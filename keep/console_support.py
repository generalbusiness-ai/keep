"""Console rendering and daemon support utilities.

Provides prompt expansion, flow/context rendering, config formatting,
directory/watch helpers, and the pending-daemon runtime used by the command
app and other integrations.
"""

import importlib.metadata
import importlib.resources
import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import typer
from typing_extensions import Annotated

from .const import (
    DAEMON_PORT,
    DAEMON_PORT_FILE,
    DAEMON_TOKEN_FILE,
    DOCUMENTS_DB,
    OPS_LOG_FILE,
    PENDING_SUMMARIES_DB,
    WORK_QUEUE_DB,
)

logger = logging.getLogger(__name__)

_URI_SCHEME_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9+.-]*://')

from .config import get_default_provider_model, get_tool_directory
from .logging_config import configure_quiet_mode, enable_debug_mode
from .provider_identity import provider_model_name
from .projections import plan_find_context_render, render_find_context_plan
from .types import (
    SYSTEM_TAG_PREFIX,
    EdgeRef,
    Item,
    ItemContext,
    MetaRef,
    PartRef,
    SimilarRef,
    VersionRef,
    local_date,
    note_display_name,
    tag_values,
)
from .utils import _text_content_id

if TYPE_CHECKING:
    from .api import Keeper


def _is_filesystem_path(source: str) -> Optional[Path]:
    """Return resolved Path if source is an existing filesystem path, None otherwise.

    Skips anything that looks like a URI (has ://). Uses expanduser + resolve.
    Conservative: only matches if the path actually exists on disk.
    """
    if _URI_SCHEME_PATTERN.match(source):
        return None
    try:
        resolved = Path(source).expanduser().resolve()
        if resolved.exists():
            return resolved
    except (OSError, ValueError):
        pass
    return None


from .utils import _list_directory_files  # noqa: E402


def _output_width() -> int:
    """Terminal width for summary truncation. Use generous default when not a TTY."""
    if not sys.stdout.isatty():
        return 200
    return shutil.get_terminal_size((120, 24)).columns


def _progress_bar(current: int, total: int, label: str, *, err: bool = False) -> None:
    """Render an inline progress bar to stderr, overwriting the current line."""
    cols = shutil.get_terminal_size((80, 24)).columns
    pct = current * 100 // total
    counter = f" {current}/{total}"
    # bar takes ~30 chars, leave rest for label
    bar_width = 20
    filled = current * bar_width // total
    bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
    prefix = f"\r  [{bar}] {pct:3d}%{counter} "
    # Truncate label to fit
    max_label = cols - len(prefix) - 2
    if len(label) > max_label > 0:
        label = label[:max_label - 1] + "\u2026"
    line = f"{prefix}{label}"
    # Pad to overwrite previous longer lines
    line = line.ljust(cols - 1)
    stream = sys.stderr if err else sys.stdout
    stream.write(line)
    stream.flush()



def _tag_display_value(value) -> str:
    """Render scalar/list tag values for compact display."""
    if isinstance(value, list):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return str(value)


def _quote_scalar_tag_value(value) -> str:
    """Render a tag scalar as a quoted JSON/YAML-compatible string."""
    return json.dumps(str(value))


def _render_edge_ref_value(ref: EdgeRef) -> str:
    """Render an edge reference as: id [date] "summary"."""
    ref_id = _shell_quote_id(ref.source_id)
    date_part = f" [{ref.date}]" if ref.date else ""
    summary_part = _quote_scalar_tag_value(ref.summary or "")
    return f"{ref_id}{date_part} {summary_part}"


def _render_tags_frontmatter(
    tags: dict,
    edge_refs: Optional[dict[str, list[EdgeRef]]] = None,
) -> list[str]:
    """Render unified tags block with quoted scalars and edge references."""
    edge_refs = edge_refs or {}
    keys = sorted(set(tags) | set(edge_refs))
    if not keys:
        return []

    lines = ["tags:"]
    for key in keys:
        refs = edge_refs.get(key, [])
        if refs:
            rendered_refs: list[str] = []
            seen_refs: set[str] = set()
            for ref in refs:
                rv = _render_edge_ref_value(ref)
                if rv in seen_refs:
                    continue
                seen_refs.add(rv)
                rendered_refs.append(rv)
            if not rendered_refs:
                continue
            lines.append(f"  {key}:")
            for rv in rendered_refs:
                lines.append(f"    - {rv}")
            continue

        values = tag_values(tags, key)
        if not values:
            continue
        if len(values) == 1:
            lines.append(f"  {key}: {_quote_scalar_tag_value(values[0])}")
        else:
            lines.append(f"  {key}:")
            for v in values:
                lines.append(f"    - {_quote_scalar_tag_value(v)}")
    return lines


# Configure quiet mode by default (suppress verbose library output)
# Set KEEP_VERBOSE=1 to enable debug mode via environment
if os.environ.get("KEEP_VERBOSE") == "1":
    enable_debug_mode()
else:
    configure_quiet_mode(quiet=True)


# Global state for CLI options (set by thin_cli callbacks)
_json_output = False
_ids_output = False
_full_output = False
_store_override: Optional[Path] = None


def _get_json_output() -> bool:
    return _json_output


def _get_ids_output() -> bool:
    return _ids_output


def _get_full_output() -> bool:
    return _full_output


def _get_store_override() -> Optional[Path]:
    return _store_override




# Shell-safe character set for IDs (no quoting needed)
_SHELL_SAFE_PATTERN = re.compile(r'^[a-zA-Z0-9_./:@{}\-%]+$')


def _shell_quote_id(id: str) -> str:
    """Quote an ID for safe shell usage if it contains non-shell-safe characters.

    IDs containing only [a-zA-Z0-9_./:@{}%-] are returned as-is.
    Others are wrapped in single quotes with internal single quotes escaped.
    """
    if _SHELL_SAFE_PATTERN.match(id):
        return id
    # Escape any single quotes within the ID: ' → '\''
    escaped = id.replace("'", "'\\''")
    return f"'{escaped}'"


# -----------------------------------------------------------------------------
# Output Formatting
#
# Three output formats, controlled by global flags:
#   --ids:  versioned ID only (id@V{N})
#   --full: YAML frontmatter with tags, similar items, version nav
#   default: summary line (id@V{N} date summary)
#
# JSON output (--json) works with any of the above.
# -----------------------------------------------------------------------------

def _filter_display_tags(tags: dict) -> dict:
    """Filter out internal-only tags for display."""
    from .types import INTERNAL_TAGS
    return {k: v for k, v in tags.items() if k not in INTERNAL_TAGS}


def render_context(ctx: ItemContext, as_json: bool = False) -> str:
    """Render an ItemContext for display.

    This is the single renderer — CLI local, CLI remote, and REST
    all produce ItemContext; this function turns it into output.
    """
    if as_json:
        return json.dumps(ctx.to_dict(), indent=2)
    return _render_item_context(ctx)


def _dicts_to_items(raw: list, summary_limit: int = 200) -> "list[Item]":
    """Convert find-action result dicts to Item objects for rendering.

    Truncates summaries to *summary_limit* chars to keep output bounded.
    """
    from .types import Item

    items = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        score = r.get("score")
        summary = str(r.get("summary", ""))
        if len(summary) > summary_limit:
            summary = summary[:summary_limit] + "..."
        items.append(Item(
            id=str(r.get("id", "")),
            summary=summary,
            tags=dict(r.get("tags", {})),
            score=float(score) if isinstance(score, (int, float)) else None,
        ))
    return items


def render_flow_response(
    result: "FlowResult",
    token_budget: int = 4000,
    keeper=None,
) -> str:
    """Render a FlowResult as token-budgeted text for LLM consumption.

    Reuses render_find_context() for search result lists found in the
    flow's return data.
    """
    def _tok(text: str) -> int:
        return len(text) // 4

    lines: list[str] = []
    remaining = token_budget

    # Header
    header = f"flow: {result.status} ({result.ticks} ticks)"
    if result.history:
        header += f" via {' > '.join(result.history)}"
    lines.append(header)
    remaining -= _tok(header)

    # Some state docs have an explicitly declared note-first render contract
    # on the text surface. Prompt expansion still consumes bindings directly;
    # this only affects rendered flow/tool output.
    from .const import FLOW_NOTE_FIRST_RENDER_STATES
    state_name = next(
        (name for name in reversed(result.history) if name in FLOW_NOTE_FIRST_RENDER_STATES),
        None,
    )
    if state_name in FLOW_NOTE_FIRST_RENDER_STATES and result.bindings and remaining > 0:
        binding_render = _render_context_from_flow_bindings(result.bindings, kp=keeper)
        if binding_render:
            section = f"\n{binding_render}"
            lines.append(section)
            remaining -= _tok(section)

            if result.tried_queries and remaining > 0:
                line = f"queries tried: {', '.join(repr(q) for q in result.tried_queries)}"
                lines.append(line)
            if result.cursor:
                lines.append(f"\ncursor: {result.cursor}")
            return "\n".join(lines)

    # Render data fields — auto-detect result lists, meta sections, scalars
    def _render_result_list(label: str, raw: list) -> None:
        nonlocal remaining
        items = _dicts_to_items(raw)
        if not items or remaining <= 0:
            return
        section_budget = min(remaining, token_budget // 3)
        rendered = render_find_context(
            items, keeper=keeper, token_budget=section_budget,
        )
        section = f"\n{label}:\n{rendered}"
        lines.append(section)
        remaining -= _tok(section)

    if result.data and remaining > 0:
        for key, val in result.data.items():
            if remaining <= 0:
                break
            if val is None:
                continue
            # Direct result list (e.g., data.results)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                _render_result_list(key, val)
            # Action binding with results (e.g., data.similar = {results: [...]})
            elif isinstance(val, dict) and "results" in val:
                raw_results = val["results"]
                if isinstance(raw_results, list) and raw_results:
                    _render_result_list(key, raw_results)
            # Meta sections (e.g., data.meta = {sections: {learnings: [...], todo: [...]}})
            elif isinstance(val, dict) and "sections" in val:
                for section_name, section_items in val["sections"].items():
                    if isinstance(section_items, list) and section_items and remaining > 0:
                        _render_result_list(f"meta/{section_name}", section_items)
            # Versions (e.g., data.versions = {versions: [{offset, summary, date}], count})
            elif isinstance(val, dict) and "versions" in val:
                vers = val["versions"]
                if isinstance(vers, list) and vers and remaining > 0:
                    lines.append(f"\n{key}:")
                    for v in vers:
                        if remaining <= 0:
                            break
                        line = f"  - @V{{{v.get('offset', '?')}}}  [{v.get('date', '')[:10]}]  {str(v.get('summary', ''))[:150]}"
                        lines.append(line)
                        remaining -= _tok(line)
            # Edges (e.g., data.edges = {edges: {predicate: [{id, summary, ...}]}, count})
            elif isinstance(val, dict) and "edges" in val:
                edge_groups = val["edges"]
                if isinstance(edge_groups, dict) and edge_groups and remaining > 0:
                    lines.append(f"\n{key}:")
                    for predicate, refs in edge_groups.items():
                        if remaining <= 0:
                            break
                        if isinstance(refs, list):
                            for ref in refs:
                                if remaining <= 0:
                                    break
                                line = f"  - [{predicate}] {ref.get('id', '')}  {str(ref.get('summary', ''))[:100]}"
                                lines.append(line)
                                remaining -= _tok(line)
            # Scalar (margin, entropy, reason, item_id, etc.)
            elif not isinstance(val, (dict, list)):
                line = f"{key}: {val}"
                lines.append(line)
                remaining -= _tok(line)

    # Some flows return their useful payload only in bindings, not data.
    # If we only have the status header so far, render bindings directly.
    if len(lines) == 1 and result.bindings and remaining > 0:
        binding_render = _render_context_from_flow_bindings(result.bindings, kp=keeper)
        if not binding_render:
            binding_sections = []
            for name, binding in result.bindings.items():
                if remaining <= 0 or not isinstance(binding, dict):
                    continue
                rendered = _render_binding(
                    name,
                    binding,
                    kp=keeper,
                    token_budget=min(remaining, token_budget),
                )
                if not rendered:
                    continue
                section = f"\n{name}:\n{rendered}"
                binding_sections.append(section)
                remaining -= _tok(section)
        else:
            section = f"\n{binding_render}"
            binding_sections = [section]
            remaining -= _tok(section)

        lines.extend(binding_sections)

    # Tried queries (for stopped flows)
    if result.tried_queries and remaining > 0:
        line = f"queries tried: {', '.join(repr(q) for q in result.tried_queries)}"
        lines.append(line)
        remaining -= _tok(line)

    # Cursor for stopped flows
    if result.cursor:
        lines.append(f"\ncursor: {result.cursor}")

    return "\n".join(lines)


def render_find_context(
    items: "list[Item]",
    keeper=None,
    token_budget: int = 4000,
    show_tags: bool = False,
    deep_primary_cap: int | None = None,
) -> str:
    """Render find results for prompt injection, filling a token budget.

    Three-pass rendering:
      Pass 1: Lay down summary lines for all items (breadth-first).
      Pass 2: Render deep sub-items and local windows from remaining budget.
      Pass 3: Backfill detail (parts, versions, tags) with what is left.

    When *deep_primary_cap* is set and deep groups exist, only the top N
    primaries are rendered (preferring those with deep groups) and pass 3
    is skipped — this gives maximum budget to deep-discovered evidence,
    useful for multi-hop queries.

    Used by expand_prompt() for {find} expansion and MCP keep_flow.
    """
    plan = plan_find_context_render(
        items,
        keeper=keeper,
        token_budget=token_budget,
        show_tags=show_tags,
        deep_primary_cap=deep_primary_cap,
    )
    if not plan.blocks:
        return "No results."
    return render_find_context_plan(plan)


def _render_item_context(ctx: ItemContext) -> str:
    """Render a complete item: YAML-style metadata block, then body text."""
    cols = _output_width()
    item = ctx.item

    def _truncate(summary: str, prefix_len: int) -> str:
        max_width = max(cols - prefix_len, 20)
        s = summary.replace("\n", " ")
        if len(s) > max_width:
            s = s[:max_width - 3].rsplit(" ", 1)[0] + "..."
        return s

    version_suffix = f"@V{{{ctx.viewing_offset}}}" if ctx.viewing_offset > 0 else ""
    lines = ["---", f"id: {_shell_quote_id(item.id)}{version_suffix}"]

    display_tags = _filter_display_tags(item.tags)
    lines.extend(_render_tags_frontmatter(display_tags, ctx.edges))

    if item.score is not None:
        lines.append(f"score: {item.score:.2f}")

    # Similar items
    if ctx.similar:
        sim_ids = []
        for s in ctx.similar:
            sid = _shell_quote_id(s.id)
            if s.offset > 0:
                sid += f"@V{{{s.offset}}}"
            sim_ids.append(sid)
        id_width = min(max(len(s) for s in sim_ids), 20)
        lines.append("similar:")
        for s, sid in zip(ctx.similar, sim_ids):
            score_str = f"({s.score:.2f})" if s.score else ""
            actual_id_len = max(len(sid), id_width)
            prefix_len = 4 + actual_id_len + 1 + len(score_str) + 1 + len(s.date) + 1
            summary_preview = _truncate(s.summary, prefix_len)
            lines.append(f"  - {sid.ljust(id_width)} {score_str} {s.date} {summary_preview}")

    # Meta sections
    if ctx.meta:
        for name, refs in ctx.meta.items():
            meta_ids = [_shell_quote_id(r.id) for r in refs]
            id_width = min(max(len(s) for s in meta_ids), 20)
            lines.append(f"meta/{name}:")
            for ref, mid in zip(refs, meta_ids):
                actual_id_len = max(len(mid), id_width)
                prefix_len = 4 + actual_id_len + 1
                summary_preview = _truncate(ref.summary, prefix_len)
                lines.append(f"  - {mid.ljust(id_width)} {summary_preview}")

    # Parts manifest
    if ctx.parts:
        total = len(ctx.parts)
        if ctx.expand_parts:
            visible = ctx.parts  # --parts: show all
        elif ctx.focus_part is not None:
            visible = [p for p in ctx.parts if abs(p.part_num - ctx.focus_part) <= 1]
        elif total <= 3:
            visible = ctx.parts
        else:
            visible = [ctx.parts[0], ctx.parts[-1]]  # first + last
        part_ids = [f"@P{{{p.part_num}}}" for p in visible]
        id_width = max(len(s) for s in part_ids)
        label = "parts:" if ctx.focus_part is None else f"parts: # {total} total, showing around @P{{{ctx.focus_part}}}"
        lines.append(label)
        if ctx.expand_parts or ctx.focus_part is not None or total <= 3:
            for part, pid in zip(visible, part_ids):
                marker = " *" if ctx.focus_part is not None and part.part_num == ctx.focus_part else ""
                prefix_len = 4 + id_width + 1 + len(marker)
                summary_preview = _truncate(part.summary, prefix_len)
                lines.append(f"  - {pid.ljust(id_width)} {summary_preview}{marker}")
        else:
            # Compact: first, "N more...", last
            first, last = visible[0], visible[1]
            first_pid, last_pid = part_ids[0], part_ids[1]
            prefix_len = 4 + id_width + 1
            lines.append(f"  - {first_pid.ljust(id_width)} {_truncate(first.summary, prefix_len)}")
            lines.append(f"  # (...{total - 2} more...)")
            lines.append(f"  - {last_pid.ljust(id_width)} {_truncate(last.summary, prefix_len)}")

    # Version navigation
    if ctx.prev:
        prev_ids = [f"@V{{{v.offset}}}" for v in ctx.prev]
        id_width = max(len(s) for s in prev_ids)
        lines.append("prev:")
        for vid, v in zip(prev_ids, ctx.prev):
            prefix_len = 4 + id_width + 1 + len(v.date) + 1
            summary_preview = _truncate(v.summary, prefix_len)
            lines.append(f"  - {vid.ljust(id_width)} {v.date} {summary_preview}")
    if ctx.next:
        next_ids = [f"@V{{{v.offset}}}" for v in ctx.next]
        id_width = max(len(s) for s in next_ids)
        lines.append("next:")
        for vid, v in zip(next_ids, ctx.next):
            prefix_len = 4 + id_width + 1 + len(v.date) + 1
            summary_preview = _truncate(v.summary, prefix_len)
            lines.append(f"  - {vid.ljust(id_width)} {v.date} {summary_preview}")
    elif ctx.viewing_offset > 0:
        lines.append("next:")
        lines.append(f"  - @V{{0}}")

    lines.append("---")
    lines.append(item.summary)
    return "\n".join(lines)


def _format_summary_line(item: Item, id_width: int = 0, show_tags: bool = False) -> str:
    """Format item as single summary line: id date summary (with @V{N} only for old versions).

    Args:
        item: The item to format
        id_width: Minimum width for ID column (for alignment across items)
        show_tags: Show non-system tags between date and summary
    """
    # Get version/part-scoped ID
    base_id = item.tags.get("_base_id", item.id)
    part_num = item.tags.get("_part_num")
    version = item.tags.get("_version", "0")
    if part_num:
        suffix = f"@P{{{part_num}}}"
    elif version != "0":
        suffix = f"@V{{{version}}}"
    else:
        suffix = ""
    versioned_id = f"{_shell_quote_id(base_id)}{suffix}"

    # Pad ID for column alignment
    padded_id = versioned_id.ljust(id_width) if id_width else versioned_id

    # Score (when available from find results)
    score_str = f" ({item.score:.2f})" if item.score is not None else ""

    # Get date in local timezone
    date = local_date(item.tags.get("_created") or item.tags.get("_updated", ""))

    # Truncate display name/summary to fit terminal width, collapse newlines
    cols = _output_width()
    prefix_len = len(padded_id) + len(score_str) + 1 + len(date) + 1  # "id (score) date "
    max_summary = max(cols - prefix_len, 20)
    summary = note_display_name(item.tags, item.summary, max_len=max_summary)

    line = f"{padded_id}{score_str} {date} {summary}"

    # Show matched part summary when item was uplifted from a part hit
    focus_part = item.tags.get("_focus_part")
    focus_summary = item.tags.get("_focus_summary")
    if focus_part and focus_summary:
        focus_text = focus_summary.replace("\n", " ")
        max_focus = max(cols - 4, 20)  # "  > " prefix
        if len(focus_text) > max_focus:
            focus_text = focus_text[:max_focus - 3].rsplit(" ", 1)[0] + "..."
        line += f"\n  > {focus_text}"

    # Optional non-system tags on a separate line
    if show_tags:
        from .types import SYSTEM_TAG_PREFIX
        user_tags = {k: v for k, v in item.tags.items() if not k.startswith(SYSTEM_TAG_PREFIX)}
        if user_tags:
            pairs = ", ".join(
                f"{k}: {_tag_display_value(v)}" for k, v in sorted(user_tags.items())
            )
            line += f"\n  {{{pairs}}}"

    return line


def _format_versioned_id(item: Item) -> str:
    """Format item ID with version suffix only for old versions: id or id@V{N}."""
    base_id = item.tags.get("_base_id", item.id)
    version = item.tags.get("_version", "0")
    version_suffix = f"@V{{{version}}}" if version != "0" else ""
    return f"{_shell_quote_id(base_id)}{version_suffix}"




# -----------------------------------------------------------------------------
# Common Options
# -----------------------------------------------------------------------------

StoreOption = Annotated[
    Optional[Path],
    typer.Option(
        "--store", "-s",
        envvar="KEEP_STORE_PATH",
        help="Path to the store directory (default: ~/.keep/)"
    )
]



def _format_items(items: list[Item], as_json: bool = False, keeper=None, show_tags: bool = False) -> str:
    """Format multiple items for display.

    Args:
        items: List of items to format.
        as_json: Output as JSON instead of text.
        keeper: Optional Keeper instance. When provided, items with
                _focus_part (uplifted from part hits) get their parts
                manifest loaded and windowed around the hit.
        show_tags: Show non-system tags in summary lines (used by --deep).
    """
    if _get_ids_output():
        ids = [_format_versioned_id(item) for item in items]
        return json.dumps(ids) if as_json else "\n".join(ids)

    if as_json:
        deep_groups = getattr(items, "deep_groups", {})

        def _item_dict(item):
            from .types import SYSTEM_TAG_PREFIX
            user_tags = {k: v for k, v in item.tags.items()
                         if not k.startswith(SYSTEM_TAG_PREFIX)}
            return {
                "id": item.id,
                "summary": item.summary,
                "tags": user_tags,
                "score": item.score,
                "created": item.created,
                "updated": item.updated,
            }

        result = []
        seen_deep: set[str] = set()
        for item in items:
            d = _item_dict(item)
            parent_id = item.id.split("@")[0] if "@" in item.id else item.id
            group = deep_groups.get(parent_id, []) or deep_groups.get(item.id, [])
            unseen = [di for di in group if di.id not in seen_deep]
            if unseen:
                d["deep"] = [_item_dict(di) for di in unseen]
                seen_deep.update(di.id for di in unseen)
            result.append(d)
        return json.dumps(result, indent=2)

    if not items:
        return "No results."

    # Full format: YAML frontmatter with double-newline separator
    # Default: summary lines with single-newline separator
    if _get_full_output():
        parts = []
        for item in items:
            focus = item.tags.get("_focus_part")
            focus_int = int(focus) if focus else None
            manifest = None
            if focus_int and keeper:
                manifest = keeper.list_parts(item.id) or None
            ctx = ItemContext(
                item=item,
                parts=[PartRef(part_num=p.part_num, summary=p.summary) for p in manifest] if manifest else [],
                focus_part=focus_int,
            )
            parts.append(_render_item_context(ctx))
        return "\n\n".join(parts)

    # Compute ID column width for alignment (capped to avoid long URIs dominating)
    all_items = list(items)
    deep_groups = getattr(items, "deep_groups", {})
    for group in deep_groups.values():
        all_items.extend(group)
    max_id = max(len(_format_versioned_id(item)) for item in all_items)
    id_width = min(max_id, 20)

    # Render with nested deep groups (dedup across items sharing a parent)
    if deep_groups:
        lines = []
        seen_deep: set[str] = set()
        for item in items:
            lines.append(_format_summary_line(item, id_width, show_tags=show_tags))
            parent_id = item.id.split("@")[0] if "@" in item.id else item.id
            for deep_item in deep_groups.get(parent_id, []) or deep_groups.get(item.id, []):
                if deep_item.id in seen_deep:
                    continue
                seen_deep.add(deep_item.id)
                deep_line = _format_summary_line(deep_item, id_width, show_tags=show_tags)
                lines.append("  " + deep_line.replace("\n", "\n  "))
        return "\n".join(lines)

    return "\n".join(_format_summary_line(item, id_width, show_tags=show_tags) for item in items)


NO_PROVIDER_ERROR = """
No embedding provider configured.

To use keep, configure a provider:

  Hosted (simplest — no local setup):
    export KEEPNOTES_API_KEY=...   # Sign up at https://keepnotes.ai

  API-based:
    export VOYAGE_API_KEY=...      # Get at dash.voyageai.com
    export ANTHROPIC_API_KEY=...   # Optional: for better summaries

  Local (macOS Apple Silicon):
    pip install 'keep-skill[local]'

See: https://github.com/keepnotes-ai/keep#installation
"""


def _authoritative_remote_store(config):
    """Return the ``remote_store`` config if valid, else None."""
    remote_store = getattr(config, "remote_store", None)
    if remote_store is None:
        return None
    api_url = getattr(remote_store, "api_url", None)
    api_key = getattr(remote_store, "api_key", None)
    project = getattr(remote_store, "project", None)
    if not isinstance(api_url, str) or not api_url.strip():
        return None
    if not isinstance(api_key, str) or not api_key.strip():
        return None
    if project is not None and not isinstance(project, str):
        return None
    return remote_store


def _get_keeper(store: Optional[Path], *, _force_local: bool = False) -> "Keeper":
    """Initialize memory, handling errors gracefully.

    Returns a local Keeper or RemoteKeeper depending on authoritative-store
    config.
    Local daemon commands (put directory, pending, etc.)
    always use a local Keeper — the command app handles the daemon HTTP path.

    When ``_force_local`` is True, skips the remote backend check
    (used by ``keep daemon`` and the hidden ``keep pending`` compatibility alias).
    """
    import atexit
    from .api import Keeper

    # Env vars target the remote authoritative store.
    api_url = os.environ.get("KEEPNOTES_API_URL", "https://api.keepnotes.ai")
    api_key = os.environ.get("KEEPNOTES_API_KEY")
    if api_url and api_key and not _force_local:
        from .config import get_config_dir, load_or_create_config
        from .remote import RemoteKeeper
        try:
            config_dir = get_config_dir()
            config = load_or_create_config(config_dir)
            kp = RemoteKeeper(api_url, api_key, config)
            atexit.register(kp.close)
            return kp
        except Exception as e:
            typer.echo(f"Error connecting to remote: {e}", err=True)
            raise typer.Exit(1)

    # Check global override from --store on main command
    actual_store = store if store is not None else _get_store_override()
    try:
        # Run setup wizard on first use (no existing config)
        from .paths import get_config_dir
        from .setup_wizard import needs_wizard, run_wizard
        if os.environ.get("KEEP_CONFIG"):
            wizard_config_dir = get_config_dir()
        elif actual_store is not None:
            wizard_config_dir = Path(actual_store).resolve()
        else:
            wizard_config_dir = get_config_dir()
        wizard_config = None
        if needs_wizard(wizard_config_dir):
            store_path = Path(actual_store).resolve() if actual_store else None
            wizard_config = run_wizard(wizard_config_dir, store_path)

        kp = Keeper(actual_store, config=wizard_config)
        # Ensure close() runs before interpreter shutdown to release model locks
        atexit.register(kp.close)

        # After first-time setup: populate store with system docs immediately
        # so prompts, tags, state docs, etc. are available right away.
        if wizard_config and kp._needs_sysdoc_migration:
            is_tty = sys.stderr.isatty()
            try:
                def _setup_progress(current, total, label):
                    if is_tty:
                        _progress_bar(current, total, label, err=True)
                result = kp._migrate_system_documents(progress=_setup_progress)
                kp._needs_sysdoc_migration = False
                n_loaded = result.get("created", 0) + result.get("migrated", 0)
                if is_tty:
                    # Replace progress bar with final summary on same line
                    cols = shutil.get_terminal_size((80, 24)).columns
                    msg = f"  Loaded {n_loaded} system docs." if n_loaded else ""
                    sys.stderr.write("\r" + msg.ljust(cols - 1) + "\n")
                    sys.stderr.flush()
                elif n_loaded:
                    typer.echo(f"  Loaded {n_loaded} system docs.", err=True)
            except Exception as e:
                logger.warning("System doc setup deferred: %s", e)

        # Check for remote authoritative-store config in TOML.
        remote_store = _authoritative_remote_store(kp.config) if kp.config else None
        if remote_store and not _force_local:
            from .remote import RemoteKeeper
            remote = RemoteKeeper(
                remote_store.api_url,
                remote_store.api_key,
                kp.config,
                project=remote_store.project,
            )
            atexit.register(remote.close)
            kp.close()  # Don't need the local Keeper
            return remote

        # Warn (don't exit) if no embedding provider — read-only ops still work
        if kp.config and kp.config.embedding is None:
            typer.echo(NO_PROVIDER_ERROR.strip(), err=True)
        return kp
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def _handle_watch(
    kp: "Keeper",
    watch: bool,
    unwatch: bool,
    source: str,
    kind: str,
    parsed_tags: dict,
    *,
    recurse: bool = False,
    exclude: list[str] | None = None,
    interval: str | None = None,
) -> None:
    """Add or remove a watch entry if --watch or --unwatch was given."""
    if not watch and not unwatch:
        return
    from .watches import add_watch, remove_watch
    if watch:
        kwargs: dict = {}
        if interval:
            kwargs["interval"] = interval
        try:
            entry = add_watch(
                kp, source, kind,
                tags=parsed_tags or {},
                recurse=recurse,
                exclude=exclude or [],
                max_watches=kp.config.max_watches,
                **kwargs,
            )
            typer.echo(f"Watching {kind}: {source} (interval {entry.interval})", err=True)
        except ValueError as e:
            typer.echo(f"Watch error: {e}", err=True)
    elif unwatch:
        if remove_watch(kp, source):
            typer.echo(f"Stopped watching: {source}", err=True)
        else:
            typer.echo(f"Not watching: {source}", err=True)


def _render_binding(name: str, binding: dict, kp=None, token_budget: int = 4000) -> str:
    """Render a single flow binding to text by dispatching on shape."""
    from .types import Item

    # Find-action results: {"results": [...], "count": N}
    results = binding.get("results")
    if isinstance(results, list) and results:
        items = _dicts_to_items(results, summary_limit=500)
        if items:
            return render_find_context(items, keeper=kp, token_budget=token_budget)

    # Get-action result: {"id": ..., "summary": ..., "tags": ...}
    if "id" in binding and "summary" in binding:
        item = Item(
            id=str(binding["id"]),
            summary=str(binding.get("summary", "")),
            tags=dict(binding.get("tags") or {}),
        )
        from ._context_resolution import ItemContext
        ctx = ItemContext(item=item, viewing_offset=0, similar=[], meta={}, edges={}, parts=[], prev=[], next=[])
        return render_context(ctx)

    # Meta sections: {"sections": {"learnings": [...], "todo": [...]}}
    sections = binding.get("sections")
    if isinstance(sections, dict):
        lines = []
        for section_name, section_items in sections.items():
            if isinstance(section_items, list) and section_items:
                items = _dicts_to_items(section_items, summary_limit=300)
                if items:
                    lines.append(f"meta/{section_name}:")
                    for item in items:
                        sid = item.id[:20]
                        summary = item.summary.replace("\n", " ")[:120]
                        lines.append(f"  - {sid} {summary}")
        return "\n".join(lines)

    # Edges: {"edges": {...}}
    edges = binding.get("edges")
    if isinstance(edges, dict) and edges:
        lines = []
        for tag_key, edge_list in edges.items():
            if isinstance(edge_list, list):
                for e in edge_list:
                    if isinstance(e, dict):
                        lines.append(f"  {tag_key}: {e.get('id', '')} {str(e.get('summary', ''))[:100]}")
        return "\n".join(lines)

    return ""


def _render_context_from_flow_bindings(bindings: dict[str, dict], kp=None) -> str:
    """Render a ``{get}`` compatibility view from prompt flow bindings."""
    if not isinstance(bindings, dict):
        return ""

    item_binding = bindings.get("item")
    if not isinstance(item_binding, dict):
        return ""
    if "id" not in item_binding or "summary" not in item_binding:
        return ""

    item_id = str(item_binding["id"])
    if kp is not None:
        try:
            # `now` is a virtual/default note until first materialized. Ensure
            # it exists before asking get_context() to assemble the full view.
            if item_id == "now" and hasattr(kp, "get_now"):
                kp.get_now()
            ctx = kp.get_context(item_id)
            if ctx is not None and getattr(ctx, "item", None) is not None:
                return render_context(ctx)
        except Exception as exc:
            logger.info(
                "Failed to render full context for %s; falling back to item binding: %s",
                item_id,
                exc,
                exc_info=True,
            )

    from ._context_resolution import ItemContext
    from .types import Item

    item = Item(
        id=item_id,
        summary=str(item_binding.get("summary", "")),
        tags=dict(item_binding.get("tags") or {}),
    )

    def _collect_similar() -> list[SimilarRef]:
        binding = bindings.get("similar")
        if not isinstance(binding, dict):
            return []
        results = binding.get("results")
        if not isinstance(results, list):
            return []
        similar: list[SimilarRef] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            tags = dict(result.get("tags") or {})
            similar.append(
                SimilarRef(
                    id=str(result.get("id", "")),
                    offset=0,
                    score=result.get("score"),
                    date=local_date(str(tags.get("_created") or tags.get("_updated") or "")),
                    summary=str(result.get("summary", "")),
                )
            )
        return similar

    def _collect_parts() -> list[PartRef]:
        binding = bindings.get("parts")
        if not isinstance(binding, dict):
            return []
        results = binding.get("results")
        if not isinstance(results, list):
            return []
        parts: list[PartRef] = []
        for idx, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue
            tags = dict(result.get("tags") or {})
            raw_part_num = tags.get("_part_num", idx)
            try:
                part_num = int(raw_part_num)
            except (TypeError, ValueError):
                part_num = idx
            parts.append(
                PartRef(
                    part_num=part_num,
                    summary=str(result.get("summary", "")),
                    tags=tags,
                )
            )
        return parts

    meta: dict[str, list[MetaRef]] = {}
    meta_binding = bindings.get("meta")
    if isinstance(meta_binding, dict):
        sections = meta_binding.get("sections")
        if isinstance(sections, dict):
            for key, results in sections.items():
                if isinstance(results, list):
                    meta[key] = [
                        MetaRef(id=str(result.get("id", "")), summary=str(result.get("summary", "")))
                        for result in results
                        if isinstance(result, dict)
                    ]

    edges: dict[str, list[EdgeRef]] = {}
    edge_binding = bindings.get("edges")
    if isinstance(edge_binding, dict):
        groups = edge_binding.get("edges")
        if isinstance(groups, dict):
            for key, results in groups.items():
                if isinstance(results, list):
                    rendered: list[EdgeRef] = []
                    for result in results:
                        if not isinstance(result, dict):
                            continue
                        tags = dict(result.get("tags") or {})
                        rendered.append(
                            EdgeRef(
                                source_id=str(result.get("id", "")),
                                date=local_date(str(tags.get("_created") or tags.get("_updated") or "")),
                                summary=str(result.get("summary", "")),
                            )
                        )
                    edges[key] = rendered

    ctx = ItemContext(
        item=item,
        viewing_offset=0,
        similar=_collect_similar(),
        meta=meta,
        edges=edges,
        parts=_collect_parts(),
        prev=[],
        next=[],
    )
    return render_context(ctx)


def _find_results_from_prompt_result(result: "PromptResult"):
    """Return prompt search results from either direct retrieval or flow bindings."""
    if result.search_results:
        return result.search_results
    bindings = result.flow_bindings or {}
    for name in ("find_results", "search"):
        binding = bindings.get(name)
        if not isinstance(binding, dict):
            continue
        results = binding.get("results")
        if isinstance(results, list) and results:
            return _dicts_to_items(results, summary_limit=500)
    return None


def expand_prompt(result: "PromptResult", kp=None) -> str:
    """Expand placeholders in a prompt template.

    Supports:
      {get}                 — rendered item context
      {find[:deep][:budget]} — search results
      {text}, {since}, {until} — raw filter values
      {binding_name}        — flow binding (when state doc is used)
    """
    output = result.prompt

    # Expand {get} with rendered context
    if result.context:
        get_rendered = render_context(result.context)
    elif result.flow_bindings:
        get_rendered = _render_context_from_flow_bindings(result.flow_bindings, kp=kp)
    else:
        get_rendered = ""
    output = output.replace("{get}", get_rendered)

    # Expand {find} variants with token-budgeted context.
    # Syntax: {find[:deep][:budget]}
    # When :deep is present, primary cap is applied automatically.
    _find_re = re.compile(r'\{find(?::(deep))?(?::(\d+))?\}')
    def _expand_find(m):
        search_results = _find_results_from_prompt_result(result)
        if not search_results:
            return ""
        is_deep = m.group(1) is not None
        budget_str = m.group(2)
        # Explicit CLI --tokens always wins over template budget
        if result.token_budget is not None:
            budget = result.token_budget
        elif budget_str:
            budget = int(budget_str)
        else:
            budget = 4000
        cap = 3 if is_deep else None
        return render_find_context(
            search_results, keeper=kp,
            token_budget=budget,
            deep_primary_cap=cap,
        )
    output = _find_re.sub(_expand_find, output)

    # Expand flow bindings: {binding_name} -> rendered binding
    if result.flow_bindings:
        budget = result.token_budget or 4000
        # Distribute budget across bindings
        n_bindings = sum(1 for b in result.flow_bindings.values() if b)
        per_binding = budget // max(n_bindings, 1)
        for name, binding in result.flow_bindings.items():
            # Match {name} or {name|fallback} — fallback text is used when
            # the binding renders empty (e.g. {edges|(none)}).
            _pat = re.compile(r'\{' + re.escape(name) + r'(?:\|([^}]*))?\}')
            for m in _pat.finditer(output):
                rendered = _render_binding(name, binding, kp=kp, token_budget=per_binding)
                if not rendered.strip() and m.group(1) is not None:
                    rendered = m.group(1)
                output = output.replace(m.group(0), rendered)

    # Replace remaining binding placeholders whose rules didn't fire
    # (e.g. {similar} when search ran instead).  Supports {name|fallback}.
    _known_value_placeholders = {"{text}", "{since}", "{until}"}
    _unfired_re = re.compile(r'\{([a-z_]+)(?:\|([^}]*))?\}')
    for m in _unfired_re.finditer(output):
        if m.group(0) not in _known_value_placeholders:
            fallback = m.group(2) if m.group(2) is not None else ""
            output = output.replace(m.group(0), fallback)

    # Expand {text}, {since}, {until} with raw filter values
    output = output.replace("{text}", result.text or "")
    output = output.replace("{since}", result.since or "")
    output = output.replace("{until}", result.until or "")

    # Clean up blank lines from empty expansions
    while "\n\n\n" in output:
        output = output.replace("\n\n\n", "\n\n")

    return output.strip()


def _get_config_value(cfg, store_path: Path, path: str):
    """Get config value by dotted path.

    Special paths (not in TOML):
        file - config file location
        tool - package directory (SKILL.md location)
        openclaw-plugin - OpenClaw plugin directory
        mcpb - generate .mcpb bundle for Claude Desktop
        store - store path

    Dotted paths into config:
        providers - all provider config
        providers.embedding - embedding provider name
        providers.summarization - summarization provider name
        embedding.* - embedding config details
        summarization.* - summarization config details
        tags - default tags
    """
    # Special built-in paths (not in TOML)
    if path == "file":
        return str(cfg.config_path) if cfg else None
    if path == "tool":
        return str(get_tool_directory())
    if path == "openclaw-plugin":
        return str(Path(str(importlib.resources.files("keep"))) / "data" / "openclaw-plugin")
    if path == "mcpb":
        from .mcpb import generate_mcpb
        out = generate_mcpb(store_path=store_path)
        if platform.system() == "Darwin":
            subprocess.Popen(["open", str(out)])
        elif platform.system() == "Windows":
            os.startfile(str(out))
        else:
            subprocess.Popen(["xdg-open", str(out)])
        return (
            'After installation, just say to Claude:\n'
            'Please read all the keep_help documentation, and then use keep_prompt(name="reflect") to save some notes about what you learn.'
        )
    if path == "docs":
        return str(get_tool_directory() / "docs")
    if path == "store":
        return str(store_path)
    # Provider shortcuts
    if path == "providers":
        if cfg:
            return {
                "embedding": cfg.embedding.name if cfg.embedding else None,
                "summarization": cfg.summarization.name,
                "document": cfg.document.name,
            }
        return None
    if path == "providers.embedding":
        return cfg.embedding.name if cfg and cfg.embedding else None
    if path == "providers.summarization":
        return cfg.summarization.name if cfg else None
    if path == "providers.document":
        return cfg.document.name if cfg else None

    # Tags shortcut
    if path == "tags":
        return cfg.default_tags if cfg else {}

    # Dotted path into config attributes
    if not cfg:
        raise typer.BadParameter(f"No config loaded, cannot access: {path}")

    parts = path.split(".")
    value = cfg
    for part in parts:
        if hasattr(value, part):
            value = getattr(value, part)
        elif hasattr(value, "params") and part in value.params:
            # Provider config params
            value = value.params[part]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise typer.BadParameter(f"Unknown config path: {path}")

    # Return name for provider objects
    if hasattr(value, "name") and hasattr(value, "params"):
        return value.name
    return value


def _format_config_with_defaults(cfg, store_path: Path) -> str:
    """Format config output with commented defaults for unused settings."""
    config_path = cfg.config_path if cfg else None
    lines = []

    # Show paths
    lines.append(f"file: {config_path}")
    lines.append(f"tool: {get_tool_directory()}")
    lines.append(f"docs: {get_tool_directory() / 'docs'}")
    lines.append(f"store: {store_path}")

    lines.append(f"openclaw-plugin: {Path(str(importlib.resources.files('keep'))) / 'data' / 'openclaw-plugin'}")

    if cfg:
        lines.append("")
        lines.append("providers:")
        lines.append(f"  embedding: {cfg.embedding.name if cfg.embedding else 'none'}")
        if cfg.embedding and cfg.embedding.params.get("model"):
            lines.append(f"    model: {cfg.embedding.params['model']}")
        lines.append(f"  summarization: {cfg.summarization.name if cfg.summarization else 'none'}")
        if cfg.summarization and cfg.summarization.params.get("model"):
            lines.append(f"    model: {cfg.summarization.params['model']}")

        # Show limits
        lines.append("")
        lines.append("limits:")
        lines.append(f"  max_summary_length: {cfg.max_summary_length}")
        lines.append(f"  max_inline_length: {cfg.max_inline_length}")

        # Show configured tags or example
        if cfg.default_tags:
            lines.append("")
            lines.append("tags:")
            for key, value in cfg.default_tags.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append("")
            lines.append("# tags:")
            lines.append("#   project: myproject")

        # Show integrations status
        from .integrations import TOOL_CONFIGS
        if cfg.integrations:
            lines.append("")
            lines.append("integrations:")
            for tool_key in TOOL_CONFIGS:
                if tool_key in cfg.integrations:
                    status = cfg.integrations[tool_key]
                    lines.append(f"  {tool_key}: {status}")
            for tool_key in TOOL_CONFIGS:
                if tool_key not in cfg.integrations:
                    lines.append(f"  # {tool_key}: false")
        else:
            lines.append("")
            lines.append("# integrations:")
            for tool_key in TOOL_CONFIGS:
                lines.append(f"#   {tool_key}: false")

        # Show available options as comments
        lines.append("")
        lines.append("# --- Configuration Options ---")
        lines.append("#")
        lines.append("# API Keys (set in environment):")
        lines.append("#   VOYAGE_API_KEY     → embedding: voyage (Anthropic's partner)")
        lines.append("#   ANTHROPIC_API_KEY  → summarization: anthropic")
        lines.append("#   OPENAI_API_KEY     → embedding: openai, summarization: openai")
        lines.append("#   OPENROUTER_API_KEY → embedding: openrouter, summarization: openrouter")
        lines.append("#   GEMINI_API_KEY     → embedding: gemini, summarization: gemini")
        lines.append("#   GOOGLE_CLOUD_PROJECT → Vertex AI (uses Workload Identity / ADC)")
        lines.append("#")
        lines.append("# Models (configure in keep.toml):")
        lines.append(
            f"#   voyage: {get_default_provider_model('embedding', 'voyage') or 'voyage-3.5-lite'} (default), "
            "voyage-3-large, voyage-code-3"
        )
        lines.append(
            f"#   anthropic: {get_default_provider_model('summarization', 'anthropic') or 'claude-haiku-4-5-20251001'} "
            "(default), claude-3-5-haiku-20241022"
        )
        lines.append(
            f"#   openai embedding: {get_default_provider_model('embedding', 'openai') or 'text-embedding-3-small'} "
            "(default), text-embedding-3-large"
        )
        lines.append(
            f"#   openai summarization: {get_default_provider_model('summarization', 'openai') or 'gpt-4.1-mini'} "
            "(default)"
        )
        lines.append(
            f"#   openrouter embedding: {get_default_provider_model('embedding', 'openrouter') or 'openai/text-embedding-3-small'} "
            "(default)"
        )
        lines.append(
            f"#   openrouter summarization: {get_default_provider_model('summarization', 'openrouter') or 'openai/gpt-4.1-mini'} "
            "(default)"
        )
        lines.append(
            f"#   gemini embedding: {get_default_provider_model('embedding', 'gemini') or 'text-embedding-004'} "
            "(default)"
        )
        lines.append(
            f"#   gemini summarization: {get_default_provider_model('summarization', 'gemini') or 'gemini-2.5-flash'} "
            "(default)"
        )
        lines.append("#")
        lines.append("# Ollama (auto-detected if running, no API key needed):")
        lines.append("#   OLLAMA_HOST        → default: http://localhost:11434")
        lines.append("#   ollama embedding: any model (prefer nomic-embed-text, mxbai-embed-large)")
        lines.append("#   ollama summarization: any generative model (e.g. llama3.2, mistral)")

    return "\n".join(lines)



def run_pending_daemon(
    kp,
    *,
    processor_lock=None,
    bind_host: str | None = None,
    advertised_url: str | None = None,
    trusted_proxy: bool | None = None,
) -> None:
    """Run the background processing daemon loop.

    Manages HTTP server, signal handlers, work processing, watches,
    timer events, and version-aware restart. Called from the CLI
    daemon command and the hidden ``pending --daemon`` compatibility path.
    """
    from .model_lock import ModelLock
    from .shutdown import clear_shutdown
    from .tracing import init_tracing
    init_tracing(tree_log=True)
    clear_shutdown()

    _daemon_logger = logging.getLogger("keep.console_support.daemon")
    pid_path = kp._processor_pid_path
    processor_lock_acquired_here = processor_lock is None
    if processor_lock is None:
        processor_lock = ModelLock(kp._store_path / ".processor.lock")
    flow_worker_id = f"pending-daemon:{os.getpid()}"
    shutdown_requested = False
    export_keeper = kp

    remote_store = _authoritative_remote_store(kp.config) if kp.config else None
    if remote_store:
        from .remote import RemoteKeeper

        export_keeper = RemoteKeeper(
            remote_store.api_url,
            remote_store.api_key,
            kp.config,
            project=remote_store.project,
        )

    if processor_lock_acquired_here:
        if not processor_lock.acquire(blocking=False):
            _daemon_logger.info("Daemon: another processor already running, exiting")
            kp.close()
            return

    _ver = _daemon_version()
    _daemon_logger.info(
        "Daemon started (pid=%d) keep-skill=%s python=%s store=%s",
        os.getpid(), _ver, sys.executable, kp._store_path,
    )
    try:
        (kp._store_path / ".processor.version").write_text(_ver)
    except Exception:
        pass

    # Materialize system notes before serving requests. On a fresh store the
    # first daemon-backed `get .meta/...` request would otherwise trigger
    # ensure_sysdocs() concurrently with the daemon loop's own startup work.
    # That race is enough to make SQLite fall over on brand-new stores.
    kp.ensure_sysdocs()

    _daemon_server, _port_path, _token_path = _start_daemon_query_server(
        kp,
        _daemon_logger,
        export_keeper=export_keeper,
        bind_host=bind_host,
        advertised_url=advertised_url,
        trusted_proxy=trusted_proxy,
    )

    from .shutdown import wait_or_shutdown

    shutdown_state = _install_daemon_signal_handlers(
        logger=_daemon_logger,
        port_path=_port_path,
        token_path=_token_path,
    )

    if kp.start_deferred_startup_maintenance():
        _daemon_logger.info("Deferred startup maintenance running in background")

    _release_stale_daemon_leases(kp, flow_worker_id, _daemon_logger)
    _enable_work_queue_auto_vacuum(kp, _daemon_logger)
    _log_daemon_startup_state(kp, _daemon_logger)

    _last_cleanup_ts = 0.0
    _CLEANUP_INTERVAL = 86400
    _CLEANUP_MAX_AGE = 86400
    _last_prune_ts = 0.0  # force immediate prune on first tick
    _PRUNE_INTERVAL = 86400
    _REPLENISH_INTERVAL = 1800
    _last_replenish_ts = _load_daemon_replenish_timestamp(kp._store_path)
    _pending_batch_limit = 64

    _version_file = kp._store_path / ".processor.version"

    try:
        pid_path.write_text(str(os.getpid()))
        while not shutdown_state["requested"]:
            _check_daemon_version_restart(_version_file, _ver, kp, _daemon_logger)
            _daemon_logger.debug("Tick: process_pending_work")
            flow_result = kp.process_pending_work(
                limit=1, worker_id=flow_worker_id,
                lease_seconds=180, shutdown_check=lambda: bool(shutdown_state["requested"]),
            )
            if shutdown_state["requested"]:
                break
            _daemon_logger.debug("Tick: process_pending")
            result = kp.process_pending(
                limit=_pending_batch_limit,
                shutdown_check=lambda: bool(shutdown_state["requested"]),
            )
            delegated = result.get("delegated", 0)
            if shutdown_state["requested"]:
                break

            _daemon_logger.debug("Tick: poll_watches")
            from .watches import poll_watches as _poll_watches
            watch_result = _poll_watches(kp)
            if watch_result["checked"] > 0:
                _daemon_logger.info(
                    "Watches: checked=%d changed=%d stale=%d errors=%d",
                    watch_result["checked"], watch_result["changed"],
                    watch_result["stale"], watch_result["errors"],
                )

            _daemon_logger.debug("Tick: poll_markdown_mirrors")
            from .markdown_mirrors import poll_markdown_mirrors as _poll_markdown_mirrors
            mirror_result = _poll_markdown_mirrors(kp, source_keeper=export_keeper)
            if mirror_result["checked"] > 0:
                _daemon_logger.info(
                    "Markdown mirrors: checked=%d exported=%d errors=%d",
                    mirror_result["checked"],
                    mirror_result["exported"],
                    mirror_result["errors"],
                )

            if shutdown_state["requested"]:
                break
            _last_cleanup_ts = _cleanup_temp_files_if_due(
                logger=_daemon_logger,
                last_cleanup_ts=_last_cleanup_ts,
                cleanup_interval=_CLEANUP_INTERVAL,
                cleanup_max_age=_CLEANUP_MAX_AGE,
            )
            _last_prune_ts = _prune_work_if_due(
                kp,
                last_prune_ts=_last_prune_ts,
                prune_interval=_PRUNE_INTERVAL,
            )
            _last_replenish_ts = _replenish_supernodes_if_due(
                kp,
                logger=_daemon_logger,
                last_replenish_ts=_last_replenish_ts,
                replenish_interval=_REPLENISH_INTERVAL,
            )
            if shutdown_state["requested"]:
                break

            _log_daemon_batch_result(
                logger=_daemon_logger,
                result=result,
                delegated=delegated,
                flow_result=flow_result,
            )
            flow_activity = _daemon_has_flow_activity(flow_result)
            if result["processed"] == 0 and result["failed"] == 0 and delegated == 0 and not flow_activity:
                if _maybe_wait_for_daemon_idle(
                    kp,
                    logger=_daemon_logger,
                    last_replenish_ts=_last_replenish_ts,
                    replenish_interval=_REPLENISH_INTERVAL,
                    wait_or_shutdown=wait_or_shutdown,
                ):
                    continue

                wait_or_shutdown(1)
                if shutdown_state["requested"]:
                    break
                flow_result = kp.process_pending_work(
                    limit=1, worker_id=flow_worker_id,
                    lease_seconds=180, shutdown_check=lambda: bool(shutdown_state["requested"]),
                )
                result = kp.process_pending(
                    limit=_pending_batch_limit,
                    shutdown_check=lambda: bool(shutdown_state["requested"]),
                )
                flow_activity = _daemon_has_flow_activity(flow_result)
                if (
                    result["processed"] == 0
                    and result["failed"] == 0
                    and result.get("delegated", 0) == 0
                    and not flow_activity
                ):
                    break
                _log_daemon_batch_result(
                    logger=_daemon_logger,
                    result=result,
                    delegated=result.get("delegated", 0),
                    flow_result=flow_result,
                    drain=True,
                )
    finally:
        _daemon_logger.info("Daemon shutting down")
        try:
            _daemon_server.stop()
        except Exception:
            pass
        for p in (_port_path, _token_path, pid_path):
            try:
                p.unlink()
            except OSError:
                pass
        kp.close()
        if export_keeper is not kp:
            export_keeper.close()
        processor_lock.release()


def _daemon_version() -> str:
    """Best-effort package version for daemon lifecycle logging."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("keep-skill")
    except Exception:
        return "unknown"


def _start_daemon_query_server(
    kp,
    daemon_logger,
    *,
    export_keeper=None,
    bind_host: str | None = None,
    advertised_url: str | None = None,
    trusted_proxy: bool | None = None,
):
    """Start the daemon HTTP server and publish discovery files."""
    from .daemon_server import DaemonServer

    daemon_port = int(os.environ.get("KEEP_DAEMON_PORT", "0")) or DAEMON_PORT
    daemon_bind_host = bind_host or os.environ.get("KEEP_DAEMON_BIND_HOST", "127.0.0.1")
    daemon_advertised_url = advertised_url or os.environ.get("KEEP_DAEMON_ADVERTISED_URL")
    daemon_trusted_proxy = trusted_proxy
    if daemon_trusted_proxy is None:
        daemon_trusted_proxy = os.environ.get("KEEP_DAEMON_TRUSTED_PROXY", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
    daemon_server = DaemonServer(
        kp,
        port=daemon_port,
        export_keeper=export_keeper,
        bind_host=daemon_bind_host,
        advertised_url=daemon_advertised_url,
        trusted_proxy=bool(daemon_trusted_proxy),
    )
    actual_port = daemon_server.start()
    port_path = kp._store_path / DAEMON_PORT_FILE
    token_path = kp._store_path / DAEMON_TOKEN_FILE
    port_path.write_text(str(actual_port))
    token_path.touch(mode=0o600, exist_ok=True)
    token_path.write_text(daemon_server.auth_token)
    daemon_logger.info("Query server on %s:%d", daemon_bind_host, actual_port)
    return daemon_server, port_path, token_path


def _install_daemon_signal_handlers(*, logger, port_path: Path, token_path: Path) -> dict[str, bool]:
    """Install signal handlers that hide the daemon immediately on shutdown."""
    from .shutdown import request_shutdown

    state = {"requested": False}

    def handle_signal(signum, frame):
        state["requested"] = True
        request_shutdown()
        for path in (port_path, token_path):
            try:
                path.unlink()
            except OSError:
                pass
        from .providers.http import close_http_session

        close_http_session()
        logger.info("Received signal %d, shutting down", signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    _install_daemon_stack_dump_signal(logger)
    return state


def _install_daemon_stack_dump_signal(logger) -> None:
    """Register a non-terminating signal for live daemon stack diagnostics.

    The daemon redirects stderr to keep-ops.log, so SIGUSR1 gives operators a
    cheap way to capture all Python thread stacks while a process is still
    wedged.  This complements slow SQLite query fingerprints, which only help
    once the relevant statement reaches Python code.
    """
    signum = getattr(signal, "SIGUSR1", None)
    if signum is None:
        logger.debug("SIGUSR1 unavailable; daemon stack dump signal disabled")
        return
    try:
        import faulthandler

        faulthandler.enable(file=sys.stderr, all_threads=True)
        faulthandler.register(signum, file=sys.stderr, all_threads=True, chain=False)
        logger.info("Daemon stack dumps enabled: kill -USR1 %d", os.getpid())
    except Exception as exc:
        logger.warning("Failed to install daemon stack dump signal: %s", exc)


def _release_stale_daemon_leases(kp, flow_worker_id: str, daemon_logger) -> None:
    """Release work leases left behind by a prior daemon instance."""
    wq = kp._get_work_queue()
    released = wq.release_stale_leases(flow_worker_id) if wq is not None else 0
    if released:
        daemon_logger.info("Released %d stale leases from previous daemon", released)


def _enable_work_queue_auto_vacuum(kp, daemon_logger) -> None:
    """Enable auto_vacuum on continuation.db if not already set."""
    wq = kp._get_work_queue()
    if wq is None:
        return
    if wq.enable_auto_vacuum():
        daemon_logger.info("Enabled auto_vacuum on continuation.db (one-time VACUUM)")


def _prune_work_if_due(kp, *, last_prune_ts: float, prune_interval: float) -> float:
    """Run periodic work queue pruning."""
    now = time.time()
    if now - last_prune_ts < prune_interval:
        return last_prune_ts
    try:
        wq = kp._get_work_queue()
        if wq is not None:
            wq.prune()
    except Exception:
        logging.getLogger("keep.console_support.daemon").warning(
            "Work queue prune failed", exc_info=True,
        )
    return now


def _log_daemon_startup_state(kp, daemon_logger) -> None:
    """Log initial queue and provider state once the daemon is reachable."""
    mirror_count = 0
    try:
        from .markdown_mirrors import list_markdown_mirrors
        mirror_count = len(list_markdown_mirrors(kp))
    except Exception:
        pass
    daemon_logger.info(
        "Queue: %d pending, %d flow, %d failed",
        kp._pending_queue.count(),
        kp.pending_work_count() if hasattr(kp, "pending_work_count") else 0,
        0,
    )
    daemon_logger.info("Markdown mirrors: %d configured", mirror_count)
    daemon_logger.info(
        "Embedding: %s/%s",
        kp._config.embedding.name if kp._config.embedding else "none",
        kp._config.embedding.params.get("model", "") if kp._config.embedding else "",
    )


def _load_daemon_replenish_timestamp(store_path: Path) -> float:
    """Restore the last supernode replenishment timestamp from timer state."""
    try:
        from .timer_state import read_timer_state

        saved_timers = read_timer_state(store_path)
        return saved_timers.get("supernode-replenish", {}).get("last_run", 0.0)
    except Exception:
        return 0.0


def _replenish_supernodes_if_due(
    kp,
    *,
    logger,
    last_replenish_ts: float,
    replenish_interval: float,
) -> float:
    """Run periodic supernode queue replenishment when its timer is due."""
    now = time.time()
    if now - last_replenish_ts < replenish_interval:
        return last_replenish_ts

    try:
        enqueued = kp.replenish_supernode_queue()
        detail = f"{enqueued} enqueued" if enqueued else "no candidates"
        if enqueued > 0:
            logger.info("Replenished %d supernode review(s)", enqueued)
    except Exception as exc:
        detail = f"error: {exc}"
        logger.debug("Supernode replenishment error: %s", exc)

    try:
        from .timer_state import write_timer_event

        write_timer_event(
            kp._store_path,
            "supernode-replenish",
            interval=replenish_interval,
            detail=detail,
        )
    except Exception as timer_exc:
        logger.warning("Failed to write timer state: %s", timer_exc)

    return now


def _cleanup_temp_files_if_due(
    *,
    logger,
    last_cleanup_ts: float,
    cleanup_interval: float,
    cleanup_max_age: float,
) -> float:
    """Run periodic temp-file cleanup for cache directories."""
    now = time.time()
    if now - last_cleanup_ts < cleanup_interval:
        return last_cleanup_ts

    cache_dirs = [Path.home() / ".cache" / "keep" / "email-att"]
    total_removed = 0
    for cache_dir in cache_dirs:
        if not cache_dir.is_dir():
            continue
        for entry in cache_dir.iterdir():
            if not entry.is_dir():
                continue
            try:
                age = now - entry.stat().st_mtime
                if age > cleanup_max_age:
                    shutil.rmtree(entry)
                    total_removed += 1
            except OSError:
                pass
    if total_removed:
        logger.info("Cleaned up %d temp directories", total_removed)
    return now


def _check_daemon_version_restart(version_file: Path, running_version: str, kp, daemon_logger) -> None:
    """Exec-restart the daemon when the requested package version changes."""
    try:
        if not version_file.exists():
            return
        requested = version_file.read_text().strip()
        if requested and requested != running_version:
            daemon_logger.info(
                "Version changed (%s → %s), restarting daemon",
                running_version,
                requested,
            )
            try:
                kp.close()
            except Exception:
                pass
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as exc:
        daemon_logger.debug("Version check failed: %s", exc)


def _daemon_has_flow_activity(flow_result: dict[str, Any]) -> bool:
    """Return True if a flow work batch claimed or changed any items."""
    return (
        int(flow_result.get("claimed", 0)) > 0
        or int(flow_result.get("processed", 0)) > 0
        or int(flow_result.get("failed", 0)) > 0
        or int(flow_result.get("dead_lettered", 0)) > 0
    )


def _log_daemon_batch_result(*, logger, result: dict[str, Any], delegated: int, flow_result: dict[str, Any], drain: bool = False) -> None:
    """Emit a consistent batch-progress log line for daemon work ticks."""
    flow_processed = int(flow_result.get("processed", 0))
    flow_failed = int(flow_result.get("failed", 0)) + int(flow_result.get("dead_lettered", 0))
    if (
        int(result.get("processed", 0)) == 0
        and int(result.get("failed", 0)) == 0
        and int(delegated) == 0
        and flow_processed == 0
        and flow_failed == 0
    ):
        return
    label = "Daemon batch (drain)" if drain else "Daemon batch"
    logger.info(
        "%s: processed=%d failed=%d delegated=%d flow_processed=%d flow_failed=%d",
        label,
        result["processed"],
        result["failed"],
        delegated,
        flow_processed,
        flow_failed,
    )


def _maybe_wait_for_daemon_idle(
    kp,
    *,
    logger,
    last_replenish_ts: float,
    replenish_interval: float,
    wait_or_shutdown,
) -> bool:
    """Sleep when the daemon is idle but has background timers or queued retries."""
    from .markdown_mirrors import (
        has_active_markdown_mirrors,
        list_markdown_mirrors,
        next_markdown_mirror_delay,
    )
    from .watches import has_active_watches, load_watches, next_check_delay

    delegated_remaining = (
        kp._pending_queue.count_delegated()
        if hasattr(kp._pending_queue, "count_delegated")
        else 0
    )
    flow_remaining = kp.pending_work_count() if hasattr(kp, "pending_work_count") else 0
    pending_remaining = kp._pending_queue.count()
    if delegated_remaining > 0:
        logger.info("Waiting for %d delegated tasks", delegated_remaining)
        wait_or_shutdown(5)
        return True
    if flow_remaining > 0:
        logger.info("Waiting for %d flow work items", flow_remaining)
        wait_or_shutdown(1)
        return True
    if pending_remaining > 0:
        logger.info("Waiting for %d pending items (retry backoff)", pending_remaining)
        wait_or_shutdown(5)
        return True

    has_timers = has_active_watches(kp)
    mirrors = list_markdown_mirrors(kp) if not has_timers else []
    if not has_timers:
        has_timers = any(m.enabled for m in mirrors)
    if not has_timers:
        has_timers = (
            last_replenish_ts == 0
            or (time.time() - last_replenish_ts) < replenish_interval
        )
    if not has_timers:
        return False

    if has_active_watches(kp):
        delay = next_check_delay(load_watches(kp))
    elif mirrors and any(m.enabled for m in mirrors):
        delay = next_markdown_mirror_delay(mirrors)
    else:
        time_to_replenish = max(0, replenish_interval - (time.time() - last_replenish_ts))
        delay = min(time_to_replenish, 60.0)
    delay = max(1.0, min(delay, 60.0))
    logger.debug("Sleeping %.1fs (timer events pending)", delay)
    wait_or_shutdown(delay)
    return True


def print_pending_list_lightweight(store_path: "Path") -> None:
    """Print pending work items without creating a Keeper.

    Opens SQLite queues directly — no ChromaDB, no embedding models,
    no lock contention with the running daemon.
    """
    from .pending_summaries import PendingSummaryQueue
    from .work_queue import WorkQueue

    pq = PendingSummaryQueue(store_path / PENDING_SUMMARIES_DB)
    items = pq.list_pending()
    failed = pq.list_failed()
    try:
        wq = WorkQueue(store_path / WORK_QUEUE_DB)
        flow_items = wq.list_pending(limit=-1)
    except Exception:
        wq = None
        flow_items = []

    if not items and not failed and not flow_items:
        typer.echo("Nothing pending.")
    else:
        if items:
            for item in items:
                retry_str = f" (retry after {item['retry_after']})" if item.get("retry_after") else ""
                typer.echo(f"  {item['task_type']:15s} {item['supersede_key'] or item['work_id']}{retry_str}")
        if flow_items:
            if items:
                typer.echo()
            if wq:
                by_kind = wq.count_by_kind()
                for kind, count in by_kind.items():
                    if kind != "flow":
                        typer.echo(f"  {kind:20s} {count}")
            flow_by_state: dict[str, list] = {}
            for fi in flow_items:
                if fi.get("kind") != "flow":
                    continue
                inp = fi.get("input") or {}
                state = inp.get("state", "unknown")
                flow_by_state.setdefault(state, []).append(fi)
            for state, state_items in sorted(flow_by_state.items()):
                if len(state_items) <= 3:
                    for si in state_items:
                        inp = si.get("input") or {}
                        target = inp.get("item_id") or inp.get("params", {}).get("item_id", "")
                        if target:
                            typer.echo(f"  flow:{state:16s} {target}")
                        else:
                            typer.echo(f"  flow:{state}")
                else:
                    typer.echo(f"  flow:{state:16s} ({len(state_items)} items)")
        if failed:
            typer.echo(f"\nFailed ({len(failed)}):")
            for item in failed[:10]:
                error = item.get("last_error", "unknown")
                typer.echo(f"  {item['task_type']:15s} {item['id']}: {error}")

    # Timer events
    try:
        from .timer_state import KNOWN_TIMERS, format_timer_events
    except Exception:
        format_timer_events = None
        KNOWN_TIMERS = {}
    if KNOWN_TIMERS:
        typer.echo("\nTimer events:")
        if format_timer_events:
            try:
                lines = format_timer_events(store_path)
                for line in lines:
                    typer.echo(line)
            except Exception:
                pass

    pq.close()
    if wq:
        wq.close()


def print_pending_interactive(kp) -> None:
    """Show pending status, ensure daemon running, tail log."""
    pending_count = kp.pending_count()
    flow_pending_count = kp.pending_work_count() if hasattr(kp, "pending_work_count") else 0
    queue_stats = kp._pending_queue.stats()
    failed_count = queue_stats.get("failed", 0)
    processing_count = queue_stats.get("processing", 0)
    mirror_count = 0

    _has_timer_events = False
    try:
        from .watches import has_active_watches
        _has_timer_events = has_active_watches(kp)
    except Exception:
        pass
    try:
        from .markdown_mirrors import list_markdown_mirrors
        mirror_count = len([entry for entry in list_markdown_mirrors(kp) if entry.enabled])
        if mirror_count:
            _has_timer_events = True
    except Exception:
        pass
    if not _has_timer_events:
        try:
            from .timer_state import KNOWN_TIMERS, read_timer_state
            _timer_state = read_timer_state(kp._store_path)
            for name, info in KNOWN_TIMERS.items():
                saved = _timer_state.get(name, {})
                last_run = saved.get("last_run", 0)
                interval = info.get("interval", 0)
                if interval and (time.time() - last_run) >= interval:
                    _has_timer_events = True
                    break
        except Exception:
            pass

    if pending_count == 0 and processing_count == 0 and flow_pending_count == 0 and not _has_timer_events:
        if failed_count:
            typer.echo(f"Nothing pending. {failed_count} failed (use --retry to requeue).", err=True)
            failed_items = kp._pending_queue.list_failed()
            for item in failed_items[:5]:
                error = item.get("last_error", "unknown")
                typer.echo(f"  {item['id']} ({item['task_type']}): {error}", err=True)
            if len(failed_items) > 5:
                typer.echo(f"  ... and {len(failed_items) - 5} more", err=True)
        else:
            typer.echo("Nothing pending.")
        return

    if pending_count > 0 or processing_count > 0 or flow_pending_count > 0:
        typer.echo(_queue_status_line(kp, queue_stats), err=True)
    elif _has_timer_events:
        typer.echo("No work queued, but timer events need servicing.", err=True)
    if mirror_count:
        typer.echo(f"Markdown mirrors active: {mirror_count}", err=True)

    if not kp._is_processor_running():
        kp._spawn_processor()
        typer.echo("Started background processor.", err=True)
    else:
        typer.echo("Background processor already running.", err=True)

    log_path = kp._store_path / OPS_LOG_FILE
    _tail_ops_log(log_path, kp)


def _queue_status_line(kp, queue_stats: dict) -> str:
    """Build a consistent queue status string like '2 queued, 1 processing (1 ocr)'."""
    pending = queue_stats.get("pending", 0)
    processing = queue_stats.get("processing", 0)
    delegated = queue_stats.get("delegated", 0)
    failed = queue_stats.get("failed", 0)
    flow_pending = 0
    if hasattr(kp, "pending_work_count"):
        try:
            flow_pending = int(kp.pending_work_count())
        except Exception:
            flow_pending = 0

    by_type = kp.pending_stats_by_type()
    type_parts = ", ".join(f"{c} {t}" for t, c in sorted(by_type.items()))

    # Flow work queue breakdown by kind, sorted by processing priority
    flow_by_kind = ""
    if flow_pending:
        try:
            wq = kp._get_work_queue()
            by_kind = wq.count_by_kind()  # pre-sorted by priority
            if by_kind:
                flow_by_kind = ", ".join(f"{c} {k}" for k, c in by_kind.items())
        except Exception:
            pass

    parts = [f"{pending} queued"]
    if processing:
        parts.append(f"{processing} processing")
    if delegated:
        parts.append(f"{delegated} delegated")
    if flow_pending:
        parts.append(f"{flow_pending} flow")

    line = ", ".join(parts)
    if type_parts:
        line += f" ({type_parts})"
    elif flow_by_kind:
        line += f" ({flow_by_kind})"
    if failed:
        line += f" + {failed} failed"
    return line


def _tail_ops_log(log_path: Path, kp) -> None:
    """Tail the ops log, showing new lines until daemon finishes or Ctrl-C."""
    # Grace period for daemon startup (takes a moment to acquire lock)
    time.sleep(1.0)

    try:
        # Ensure log file exists (may not on fresh stores)
        log_path.touch(exist_ok=True)
        with open(log_path) as f:
            f.seek(0, 2)  # Seek to end
            idle_checks = 0
            while True:
                line = f.readline()
                if line:
                    typer.echo(line.rstrip(), err=True)
                    idle_checks = 0
                else:
                    # No new line — check if daemon is still running
                    idle_checks += 1
                    if idle_checks >= 5 and not kp._is_processor_running():
                        stats = kp._pending_queue.stats()
                        flow_pending = kp.pending_work_count() if hasattr(kp, "pending_work_count") else 0
                        if (
                            stats.get("pending", 0) == 0
                            and stats.get("processing", 0) == 0
                            and flow_pending == 0
                        ):
                            typer.echo("Done.", err=True)
                        else:
                            typer.echo(_queue_status_line(kp, stats), err=True)
                        break
                    time.sleep(0.5)
    except (KeyboardInterrupt, EOFError):
        stats = kp._pending_queue.stats()
        typer.echo(
            f"\nDetached. Daemon still running. {_queue_status_line(kp, stats)}",
            err=True,
        )


# -----------------------------------------------------------------------------
# Data Management
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def doctor(
    store: StoreOption = None,
    log: Annotated[bool, typer.Option(
        "--log", "-l", help="Tail the ops log (Ctrl-C to stop)"
    )] = False,
    use_faulthandler: Annotated[bool, typer.Option(
        "--faulthandler", help="Enable faulthandler for native crash traces"
    )] = False,
):
    """Diagnostic checks for debugging setup and crash issues."""
    if log:
        from .config import load_or_create_config
        from .paths import get_config_dir, get_default_store_path
        actual_store = store if store is not None else _get_store_override()
        config_dir = Path(actual_store).resolve() if actual_store else get_config_dir()
        cfg = load_or_create_config(config_dir)
        sp = Path(get_default_store_path(cfg) if actual_store is None else actual_store)
        log_path = sp / OPS_LOG_FILE
        if not log_path.exists():
            typer.echo(f"No ops log at {log_path}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Tailing {log_path} (Ctrl-C to stop)\n", err=True)
        try:
            with open(log_path) as f:
                f.seek(0, 2)  # seek to end
                while True:
                    line = f.readline()
                    if line:
                        typer.echo(line.rstrip())
                    else:
                        time.sleep(0.3)
        except (KeyboardInterrupt, EOFError):
            pass
        return

    if use_faulthandler:
        import faulthandler
        faulthandler.enable()
        typer.echo("faulthandler enabled (native crash traces will print to stderr)\n")

    def ok(msg):
        typer.echo(f"  [ok]   {msg}")

    def fail(msg):
        typer.echo(f"  [FAIL] {msg}")

    def warn(msg):
        typer.echo(f"  [WARN] {msg}")

    # 1. Environment
    try:
        kv = importlib.metadata.version("keep-skill")
    except Exception:
        kv = "?"
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    plat = f"{platform.system()} {platform.release()} ({platform.machine()})"
    ok(f"Environment: Python {py_ver}, {plat}, keep-skill {kv}")

    # 2. Key packages
    pkg_names = ["chromadb", "sentence_transformers", "torch", "pydantic", "sqlite3"]
    pkg_versions = {}
    for pkg in pkg_names:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", getattr(mod, "version", "imported"))
        except ImportError:
            pkg_versions[pkg] = "not installed"
    pkg_str = ", ".join(f"{k} {v}" for k, v in pkg_versions.items())
    ok(f"Packages: {pkg_str}")

    # 3. Store config
    from .config import load_or_create_config
    from .paths import get_config_dir, get_default_store_path
    try:
        actual_store = store if store is not None else _get_store_override()
        config_dir = Path(actual_store).resolve() if actual_store else get_config_dir()
        cfg = load_or_create_config(config_dir)
        store_path = Path(get_default_store_path(cfg) if actual_store is None else actual_store)
        emb_name = cfg.embedding.name if cfg and cfg.embedding else "none"
        sum_name = cfg.summarization.name if cfg and cfg.summarization else "none"
        ok(f"Store config: {store_path} (embedding: {emb_name}, summarization: {sum_name})")
    except Exception as e:
        fail(f"Store config: {e}")
        store_path = None

    # 4. SQLite (DocumentStore)
    db_path = store_path / DOCUMENTS_DB if store_path else None
    if db_path and db_path.exists():
        try:
            from .document_store import DocumentStore
            ds = DocumentStore(db_path)
            doc_count = ds.count("default")
            ver_count = ds.count_versions("default")
            ok(f"SQLite: {doc_count} documents, {ver_count} versions")
            ds.close()
        except Exception as e:
            fail(f"SQLite: {e}")
    elif store_path:
        ok("SQLite: no documents.db yet (new store)")
    else:
        fail("SQLite: skipped (no store path)")

    # 5. Embedding provider
    if cfg and cfg.embedding:
        try:
            from .providers.base import get_registry
            registry = get_registry()
            provider = registry.create_embedding(cfg.embedding.name, cfg.embedding.params)
            t0 = time.perf_counter()
            vec = provider.embed("keep doctor test")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            dim = len(vec)
            model = provider_model_name(provider, "?")
            ok(f"Embedding: {model}, dim={dim}, {elapsed_ms:.0f}ms")
        except Exception as e:
            fail(f"Embedding: {e}")
    else:
        ok("Embedding: no provider configured")

    # 6. Summarization provider
    if cfg and cfg.summarization and cfg.summarization.name != "passthrough":
        try:
            from .providers.base import get_registry
            registry = get_registry()
            provider = registry.create_summarization(cfg.summarization.name, cfg.summarization.params)
            t0 = time.perf_counter()
            result = provider.summarize("The quick brown fox jumps over the lazy dog.")
            elapsed_ms = (time.perf_counter() - t0) * 1000
            model = provider_model_name(provider, cfg.summarization.name)
            ok(f"Summarization: {model}, {elapsed_ms:.0f}ms")
        except Exception as e:
            fail(f"Summarization: {e}")
    else:
        ok(f"Summarization: {'passthrough' if cfg and cfg.summarization else 'none'}")

    # 7. Media describer
    if cfg and cfg.media:
        try:
            from .providers.base import get_registry
            registry = get_registry()
            provider = registry.create_media(cfg.media.name, cfg.media.params)
            model = provider_model_name(provider, cfg.media.name)
            ok(f"Media: {cfg.media.name} ({model})")
        except Exception as e:
            fail(f"Media: {e}")
    elif cfg:
        from .config import (
            OLLAMA_DEFAULT_OCR_MODEL,
            OLLAMA_DEFAULT_VISION_MODEL,
            ProviderConfig,
            _detect_ollama,
            _ollama_vision_models,
            ollama_pull,
            save_config,
        )
        ollama = _detect_ollama()
        if ollama:
            vision = _ollama_vision_models(ollama["models"])
            if vision:
                # Vision model available but not configured — auto-configure
                model_name = vision[0]
                params: dict[str, Any] = {"model": model_name}
                if ollama["base_url"] != "http://localhost:11434":
                    params["base_url"] = ollama["base_url"]
                cfg.media = ProviderConfig("ollama", params)
                save_config(cfg)
                ok(f"Media: auto-configured ollama ({model_name})")
            else:
                # Ollama running but no vision model — pull one
                target = OLLAMA_DEFAULT_VISION_MODEL
                typer.echo(f"         Pulling {target}...", nl=False)
                last_status = [""]
                def _progress(s: str) -> None:
                    if s != last_status[0]:
                        last_status[0] = s
                        typer.echo(f"\r         Pulling {target}... {s}    ", nl=False)
                if ollama_pull(target, ollama["base_url"], on_progress=_progress):
                    typer.echo(f"\r         Pulling {target}... done.           ")
                    params = {"model": target}
                    if ollama["base_url"] != "http://localhost:11434":
                        params["base_url"] = ollama["base_url"]
                    cfg.media = ProviderConfig("ollama", params)
                    save_config(cfg)
                    ok(f"Media: pulled and configured ollama ({target})")
                else:
                    typer.echo()
                    warn(f"Media: failed to pull {target}")
        else:
            ok("Media: none available (metadata-only indexing)")

    # 7b. Content extractor (OCR for PDFs/images)
    if cfg and cfg.content_extractor:
        # Verify the model is actually available if it's Ollama
        if cfg.content_extractor.name == "ollama":
            from .config import OLLAMA_DEFAULT_OCR_MODEL, _detect_ollama, _ollama_has_model, ollama_pull, save_config
            ce_model = cfg.content_extractor.params.get("model", OLLAMA_DEFAULT_OCR_MODEL)
            if _ollama_has_model(ce_model):
                ok(f"Content extractor: {cfg.content_extractor.name} ({ce_model})")
            else:
                typer.echo(f"         Pulling {ce_model}...", nl=False)
                base_url = cfg.content_extractor.params.get("base_url")
                last_status = [""]
                def _ce_progress(s: str) -> None:
                    if s != last_status[0]:
                        last_status[0] = s
                        typer.echo(f"\r         Pulling {ce_model}... {s}    ", nl=False)
                if ollama_pull(ce_model, base_url, on_progress=_ce_progress):
                    typer.echo(f"\r         Pulling {ce_model}... done.           ")
                    ok(f"Content extractor: pulled {ce_model}")
                else:
                    typer.echo()
                    warn(f"Content extractor: {ce_model} not available, pull it: ollama pull {ce_model}")
        else:
            ok(f"Content extractor: {cfg.content_extractor.name}")
    elif cfg:
        from .config import OLLAMA_DEFAULT_OCR_MODEL, ProviderConfig, _detect_ollama, ollama_pull, save_config
        ollama = _detect_ollama()
        if ollama:
            # Ollama running — pull OCR model and configure
            target = OLLAMA_DEFAULT_OCR_MODEL
            if any(m.split(":")[0] == target.split(":")[0] for m in ollama["models"]):
                # Model already present, just configure
                params: dict[str, Any] = {"model": target}
                if ollama["base_url"] != "http://localhost:11434":
                    params["base_url"] = ollama["base_url"]
                cfg.content_extractor = ProviderConfig("ollama", params)
                save_config(cfg)
                ok(f"Content extractor: auto-configured ollama ({target})")
            else:
                typer.echo(f"         Pulling {target}...", nl=False)
                last_status = [""]
                def _ocr_progress(s: str) -> None:
                    if s != last_status[0]:
                        last_status[0] = s
                        typer.echo(f"\r         Pulling {target}... {s}    ", nl=False)
                if ollama_pull(target, ollama["base_url"], on_progress=_ocr_progress):
                    typer.echo(f"\r         Pulling {target}... done.           ")
                    params = {"model": target}
                    if ollama["base_url"] != "http://localhost:11434":
                        params["base_url"] = ollama["base_url"]
                    cfg.content_extractor = ProviderConfig("ollama", params)
                    save_config(cfg)
                    ok(f"Content extractor: pulled and configured ollama ({target})")
                else:
                    typer.echo()
                    warn(f"Content extractor: failed to pull {target}")
        else:
            ok("Content extractor: none (PDFs use text extraction only)")

    # 8. Analyzer
    if cfg and cfg.analyzer:
        try:
            from .providers.base import get_registry
            registry = get_registry()
            provider = registry.create_analyzer(cfg.analyzer.name, cfg.analyzer.params)
            ok(f"Analyzer: {cfg.analyzer.name}")
        except Exception as e:
            fail(f"Analyzer: {e}")
    else:
        ok("Analyzer: default (uses summarization provider)")

    # 9. ChromaDB
    if store_path and (store_path / "chroma").exists():
        try:
            from .store import ChromaStore
            cs = ChromaStore(store_path)
            collections = cs.list_collections()
            counts = {c: cs.count(c) for c in collections}
            parts = [f"{c} ({n})" for c, n in counts.items()]
            ok(f"ChromaDB: {', '.join(parts) if parts else 'no collections'}")
        except Exception as e:
            fail(f"ChromaDB: {e}")
    elif store_path:
        ok("ChromaDB: no chroma/ yet (new store)")
    else:
        fail("ChromaDB: skipped (no store path)")

    # 10. Model locks
    if store_path:
        from .model_lock import ModelLock
        for lock_name in [".embedding.lock", ".summarization.lock"]:
            lock_file = store_path / lock_name
            if lock_file.exists():
                probe = ModelLock(lock_file)
                if probe.is_locked():
                    fail(f"Lock: {lock_name} is held by another process (lsof {lock_file})")
                else:
                    ok(f"Lock: {lock_name} available")
            else:
                ok(f"Lock: {lock_name} (not yet created)")
    else:
        ok("Lock: skipped (no store path)")

    # 11. Round-trip (temp store, isolates stack from store data)
    from .config import ProviderConfig, StoreConfig
    from .api import Keeper
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp(prefix="keep_doctor_")
        tmp_path = Path(tmp_dir)
        # Minimal config: use store's embedding provider, passthrough summarization
        test_config = StoreConfig(
            path=tmp_path,
            embedding=cfg.embedding if cfg else None,
            summarization=ProviderConfig("passthrough", {"max_chars": 10000}),
            max_summary_length=10000,
        )
        t0 = time.perf_counter()
        kp = Keeper(tmp_dir, config=test_config)
        kp.put("The quick brown fox jumps over the lazy dog", id="doctor_test")
        item = kp.get("doctor_test")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if item and "fox" in item.summary:
            ok(f"Round-trip: put + get in {elapsed_ms:.0f}ms")
        else:
            fail("Round-trip: put succeeded but get returned unexpected result")
    except Exception as e:
        fail(f"Round-trip: {e}")
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # 12. System doc validation
    if store_path:
        try:
            from .validate import validate_system_doc
            kp = _get_keeper(store)
            doc_coll = kp._resolve_doc_collection()
            docs_by_id: dict[str, tuple[str, dict]] = {}
            for sys_prefix in (".tag/", ".meta/", ".prompt/", ".state/"):
                for rec in kp._document_store.query_by_id_prefix(doc_coll, sys_prefix):
                    doc_id = str(getattr(rec, "id", ""))
                    if doc_id:
                        summary = str(getattr(rec, "summary", "") or "")
                        raw_tags = getattr(rec, "tags", None)
                        tags_dict = dict(raw_tags) if isinstance(raw_tags, dict) else {}
                        docs_by_id[doc_id] = (summary, tags_dict)
            if docs_by_id:
                errors = 0
                warnings = 0
                bad_docs = []
                for doc_id in sorted(docs_by_id):
                    content, tags_dict = docs_by_id[doc_id]
                    result = validate_system_doc(doc_id, content, tags_dict)
                    errors += len(result.errors)
                    warnings += len(result.warnings)
                    if result.errors:
                        bad_docs.append(doc_id)
                if errors:
                    fail(f"System docs: {len(docs_by_id)} docs, {errors} error(s) in {', '.join(bad_docs)}")
                elif warnings:
                    ok(f"System docs: {len(docs_by_id)} docs, {warnings} warning(s)")
                else:
                    ok(f"System docs: {len(docs_by_id)} docs validated")
            else:
                ok("System docs: none found (new store)")
        except Exception as e:
            fail(f"System docs: {e}")
    else:
        ok("System docs: skipped (no store path)")

    typer.echo()
