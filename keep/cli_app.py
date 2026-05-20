"""Typer command app for keep.

Primarily daemon-backed commands for note/query flows, plus a small set of
retained local or delegated commands for setup, daemon control, MCP, and
data import/export.
"""

import json
import os
import shutil
import sys
import importlib.metadata
import importlib.resources
import select
import signal
import subprocess as sp
import tempfile
import time as _time
from pathlib import Path
from typing import Annotated, Optional
from urllib.parse import quote

import typer

from keep.help import get_help_topic
from keep.mcp import main as mcp_main
from keep.types import INTERNAL_TAGS, local_date, note_display_name
from keep.utils import _extract_markdown_frontmatter
from keep.validate import state_doc_diagram

from . import api as _api
from . import console_support as _console_support
from . import daemon_client as _daemon_client
from . import markdown_import as _markdown_import
from . import markdown_mirrors as _markdown_mirrors
from .config import load_or_create_config
from .daemon_client import get_port as _daemon_get_port
from .const import DAEMON_PORT_FILE, DAEMON_TOKEN_FILE, STATE_PROMPT
from .console_support import (
    _format_config_with_defaults,
    _get_config_value,
    _get_keeper,
    _progress_bar,
    doctor as doctor_impl,
    get_tool_directory,
    print_pending_interactive,
    print_pending_list_lightweight,
    run_pending_daemon,
)
from .ignore import merge_excludes, parse_ignore_patterns
from . import markdown_export as _markdown_export
from .markdown_import import count_markdown_import_files
from .markdown_mirrors import run_markdown_export_once
from .paths import get_config_dir, get_default_store_path
from .setup_wizard import run_wizard
from .utils import _list_directory_files
from .workstream import derive_workstream_slug

_ExportCollisionError = _markdown_export._ExportCollisionError
_MAX_FILENAME_BYTES = _markdown_export._MAX_FILENAME_BYTES
_id_to_rel_path = _markdown_export.id_to_rel_path
_md_link_target = _markdown_export._md_link_target
_render_doc_markdown = _markdown_export._render_doc_markdown
_render_part_markdown = _markdown_export._render_part_markdown
_render_version_markdown = _markdown_export._render_version_markdown
_write_markdown_export = _markdown_export.write_markdown_export

app = typer.Typer(
    name="keep",
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode=None,
)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _q(id: str) -> str:
    """URL-encode an ID for path segments."""
    return quote(id, safe="")


# ---------------------------------------------------------------------------
# Stdin JSON template expansion
# ---------------------------------------------------------------------------
# Hooks pipe JSON on stdin.  ${.field}, ${.field|text} (strip XML tags),
# and ${.field:N} (truncate to N chars) expand from that JSON.

import re

_TEMPLATE_RE = re.compile(r'\$\{\.([A-Za-z_][A-Za-z0-9_]*)(?:\|([a-z]+))?(?::(\d+))?\}')
# Strip XML-style tags from text (used by |text template filter)
_XML_TAG_RE = re.compile(r'<([a-z][\w-]*)[\s>].*?</\1>', re.DOTALL)
_STDIN_JSON_SENTINEL = object()
_stdin_json_cache = _STDIN_JSON_SENTINEL


def _has_stdin_data() -> bool:
    try:
        if sys.stdin.isatty():
            return False
        # Use select to avoid hanging on socket stdin (exec sandboxes)
        return bool(select.select([sys.stdin], [], [], 0)[0])
    except Exception:
        return False


def _read_stdin_json() -> dict:
    """Read and cache JSON object from stdin.  Returns {} on failure."""
    global _stdin_json_cache
    if _stdin_json_cache is not _STDIN_JSON_SENTINEL:
        return _stdin_json_cache  # type: ignore[return-value]
    _stdin_json_cache = {}
    if _has_stdin_data():
        try:
            raw = sys.stdin.read()
            obj = json.loads(raw)
            if isinstance(obj, dict):
                _stdin_json_cache = obj
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            pass
    return _stdin_json_cache


def _has_templates(s: str | None) -> bool:
    return s is not None and '${.' in s


def _expand_template(s: str, data: dict) -> str:
    """Expand ${.field}, ${.field|filter}, and ${.field:N} in *s* from *data*.

    Filters: |text — strip XML-style tags from the value.
    """
    def _replace(m: re.Match) -> str:
        key, filt, limit = m.group(1), m.group(2), m.group(3)
        val = data.get(key)
        if val is None:
            return ''
        result = str(val)
        if filt == 'text':
            result = _XML_TAG_RE.sub('', result).strip()
        if limit:
            result = result[:int(limit)]
        return result
    return _TEMPLATE_RE.sub(_replace, s)


def _expand_stdin_templates(*strings: str | None) -> tuple[str | None, ...]:
    """Expand ${.field} templates in strings from stdin JSON."""
    if not any(_has_templates(s) for s in strings):
        return strings
    data = _read_stdin_json()
    return tuple(
        _expand_template(s, data) if _has_templates(s) else s
        for s in strings
    )


def _expand_stdin_tag_list(tags: list[str] | None) -> list[str] | None:
    """Expand templates in a tag list."""
    if not tags or not any(_has_templates(t) for t in tags):
        return tags
    data = _read_stdin_json()
    return [_expand_template(t, data) if _has_templates(t) else t for t in tags]


def _read_stdin_text() -> str:
    """Read stdin as strict UTF-8 text.

    Python's text wrapper may use a permissive error handler for stdin on some
    platforms, so decode from the buffered bytes directly when possible.
    """
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is not None:
        raw = buffer.read()
        return raw.decode("utf-8")

    text = sys.stdin.read()
    if any(0xDC80 <= ord(ch) <= 0xDCFF for ch in text):
        raise ValueError("stdin contains surrogate-escaped bytes")
    return text


def _get_export_host():
    store = Path(_global_store).resolve() if _global_store else None
    return _get_keeper(store)


def _supports_local_markdown_export(host: object) -> bool:
    return _markdown_export.supports_local_markdown_export_graph(host)


def _supports_markdown_export(host: object) -> bool:
    if _supports_local_markdown_export(host):
        return True
    if hasattr(host, "supports_capability"):
        try:
            return bool(host.supports_capability("export_bundle"))
        except Exception:
            return False
    return hasattr(host, "export_bundle")


def _require_markdown_export_host():
    host = _get_export_host()
    if _supports_markdown_export(host):
        return host
    host.close()
    typer.echo(
        "Error: markdown export requires a host with local markdown access "
        "or note-bundle export support.",
        err=True,
    )
    raise SystemExit(1)


def _daemon_error_message(body: dict, default: str = "unknown") -> str:
    """Format daemon errors with request_id so users can correlate logs."""
    message = str(body.get("error", default))
    request_id = body.get("request_id")
    if request_id:
        return f"{message} (request_id={request_id})"
    return message


def _get(port: int, path: str) -> dict:
    status, body = _daemon_request("GET", port, path)
    if status == 404:
        typer.echo(f"Not found", err=True)
        raise typer.Exit(1)
    if status != 200:
        typer.echo(f"Error: {_daemon_error_message(body)}", err=True)
        raise typer.Exit(1)
    return body


def _post(port: int, path: str, body: dict) -> dict:
    status, result = _daemon_request("POST", port, path, body)
    if status != 200:
        typer.echo(f"Error: {_daemon_error_message(result)}", err=True)
        raise typer.Exit(1)
    return result


def _patch(port: int, path: str, body: dict) -> dict:
    status, result = _daemon_request("PATCH", port, path, body)
    if status != 200:
        typer.echo(f"Error: {_daemon_error_message(result)}", err=True)
        raise typer.Exit(1)
    return result


def _delete(port: int, path: str) -> dict:
    status, result = _daemon_request("DELETE", port, path)
    if status != 200:
        typer.echo(f"Error: {_daemon_error_message(result)}", err=True)
        raise typer.Exit(1)
    return result


def _emit_context(data: dict, json_output: bool) -> None:
    if _is_json(json_output):
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(_render_context(data))


def _show_now(port: int, json_output: bool, doc_id: str = "now") -> None:
    data = _get(port, f"/v1/notes/{_q(doc_id)}/context")
    _emit_context(data, json_output)


def _resolve_now_scope(scope: Optional[str], auto_scope: bool) -> Optional[str]:
    """Resolve a ``--scope`` / ``--auto-scope`` pair to a concrete slug.

    Explicit ``--scope`` always wins. ``--auto-scope`` derives the slug from
    the cwd via :func:`keep.workstream.derive_workstream_slug` so plugin
    hooks can stay shell-simple.
    """
    if scope:
        return scope
    if auto_scope:
        return derive_workstream_slug()
    return None


def _now_doc_id(scope: Optional[str]) -> str:
    """``now`` when unscoped, ``now:{scope}`` otherwise — matches flow_client."""
    return f"now:{scope}" if scope else "now"


def _should_use_now_put_path(
    source: str | None,
    *,
    id: str | None,
    summary: str | None,
    watch: bool,
    unwatch: bool,
    interval: str | None,
    force: bool,
) -> bool:
    """True when `put` should behave exactly like `now`.

    Only plain text/stdin writes should collapse into the now command. File,
    URI, directory, watch, and force/update semantics remain normal put mode.
    """
    if id != "now" or summary is not None or watch or unwatch or interval or force:
        return False
    if source == "-" or source is None:
        return True
    if source.startswith(("file://", "http://", "https://")):
        return False
    path = Path(source)
    return not path.exists()


def _daemon_request(method: str, port: int, path: str, body: dict | None = None) -> tuple[int, dict]:
    """Make one daemon request using the shared daemon-client retry policy."""
    return _daemon_client.http_request_with_discovery_retry(
        method,
        port,
        path,
        body,
        store_override=_global_store,
    )


# ---------------------------------------------------------------------------
# Daemon port resolution (delegates to daemon_client)
# ---------------------------------------------------------------------------

def _get_port() -> int:
    """Get daemon port, auto-starting if needed."""
    return _daemon_get_port(_global_store)


def _resolved_store_path() -> Path:
    """Resolve the active CLI store through the shared daemon-client helper."""
    return _daemon_client.resolve_store_path(_global_store)


def _config_dir_for_store() -> Path:
    """Return the config directory for commands that read local settings."""
    return _resolved_store_path() if _global_store else get_config_dir()


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _width() -> int:
    return shutil.get_terminal_size((200, 24)).columns


def _truncate(text: str, max_len: int) -> str:
    """Replace newlines with spaces and truncate at word boundary."""
    text = text.replace("\n", " ")
    if len(text) > max_len:
        return text[:max_len - 3].rsplit(" ", 1)[0] + "..."
    return text


def _parse_tag_args(raw: list[str] | None) -> tuple[dict[str, str], list[str]]:
    """Parse --tag args into (key=value dict, key-only list)."""
    tag_dict: dict[str, str] = {}
    tag_keys: list[str] = []
    for t in (raw or []):
        if "=" in t:
            k, v = t.split("=", 1)
            tag_dict[k] = v
        else:
            tag_keys.append(t)
    return tag_dict, tag_keys


def _date(tags: dict) -> str:
    """Extract display date from tags."""
    for key in ("_updated", "_created"):
        val = tags.get(key, "")
        if val:
            return local_date(val)
    return ""


def _flow_items(resp: dict) -> list[dict]:
    """Extract item list from a flow response (data.results.results)."""
    return resp.get("data", {}).get("results", {}).get("results", [])


def _display_tags(tags: dict) -> dict:
    """Filter to user-visible tags (matches keep/types.py:INTERNAL_TAGS)."""
    return {k: v for k, v in tags.items()
            if k not in INTERNAL_TAGS
            and not k.startswith("_tk::")
            and not k.startswith("_tv::")}


def _yaml_quote(v: str) -> str:
    """Quote a YAML scalar value (matches cli.py's _quote_scalar_tag_value)."""
    return json.dumps(str(v))


def _render_tags_block(tags: dict) -> str:
    """Render tags as YAML-style indented block matching old CLI output."""
    display = _display_tags(tags)
    if not display:
        return ""
    lines = []
    for k, v in sorted(display.items()):
        if isinstance(v, list):
            lines.append(f"  {k}: [{', '.join(str(x) for x in v)}]")
        else:
            lines.append(f"  {k}: {_yaml_quote(str(v))}")
    return "\n".join(lines)


def _render_item_line(item: dict, w: int, id_width: int = 0) -> str:
    """Format a single item as a summary line."""
    id = item.get("id", "")
    score = item.get("score")
    tags = item.get("tags", {})
    date = _date(tags)
    padded = id.ljust(id_width) if id_width else id
    score_str = f" ({score:.2f})" if score is not None else ""
    date_str = f" {date}" if date else ""
    prefix_len = len(padded) + len(score_str) + len(date_str) + 2
    summary = note_display_name(
        tags, item.get("summary", ""), max_len=max(w - prefix_len, 20)
    )
    return f"{padded}{score_str}{date_str} {summary}"


def _render_context(data: dict) -> str:
    """Render ItemContext JSON as YAML frontmatter."""
    item = data.get("item", {})
    w = _width()
    lines = ["---"]

    # ID + version
    id_str = item.get("id", "")
    offset = data.get("viewing_offset", 0)
    if offset > 0:
        id_str += f"@V{{{offset}}}"
    lines.append(f"id: {id_str}")

    # Tags
    tags_block = _render_tags_block(item.get("tags", {}))
    if tags_block:
        lines.append("tags:")
        lines.append(tags_block)

    # Similar
    similar = data.get("similar", [])
    if similar:
        sim_ids = [s.get("id", "") for s in similar]
        id_width = min(max(len(s) for s in sim_ids), 20) if sim_ids else 0
        lines.append("similar:")
        for s in similar:
            sid = s.get("id", "")
            score = s.get("score")
            date = s.get("date", "")
            padded = sid.ljust(id_width)
            score_str = f"({score:.2f})" if score is not None else ""
            summary = _truncate(s.get("summary", ""), w - id_width - 25)
            lines.append(f"  - {padded} {score_str} {date} {summary}")

    # Meta sections
    meta = data.get("meta", {})
    for section, items in sorted(meta.items()):
        if items:
            meta_ids = [m.get("id", "") for m in items]
            id_width = min(max(len(s) for s in meta_ids), 20) if meta_ids else 0
            lines.append(f"meta/{section}:")
            for m in items:
                mid = m.get("id", "")
                padded = mid.ljust(id_width)
                msummary = _truncate(m.get("summary", ""), w - id_width - 10)
                lines.append(f"  - {padded} {msummary}")

    # Edges
    edges = data.get("edges", {})
    for pred, refs in sorted(edges.items()):
        if refs:
            lines.append(f"edges/{pred}:")
            for e in refs:
                eid = e.get("source_id", "")
                edate = e.get("date", "")
                esummary = _truncate(e.get("summary", ""), w - len(eid) - 15)
                lines.append(f"  - {eid} {edate} {esummary}")

    # Parts (compact: first + "N more" + last when many)
    parts = data.get("parts", [])
    if parts:
        lines.append("parts:")
        if len(parts) <= 3:
            for p in parts:
                pnum = p.get("part_num", 0)
                psummary = _truncate(p.get("summary", ""), w - 15)
                lines.append(f"  - @P{{{pnum}}} {psummary}")
        else:
            first, last = parts[0], parts[-1]
            lines.append(f"  - @P{{{first.get('part_num', 0)}}} {_truncate(first.get('summary', ''), w - 15)}")
            lines.append(f"  # (...{len(parts) - 2} more...)")
            lines.append(f"  - @P{{{last.get('part_num', 0)}}} {_truncate(last.get('summary', ''), w - 15)}")

    # Version navigation
    prev = data.get("prev", [])
    if prev:
        lines.append("prev:")
        for v in prev:
            voff = v.get("offset", 0)
            vdate = v.get("date", "")
            vsummary = _truncate(v.get("summary", ""), w - 20)
            lines.append(f"  - @V{{{voff}}} {vdate} {vsummary}")

    nxt = data.get("next", [])
    if nxt:
        lines.append("next:")
        for v in nxt:
            voff = v.get("offset", 0)
            vdate = v.get("date", "")
            vsummary = _truncate(v.get("summary", ""), w - 20)
            lines.append(f"  - @V{{{voff}}} {vdate} {vsummary}")

    lines.append("---")

    # Summary body
    summary = item.get("summary", "")
    if summary:
        lines.append(summary)

    return "\n".join(lines)


def _render_find(data: dict, port: int = 0) -> str:
    """Render search results. Respects --ids and --full global flags."""
    notes = data.get("notes", [])

    if _is_ids():
        return "\n".join(n.get("id", "") for n in notes) if notes else "No results."

    if _is_full() and port:
        # Full context for each result
        parts = []
        for n in notes:
            nid = n.get("id", "")
            status, ctx = _daemon_request("GET", port, f"/v1/notes/{_q(nid)}/context")
            if status == 200:
                parts.append(_render_context(ctx))
        return "\n\n".join(parts) if parts else "No results."

    w = _width()
    deep_groups = {g["id"]: g["items"] for g in data.get("deep_groups", []) if g.get("id")}

    # Compute aligned ID width
    all_ids = [n.get("id", "") for n in notes]
    id_width = min(max((len(i) for i in all_ids), default=0), 20)

    lines = []
    for item in notes:
        lines.append(_render_item_line(item, w, id_width))
        # Deep group items
        item_id = item.get("id", "")
        base_id = item_id.split("@")[0] if "@" in item_id else item_id
        group = deep_groups.get(base_id, deep_groups.get(item_id, []))
        for deep in group:
            deep_line = _render_item_line(deep, w - 2)
            lines.append(f"  {deep_line}")

    if not lines:
        return "No results."
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

StoreOption = Annotated[Optional[str], typer.Option("--store", "-s", envvar="KEEP_STORE_PATH", help="Store path")]
JsonFlag = Annotated[bool, typer.Option("--json", "-j", help="JSON output")]
LimitOption = Annotated[int, typer.Option("--limit", "-l", help="Result limit")]

# Global flags — set by app callback, read by helpers.
_global_json: bool = False
_global_ids: bool = False
_global_full: bool = False
_global_store: str | None = None


def _is_json(local_flag: bool = False) -> bool:
    """Check if JSON output is requested (local --json or global --json)."""
    return local_flag or _global_json


def _is_ids() -> bool:
    return _global_ids


def _is_full() -> bool:
    return _global_full


def _version_callback(value: bool | None):
    if value:
        try:
            typer.echo(importlib.metadata.version("keep-skill"))
        except Exception:
            typer.echo("unknown")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def default(
    ctx: typer.Context,
    store: StoreOption = None,
    json_output: JsonFlag = False,
    ids_only: Annotated[bool, typer.Option(
        "--ids", "-I", help="Output only IDs (for piping)"
    )] = False,
    full_output: Annotated[bool, typer.Option(
        "--full", "-F", help="Output full notes with context"
    )] = False,
    version: Annotated[Optional[bool], typer.Option(
        "--version", help="Show version and exit",
        callback=_version_callback, is_eager=True,
    )] = None,
):
    """Keep — reflective memory for AI agents."""
    global _global_json, _global_ids, _global_full, _global_store
    _global_json = json_output
    _global_ids = ids_only
    _global_full = full_output
    if store is not None:
        _global_store = store
    if ctx.invoked_subcommand is None:
        # No subcommand → show "now"
        port = _get_port()
        _show_now(port, json_output)


@app.command()
def get(
    id: Annotated[list[str], typer.Argument(help="Item ID(s)")],
    version: Annotated[Optional[int], typer.Option(
        "--version", "-V", help="Version selector (>=0 from current, <0 from oldest)"
    )] = None,
    history: Annotated[bool, typer.Option(
        "--history", "-H", help="List all versions"
    )] = False,
    similar: Annotated[bool, typer.Option(
        "--similar", "-S", help="List similar notes"
    )] = False,
    meta: Annotated[bool, typer.Option(
        "--meta", "-M", help="List meta notes"
    )] = False,
    parts: Annotated[bool, typer.Option(
        "--parts", "-P", help="List structural parts (from analyze)"
    )] = False,
    tag: Annotated[Optional[list[str]], typer.Option(
        "--tag", "-t", help="Require tag (key or key=value, repeatable)"
    )] = None,
    limit: Annotated[int, typer.Option(
        "--limit", "-n", help="Max items per section (default: 10)"
    )] = 10,
    json_output: JsonFlag = False,
):
    """Retrieve note(s) by ID.

    \b
    Accepts one or more IDs. Version: append @V{N}. Part: append @P{N}.
    Examples:
        keep get doc:1                  # Current version with context
        keep get doc:1 doc:2 doc:3      # Multiple notes
        keep get doc:1 -V 1             # Previous version
        keep get "doc:1@P{1}"           # Part 1
        keep get doc:1 --history        # List all versions
        keep get doc:1 --similar        # List similar items
        keep get doc:1 --meta           # List meta items
        keep get doc:1 --parts          # List structural parts
        keep get doc:1 -t project=myapp # Only if tag matches
    """
    port = _get_port()
    outputs = []
    errors = 0
    for item_id in id:
        result = _get_one_item(
            port, item_id, version=version, limit=limit,
            similar=similar, meta=meta, parts=parts, history=history,
            tag=tag, json_output=json_output,
        )
        if result is None:
            errors += 1
        else:
            outputs.append(result)
    if outputs:
        sep = "\n" if _is_ids() else "\n---\n" if len(outputs) > 1 else ""
        typer.echo(sep.join(outputs))
    if errors and not outputs:
        raise typer.Exit(1)


def _get_one_item(
    port: int, item_id: str, *, version: Optional[int], limit: int,
    similar: bool, meta: bool, parts: bool, history: bool,
    tag: Optional[list[str]], json_output: bool,
) -> Optional[str]:
    """Fetch and render a single item. Returns formatted string or None on error."""
    if (
        item_id == "now"
        and version is None
        and limit == 10
        and not similar
        and not meta
        and not parts
        and not history
        and not tag
    ):
        data = _get(port, f"/v1/notes/{_q('now')}/context")
        if _is_json(json_output):
            return json.dumps(data, indent=2)
        return _render_context(data)

    # --similar: flat list via search endpoint
    if similar:
        data = _post(port, "/v1/search", {
            "similar_to": item_id, "limit": limit,
        })
        if _is_json(json_output):
            return json.dumps(data, indent=2)
        return _render_find(data, port)

    # Section-specific context modes
    _SECTION_QS = {
        "parts": "include_similar=false&include_meta=false&include_versions=false&parts_limit={}",
        "meta": "include_similar=false&include_parts=false&include_versions=false&meta_limit={}",
        "history": "include_similar=false&include_meta=false&include_parts=false&versions_limit={}",
    }
    section = "parts" if parts else "meta" if meta else "history" if history else None
    if section:
        qs = "?" + _SECTION_QS[section].format(limit)
    else:
        qs_parts = [f"similar_limit={limit}", f"meta_limit={limit}", f"edges_limit={limit}"]
        if version is not None:
            qs_parts.append(f"version={version}")
        qs = "?" + "&".join(qs_parts)

    status, data = _daemon_request("GET", port, f"/v1/notes/{_q(item_id)}/context{qs}")
    if status == 404:
        typer.echo(f"Not found: {item_id}", err=True)
        return None
    if status != 200:
        typer.echo(f"Error: {_daemon_error_message(data)}", err=True)
        return None
    if _is_json(json_output):
        return json.dumps(data, indent=2)

    # Render section-specific output
    if parts:
        ctx_parts = data.get("parts", [])
        if not ctx_parts:
            return f"No parts for {item_id}. Use 'keep analyze {item_id}' to create parts."
        lines = [f"Parts of {item_id}:"]
        for p in ctx_parts:
            lines.append(f"  @P{{{p.get('part_num', 0)}}} {p.get('summary', '')[:80]}")
        return "\n".join(lines)
    if meta:
        meta_sections = data.get("meta", {})
        lines = [f"Meta for {item_id}:"]
        for sec, items in sorted(meta_sections.items()):
            if items:
                lines.append(f"  {sec}:")
                for m in items:
                    lines.append(f"    {m.get('id', '')}  {m.get('summary', '')[:50]}")
        if len(lines) == 1:
            lines.append("  No meta notes found.")
        return "\n".join(lines)
    if history:
        prev = data.get("prev", [])
        lines = [f"Versions of {item_id}:"]
        lines.append(f"  @V{{0}} (current)")
        for v in prev:
            lines.append(f"  @V{{{v.get('offset', 0)}}} {v.get('date', '')} {v.get('summary', '')[:60]}")
        if not prev:
            lines.append("  No previous versions.")
        return "\n".join(lines)

    # --tag: post-filter
    if tag:
        item_tags = data.get("item", {}).get("tags", {})
        for t in tag:
            if "=" in t:
                k, v = t.split("=", 1)
                if str(item_tags.get(k, "")).lower() != v.lower():
                    typer.echo(f"Tag filter not matched: {item_id}", err=True)
                    return None
            else:
                if t not in item_tags:
                    typer.echo(f"Tag filter not matched: {item_id}", err=True)
                    return None

    if _is_ids():
        return data.get("item", {}).get("id", item_id)
    if _is_json(json_output):
        return json.dumps(data, indent=2)
    return _render_context(data)


@app.command()
def find(
    query: Annotated[Optional[str], typer.Argument(help="Search query text")] = None,
    id: Annotated[Optional[str], typer.Option(
        "--id", help="Find notes similar to this ID (instead of text search)"
    )] = None,
    tag: Annotated[Optional[list[str]], typer.Option(
        "--tag", "-t", help="Filter by tag (key or key=value, repeatable)"
    )] = None,
    limit: LimitOption = 10,
    since: Annotated[Optional[str], typer.Option("--since", help="Updated since")] = None,
    until: Annotated[Optional[str], typer.Option("--until", help="Updated before")] = None,
    deep: Annotated[bool, typer.Option("--deep", "-D", help="Follow tags for related items")] = False,
    show_all: Annotated[bool, typer.Option(
        "--all", "-a", help="Include hidden system notes (IDs starting with '.')"
    )] = False,
    scope: Annotated[Optional[str], typer.Option(
        "--scope", "-S", help="ID glob to constrain results (e.g. 'file:///path/to/dir*')"
    )] = None,
    json_output: JsonFlag = False,
):
    """Find notes by hybrid search (semantic + full-text) or similarity.

    \b
    Examples:
        keep find "auth patterns"           # Semantic + full-text search
        keep find --id %abc123              # Find similar to item
        keep find "auth" -t project=myapp   # Search + filter by tag
        keep find "auth" --deep             # Follow tags for related items
        keep find "auth" --since P7D        # Last 7 days only
        keep find "auth" -S "file://.*"     # Scope to file items
    """
    if id and query:
        typer.echo("Error: Specify either a query or --id, not both", err=True)
        raise typer.Exit(1)
    if not id and not query:
        typer.echo("Error: Specify a query or --id", err=True)
        raise typer.Exit(1)

    port = _get_port()
    tag_filter, _ = _parse_tag_args(tag)

    body: dict = {
        "query": query if not id else None,
        "similar_to": id,
        "tags": tag_filter or None,
        "limit": limit,
        "deep": deep or None,
        "since": since,
        "until": until,
        "scope": scope,
        "include_hidden": show_all or None,
    }
    data = _post(port, "/v1/search", body)
    if _is_json(json_output):
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(_render_find(data, port))


def _put_directory(
    port: int, resolved_path: Path, parsed_tags: dict, *,
    recurse: bool, exclude: list[str] | None,
    watch: bool, unwatch: bool, interval: str | None,
    force: bool, json_output: bool,
) -> None:
    """Index files from a directory via the daemon HTTP API."""
    max_dir_files = 1000

    # Fast local precheck so obviously ineligible directories fail before
    # we try to bootstrap the daemon just to discover there is nothing to do.
    preflight_files = _list_directory_files(
        resolved_path,
        recurse=recurse,
        exclude=exclude or None,
    )
    if not preflight_files:
        typer.echo(f"Error: no eligible files in {resolved_path}/", err=True)
        hint = "hidden files and symlinks are skipped"
        if not recurse:
            hint += "; use -r to recurse into subdirectories"
        typer.echo(f"Hint: {hint}", err=True)
        raise typer.Exit(1)
    if len(preflight_files) > max_dir_files:
        typer.echo(
            f"Error: directory has {len(preflight_files)} files (max {max_dir_files})",
            err=True,
        )
        typer.echo(
            "Hint: increase max_dir_files in keep.toml or index files individually",
            err=True,
        )
        raise typer.Exit(1)

    # Get ignore patterns from the store's .ignore doc (if any)
    ignore_patterns: list[str] = []
    try:
        status, data = _daemon_request("GET", port, f"/v1/notes/{_q('.ignore')}")
        if status == 200 and data.get("summary"):
            ignore_patterns = parse_ignore_patterns(data["summary"])
    except Exception:
        pass

    combined_exclude = merge_excludes(ignore_patterns, exclude)
    files = _list_directory_files(resolved_path, recurse=recurse, exclude=combined_exclude or None)
    if not files:
        typer.echo(f"Error: no eligible files in {resolved_path}/", err=True)
        hint = "hidden files and symlinks are skipped"
        if not recurse:
            hint += "; use -r to recurse into subdirectories"
        typer.echo(f"Hint: {hint}", err=True)
        raise typer.Exit(1)

    if len(files) > max_dir_files:
        typer.echo(
            f"Error: directory has {len(files)} files (max {max_dir_files})",
            err=True,
        )
        typer.echo("Hint: increase max_dir_files in keep.toml or index files individually", err=True)
        raise typer.Exit(1)

    indexed = 0
    errors: list[str] = []
    total = len(files)
    is_tty = sys.stderr.isatty()
    for i, fpath in enumerate(files, 1):
        file_uri = f"file://{fpath}"
        rel = str(fpath.relative_to(resolved_path) if recurse else fpath.name)
        body: dict = {"uri": file_uri, "tags": parsed_tags or None, "force": force or None}
        try:
            status, data = _daemon_request("POST", port, "/v1/notes", body)
            if status == 200:
                indexed += 1
                if is_tty:
                    pct = i * 100 // total
                    typer.echo(f"\r[{pct:3d}%] {rel[:60]}", err=True, nl=False)
                else:
                    typer.echo(f"[{i}/{total}] {rel} ok", err=True)
            else:
                errors.append(f"{rel}: {_daemon_error_message(data)}")
                if not is_tty:
                    typer.echo(f"[{i}/{total}] {rel} error", err=True)
        except Exception as e:
            errors.append(f"{rel}: {e}")

    if is_tty:
        typer.echo("", err=True)
    typer.echo(f"{indexed} indexed, {len(errors)} errors from {resolved_path.name}/", err=True)
    for e in errors:
        typer.echo(f"  error: {e}", err=True)

    # Queue initial git-history ingest for any repos under the imported tree.
    # This is separate from file indexing because commit items are background
    # work owned by the daemon, not inline CLI work.
    dir_body: dict = {
        "uri": f"file://{resolved_path}",
        "tags": parsed_tags or None,
        "force": None,
        "watch_kind": "directory",
    }

    # Watch management
    if watch or unwatch:
        watch_body = dict(dir_body)
        if watch:
            watch_body["enqueue_git"] = True
            watch_body["watch"] = True
            watch_body["recurse"] = recurse
            watch_body["exclude"] = exclude
            if interval:
                watch_body["interval"] = interval
        elif unwatch:
            watch_body["unwatch"] = True
        try:
            status, data = _daemon_request("POST", port, "/v1/notes", watch_body)
            if status == 200:
                if data.get("git", {}).get("queued"):
                    typer.echo("git: changelog ingest queued", err=True)
                if watch and data.get("watch"):
                    typer.echo(
                        f"watching {resolved_path}/ "
                        f"(interval {data['watch']['interval']})",
                        err=True,
                    )
                elif unwatch:
                    if data.get("unwatch"):
                        typer.echo(f"watch removed: {resolved_path}/", err=True)
                    else:
                        typer.echo(f"not watching: {resolved_path}/", err=True)
            else:
                err = _daemon_error_message(data) if isinstance(data, dict) else str(data)
                typer.echo(
                    f"Warning: watch registration failed: {err or 'unknown error'}",
                    err=True,
                )
        except Exception as e:
            typer.echo(
                f"Warning: watch registration failed: {e}",
                err=True,
            )
    else:
        git_body = dict(dir_body)
        git_body["enqueue_git"] = True
        git_body["recurse"] = recurse
        git_body["exclude"] = exclude
        try:
            status, data = _daemon_request("POST", port, "/v1/notes", git_body)
            if status == 200:
                if data.get("git", {}).get("queued"):
                    typer.echo("git: changelog ingest queued", err=True)
            else:
                err = _daemon_error_message(data) if isinstance(data, dict) else str(data)
                typer.echo(
                    f"Warning: git ingest enqueue failed: {err or 'unknown error'}",
                    err=True,
                )
        except Exception as e:
            typer.echo(
                f"Warning: git ingest enqueue failed: {e}",
                err=True,
            )


@app.command()
def put(
    source: Annotated[Optional[str], typer.Argument(help="Content, file path, URI, or '-' for stdin")] = None,
    id: Annotated[Optional[str], typer.Option("--id", "-i", help="Item ID")] = None,
    tags: Annotated[Optional[list[str]], typer.Option("-t", "--tag", help="Tags (key=value)")] = None,
    summary: Annotated[Optional[str], typer.Option("--summary", help="Explicit summary")] = None,
    recurse: Annotated[bool, typer.Option("--recurse", "-r", help="Recurse into subdirectories (directory mode)")] = False,
    exclude: Annotated[Optional[list[str]], typer.Option(
        "--exclude", "-x", help="Glob pattern to exclude files (directory mode, repeatable)"
    )] = None,
    watch: Annotated[bool, typer.Option("--watch", help="Watch source for changes (daemon re-imports on change)")] = False,
    unwatch: Annotated[bool, typer.Option("--unwatch", help="Stop watching source for changes")] = False,
    interval: Annotated[Optional[str], typer.Option(
        "--interval", help="Watch poll interval as ISO 8601 duration (default PT30S)"
    )] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Force update")] = False,
    json_output: JsonFlag = False,
):
    """Add or update a note in the store.

    \b
    Input modes (auto-detected):
      keep put /path/to/file.pdf     # File mode
      keep put https://example.com   # URI mode
      keep put /path/to/folder/      # Directory mode
      keep put /path/to/folder/ -r   # Directory mode (recurse)
      keep put "my note"             # Text mode (content-addressed ID)
      keep put -                     # Stdin mode
    """
    if watch and unwatch:
        typer.echo("Error: --watch and --unwatch are mutually exclusive", err=True)
        raise typer.Exit(1)
    if interval and not watch:
        typer.echo("Error: --interval requires --watch", err=True)
        raise typer.Exit(1)

    # Expand ${.field} templates from stdin JSON (hook support)
    (source,) = _expand_stdin_templates(source)
    tags = _expand_stdin_tag_list(tags)

    parsed_tags: dict = {}
    for t in (tags or []):
        if "=" not in t:
            hint = f"Invalid tag format: {t!r}."
            if ":" in t:
                k2, v2 = t.split(":", 1)
                hint += f" Did you mean: {k2}={v2}?"
            else:
                hint += " Use key=value"
            typer.echo(hint, err=True)
            raise typer.Exit(1)
        k, v = t.split("=", 1)
        key = k.casefold()
        existing = parsed_tags.get(key)
        if existing is None:
            parsed_tags[key] = v
        elif isinstance(existing, list):
            if v not in existing:
                existing.append(v)
        elif existing != v:
            parsed_tags[key] = [existing, v]

    use_now_put_path = _should_use_now_put_path(
        source,
        id=id,
        summary=summary,
        watch=watch,
        unwatch=unwatch,
        interval=interval,
        force=force,
    )

    # Stdin mode
    if source == "-" or (source is None and _has_stdin_data()):
        try:
            content = _read_stdin_text()
        except (UnicodeDecodeError, ValueError):
            typer.echo("Error: stdin contains binary data (not valid UTF-8)", err=True)
            typer.echo("Hint: for binary files, use: keep put file:///path/to/file", err=True)
            raise typer.Exit(1)
        if summary is not None:
            typer.echo("Error: --summary cannot be used with stdin (original content would be lost)", err=True)
            raise typer.Exit(1)
        # Extract YAML frontmatter as tags (CLI tags override)
        body_text, fm_tags = _extract_markdown_frontmatter(content)
        if fm_tags:
            merged = {**fm_tags, **parsed_tags}
            content = body_text
        else:
            merged = parsed_tags
        if use_now_put_path:
            port = _get_port()
            _post(port, "/v1/notes", {"content": content, "id": "now", "tags": merged or None})
            _show_now(port, json_output)
            return
        port = _get_port()
        body: dict = {"content": content, "id": id, "tags": merged or None, "force": force or None}
        data = _post(port, "/v1/notes", body)
    elif source is None:
        typer.echo("Error: provide content, URI, or '-' for stdin", err=True)
        raise typer.Exit(1)
    else:
        # Detect file/URL/directory
        content = source
        uri = None
        watch_kind = "file"
        is_directory = False
        if source.startswith(("file://", "http://", "https://")):
            uri = source
            content = None
            watch_kind = "url" if source.startswith(("http://", "https://")) else "file"
        else:
            # Inline note content can be arbitrarily long; probing it as a
            # filesystem path must never turn validation into an OS error.
            try:
                is_directory = Path(source).is_dir()
            except OSError:
                is_directory = False
        if is_directory:
            if summary is not None:
                typer.echo("Error: --summary cannot be used with directory mode", err=True)
                raise typer.Exit(1)
            if id is not None:
                typer.echo("Error: --id cannot be used with directory mode", err=True)
                raise typer.Exit(1)
            # Directory mode — iterate locally, put each file via HTTP
            resolved_dir = Path(source).resolve()
            max_dir_files = 1000
            preflight_files = _list_directory_files(
                resolved_dir,
                recurse=recurse,
                exclude=exclude or None,
            )
            if not preflight_files:
                typer.echo(f"Error: no eligible files in {resolved_dir}/", err=True)
                hint = "hidden files and symlinks are skipped"
                if not recurse:
                    hint += "; use -r to recurse into subdirectories"
                typer.echo(f"Hint: {hint}", err=True)
                raise typer.Exit(1)
            if len(preflight_files) > max_dir_files:
                typer.echo(
                    f"Error: directory has {len(preflight_files)} files (max {max_dir_files})",
                    err=True,
                )
                typer.echo(
                    "Hint: increase max_dir_files in keep.toml or index files individually",
                    err=True,
                )
                raise typer.Exit(1)
            port = _get_port()
            _put_directory(
                port, resolved_dir, parsed_tags,
                recurse=recurse, exclude=exclude,
                watch=watch, unwatch=unwatch, interval=interval,
                force=force, json_output=json_output,
            )
            return
        elif not source.startswith("%"):
            try:
                if Path(source).exists():
                    uri = f"file://{Path(source).resolve()}"
                    content = None
            except OSError:
                # Overlong inline text is not a filesystem path.
                pass

        if content is not None and uri is None:
            config_dir = _config_dir_for_store()
            cfg = load_or_create_config(config_dir)
            if len(content) > cfg.max_inline_length:
                typer.echo(
                    (
                        f"Error: inline text too long ({len(content)} chars, "
                        f"max {cfg.max_inline_length})"
                    ),
                    err=True,
                )
                typer.echo(
                    "Hint: write it to a file and run: keep put file:///path/to/file",
                    err=True,
                )
                raise typer.Exit(1)

        # Inline text + --summary rejected (original content would be lost)
        if content is not None and uri is None and summary is not None:
            typer.echo("Error: --summary cannot be used with inline text (original content would be lost)", err=True)
            typer.echo("Hint: write to a file first, then: keep put file:///path/to/file --summary '...'", err=True)
            raise typer.Exit(1)

        if use_now_put_path and content is not None and uri is None:
            port = _get_port()
            _post(port, "/v1/notes", {"content": content, "id": "now", "tags": parsed_tags or None})
            _show_now(port, json_output)
            return

        port = _get_port()
        body = {
            "content": content,
            "uri": uri,
            "id": id,
            "tags": parsed_tags or None,
            "summary": summary,
            "force": force or None,
        }
        # Watch params → daemon handles the timer
        if watch:
            body["watch"] = True
            body["watch_kind"] = watch_kind
            if interval:
                body["interval"] = interval
        elif unwatch:
            body["unwatch"] = True
        data = _post(port, "/v1/notes", body)

    if _is_json(json_output):
        typer.echo(json.dumps(data, indent=2))
    else:
        msg = f"{data.get('id', '')} stored."
        if data.get("watch"):
            msg += f" watching (interval {data['watch']['interval']})"
        elif data.get("unwatch"):
            msg += " watch removed."
        typer.echo(msg)


# Hidden aliases for put
@app.command("update", hidden=True)
def update_alias(
    source: Annotated[Optional[str], typer.Argument()] = None,
    id: Annotated[Optional[str], typer.Option("--id", "-i")] = None,
    tags: Annotated[Optional[list[str]], typer.Option("-t", "--tag")] = None,
    summary: Annotated[Optional[str], typer.Option("--summary")] = None,
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
    json_output: JsonFlag = False,
):
    """Alias for 'put'."""
    put(source=source, id=id, tags=tags, summary=summary, force=force, json_output=json_output)


@app.command("add", hidden=True)
def add_alias(
    source: Annotated[Optional[str], typer.Argument()] = None,
    id: Annotated[Optional[str], typer.Option("--id", "-i")] = None,
    tags: Annotated[Optional[list[str]], typer.Option("-t", "--tag")] = None,
    summary: Annotated[Optional[str], typer.Option("--summary")] = None,
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
    json_output: JsonFlag = False,
):
    """Alias for 'put'."""
    put(source=source, id=id, tags=tags, summary=summary, force=force, json_output=json_output)


@app.command()
def tag(
    ids: Annotated[list[str], typer.Argument(help="Note ID(s) to tag")],
    tags: Annotated[Optional[list[str]], typer.Option(
        "--tag", "-t", help="Tag to add/update as key=value"
    )] = None,
    remove: Annotated[Optional[list[str]], typer.Option(
        "--remove", "-r", help="Tag keys to remove"
    )] = None,
    remove_values: Annotated[Optional[list[str]], typer.Option(
        "--remove-value", "-R", help="Tag values to remove as key=value"
    )] = None,
    json_output: JsonFlag = False,
):
    """Add, update, or remove tags on existing notes.

    \b
    Does not re-process the note — only updates tags.
    Examples:
        keep tag doc:1 -t project=myapp
        keep tag doc:1 doc:2 -t status=reviewed
        keep tag doc:1 --remove obsolete_tag
        keep tag doc:1 --remove-value speaker=Bob
    """
    set_tags, _ = _parse_tag_args(tags)
    rv_dict, _ = _parse_tag_args(remove_values)

    if not set_tags and not remove and not rv_dict:
        typer.echo("Error: Specify at least one --tag, --remove, or --remove-value", err=True)
        raise typer.Exit(1)

    port = _get_port()
    body: dict = {}
    if set_tags:
        body["set"] = set_tags
    if remove:
        body["remove"] = remove
    if rv_dict:
        body["remove_values"] = rv_dict

    for doc_id in ids:
        data = _patch(port, f"/v1/notes/{_q(doc_id)}/tags", body)
        if _is_json(json_output):
            typer.echo(json.dumps(data, indent=2))
        else:
            typer.echo(f"{data.get('id', doc_id)} tagged.")


@app.command("tag-update", hidden=True)
def tag_update_alias(
    ids: Annotated[list[str], typer.Argument(help="Note ID(s)")],
    tags: Annotated[Optional[list[str]], typer.Option("-t", "--tag")] = None,
    remove: Annotated[Optional[list[str]], typer.Option("--remove", "-r")] = None,
    json_output: JsonFlag = False,
):
    """Alias for 'tag'."""
    tag(ids=ids, tags=tags, remove=remove, json_output=json_output)


@app.command("del")
def delete_cmd(
    id: Annotated[list[str], typer.Argument(help="Item ID(s)")],
):
    """Delete the current version of note(s), or a specific version."""
    port = _get_port()
    for item_id in id:
        data = _delete(port, f"/v1/notes/{_q(item_id)}")
        if data.get("deleted"):
            typer.echo(f"{item_id} deleted.")
        else:
            typer.echo(f"{item_id} not found.", err=True)


@app.command("delete", hidden=True)
def delete_alias(id: Annotated[list[str], typer.Argument(help="Item ID(s)")]):
    """Alias for 'del'."""
    delete_cmd(id=id)


@app.command()
def now(
    content: Annotated[Optional[str], typer.Argument(help="New content")] = None,
    tags: Annotated[Optional[list[str]], typer.Option("-t", "--tag", help="Tags")] = None,
    truncate_flag: Annotated[bool, typer.Option("--truncate", help="Truncate content to max_inline_length instead of failing")] = False,
    scope: Annotated[Optional[str], typer.Option("--scope", help="Workstream scope name; writes to 'now:{scope}' instead of bare 'now'")] = None,
    auto_scope: Annotated[bool, typer.Option("--auto-scope", help="Derive scope from cwd (git project+branch, else cwd basename)")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress the trailing context dump on stdout (use in Stop hooks)")] = False,
    json_output: JsonFlag = False,
):
    """Get or set the current working intentions.

    \b
    With no arguments, displays the current intentions.
    With content, replaces them (previous version is preserved).

    \b
    --scope/--auto-scope route reads and writes to a per-workstream version
    chain (doc id ``now:{scope}``). Lets parallel sessions stop clobbering
    one shared ``now``.
    """
    # Expand ${.field} templates from stdin JSON (hook support)
    (content,) = _expand_stdin_templates(content)
    tags = _expand_stdin_tag_list(tags)

    port = _get_port()
    resolved_scope = _resolve_now_scope(scope, auto_scope)
    doc_id = _now_doc_id(resolved_scope)
    if content:
        if truncate_flag:
            config_dir = _config_dir_for_store()
            cfg = load_or_create_config(config_dir)
            if len(content) > cfg.max_inline_length:
                content = content[:cfg.max_inline_length]
        parsed_tags, _ = _parse_tag_args(tags)
        # Mirror flow_client.set_now_item: a scoped now also carries a
        # ``user`` tag so downstream queries can find sibling chains.
        if resolved_scope:
            parsed_tags = parsed_tags or {}
            parsed_tags.setdefault("user", resolved_scope)
        _post(port, "/v1/notes", {"content": content, "id": doc_id, "tags": parsed_tags or None})
    if not quiet:
        _show_now(port, json_output, doc_id=doc_id)


@app.command("list")
def list_cmd(
    prefix: Annotated[Optional[str], typer.Argument(
        help="ID filter — prefix (e.g. '.tag') or glob (e.g. 'session-*', '*auth*')"
    )] = None,
    tag: Annotated[Optional[list[str]], typer.Option(
        "--tag", "-t", help="Filter by tag (key or key=value, repeatable)"
    )] = None,
    sort: Annotated[str, typer.Option(
        "--sort", help="Sort order: 'updated' (default), 'accessed', 'created', or 'id'"
    )] = "updated",
    limit: LimitOption = 20,
    since: Annotated[Optional[str], typer.Option("--since")] = None,
    until: Annotated[Optional[str], typer.Option("--until")] = None,
    show_all: Annotated[bool, typer.Option(
        "--all", "-a", help="Include hidden system notes (IDs starting with '.')"
    )] = False,
    json_output: JsonFlag = False,
):
    """List recent notes, filter by tags, or browse by prefix.

    \b
    Examples:
        keep list                      # Recent notes (by update time)
        keep list .tag                 # All .tag/* system docs
        keep list session-*            # Glob match
        keep list --sort accessed      # By access time
        keep list -t project=myapp     # Filter by tag
        keep list --since P7D          # Last 7 days
        keep --ids list                # IDs only (for piping)
    """
    port = _get_port()
    tag_dict, tag_keys = _parse_tag_args(tag)

    flow_params: dict = {
        "limit": limit,
        "order_by": sort,
        "since": since,
        "until": until,
        "include_hidden": show_all or (prefix.startswith(".") if prefix else False),
    }
    if prefix:
        flow_params["prefix"] = prefix
    if tag_dict:
        flow_params["tags"] = tag_dict
    if tag_keys:
        flow_params["tag_keys"] = tag_keys

    resp = _post(port, "/v1/flow", {"state": "list", "params": flow_params})
    results = _flow_items(resp)
    data: dict = {"notes": results}

    if _is_json(json_output):
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(_render_find(data, port))


@app.command()
def move(
    name: Annotated[str, typer.Argument(help="Target collection name")],
    source: Annotated[str, typer.Option("--source", help="Source item ID")] = "now",
    json_output: JsonFlag = False,
):
    """Move versions from now (or another item) into a named item."""
    port = _get_port()
    data = _post(port, "/v1/flow", {
        "state": "move", "params": {"name": name, "source": source},
    })
    typer.echo(f"Moved to {name}.")


@app.command()
def prompt(
    name: Annotated[str, typer.Argument(help="Prompt name (e.g. 'reflect')")] = "",
    text: Annotated[Optional[str], typer.Argument(help="Text for context search")] = None,
    list_prompts: Annotated[bool, typer.Option("--list", "-l", help="List available prompts")] = False,
    id: Annotated[Optional[str], typer.Option("--id", help="Item ID for {get} context")] = None,
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Filter by tag (key=value)")] = None,
    since: Annotated[Optional[str], typer.Option("--since", help="Updated since")] = None,
    until: Annotated[Optional[str], typer.Option("--until", help="Updated before")] = None,
    deep: Annotated[bool, typer.Option("--deep", "-D", help="Follow tags to discover related items")] = False,
    scope: Annotated[Optional[str], typer.Option("--scope", "-S", help="ID glob scope")] = None,
    auto_scope: Annotated[bool, typer.Option("--auto-scope", help="When --id is unset, default it to the workstream-scoped 'now' derived from cwd")] = False,
    token_budget: Annotated[Optional[int], typer.Option("--tokens", help="Token budget for {find}")] = None,
    json_output: JsonFlag = False,
):
    """Render an agent prompt with injected context.

    \b
    The prompt doc may contain {get} and {find} placeholders:
      {get}  — expanded with context for --id (default: now)
      {find} — expanded with search results for the text argument

    \b
    Examples:
        keep prompt --list                        # List available prompts
        keep prompt reflect                       # Reflect on current work
        keep prompt reflect "auth flow"           # With search context
        keep prompt reflect --since P7D           # Recent context only
    """
    # Expand ${.field} templates from stdin JSON (hook support)
    tag = _expand_stdin_tag_list(tag)

    # --auto-scope is a per-workstream override of the {get} target. Only
    # apply it when --id wasn't specified explicitly, so an explicit --id
    # always wins.
    if auto_scope and not id:
        id = _now_doc_id(_resolve_now_scope(None, True))

    port = _get_port()

    if list_prompts or not name:
        # List prompts via flow
        data = _post(port, "/v1/flow", {"state": STATE_PROMPT, "params": {"list": True}})
        prompts = data.get("data", {}).get("prompts", [])
        if not prompts:
            typer.echo("No agent prompts available.", err=True)
            raise typer.Exit(1)
        if _is_json(json_output):
            typer.echo(json.dumps({"prompts": prompts}, indent=2))
        else:
            for p in prompts:
                typer.echo(f"{p['name']:20s} {p.get('summary', '')}")
        return

    tags_dict = {}
    for t in (tag or []):
        if "=" in t:
            k, v = t.split("=", 1)
            tags_dict[k] = v

    # Render prompt via flow
    data = _post(port, "/v1/flow", {"state": STATE_PROMPT, "params": {
        "name": name,
        "text": text,
        "id": id,
        "tags": tags_dict or None,
        "since": since,
        "until": until,
        "deep": deep or None,
        "scope": scope,
        "token_budget": token_budget,
    }})
    flow_data = data.get("data", {})
    if data.get("status") == "error":
        typer.echo(f"Error: {_daemon_error_message(flow_data)}", err=True)
        raise typer.Exit(1)
    if _is_json(json_output):
        typer.echo(json.dumps(flow_data, indent=2))
    else:
        typer.echo(flow_data.get("text", ""))


@app.command(hidden=True)
def reflect(
    text: Annotated[Optional[str], typer.Argument(help="Text for context search")] = None,
    id: Annotated[Optional[str], typer.Option("--id", help="Item ID for {get} context")] = None,
    json_output: JsonFlag = False,
):
    """Reflect on current actions (alias for 'keep prompt reflect')."""
    prompt(name="reflect", text=text, id=id, json_output=json_output)


@app.command("flow")
def flow_cmd(
    state: Annotated[Optional[str], typer.Argument(help="State doc name")] = None,
    target: Annotated[Optional[str], typer.Option("--target", "-t", help="Target note ID")] = None,
    file: Annotated[Optional[str], typer.Option("--file", "-f", help="YAML state doc file or '-' for stdin")] = None,
    budget: Annotated[Optional[int], typer.Option("--budget", "-b", help="Max ticks")] = None,
    cursor: Annotated[Optional[str], typer.Option("--cursor", "-c", help="Resume cursor")] = None,
    param: Annotated[Optional[list[str]], typer.Option("--param", "-p", help="Parameter as key=value")] = None,
    json_output: JsonFlag = False,
):
    """Run a state-doc flow synchronously.

    \b
    Examples:
        keep flow after-write --target %abc123
        keep flow query-resolve -p query="auth patterns"
        keep flow --file review.yaml --target myproject
        keep flow --cursor <token> --budget 5
    """
    if state is None and file is None and cursor is None:
        typer.echo("Error: provide a state name, --file, or --cursor", err=True)
        raise typer.Exit(1)

    flow_params: dict = {}
    if target:
        flow_params["id"] = target
    for p in (param or []):
        if "=" not in p:
            typer.echo(f"Error: param must be key=value, got: {p!r}", err=True)
            raise typer.Exit(1)
        k, v = p.split("=", 1)
        try:
            flow_params[k] = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            flow_params[k] = v

    state_doc_yaml: Optional[str] = None
    if file is not None:
        if file == "-":
            state_doc_yaml = sys.stdin.read()
        else:
            try:
                state_doc_yaml = Path(file).read_text()
            except FileNotFoundError:
                typer.echo(f"Error: file not found: {file}", err=True)
                raise typer.Exit(1)
        if state is None:
            state = "inline"

    if cursor and state is None:
        state = "__cursor__"

    port = _get_port()
    body: dict = {"state": state, "params": flow_params}
    if budget is not None:
        body["budget"] = budget
    if cursor:
        body["cursor_token"] = cursor
    if state_doc_yaml:
        body["state_doc_yaml"] = state_doc_yaml

    data = _post(port, "/v1/flow", body)
    if _is_json(json_output):
        typer.echo(json.dumps(data, ensure_ascii=False))
    else:
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2))


@app.command("edit")
def edit_cmd(
    id: Annotated[str, typer.Argument(help="ID of note to edit")],
):
    """Edit a note's content in $EDITOR.

    \b
    Opens the current content in your editor. On save, updates the note
    if the content changed.
    Examples:
        keep edit .ignore                    # Edit ignore patterns
        keep edit .prompt/agent/reflect      # Edit a prompt template
        keep edit now                        # Edit current intentions
    """
    port = _get_port()
    data = _get(port, f"/v1/notes/{_q(id)}")
    content = data.get("summary", "")

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    suffix = ".md" if not id.endswith((".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml")) else ""
    with tempfile.NamedTemporaryFile(suffix=suffix or Path(id).suffix, mode="w", delete=False, prefix="keep-edit-") as f:
        f.write(content)
        tmp = f.name

    try:
        sp.run([editor, tmp], check=True)
        new_content = Path(tmp).read_text()
    except (sp.CalledProcessError, KeyboardInterrupt):
        typer.echo("Editor exited abnormally, no changes saved", err=True)
        raise typer.Exit(1)
    finally:
        Path(tmp).unlink(missing_ok=True)

    if new_content == content:
        typer.echo("No changes", err=True)
        return

    _post(port, "/v1/notes", {"content": new_content, "id": id})
    typer.echo(f"Updated {id}", err=True)


@app.command()
def analyze(
    id: Annotated[str, typer.Argument(help="ID of note to analyze")],
    tag: Annotated[Optional[list[str]], typer.Option("--tag", "-t", help="Guidance tag keys")] = None,
    foreground: Annotated[bool, typer.Option("--foreground", "--fg", help="Run in foreground")] = False,
    force: Annotated[bool, typer.Option("--force", help="Re-analyze even if current")] = False,
    json_output: JsonFlag = False,
):
    """Decompose a note or string into meaningful parts.

    \b
    Uses an LLM to identify sections, each with its own summary, tags,
    and embedding. Runs in background by default; use --fg to wait.
    """
    port = _get_port()
    body: dict = {"id": id, "foreground": foreground, "force": force}
    if tag:
        body["tags"] = tag
    data = _post(port, "/v1/analyze", body)

    if _is_json(json_output):
        typer.echo(json.dumps(data, indent=2))
    else:
        if data.get("parts") is not None:
            parts = data["parts"]
            if parts:
                typer.echo(f"Analyzed {id} into {len(parts)} parts:")
                for p in parts:
                    summary = str(p.get("summary", ""))[:60].replace("\n", " ")
                    typer.echo(f"  @P{{{p.get('part_num', '?')}}} {summary}")
            else:
                typer.echo(f"Content not decomposable into multiple parts: {id}")
        elif data.get("queued"):
            typer.echo(f"Queued {id} for background analysis.", err=True)
        elif data.get("skipped"):
            typer.echo(f"Already analyzed, skipping {id}.", err=True)


@app.command("help")
def help_cmd(
    topic: Annotated[Optional[str], typer.Argument(help="Documentation topic")] = None,
):
    """Browse keep documentation.

    \b
    Examples:
        keep help              # Documentation index
        keep help quickstart   # CLI Quick Start guide
        keep help keep-put     # keep put reference
    """
    typer.echo(get_help_topic(topic or "index", link_style="cli"))


# ---------------------------------------------------------------------------
# Delegate commands — these stay with the full CLI
# ---------------------------------------------------------------------------

def _daemon_command_impl(
    *,
    stop: Annotated[bool, typer.Option("--stop", help="Stop the background daemon")] = False,
    list_items: Annotated[bool, typer.Option("--list", "-l", help="List pending work items")] = False,
    reindex: Annotated[bool, typer.Option("--reindex", help="Enqueue all items for re-embedding")] = False,
    retry: Annotated[bool, typer.Option("--retry", help="Reset failed items back to pending")] = False,
    purge: Annotated[bool, typer.Option("--purge", help="Delete all pending work items")] = False,
    daemon: bool = False,
    bind_host: str | None = None,
    advertised_url: str | None = None,
    trusted_proxy: bool = False,
    interactive: bool = True,
) -> None:
    if stop:
        store_path = _resolved_store_path()
        pid_file = store_path / "processor.pid"
        if not pid_file.exists():
            typer.echo("No daemon running.")
            (store_path / DAEMON_PORT_FILE).unlink(missing_ok=True)
            (store_path / DAEMON_TOKEN_FILE).unlink(missing_ok=True)
            return
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            typer.echo(f"Stopping daemon (pid {pid})...", err=True)
            # Wait up to 5s for graceful shutdown
            deadline = _time.monotonic() + 10.0
            while _time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)  # check if still alive
                except ProcessLookupError:
                    break
                _time.sleep(0.2)
            else:
                # Still alive — force kill
                try:
                    os.kill(pid, signal.SIGKILL)
                    typer.echo("Force-killed (was stuck in long operation).", err=True)
                except ProcessLookupError:
                    pass
            pid_file.unlink(missing_ok=True)
        except ProcessLookupError:
            typer.echo("Daemon not running (stale PID file).", err=True)
            pid_file.unlink(missing_ok=True)
        except (ValueError, OSError) as e:
            typer.echo(f"Error stopping daemon: {e}", err=True)
        (store_path / DAEMON_PORT_FILE).unlink(missing_ok=True)
        (store_path / DAEMON_TOKEN_FILE).unlink(missing_ok=True)
        return

    if list_items:
        store_path = _resolved_store_path()
        _console_support.print_pending_list_lightweight(store_path)
        # Ensure daemon is running so pending items get processed
        _daemon_client.get_port(_global_store)
        return

    kp = _api.Keeper(store_path=_resolved_store_path())

    if daemon:
        _console_support.run_pending_daemon(
            kp,
            bind_host=bind_host,
            advertised_url=advertised_url,
            trusted_proxy=trusted_proxy,
        )
        return

    try:
        if purge:
            wq = kp._get_work_queue()
            n = wq.purge()
            typer.echo(f"Purged {n} pending work items.", err=True)
            return

        if retry:
            n = kp._pending_queue.retry_failed()
            if n:
                typer.echo(f"Reset {n} failed items back to pending.", err=True)
            else:
                typer.echo("No failed items to retry.", err=True)
                return

        if reindex:
            count = kp.count()
            if count == 0:
                typer.echo("No notes to reindex.")
                raise typer.Exit(0)
            typer.echo(f"Enqueuing {count} notes for reindex...", err=True)
            stats = kp.enqueue_reindex()
            typer.echo(f"Enqueued {stats['enqueued']} items + {stats['versions']} versions", err=True)

        if interactive:
            # Interactive mode: show status, ensure daemon running, tail log
            _console_support.print_pending_interactive(kp)
        else:
            _console_support.run_pending_daemon(
                kp,
                bind_host=bind_host,
                advertised_url=advertised_url,
                trusted_proxy=trusted_proxy,
            )
    finally:
        kp.close()


@app.command()
def daemon(
    stop: Annotated[bool, typer.Option("--stop", help="Stop the background daemon")] = False,
    list_items: Annotated[bool, typer.Option("--list", "-l", help="List pending work items")] = False,
    reindex: Annotated[bool, typer.Option("--reindex", help="Enqueue all items for re-embedding")] = False,
    retry: Annotated[bool, typer.Option("--retry", help="Reset failed items back to pending")] = False,
    purge: Annotated[bool, typer.Option("--purge", help="Delete all pending work items")] = False,
    bind: Annotated[Optional[str], typer.Option("--bind", help="Bind host for the daemon HTTP server")] = None,
    advertised_url: Annotated[Optional[str], typer.Option(
        "--advertised-url",
        help="Advertised base URL used for remote-mode host allowlisting",
    )] = None,
    trusted_proxy: Annotated[bool, typer.Option(
        "--trusted-proxy",
        help="Acknowledge that TLS is terminated by a trusted reverse proxy",
    )] = False,
):
    """Run or manage the background daemon."""
    _daemon_command_impl(
        stop=stop,
        list_items=list_items,
        reindex=reindex,
        retry=retry,
        purge=purge,
        daemon=not any((stop, list_items, reindex, retry, purge)),
        bind_host=bind,
        advertised_url=advertised_url,
        trusted_proxy=trusted_proxy,
        interactive=retry or reindex,
    )


@app.command("pending", hidden=True)
def pending(
    stop: Annotated[bool, typer.Option("--stop", help="Stop the background daemon")] = False,
    list_items: Annotated[bool, typer.Option("--list", "-l", help="List pending work items")] = False,
    reindex: Annotated[bool, typer.Option("--reindex", help="Enqueue all items for re-embedding")] = False,
    retry: Annotated[bool, typer.Option("--retry", help="Reset failed items back to pending")] = False,
    purge: Annotated[bool, typer.Option("--purge", help="Delete all pending work items")] = False,
    daemon: Annotated[bool, typer.Option("--daemon", hidden=True, help="Run as background daemon")] = False,
):
    """Hidden compatibility alias for older daemon workflows."""
    _daemon_command_impl(
        stop=stop,
        list_items=list_items,
        reindex=reindex,
        retry=retry,
        purge=purge,
        daemon=daemon,
        interactive=not daemon,
    )


@app.command()
def config(
    ctx: typer.Context,
    path: Annotated[Optional[str], typer.Argument(
        help="Config path (e.g., 'file', 'tool', 'store', 'providers.embedding')"
    )] = None,
    setup: Annotated[bool, typer.Option("--setup", help="Run setup wizard")] = False,
    reset_system_docs: Annotated[bool, typer.Option(
        "--reset-system-docs", help="Force reload system documents from bundled content"
    )] = False,
    state_diagram: Annotated[bool, typer.Option(
        "--state-diagram", help="Print Mermaid state-transition diagram for .state/* docs"
    )] = False,
    json_output: JsonFlag = False,
):
    """Show configuration. Optionally get a specific value by path.

    \b
    Special paths: file, tool, store, docs, mcpb, providers
    Dotted paths: providers.embedding, tags, etc.
    """
    if reset_system_docs:
        port = _get_port()
        data = _post(port, "/v1/admin/reset-system-docs", {})
        count = data.get("reset", 0)
        typer.echo(f"Reset {count} system documents")
        return

    if state_diagram:
        port = _get_port()
        resp = _post(port, "/v1/flow", {
            "state": "list",
            "params": {"prefix": ".state/", "include_hidden": True, "limit": 100},
        })
        results = _flow_items(resp)
        state_docs: dict[str, str] = {}
        for note in results:
            nid = note.get("id", "")
            body = (note.get("summary") or "").strip()
            if nid.startswith(".state/") and body:
                state_docs[nid.removeprefix(".state/")] = body
        if not state_docs:
            typer.echo("No .state/* documents found.", err=True)
            raise typer.Exit(1)
        typer.echo(state_doc_diagram(state_docs))
        return

    if setup:
        store_path = _resolved_store_path()
        config_dir = _config_dir_for_store()
        run_wizard(config_dir, store_path, restart_command="keep config --setup")
        return

    config_dir = _config_dir_for_store()
    cfg = load_or_create_config(config_dir)
    store_path = get_default_store_path(cfg) if not _global_store else _resolved_store_path()

    if path:
        try:
            value = _get_config_value(cfg, Path(str(store_path)), path)
        except typer.BadParameter as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
        is_json = json_output or (ctx.parent and ctx.parent.params.get("json_output", False))
        if is_json:
            typer.echo(json.dumps({path: value}, indent=2))
        elif isinstance(value, (list, dict)):
            typer.echo(json.dumps(value))
        else:
            typer.echo(value)
        return

    is_json = json_output or (ctx.parent and ctx.parent.params.get("json_output", False))
    if is_json:
        result = {
            "file": str(cfg.config_path) if cfg else None,
            "tool": str(get_tool_directory()),
            "docs": str(get_tool_directory() / "docs"),
            "store": str(store_path),
            "openclaw-plugin": str(Path(str(importlib.resources.files("keep"))) / "data" / "openclaw-plugin"),
            "providers": {
                "embedding": cfg.embedding.name if cfg and cfg.embedding else None,
                "summarization": cfg.summarization.name if cfg else None,
                "document": cfg.document.name if cfg else None,
            },
        }
        if cfg and cfg.default_tags:
            result["tags"] = cfg.default_tags
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(_format_config_with_defaults(cfg, Path(str(store_path))))


@app.command(hidden=True)
def doctor(
    log: Annotated[bool, typer.Option("--log", "-l", help="Tail the ops log")] = False,
    use_faulthandler: Annotated[bool, typer.Option("--faulthandler", help="Enable faulthandler")] = False,
):
    """Diagnostic checks for debugging setup and crash issues."""
    doctor_impl(log=log, use_faulthandler=use_faulthandler)


@app.command()
def mcp(ctx: typer.Context):
    """Start MCP stdio server."""
    if _global_store:
        os.environ["KEEP_STORE_PATH"] = _global_store
    mcp_main()


# ---------------------------------------------------------------------------
# Data management subcommands
# ---------------------------------------------------------------------------

data_app = typer.Typer(
    name="data",
    help="Data management — export, import.",
    rich_markup_mode=None,
)
app.add_typer(data_app)


def _clear_progress_line() -> None:
    cols = shutil.get_terminal_size((80, 24)).columns
    sys.stderr.write("\r" + " " * (cols - 1) + "\r")
    sys.stderr.flush()


@data_app.command("export")
def data_export(
    output: Annotated[Optional[str], typer.Argument(help="Output file path (JSON) or directory (markdown). Use '-' for stdout in JSON mode.", show_default=False)] = None,
    include_system: Annotated[bool, typer.Option("--include-system", help="Include system documents (.tag/*, .meta/*, .now, etc.). Excluded by default.")] = False,
    include_parts: Annotated[bool, typer.Option("--include-parts", help="Markdown mode: emit analysis parts as <note>/@P{N}.md sidecars. Off by default.")] = False,
    include_versions: Annotated[bool, typer.Option("--include-versions", help="Markdown mode: emit archived versions as <note>/@V{N}.md sidecars. Off by default.")] = False,
    list_sync: Annotated[bool, typer.Option("--list", help="List active markdown sync directories.")] = False,
    sync: Annotated[bool, typer.Option("--sync", help="Markdown mode: continuously mirror from keep for this directory.")] = False,
    stop: Annotated[bool, typer.Option("--stop", help="With --sync, stop mirroring this directory.")] = False,
    interval: Annotated[Optional[str], typer.Option("--interval", help="Sync poll interval as ISO 8601 duration (default PT30S, e.g. PT60S, PT5M)")] = None,
    format: Annotated[str, typer.Option("--format", "-f", help="Export format: 'json' (default) or 'md' (markdown directory, one file per note)")] = "json",
):
    """Export the store for backup or migration.

    JSON mode writes a single file with full history (versions, parts).
    Markdown mode writes a directory with one .md file per note, containing
    YAML frontmatter (`_id`, reserved `_...` metadata, promoted tags) and the
    note summary as the body.
    Analysis parts and archived versions are off by default in markdown mode;
    pass ``--include-parts`` and/or ``--include-versions`` to emit them as
    ``<note>/@P{N}.md`` and ``<note>/@V{N}.md`` sidecar files.

    System documents (dot-prefix ids like ``.tag/*``, ``.meta/*``, ``.now``)
    are excluded by default; pass ``--include-system`` to include them.
    """
    if list_sync:
        if output is not None:
            typer.echo("Error: --list does not take an output path", err=True)
            raise SystemExit(1)
        if sync or stop:
            typer.echo("Error: --list cannot be combined with --sync or --stop", err=True)
            raise SystemExit(1)
        port = _get_port()
        status, data = _daemon_request(
            "POST",
            port,
            "/v1/admin/markdown-export",
            {"list": True},
        )
        if status == 404 and data.get("error") == "not found":
            if _is_json(False):
                typer.echo(json.dumps({"mirrors": []}, indent=2))
            else:
                typer.echo("No markdown sync directories.")
            return
        if status >= 400:
            typer.echo(f"Error: {_daemon_error_message(data, 'markdown sync list failed')}", err=True)
            raise SystemExit(1)
        mirrors = data.get("mirrors", [])
        if _is_json(False):
            typer.echo(json.dumps({"mirrors": mirrors}, indent=2))
            return
        if not mirrors:
            typer.echo("No markdown sync directories.")
            return
        for entry in mirrors:
            status_bits = []
            if entry.get("last_error"):
                status_bits.append(f"error={entry['last_error']}")
            elif entry.get("pending_since"):
                status_bits.append(f"pending since {entry['pending_since']}")
            elif entry.get("last_run"):
                status_bits.append(f"last run {entry['last_run']}")
            else:
                status_bits.append("idle")
            if entry.get("include_parts"):
                status_bits.append("parts")
            if entry.get("include_versions"):
                status_bits.append("versions")
            if entry.get("include_system"):
                status_bits.append("system")
            entry_interval = entry.get("interval", "PT30S")
            if entry_interval != "PT30S":
                status_bits.append(f"interval={entry_interval}")
            typer.echo(f"{entry.get('root', '')} ({', '.join(status_bits)})")
        return

    if output is None:
        typer.echo("Error: missing output path", err=True)
        raise SystemExit(1)

    fmt = format.lower()
    # --interval implies --sync for markdown exports
    if interval and not sync and not stop:
        sync = True
    if (sync or stop) and fmt == "json":
        fmt = "md"
    if interval and fmt not in ("md", "markdown"):
        typer.echo("Error: --interval requires markdown sync mode (--sync or --format md)", err=True)
        raise SystemExit(1)
    if fmt in ("md", "markdown"):
        if stop and not sync:
            typer.echo("Error: --stop requires --sync", err=True)
            raise SystemExit(1)
        if sync:
            count = 0
            source_cursor = ""
            if not stop:
                if output == "-":
                    typer.echo("Error: markdown export requires a directory path, not '-'", err=True)
                    raise SystemExit(1)
                port = _get_port()
                status, data = _daemon_request(
                    "POST",
                    port,
                    "/v1/admin/markdown-export",
                    {
                        "root": output,
                        "include_system": include_system,
                        "include_parts": include_parts,
                        "include_versions": include_versions,
                        "sync": True,
                        "validate_only": True,
                        **({"interval": interval} if interval else {}),
                    },
                )
                if status >= 400:
                    typer.echo(f"Error: {_daemon_error_message(data, 'markdown sync failed')}", err=True)
                    raise SystemExit(1)
                kp = _require_markdown_export_host()
                is_tty = sys.stderr.isatty()
                progress = None
                if is_tty:
                    def progress(current: int, total: int, label: str) -> None:
                        if total > 0:
                            _console_support._progress_bar(current, total, label, err=True)

                try:
                    try:
                        count, _info = _markdown_mirrors.run_markdown_export_once(
                            kp,
                            output,
                            include_system=include_system,
                            include_parts=include_parts,
                            include_versions=include_versions,
                            allow_existing=True,
                            mirror_entry=None,
                            progress=progress,
                        )
                    except _ExportCollisionError as e:
                        if is_tty:
                            _clear_progress_line()
                        typer.echo(
                            f"Error: id collision — '{e.first_id}' and '{e.second_id}' "
                            f"both map to '{e.path}'. Rename one of these ids before exporting.",
                            err=True,
                        )
                        raise SystemExit(1) from None
                    if (
                        not _supports_local_markdown_export(kp)
                        and hasattr(kp, "export_changes")
                    ):
                        feed = kp.export_changes(limit=0)
                        if isinstance(feed, dict):
                            source_cursor = str(feed.get("cursor") or "")
                finally:
                    kp.close()

                if is_tty:
                    _clear_progress_line()

            port = _get_port()
            status, data = _daemon_request(
                "POST",
                port,
                "/v1/admin/markdown-export",
                {
                    "root": output,
                    "include_system": include_system,
                    "include_parts": include_parts,
                    "include_versions": include_versions,
                    "sync": True,
                    "stop": stop,
                    "register_only": True,
                    "baseline_complete": not stop,
                    "source_cursor": source_cursor,
                    **({"interval": interval} if interval else {}),
                },
            )
            if status >= 400:
                typer.echo(f"Error: {_daemon_error_message(data, 'markdown sync failed')}", err=True)
                raise SystemExit(1)
            if stop:
                if data.get("stopped"):
                    typer.echo(f"Stopped markdown sync: {output}", err=True)
                else:
                    typer.echo(f"Not syncing: {output}", err=True)
                return
            sync_info = data.get("sync", {})
            sync_interval = sync_info.get("interval", "PT30S")
            interval_suffix = f", interval={sync_interval}" if sync_interval != "PT30S" else ""
            typer.echo(
                f"Markdown sync active: {sync_info.get('root', output)} "
                f"({count} notes exported{interval_suffix})",
                err=True,
            )
            return
        _data_export_markdown(
            output,
            include_system=include_system,
            include_parts=include_parts,
            include_versions=include_versions,
        )
        return
    if fmt != "json":
        typer.echo(f"Error: --format must be 'json' or 'md', got '{format}'", err=True)
        raise SystemExit(1)

    if include_parts or include_versions:
        bad = []
        if include_parts:
            bad.append("--include-parts")
        if include_versions:
            bad.append("--include-versions")
        typer.echo(
            f"Error: {', '.join(bad)} only applies to --format md "
            f"(JSON exports always include full history)",
            err=True,
        )
        raise SystemExit(1)

    kp = _get_export_host()
    it = kp.export_iter(include_system=include_system)
    header = next(it)
    info = header["store_info"]
    total = info["document_count"]
    # Only show a progress bar when stderr is a real terminal.  Also skip
    # when writing JSON to stdout ('-') — a user piping the export is
    # likely capturing the stream and doesn't want decorative output
    # written to stderr either.
    show_progress = output != "-" and total > 0 and sys.stderr.isatty()

    dest = sys.stdout if output == "-" else open(output, "w", encoding="utf-8")
    try:
        dest.write("{\n")
        for key in ("format", "version", "exported_at", "store_info"):
            dest.write(f"  {json.dumps(key)}: {json.dumps(header[key], ensure_ascii=False)},\n")
        dest.write('  "documents": [\n')
        first = True
        count = 0
        for doc in it:
            if not first:
                dest.write(",\n")
            dest.write("    " + json.dumps(doc, ensure_ascii=False))
            first = False
            count += 1
            if show_progress:
                _console_support._progress_bar(count, total, doc.get("id", ""), err=True)
        dest.write("\n  ]\n}\n")
    finally:
        if dest is not sys.stdout:
            dest.close()
    kp.close()

    if show_progress:
        # Clear the progress bar line before the final summary.
        cols = shutil.get_terminal_size((80, 24)).columns
        sys.stderr.write("\r" + " " * (cols - 1) + "\r")
        sys.stderr.flush()

    if output != "-":
        typer.echo(
            f"Exported {info['document_count']} documents "
            f"({info['version_count']} versions, {info['part_count']} parts) "
            f"to {output}",
            err=True,
        )


def _data_export_markdown(
    output: str,
    *,
    include_system: bool,
    include_parts: bool = False,
    include_versions: bool = False,
) -> None:
    """Write a markdown-per-note export into ``output`` (a directory)."""
    if output == "-":
        typer.echo("Error: markdown export requires a directory path, not '-'", err=True)
        raise SystemExit(1)

    out_dir = Path(output)
    if out_dir.exists():
        if not out_dir.is_dir():
            typer.echo(f"Error: {output} exists and is not a directory", err=True)
            raise SystemExit(1)
        if any(out_dir.iterdir()):
            typer.echo(f"Error: output directory {output} is not empty", err=True)
            raise SystemExit(1)
    else:
        out_dir.mkdir(parents=True)

    kp = _require_markdown_export_host()

    is_tty = sys.stderr.isatty()
    progress = None
    if is_tty:
        def progress(current: int, total: int, label: str) -> None:
            if total > 0:
                _console_support._progress_bar(current, total, label, err=True)

    try:
        try:
            count, info = _write_markdown_export(
                kp, out_dir,
                include_system=include_system,
                include_parts=include_parts,
                include_versions=include_versions,
                progress=progress,
            )
        except _ExportCollisionError as e:
            if is_tty:
                cols = shutil.get_terminal_size((80, 24)).columns
                sys.stderr.write("\r" + " " * (cols - 1) + "\r")
                sys.stderr.flush()
            typer.echo(
                f"Error: id collision — '{e.first_id}' and '{e.second_id}' "
                f"both map to '{e.path}'. Rename one of these ids before exporting.",
                err=True,
            )
            raise SystemExit(1) from None
    finally:
        kp.close()

    if is_tty:
        # Clear the progress bar line before the final summary.
        _clear_progress_line()

    extras = []
    if include_parts:
        extras.append(f"{info['part_count']} parts")
    if include_versions:
        extras.append(f"{info['version_count']} versions")
    if extras:
        suffix = f" with {' and '.join(extras)}"
    else:
        suffix = (
            f" (skipped {info['version_count']} versions, "
            f"{info['part_count']} parts)"
        )
    typer.echo(f"Exported {count} notes to {output}{suffix}", err=True)


@data_app.command("import")
def data_import(
    file: Annotated[str, typer.Argument(help="JSON export file, markdown file, or markdown directory to import")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="merge or replace")] = "merge",
    format: Annotated[str, typer.Option("--format", "-f", help="auto, json, or md")] = "auto",
):
    """Import documents from a JSON export or recursive markdown source."""
    if mode not in ("merge", "replace"):
        typer.echo(f"Error: --mode must be 'merge' or 'replace', got '{mode}'", err=True)
        raise SystemExit(1)
    if format not in ("auto", "json", "md"):
        typer.echo(
            f"Error: --format must be 'auto', 'json', or 'md', got '{format}'",
            err=True,
        )
        raise SystemExit(1)

    p: Path | None = None
    if file != "-":
        p = Path(file)
        if not p.exists():
            typer.echo(f"Error: file not found: {file}", err=True)
            raise SystemExit(1)

    import_format = format
    if import_format == "auto":
        if file == "-":
            import_format = "json"
        elif p is not None and (p.is_dir() or p.suffix.lower() == ".md"):
            import_format = "md"
        else:
            import_format = "json"

    if mode == "replace":
        if import_format == "json":
            if file == "-":
                data = json.loads(sys.stdin.read())
            else:
                assert p is not None
                data = json.loads(p.read_text(encoding="utf-8"))
            doc_count = len(data.get("documents", []))
            if not typer.confirm(
                f"This will delete all existing documents and import {doc_count} from {file}. Continue?"
            ):
                raise SystemExit(0)
        else:
            if not typer.confirm(
                f"This will delete all existing documents and import markdown from {file}. Continue?"
            ):
                raise SystemExit(0)

    kp = _api.Keeper(store_path=_resolved_store_path())
    try:
        if import_format == "json":
            if mode != "replace":
                if file == "-":
                    data = json.loads(sys.stdin.read())
                else:
                    assert p is not None
                    data = json.loads(p.read_text(encoding="utf-8"))
            stats = kp.import_data(data, mode=mode)
        else:
            if file == "-":
                typer.echo("Error: markdown import does not support stdin", err=True)
                raise SystemExit(1)
            assert p is not None
            show_progress = False
            progress = None
            total_files = _markdown_import.count_markdown_import_files(p)
            if total_files > 1 and sys.stderr.isatty():
                show_progress = True

                def progress(current: int, total: int, label: str) -> None:
                    if total > 1:
                        _console_support._progress_bar(current, total, label, err=True)

            try:
                stats = kp.import_markdown(p, mode=mode, progress=progress)
            finally:
                if show_progress:
                    _clear_progress_line()
    finally:
        kp.close()

    typer.echo(
        f"Imported {stats['imported']} documents "
        f"({stats['versions']} versions, {stats['parts']} parts), "
        f"skipped {stats['skipped']}. "
        f"Queued {stats['queued']} for embedding.",
        err=True,
    )
    if stats["queued"] > 0:
        typer.echo("Run 'keep daemon' to process embeddings.", err=True)


def main():
    app()


if __name__ == "__main__":
    main()
