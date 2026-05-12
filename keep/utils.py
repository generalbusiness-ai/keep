"""Pure utility functions extracted from api.py.

These have no dependency on Keeper or any store — they are pure data
transforms for timestamps, tags, content parsing, and hashing.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from .markdown_frontmatter import (
    MARKDOWN_FRONTMATTER_TAGS_KEY,
    classify_markdown_frontmatter_key,
    is_writable_markdown_frontmatter_key,
)
from .types import (
    Item,
    SYSTEM_TAG_PREFIX,
    is_system_id,
    iter_tag_pairs,
    set_tag_values,
    tag_values,
    validate_tag_key,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Environment variable prefix for auto-applied tags
ENV_TAG_PREFIX = "KEEP_TAG_"

# Pattern for meta-doc query lines: key=value pairs separated by spaces
_META_QUERY_PAIR = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*=\S+$')
# Pattern for context-match lines: key= (bare, no value)
_META_CONTEXT_KEY = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)=$')
# Pattern for prerequisite lines: key=* (item must have this tag)
_META_PREREQ_KEY = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)=\*$')

# Markdown extensions that may contain YAML frontmatter
_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx"}


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------

def _parse_date_param(value: str) -> str:
    """Parse a date/duration parameter and return a YYYY-MM-DD date.

    Accepts:
    - ISO 8601 duration: P3D (3 days), P1W (1 week), PT1H (1 hour), P1DT12H, etc.
    - ISO date: 2026-01-15
    - Date with slashes: 2026/01/15

    Returns:
        YYYY-MM-DD string
    """
    since = value.strip()

    # ISO 8601 duration: P[n]Y[n]M[n]W[n]DT[n]H[n]M[n]S
    if since.upper().startswith("P"):
        duration_str = since.upper()

        # Parse duration components
        years = months = weeks = days = hours = minutes = seconds = 0

        # Split on T to separate date and time parts
        if "T" in duration_str:
            date_part, time_part = duration_str.split("T", 1)
        else:
            date_part = duration_str
            time_part = ""

        # Parse date part (P[n]Y[n]M[n]W[n]D)
        date_part = date_part[1:]  # Remove leading P
        for match in re.finditer(r"(\d+)([YMWD])", date_part):
            val, unit = int(match.group(1)), match.group(2)
            if unit == "Y":
                years = val
            elif unit == "M":
                months = val
            elif unit == "W":
                weeks = val
            elif unit == "D":
                days = val

        # Parse time part ([n]H[n]M[n]S)
        for match in re.finditer(r"(\d+)([HMS])", time_part):
            val, unit = int(match.group(1)), match.group(2)
            if unit == "H":
                hours = val
            elif unit == "M":
                minutes = val
            elif unit == "S":
                seconds = val

        # Convert to timedelta (approximate months/years)
        total_days = years * 365 + months * 30 + weeks * 7 + days
        delta = timedelta(days=total_days, hours=hours, minutes=minutes, seconds=seconds)
        cutoff = datetime.now(timezone.utc) - delta
        return cutoff.strftime("%Y-%m-%d")

    # Try parsing as date
    # ISO format: 2026-01-15 or 2026-01-15T...
    # Slash format: 2026/01/15
    date_str = since.replace("/", "-").split("T")[0]

    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
        return parsed.strftime("%Y-%m-%d")
    except ValueError:
        pass

    raise ValueError(
        f"Invalid date/duration format: {value}. "
        "Use ISO duration (P3D, PT1H, P1W) or date (2026-01-15)"
    )


def _truncate_ts(ts: str) -> str:
    """Normalize timestamp to canonical format: YYYY-MM-DDTHH:MM:SS.

    New data is already in this format (via utc_now()). This handles
    legacy timestamps that may have microseconds, 'Z', or '+00:00'.
    """
    # Strip fractional seconds
    dot = ts.find(".", 19)
    if dot != -1:
        # Skip past digits to any tz suffix
        end = dot
        for i in range(dot + 1, len(ts)):
            if ts[i] in "+-Z":
                break
        else:
            i = len(ts)
        ts = ts[:dot] + ts[i:] if i < len(ts) else ts[:dot]
    # Strip timezone suffix — all timestamps are UTC by convention
    if ts.endswith("+00:00"):
        ts = ts[:-6]
    elif ts.endswith("Z"):
        ts = ts[:-1]
    return ts


def _record_to_item(rec, score: float = None, changed: bool = None) -> Item:
    """Convert a DocumentRecord to an Item with timestamp tags.

    Adds _updated, _created, _updated_date from the record's columns
    to ensure consistent timestamp exposure across all retrieval methods.
    """
    updated = _truncate_ts(rec.updated_at) if rec.updated_at else ""
    created = _truncate_ts(rec.created_at) if rec.created_at else ""
    accessed = _truncate_ts(rec.accessed_at or rec.updated_at) if (rec.accessed_at or rec.updated_at) else ""
    tags = {
        **rec.tags,
        "_updated": updated,
        "_created": created,
        "_updated_date": updated[:10],
        "_accessed": accessed,
        "_accessed_date": accessed[:10],
    }
    return Item(id=rec.id, summary=rec.summary, tags=tags, score=score, changed=changed)


def _filter_by_date(
    items: list,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> list:
    """Filter items by date range (since <= date, date < until)."""
    since_cutoff = _parse_date_param(since) if since else None
    until_cutoff = _parse_date_param(until) if until else None
    result = []
    for item in items:
        d = item.tags.get("_updated_date", "0000-00-00")
        if since_cutoff and d < since_cutoff:
            continue
        if until_cutoff and d >= until_cutoff:
            continue
        result.append(item)
    return result


def _enrich_updated_date(items: list, doc_store, doc_coll: str) -> None:
    """Fill in missing _updated_date tags from SQLite for date filtering.

    Version references (``id@v{N}``) stored in ChromaDB often lack
    ``_updated_date`` because the tag is synthesized by ``_record_to_item``
    from the SQLite ``updated_at`` column rather than stored in the raw
    tags dict.  Without enrichment, ``_filter_by_date`` defaults to
    ``"0000-00-00"`` and drops them.

    Uses a single batch lookup (``get_many``) to avoid N+1 queries.
    """
    # Collect unique base IDs that need enrichment
    needs: dict[str, list] = {}  # base_id → [items needing it]
    for item in items:
        if "_updated_date" in item.tags:
            continue
        base_id = item.tags.get("_base_id",
                                item.id.split("@")[0] if "@" in item.id else item.id)
        needs.setdefault(base_id, []).append(item)
    if not needs:
        return
    # Single batch fetch
    docs = doc_store.get_many(doc_coll, list(needs.keys()))
    for base_id, waiting in needs.items():
        doc = docs.get(base_id)
        if doc and doc.updated_at:
            date = doc.updated_at[:10]
            for item in waiting:
                item.tags["_updated_date"] = date


def _is_hidden(item) -> bool:
    """System notes (dot-prefix IDs like .conversations) are hidden by default."""
    base_id = item.tags.get("_base_id", item.id)
    if isinstance(base_id, list):
        base_id = base_id[0] if base_id else item.id
    return is_system_id(base_id)


# ---------------------------------------------------------------------------
# Content parsing
# ---------------------------------------------------------------------------

def _extract_markdown_frontmatter(content: str) -> tuple[str, dict]:
    """Extract YAML frontmatter from markdown content.

    Returns (body, tags) where:
    - body: content with frontmatter stripped
    - tags: all scalar frontmatter values as string tags, plus
            values from a ``tags`` dict if present.
            Reserved ``_...`` keys are skipped.
            Non-scalar values (lists, nested dicts) are dropped,
            except for the ``tags`` key which is expected to be a dict.
    """
    if not content.startswith("---"):
        return content, {}

    parts = content.split("---", 2)
    if len(parts) < 3:
        return content, {}

    try:
        import yaml
        frontmatter = yaml.safe_load(parts[1])
    except Exception:
        return content, {}

    body = parts[2].lstrip("\n")
    if not isinstance(frontmatter, dict):
        return body, {}

    tags: dict[str, str | list[str]] = {}

    for key, value in frontmatter.items():
        key_str = str(key)
        kind = classify_markdown_frontmatter_key(key_str)
        if kind not in ("writable_tag", "tag_namespace"):
            continue

        if key_str == MARKDOWN_FRONTMATTER_TAGS_KEY and isinstance(value, dict):
            # Obsidian/keep-style tags dict: {topic: foo, project: bar}
            for tk, tv in value.items():
                tk_str = str(tk)
                if not is_writable_markdown_frontmatter_key(tk_str):
                    continue
                if isinstance(tv, (str, int, float, bool)):
                    tags[tk_str] = str(tv)
                elif isinstance(tv, list):
                    vals = [str(v) for v in tv if isinstance(v, (str, int, float, bool))]
                    if vals:
                        tags[tk_str] = vals
        elif isinstance(value, (str, int, float, bool)):
            # Top-level scalar: title, author, date, etc. → tag
            tags[key_str] = str(value)
        elif isinstance(value, list):
            vals = [str(v) for v in value if isinstance(v, (str, int, float, bool))]
            if vals:
                tags[key_str] = vals
        elif hasattr(value, "isoformat"):
            # datetime.date / datetime.datetime (YAML auto-parses dates)
            tags[key_str] = value.isoformat()
        # else: lists, nested dicts, None → drop silently

    return body, tags


def _parse_meta_doc(content: str) -> tuple[list[dict[str, str]], list[str], list[str]]:
    """Parse meta-doc content into query lines, context-match keys, and prerequisites.

    Returns:
        (query_lines, context_keys, prereq_keys) where:
        - query_lines: list of dicts, each {key: value, ...} for AND queries
        - context_keys: list of tag keys for context matching
        - prereq_keys: list of tag keys the current item must have
    """
    query_lines: list[dict[str, str]] = []
    context_keys: list[str] = []
    prereq_keys: list[str] = []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check for prerequisite: exactly "key=*"
        prereq_match = _META_PREREQ_KEY.match(line)
        if prereq_match:
            prereq_keys.append(prereq_match.group(1))
            continue

        # Check for context-match: exactly "key=" with no value
        ctx_match = _META_CONTEXT_KEY.match(line)
        if ctx_match:
            context_keys.append(ctx_match.group(1))
            continue

        # Check for query line: all space-separated tokens are key=value
        tokens = line.split()
        pairs: dict[str, str] = {}
        is_query = True
        for token in tokens:
            if _META_QUERY_PAIR.match(token):
                k, v = token.split("=", 1)
                pairs[k] = v
            else:
                is_query = False
                break

        if is_query and pairs:
            query_lines.append(pairs)

    return query_lines, context_keys, prereq_keys


# ---------------------------------------------------------------------------
# Tag utilities
# ---------------------------------------------------------------------------

def _get_env_tags() -> dict[str, str]:
    """Collect tags from KEEP_TAG_* environment variables.

    KEEP_TAG_PROJECT=foo -> {"project": "foo"}
    KEEP_TAG_MyTag=bar   -> {"mytag": "bar"}

    Tag keys are lowercased for consistency.
    """
    tags = {}
    for key, value in os.environ.items():
        if key.startswith(ENV_TAG_PREFIX) and value:
            tag_key = key[len(ENV_TAG_PREFIX):].lower()
            tags[tag_key] = value
    return tags


def _user_tags_changed(old_tags: dict, new_tags: dict) -> bool:
    """Check if non-system tags differ between old and new."""
    old_user = {
        (k, v)
        for k, v in iter_tag_pairs(old_tags, include_system=False)
    }
    new_user = {
        (k, v)
        for k, v in iter_tag_pairs(new_tags, include_system=False)
    }
    return old_user != new_user


def _merge_tags_additive(dst: dict, src: dict, *, replace_system: bool = False) -> None:
    """Merge tags by adding distinct values per key.

    Non-system keys are additive set union. System keys are replaced.
    """
    for raw_key in src:
        key = str(raw_key)
        vals = tag_values(src, raw_key)
        if not vals:
            continue
        if key.startswith(SYSTEM_TAG_PREFIX):
            if replace_system:
                dst[key] = vals[-1]
            continue
        existing = tag_values(dst, key)
        seen = set(existing)
        merged = list(existing)
        for value in vals:
            if value not in seen:
                merged.append(value)
                seen.add(value)
        set_tag_values(dst, key, merged)


def _split_tag_additions(tags: dict[str, Any]) -> tuple[set[str], dict[str, list[str]]]:
    """Split additive tag changes and legacy empty-value key deletions."""
    delete_keys: set[str] = set()
    add_by_key: dict[str, list[str]] = {}
    for key in tags:
        values = tag_values(tags, key)
        if not values:
            continue
        if "" in values:
            delete_keys.add(key)
            values = [v for v in values if v != ""]
        if values:
            add_by_key[key] = values
    for key in delete_keys:
        add_by_key.pop(key, None)
    return delete_keys, add_by_key


def _normalize_remove_keys(keys: Optional[list[str]]) -> set[str]:
    """Normalize and validate explicit key removals."""
    out: set[str] = set()
    if not keys:
        return out
    for raw_key in keys:
        key = str(raw_key).casefold()
        if key.startswith(SYSTEM_TAG_PREFIX):
            continue
        validate_tag_key(key)
        out.add(key)
    return out


def _apply_tag_mutations(
    base_tags: dict[str, Any],
    add_by_key: dict[str, Any],
    *,
    remove_keys: Optional[set[str]] = None,
    remove_by_key: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Apply explicit tag adds/removals and return updated tags."""
    merged = dict(base_tags)
    for key in (remove_keys or set()):
        merged.pop(key, None)

    remove_map = remove_by_key or {}
    for key in remove_map:
        current_vals = tag_values(merged, key)
        if not current_vals:
            continue
        remove_set = set(tag_values(remove_map, key))
        if not remove_set:
            continue
        kept = [v for v in current_vals if v not in remove_set]
        set_tag_values(merged, key, kept)

    if add_by_key:
        _merge_tags_additive(merged, add_by_key)

    return merged


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

def _text_content_id(content: str) -> str:
    """Generate a content-addressed ID for text updates.

    This makes text updates versioned by content:
    - ``keep put "my note"`` -> ID = %{hash[:12]}
    - ``keep put "my note" -t status=done`` -> same ID, new version
    - ``keep put "different note"`` -> different ID

    Args:
        content: The text content

    Returns:
        Content-addressed ID in format %{hash[:12]}
    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"%{content_hash}"


# ---------------------------------------------------------------------------
# Directory walking (shared by cli.py and watches.py)
# ---------------------------------------------------------------------------

def _has_nested_git_boundary(entry: Path, directory: Path) -> bool:
    """Return true when a file is inside a nested Git checkout or worktree.

    Linked worktrees use a `.git` file rather than a `.git/` directory, so
    hidden-name filtering alone is not enough to keep a recursive import from
    crossing into a separate repository.  The root directory is not considered
    nested; this lets `keep put -r <worktree>` index that worktree normally.
    """
    root = directory.resolve()
    current = entry.parent.resolve()
    while current != root and current != current.parent:
        if (current / ".git").exists():
            return True
        current = current.parent
    return False


def _git_visible_files(
    directory: Path, recurse: bool, exclude: list[str] | None = None,
) -> list[Path] | None:
    """Return files visible to git (tracked + untracked, excluding ignored).

    Returns None if the directory is not inside a git repository or git
    is not available, signalling the caller to fall back to plain walk.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=directory, capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    raw = result.stdout.decode("utf-8", errors="replace")
    if not raw:
        return []

    files = []
    for relpath in raw.split("\0"):
        if not relpath:
            continue
        # Skip paths with any hidden component (.github/, .vscode/, etc.)
        parts = relpath.split("/")
        if any(p.startswith(".") for p in parts):
            continue
        entry = (directory / relpath).resolve()
        # Scope to directory
        if not str(entry).startswith(str(directory.resolve())):
            continue
        # Apply recurse constraint: without -r, only top-level files
        if not recurse and len(parts) > 1:
            continue
        if entry.is_symlink() or not entry.is_file():
            continue
        if _has_nested_git_boundary(entry, directory):
            continue
        # Apply user-specified exclude patterns against relative path
        if exclude and any(fnmatch.fnmatch(relpath, pat) for pat in exclude):
            continue
        files.append(entry)
    return sorted(files)


def _list_directory_files(
    directory: Path, *, recurse: bool = False, exclude: list[str] | None = None,
) -> list[Path]:
    """List regular files in a directory, sorted by name.

    Respects .gitignore when the directory is inside a git repository.
    Skips symlinks, hidden files/dirs (names starting with '.').
    When recurse=True, walks subdirectories.
    Exclude patterns use fnmatch against relative paths (gitignore-style globs).
    """
    # Try git-aware listing first
    git_files = _git_visible_files(directory, recurse, exclude=exclude)
    if git_files is not None:
        return git_files

    # Fallback: plain walk (non-git directories)
    files = []
    if recurse:
        for root, dirs, filenames in os.walk(directory, followlinks=False):
            # Skip hidden directories and nested Git checkouts/worktrees.
            # Modifying dirs in-place prunes the walk before child files are
            # considered, which keeps recursive imports scoped to one tree.
            dirs[:] = sorted(
                d for d in dirs
                if not d.startswith(".") and not (Path(root) / d / ".git").exists()
            )
            # Prune excluded directories early
            if exclude:
                rel_root = Path(root).relative_to(directory)
                dirs[:] = [
                    d for d in dirs
                    if not any(fnmatch.fnmatch(str(rel_root / d), pat) for pat in exclude)
                ]
            root_path = Path(root)
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                entry = root_path / name
                if entry.is_symlink():
                    continue
                if exclude:
                    relpath = str(entry.relative_to(directory))
                    if any(fnmatch.fnmatch(relpath, pat) for pat in exclude):
                        continue
                files.append(entry)
    else:
        for entry in sorted(directory.iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_symlink():
                continue
            if entry.is_dir():
                continue
            if exclude and any(fnmatch.fnmatch(entry.name, pat) for pat in exclude):
                continue
            files.append(entry)
    return files
