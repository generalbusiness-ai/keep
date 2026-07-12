"""Daemon-owned markdown mirror registry and export passes."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .dependencies import NoteDependencyService
from .markdown_export import (
    _resolve_export_destination,
    bundle_export_refs,
    get_edge_data,
    get_export_doc,
    local_edge_tag_resolver,
    render_doc_bundle,
    resolve_remote_render_bundle,
    supports_local_markdown_export_graph,
    write_markdown_export,
)
from .markdown_sync import (
    DOC_STRUCTURAL_MUTATIONS,
    DOC_UPDATE_MUTATION,
    EDGE_MUTATIONS,
    PART_MUTATIONS,
    VERSION_EDGE_MUTATIONS,
    VERSION_MUTATIONS,
    decode_sync_event_payload,
)
from .types import file_uri_to_path, utc_now as _utc_now
from .watches import _DEFAULT_INTERVAL, _DOC_COLLECTION, load_watches, parse_duration

if TYPE_CHECKING:
    from .api import Keeper

logger = logging.getLogger(__name__)

_MIRRORS_ID = ".markdown-mirrors"
_SYNC_DIR = ".keep-sync"
_MAP_FILE = "map.tsv"
_STATE_FILE = "state.json"
_SYNC_OUTBOX_POLL_SECONDS = 1.0
_SYNC_OUTBOX_MAX_EVENTS_PER_POLL = 1000


@dataclass
class MarkdownMirrorEntry:
    """One daemon-managed markdown mirror root."""

    root: str
    include_system: bool = False
    include_parts: bool = False
    include_versions: bool = False
    interval: str = _DEFAULT_INTERVAL
    enabled: bool = True
    added_at: str = ""
    pending_since: str = ""
    pending_full_replan: bool = False
    pending_note_ids: list[str] = field(default_factory=list)
    last_run: str = ""
    last_error: str = ""
    source_cursor: str = ""

    def is_due(self, now: datetime | None = None) -> bool:
        if not self.enabled:
            return False
        if not self.pending_since:
            return False
        now = now or datetime.now(timezone.utc)
        try:
            pending = datetime.fromisoformat(self.pending_since)
            if pending.tzinfo is None:
                pending = pending.replace(tzinfo=timezone.utc)
            return now >= pending + parse_duration(self.interval)
        except (ValueError, TypeError):
            return True


@dataclass(frozen=True)
class MarkdownMirrorUpdatePlan:
    """One mirror pass plan derived from sync outbox events."""

    full_replan: bool
    note_ids: tuple[str, ...]


class _IncrementalExportNeedsFullReplan(RuntimeError):
    """Raised when bounded export cannot preserve the existing namespace."""


def _mirror_to_dict(entry: MarkdownMirrorEntry) -> dict[str, Any]:
    data = asdict(entry)
    # Drop empty defaults for compact storage, but never drop ``enabled``
    # — its default is True, so omitting ``enabled=False`` would silently
    # re-enable a disabled mirror on the next load.
    return {
        k: v for k, v in data.items()
        if k == "enabled" or v not in ("", [], False, _DEFAULT_INTERVAL)
    }


def _dict_to_mirror(data: dict[str, Any]) -> MarkdownMirrorEntry:
    known = {f.name for f in MarkdownMirrorEntry.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known}
    for key, value in list(filtered.items()):
        if isinstance(value, datetime):
            filtered[key] = value.isoformat()
    if isinstance(filtered.get("pending_note_ids"), tuple):
        filtered["pending_note_ids"] = list(filtered["pending_note_ids"])
    return MarkdownMirrorEntry(**filtered)


def _resolve_root(root: str | Path) -> Path:
    raw = str(root)
    if raw.startswith("file://"):
        raw = file_uri_to_path(raw)
    return Path(raw).expanduser().resolve()


def _mirror_registry_path(keeper) -> Path:
    config = getattr(keeper, "config", None) or getattr(keeper, "_config", None)
    candidates = [
        getattr(config, "config_dir", None),
        getattr(config, "path", None),
        getattr(keeper, "_store_path", None),
    ]
    config_dir = next(
        (candidate for candidate in candidates if isinstance(candidate, (str, Path))),
        None,
    )
    if config_dir is None:
        raise ValueError("markdown mirror registry requires a local config directory")
    return Path(config_dir).expanduser().resolve() / "markdown-mirrors.yaml"


def _paths_overlap(a: str | Path, b: str | Path) -> bool:
    pa = _resolve_root(a)
    pb = _resolve_root(b)
    try:
        pa.relative_to(pb)
        return True
    except ValueError:
        pass
    try:
        pb.relative_to(pa)
        return True
    except ValueError:
        return False


def _load_markdown_mirrors_legacy_store(keeper: Keeper) -> list[MarkdownMirrorEntry]:
    if not hasattr(keeper, "_document_store"):
        return []
    rec = keeper._document_store.get(_DOC_COLLECTION, _MIRRORS_ID)
    if rec is None or not rec.summary:
        return []
    try:
        items = yaml.safe_load(rec.summary)
    except yaml.YAMLError:
        logger.warning("Failed to parse %s", _MIRRORS_ID)
        return []
    if not isinstance(items, list):
        return []
    return [
        _dict_to_mirror(item)
        for item in items
        if isinstance(item, dict)
    ]


def load_markdown_mirrors(keeper: Keeper) -> list[MarkdownMirrorEntry]:
    registry_path = _mirror_registry_path(keeper)
    if registry_path.exists():
        try:
            items = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            logger.warning("Failed to parse markdown mirror registry: %s", registry_path)
            return []
        if not isinstance(items, list):
            return []
        return [
            _dict_to_mirror(item)
            for item in items
            if isinstance(item, dict)
        ]
    return _load_markdown_mirrors_legacy_store(keeper)


def save_markdown_mirrors(keeper: Keeper, entries: list[MarkdownMirrorEntry]) -> None:
    payload = [_mirror_to_dict(entry) for entry in entries]
    registry_path = _mirror_registry_path(keeper)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if payload:
        content = yaml.safe_dump(payload, default_flow_style=False)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=registry_path.parent,
            prefix="markdown-mirrors-",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        tmp_path.replace(registry_path)
    else:
        try:
            registry_path.unlink()
        except FileNotFoundError:
            pass

    if hasattr(keeper, "_document_store"):
        try:
            keeper._document_store.delete(_DOC_COLLECTION, _MIRRORS_ID)
        except Exception:
            logger.debug("Failed to clean up legacy mirror registry doc", exc_info=True)


def list_markdown_mirrors(keeper: Keeper) -> list[MarkdownMirrorEntry]:
    return load_markdown_mirrors(keeper)


def has_active_markdown_mirrors(keeper: Keeper) -> bool:
    return any(entry.enabled for entry in load_markdown_mirrors(keeper))


def next_markdown_mirror_delay(entries: list[MarkdownMirrorEntry]) -> float:
    if not entries:
        return 30.0
    now = datetime.now(timezone.utc)
    min_delay = _SYNC_OUTBOX_POLL_SECONDS
    for entry in entries:
        if not entry.enabled:
            continue
        if not entry.pending_since:
            continue
        try:
            pending = datetime.fromisoformat(entry.pending_since)
            if pending.tzinfo is None:
                pending = pending.replace(tzinfo=timezone.utc)
            due_at = pending + parse_duration(entry.interval or _DEFAULT_INTERVAL)
            min_delay = min(min_delay, (due_at - now).total_seconds())
        except (ValueError, TypeError):
            return 0.0
    return max(0.0, min_delay)


def record_markdown_mirror_export_success(
    keeper: Keeper,
    root: str | Path,
    *,
    error: str = "",
    source_cursor: str | None = None,
) -> bool:


    resolved_root = str(_resolve_root(root))
    entries = load_markdown_mirrors(keeper)
    changed = False
    now_ts = _utc_now()
    for entry in entries:
        if entry.root != resolved_root:
            continue
        entry.last_run = now_ts
        entry.pending_since = ""
        entry.pending_full_replan = False
        entry.pending_note_ids = []
        entry.last_error = error
        if source_cursor is not None:
            entry.source_cursor = str(source_cursor)
        changed = True
        break
    if changed:
        save_markdown_mirrors(keeper, entries)
    return changed


def path_inside_markdown_mirror(keeper: Keeper, path: str | Path) -> bool:
    candidate = _resolve_root(path)
    return any(
        _paths_overlap(candidate, entry.root)
        for entry in load_markdown_mirrors(keeper)
        if entry.enabled
    )


def add_markdown_mirror(
    keeper: Keeper,
    root: str | Path,
    *,
    include_system: bool = False,
    include_parts: bool = False,
    include_versions: bool = False,
    interval: str = _DEFAULT_INTERVAL,
) -> MarkdownMirrorEntry:


    resolved_root, entries = validate_markdown_mirror(
        keeper,
        root,
        interval=interval,
    )

    for entry in entries:
        if entry.root != resolved_root:
            continue
        entry.include_system = include_system
        entry.include_parts = include_parts
        entry.include_versions = include_versions
        entry.interval = interval
        entry.enabled = True
        save_markdown_mirrors(keeper, entries)
        return entry

    entry = MarkdownMirrorEntry(
        root=resolved_root,
        include_system=include_system,
        include_parts=include_parts,
        include_versions=include_versions,
        interval=interval,
        enabled=True,
        added_at=_utc_now(),
    )
    entries.append(entry)
    save_markdown_mirrors(keeper, entries)
    return entry


def validate_markdown_mirror(
    keeper: Keeper,
    root: str | Path,
    *,
    interval: str = _DEFAULT_INTERVAL,
) -> tuple[str, list[MarkdownMirrorEntry]]:
    """Validate mirror registration without mutating store state."""
    parse_duration(interval)
    resolved_root = str(_resolve_root(root))
    entries = load_markdown_mirrors(keeper)

    for watch in load_watches(keeper):
        if watch.kind not in ("file", "directory"):
            continue
        if _paths_overlap(resolved_root, watch.source):
            raise ValueError(
                f"Markdown sync root overlaps watched source: {watch.source}"
            )

    for entry in entries:
        if _paths_overlap(resolved_root, entry.root) and entry.root != resolved_root:
            raise ValueError(
                f"Markdown sync root overlaps existing mirror root: {entry.root}"
            )

    return resolved_root, entries


def remove_markdown_mirror(keeper: Keeper, root: str | Path) -> bool:
    resolved_root = str(_resolve_root(root))
    entries = load_markdown_mirrors(keeper)
    kept = [entry for entry in entries if entry.root != resolved_root]
    if len(kept) == len(entries):
        return False
    save_markdown_mirrors(keeper, kept)
    return True


def _sync_paths(root: Path) -> tuple[Path, Path, Path]:
    sync_dir = root / _SYNC_DIR
    return sync_dir, sync_dir / _MAP_FILE, sync_dir / _STATE_FILE


def _load_map(root: Path) -> dict[str, str]:
    _sync_dir, map_path, _state_path = _sync_paths(root)
    if not map_path.is_file():
        return {}
    lines = map_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    entries: dict[str, str] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            export_ref, keep_id = line.split("\t", 1)
        except ValueError:
            continue
        entries[export_ref] = keep_id
    return entries


def _write_map(root: Path, entries: dict[str, str]) -> None:
    sync_dir, map_path, _state_path = _sync_paths(root)
    sync_dir.mkdir(parents=True, exist_ok=True)
    lines = ["export_ref\tkeep_id"]
    for export_ref, keep_id in sorted(entries.items()):
        lines.append(f"{export_ref}\t{keep_id}")
    map_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_state(
    root: Path,
    *,
    entry: MarkdownMirrorEntry | None,
    count: int,
    info: dict[str, Any],
) -> None:


    sync_dir, _map_path, state_path = _sync_paths(root)
    sync_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "keep-markdown-sync",
        "version": 1,
        "updated_at": _utc_now(),
        "count": count,
        "store_info": info,
    }
    if entry is not None:
        payload["mirror"] = {
            "root": entry.root,
            "include_system": entry.include_system,
            "include_parts": entry.include_parts,
            "include_versions": entry.include_versions,
            "interval": entry.interval,
            "enabled": entry.enabled,
        }
    state_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _keep_id_from_exported_file(rel_path: Path, text: str) -> str | None:
    if not text.startswith("---\n"):
        return None
    try:
        frontmatter = yaml.safe_load(text.split("---", 2)[1])
    except Exception:
        return None
    if not isinstance(frontmatter, dict):
        return None
    parent_id = frontmatter.get("_id")
    if not isinstance(parent_id, str) or not parent_id:
        return None
    name = rel_path.name
    if name.startswith("@P{") and isinstance(frontmatter.get("_part_num"), int):
        return f"{parent_id}@P{{{frontmatter['_part_num']}}}"
    if name.startswith("@V{") and isinstance(frontmatter.get("_version_offset"), int):
        return f"{parent_id}@V{{{frontmatter['_version_offset']}}}"
    return parent_id


def _prune_empty_dirs(root: Path, start: Path) -> None:
    current = start
    while current != root and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _is_exported_note_id(doc_id: str, *, include_system: bool) -> bool:
    return include_system or not doc_id.startswith(".")


def _map_reverse(entries: dict[str, str]) -> dict[str, str]:
    return {keep_id: export_ref for export_ref, keep_id in entries.items()}


def _bundle_map_entries(
    existing_map: dict[str, str],
    doc_id: str,
) -> dict[str, str]:
    # Bundle ids use the reserved `@P{n}` / `@V{n}` suffix syntax, so prefix
    # matching stays scoped to that note's exported sidecars.
    return {
        export_ref: keep_id
        for export_ref, keep_id in existing_map.items()
        if keep_id == doc_id
        or keep_id.startswith(f"{doc_id}@P{{")
        or keep_id.startswith(f"{doc_id}@V{{")
        or keep_id.startswith(f"{doc_id}@p")
    }


def _delete_rel_path(root: Path, rel_path: Path) -> None:
    # rel_path can come from the on-disk .keep-sync map, which is not
    # trusted: refuse to delete anything outside the mirror root (the full
    # export path enforces the same containment on writes). Validate with
    # the resolved path, but unlink the original lexical path — unlinking
    # the resolved path would follow an in-root symlink and delete its
    # target instead of the stale link itself.
    try:
        _resolve_export_destination(root, rel_path)
    except ValueError:
        logger.warning("Skipping mirror delete outside root: %s", rel_path)
        return
    target = root / rel_path
    if target.is_symlink() or target.exists():
        target.unlink()
        _prune_empty_dirs(root, target.parent)


def _current_store_info(keeper: Keeper, *, include_system: bool) -> dict[str, Any]:
    it = keeper.export_iter(include_system=include_system)
    try:
        header = next(it)
        return dict(header.get("store_info") or {})
    finally:
        close = getattr(it, "close", None)
        if callable(close):
            close()


_FULL_REPLAN = object()
_SKIP_EVENT = object()


def _resolve_local_markdown_sync_targets(
    dependencies: NoteDependencyService,
    *,
    mutation: str,
    doc_id: str,
    payload: dict[str, Any],
    include_system: bool,
    expanded_targets: dict[str, list[str]],
    row: dict[str, Any],
):
    if mutation in DOC_STRUCTURAL_MUTATIONS:
        if _is_exported_note_id(doc_id, include_system=include_system):
            return _FULL_REPLAN
        return _SKIP_EVENT

    if mutation == DOC_UPDATE_MUTATION:
        if doc_id not in expanded_targets:
            expanded_targets[doc_id] = dependencies.all_target_ids(doc_id)
        return [doc_id, *expanded_targets[doc_id]]

    if mutation in PART_MUTATIONS or mutation in VERSION_MUTATIONS:
        return [doc_id]

    if mutation in EDGE_MUTATIONS or mutation in VERSION_EDGE_MUTATIONS:
        return [
            doc_id,
            str(payload.get("target_id") or ""),
            str(payload.get("old_target_id") or ""),
        ]

    return _FULL_REPLAN


def _resolve_remote_markdown_sync_targets(
    *,
    mutation: str,
    doc_id: str,
    payload: dict[str, Any],
    include_system: bool,
    expanded_targets: dict[str, list[str]],
    row: dict[str, Any],
):
    if mutation in DOC_STRUCTURAL_MUTATIONS:
        if _is_exported_note_id(doc_id, include_system=include_system):
            return _FULL_REPLAN
        return _SKIP_EVENT

    affected_note_ids = row.get("affected_note_ids")
    if not isinstance(affected_note_ids, list):
        affected_note_ids = []
    affected_note_ids = [str(note_id) for note_id in affected_note_ids if note_id]

    if mutation == DOC_UPDATE_MUTATION:
        return affected_note_ids or _FULL_REPLAN

    if mutation in PART_MUTATIONS or mutation in VERSION_MUTATIONS:
        return affected_note_ids or [doc_id]

    if mutation in EDGE_MUTATIONS or mutation in VERSION_EDGE_MUTATIONS:
        return affected_note_ids or [
            doc_id,
            str(payload.get("target_id") or ""),
            str(payload.get("old_target_id") or ""),
        ]

    return _FULL_REPLAN


def _plan_markdown_mirror_update_common(
    entry: MarkdownMirrorEntry,
    events: list[dict[str, Any]],
    *,
    resolve_targets,
) -> MarkdownMirrorUpdatePlan:
    if not events:
        return MarkdownMirrorUpdatePlan(full_replan=True, note_ids=())
    note_ids: set[str] = set()
    expanded_targets: dict[str, list[str]] = {}

    def _maybe_add(doc_id: str) -> None:
        if not doc_id:
            return
        if _is_exported_note_id(doc_id, include_system=entry.include_system):
            note_ids.add(doc_id)

    for row in events:
        if row.get("collection") != _DOC_COLLECTION:
            continue
        mutation = str(row.get("mutation") or "")
        doc_id = str(row.get("entity_id") or "")
        payload = decode_sync_event_payload(row.get("payload_json"))
        resolved = resolve_targets(
            mutation=mutation,
            doc_id=doc_id,
            payload=payload,
            include_system=entry.include_system,
            expanded_targets=expanded_targets,
            row=row,
        )
        if resolved is _FULL_REPLAN:
            return MarkdownMirrorUpdatePlan(full_replan=True, note_ids=())
        if resolved is _SKIP_EVENT:
            continue
        for note_id in resolved:
            _maybe_add(note_id)

    return MarkdownMirrorUpdatePlan(
        full_replan=False,
        note_ids=tuple(sorted(note_ids)),
    )


def _plan_markdown_mirror_update(
    keeper: Keeper,
    entry: MarkdownMirrorEntry,
    events: list[dict[str, Any]],
) -> MarkdownMirrorUpdatePlan:
    doc_coll = keeper._resolve_doc_collection()
    dependencies = NoteDependencyService(keeper._document_store, doc_coll)
    return _plan_markdown_mirror_update_common(
        entry,
        events,
        resolve_targets=lambda **kwargs: _resolve_local_markdown_sync_targets(
            dependencies,
            **kwargs,
        ),
    )


def _plan_markdown_mirror_update_remote(
    entry: MarkdownMirrorEntry,
    events: list[dict[str, Any]],
) -> MarkdownMirrorUpdatePlan:
    return _plan_markdown_mirror_update_common(
        entry,
        events,
        resolve_targets=_resolve_remote_markdown_sync_targets,
    )


def run_markdown_export_incremental(
    keeper: Keeper,
    root: str | Path,
    *,
    note_ids: list[str],
    include_system: bool,
    include_parts: bool = False,
    include_versions: bool = False,
    mirror_entry: MarkdownMirrorEntry | None = None,
) -> tuple[int, dict[str, Any]]:
    """Rewrite only selected exported note bundles under an existing mirror."""
    out_dir = _resolve_root(root)
    if not out_dir.is_dir():
        raise ValueError(f"{out_dir} does not exist")

    existing_map = _load_map(out_dir)
    reverse_map = _map_reverse(existing_map)
    export_refs = dict(reverse_map)
    local_graph = supports_local_markdown_export_graph(keeper)
    if local_graph:
        local_is_edge_tag = local_edge_tag_resolver(keeper)
        current_inverse_lookup, version_inverse_lookup = get_edge_data(
            keeper, export_refs=export_refs,
        )
    elif not hasattr(keeper, "export_bundle"):
        raise ValueError(
            "incremental markdown export requires a host with either local "
            "graph access or export_bundle() support"
        )

    updated_entries = dict(existing_map)
    rewritten = 0

    for doc_id in note_ids:
        existing_export_ref = reverse_map.get(doc_id)
        if existing_export_ref is None:
            raise _IncrementalExportNeedsFullReplan(
                f"incremental export missing namespace mapping for {doc_id}"
            )
        rel_path = Path(f"{existing_export_ref}.md")
        if local_graph:
            doc = get_export_doc(keeper, doc_id)
            if doc is None:
                raise _IncrementalExportNeedsFullReplan(
                    f"incremental export missing note {doc_id}"
                )
            current_inverse = current_inverse_lookup(doc_id)
            version_inverse = version_inverse_lookup(doc_id)
            is_edge_tag = local_is_edge_tag
        else:
            bundle = keeper.export_bundle(
                doc_id,
                include_system=include_system,
                include_parts=include_parts,
                include_versions=include_versions,
            )
            if not isinstance(bundle, dict):
                raise _IncrementalExportNeedsFullReplan(
                    f"incremental export missing note bundle for {doc_id}"
                )
            remote_bundle = resolve_remote_render_bundle(
                bundle,
                export_refs=export_refs,
            )
            if not isinstance(remote_bundle.document, dict):
                raise _IncrementalExportNeedsFullReplan(
                    f"incremental export missing document payload for {doc_id}"
                )
            doc = remote_bundle.document
            current_inverse = remote_bundle.current_inverse
            version_inverse = remote_bundle.version_inverse
            is_edge_tag = remote_bundle.is_edge_tag
        if not _is_exported_note_id(doc_id, include_system=include_system):
            continue

        bundle_refs = bundle_export_refs(
            doc,
            rel_path,
            include_parts=include_parts,
            include_versions=include_versions,
        )
        export_refs.update(bundle_refs)
        bundle_entries = {
            export_ref: keep_id
            for keep_id, export_ref in bundle_refs.items()
            if keep_id == doc_id or "@P{" in keep_id or "@V{" in keep_id
        }
        files = render_doc_bundle(
            keeper,
            doc,
            rel_path,
            include_system=include_system,
            include_parts=include_parts,
            include_versions=include_versions,
            export_refs=export_refs,
            current_inverse=lambda _doc_id, edges=current_inverse: edges,
            version_inverse=lambda _doc_id, edges=version_inverse: edges,
            is_edge_tag=is_edge_tag,
        )
        for file_rel, text in files.items():
            # Same containment guard as the full-export writer: refs derived
            # from note ids / the on-disk map must not escape the mirror root.
            try:
                dest = _resolve_export_destination(out_dir, Path(file_rel))
            except ValueError:
                logger.warning("Skipping mirror write outside root: %s", file_rel)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")

        previous_entries = _bundle_map_entries(updated_entries, doc_id)
        stale_refs = set(previous_entries) - set(bundle_entries)
        for stale_ref in stale_refs:
            _delete_rel_path(out_dir, Path(f"{stale_ref}.md"))
            updated_entries.pop(stale_ref, None)

        for old_ref in previous_entries:
            updated_entries.pop(old_ref, None)
        updated_entries.update(bundle_entries)
        rewritten += 1

    _write_map(out_dir, updated_entries)
    info = _current_store_info(keeper, include_system=include_system)
    _write_state(out_dir, entry=mirror_entry, count=info["document_count"], info=info)
    return rewritten, info


def run_markdown_export_once(
    keeper: Keeper,
    root: str | Path,
    *,
    include_system: bool,
    include_parts: bool = False,
    include_versions: bool = False,
    allow_existing: bool = False,
    mirror_entry: MarkdownMirrorEntry | None = None,
    progress: Any | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run one markdown export pass, optionally preserving an existing mirror."""
    out_dir = _resolve_root(root)
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"{out_dir} exists and is not a directory")
        if not allow_existing and any(out_dir.iterdir()):
            raise ValueError(f"output directory {out_dir} is not empty")
    else:
        out_dir.mkdir(parents=True)

    prev_map = _load_map(out_dir) if allow_existing else {}

    with tempfile.TemporaryDirectory(prefix="keep-md-export-", dir=out_dir.parent) as tmp:
        tmpdir = Path(tmp)
        new_map: dict[str, str] = {}
        new_paths: set[Path] = set()
        count, info = write_markdown_export(
            keeper,
            tmpdir,
            include_system=include_system,
            include_parts=include_parts,
            include_versions=include_versions,
            progress=progress,
            export_map=new_map,
            written_paths=new_paths,
        )

        for rel_path in sorted(new_paths):
            try:
                dest = _resolve_export_destination(out_dir, rel_path)
            except ValueError:
                logger.warning("Skipping mirror write outside root: %s", rel_path)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmpdir / rel_path, dest)

    if allow_existing:
        # prev_map is loaded from the on-disk .keep-sync map (untrusted);
        # _delete_rel_path enforces containment within out_dir.
        old_paths = {Path(f"{export_ref}.md") for export_ref in prev_map}
        stale_paths = sorted(old_paths - new_paths, reverse=True)
        for rel_path in stale_paths:
            _delete_rel_path(out_dir, rel_path)

    _write_map(out_dir, new_map)
    _write_state(out_dir, entry=mirror_entry, count=count, info=info)
    return count, info


def _drain_sync_outbox(
    keeper: Keeper,
    *,
    discard: bool,
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    # Bound each poll's outbox work so large mutation bursts do not turn one
    # daemon tick into an unbounded drain-and-plan pause.
    events: list[dict[str, Any]] = []
    discarded = 0
    while True:
        remaining = None if max_rows is None else max_rows - (len(events) + discarded)
        if remaining is not None and remaining <= 0:
            break
        rows = keeper._document_store.dequeue_sync_outbox(
            limit=200 if remaining is None else min(200, remaining),
        )
        if not rows:
            break
        outbox_ids = [row["outbox_id"] for row in rows]
        if discard:
            keeper._document_store.complete_sync_outbox(outbox_ids)
            discarded += len(rows)
            continue
        events.extend(rows)
        keeper._document_store.complete_sync_outbox(outbox_ids)
    return events, discarded


def clear_sync_outbox(keeper: Keeper) -> int:
    """Discard all pending sync-outbox rows after a baseline full export.

    This is for one-shot baseline exports that have already materialized the
    entire store state into a mirror root, such as the initial `--sync`
    registration path. It should not run after ordinary mirror polling passes,
    because that would drop events that arrived while another mirror was
    exporting.
    """
    _events, discarded = _drain_sync_outbox(keeper, discard=True)
    return discarded


def _supports_export_change_feed(keeper: Any) -> bool:
    if hasattr(keeper, "supports_capability"):
        try:
            return bool(keeper.supports_capability("export_changes"))
        except Exception:
            return False
    return hasattr(keeper, "export_changes")


def poll_markdown_mirrors(keeper: Keeper, *, source_keeper=None) -> dict[str, int]:
    # v1 uses the sync outbox only as a precise trigger boundary. Any activity
    # is coalesced by the mirror interval. Structural changes still force a
    # whole-mirror replan; ordinary note/part/version/edge changes can now be
    # handled as bounded incremental rewrites through the shared dependency
    # service.
    source_keeper = source_keeper or keeper
    entries = load_markdown_mirrors(keeper)


    if not entries:
        if source_keeper is keeper:
            _events, discarded = _drain_sync_outbox(keeper, discard=True)
        else:
            discarded = 0
        return {"checked": 0, "exported": 0, "errors": 0, "discarded": discarded}

    now = datetime.now(timezone.utc)
    now_ts = _utc_now()
    dirty = False
    stats = {"checked": 0, "exported": 0, "errors": 0, "discarded": 0}
    if source_keeper is keeper:
        events, _discarded = _drain_sync_outbox(
            keeper,
            discard=False,
            max_rows=_SYNC_OUTBOX_MAX_EVENTS_PER_POLL,
        )
        if events:
            for entry in entries:
                if not entry.enabled:
                    continue
                plan = _plan_markdown_mirror_update(keeper, entry, events)
                if not plan.full_replan and not plan.note_ids:
                    continue
                event_times = [
                    (row.get("created_at") or now_ts)
                    for row in events
                    if row.get("collection") == _DOC_COLLECTION
                ]
                pending_since = min(
                    [entry.pending_since] + event_times
                    if entry.pending_since else event_times
                )
                if entry.pending_since != pending_since:
                    entry.pending_since = pending_since
                if plan.full_replan:
                    entry.pending_full_replan = True
                    entry.pending_note_ids = []
                elif not entry.pending_full_replan:
                    merged = set(entry.pending_note_ids)
                    merged.update(plan.note_ids)
                    entry.pending_note_ids = sorted(merged)
                dirty = True
    elif _supports_export_change_feed(source_keeper):
        for entry in entries:
            if not entry.enabled:
                continue
            try:
                feed = source_keeper.export_changes(
                    cursor=entry.source_cursor or "0",
                    limit=_SYNC_OUTBOX_MAX_EVENTS_PER_POLL,
                )
            except Exception as exc:
                entry.last_error = str(exc)
                stats["errors"] += 1
                dirty = True
                continue

            if not isinstance(feed, dict):
                entry.last_error = "invalid export change feed response"
                stats["errors"] += 1
                dirty = True
                continue

            head_cursor = str(feed.get("head_cursor") or entry.source_cursor or "0")
            next_cursor = str(feed.get("cursor") or entry.source_cursor or head_cursor)
            compacted = bool(feed.get("compacted"))
            truncated = bool(feed.get("truncated"))
            events = feed.get("events")
            if not isinstance(events, list):
                events = []

            target_cursor = head_cursor if (compacted or truncated) else next_cursor
            if entry.source_cursor != target_cursor:
                entry.source_cursor = target_cursor
                dirty = True

            if not compacted and not truncated and not events:
                continue

            if compacted or truncated:
                plan = MarkdownMirrorUpdatePlan(full_replan=True, note_ids=())
            else:
                plan = _plan_markdown_mirror_update_remote(entry, events)
                if not plan.full_replan and not plan.note_ids:
                    continue

            event_times = [
                str(row.get("created_at") or now_ts)
                for row in events
                if isinstance(row, dict)
            ]
            if event_times:
                pending_since = min(
                    [entry.pending_since] + event_times
                    if entry.pending_since else event_times
                )
            else:
                pending_since = entry.pending_since or entry.last_run or entry.added_at or now_ts
            if entry.pending_since != pending_since:
                entry.pending_since = pending_since
            if plan.full_replan:
                entry.pending_full_replan = True
                entry.pending_note_ids = []
            elif not entry.pending_full_replan:
                merged = set(entry.pending_note_ids)
                merged.update(plan.note_ids)
                entry.pending_note_ids = sorted(merged)
            entry.last_error = ""
            dirty = True
    else:
        for entry in entries:
            if not entry.enabled:
                continue
            pending_since = entry.pending_since or entry.last_run or entry.added_at or now_ts
            if entry.pending_since != pending_since:
                entry.pending_since = pending_since
                dirty = True
    if source_keeper is keeper:
        for entry in entries:
            if entry.pending_since and not entry.pending_full_replan and not entry.pending_note_ids:
                entry.pending_since = ""
                dirty = True
    for entry in entries:
        if not entry.is_due(now):
            continue
        plan = MarkdownMirrorUpdatePlan(
            full_replan=entry.pending_full_replan or not entry.pending_note_ids,
            note_ids=tuple(entry.pending_note_ids),
        )
        if not plan.full_replan and not plan.note_ids:
            entry.pending_since = ""
            entry.last_error = ""
            entry.pending_note_ids = []
            entry.pending_full_replan = False
            dirty = True
            continue
        stats["checked"] += 1
        try:
            if plan.full_replan:
                run_markdown_export_once(
                    source_keeper,
                    entry.root,
                    include_system=entry.include_system,
                    include_parts=entry.include_parts,
                    include_versions=entry.include_versions,
                    allow_existing=True,
                    mirror_entry=entry,
                )
            else:
                run_markdown_export_incremental(
                    source_keeper,
                    entry.root,
                    note_ids=list(plan.note_ids),
                    include_system=entry.include_system,
                    include_parts=entry.include_parts,
                    include_versions=entry.include_versions,
                    mirror_entry=entry,
                )
            entry.last_run = now_ts
            entry.pending_since = ""
            entry.pending_note_ids = []
            entry.pending_full_replan = False
            entry.last_error = ""
            stats["exported"] += 1
        except _IncrementalExportNeedsFullReplan:
            run_markdown_export_once(
                source_keeper,
                entry.root,
                include_system=entry.include_system,
                include_parts=entry.include_parts,
                include_versions=entry.include_versions,
                allow_existing=True,
                mirror_entry=entry,
            )
            entry.last_run = now_ts
            entry.pending_since = ""
            entry.pending_note_ids = []
            entry.pending_full_replan = False
            entry.last_error = ""
            stats["exported"] += 1
        except Exception as exc:
            entry.last_error = str(exc)
            logger.warning("Markdown mirror export failed for %s: %s", entry.root, exc)
            stats["errors"] += 1
        dirty = True

    if dirty:
        save_markdown_mirrors(keeper, entries)
    return stats
