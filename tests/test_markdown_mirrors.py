"""Tests for daemon-owned markdown mirror registry and export passes."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from keep.api import Keeper
from keep.document_store import PartInfo
from keep.markdown_mirrors import (
    MarkdownMirrorEntry,
    _dict_to_mirror,
    _mirror_registry_path,
    add_markdown_mirror,
    clear_sync_outbox,
    list_markdown_mirrors,
    next_markdown_mirror_delay,
    poll_markdown_mirrors,
    record_markdown_mirror_export_success,
    run_markdown_export_incremental,
    run_markdown_export_once,
    save_markdown_mirrors,
)
from keep.types import utc_now
from keep.watches import add_watch


def _create_tagdoc(kp: Keeper, key: str, inverse: str) -> None:
    doc_coll = kp._resolve_doc_collection()
    now = utc_now()
    kp._document_store.upsert(
        collection=doc_coll,
        id=f".tag/{key}",
        summary=f"Tag: {key}",
        tags={
            "_inverse": inverse,
            "_created": now,
            "_updated": now,
            "_source": "inline",
            "category": "system",
        },
        archive=False,
    )


def test_add_markdown_mirror_rejects_watch_overlap(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        watched = tmp_path / "watched"
        watched.mkdir()
        add_watch(kp, str(watched), "directory")
        with pytest.raises(ValueError, match="overlaps watched source"):
            add_markdown_mirror(kp, watched / "vault")
    finally:
        kp.close()


def test_add_watch_rejects_markdown_mirror_overlap(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        root = tmp_path / "vault"
        root.mkdir()
        add_markdown_mirror(kp, root)
        with pytest.raises(ValueError, match="overlaps markdown sync root"):
            add_watch(kp, str(root / "notes"), "directory")
    finally:
        kp.close()


def test_put_from_markdown_mirror_root_rejected(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        root = tmp_path / "vault"
        root.mkdir()
        add_markdown_mirror(kp, root)
        note_path = root / "note.md"
        note_path.write_text("hello", encoding="utf-8")
        with pytest.raises(ValueError, match="synced markdown mirror root"):
            kp.put(uri=f"file://{note_path}")
    finally:
        kp.close()


def test_run_markdown_export_once_writes_map_state_and_cleans_stale(
    mock_providers, tmp_path,
):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        kp.put("Alpha body", id="alpha")
        root = tmp_path / "vault"
        root.mkdir()
        obsidian = root / ".obsidian"
        obsidian.mkdir()
        (obsidian / "workspace.json").write_text("{}", encoding="utf-8")

        count, info = run_markdown_export_once(
            kp,
            root,
            include_system=False,
            allow_existing=True,
        )
        assert count == 1
        assert info["document_count"] == 1
        assert (root / "alpha.md").is_file()
        assert (root / ".keep-sync" / "map.tsv").is_file()
        assert (root / ".keep-sync" / "state.json").is_file()
        assert (root / ".obsidian" / "workspace.json").is_file()

        kp.delete("alpha")
        kp.put("Beta body", id="beta")
        run_markdown_export_once(
            kp,
            root,
            include_system=False,
            allow_existing=True,
        )

        assert not (root / "alpha.md").exists()
        assert (root / "beta.md").is_file()
        state = json.loads((root / ".keep-sync" / "state.json").read_text(encoding="utf-8"))
        assert state["count"] == 1
        map_text = (root / ".keep-sync" / "map.tsv").read_text(encoding="utf-8")
        assert "beta\tbeta" in map_text
        assert "alpha\talpha" not in map_text
    finally:
        kp.close()


def test_run_markdown_export_once_maps_read_only_sidecars(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        kp.put("Version one " + ("a" * 700), id="doc1")
        kp.put("Version two " + ("b" * 700), id="doc1")
        doc_coll = kp._resolve_doc_collection()
        kp._document_store.upsert_single_part(
            doc_coll,
            "doc1",
            PartInfo(
                part_num=1,
                summary="Part one",
                tags={"_part_num": "1", "_base_id": "doc1"},
                created_at=utc_now(),
            ),
        )

        root = tmp_path / "vault"
        count, _info = run_markdown_export_once(
            kp,
            root,
            include_system=False,
            include_parts=True,
            include_versions=True,
            allow_existing=True,
        )

        assert count == 1
        entries = (root / ".keep-sync" / "map.tsv").read_text(encoding="utf-8")
        assert "doc1\tdoc1" in entries
        assert "doc1/@P{1}\tdoc1@P{1}" in entries
        assert "doc1/@V{1}\tdoc1@V{1}" in entries
    finally:
        kp.close()


def test_run_markdown_export_once_handles_uri_parent_with_md_suffix_sidecars(
    mock_providers, tmp_path,
):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        kp.put(
            "Current body",
            id="file:///Users/openclaw/keep-uri-test/brief.md",
        )
        kp.put(
            "Older body " + ("x" * 700),
            id="file:///Users/openclaw/keep-uri-test/brief.md",
        )

        root = tmp_path / "vault"
        count, _info = run_markdown_export_once(
            kp,
            root,
            include_system=False,
            include_versions=True,
            allow_existing=True,
        )

        assert count == 1
        assert (root / "file" / "Users" / "openclaw" / "keep-uri-test" / "brief.md.md").is_file()
        assert (root / "file" / "Users" / "openclaw" / "keep-uri-test" / "brief.md" / "@V{1}.md").is_file()
        entries = (root / ".keep-sync" / "map.tsv").read_text(encoding="utf-8")
        assert (
            "file/Users/openclaw/keep-uri-test/brief.md/@V{1}\t"
            "file:///Users/openclaw/keep-uri-test/brief.md@V{1}"
        ) in entries
    finally:
        kp.close()


def test_run_markdown_export_bundle_host_supports_full_and_incremental(tmp_path):
    class BundleHost:
        def __init__(self) -> None:
            self.docs = {
                "alpha": {
                    "id": "alpha",
                    "summary": "Alpha body",
                    "tags": {},
                    "created_at": "2026-04-09T18:00:00",
                    "updated_at": "2026-04-09T18:00:00",
                    "accessed_at": "2026-04-09T18:00:00",
                },
                "beta/path": {
                    "id": "beta/path",
                    "summary": "Beta body",
                    "tags": {"speaker": "alpha"},
                    "created_at": "2026-04-09T18:00:00",
                    "updated_at": "2026-04-09T18:00:00",
                    "accessed_at": "2026-04-09T18:00:00",
                },
            }

        def export_iter(self, *, include_system: bool = True):
            yield {
                "format": "keep-export",
                "version": 3,
                "exported_at": "2026-04-09T18:00:00",
                "store_info": {
                    "document_count": len(self.docs),
                    "version_count": 0,
                    "part_count": 0,
                    "collection": "default",
                },
            }
            for doc_id in ("alpha", "beta/path"):
                yield dict(self.docs[doc_id])

        def export_bundle(
            self,
            id: str,
            *,
            include_system: bool = True,
            include_parts: bool = True,
            include_versions: bool = True,
        ):
            if id == "alpha":
                return {
                    "document": dict(self.docs["alpha"]),
                    "current_inverse": [["said", "beta/path"]],
                    "version_inverse": [],
                    "edge_tag_keys": [],
                }
            if id == "beta/path":
                return {
                    "document": dict(self.docs["beta/path"]),
                    "current_inverse": [],
                    "version_inverse": [],
                    "edge_tag_keys": ["speaker"],
                }
            return None

    host = BundleHost()
    root = tmp_path / "vault"

    count, info = run_markdown_export_once(
        host,
        root,
        include_system=False,
        allow_existing=True,
    )
    assert count == 2
    assert info["document_count"] == 2
    assert "[[beta/path]]" in (root / "alpha.md").read_text(encoding="utf-8")
    beta_path = root / "beta" / "path.md"
    assert beta_path.is_file()
    assert 'speaker: "[[alpha]]"' in beta_path.read_text(encoding="utf-8")

    alpha_path = root / "alpha.md"
    alpha_before = alpha_path.stat().st_mtime_ns
    beta_before = beta_path.stat().st_mtime_ns

    time.sleep(0.01)
    host.docs["alpha"]["summary"] = "Alpha body updated"
    run_markdown_export_incremental(
        host,
        root,
        note_ids=["alpha"],
        include_system=False,
    )

    assert alpha_path.stat().st_mtime_ns > alpha_before
    assert beta_path.stat().st_mtime_ns == beta_before
    assert "Alpha body updated" in alpha_path.read_text(encoding="utf-8")


def test_list_markdown_mirrors_returns_saved_entries(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        root = tmp_path / "vault"
        add_markdown_mirror(kp, root, include_parts=True, interval="PT5M")
        entries = list_markdown_mirrors(kp)
        assert len(entries) == 1
        assert entries[0].root == str(root.resolve())
        assert entries[0].include_parts is True
        assert entries[0].interval == "PT5M"
    finally:
        kp.close()


def test_markdown_mirror_registry_is_local_file_not_store_doc(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        root = tmp_path / "vault"
        add_markdown_mirror(kp, root, include_parts=True, interval="PT5M")

        config_dir = kp.config.config_dir or kp.config.path
        registry_path = Path(config_dir) / "markdown-mirrors.yaml"
        assert registry_path.is_file()
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        assert payload[0]["root"] == str(root.resolve())
        assert kp._document_store.get("default", ".markdown-mirrors") is None
    finally:
        kp.close()


def test_markdown_mirror_registry_migrates_legacy_store_doc(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        legacy_payload = yaml.safe_dump([{
            "root": str((tmp_path / "vault").resolve()),
            "include_parts": True,
            "interval": "PT5M",
        }])
        kp._document_store.upsert(
            collection="default",
            id=".markdown-mirrors",
            summary=legacy_payload,
            tags={},
            archive=False,
        )

        entries = list_markdown_mirrors(kp)
        assert len(entries) == 1
        assert entries[0].include_parts is True

        save_markdown_mirrors(kp, entries)

        config_dir = kp.config.config_dir or kp.config.path
        registry_path = Path(config_dir) / "markdown-mirrors.yaml"
        assert registry_path.is_file()
        assert kp._document_store.get("default", ".markdown-mirrors") is None
    finally:
        kp.close()


def test_mirror_registry_path_ignores_mock_config_dir_and_falls_back_to_store_path(tmp_path):
    keeper = MagicMock()
    keeper.config = MagicMock()
    keeper.config.config_dir = MagicMock()
    keeper.config.path = MagicMock()
    keeper._store_path = tmp_path / "store"

    registry_path = _mirror_registry_path(keeper)

    assert registry_path == (keeper._store_path / "markdown-mirrors.yaml").resolve()


def test_dict_to_mirror_coerces_any_datetime_fields():
    entry = _dict_to_mirror({
        "root": "/tmp/vault",
        "added_at": datetime(2026, 4, 9, 12, 0, 0),
        "pending_since": datetime(2026, 4, 9, 12, 1, 0),
        "last_run": datetime(2026, 4, 9, 12, 2, 0),
    })

    assert entry.added_at == "2026-04-09T12:00:00"
    assert entry.pending_since == "2026-04-09T12:01:00"
    assert entry.last_run == "2026-04-09T12:02:00"


def test_poll_markdown_mirrors_debounces_sync_outbox(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        root = tmp_path / "vault"
        root.mkdir()
        run_markdown_export_once(
            kp,
            root,
            include_system=False,
            allow_existing=True,
        )
        add_markdown_mirror(kp, root, interval="PT5M")

        kp.put("Alpha body", id="alpha")
        stats = poll_markdown_mirrors(kp)
        assert stats["exported"] == 0
        assert not (root / "alpha.md").exists()

        entries = list_markdown_mirrors(kp)
        assert len(entries) == 1
        assert entries[0].pending_since
        entries[0].pending_since = "2000-01-01T00:00:00"
        save_markdown_mirrors(kp, entries)

        stats = poll_markdown_mirrors(kp)
        assert stats["exported"] == 1
        assert (root / "alpha.md").is_file()

        refreshed = list_markdown_mirrors(kp)
        assert refreshed[0].pending_since == ""
        assert refreshed[0].last_run
    finally:
        kp.close()


def test_poll_markdown_mirrors_incremental_rewrites_only_changed_note(
    mock_providers, tmp_path,
):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        kp.put("Alpha body", id="alpha")
        kp.put("Beta body", id="beta")
        root = tmp_path / "vault"
        root.mkdir()
        run_markdown_export_once(
            kp,
            root,
            include_system=False,
            allow_existing=True,
        )
        clear_sync_outbox(kp)
        # Long interval (debounce window) on purpose — shared by the poll tests
        # below. They assert the FIRST poll does NOT export (the change is still
        # within the window), then force a re-export by backdating pending_since.
        # A short interval like PT1S races with put→poll latency: on a loaded CI
        # runner >1s elapses, the window expires, and the first poll exports
        # unexpectedly (a real flake observed on Python 3.13 CI). PT1H can never
        # expire mid-test; the backdated pending_since still makes is_due() true.
        add_markdown_mirror(kp, root, interval="PT1H")

        alpha_path = root / "alpha.md"
        beta_path = root / "beta.md"
        alpha_before = alpha_path.stat().st_mtime_ns
        beta_before = beta_path.stat().st_mtime_ns

        time.sleep(0.01)
        kp.put("Alpha body updated", id="alpha")
        stats = poll_markdown_mirrors(kp)
        assert stats["exported"] == 0
        entries = list_markdown_mirrors(kp)
        entries[0].pending_since = "2000-01-01T00:00:00"
        save_markdown_mirrors(kp, entries)
        stats = poll_markdown_mirrors(kp)

        assert stats["exported"] == 1
        assert alpha_path.stat().st_mtime_ns > alpha_before
        assert beta_path.stat().st_mtime_ns == beta_before
    finally:
        kp.close()


def test_poll_markdown_mirrors_remote_source_reexports_on_interval(
    mock_providers, tmp_path,
):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        class BundleHost:
            def __init__(self) -> None:
                self.docs = {
                    "alpha": {
                        "id": "alpha",
                        "summary": "Alpha body",
                        "tags": {},
                        "created_at": "2026-04-09T18:00:00",
                        "updated_at": "2026-04-09T18:00:00",
                        "accessed_at": "2026-04-09T18:00:00",
                    },
                }

            def export_iter(self, *, include_system: bool = True):
                yield {
                    "format": "keep-export",
                    "version": 3,
                    "exported_at": "2026-04-09T18:00:00",
                    "store_info": {
                        "document_count": 1,
                        "version_count": 0,
                        "part_count": 0,
                        "collection": "default",
                    },
                }
                yield dict(self.docs["alpha"])

            def export_bundle(
                self,
                id: str,
                *,
                include_system: bool = True,
                include_parts: bool = True,
                include_versions: bool = True,
            ):
                if id != "alpha":
                    return None
                return {
                    "document": dict(self.docs["alpha"]),
                    "current_inverse": [],
                    "version_inverse": [],
                    "edge_tag_keys": [],
                }

        host = BundleHost()
        root = tmp_path / "vault"
        root.mkdir()
        run_markdown_export_once(
            host,
            root,
            include_system=False,
            allow_existing=True,
        )
        add_markdown_mirror(kp, root, interval="PT1H")
        record_markdown_mirror_export_success(kp, root)

        alpha_path = root / "alpha.md"
        alpha_before = alpha_path.stat().st_mtime_ns

        host.docs["alpha"]["summary"] = "Alpha body updated remotely"
        stats = poll_markdown_mirrors(kp, source_keeper=host)
        assert stats["exported"] == 0

        entries = list_markdown_mirrors(kp)
        entries[0].pending_since = "2000-01-01T00:00:00"
        save_markdown_mirrors(kp, entries)

        time.sleep(0.01)
        stats = poll_markdown_mirrors(kp, source_keeper=host)
        assert stats["exported"] == 1
        assert alpha_path.stat().st_mtime_ns > alpha_before
        assert "Alpha body updated remotely" in alpha_path.read_text(encoding="utf-8")
    finally:
        kp.close()


def test_poll_markdown_mirrors_remote_change_feed_only_reexports_after_changes(
    mock_providers, tmp_path,
):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        class FeedHost:
            def __init__(self) -> None:
                self.docs = {
                    "alpha": {
                        "id": "alpha",
                        "summary": "Alpha body",
                        "tags": {},
                        "created_at": "2026-04-09T18:00:00",
                        "updated_at": "2026-04-09T18:00:00",
                        "accessed_at": "2026-04-09T18:00:00",
                    },
                }
                self.events = [{
                    "outbox_id": 1,
                    "mutation": "doc_insert",
                    "entity_id": "alpha",
                    "collection": "default",
                    "payload_json": "{}",
                    "created_at": "2026-04-09T18:00:00",
                }]

            def export_iter(self, *, include_system: bool = True):
                yield {
                    "format": "keep-export",
                    "version": 3,
                    "exported_at": "2026-04-09T18:00:00",
                    "store_info": {
                        "document_count": 1,
                        "version_count": 0,
                        "part_count": 0,
                        "collection": "default",
                    },
                }
                yield dict(self.docs["alpha"])

            def export_bundle(
                self,
                id: str,
                *,
                include_system: bool = True,
                include_parts: bool = True,
                include_versions: bool = True,
            ):
                if id != "alpha":
                    return None
                return {
                    "document": dict(self.docs["alpha"]),
                    "current_inverse": [],
                    "version_inverse": [],
                    "edge_tag_keys": [],
                }

            def export_changes(
                self,
                *,
                cursor: str | None = None,
                limit: int = 1000,
            ) -> dict:
                after = int(cursor or "0")
                visible = [row for row in self.events if row["outbox_id"] > after][:limit]
                head = self.events[-1]["outbox_id"] if self.events else after
                next_cursor = visible[-1]["outbox_id"] if visible else head
                return {
                    "format": "keep-export-changes",
                    "version": 1,
                    "cursor": str(next_cursor),
                    "head_cursor": str(head),
                    "compacted": False,
                    "truncated": False,
                    "events": visible,
                }

        host = FeedHost()
        root = tmp_path / "vault"
        root.mkdir()
        run_markdown_export_once(
            host,
            root,
            include_system=False,
            allow_existing=True,
        )
        add_markdown_mirror(kp, root, interval="PT1H")
        record_markdown_mirror_export_success(kp, root, source_cursor="1")

        alpha_path = root / "alpha.md"
        alpha_before = alpha_path.stat().st_mtime_ns

        stats = poll_markdown_mirrors(kp, source_keeper=host)
        assert stats["exported"] == 0
        assert alpha_path.stat().st_mtime_ns == alpha_before

        host.docs["alpha"]["summary"] = "Alpha body updated remotely"
        host.events.append({
            "outbox_id": 2,
            "mutation": "doc_update",
            "entity_id": "alpha",
            "collection": "default",
            "payload_json": "{}",
            "created_at": utc_now(),
            "affected_note_ids": ["alpha"],
        })

        stats = poll_markdown_mirrors(kp, source_keeper=host)
        assert stats["exported"] == 0

        entries = list_markdown_mirrors(kp)
        assert entries[0].source_cursor == "2"
        assert entries[0].pending_full_replan is False
        assert entries[0].pending_note_ids == ["alpha"]
        entries[0].pending_since = "2000-01-01T00:00:00"
        save_markdown_mirrors(kp, entries)

        time.sleep(0.01)
        stats = poll_markdown_mirrors(kp, source_keeper=host)
        assert stats["exported"] == 1
        assert alpha_path.stat().st_mtime_ns > alpha_before
        assert "Alpha body updated remotely" in alpha_path.read_text(encoding="utf-8")
    finally:
        kp.close()


def test_poll_markdown_mirrors_remote_change_feed_rewrites_inverse_target_on_source_display_change(
    mock_providers, tmp_path,
):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        class FeedHost:
            def __init__(self) -> None:
                self.docs = {
                    "Joanna": {
                        "id": "Joanna",
                        "summary": "Joanna note",
                        "tags": {},
                        "created_at": "2026-04-09T18:00:00",
                        "updated_at": "2026-04-09T18:00:00",
                        "accessed_at": "2026-04-09T18:00:00",
                    },
                    "session-1": {
                        "id": "session-1",
                        "summary": "Session body",
                        "tags": {"speaker": "Joanna"},
                        "created_at": "2026-04-09T18:00:00",
                        "updated_at": "2026-04-09T18:00:00",
                        "accessed_at": "2026-04-09T18:00:00",
                    },
                    "other": {
                        "id": "other",
                        "summary": "Unrelated body",
                        "tags": {},
                        "created_at": "2026-04-09T18:00:00",
                        "updated_at": "2026-04-09T18:00:00",
                        "accessed_at": "2026-04-09T18:00:00",
                    },
                }
                self.events = []

            def export_iter(self, *, include_system: bool = True):
                yield {
                    "format": "keep-export",
                    "version": 3,
                    "exported_at": "2026-04-09T18:00:00",
                    "store_info": {
                        "document_count": len(self.docs),
                        "version_count": 0,
                        "part_count": 0,
                        "collection": "default",
                    },
                }
                for doc_id in ("Joanna", "session-1", "other"):
                    yield dict(self.docs[doc_id])

            def export_bundle(
                self,
                id: str,
                *,
                include_system: bool = True,
                include_parts: bool = True,
                include_versions: bool = True,
            ):
                bundles = {
                    "Joanna": {
                        "document": dict(self.docs["Joanna"]),
                        "current_inverse": [["said", "session-1"]],
                        "version_inverse": [],
                        "edge_tag_keys": [],
                    },
                    "session-1": {
                        "document": dict(self.docs["session-1"]),
                        "current_inverse": [],
                        "version_inverse": [],
                        "edge_tag_keys": ["speaker"],
                    },
                    "other": {
                        "document": dict(self.docs["other"]),
                        "current_inverse": [],
                        "version_inverse": [],
                        "edge_tag_keys": [],
                    },
                }
                return bundles.get(id)

            def export_changes(
                self,
                *,
                cursor: str | None = None,
                limit: int = 1000,
            ) -> dict:
                after = int(cursor or "0")
                visible = [row for row in self.events if row["outbox_id"] > after][:limit]
                head = self.events[-1]["outbox_id"] if self.events else after
                next_cursor = visible[-1]["outbox_id"] if visible else head
                return {
                    "format": "keep-export-changes",
                    "version": 1,
                    "cursor": str(next_cursor),
                    "head_cursor": str(head),
                    "compacted": False,
                    "truncated": False,
                    "events": visible,
                }

        host = FeedHost()
        root = tmp_path / "vault"
        root.mkdir()
        run_markdown_export_once(host, root, include_system=False, allow_existing=True)
        add_markdown_mirror(kp, root, interval="PT1H")
        record_markdown_mirror_export_success(kp, root, source_cursor="0")

        source_path = root / "session-1.md"
        target_path = root / "Joanna.md"
        other_path = root / "other.md"
        source_before = source_path.stat().st_mtime_ns
        target_before = target_path.stat().st_mtime_ns
        other_before = other_path.stat().st_mtime_ns

        host.docs["session-1"]["summary"] = "Session renamed"
        host.events.append({
            "outbox_id": 1,
            "mutation": "doc_update",
            "entity_id": "session-1",
            "collection": "default",
            "payload_json": "{}",
            "created_at": utc_now(),
            "affected_note_ids": ["session-1", "Joanna"],
        })

        stats = poll_markdown_mirrors(kp, source_keeper=host)
        assert stats["exported"] == 0
        entries = list_markdown_mirrors(kp)
        assert entries[0].pending_note_ids == ["Joanna", "session-1"]
        entries[0].pending_since = "2000-01-01T00:00:00"
        save_markdown_mirrors(kp, entries)

        time.sleep(0.01)
        stats = poll_markdown_mirrors(kp, source_keeper=host)
        assert stats["exported"] == 1
        assert source_path.stat().st_mtime_ns > source_before
        assert target_path.stat().st_mtime_ns > target_before
        assert other_path.stat().st_mtime_ns == other_before
    finally:
        kp.close()


def test_poll_markdown_mirrors_remote_change_feed_rewrites_old_and_new_edge_targets(
    mock_providers, tmp_path,
):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        class FeedHost:
            def __init__(self) -> None:
                self.docs = {
                    "Joanna": {
                        "id": "Joanna",
                        "summary": "Joanna note",
                        "tags": {},
                        "created_at": "2026-04-09T18:00:00",
                        "updated_at": "2026-04-09T18:00:00",
                        "accessed_at": "2026-04-09T18:00:00",
                    },
                    "Nate": {
                        "id": "Nate",
                        "summary": "Nate note",
                        "tags": {},
                        "created_at": "2026-04-09T18:00:00",
                        "updated_at": "2026-04-09T18:00:00",
                        "accessed_at": "2026-04-09T18:00:00",
                    },
                    "session-2": {
                        "id": "session-2",
                        "summary": "Session body",
                        "tags": {"speaker": "Joanna"},
                        "created_at": "2026-04-09T18:00:00",
                        "updated_at": "2026-04-09T18:00:00",
                        "accessed_at": "2026-04-09T18:00:00",
                    },
                    "other": {
                        "id": "other",
                        "summary": "Unrelated body",
                        "tags": {},
                        "created_at": "2026-04-09T18:00:00",
                        "updated_at": "2026-04-09T18:00:00",
                        "accessed_at": "2026-04-09T18:00:00",
                    },
                }
                self.events = []

            def export_iter(self, *, include_system: bool = True):
                yield {
                    "format": "keep-export",
                    "version": 3,
                    "exported_at": "2026-04-09T18:00:00",
                    "store_info": {
                        "document_count": len(self.docs),
                        "version_count": 0,
                        "part_count": 0,
                        "collection": "default",
                    },
                }
                for doc_id in ("Joanna", "Nate", "session-2", "other"):
                    yield dict(self.docs[doc_id])

            def export_bundle(
                self,
                id: str,
                *,
                include_system: bool = True,
                include_parts: bool = True,
                include_versions: bool = True,
            ):
                bundles = {
                    "Joanna": {
                        "document": dict(self.docs["Joanna"]),
                        "current_inverse": [],
                        "version_inverse": [],
                        "edge_tag_keys": [],
                    },
                    "Nate": {
                        "document": dict(self.docs["Nate"]),
                        "current_inverse": [["said", "session-2"]],
                        "version_inverse": [],
                        "edge_tag_keys": [],
                    },
                    "session-2": {
                        "document": dict(self.docs["session-2"]),
                        "current_inverse": [],
                        "version_inverse": [],
                        "edge_tag_keys": ["speaker"],
                    },
                    "other": {
                        "document": dict(self.docs["other"]),
                        "current_inverse": [],
                        "version_inverse": [],
                        "edge_tag_keys": [],
                    },
                }
                if id == "Joanna" and self.docs["session-2"]["tags"].get("speaker") == "Joanna":
                    bundles["Joanna"]["current_inverse"] = [["said", "session-2"]]
                return bundles.get(id)

            def export_changes(
                self,
                *,
                cursor: str | None = None,
                limit: int = 1000,
            ) -> dict:
                after = int(cursor or "0")
                visible = [row for row in self.events if row["outbox_id"] > after][:limit]
                head = self.events[-1]["outbox_id"] if self.events else after
                next_cursor = visible[-1]["outbox_id"] if visible else head
                return {
                    "format": "keep-export-changes",
                    "version": 1,
                    "cursor": str(next_cursor),
                    "head_cursor": str(head),
                    "compacted": False,
                    "truncated": False,
                    "events": visible,
                }

        host = FeedHost()
        root = tmp_path / "vault"
        root.mkdir()
        run_markdown_export_once(host, root, include_system=False, allow_existing=True)
        add_markdown_mirror(kp, root, interval="PT1H")
        record_markdown_mirror_export_success(kp, root, source_cursor="0")

        source_path = root / "session-2.md"
        joanna_path = root / "Joanna.md"
        nate_path = root / "Nate.md"
        other_path = root / "other.md"
        source_before = source_path.stat().st_mtime_ns
        joanna_before = joanna_path.stat().st_mtime_ns
        nate_before = nate_path.stat().st_mtime_ns
        other_before = other_path.stat().st_mtime_ns

        host.docs["session-2"]["tags"] = {"speaker": "Nate"}
        host.events.append({
            "outbox_id": 1,
            "mutation": "edge_update",
            "entity_id": "session-2",
            "collection": "default",
            "payload_json": json.dumps({
                "target_id": "Nate",
                "old_target_id": "Joanna",
            }),
            "created_at": utc_now(),
            "affected_note_ids": ["session-2", "Joanna", "Nate"],
        })

        stats = poll_markdown_mirrors(kp, source_keeper=host)
        assert stats["exported"] == 0
        entries = list_markdown_mirrors(kp)
        assert entries[0].pending_note_ids == ["Joanna", "Nate", "session-2"]
        entries[0].pending_since = "2000-01-01T00:00:00"
        save_markdown_mirrors(kp, entries)

        time.sleep(0.01)
        stats = poll_markdown_mirrors(kp, source_keeper=host)
        assert stats["exported"] == 1
        assert source_path.stat().st_mtime_ns > source_before
        assert joanna_path.stat().st_mtime_ns > joanna_before
        assert nate_path.stat().st_mtime_ns > nate_before
        assert other_path.stat().st_mtime_ns == other_before
    finally:
        kp.close()


def test_poll_markdown_mirrors_incremental_rewrites_inverse_target_on_source_display_change(
    mock_providers, tmp_path,
):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        _create_tagdoc(kp, "speaker", "said")
        kp.put("Joanna note", id="Joanna")
        kp.put("Session body", id="session-1", tags={"speaker": "Joanna"})
        kp.put("Unrelated body", id="other")

        root = tmp_path / "vault"
        root.mkdir()
        run_markdown_export_once(
            kp,
            root,
            include_system=False,
            allow_existing=True,
        )
        clear_sync_outbox(kp)
        add_markdown_mirror(kp, root, interval="PT1H")

        source_path = root / "session-1.md"
        target_path = root / "Joanna.md"
        other_path = root / "other.md"
        source_before = source_path.stat().st_mtime_ns
        target_before = target_path.stat().st_mtime_ns
        other_before = other_path.stat().st_mtime_ns

        time.sleep(0.01)
        kp.put("Session body renamed", id="session-1", summary="Session renamed", tags={"speaker": "Joanna"})
        stats = poll_markdown_mirrors(kp)
        assert stats["exported"] == 0
        entries = list_markdown_mirrors(kp)
        entries[0].pending_since = "2000-01-01T00:00:00"
        save_markdown_mirrors(kp, entries)
        stats = poll_markdown_mirrors(kp)

        assert stats["exported"] == 1
        assert source_path.stat().st_mtime_ns > source_before
        assert target_path.stat().st_mtime_ns > target_before
        assert other_path.stat().st_mtime_ns == other_before
    finally:
        kp.close()


def test_poll_markdown_mirrors_incremental_rewrites_old_and_new_edge_targets(
    mock_providers, tmp_path,
):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        _create_tagdoc(kp, "speaker", "said")
        kp.put("Joanna note", id="Joanna")
        kp.put("Nate note", id="Nate")
        kp.put("Session body", id="session-2", tags={"speaker": "Joanna"})
        kp.put("Unrelated body", id="other")

        root = tmp_path / "vault"
        root.mkdir()
        run_markdown_export_once(
            kp,
            root,
            include_system=False,
            allow_existing=True,
        )
        clear_sync_outbox(kp)
        add_markdown_mirror(kp, root, interval="PT1H")

        source_path = root / "session-2.md"
        joanna_path = root / "Joanna.md"
        nate_path = root / "Nate.md"
        other_path = root / "other.md"
        source_before = source_path.stat().st_mtime_ns
        joanna_before = joanna_path.stat().st_mtime_ns
        nate_before = nate_path.stat().st_mtime_ns
        other_before = other_path.stat().st_mtime_ns

        time.sleep(0.01)
        kp.put("Session body", id="session-2", tags={"speaker": "Nate"})
        stats = poll_markdown_mirrors(kp)
        assert stats["exported"] == 0
        entries = list_markdown_mirrors(kp)
        entries[0].pending_since = "2000-01-01T00:00:00"
        save_markdown_mirrors(kp, entries)
        stats = poll_markdown_mirrors(kp)

        assert stats["exported"] == 1
        assert source_path.stat().st_mtime_ns > source_before
        assert joanna_path.stat().st_mtime_ns > joanna_before
        assert nate_path.stat().st_mtime_ns > nate_before
        assert other_path.stat().st_mtime_ns == other_before
    finally:
        kp.close()


def test_clear_sync_outbox_discards_pending_rows(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        kp.put("Alpha body", id="alpha")
        assert kp._document_store.sync_outbox_depth() > 0
        discarded = clear_sync_outbox(kp)
        assert discarded > 0
        assert kp._document_store.sync_outbox_depth() == 0
    finally:
        kp.close()


def test_poll_markdown_mirrors_falls_back_to_full_replan_when_map_is_missing_note(
    mock_providers, tmp_path,
):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        kp.put("Alpha body", id="alpha")
        root = tmp_path / "vault"
        root.mkdir()
        run_markdown_export_once(
            kp,
            root,
            include_system=False,
            allow_existing=True,
        )
        clear_sync_outbox(kp)
        add_markdown_mirror(kp, root, interval="PT1H")

        map_path = root / ".keep-sync" / "map.tsv"
        map_path.write_text("export_ref\tkeep_id\n", encoding="utf-8")

        kp.put("Alpha body updated", id="alpha")
        stats = poll_markdown_mirrors(kp)
        assert stats["exported"] == 0
        entries = list_markdown_mirrors(kp)
        entries[0].pending_since = "2000-01-01T00:00:00"
        save_markdown_mirrors(kp, entries)

        stats = poll_markdown_mirrors(kp)
        assert stats["exported"] == 1
        assert (root / "alpha.md").read_text(encoding="utf-8").endswith(
            "Alpha body updated\n"
        )
        map_text = map_path.read_text(encoding="utf-8")
        assert "alpha\talpha" in map_text

        refreshed = list_markdown_mirrors(kp)
        assert refreshed[0].last_error == ""
        assert refreshed[0].pending_since == ""
        assert refreshed[0].pending_note_ids == []
        assert refreshed[0].pending_full_replan is False
    finally:
        kp.close()


def test_poll_markdown_mirrors_limits_outbox_drain_per_tick(
    mock_providers, tmp_path, monkeypatch,
):
    kp = Keeper(store_path=tmp_path / "store")
    try:
        kp.put("Alpha body", id="alpha")
        root = tmp_path / "vault"
        root.mkdir()
        run_markdown_export_once(
            kp,
            root,
            include_system=False,
            allow_existing=True,
        )
        clear_sync_outbox(kp)
        add_markdown_mirror(kp, root, interval="PT1H")

        for idx in range(5):
            kp.put(f"Body {idx}", id=f"doc-{idx}")

        total_rows = kp._document_store.sync_outbox_depth()
        assert total_rows >= 5
        monkeypatch.setattr(
            "keep.markdown_mirrors._SYNC_OUTBOX_MAX_EVENTS_PER_POLL",
            3,
        )

        stats = poll_markdown_mirrors(kp)
        assert stats["exported"] == 0
        assert kp._document_store.sync_outbox_depth() == total_rows - 3
    finally:
        kp.close()


def test_next_markdown_mirror_delay_prefers_pending_due_time():
    ready = MarkdownMirrorEntry(
        root="/tmp/one",
        interval="PT30S",
        enabled=True,
        pending_since="2000-01-01T00:00:00",
    )
    idle = MarkdownMirrorEntry(
        root="/tmp/two",
        interval="PT30S",
        enabled=True,
        pending_since="",
    )
    delay = next_markdown_mirror_delay([idle, ready])
    assert delay == 0.0


def test_delete_rel_path_refuses_escape(tmp_path):
    """_delete_rel_path must not unlink targets outside the mirror root."""
    from keep.markdown_mirrors import _delete_rel_path

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("keep me", encoding="utf-8")

    _delete_rel_path(root, Path("../outside.md"))
    assert outside.exists()

    # In-tree deletes still work, including empty-dir pruning up to the root
    inside = root / "sub" / "note.md"
    inside.parent.mkdir()
    inside.write_text("x", encoding="utf-8")
    _delete_rel_path(root, Path("sub/note.md"))
    assert not inside.exists()
    assert not (root / "sub").exists()
    assert root.exists()


def test_poisoned_map_entry_cannot_delete_outside_root(mock_providers, tmp_path):
    """A tampered .keep-sync map ref must not delete files outside the mirror."""
    kp = Keeper(store_path=tmp_path / "store")
    try:
        kp.put("Alpha body", id="alpha")
        root = tmp_path / "vault"
        root.mkdir()
        run_markdown_export_once(kp, root, include_system=False, allow_existing=True)

        victim = tmp_path / "victim.md"
        victim.write_text("do not delete", encoding="utf-8")
        map_path = root / ".keep-sync" / "map.tsv"
        map_path.write_text(
            map_path.read_text(encoding="utf-8") + "../victim\tsome-id\n",
            encoding="utf-8",
        )

        # The poisoned ref is stale (not re-exported) → hits the delete path
        run_markdown_export_once(kp, root, include_system=False, allow_existing=True)
        assert victim.exists()
        assert (root / "alpha.md").is_file()
    finally:
        kp.close()


def test_delete_rel_path_unlinks_symlink_not_target(tmp_path):
    """Regression test for symlink-following deletes.

    Deleting a stale in-root symlink must remove the link itself, not
    follow it and delete the linked note.
    """
    from keep.markdown_mirrors import _delete_rel_path

    root = tmp_path / "root"
    root.mkdir()
    valuable = root / "valuable.md"
    valuable.write_text("keep me", encoding="utf-8")
    stale = root / "stale.md"
    stale.symlink_to(valuable)

    _delete_rel_path(root, Path("stale.md"))

    assert valuable.exists()
    assert not stale.exists() and not stale.is_symlink()
