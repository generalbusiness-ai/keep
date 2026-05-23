"""Tests for keep data export/import."""

import json
import pytest
import yaml
from pathlib import Path

from keep.api import Keeper
from keep.cli_app import (
    _MAX_FILENAME_BYTES,
    _id_to_rel_path,
    _md_link_target,
    _render_doc_markdown,
    _render_part_markdown,
    _render_version_markdown,
    _write_markdown_export,
)
from keep.markdown_export import _resolve_export_destination
from keep.config import StoreConfig, ProviderConfig
from keep.markdown_import import load_markdown_import
from tests.conftest import _write_test_store_config


@pytest.fixture
def keeper(tmp_path):
    """Create a real Keeper with passthrough summarization and no embedding.

    Uses real SQLite (DocumentStore) but no ML models.
    """
    config = StoreConfig(
        path=tmp_path,
        embedding=None,
        summarization=ProviderConfig("passthrough", {"max_chars": 10000}),
        max_summary_length=10000,
    )
    kp = Keeper(str(tmp_path), config=config)
    yield kp
    kp.close()


@pytest.fixture
def fresh_keeper(tmp_path):
    """Create a second Keeper for import testing."""
    fresh_path = tmp_path / "fresh"
    fresh_path.mkdir()
    config = StoreConfig(
        path=fresh_path,
        embedding=None,
        summarization=ProviderConfig("passthrough", {"max_chars": 10000}),
        max_summary_length=10000,
    )
    kp = Keeper(str(fresh_path), config=config)
    yield kp
    kp.close()


def _seed(keeper, docs):
    """Seed documents into keeper via import_batch (bypasses embedding)."""
    ds = keeper._document_store
    coll = keeper._resolve_doc_collection()
    ds.import_batch(coll, docs)


def _make_doc(id, summary, tags=None, versions=None, parts=None,
              created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
              accessed_at="2026-01-01T00:00:00"):
    """Build a document dict for seeding or import."""
    doc = {
        "id": id,
        "summary": summary,
        "tags": tags or {},
        "created_at": created_at,
        "updated_at": updated_at,
        "accessed_at": accessed_at,
    }
    if versions:
        doc["versions"] = versions
    if parts:
        doc["parts"] = parts
    return doc


class TestExportImport:
    """Round-trip export/import tests."""

    def test_export_empty_store(self, keeper):
        """Export from empty store produces valid structure."""
        data = keeper.export_data()
        assert data["format"] == "keep-export"
        assert data["version"] == 3
        assert data["exported_at"]
        assert data["store_info"]["document_count"] >= 0
        assert isinstance(data["documents"], list)

    def test_export_with_documents(self, keeper):
        """Export captures documents with tags and timestamps."""
        _seed(keeper, [
            _make_doc("rust-doc", "Test document about Rust", tags={"topic": "rust"}),
            _make_doc("python-doc", "Another doc about Python", tags={"topic": "python"}),
        ])

        data = keeper.export_data()
        docs_by_id = {d["id"]: d for d in data["documents"]}

        assert "rust-doc" in docs_by_id
        assert "python-doc" in docs_by_id

        doc = docs_by_id["python-doc"]
        assert doc["summary"]
        assert doc["tags"]["topic"] == "python"
        assert doc["created_at"]
        assert doc["updated_at"]
        assert doc["accessed_at"]

    def test_export_with_versions(self, keeper):
        """Export captures version history."""
        _seed(keeper, [
            _make_doc("versioned", "Version 2", versions=[{
                "version": 1,
                "summary": "Version 1",
                "tags": {},
                "content_hash": None,
                "created_at": "2025-12-01T00:00:00",
            }]),
        ])

        data = keeper.export_data()
        doc = next(d for d in data["documents"] if d["id"] == "versioned")

        # Current doc has latest
        assert "Version 2" in doc["summary"]
        # Version history exists
        assert "versions" in doc
        assert len(doc["versions"]) >= 1
        assert data["store_info"]["version_count"] >= 1

    def test_export_include_system_flag(self, keeper):
        """include_system=False (the CLI default) skips dot-prefix IDs."""
        _seed(keeper, [
            _make_doc("user-doc", "User doc"),
            _make_doc(".system-doc", "System doc"),
        ])

        data_all = keeper.export_data(include_system=True)
        data_no_sys = keeper.export_data(include_system=False)

        all_ids = {d["id"] for d in data_all["documents"]}
        no_sys_ids = {d["id"] for d in data_no_sys["documents"]}

        assert ".system-doc" in all_ids
        assert "user-doc" in no_sys_ids
        # No dot-prefix IDs in filtered export
        assert all(not id.startswith(".") for id in no_sys_ids)
        # Filtered should have fewer docs
        assert len(data_no_sys["documents"]) < len(data_all["documents"])

    def test_round_trip(self, keeper, fresh_keeper):
        """Export then import into fresh store preserves data."""
        _seed(keeper, [
            _make_doc("auth-learning", "Important learning about auth", tags={"topic": "auth"}),
            _make_doc("notes", "Version 2 of notes", versions=[{
                "version": 1,
                "summary": "Version 1 of notes",
                "tags": {},
                "content_hash": None,
                "created_at": "2025-12-01T00:00:00",
            }]),
        ])

        data = keeper.export_data()
        stats = fresh_keeper.import_data(data, mode="merge")

        assert stats["imported"] > 0
        assert stats["skipped"] == 0

        # Verify imported data
        item = fresh_keeper.get("auth-learning")
        assert item is not None
        assert item.tags.get("topic") == "auth"

        # Verify versions imported
        versions = fresh_keeper.list_versions("notes")
        assert len(versions) >= 1

    def test_import_markdown_honors_id_and_source_uri(self, keeper, tmp_path):
        source = tmp_path / "cwe.md"
        source.write_text(
            """---
_id: CWE-9999
title: Example Weakness
_source_uri: https://cwe.mitre.org/data/definitions/9999.html
status: Draft
taxonomy_mappings:
  taxonomy_name: Foo
tags:
  topic: security
  _content_type: text/markdown
---
Example markdown body.
""",
            encoding="utf-8",
        )

        stats = keeper.import_markdown(source)

        item = keeper.get("CWE-9999")
        assert stats["imported"] == 1
        assert item is not None
        assert item.summary == "Example markdown body.\n"
        assert item.tags["title"] == "Example Weakness"
        assert item.tags["_source_uri"] == "https://cwe.mitre.org/data/definitions/9999.html"
        assert item.tags["topic"] == "security"
        assert item.tags["_content_type"] == "text/markdown"
        assert "taxonomy_mappings" not in item.tags

    def test_import_markdown_rewrites_export_local_edge_refs(self, keeper, tmp_path):
        paper_dir = tmp_path / "https" / "example.com"
        paper_dir.mkdir(parents=True)
        (paper_dir / "paper.md").write_text(
            """---
_id: https://example.com/paper
title: Example Paper
---
Reference body.
""",
            encoding="utf-8",
        )
        (tmp_path / "note.md").write_text(
            """---
_id: source-note
title: Source Note
references:
  - "[[https/example.com/paper|Example Paper]]"
---
Source body.
""",
            encoding="utf-8",
        )

        keeper.import_markdown(tmp_path)

        item = keeper.get("source-note")
        assert item is not None
        assert item.tags["references"] == "[[https://example.com/paper|Example Paper]]"

    def test_import_markdown_restores_versions_and_parts(self, keeper, tmp_path):
        (tmp_path / "doc.md").write_text(
            """---
_id: doc
_created: 2026-01-01T00:00:00
_updated: 2026-01-02T00:00:00
---
Current body.
""",
            encoding="utf-8",
        )
        sidecar_dir = tmp_path / "doc"
        sidecar_dir.mkdir()
        (sidecar_dir / "@P{1}.md").write_text(
            """---
_id: doc
_part_num: 1
_created: 2026-01-03T00:00:00
---
Part body.
""",
            encoding="utf-8",
        )
        (sidecar_dir / "@V{1}.md").write_text(
            """---
_id: doc
_version: 7
_version_offset: 1
_created: 2025-12-31T00:00:00
---
Older body.
""",
            encoding="utf-8",
        )

        stats = keeper.import_markdown(tmp_path)

        assert stats["imported"] == 1
        versions = keeper.list_versions("doc")
        assert len(versions) == 1
        assert versions[0].summary == "Older body.\n"
        part = keeper._document_store.get_part(
            keeper._resolve_doc_collection(),
            "doc",
            1,
        )
        assert part is not None
        assert part.summary == "Part body.\n"

    def test_load_markdown_import_reports_progress(self, tmp_path):
        (tmp_path / "one.md").write_text("---\n_id: one\n---\nOne.\n", encoding="utf-8")
        subdir = tmp_path / "nested"
        subdir.mkdir()
        (subdir / "two.md").write_text("---\n_id: two\n---\nTwo.\n", encoding="utf-8")

        events: list[tuple[int, int, str]] = []
        documents, _ref_map = load_markdown_import(
            tmp_path,
            progress=lambda cur, total, label: events.append((cur, total, label)),
        )

        assert [doc["id"] for doc in documents] == ["one", "two"]
        assert events == [
            (1, 2, "nested/two.md"),
            (2, 2, "one.md"),
        ]

    def test_merge_skips_existing(self, keeper):
        """Merge mode skips documents with existing IDs."""
        _seed(keeper, [_make_doc("existing-doc", "Original content")])

        data = {
            "format": "keep-export",
            "version": 1,
            "exported_at": "2026-01-01T00:00:00",
            "store_info": {"document_count": 1, "version_count": 0,
                          "part_count": 0, "collection": "default"},
            "documents": [{
                "id": "existing-doc",
                "summary": "Different content",
                "tags": {},
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "accessed_at": "2026-01-01T00:00:00",
            }],
        }

        stats = keeper.import_data(data, mode="merge")
        assert stats["imported"] == 0
        assert stats["skipped"] == 1

        # Original content preserved
        item = keeper.get("existing-doc")
        assert "Original" in item.summary

    def test_replace_clears_store(self, keeper):
        """Replace mode clears existing data before import."""
        _seed(keeper, [_make_doc("old-doc", "Old data")])

        data = {
            "format": "keep-export",
            "version": 1,
            "exported_at": "2026-01-01T00:00:00",
            "store_info": {"document_count": 1, "version_count": 0,
                          "part_count": 0, "collection": "default"},
            "documents": [{
                "id": "new-doc",
                "summary": "New data",
                "tags": {"type": "imported"},
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "accessed_at": "2026-01-01T00:00:00",
            }],
        }

        stats = keeper.import_data(data, mode="replace")
        assert stats["imported"] == 1
        assert stats["skipped"] == 0

        # Old doc gone
        assert keeper.get("old-doc") is None
        # New doc present
        item = keeper.get("new-doc")
        assert item is not None
        assert "New data" in item.summary

    def test_import_invalid_format(self, keeper):
        """Import rejects invalid format."""
        with pytest.raises(ValueError, match="Invalid export format"):
            keeper.import_data({"format": "wrong"})

    def test_import_future_version(self, keeper):
        """Import rejects future format versions."""
        with pytest.raises(ValueError, match="not supported"):
            keeper.import_data({"format": "keep-export", "version": 99})

    def test_import_rejects_invalid_tag_key(self, keeper):
        """Import rejects tag keys that violate tag-key validation."""
        data = {
            "format": "keep-export",
            "version": 1,
            "exported_at": "2026-01-01T00:00:00",
            "store_info": {"document_count": 1, "version_count": 0,
                          "part_count": 0, "collection": "default"},
            "documents": [{
                "id": "bad-tags",
                "summary": "Doc with invalid tag key",
                "tags": {"bad!key": "x"},
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "accessed_at": "2026-01-01T00:00:00",
            }],
        }

        with pytest.raises(ValueError, match="Import document tags"):
            keeper.import_data(data, mode="merge")

    def test_timestamp_preservation(self, keeper):
        """Import preserves original timestamps."""
        data = {
            "format": "keep-export",
            "version": 1,
            "exported_at": "2026-01-01T00:00:00",
            "store_info": {"document_count": 1, "version_count": 0,
                          "part_count": 0, "collection": "default"},
            "documents": [{
                "id": "timestamped",
                "summary": "Doc with specific timestamps",
                "tags": {},
                "created_at": "2025-06-15T10:30:00",
                "updated_at": "2025-12-01T14:00:00",
                "accessed_at": "2026-01-15T09:00:00",
            }],
        }

        stats = keeper.import_data(data, mode="merge")
        assert stats["imported"] == 1

        doc_coll = keeper._resolve_doc_collection()
        record = keeper._document_store.get(doc_coll, "timestamped")
        assert record.created_at == "2025-06-15T10:30:00"
        assert record.updated_at == "2025-12-01T14:00:00"
        assert record.accessed_at == "2026-01-15T09:00:00"

    def test_import_queues_reindex(self, keeper):
        """Imported documents are queued for re-embedding."""
        data = {
            "format": "keep-export",
            "version": 1,
            "exported_at": "2026-01-01T00:00:00",
            "store_info": {"document_count": 2, "version_count": 0,
                          "part_count": 0, "collection": "default"},
            "documents": [
                {
                    "id": "doc1", "summary": "First", "tags": {},
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                },
                {
                    "id": "doc2", "summary": "Second", "tags": {},
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                },
            ],
        }

        stats = keeper.import_data(data, mode="merge")
        assert stats["queued"] == 2

    def test_import_v1_part_content_normalizes_into_summary(self, keeper):
        """Legacy v1 part content is promoted into the canonical part summary."""
        data = {
            "format": "keep-export",
            "version": 1,
            "exported_at": "2026-01-01T00:00:00",
            "store_info": {"document_count": 1, "version_count": 0,
                          "part_count": 1, "collection": "default"},
            "documents": [{
                "id": "legacy-part-doc",
                "summary": "Parent summary",
                "tags": {},
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "accessed_at": "2026-01-01T00:00:00",
                "parts": [{
                    "part_num": 1,
                    "summary": "Old short prefix",
                    "content": "Full legacy part text",
                    "tags": {"topic": "legacy"},
                    "created_at": "2026-01-01T00:00:00",
                }],
            }],
        }

        stats = keeper.import_data(data, mode="merge")
        assert stats["queued"] == 2

        part = keeper._document_store.get_part(
            keeper._resolve_doc_collection(),
            "legacy-part-doc",
            1,
        )
        assert part is not None
        assert part.summary == "Full legacy part text"

    def test_import_legacy_labeled_refs_reexports_canonical_form(self, keeper):
        """Old export files with legacy labeled refs import into canonical form."""
        data = {
            "format": "keep-export",
            "version": 1,
            "exported_at": "2026-01-01T00:00:00",
            "store_info": {"document_count": 2, "version_count": 1,
                          "part_count": 1, "collection": "default"},
            "documents": [
                {
                    "id": ".tag/speaker",
                    "summary": "speaker tagdoc",
                    "tags": {"_inverse": "said"},
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                    "accessed_at": "2026-01-01T00:00:00",
                },
                {
                    "id": "conv1",
                    "summary": "Conversation",
                    "tags": {"speaker": "contact:telegram:42[[Alice]]"},
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                    "accessed_at": "2026-01-01T00:00:00",
                    "versions": [{
                        "version": 1,
                        "summary": "Older conversation",
                        "tags": {"speaker": "contact:telegram:42[[Alice]]"},
                        "content_hash": None,
                        "created_at": "2025-12-01T00:00:00",
                    }],
                    "parts": [{
                        "part_num": 1,
                        "summary": "Part summary",
                        "tags": {"speaker": "contact:telegram:42[[Alice]]"},
                        "created_at": "2026-01-01T00:00:00",
                    }],
                },
            ],
        }

        keeper.import_data(data, mode="merge")
        exported = keeper.export_data(include_system=True)
        docs = {d["id"]: d for d in exported["documents"]}
        conv = docs["conv1"]

        assert exported["version"] == 3
        assert conv["tags"]["speaker"] == "[[contact:telegram:42|Alice]]"
        assert conv["versions"][0]["tags"]["speaker"] == "[[contact:telegram:42|Alice]]"
        assert conv["parts"][0]["tags"]["speaker"] == "[[contact:telegram:42|Alice]]"

    def test_export_parts_omit_legacy_content_field(self, keeper):
        """Version 3 exports write summary-only part records."""
        _seed(keeper, [_make_doc(
            "summary-only",
            "Parent summary",
            parts=[{
                "part_num": 1,
                "summary": "Canonical part text",
                "tags": {"topic": "x"},
                "created_at": "2026-01-02T00:00:00",
            }],
        )])

        data = keeper.export_data()
        doc = next(d for d in data["documents"] if d["id"] == "summary-only")
        assert data["version"] == 3
        assert doc["parts"] == [{
            "part_num": 1,
            "summary": "Canonical part text",
            "tags": {"topic": "x"},
            "created_at": "2026-01-02T00:00:00",
        }]


class TestDocumentStoreImport:
    """Direct DocumentStore import method tests."""

    def test_import_batch_basic(self, keeper):
        """import_batch inserts documents correctly."""
        ds = keeper._document_store
        coll = keeper._resolve_doc_collection()

        docs = [{
            "id": "batch-1",
            "summary": "First doc",
            "tags": {"topic": "test"},
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-02T00:00:00",
            "accessed_at": "2026-01-03T00:00:00",
            "versions": [{
                "version": 1,
                "summary": "Old version",
                "tags": {},
                "content_hash": None,
                "created_at": "2025-12-01T00:00:00",
            }],
            "parts": [{
                "part_num": 1,
                "summary": "Part one",
                "tags": {"section": "intro"},
                "created_at": "2026-01-02T00:00:00",
            }],
        }]

        stats = ds.import_batch(coll, docs)
        assert stats == {"documents": 1, "versions": 1, "parts": 1}

        record = ds.get(coll, "batch-1")
        assert record is not None
        assert record.summary == "First doc"
        assert record.tags.get("topic") == "test"
        assert record.created_at == "2026-01-01T00:00:00"
        assert record.updated_at == "2026-01-02T00:00:00"

        versions = ds.list_versions(coll, "batch-1")
        assert len(versions) == 1
        assert versions[0].summary == "Old version"

        parts = ds.list_parts(coll, "batch-1")
        assert len(parts) == 1
        assert parts[0].summary == "Part one"

    def test_delete_collection_all(self, keeper):
        """delete_collection_all clears documents, versions, and parts."""
        ds = keeper._document_store
        coll = keeper._resolve_doc_collection()

        docs = [{
            "id": "to-delete",
            "summary": "Will be deleted",
            "tags": {},
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "versions": [{
                "version": 1, "summary": "v1", "tags": {},
                "content_hash": None, "created_at": "2025-01-01T00:00:00",
            }],
            "parts": [{
                "part_num": 1, "summary": "p1", "tags": {},
                "created_at": "2026-01-01T00:00:00",
            }],
        }]
        ds.import_batch(coll, docs)
        assert ds.get(coll, "to-delete") is not None

        count = ds.delete_collection_all(coll)
        assert count >= 1
        assert ds.get(coll, "to-delete") is None
        assert ds.list_versions(coll, "to-delete") == []
        assert ds.list_parts(coll, "to-delete") == []


def _parse_markdown(path: Path) -> tuple[dict, str]:
    """Parse an exported markdown file into (frontmatter_dict, body)."""
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"missing frontmatter: {text[:40]!r}"
    _, fm_text, body = text.split("---", 2)
    return yaml.safe_load(fm_text), body.lstrip("\n")


class TestMarkdownExport:
    """Tests for the markdown export mode (directory, one .md per note)."""

    def test_id_to_rel_path_simple(self):
        assert _id_to_rel_path("auth-notes") == Path("auth-notes.md")

    def test_id_to_rel_path_file_uri_becomes_dir_tree(self):
        # file:///Users/x/README.md mirrors the directory structure
        # with the URI scheme as a top-level directory (wget -m style).
        rel = _id_to_rel_path("file:///Users/x/README.md")
        assert rel == Path("file") / "Users" / "x" / "README.md.md"

    def test_id_to_rel_path_system_ids_nest(self):
        # Hierarchical ids split on `/` and become nested directories.
        assert _id_to_rel_path(".tag/act/commitment") == Path(".tag/act/commitment.md")

    def test_id_to_rel_path_hierarchical_without_scheme(self):
        assert _id_to_rel_path("notes/2024/jan") == Path("notes/2024/jan.md")

    def test_id_to_rel_path_content_addressed_inline_groups_under_inline(self):
        rel = _id_to_rel_path("%9f86d081884c")
        assert rel == Path("_inline") / "9f86d081884c.md"

    def test_id_to_rel_path_bare_email_groups_under_email_domain(self):
        rel = _id_to_rel_path("alice@example.com")
        assert rel == Path("_email") / "example.com" / "alice@example.com.md"

    def test_id_to_rel_path_encodes_unsafe_chars_within_component(self):
        # ':' and '#' inside a single path component are percent-encoded.
        rel = _id_to_rel_path("file:///a/b:c#d/e")
        assert rel == Path("file") / "a" / "b%3Ac%23d" / "e.md"

    def test_id_to_rel_path_truncates_overlong_thread_id(self):
        # Real-world case: long thread ids with embedded base64 blobs
        # exceed the filesystem's per-component byte limit.  With the
        # bare-scheme rule, `thread:` peels off as a directory and the
        # long body is one component that gets truncated + hashed.
        long_id = (
            "thread:3d0b982edfb41fea27a4e3f4b86fe0598c4947f0@hey.com"
            "#" + "BAh7CEkiCGdpZAY6BkVUSSI9Z2lkOi8vaGF5c3RhY2svQWN0aXZlU3RvcmFnZTo6"
            "QmxvYi8xMDc3MTA1MzExP2V4cGlyZXNfaW4GOwBUSSIMcHVycG9zZQY7A"
            "FRJIg9hdHRhY2hhYmxlBjsAVEkiD2V4cGlyZXNfYXQGOwBUMA%3D%3D"
            "--5d3054b37eb866d54f494788389df9126e377521@hey.com"
        )
        rel = _id_to_rel_path(long_id)
        # Two parts: 'thread' dir + the (truncated) body file.
        assert rel.parts[0] == "thread"
        assert len(rel.parts) == 2
        body = rel.parts[1]
        # Each path component must fit the filesystem byte budget.
        assert len(body.encode("utf-8")) <= _MAX_FILENAME_BYTES
        assert body.endswith(".md")
        # The body prefix should still be human-recognisable.
        assert body.startswith(
            "3d0b982edfb41fea27a4e3f4b86fe0598c4947f0@hey.com"
        )

    def test_id_to_rel_path_long_component_is_unique(self):
        # Two distinct long components must produce distinct filenames
        # even when both are truncated.
        base = "thread:" + "x" * 300
        a = _id_to_rel_path(base + "alpha")
        b = _id_to_rel_path(base + "beta")
        assert a != b
        for rel in (a, b):
            for part in rel.parts:
                assert len(part.encode("utf-8")) <= _MAX_FILENAME_BYTES

    def test_id_to_rel_path_no_dangling_percent_escape(self):
        # Build an id that is certain to truncate exactly where a
        # percent-escape would otherwise be sliced in half.  'x' * N
        # followed by a non-ASCII char forces a `%XX` at a controlled
        # offset.
        for pad in range(195, 202):
            doc_id = "x" * pad + "é"  # é encodes to %C3%A9
            rel = _id_to_rel_path(doc_id)
            # No bare trailing '%' in the truncated prefix — the
            # digest suffix always starts with a literal '.'.
            name = rel.parts[-1]
            body = name.removesuffix(".md")
            # Strip the `.{digest}` tail (12 hex chars) to get the prefix.
            prefix = body.rsplit(".", 1)[0]
            assert not prefix.endswith("%")
            assert not (len(prefix) >= 2 and prefix[-2] == "%")
            assert len(name.encode("utf-8")) <= _MAX_FILENAME_BYTES

    def test_id_to_rel_path_http_scheme(self):
        rel = _id_to_rel_path("https://example.com/docs/guide")
        assert rel == Path("https") / "example.com" / "docs" / "guide.md"

    def test_id_to_rel_path_thread_urn_style(self):
        # URN-style scheme (no '//' after the colon) — `thread:` becomes
        # a top-level directory and the body is a single component.
        # `@` is filesystem-safe on every modern OS so it stays literal;
        # `#` is encoded.
        rel = _id_to_rel_path(
            "thread:B1B58B3B-1943-4228-8DC6-4CDDEA5A6B38@northeastern.edu"
            "#185b1a36f1367a087003"
        )
        assert rel == Path("thread") / (
            "B1B58B3B-1943-4228-8DC6-4CDDEA5A6B38@northeastern.edu"
            "%23185b1a36f1367a087003.md"
        )

    def test_id_to_rel_path_mailto(self):
        rel = _id_to_rel_path("mailto:foo@bar.com")
        assert rel == Path("mailto") / "foo@bar.com.md"

    def test_id_to_rel_path_tel(self):
        # '+' is in the safe set so it stays literal.
        rel = _id_to_rel_path("tel:+15551234")
        assert rel == Path("tel") / "+15551234.md"

    def test_id_to_rel_path_bare_scheme_with_slash_body(self):
        # Option B: when a URN-style id's body contains '/', it splits
        # on '/' just like an authority-style URI would.  The fragment
        # marker '#' is not a separator, so '#a' stays glued to the
        # first component (with '#' percent-encoded).
        rel = _id_to_rel_path("thread:UUID@host#a/b/c")
        assert rel == Path("thread") / "UUID@host%23a" / "b" / "c.md"

    def test_id_to_rel_path_simple_colon_id(self):
        # Even ids that "look like tags" get reshaped — but the result
        # is still a sensible browse layout.
        assert _id_to_rel_path("notes:january") == Path("notes") / "january.md"
        assert _id_to_rel_path("topic:rust") == Path("topic") / "rust.md"

    def test_id_to_rel_path_does_not_match_non_uri(self):
        # Leading-dot ids and digit-prefixed ids must not be parsed as
        # schemes (RFC 3986 requires the scheme to start with ALPHA).
        # `.tag/act/commitment` falls through to the '/'-split branch.
        assert _id_to_rel_path(".tag/act/commitment") == Path(".tag/act/commitment.md")
        # A digit-prefixed pseudo-scheme stays flat (encoded).
        assert _id_to_rel_path("2024:01:summary") == Path("2024%3A01%3Asummary.md")
        # Non-content-addressed `%...` ids should remain flat too.
        assert _id_to_rel_path("%not-inline") == Path("%25not-inline.md")

    def test_render_doc_markdown_frontmatter(self):
        doc = {
            "id": "auth-notes",
            "summary": "Body content here.",
            "tags": {"topic": "auth", "_source": "inline"},
            "content_hash": "abc123",
            "content_hash_full": "def456",
            "created_at": "2026-01-15T10:30:00",
            "updated_at": "2026-02-01T14:22:00",
            "accessed_at": "2026-02-19T09:00:00",
        }
        text = _render_doc_markdown(doc)
        assert text.startswith("---\n")
        assert text.endswith("\n")
        # body comes after the closing fence
        parts = text.split("---", 2)
        assert len(parts) == 3
        meta = yaml.safe_load(parts[1])
        # Frontmatter is flat: _id first, then reserved export metadata,
        # then every stored tag (user + system) promoted to its own
        # top-level key.  No nested ``tags:`` block, and no duplicated
        # created_at/updated_at/accessed_at columns.
        assert meta["_id"] == "auth-notes"
        assert meta["_content_hash"] == "abc123"
        assert meta["_content_hash_full"] == "def456"
        assert meta["topic"] == "auth"
        assert meta["_source"] == "inline"
        assert "tags" not in meta
        assert "created_at" not in meta
        assert "updated_at" not in meta
        assert "accessed_at" not in meta
        assert parts[2].lstrip("\n").rstrip("\n") == "Body content here."

    def test_render_doc_markdown_column_timestamp_fallback(self):
        # The stored tag map is empty but the schema columns carry the
        # timestamps — that's the common case in practice because
        # ``documents.tags_json`` usually doesn't duplicate
        # ``_created``/``_updated``/``_accessed`` (they're synthesised
        # at display time by ``keep get``).  The exporter must fall
        # back to the columns so the exported frontmatter isn't missing
        # timestamps for the majority of docs.
        doc = {
            "id": "no-tags",
            "summary": "body",
            "tags": {},
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-02-01T00:00:00",
            "accessed_at": "2026-03-01T00:00:00",
        }
        text = _render_doc_markdown(doc)
        meta = yaml.safe_load(text.split("---", 2)[1])
        assert meta["_id"] == "no-tags"
        assert meta["_created"] == "2026-01-01T00:00:00"
        assert meta["_updated"] == "2026-02-01T00:00:00"
        assert meta["_accessed"] == "2026-03-01T00:00:00"
        # No nested ``tags:`` block, no legacy column spellings.
        assert "tags" not in meta
        assert "created_at" not in meta
        assert "updated_at" not in meta
        assert "accessed_at" not in meta

    def test_render_doc_markdown_stored_timestamp_tags_win(self):
        # When the stored tag map already carries ``_created`` /
        # ``_updated`` / ``_accessed``, those values win over the
        # schema columns.  Guards the "tags are authoritative when
        # present" half of the fallback.
        doc = {
            "id": "stored-ts",
            "summary": "body",
            "tags": {
                "_created": "2020-01-01T00:00:00",
                "_updated": "2020-02-01T00:00:00",
            },
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-02-01T00:00:00",
        }
        text = _render_doc_markdown(doc)
        meta = yaml.safe_load(text.split("---", 2)[1])
        # Stored tag values win — columns are only a fallback.
        assert meta["_created"] == "2020-01-01T00:00:00"
        assert meta["_updated"] == "2020-02-01T00:00:00"

    def test_render_doc_markdown_omits_versions_and_parts(self):
        doc = _make_doc(
            "mixed", "Summary body",
            tags={"topic": "x"},
            versions=[{
                "version": 1, "summary": "old", "tags": {},
                "content_hash": None, "created_at": "2025-12-01T00:00:00",
            }],
            parts=[{
                "part_num": 1, "summary": "p1", "tags": {},
                "created_at": "2026-01-01T00:00:00",
            }],
        )
        text = _render_doc_markdown(doc)
        # Neither versions nor parts should appear anywhere in the markdown.
        assert "versions" not in text
        assert "parts" not in text
        meta = yaml.safe_load(text.split("---", 2)[1])
        assert "versions" not in meta
        assert "parts" not in meta

    def test_write_markdown_export_basic(self, keeper, tmp_path):
        _seed(keeper, [
            _make_doc("rust-doc", "About Rust", tags={"topic": "rust"}),
            _make_doc("python-doc", "About Python", tags={"topic": "python"}),
        ])
        out = tmp_path / "md-export"
        out.mkdir()

        count, info = _write_markdown_export(keeper, out, include_system=True)
        assert count >= 2

        # One file per user doc (plus any system docs that may exist)
        files = {p.name: p for p in out.iterdir() if p.is_file()}
        assert "rust-doc.md" in files
        assert "python-doc.md" in files

        meta, body = _parse_markdown(files["python-doc.md"])
        assert meta["_id"] == "python-doc"
        # Tags are flat top-level keys, not nested under ``tags:``.
        assert meta["topic"] == "python"
        assert "About Python" in body

    def test_write_markdown_export_rejects_escape_path(self, tmp_path):
        out = tmp_path / "md-export"
        out.mkdir()

        with pytest.raises(ValueError, match="escapes output directory"):
            _resolve_export_destination(out, Path("..") / "outside.md")

    def test_write_markdown_export_include_system_false(self, keeper, tmp_path):
        _seed(keeper, [
            _make_doc("user-doc", "User doc", tags={"topic": "user"}),
            _make_doc(".system-doc", "System doc"),
        ])
        out = tmp_path / "md-user-only"
        out.mkdir()

        count, _info = _write_markdown_export(keeper, out, include_system=False)
        names = {p.name for p in out.rglob("*.md")}
        assert "user-doc.md" in names
        # System docs (dot-prefix ids) must not appear when include_system=False.
        assert ".system-doc.md" not in names
        assert count >= 1

    def test_write_markdown_export_include_system_true(self, keeper, tmp_path):
        _seed(keeper, [
            _make_doc("user-doc", "User doc"),
            _make_doc(".system-doc", "System doc"),
        ])
        out = tmp_path / "md-with-system"
        out.mkdir()

        _write_markdown_export(keeper, out, include_system=True)
        names = {p.name for p in out.rglob("*.md")}
        assert "user-doc.md" in names
        assert ".system-doc.md" in names

    def test_write_markdown_export_path_like_id(self, keeper, tmp_path):
        file_id = "file:///tmp/readme.md"
        _seed(keeper, [_make_doc(file_id, "# Readme\n\nHello.", tags={"topic": "x"})])
        out = tmp_path / "md-file"
        out.mkdir()

        _write_markdown_export(keeper, out, include_system=False)
        # Expect the note to land at file/tmp/readme.md.md (wget -m style)
        expected = out / "file" / "tmp" / "readme.md.md"
        assert expected.is_file(), f"expected {expected}, got {list(out.rglob('*'))}"

        meta, body = _parse_markdown(expected)
        assert meta["_id"] == file_id
        assert "Hello." in body

    def test_write_markdown_export_mixed_hierarchy(self, keeper, tmp_path):
        """Mixed flat, hierarchical, and URI-scheme ids coexist cleanly."""
        _seed(keeper, [
            _make_doc("auth-notes", "body A"),
            _make_doc("notes/2024/jan", "body B"),
            _make_doc("file:///Users/x/readme.md", "body C"),
        ])
        out = tmp_path / "md-mixed"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)

        assert (out / "auth-notes.md").is_file()
        assert (out / "notes" / "2024" / "jan.md").is_file()
        assert (out / "file" / "Users" / "x" / "readme.md.md").is_file()

    # ------------------------------------------------------------------
    # Parts and versions sidecars
    # ------------------------------------------------------------------

    def test_render_part_markdown_shape(self):
        text = _render_part_markdown("auth-notes", {
            "part_num": 3,
            "summary": "The analysis text for this section.",
            "tags": {"section": "intro"},
            "created_at": "2026-01-15T10:30:00",
        })
        meta = yaml.safe_load(text.split("---", 2)[1])
        body = text.split("---", 2)[2].lstrip("\n").rstrip("\n")
        # Part frontmatter is flat: _id (parent), _part_num, then any
        # part tags promoted to top-level.  ``_created`` is synthesized
        # from the column when there's no stored ``_created`` tag
        # (which parts don't carry).
        assert meta["_id"] == "auth-notes"
        assert meta["_part_num"] == 3
        assert meta["section"] == "intro"
        assert meta["_created"] == "2026-01-15T10:30:00"
        assert "tags" not in meta
        assert body == "The analysis text for this section."
        assert "summary" not in meta

    def test_render_part_markdown_uses_summary_body(self):
        text = _render_part_markdown("notes", {
            "part_num": 1,
            "summary": "Just the analysis text.",
            "tags": {},
            "created_at": "2026-02-01T12:00:00",
        })
        body = text.split("---", 2)[2].lstrip("\n").rstrip("\n")
        assert body == "Just the analysis text."

    def test_part_frontmatter_wins_over_stored_part_num_tag(self):
        text = _render_part_markdown("p", {
            "part_num": 3,
            "summary": "x",
            "tags": {"_part_num": "3", "_base_id": "p", "section": "intro"},
            "created_at": "2026-01-15T10:30:00",
        })
        meta = yaml.safe_load(text.split("---", 2)[1])
        assert meta["_id"] == "p"
        assert meta["_part_num"] == 3
        assert isinstance(meta["_part_num"], int)
        assert meta["section"] == "intro"
        assert "_base_id" in meta

    def test_part_stored_created_tag_flows_through(self):
        # ``_created`` is not in ``_EXPORT_RESERVED_KEYS`` — if a part
        # happens to carry it in its stored tag map, the value should
        # flow through to the frontmatter instead of being dropped.
        # The column fallback only kicks in when the tag is absent.
        text = _render_part_markdown("p", {
            "part_num": 1,
            "summary": "x",
            "tags": {"_created": "2020-06-01T00:00:00", "section": "intro"},
            "created_at": "2026-01-15T10:30:00",
        })
        meta = yaml.safe_load(text.split("---", 2)[1])
        # Stored tag wins over the column.
        assert meta["_created"] == "2020-06-01T00:00:00"
        assert meta["section"] == "intro"

    def test_render_version_markdown_shape(self):
        text = _render_version_markdown(
            "auth-notes",
            {
                "version": 7,
                "summary": "Older summary text",
                "tags": {"topic": "auth"},
                "content_hash": "abc123",
                "created_at": "2025-12-01T09:00:00",
            },
            offset=2,
        )
        meta = yaml.safe_load(text.split("---", 2)[1])
        body = text.split("---", 2)[2].lstrip("\n").rstrip("\n")
        # The @V{N} offset (2 = two steps back from current) is stored
        # alongside the absolute database version number for reference.
        # Tags are promoted flat onto the version frontmatter, no
        # nested ``tags:`` block.
        assert meta["_id"] == "auth-notes"
        assert meta["_version_offset"] == 2
        assert meta["_version"] == 7
        assert meta["_content_hash"] == "abc123"
        assert meta["_created"] == "2025-12-01T09:00:00"
        assert meta["topic"] == "auth"
        assert "tags" not in meta
        assert body == "Older summary text"

    def test_version_frontmatter_wins_over_stored_version_tag(self):
        text = _render_version_markdown(
            "auth-notes",
            {
                "version": 7,
                "summary": "Older summary text",
                "tags": {"_version": "7", "_base_id": "auth-notes", "topic": "auth"},
                "content_hash": "abc123",
                "created_at": "2025-12-01T09:00:00",
            },
            offset=2,
        )
        meta = yaml.safe_load(text.split("---", 2)[1])
        assert meta["_version_offset"] == 2
        assert meta["_version"] == 7
        assert isinstance(meta["_version"], int)
        assert meta["topic"] == "auth"
        assert "_base_id" in meta

    def test_write_markdown_export_parts_off_by_default(self, keeper, tmp_path):
        _seed(keeper, [
            _make_doc("with-parts", "Parent summary", parts=[
                {"part_num": 1, "summary": "p1", "tags": {},
                 "created_at": "2026-01-01T00:00:00"},
                {"part_num": 2, "summary": "p2", "tags": {},
                 "created_at": "2026-01-01T00:00:00"},
            ]),
        ])
        out = tmp_path / "md-no-parts"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)

        # Parent file present, sidecar dir NOT created (no parts emitted).
        assert (out / "with-parts.md").is_file()
        assert not (out / "with-parts").exists()

    def test_write_markdown_export_parts_sidecar(self, keeper, tmp_path):
        _seed(keeper, [
            _make_doc("with-parts", "Parent summary", parts=[
                {"part_num": 1, "summary": "Introduction analysis text.",
                 "tags": {"section": "intro"},
                 "created_at": "2026-01-01T00:00:00"},
                {"part_num": 2, "summary": "Body analysis text.",
                 "tags": {"section": "body"},
                 "created_at": "2026-01-02T00:00:00"},
                {"part_num": 5, "summary": "Non-contiguous part text.",
                 "tags": {},
                 "created_at": "2026-01-03T00:00:00"},
            ]),
        ])
        out = tmp_path / "md-parts"
        out.mkdir()
        _write_markdown_export(
            keeper, out, include_system=False, include_parts=True,
        )

        # Sidecar dir holds @P{N}.md per part, named after part_num.
        sidecar = out / "with-parts"
        assert sidecar.is_dir()
        assert (sidecar / "@P{1}.md").is_file()
        assert (sidecar / "@P{2}.md").is_file()
        assert (sidecar / "@P{5}.md").is_file()
        # Non-existent part numbers don't get files.
        assert not (sidecar / "@P{3}.md").exists()

        # Each part file's body is just the part summary; chain
        # navigation lives in ``_prev_part`` / ``_next_part``
        # frontmatter keys instead of body markdown links.
        meta, body = _parse_markdown(sidecar / "@P{1}.md")
        assert meta["_id"] == "with-parts"
        assert meta["_part_num"] == 1
        assert meta["section"] == "intro"
        assert "tags" not in meta
        assert body.strip() == "Introduction analysis text."
        # _prev_part points back to the parent doc id (the chain head).
        assert meta["_prev_part"] == "[[with-parts]]"
        # _next_part points to the next-higher part_num's exported path.
        assert meta["_next_part"] == "[[with-parts/@P{2}]]"

    def test_write_markdown_export_versions_sidecar_offsets(self, keeper, tmp_path):
        # `versions` in the seed represents archived (non-current) versions.
        # `list_versions` returns them DESC by version number, so the
        # iterator yields them newest-first → @V{1}, @V{2}, @V{3}.
        _seed(keeper, [
            _make_doc("with-history", "Latest summary", versions=[
                {"version": 4, "summary": "Three steps back", "tags": {},
                 "content_hash": "h4", "created_at": "2025-12-04T00:00:00"},
                {"version": 3, "summary": "Two steps back", "tags": {},
                 "content_hash": "h3", "created_at": "2025-12-03T00:00:00"},
                {"version": 2, "summary": "One step back", "tags": {},
                 "content_hash": "h2", "created_at": "2025-12-02T00:00:00"},
                {"version": 1, "summary": "Oldest", "tags": {},
                 "content_hash": "h1", "created_at": "2025-12-01T00:00:00"},
            ]),
        ])
        out = tmp_path / "md-versions"
        out.mkdir()
        _write_markdown_export(
            keeper, out, include_system=False, include_versions=True,
        )

        # Parent file holds the current version; sidecar holds the @V{N}
        # archived versions, where N counts steps back from current.
        assert (out / "with-history.md").is_file()
        sidecar = out / "with-history"
        assert sidecar.is_dir()
        # The export iterator yields archived versions newest-first, so
        # the first one (the most recent prior version, db version=4) is
        # @V{1}, the next (v3) is @V{2}, etc.
        assert (sidecar / "@V{1}.md").is_file()
        assert (sidecar / "@V{2}.md").is_file()
        assert (sidecar / "@V{3}.md").is_file()
        assert (sidecar / "@V{4}.md").is_file()

        meta1, body1 = _parse_markdown(sidecar / "@V{1}.md")
        assert meta1["_id"] == "with-history"
        assert meta1["_version_offset"] == 1
        assert meta1["_version"] == 4  # absolute db version number for reference
        assert meta1["_content_hash"] == "h4"
        assert meta1["_created"] == "2025-12-04T00:00:00"
        assert body1.strip() == "Three steps back"
        # @V{1} is the chain entry from the parent: _prev_version
        # points to the next-older archived version, _next_version
        # points back to the parent doc.
        assert meta1["_prev_version"] == "[[with-history/@V{2}]]"
        assert meta1["_next_version"] == "[[with-history]]"

        # Sanity check the deepest archived version maps correctly.
        meta4, body4 = _parse_markdown(sidecar / "@V{4}.md")
        assert meta4["_version_offset"] == 4
        assert meta4["_version"] == 1
        assert body4.strip() == "Oldest"
        # The chain tail has only a "next" link (no older version).
        assert "_prev_version" not in meta4
        assert meta4["_next_version"] == "[[with-history/@V{3}]]"

        # The current (latest) version stays in the parent file, NOT in
        # the sidecar — there is no @V{0}.md.
        assert not (sidecar / "@V{0}.md").exists()
        parent_meta, parent_body = _parse_markdown(out / "with-history.md")
        assert parent_body.strip() == "Latest summary"
        # Parent doc records a `_prev_version` chain entry pointing
        # at @V{1}.  No `_next_version` — the parent IS current.
        assert parent_meta["_prev_version"] == "[[with-history/@V{1}]]"
        assert "_next_version" not in parent_meta

    def test_write_markdown_export_parts_and_versions_combined(self, keeper, tmp_path):
        _seed(keeper, [
            _make_doc("combo", "current", parts=[
                {"part_num": 1, "summary": "part one text", "tags": {},
                 "created_at": "2026-01-01T00:00:00"},
            ], versions=[
                {"version": 1, "summary": "old", "tags": {},
                 "content_hash": None, "created_at": "2025-12-01T00:00:00"},
            ]),
        ])
        out = tmp_path / "md-combo"
        out.mkdir()
        _write_markdown_export(
            keeper, out, include_system=False,
            include_parts=True, include_versions=True,
        )

        sidecar = out / "combo"
        assert (out / "combo.md").is_file()
        assert (sidecar / "@P{1}.md").is_file()
        assert (sidecar / "@V{1}.md").is_file()

    def test_write_markdown_export_no_sidecar_when_no_parts_or_versions(self, keeper, tmp_path):
        # Notes that have no parts/versions get no sidecar dir even when
        # both flags are on.
        _seed(keeper, [_make_doc("plain", "just a summary")])
        out = tmp_path / "md-plain"
        out.mkdir()
        _write_markdown_export(
            keeper, out, include_system=False,
            include_parts=True, include_versions=True,
        )
        assert (out / "plain.md").is_file()
        assert not (out / "plain").exists()

    def test_write_markdown_export_case_fold_file_vs_sidecar_dir(
        self, keeper, tmp_path,
    ):
        """File from one id auto-disambiguates from sidecar dir of another.

        Reproduces a bug where two file:// ids whose paths case-fold
        to the same on-disk slot crashed with a raw FileExistsError on
        case-insensitive filesystems when the sidecar dir creation
        tried to overwrite an existing file:

            ``file:///abc/def``    → file/abc/def.md       (file)
            ``file:///abc/DEF.md`` → file/abc/DEF.md.md    (file)
                                     file/abc/DEF.md/      (sidecar dir)

        On macOS APFS the sidecar dir ``file/abc/DEF.md`` case-folds
        to ``file/abc/def.md`` — already a regular file from the
        first id.  The export now appends a hash suffix to the
        second doc's stem so both notes coexist on disk and the
        sidecar dir lives at a non-colliding path.
        """
        _seed(keeper, [
            _make_doc("file:///abc/def", "A body"),
            _make_doc("file:///abc/DEF.md", "B body", parts=[
                {"part_num": 1, "summary": "p1", "tags": {},
                 "created_at": "2026-01-01T00:00:00"},
            ]),
        ])
        out = tmp_path / "md-casefold"
        out.mkdir()
        _write_markdown_export(
            keeper, out, include_system=False, include_parts=True,
        )

        # The first doc landed at the canonical path.
        canonical_a = out / "file" / "abc" / "def.md"
        assert canonical_a.is_file()

        # The second doc was disambiguated — its main file is no
        # longer at ``DEF.md.md`` but at ``DEF.md.<hash>.md``, and
        # its sidecar dir is at ``DEF.md.<hash>/``.  Find them.
        b_files = [
            p for p in (out / "file" / "abc").iterdir()
            if p.is_file() and p != canonical_a
        ]
        assert len(b_files) == 1, f"expected one B file, got {b_files}"
        b_main = b_files[0]
        assert b_main.name.startswith("DEF.md.")
        assert b_main.name.endswith(".md")
        # The disambiguation hash is 8 hex chars wedged into the stem.
        # The frontmatter still records the canonical _id.
        meta_b, _ = _parse_markdown(b_main)
        assert meta_b["_id"] == "file:///abc/DEF.md"

        # The sidecar dir matches the disambiguated stem (so its case-
        # folded name no longer collides with ``def.md``) and the part
        # file lives inside it.
        b_sidecar = b_main.with_suffix("")
        assert b_sidecar.is_dir()
        assert (b_sidecar / "@P{1}.md").is_file()

    def test_write_markdown_export_case_fold_two_main_files(
        self, keeper, tmp_path,
    ):
        """Two main-file paths that differ only in case auto-disambiguate."""
        _seed(keeper, [
            _make_doc("readme", "lower"),
            _make_doc("README", "upper"),
        ])
        out = tmp_path / "md-case-files"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)

        files = sorted(p.name for p in out.iterdir() if p.is_file())
        # Two distinct on-disk files exist — one canonical, one with
        # an 8-hex-char disambiguation suffix.
        assert len(files) == 2
        canonical = {"readme.md", "README.md"}
        disambiguated = [f for f in files if f not in canonical]
        canonical_files = [f for f in files if f in canonical]
        assert len(canonical_files) == 1
        assert len(disambiguated) == 1
        # The disambiguated file's stem is `<canonical_stem>.<hash>`.
        assert disambiguated[0].count(".") == 2

        # Both notes' frontmatter still records the canonical _id —
        # the disambiguation is purely an on-disk detail.
        ids = set()
        for f in files:
            meta, _ = _parse_markdown(out / f)
            ids.add(meta["_id"])
        assert ids == {"readme", "README"}

    def test_write_markdown_export_disambiguated_doc_keeps_inverse_edges(
        self, keeper, tmp_path,
    ):
        """A disambiguated doc still records its inverse edges in frontmatter.

        Inverse-edge values are rendered as exported wikilinks, so the
        doc that got disambiguated still needs to carry its incoming
        edge data through to its renamed file.
        """
        _seed(keeper, [
            _make_doc("readme", "lower"),
            _make_doc("README", "upper"),
            _make_doc("source", "Source body"),
        ])
        # 'source' has an edge pointing at the lowercase 'readme'.
        ds = keeper._document_store
        coll = keeper._resolve_doc_collection()
        ds.upsert_edge(
            coll, "source", "speaker", "readme", "said",
            "2026-01-15T10:00:00",
        )

        out = tmp_path / "md-disambig-edges"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)

        # Find each file by walking the directory and matching the
        # canonical _id from its frontmatter.  One of the readmes will
        # have been disambiguated with a hash suffix.
        files_by_id: dict[str, Path] = {}
        for p in out.iterdir():
            if not p.is_file():
                continue
            meta, _ = _parse_markdown(p)
            files_by_id[meta["_id"]] = p

        # All three notes are present.
        assert set(files_by_id) == {"readme", "README", "source"}

        # The lowercase 'readme' has its incoming edge as a flat
        # frontmatter key, regardless of which file the disambiguator
        # picked. The source note still links by its exported path.
        readme_meta, _ = _parse_markdown(files_by_id["readme"])
        said = readme_meta["said"]
        assert isinstance(said, list)
        assert any(v == "[[source|Source body]]" for v in said)

        # The uppercase 'README' has no incoming edges and shouldn't
        # carry a stray ``said`` key.
        upper_meta, _ = _parse_markdown(files_by_id["README"])
        assert "said" not in upper_meta

    def test_forward_edge_tags_use_export_paths_for_uri_targets(
        self, keeper, tmp_path,
    ):
        """Edge-tag refs are rewritten to the exported vault path namespace."""
        file_id = "file:///tmp/readme.md"
        _seed(keeper, [
            {
                "id": ".tag/speaker",
                "summary": "speaker tagdoc",
                "tags": {"_inverse": "said"},
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "accessed_at": "2026-01-01T00:00:00",
            },
            _make_doc(file_id, "Readme"),
            _make_doc("conv", "Conversation", tags={"speaker": file_id}),
        ])
        out = tmp_path / "md-uri-edge"
        out.mkdir()

        _write_markdown_export(keeper, out, include_system=False)

        meta, _ = _parse_markdown(out / "conv.md")
        assert meta["speaker"] == "[[file/tmp/readme.md]]"

    def test_non_edge_tag_values_are_not_rewritten_to_export_refs(
        self, keeper, tmp_path,
    ):
        """Only actual edge tags become export-local wikilinks."""
        file_id = "file:///tmp/readme.md"
        _seed(keeper, [
            _make_doc(file_id, "Readme"),
            _make_doc("conv", "Conversation", tags={"topic": file_id}),
        ])
        out = tmp_path / "md-non-edge-ref"
        out.mkdir()

        _write_markdown_export(keeper, out, include_system=False)

        meta, _ = _parse_markdown(out / "conv.md")
        assert meta["topic"] == file_id

    def test_chain_links_use_export_paths_for_uri_backed_parent(
        self, keeper, tmp_path,
    ):
        """Prev/next chain refs follow the exported sidecar paths."""
        file_id = "file:///tmp/readme.md"
        _seed(keeper, [
            _make_doc(file_id, "Current", parts=[
                {"part_num": 1, "summary": "p1",
                 "tags": {}, "created_at": "2026-01-01T00:00:00"},
            ], versions=[
                {"version": 1, "summary": "older",
                 "tags": {}, "content_hash": "h1",
                 "created_at": "2025-12-01T00:00:00"},
            ]),
        ])
        out = tmp_path / "md-uri-chain"
        out.mkdir()

        _write_markdown_export(
            keeper, out, include_system=False,
            include_parts=True, include_versions=True,
        )

        parent = out / "file" / "tmp" / "readme.md.md"
        parent_meta, _ = _parse_markdown(parent)
        assert parent_meta["_next_part"] == "[[file/tmp/readme.md/@P{1}]]"
        assert parent_meta["_prev_version"] == "[[file/tmp/readme.md/@V{1}]]"

        part_meta, _ = _parse_markdown(out / "file" / "tmp" / "readme.md" / "@P{1}.md")
        assert part_meta["_prev_part"] == "[[file/tmp/readme.md]]"

        version_meta, _ = _parse_markdown(
            out / "file" / "tmp" / "readme.md" / "@V{1}.md",
        )
        assert version_meta["_next_version"] == "[[file/tmp/readme.md]]"

    def test_forward_edge_tags_use_export_paths_for_email_targets(
        self, keeper, tmp_path,
    ):
        """Bare email IDs export to the grouped email path namespace."""
        _seed(keeper, [
            {
                "id": ".tag/speaker",
                "summary": "speaker tagdoc",
                "tags": {"_inverse": "said"},
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "accessed_at": "2026-01-01T00:00:00",
            },
            _make_doc("alice@example.com", "Alice"),
            _make_doc("conv", "Conversation", tags={"speaker": "alice@example.com"}),
        ])
        out = tmp_path / "md-email-edge"
        out.mkdir()

        _write_markdown_export(keeper, out, include_system=False)

        meta, _ = _parse_markdown(out / "conv.md")
        assert meta["speaker"] == "[[_email/example.com/alice@example.com]]"

    def test_inverse_edges_use_disambiguated_export_paths(
        self, keeper, tmp_path,
    ):
        """Backlinks point at the final exported filename when ids collide."""
        _seed(keeper, [
            _make_doc("readme", "lower"),
            _make_doc("README", "upper"),
            _make_doc("target", "Target"),
        ])
        ds = keeper._document_store
        coll = keeper._resolve_doc_collection()
        ds.upsert_edge(
            coll, "README", "speaker", "target", "said",
            "2026-01-15T10:00:00",
        )

        out = tmp_path / "md-disambig-export-ref"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)

        files_by_id: dict[str, Path] = {}
        for p in out.iterdir():
            if not p.is_file():
                continue
            meta, _ = _parse_markdown(p)
            files_by_id[meta["_id"]] = p

        exported_readme_ref = (
            files_by_id["README"].relative_to(out).with_suffix("").as_posix()
        )
        target_meta, _ = _parse_markdown(files_by_id["target"])
        assert target_meta["said"] == [f"[[{exported_readme_ref}|upper]]"]

    def test_cli_include_parts_and_versions_flags(self, tmp_path):
        from typer.testing import CliRunner
        from keep.cli_app import app
        r = CliRunner().invoke(app, ["data", "export", "--help"])
        assert r.exit_code == 0
        assert "--include-parts" in r.output
        assert "--include-versions" in r.output

    def test_cli_rejects_include_parts_with_json(self, tmp_path):
        from typer.testing import CliRunner
        from keep.cli_app import app
        r = CliRunner().invoke(
            app,
            ["--store", str(tmp_path), "data", "export", "out.json",
             "--include-parts"],
        )
        assert r.exit_code != 0
        assert "only applies to --format md" in r.output

    def test_cli_rejects_include_versions_with_json(self, tmp_path):
        from typer.testing import CliRunner
        from keep.cli_app import app
        r = CliRunner().invoke(
            app,
            ["--store", str(tmp_path), "data", "export", "out.json",
             "--include-versions"],
        )
        assert r.exit_code != 0
        assert "only applies to --format md" in r.output

    def test_cli_rejects_stdout_for_markdown(self, tmp_path):
        from typer.testing import CliRunner
        from keep.cli_app import app
        r = CliRunner().invoke(
            app, ["--store", str(tmp_path), "data", "export", "-", "--format", "md"],
        )
        assert r.exit_code != 0
        assert "markdown export requires a directory" in r.output

    def test_cli_rejects_nonempty_markdown_dir(self, tmp_path):
        from typer.testing import CliRunner
        from keep.cli_app import app
        out = tmp_path / "nonempty"
        out.mkdir()
        (out / "junk.txt").write_text("x")
        r = CliRunner().invoke(
            app, ["--store", str(tmp_path), "data", "export", str(out), "--format", "md"],
        )
        assert r.exit_code != 0
        assert "not empty" in r.output

    def test_cli_markdown_export_uses_bundle_host(self, tmp_path):
        from typer.testing import CliRunner
        from unittest.mock import MagicMock, patch
        from keep.cli_app import app

        out = tmp_path / "vault"

        class RemoteMarkdownHost:
            def __init__(self) -> None:
                self.close = MagicMock()

            def export_iter(self, *, include_system: bool = True):
                yield {
                    "format": "keep-export",
                    "version": 3,
                    "exported_at": "2026-04-09T18:00:00",
                    "store_info": {
                        "document_count": 2,
                        "version_count": 0,
                        "part_count": 0,
                        "collection": "default",
                    },
                }
                yield {
                    "id": "Joanna",
                    "summary": "Joanna note",
                    "tags": {},
                    "created_at": "2026-04-09T18:00:00",
                    "updated_at": "2026-04-09T18:00:00",
                    "accessed_at": "2026-04-09T18:00:00",
                }
                yield {
                    "id": "sessions/one",
                    "summary": "Session body",
                    "tags": {"speaker": "Joanna"},
                    "created_at": "2026-04-09T18:00:00",
                    "updated_at": "2026-04-09T18:00:00",
                    "accessed_at": "2026-04-09T18:00:00",
                }

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
                        "document": {
                            "id": "Joanna",
                            "summary": "Joanna note",
                            "tags": {},
                            "created_at": "2026-04-09T18:00:00",
                            "updated_at": "2026-04-09T18:00:00",
                            "accessed_at": "2026-04-09T18:00:00",
                        },
                        "current_inverse": [["said", "sessions/one"]],
                        "version_inverse": [],
                        "edge_tag_keys": [],
                    },
                    "sessions/one": {
                        "document": {
                            "id": "sessions/one",
                            "summary": "Session body",
                            "tags": {"speaker": "Joanna"},
                            "created_at": "2026-04-09T18:00:00",
                            "updated_at": "2026-04-09T18:00:00",
                            "accessed_at": "2026-04-09T18:00:00",
                        },
                        "current_inverse": [],
                        "version_inverse": [],
                        "edge_tag_keys": ["speaker"],
                    },
                }
                return bundles.get(id)

        remote_host = RemoteMarkdownHost()

        with patch("keep.cli_app._get_export_host", return_value=remote_host):
            r = CliRunner().invoke(
                app, ["--store", str(tmp_path), "data", "export", str(out), "--format", "md"],
            )

        assert r.exit_code == 0
        assert (out / "Joanna.md").is_file()
        assert (out / "sessions" / "one.md").is_file()
        assert "[[sessions/one]]" in (out / "Joanna.md").read_text(encoding="utf-8")
        assert 'speaker: "[[Joanna]]"' in (out / "sessions" / "one.md").read_text(encoding="utf-8")
        remote_host.close.assert_called_once()

    def test_cli_remote_json_import_uses_hosted_endpoint(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        from unittest.mock import patch
        from keep import cli_app
        from keep.cli_app import app

        store = tmp_path / "store"
        store.mkdir()
        _write_test_store_config(store)
        with (store / "keep.toml").open("a", encoding="utf-8") as fh:
            fh.write(
                "\n[remote]\n"
                "api_url = \"https://api.example.test\"\n"
                "api_key = \"kn_test\"\n"
                "project = \"first-user\"\n"
            )
        src = tmp_path / "export.json"
        src.write_text(json.dumps({
            "format": "keep-export",
            "version": 3,
            "documents": [_make_doc("remote-doc", "remote body")],
        }))
        calls: list[tuple[str, str, dict | None]] = []

        def fake_remote_request(method, path, body=None):
            calls.append((method, path, body))
            return 200, {
                "imported": 1,
                "skipped": 0,
                "versions": 0,
                "parts": 0,
                "queued": 1,
            }

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.setattr(cli_app, "_global_store", None)
        monkeypatch.setattr(cli_app, "_remote_request", fake_remote_request)
        monkeypatch.setattr(
            cli_app,
            "_daemon_get_port",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("local daemon must not start in remote mode")
            ),
        )

        with patch(
            "keep.api.Keeper",
            side_effect=AssertionError("local Keeper must not open in remote mode"),
        ):
            r = CliRunner().invoke(
                app,
                ["--store", str(store), "data", "import", str(src)],
                catch_exceptions=False,
            )

        assert r.exit_code == 0, r.output
        assert calls == [("POST", "/v1/import", {
            "format": "keep-export",
            "version": 3,
            "documents": [_make_doc("remote-doc", "remote body")],
            "mode": "merge",
        })]
        assert "hosted processing" in r.stderr

    def test_cli_remote_markdown_import_uses_hosted_endpoint(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        from unittest.mock import patch
        from keep import cli_app
        from keep.cli_app import app

        store = tmp_path / "store"
        store.mkdir()
        _write_test_store_config(store)
        with (store / "keep.toml").open("a", encoding="utf-8") as fh:
            fh.write(
                "\n[remote]\n"
                "api_url = \"https://api.example.test\"\n"
                "api_key = \"kn_test\"\n"
                "project = \"first-user\"\n"
            )
        src = tmp_path / "note.md"
        src.write_text("---\n_id: md-remote\ntopic: hosted\n---\nMarkdown body\n")
        calls: list[tuple[str, str, dict | None]] = []

        def fake_remote_request(method, path, body=None):
            calls.append((method, path, body))
            return 200, {
                "imported": 1,
                "skipped": 0,
                "versions": 0,
                "parts": 0,
                "queued": 1,
            }

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.setattr(cli_app, "_global_store", None)
        monkeypatch.setattr(cli_app, "_remote_request", fake_remote_request)
        monkeypatch.setattr(
            cli_app,
            "_daemon_get_port",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("local daemon must not start in remote mode")
            ),
        )

        with patch(
            "keep.api.Keeper",
            side_effect=AssertionError("local Keeper must not open in remote mode"),
        ):
            r = CliRunner().invoke(
                app,
                ["--store", str(store), "data", "import", str(src), "--format", "md"],
                catch_exceptions=False,
            )

        assert r.exit_code == 0, r.output
        assert len(calls) == 1
        method, path, body = calls[0]
        assert (method, path) == ("POST", "/v1/import")
        assert body["mode"] == "merge"
        assert body["documents"][0]["id"] == "md-remote"
        assert body["documents"][0]["summary"] == "Markdown body\n"

    def test_cli_markdown_sync_uses_remote_export_host(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        from unittest.mock import MagicMock, patch
        from keep import cli_app
        from keep.cli_app import app

        store = tmp_path / "store"
        store.mkdir()
        _write_test_store_config(store)
        with (store / "keep.toml").open("a", encoding="utf-8") as fh:
            fh.write(
                "\n[remote]\n"
                "api_url = \"https://api.example.test\"\n"
                "api_key = \"kn_test\"\n"
                "project = \"first-user\"\n"
            )

        class RemoteSyncHost:
            def __init__(self) -> None:
                self.close = MagicMock()

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
                yield {
                    "id": "remote-sync-doc",
                    "summary": "Remote sync body",
                    "tags": {},
                    "created_at": "2026-04-09T18:00:00",
                    "updated_at": "2026-04-09T18:00:00",
                    "accessed_at": "2026-04-09T18:00:00",
                }

            def export_bundle(
                self,
                id: str,
                *,
                include_system: bool = True,
                include_parts: bool = True,
                include_versions: bool = True,
            ):
                if id == "remote-sync-doc":
                    return {
                        "document": {
                            "id": "remote-sync-doc",
                            "summary": "Remote sync body",
                            "tags": {},
                            "created_at": "2026-04-09T18:00:00",
                            "updated_at": "2026-04-09T18:00:00",
                            "accessed_at": "2026-04-09T18:00:00",
                        },
                        "current_inverse": [],
                        "version_inverse": [],
                        "edge_tag_keys": [],
                    }
                return None

            def export_changes(self, *, cursor: str | None = None, limit: int = 1000):
                return {
                    "format": "keep-export-changes",
                    "version": 1,
                    "cursor": "42",
                    "head_cursor": "42",
                    "compacted": False,
                    "truncated": False,
                    "events": [],
                }

        remote_host = RemoteSyncHost()

        def fake_daemon_request(method, port, path, body=None):
            assert port == 1234
            assert path == "/v1/admin/markdown-export"
            if body and body.get("validate_only"):
                return 200, {"validated": True, "root": str((tmp_path / "vault").resolve())}
            return 200, {"sync": {"root": str((tmp_path / "vault").resolve())}}

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.setattr(cli_app, "_global_store", None)
        with patch("keep.cli_app._get_export_host", return_value=remote_host), \
             patch("keep.cli_app._get_local_daemon_port", return_value=1234) as local_port, \
             patch("keep.cli_app._remote_request", side_effect=AssertionError("remote admin call")), \
             patch("keep.cli_app._daemon_request", side_effect=fake_daemon_request) as daemon_request:
            r = CliRunner().invoke(
                app,
                [
                    "--store", str(store), "data", "export",
                    str(tmp_path / "vault"), "--format", "md", "--sync",
                ],
            )

        assert r.exit_code == 0
        assert (tmp_path / "vault" / "remote-sync-doc.md").is_file()
        assert "Markdown sync active:" in r.output
        assert local_port.call_count == 2
        register_body = daemon_request.call_args_list[1].args[3]
        assert register_body["source_cursor"] == "42"
        remote_host.close.assert_called_once()

    def test_cli_markdown_sync_list_uses_local_daemon_when_remote_configured(
        self, tmp_path, monkeypatch,
    ):
        from typer.testing import CliRunner
        from unittest.mock import patch
        from keep import cli_app
        from keep.cli_app import app

        store = tmp_path / "store"
        store.mkdir()
        _write_test_store_config(store)
        with (store / "keep.toml").open("a", encoding="utf-8") as fh:
            fh.write(
                "\n[remote]\n"
                "api_url = \"https://api.example.test\"\n"
                "api_key = \"kn_test\"\n"
                "project = \"first-user\"\n"
            )
        calls: list[tuple[str, int, str, dict | None]] = []

        def fake_daemon_request(method, port, path, body=None):
            calls.append((method, port, path, body))
            return 200, {"mirrors": []}

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.setattr(cli_app, "_global_store", None)
        with patch("keep.cli_app._get_local_daemon_port", return_value=1234), \
             patch("keep.cli_app._remote_request", side_effect=AssertionError("remote admin call")), \
             patch("keep.cli_app._daemon_request", side_effect=fake_daemon_request):
            r = CliRunner().invoke(
                app,
                ["--store", str(store), "data", "export", "--list"],
            )

        assert r.exit_code == 0, r.output
        assert "No markdown sync directories." in r.output
        assert calls == [("POST", 1234, "/v1/admin/markdown-export", {"list": True})]

    def test_cli_markdown_sync_stop_uses_local_daemon_when_remote_configured(
        self, tmp_path, monkeypatch,
    ):
        from typer.testing import CliRunner
        from unittest.mock import patch
        from keep import cli_app
        from keep.cli_app import app

        store = tmp_path / "store"
        store.mkdir()
        _write_test_store_config(store)
        with (store / "keep.toml").open("a", encoding="utf-8") as fh:
            fh.write(
                "\n[remote]\n"
                "api_url = \"https://api.example.test\"\n"
                "api_key = \"kn_test\"\n"
                "project = \"first-user\"\n"
            )
        calls: list[tuple[str, int, str, dict | None]] = []

        def fake_daemon_request(method, port, path, body=None):
            calls.append((method, port, path, body))
            return 200, {"stopped": True, "root": str(tmp_path / "vault")}

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.setattr(cli_app, "_global_store", None)
        with patch("keep.cli_app._get_local_daemon_port", return_value=1234), \
             patch("keep.cli_app._remote_request", side_effect=AssertionError("remote admin call")), \
             patch("keep.cli_app._daemon_request", side_effect=fake_daemon_request):
            r = CliRunner().invoke(
                app,
                [
                    "--store", str(store), "data", "export",
                    str(tmp_path / "vault"), "--sync", "--stop",
                ],
            )

        assert r.exit_code == 0, r.output
        assert "Stopped markdown sync:" in r.output
        assert len(calls) == 1
        method, port, path, body = calls[0]
        assert (method, port, path) == ("POST", 1234, "/v1/admin/markdown-export")
        assert body["stop"] is True
        assert body["register_only"] is True

    def test_cli_json_export_uses_export_host(self, tmp_path):
        from typer.testing import CliRunner
        from unittest.mock import MagicMock, patch
        from keep.cli_app import app

        out = tmp_path / "remote-export.json"

        class ExportHost:
            def __init__(self) -> None:
                self.close = MagicMock()

            def export_iter(self, *, include_system: bool = True):
                yield {
                    "format": "keep-export",
                    "version": 3,
                    "exported_at": "2026-04-09T17:00:00",
                    "store_info": {
                        "document_count": 1,
                        "version_count": 0,
                        "part_count": 0,
                        "collection": "default",
                    },
                }
                yield {
                    "id": "remote-doc",
                    "summary": "Remote export body",
                    "tags": {"topic": "transport"},
                    "created_at": "2026-04-09T17:00:00",
                    "updated_at": "2026-04-09T17:00:00",
                    "accessed_at": "2026-04-09T17:00:00",
                }

        host = ExportHost()

        with patch("keep.cli_app._get_export_host", return_value=host):
            r = CliRunner().invoke(
                app, ["--store", str(tmp_path), "data", "export", str(out)],
            )

        assert r.exit_code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["documents"][0]["id"] == "remote-doc"
        host.close.assert_called_once()

    def test_cli_rejects_stop_without_sync(self, tmp_path):
        from typer.testing import CliRunner
        from keep.cli_app import app

        r = CliRunner().invoke(
            app, ["--store", str(tmp_path), "data", "export", str(tmp_path / "vault"), "--format", "md", "--stop"],
        )
        assert r.exit_code != 0
        assert "--stop requires --sync" in r.output

    def test_cli_rejects_unknown_format(self, tmp_path):
        from typer.testing import CliRunner
        from keep.cli_app import app
        r = CliRunner().invoke(
            app, ["--store", str(tmp_path), "data", "export", "foo", "--format", "xml"],
        )
        assert r.exit_code != 0
        assert "must be 'json' or 'md'" in r.output

    def test_cli_help_shows_include_system_flag(self):
        from typer.testing import CliRunner
        from keep.cli_app import app
        r = CliRunner().invoke(app, ["data", "export", "--help"])
        assert r.exit_code == 0
        assert "--include-system" in r.output
        # Old --exclude-system flag must be gone.
        assert "--exclude-system" not in r.output

    def test_write_markdown_export_skips_versions_and_parts(self, keeper, tmp_path):
        _seed(keeper, [
            _make_doc(
                "with-history", "Latest summary",
                versions=[{
                    "version": 1, "summary": "Old summary", "tags": {},
                    "content_hash": None, "created_at": "2025-12-01T00:00:00",
                }],
                parts=[{
                    "part_num": 1, "summary": "p1", "tags": {},
                    "created_at": "2026-01-01T00:00:00",
                }],
            ),
        ])
        out = tmp_path / "md-history"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)

        text = (out / "with-history.md").read_text(encoding="utf-8")
        assert "Latest summary" in text
        # Old version bodies must not leak into the export.
        assert "Old summary" not in text

    def test_write_markdown_export_progress_callback(self, keeper, tmp_path):
        _seed(keeper, [
            _make_doc("a", "first note"),
            _make_doc("b", "second note"),
            _make_doc("c", "third note"),
        ])
        out = tmp_path / "md-progress"
        out.mkdir()

        events: list[tuple[int, int, str]] = []
        _write_markdown_export(
            keeper, out, include_system=False,
            progress=lambda cur, total, label: events.append((cur, total, label)),
        )

        # One callback per exported note, counter strictly increasing from 1..N.
        assert len(events) == 3
        assert [e[0] for e in events] == [1, 2, 3]
        # Total is stable across calls and matches the number of notes.
        assert {e[1] for e in events} == {3}
        # Labels are note ids that were just written.
        assert {e[2] for e in events} == {"a", "b", "c"}

    # ------------------------------------------------------------------
    # Linear-chain navigation between parent and sidecars
    # ------------------------------------------------------------------

    def test_md_link_target_same_dir(self):
        # note.md → other.md (root to root): no '..' prefix.
        assert _md_link_target(Path("other.md"), Path("note.md")) == "other.md"

    def test_md_link_target_into_subdir(self):
        # note.md → note/@P{1}.md: relative descend, '{}' percent-encoded.
        assert (
            _md_link_target(
                Path("note") / "@P{1}.md",
                Path("note.md"),
            )
            == "note/@P%7B1%7D.md"
        )

    def test_md_link_target_back_to_parent(self):
        # note/@P{1}.md → note.md: one '..' segment.
        assert (
            _md_link_target(
                Path("note.md"),
                Path("note") / "@P{1}.md",
            )
            == "../note.md"
        )

    def test_md_link_target_sibling_in_same_subdir(self):
        # note/@P{1}.md → note/@P{2}.md: same dir, no traversal.
        assert (
            _md_link_target(
                Path("note") / "@P{2}.md",
                Path("note") / "@P{1}.md",
            )
            == "@P%7B2%7D.md"
        )

    def test_md_link_target_nested_root(self):
        # notes/2024/jan.md → notes/2024/jan/@P{1}.md.
        assert (
            _md_link_target(
                Path("notes") / "2024" / "jan" / "@P{1}.md",
                Path("notes") / "2024" / "jan.md",
            )
            == "jan/@P%7B1%7D.md"
        )

    def test_md_link_target_nested_back_to_root(self):
        # notes/2024/jan/@P{1}.md → notes/2024/jan.md: '../jan.md'.
        assert (
            _md_link_target(
                Path("notes") / "2024" / "jan.md",
                Path("notes") / "2024" / "jan" / "@P{1}.md",
            )
            == "../jan.md"
        )

    def test_md_link_target_across_dirs(self):
        # nested → root sibling: traversal up + down.
        assert (
            _md_link_target(
                Path("Deborah.md"),
                Path("notes") / "2024" / "jan.md",
            )
            == "../../Deborah.md"
        )

    def test_md_link_target_url_encodes_parens(self):
        # On-disk component with '()' must percent-encode in the link
        # target — bare parens would otherwise unbalance markdown's
        # link-URL syntax.
        assert (
            _md_link_target(
                Path("foo(bar).md"),
                Path("here.md"),
            )
            == "foo%28bar%29.md"
        )

    def test_md_link_target_double_encodes_literal_percent(self):
        # `_id_to_rel_path` already percent-encodes filesystem-unsafe
        # chars inside each path component, so '#' lands on disk as
        # the LITERAL bytes '%23' inside the filename.  For a wiki
        # tool to URL-decode a markdown link target back to that
        # literal filename, every '%' has to be re-encoded as '%25'
        # — otherwise '%23' in the link decodes to '#' and resolves
        # to nothing on disk.
        rel = _id_to_rel_path("thread:abc@host#frag")
        # Sanity: the on-disk component carries the percent-escape.
        assert "%23frag" in rel.parts[-1]
        link = _md_link_target(rel, Path("anchor.md"))
        # '%23' must be re-encoded as '%2523' so URL-decoding it gives
        # back the literal '%23' that exists on disk.
        assert "%2523frag" in link

    def test_md_link_target_round_trips_to_on_disk_filename(self):
        # End-to-end check of the encode/decode round trip: the URL
        # produced by _md_link_target must, after URL-decoding,
        # exactly match the on-disk relative path.
        from urllib.parse import unquote
        for doc_id in (
            "auth-notes",
            "notes/2024/jan",
            "file:///Users/x/README.md",
            "thread:abc@host#frag",
            "%9f86d081884c",
            "alice@example.com",
        ):
            rel = _id_to_rel_path(doc_id)
            link = _md_link_target(rel, Path("anchor.md"))
            assert unquote(link) == rel.as_posix(), (
                f"round-trip failed for {doc_id!r}: "
                f"link={link!r} decoded={unquote(link)!r} "
                f"on-disk={rel.as_posix()!r}"
            )

    def test_parts_chain_links_non_contiguous(self, keeper, tmp_path):
        """Parts chain follows part_num order even with gaps."""
        _seed(keeper, [
            _make_doc("doc", "Parent body", parts=[
                {"part_num": 1, "summary": "first part",
                 "tags": {}, "created_at": "2026-01-01T00:00:00"},
                {"part_num": 2, "summary": "second part",
                 "tags": {}, "created_at": "2026-01-02T00:00:00"},
                {"part_num": 5, "summary": "fifth part",
                 "tags": {}, "created_at": "2026-01-03T00:00:00"},
            ]),
        ])
        out = tmp_path / "md-parts-chain"
        out.mkdir()
        _write_markdown_export(
            keeper, out, include_system=False, include_parts=True,
        )

        # Parent links forward to the first part in the chain via the
        # exported sidecar path.
        parent_meta, parent_body = _parse_markdown(out / "doc.md")
        assert parent_meta["_next_part"] == "[[doc/@P{1}]]"
        assert "_prev_part" not in parent_meta
        # Parent body is just its summary — no chain markdown links.
        assert parent_body.strip() == "Parent body"

        # Chain head: _prev_part → parent, _next_part → @P{2}
        m1, b1 = _parse_markdown(out / "doc" / "@P{1}.md")
        assert m1["_prev_part"] == "[[doc]]"
        assert m1["_next_part"] == "[[doc/@P{2}]]"
        assert b1.strip() == "first part"

        # Chain middle: _prev_part → @P{1}, _next_part → @P{5}
        m2, b2 = _parse_markdown(out / "doc" / "@P{2}.md")
        assert m2["_prev_part"] == "[[doc/@P{1}]]"
        assert m2["_next_part"] == "[[doc/@P{5}]]"
        assert b2.strip() == "second part"

        # Chain tail: _prev_part → @P{2}, no _next_part.
        m5, b5 = _parse_markdown(out / "doc" / "@P{5}.md")
        assert m5["_prev_part"] == "[[doc/@P{2}]]"
        assert "_next_part" not in m5
        assert b5.strip() == "fifth part"

    def test_versions_chain_links(self, keeper, tmp_path):
        """Versions chain follows @V{N} offset order."""
        _seed(keeper, [
            _make_doc("doc", "current", versions=[
                {"version": 4, "summary": "v4 (one back)",
                 "tags": {}, "content_hash": "h4",
                 "created_at": "2025-12-04T00:00:00"},
                {"version": 3, "summary": "v3 (two back)",
                 "tags": {}, "content_hash": "h3",
                 "created_at": "2025-12-03T00:00:00"},
                {"version": 2, "summary": "v2 (three back)",
                 "tags": {}, "content_hash": "h2",
                 "created_at": "2025-12-02T00:00:00"},
                {"version": 1, "summary": "v1 (four back)",
                 "tags": {}, "content_hash": "h1",
                 "created_at": "2025-12-01T00:00:00"},
            ]),
        ])
        out = tmp_path / "md-versions-chain"
        out.mkdir()
        _write_markdown_export(
            keeper, out, include_system=False, include_versions=True,
        )

        # Parent records its chain entry as a `_prev_version` key.
        parent_meta, parent_body = _parse_markdown(out / "doc.md")
        assert parent_meta["_prev_version"] == "[[doc/@V{1}]]"
        assert "_next_version" not in parent_meta
        assert parent_body.strip() == "current"

        # Chain head: _prev_version → @V{2}, _next_version → parent
        m1, _ = _parse_markdown(out / "doc" / "@V{1}.md")
        assert m1["_prev_version"] == "[[doc/@V{2}]]"
        assert m1["_next_version"] == "[[doc]]"

        # Middle: _prev_version → @V{3}, _next_version → @V{1}
        m2, _ = _parse_markdown(out / "doc" / "@V{2}.md")
        assert m2["_prev_version"] == "[[doc/@V{3}]]"
        assert m2["_next_version"] == "[[doc/@V{1}]]"

        # Chain tail (oldest): no _prev_version, _next_version → @V{3}
        m4, _ = _parse_markdown(out / "doc" / "@V{4}.md")
        assert "_prev_version" not in m4
        assert m4["_next_version"] == "[[doc/@V{3}]]"

    def test_parent_doc_chain_entries_for_both_parts_and_versions(
        self, keeper, tmp_path,
    ):
        """A doc with parts AND versions records both chain entries."""
        _seed(keeper, [
            _make_doc("doc", "current", parts=[
                {"part_num": 1, "summary": "p1",
                 "tags": {}, "created_at": "2026-01-01T00:00:00"},
            ], versions=[
                {"version": 1, "summary": "older",
                 "tags": {}, "content_hash": "h1",
                 "created_at": "2025-12-01T00:00:00"},
            ]),
        ])
        out = tmp_path / "md-both"
        out.mkdir()
        _write_markdown_export(
            keeper, out, include_system=False,
            include_parts=True, include_versions=True,
        )
        parent_meta, _ = _parse_markdown(out / "doc.md")
        assert parent_meta["_prev_version"] == "[[doc/@V{1}]]"
        assert parent_meta["_next_part"] == "[[doc/@P{1}]]"

    def test_doc_with_no_sidecars_has_no_chain_keys(self, keeper, tmp_path):
        """A plain doc records no chain-navigation frontmatter keys."""
        _seed(keeper, [_make_doc("plain", "Just a summary")])
        out = tmp_path / "md-plain-nav"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)
        meta, _ = _parse_markdown(out / "plain.md")
        assert "_prev_version" not in meta
        assert "_next_version" not in meta
        assert "_prev_part" not in meta
        assert "_next_part" not in meta

    def test_chain_links_omitted_when_flags_off(self, keeper, tmp_path):
        """Sidecars must not be linked from the parent when their flag is off."""
        _seed(keeper, [
            _make_doc("doc", "current", parts=[
                {"part_num": 1, "summary": "p1",
                 "tags": {}, "created_at": "2026-01-01T00:00:00"},
            ], versions=[
                {"version": 1, "summary": "old",
                 "tags": {}, "content_hash": None,
                 "created_at": "2025-12-01T00:00:00"},
            ]),
        ])
        out = tmp_path / "md-flags-off"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)
        meta, _ = _parse_markdown(out / "doc.md")
        assert "_prev_version" not in meta
        assert "_next_part" not in meta

    # ------------------------------------------------------------------
    # Inverse edges in the flattened frontmatter
    # ------------------------------------------------------------------

    def test_parent_inverse_edges_in_frontmatter(self, keeper, tmp_path):
        """Parent doc gets inverse edges as a multi-value frontmatter key."""
        _seed(keeper, [
            _make_doc("conv1", "First conversation"),
            _make_doc("conv2", "Second conversation"),
            _make_doc("Deborah", "About Deborah"),
        ])
        # Insert two inverse edges pointing at "Deborah" — both with
        # the inverse predicate "said" (the bundled `speaker` tag's
        # _inverse).  Bypassing the tagdoc machinery is fine here:
        # the markdown export queries the edges table directly.
        ds = keeper._document_store
        coll = keeper._resolve_doc_collection()
        ds.upsert_edge(
            coll, "conv1", "speaker", "Deborah", "said",
            "2026-01-15T10:00:00",
        )
        ds.upsert_edge(
            coll, "conv2", "speaker", "Deborah", "said",
            "2026-01-16T10:00:00",
        )

        out = tmp_path / "md-inverse"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)

        meta, body = _parse_markdown(out / "Deborah.md")
        # Inverse-edge predicate is a top-level frontmatter key with
        # the source ids as a multi-value list.  Each value is the
        # canonical ``[[id|display name]]`` form when the source has
        # a resolvable display name.
        said = meta["said"]
        assert isinstance(said, list)
        assert len(said) == 2
        # Sources are stored with canonical labeled refs.
        assert any(v == "[[conv1|First conversation]]" for v in said)
        assert any(v == "[[conv2|Second conversation]]" for v in said)

        # No body section — inverse edges live entirely in frontmatter.
        assert "## Referenced By" not in body

        # Source docs themselves don't get any inverse-edge keys —
        # nothing points at them.
        conv1_meta, _ = _parse_markdown(out / "conv1.md")
        assert "said" not in conv1_meta

    def test_inverse_edges_double_quote_wikilink_values(
        self, keeper, tmp_path,
    ):
        """Wikilink-format edge values are double-quoted in YAML output."""
        _seed(keeper, [
            _make_doc("Foo Malone", "About Foo"),
            _make_doc("contact", "Contact card"),
        ])
        ds = keeper._document_store
        coll = keeper._resolve_doc_collection()
        # Source 'Foo Malone' has a display name distinct from its id,
        # so the edge value will be ``[[Foo Malone|Foo Malone]]`` —
        # which YAML must render with double quotes for the brackets
        # to be unambiguous.
        ds.upsert_edge(
            coll, "Foo Malone", "speaker", "contact", "said",
            "2026-01-15T10:00:00",
        )

        out = tmp_path / "md-doublequote"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)

        # Read the raw frontmatter text to verify the quote style.
        text = (out / "contact.md").read_text(encoding="utf-8")
        fm_text = text.split("---", 2)[1]
        # The labeled-ref value uses double quotes, not single.
        assert '- "[[Foo Malone|About Foo]]"' in fm_text
        # And it loads back correctly.
        meta = yaml.safe_load(fm_text)
        assert meta["said"][0] == "[[Foo Malone|About Foo]]"

    def test_inverse_edges_keep_long_labeled_refs_on_one_line(
        self, keeper, tmp_path,
    ):
        """Long inverse-edge labeled refs should not wrap across YAML lines."""
        long_id = (
            "source/with/a/very/long/path/that/used/to/trigger/"
            "yaml/frontmatter/wrapping/in/export"
        )
        long_summary = (
            "A deliberately long display name for the referenced note that "
            "would exceed the default YAML wrapping width"
        )
        _seed(keeper, [
            _make_doc(long_id, long_summary),
            _make_doc("target", "Target card"),
        ])
        ds = keeper._document_store
        coll = keeper._resolve_doc_collection()
        ds.upsert_edge(
            coll, long_id, "references", "target", "referenced_by",
            "2026-01-15T10:00:00",
        )

        out = tmp_path / "md-long-inverse"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)

        text = (out / "target.md").read_text(encoding="utf-8")
        fm_text = text.split("---", 2)[1]
        meta = yaml.safe_load(fm_text)
        emitted = meta["referenced_by"][0]
        assert emitted.startswith(f"[[{long_id}|")
        assert f'- "{emitted}"' in fm_text
        assert f'- "{emitted}\\n' not in fm_text

    def test_inverse_edges_skip_system_sources(self, keeper, tmp_path):
        """When include_system=False, system-id edge sources are dropped."""
        _seed(keeper, [_make_doc("target", "Target body")])
        ds = keeper._document_store
        coll = keeper._resolve_doc_collection()
        ds.upsert_edge(
            coll, ".meta/something", "speaker", "target", "said",
            "2026-01-15T10:00:00",
        )
        ds.upsert_edge(
            coll, "real-source", "speaker", "target", "said",
            "2026-01-16T10:00:00",
        )

        out = tmp_path / "md-sys-filter"
        out.mkdir()
        _write_markdown_export(keeper, out, include_system=False)

        meta, _ = _parse_markdown(out / "target.md")
        said = meta["said"]
        # Only the non-system source survives.
        assert any(v == "[[real-source]]" for v in said)
        assert not any(".meta/something" in v for v in said)

    def test_version_sidecar_inverse_edges_in_frontmatter(
        self, keeper, tmp_path,
    ):
        """Version sidecars expose inverse edges in frontmatter, not body."""
        # Seed both the target doc (with archived versions) and a
        # source doc with versions whose tags reference the target.
        # `import_batch` calls _rebuild_version_edges_for_source so
        # version_edges gets populated automatically — but only if a
        # tagdoc with _inverse exists at import time, so include one
        # in the same batch.
        _seed(keeper, [
            {
                "id": ".tag/speaker",
                "summary": "speaker tagdoc",
                "tags": {"_inverse": "said"},
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
                "accessed_at": "2025-01-01T00:00:00",
            },
            _make_doc("target", "Target body", versions=[
                {"version": 1, "summary": "older target body",
                 "tags": {}, "content_hash": "ht1",
                 "created_at": "2025-12-01T00:00:00"},
            ]),
            _make_doc(
                "src",
                "current source",
                tags={"speaker": "target"},
                versions=[
                    {"version": 1,
                     "summary": "older source body",
                     "tags": {"speaker": "target"},
                     "content_hash": "hs1",
                     "created_at": "2025-12-01T00:00:00"},
                ],
            ),
        ])

        out = tmp_path / "md-version-inverse"
        out.mkdir()
        _write_markdown_export(
            keeper, out, include_system=False, include_versions=True,
        )

        # The single archived version sidecar of `target` should
        # surface the historical reference from `src` (through
        # version_edges) as a flat top-level frontmatter key.
        meta, body = _parse_markdown(out / "target" / "@V{1}.md")
        said = meta["said"]
        assert isinstance(said, list)
        assert any(v == "[[src|current source]]" for v in said)
        # No body section.
        assert "## Referenced By" not in body
