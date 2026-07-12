"""Tests for the document store module."""

import json
import logging
import sqlite3
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from keep.document_store import DocumentStore, DocumentRecord, _load_tags_json


class TestSchemaCompatibility:
    """Schema compatibility guards."""

    def test_rejects_store_from_future(self) -> None:
        """Opening a DB with newer user_version fails fast."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path):
                pass
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("PRAGMA user_version = 999")
                conn.commit()
            finally:
                conn.close()

            with pytest.raises(sqlite3.DatabaseError, match="newer than supported"):
                DocumentStore(db_path)


class TestLoadTagsJsonHelper:
    """Defensive decoding for stored ``tags_json`` values."""

    def test_returns_empty_for_none(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="keep.document_store")
        assert _load_tags_json(None, doc_id=".state/put", collection="default") == {}
        assert "is empty for default/.state/put" in caplog.text

    def test_returns_empty_for_blank_string(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="keep.document_store")
        assert _load_tags_json("", doc_id="x") == {}
        assert "is empty" in caplog.text

    def test_returns_empty_for_invalid_json(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="keep.document_store")
        assert _load_tags_json("{not-json", doc_id="x") == {}
        assert "invalid JSON" in caplog.text

    def test_returns_empty_for_non_object(self, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="keep.document_store")
        assert _load_tags_json("[1, 2, 3]", doc_id="x") == {}
        assert "is not an object" in caplog.text

    def test_returns_decoded_object(self) -> None:
        assert _load_tags_json('{"topic": "auth"}', doc_id="x") == {"topic": "auth"}

    def test_get_tolerates_null_tags_json_in_storage(self, tmp_path, caplog) -> None:
        """A NULL tags_json column should not break DocumentStore.get.

        Schema enforcement won't let us write NULL through the public API, so
        construct a documents table without the NOT NULL constraint and run
        DocumentStore over the resulting database.
        """
        db_path = tmp_path / "tolerant.db"

        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(
                """
                CREATE TABLE documents (
                    id TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    tags_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    content_hash TEXT,
                    content_hash_full TEXT,
                    PRIMARY KEY (id, collection)
                );
                INSERT INTO documents
                    (id, collection, summary, tags_json, created_at, updated_at)
                VALUES
                    ('.state/put', 'default', 'summary', NULL,
                     '2026-04-28T00:00:00', '2026-04-28T00:00:00');
                """
            )
            conn.commit()
        finally:
            conn.close()

        caplog.set_level(logging.WARNING, logger="keep.document_store")
        with DocumentStore(db_path) as store:
            record = store.get("default", ".state/put")

        assert record is not None
        assert record.tags == {}
        assert "is empty for default/.state/put" in caplog.text


class TestDocumentStoreBasics:
    """Basic CRUD operations."""
    
    @pytest.fixture
    def store(self):
        """Create a temporary document store."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store
    
    def test_upsert_and_get(self, store: DocumentStore) -> None:
        """upsert() stores a document, get() retrieves it."""
        record, _ = store.upsert(
            collection="default",
            id="doc:1",
            summary="Test summary",
            tags={"topic": "testing"},
        )

        assert record.id == "doc:1"
        assert record.summary == "Test summary"
        assert record.tags == {"topic": "testing"}

        retrieved = store.get("default", "doc:1")
        assert retrieved is not None
        assert retrieved.id == "doc:1"
        assert retrieved.summary == "Test summary"
        assert retrieved.tags == {"topic": "testing"}
    
    def test_get_not_found(self, store: DocumentStore) -> None:
        """get() returns None for non-existent documents."""
        result = store.get("default", "nonexistent")
        assert result is None
    
    def test_exists(self, store: DocumentStore) -> None:
        """exists() returns correct boolean."""
        assert store.exists("default", "doc:1") is False
        
        store.upsert("default", "doc:1", "Summary", {})
        
        assert store.exists("default", "doc:1") is True
    
    def test_delete(self, store: DocumentStore) -> None:
        """delete() removes documents."""
        store.upsert("default", "doc:1", "Summary", {})
        assert store.exists("default", "doc:1") is True
        
        deleted = store.delete("default", "doc:1")
        assert deleted is True
        assert store.exists("default", "doc:1") is False
    
    def test_delete_not_found(self, store: DocumentStore) -> None:
        """delete() returns False for non-existent documents."""
        deleted = store.delete("default", "nonexistent")
        assert deleted is False
    
    def test_upsert_updates_existing(self, store: DocumentStore) -> None:
        """upsert() updates existing documents."""
        store.upsert("default", "doc:1", "Original", {"v": "1"})
        store.upsert("default", "doc:1", "Updated", {"v": "2"})
        
        doc = store.get("default", "doc:1")
        assert doc.summary == "Updated"
        assert doc.tags == {"v": "2"}


class TestTimestamps:
    """Timestamp handling."""
    
    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store
    
    def test_created_at_set_on_insert(self, store: DocumentStore) -> None:
        """created_at is set when first inserted."""
        record, _ = store.upsert("default", "doc:1", "Summary", {})
        assert record.created_at is not None
        assert "T" in record.created_at  # ISO format

    def test_updated_at_set_on_insert(self, store: DocumentStore) -> None:
        """updated_at is set when first inserted."""
        record, _ = store.upsert("default", "doc:1", "Summary", {})
        assert record.updated_at is not None

    def test_created_at_preserved_on_update(self, store: DocumentStore) -> None:
        """created_at is preserved when updated."""
        store._now = lambda: "2026-01-01T00:00:00"
        original, _ = store.upsert("default", "doc:1", "Original", {})
        original_created = original.created_at

        store._now = lambda: "2026-01-01T00:00:05"
        updated, _ = store.upsert("default", "doc:1", "Updated", {})

        assert updated.created_at == original_created
        assert updated.updated_at != original_created


class TestBatchOperations:
    """Batch and collection operations."""
    
    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store
    
    def test_get_many(self, store: DocumentStore) -> None:
        """get_many() retrieves multiple documents."""
        store.upsert("default", "doc:1", "Summary 1", {})
        store.upsert("default", "doc:2", "Summary 2", {})
        store.upsert("default", "doc:3", "Summary 3", {})
        
        results = store.get_many("default", ["doc:1", "doc:3"])
        
        assert len(results) == 2
        assert "doc:1" in results
        assert "doc:3" in results
        assert "doc:2" not in results
    
    def test_get_many_missing_ids(self, store: DocumentStore) -> None:
        """get_many() omits missing IDs."""
        store.upsert("default", "doc:1", "Summary", {})
        
        results = store.get_many("default", ["doc:1", "nonexistent"])
        
        assert len(results) == 1
        assert "doc:1" in results
    
    def test_list_ids(self, store: DocumentStore) -> None:
        """list_ids() returns all document IDs."""
        store.upsert("default", "doc:1", "Summary 1", {})
        store.upsert("default", "doc:2", "Summary 2", {})
        
        ids = store.list_ids("default")
        
        assert set(ids) == {"doc:1", "doc:2"}
    
    def test_list_ids_with_limit(self, store: DocumentStore) -> None:
        """list_ids() respects limit."""
        for i in range(10):
            store.upsert("default", f"doc:{i}", f"Summary {i}", {})
        
        ids = store.list_ids("default", limit=3)
        
        assert len(ids) == 3
    
    def test_count(self, store: DocumentStore) -> None:
        """count() returns correct document count."""
        assert store.count("default") == 0
        
        store.upsert("default", "doc:1", "Summary", {})
        store.upsert("default", "doc:2", "Summary", {})
        
        assert store.count("default") == 2
    
    def test_count_all(self, store: DocumentStore) -> None:
        """count_all() counts across collections."""
        store.upsert("coll1", "doc:1", "Summary", {})
        store.upsert("coll2", "doc:2", "Summary", {})
        
        assert store.count_all() == 2


class TestCollectionManagement:
    """Collection operations."""
    
    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store
    
    def test_list_collections_empty(self, store: DocumentStore) -> None:
        """list_collections() returns empty for new store."""
        assert store.list_collections() == []
    
    def test_list_collections(self, store: DocumentStore) -> None:
        """list_collections() returns all collection names."""
        store.upsert("alpha", "doc:1", "Summary", {})
        store.upsert("beta", "doc:2", "Summary", {})
        
        collections = store.list_collections()
        
        assert set(collections) == {"alpha", "beta"}
    
    def test_delete_collection(self, store: DocumentStore) -> None:
        """delete_collection() removes all documents."""
        store.upsert("default", "doc:1", "Summary 1", {})
        store.upsert("default", "doc:2", "Summary 2", {})
        
        deleted = store.delete_collection("default")
        
        assert deleted == 2
        assert store.count("default") == 0


class TestUpdateOperations:
    """Partial update operations."""
    
    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store
    
    def test_update_summary(self, store: DocumentStore) -> None:
        """update_summary() updates just the summary."""
        store.upsert("default", "doc:1", "Original summary", {"tag": "value"})
        
        updated = store.update_summary("default", "doc:1", "New summary")
        
        assert updated is True
        doc = store.get("default", "doc:1")
        assert doc.summary == "New summary"
        assert doc.tags == {"tag": "value"}  # Tags preserved
    
    def test_update_summary_not_found(self, store: DocumentStore) -> None:
        """update_summary() returns False for missing document."""
        updated = store.update_summary("default", "nonexistent", "New")
        assert updated is False
    
    def test_update_tags(self, store: DocumentStore) -> None:
        """update_tags() updates just the tags."""
        store.upsert("default", "doc:1", "Summary", {"old": "tags"})
        
        updated = store.update_tags("default", "doc:1", {"new": "tags"})
        
        assert updated is True
        doc = store.get("default", "doc:1")
        assert doc.summary == "Summary"  # Summary preserved
        assert doc.tags == {"new": "tags"}
    
    def test_update_tags_not_found(self, store: DocumentStore) -> None:
        """update_tags() returns False for missing document."""
        updated = store.update_tags("default", "nonexistent", {})
        assert updated is False


class TestTagDedupOnWritePaths:
    """Tag values are deduplicated on all document-level write paths."""

    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store

    def test_upsert_deduplicates_tag_values(self, store: DocumentStore) -> None:
        """upsert() stores scalar-or-list tags without duplicate values."""
        store.upsert(
            "default",
            "doc:1",
            "Summary",
            {"k": ["v1", "v1", "v2"], "single": ["x", "x"]},
        )

        doc = store.get("default", "doc:1")
        assert doc is not None
        assert doc.tags == {"k": ["v1", "v2"], "single": "x"}

    def test_update_tags_deduplicates_tag_values(self, store: DocumentStore) -> None:
        """update_tags() also deduplicates values before persisting."""
        store.upsert("default", "doc:1", "Summary", {"k": "v1"})

        updated = store.update_tags(
            "default",
            "doc:1",
            {"k": ["v1", "v1", "v2"], "single": ["x", "x"]},
        )
        assert updated is True

        doc = store.get("default", "doc:1")
        assert doc is not None
        assert doc.tags == {"k": ["v1", "v2"], "single": "x"}

    def test_restore_latest_version_deduplicates_tags(self, store: DocumentStore) -> None:
        """restore_latest_version() normalizes tags from version rows."""
        store.upsert("default", "doc:1", "Current", {"status": "current"})
        store._execute(
            """
            INSERT INTO document_versions
                (id, collection, version, summary, tags_json, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "doc:1",
                "default",
                1,
                "Archived",
                json.dumps({"k": ["v1", "v1", "v2"], "single": ["x", "x"]}),
                None,
                "2026-01-01T00:00:00",
            ),
        )
        store._conn.commit()

        restored = store.restore_latest_version("default", "doc:1")
        assert restored is not None
        assert restored.tags == {"k": ["v1", "v2"], "single": "x"}

        doc = store.get("default", "doc:1")
        assert doc is not None
        assert doc.tags == {"k": ["v1", "v2"], "single": "x"}

    def test_import_batch_deduplicates_tags_everywhere(self, store: DocumentStore) -> None:
        """import_batch() deduplicates document/version/part tag values."""
        stats = store.import_batch(
            "default",
            [{
                "id": "doc:1",
                "summary": "Summary",
                "tags": {"k": ["v1", "v1", "v2"]},
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-02T00:00:00",
                "versions": [{
                    "version": 1,
                    "summary": "V1",
                    "tags": {"k": ["a", "a", "b"]},
                    "content_hash": None,
                    "created_at": "2026-01-01T00:00:00",
                }],
                "parts": [{
                    "part_num": 1,
                    "summary": "P1",
                    "tags": {"k": ["p", "p", "q"]},
                    "created_at": "2026-01-02T00:00:00",
                }],
            }],
        )
        assert stats == {"documents": 1, "versions": 1, "parts": 1}

        doc = store.get("default", "doc:1")
        assert doc is not None
        assert doc.tags == {"k": ["v1", "v2"]}

        versions = store.list_versions("default", "doc:1")
        assert len(versions) == 1
        assert versions[0].tags == {"k": ["a", "b"]}

        parts = store.list_parts("default", "doc:1")
        assert len(parts) == 1
        assert parts[0].tags == {"k": ["p", "q"]}

class TestCollectionIsolation:
    """Documents in different collections are isolated."""
    
    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store
    
    def test_same_id_different_collections(self, store: DocumentStore) -> None:
        """Same ID in different collections are separate documents."""
        store.upsert("coll1", "doc:1", "In collection 1", {})
        store.upsert("coll2", "doc:1", "In collection 2", {})
        
        doc1 = store.get("coll1", "doc:1")
        doc2 = store.get("coll2", "doc:1")
        
        assert doc1.summary == "In collection 1"
        assert doc2.summary == "In collection 2"
    
    def test_delete_does_not_affect_other_collections(self, store: DocumentStore) -> None:
        """Delete in one collection doesn't affect others."""
        store.upsert("coll1", "doc:1", "Summary", {})
        store.upsert("coll2", "doc:1", "Summary", {})

        store.delete("coll1", "doc:1")

        assert store.exists("coll1", "doc:1") is False
        assert store.exists("coll2", "doc:1") is True


class TestVersioning:
    """Document versioning tests."""

    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store

    def test_upsert_creates_version_on_content_change(self, store: DocumentStore) -> None:
        """upsert() archives current version when content changes."""
        # First insert
        store.upsert("default", "doc:1", "Version 1", {"tag": "a"}, content_hash="hash1")

        # Update with different content hash
        store.upsert("default", "doc:1", "Version 2", {"tag": "b"}, content_hash="hash2")

        # Check version was archived
        versions = store.list_versions("default", "doc:1")
        assert len(versions) == 1
        assert versions[0].summary == "Version 1"
        # Version tags include injected _created/_updated for frontmatter display
        assert versions[0].tags["tag"] == "a"

        # Current should be updated
        current = store.get("default", "doc:1")
        assert current.summary == "Version 2"

    def test_upsert_creates_version_on_tag_change_same_hash(self, store: DocumentStore) -> None:
        """upsert() creates version even when only tags change (same content hash)."""
        # First insert
        store.upsert("default", "doc:1", "Content", {"status": "draft"}, content_hash="hash1")

        # Update with same content hash but different tags
        store.upsert("default", "doc:1", "Content", {"status": "done"}, content_hash="hash1")

        # Version should still be created (for tag history)
        versions = store.list_versions("default", "doc:1")
        assert len(versions) == 1
        assert versions[0].tags["status"] == "draft"

        # Current should have new tags
        current = store.get("default", "doc:1")
        assert current.tags["status"] == "done"

    def test_upsert_returns_content_changed_flag(self, store: DocumentStore) -> None:
        """upsert() returns tuple with content_changed flag."""
        # First insert - no previous content
        _, content_changed = store.upsert("default", "doc:1", "V1", {}, content_hash="hash1")
        assert content_changed is False  # No previous version

        # Same hash - no content change
        _, content_changed = store.upsert("default", "doc:1", "V1", {"tag": "new"}, content_hash="hash1")
        assert content_changed is False

        # Different hash - content changed
        _, content_changed = store.upsert("default", "doc:1", "V2", {}, content_hash="hash2")
        assert content_changed is True

    def test_get_version_current_returns_none(self, store: DocumentStore) -> None:
        """get_version() with offset=0 returns None (use get() instead)."""
        store.upsert("default", "doc:1", "Content", {})
        result = store.get_version("default", "doc:1", offset=0)
        assert result is None

    def test_get_version_previous(self, store: DocumentStore) -> None:
        """get_version() retrieves previous versions by offset."""
        # Create history
        store.upsert("default", "doc:1", "V1", {}, content_hash="h1")
        store.upsert("default", "doc:1", "V2", {}, content_hash="h2")
        store.upsert("default", "doc:1", "V3", {}, content_hash="h3")

        # offset=1 = previous (V2)
        v = store.get_version("default", "doc:1", offset=1)
        assert v is not None
        assert v.summary == "V2"

        # offset=2 = two ago (V1)
        v = store.get_version("default", "doc:1", offset=2)
        assert v is not None
        assert v.summary == "V1"

        # offset=3 = doesn't exist
        v = store.get_version("default", "doc:1", offset=3)
        assert v is None

    def test_list_versions(self, store: DocumentStore) -> None:
        """list_versions() returns versions newest first."""
        store.upsert("default", "doc:1", "V1", {}, content_hash="h1")
        store.upsert("default", "doc:1", "V2", {}, content_hash="h2")
        store.upsert("default", "doc:1", "V3", {}, content_hash="h3")

        versions = store.list_versions("default", "doc:1")
        assert len(versions) == 2  # V1 and V2 archived, V3 is current
        assert versions[0].summary == "V2"  # Newest archived first
        assert versions[1].summary == "V1"

    def test_list_versions_with_limit(self, store: DocumentStore) -> None:
        """list_versions() respects limit."""
        for i in range(5):
            store.upsert("default", "doc:1", f"V{i+1}", {}, content_hash=f"h{i}")

        versions = store.list_versions("default", "doc:1", limit=2)
        assert len(versions) == 2

    def test_get_version_nav_for_current(self, store: DocumentStore) -> None:
        """get_version_nav() returns prev list when viewing current."""
        store.upsert("default", "doc:1", "V1", {}, content_hash="h1")
        store.upsert("default", "doc:1", "V2", {}, content_hash="h2")
        store.upsert("default", "doc:1", "V3", {}, content_hash="h3")

        nav = store.get_version_nav("default", "doc:1", current_version=None)
        assert "prev" in nav
        assert len(nav["prev"]) == 2
        assert "next" not in nav or nav.get("next") == []

    def test_get_version_nav_for_old_version(self, store: DocumentStore) -> None:
        """get_version_nav() returns prev and next when viewing old version."""
        store.upsert("default", "doc:1", "V1", {}, content_hash="h1")
        store.upsert("default", "doc:1", "V2", {}, content_hash="h2")
        store.upsert("default", "doc:1", "V3", {}, content_hash="h3")

        # Viewing version 1 (oldest archived)
        nav = store.get_version_nav("default", "doc:1", current_version=1)
        assert nav["prev"] == []  # No older versions
        assert len(nav.get("next", [])) == 1  # V2 is next

        # Viewing version 2 (has both prev and next)
        nav = store.get_version_nav("default", "doc:1", current_version=2)
        assert len(nav["prev"]) == 1  # V1 is prev
        # next is empty list meaning "current is next"
        assert "next" in nav

    def test_version_count(self, store: DocumentStore) -> None:
        """version_count() returns correct count."""
        assert store.version_count("default", "doc:1") == 0

        store.upsert("default", "doc:1", "V1", {}, content_hash="h1")
        assert store.version_count("default", "doc:1") == 0  # No archived yet

        store.upsert("default", "doc:1", "V2", {}, content_hash="h2")
        assert store.version_count("default", "doc:1") == 1

        store.upsert("default", "doc:1", "V3", {}, content_hash="h3")
        assert store.version_count("default", "doc:1") == 2

    def test_delete_removes_versions(self, store: DocumentStore) -> None:
        """delete() removes version history by default."""
        store.upsert("default", "doc:1", "V1", {}, content_hash="h1")
        store.upsert("default", "doc:1", "V2", {}, content_hash="h2")

        store.delete("default", "doc:1")

        assert store.version_count("default", "doc:1") == 0
        assert store.list_versions("default", "doc:1") == []

    def test_delete_preserves_versions_when_requested(self, store: DocumentStore) -> None:
        """delete(delete_versions=False) preserves history."""
        store.upsert("default", "doc:1", "V1", {}, content_hash="h1")
        store.upsert("default", "doc:1", "V2", {}, content_hash="h2")

        store.delete("default", "doc:1", delete_versions=False)

        # Current deleted but versions preserved
        assert store.get("default", "doc:1") is None
        assert store.version_count("default", "doc:1") == 1

    def test_first_upsert_no_version(self, store: DocumentStore) -> None:
        """First upsert doesn't create a version (nothing to archive)."""
        store.upsert("default", "doc:1", "First", {}, content_hash="h1")

        versions = store.list_versions("default", "doc:1")
        assert len(versions) == 0


class TestSystemDocVersioning:
    """System doc versioning: upgrade, reset, and delete behavior."""

    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store

    def test_upsert_archive_false_no_version_created(self, store: DocumentStore) -> None:
        """upsert(archive=False) updates head without creating a version."""
        store.upsert("default", "doc:1", "V1", {"t": "a"}, content_hash="h1")
        store.upsert("default", "doc:1", "V2", {"t": "b"}, content_hash="h2", archive=False)

        # No version archived
        assert store.version_count("default", "doc:1") == 0
        # Head updated
        current = store.get("default", "doc:1")
        assert current.summary == "V2"
        assert current.tags["t"] == "b"

    def test_find_version_by_content_hash(self, store: DocumentStore) -> None:
        """find_version_by_content_hash returns oldest matching version."""
        store.upsert("default", "doc:1", "V1", {}, content_hash="base_hash")
        store.upsert("default", "doc:1", "V2", {}, content_hash="user_hash")

        ver = store.find_version_by_content_hash("default", "doc:1", "base_hash")
        assert ver == 1

        # Not found
        ver = store.find_version_by_content_hash("default", "doc:1", "nonexistent")
        assert ver is None

    def test_replace_version_content(self, store: DocumentStore) -> None:
        """replace_version_content updates an archived version in-place."""
        store.upsert("default", "doc:1", "bundled-v1", {"bundled_hash": "bh1"}, content_hash="bh1")
        store.upsert("default", "doc:1", "user-edit", {"bundled_hash": "bh1"}, content_hash="uh1")

        # V1 in archive = bundled-v1
        v1 = store.get_version("default", "doc:1", offset=1)
        assert v1.summary == "bundled-v1"
        assert v1.content_hash == "bh1"

        # Replace V1 with new bundled content
        ok = store.replace_version_content(
            "default", "doc:1", v1.version,
            summary="bundled-v2", tags={"bundled_hash": "bh2"},
            content_hash="bh2",
        )
        assert ok is True

        # Verify replaced
        v1_updated = store.get_version("default", "doc:1", offset=1)
        assert v1_updated.summary == "bundled-v2"
        assert v1_updated.content_hash == "bh2"

        # Head unchanged
        current = store.get("default", "doc:1")
        assert current.summary == "user-edit"

    def test_delete_all_versions(self, store: DocumentStore) -> None:
        """delete_all_versions removes all archived versions."""
        store.upsert("default", "doc:1", "V1", {}, content_hash="h1")
        store.upsert("default", "doc:1", "V2", {}, content_hash="h2")
        store.upsert("default", "doc:1", "V3", {}, content_hash="h3")
        assert store.version_count("default", "doc:1") == 2

        n = store.delete_all_versions("default", "doc:1")
        assert n == 2
        assert store.version_count("default", "doc:1") == 0

        # Head still exists
        current = store.get("default", "doc:1")
        assert current.summary == "V3"

    def test_patch_head_tags(self, store: DocumentStore) -> None:
        """patch_head_tags merges tags without creating a version."""
        store.upsert("default", "doc:1", "content", {"a": "1", "bundled_hash": "old"})

        ok = store.patch_head_tags("default", "doc:1", {"bundled_hash": "new"})
        assert ok is True

        current = store.get("default", "doc:1")
        assert current.tags["bundled_hash"] == "new"
        assert current.tags["a"] == "1"  # preserved
        assert current.summary == "content"  # unchanged
        assert store.version_count("default", "doc:1") == 0  # no version created

    def test_upgrade_no_user_edit_updates_head_in_place(self, store: DocumentStore) -> None:
        """Simulates upgrade when user hasn't edited: head updated, no version."""
        # Initial system doc
        store.upsert("default", ".state/foo", "bundled-v1",
                      {"category": "system", "bundled_hash": "bh1"},
                      content_hash="bh1")

        # Upgrade: no user edit (content_hash == bundled_hash), use archive=False
        store.upsert("default", ".state/foo", "bundled-v2",
                      {"category": "system", "bundled_hash": "bh2"},
                      content_hash="bh2", archive=False)

        current = store.get("default", ".state/foo")
        assert current.summary == "bundled-v2"
        assert store.version_count("default", ".state/foo") == 0

    def test_upgrade_with_user_edit_updates_base_version(self, store: DocumentStore) -> None:
        """Simulates upgrade when user has customized: base version updated."""
        # Initial system doc
        store.upsert("default", ".state/foo", "bundled-v1",
                      {"category": "system", "bundled_hash": "bh1"},
                      content_hash="bh1")

        # User edits → archives bundled-v1 as V1, head = user content
        store.upsert("default", ".state/foo", "my-custom-rules",
                      {"category": "system", "bundled_hash": "bh1"},
                      content_hash="user1")

        # Upgrade: find base version and replace it
        base_ver = store.find_version_by_content_hash("default", ".state/foo", "bh1")
        assert base_ver is not None
        store.replace_version_content(
            "default", ".state/foo", base_ver,
            summary="bundled-v2",
            tags={"category": "system", "bundled_hash": "bh2"},
            content_hash="bh2",
        )
        store.patch_head_tags("default", ".state/foo", {"bundled_hash": "bh2"})

        # Head unchanged (user's content)
        current = store.get("default", ".state/foo")
        assert current.summary == "my-custom-rules"
        assert current.tags["bundled_hash"] == "bh2"  # updated to track new base

        # Base version updated
        v1 = store.get_version("default", ".state/foo", offset=1)
        assert v1.summary == "bundled-v2"
        assert v1.content_hash == "bh2"

    def test_revert_after_upgrade_restores_current_bundled(self, store: DocumentStore) -> None:
        """After upgrade updates base, reverting user edit restores current bundled."""
        # Setup: system doc → user edit → upgrade base
        store.upsert("default", ".state/foo", "bundled-v1", {}, content_hash="bh1")
        store.upsert("default", ".state/foo", "user-edit", {}, content_hash="uh1")
        base_ver = store.find_version_by_content_hash("default", ".state/foo", "bh1")
        store.replace_version_content(
            "default", ".state/foo", base_ver,
            summary="bundled-v2", tags={}, content_hash="bh2",
        )

        # Revert user edit → should restore bundled-v2 (not stale bundled-v1)
        restored = store.restore_latest_version("default", ".state/foo")
        assert restored.summary == "bundled-v2"

    def test_reset_clears_versions_and_restores_head(self, store: DocumentStore) -> None:
        """Simulates reset: all versions cleared, head = fresh bundled."""
        store.upsert("default", ".state/foo", "bundled-v1", {}, content_hash="bh1")
        store.upsert("default", ".state/foo", "user-edit-1", {}, content_hash="uh1")
        store.upsert("default", ".state/foo", "user-edit-2", {}, content_hash="uh2")
        assert store.version_count("default", ".state/foo") == 2

        # Reset
        store.delete_all_versions("default", ".state/foo")
        store.upsert("default", ".state/foo", "bundled-v2",
                      {"category": "system", "bundled_hash": "bh2"},
                      content_hash="bh2", archive=False)

        assert store.version_count("default", ".state/foo") == 0
        current = store.get("default", ".state/foo")
        assert current.summary == "bundled-v2"


class TestTagQueries:
    """Tag-value query behavior."""

    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store

    def test_query_by_tag_value_matches_scalar_tags(self, store: DocumentStore) -> None:
        """query_by_tag_value() matches scalar tag values without JSON errors."""
        store.upsert(
            "default",
            ".state/foo",
            "bundled",
            {"category": "system", "bundled_hash": "bh1"},
        )
        store.upsert(
            "default",
            "doc:1",
            "user",
            {"category": "user"},
        )

        docs = store.query_by_tag_value("default", "category", "system")

        assert [doc.id for doc in docs] == [".state/foo"]


class TestAccessedAt:
    """Last-accessed timestamp tracking."""

    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store

    def test_upsert_sets_accessed_at(self, store: DocumentStore) -> None:
        """upsert() sets accessed_at alongside updated_at."""
        record, _ = store.upsert("default", "doc:1", "Summary", {})
        assert record.accessed_at is not None
        assert record.accessed_at == record.updated_at

    def test_touch_updates_accessed_at(self, store: DocumentStore) -> None:
        """touch() updates accessed_at without changing updated_at."""
        store._now = lambda: "2026-01-01T00:00:00"
        record, _ = store.upsert("default", "doc:1", "Summary", {})
        original_updated = record.updated_at

        store._now = lambda: "2026-01-01T00:00:05"
        store.touch("default", "doc:1")
        doc = store.get("default", "doc:1")

        assert doc.updated_at == original_updated  # Unchanged
        assert doc.accessed_at > original_updated   # Bumped

    def test_touch_many(self, store: DocumentStore) -> None:
        """touch_many() updates accessed_at for multiple docs."""
        store._now = lambda: "2026-01-01T00:00:00"
        store.upsert("default", "doc:1", "S1", {})
        store.upsert("default", "doc:2", "S2", {})
        store.upsert("default", "doc:3", "S3", {})

        store._now = lambda: "2026-01-01T00:00:05"
        store.touch_many("default", ["doc:1", "doc:3"])

        d1 = store.get("default", "doc:1")
        d2 = store.get("default", "doc:2")
        d3 = store.get("default", "doc:3")

        # doc:1 and doc:3 should have newer accessed_at than updated_at
        assert d1.accessed_at > d1.updated_at
        assert d3.accessed_at > d3.updated_at
        # doc:2 should be unchanged
        assert d2.accessed_at == d2.updated_at

    def test_list_recent_order_by_accessed(self, store: DocumentStore) -> None:
        """list_recent(order_by='accessed') sorts by accessed_at."""
        store._now = lambda: "2026-01-01T00:00:00"
        store.upsert("default", "doc:1", "First", {})

        store._now = lambda: "2026-01-01T00:00:01"
        store.upsert("default", "doc:2", "Second", {})

        store._now = lambda: "2026-01-01T00:00:02"
        # Touch doc:1 so it has newer accessed_at than doc:2
        store.touch("default", "doc:1")

        # Default order (updated) should put doc:2 first
        by_updated = store.list_recent("default", order_by="updated")
        assert by_updated[0].id == "doc:2"

        # Access order should put doc:1 first (just touched)
        by_accessed = store.list_recent("default", order_by="accessed")
        assert by_accessed[0].id == "doc:1"

    def test_touch_many_empty_ids(self, store: DocumentStore) -> None:
        """touch_many() with empty list is a no-op."""
        store.touch_many("default", [])  # Should not raise


class TestStopwordOverrides:
    """Stopwords should come from the `.stop` store note."""

    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store

    def test_dot_stop_overrides_default_list(self, store: DocumentStore) -> None:
        store.upsert("default", ".stop", "how\nmany\nhikes", {})
        query = store._build_fts_query("how many hikes today")
        assert query == '"today"'

    def test_dot_stopwords_legacy_note_is_ignored(self, store: DocumentStore) -> None:
        store.upsert("default", ".stopwords", "today\nhiking", {})
        query = store._build_fts_query("today hiking")
        assert query is not None
        assert '"today"' in query
        assert '"hiking"' in query

    def test_stopword_cache_invalidates_when_dot_stop_changes(self, store: DocumentStore) -> None:
        store.upsert("default", ".stop", "foo", {})
        first = store._build_fts_query("foo bar")
        assert first == '"bar"'

        store.upsert("default", ".stop", "bar", {})
        second = store._build_fts_query("foo bar")
        assert second == '"foo"'


class TestFindByName:
    """Vault-wide name-based lookup."""

    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store

    def test_find_by_stem_with_md(self, store: DocumentStore) -> None:
        store.upsert("default", "file:///vault/notes/Foo.md", "Foo note", {})
        store.upsert("default", "file:///vault/other/Bar.md", "Bar note", {})
        results = store.find_by_name("default", "Foo")
        assert len(results) == 1
        assert results[0].id == "file:///vault/notes/Foo.md"

    def test_find_by_stem_without_md(self, store: DocumentStore) -> None:
        store.upsert("default", "file:///vault/notes/Foo", "Foo note", {})
        results = store.find_by_name("default", "Foo")
        assert len(results) == 1
        assert results[0].id == "file:///vault/notes/Foo"

    def test_scoped_by_prefix(self, store: DocumentStore) -> None:
        store.upsert("default", "file:///vault1/Foo.md", "Foo v1", {})
        store.upsert("default", "file:///vault2/Foo.md", "Foo v2", {})
        results = store.find_by_name(
            "default", "Foo", id_prefix="file:///vault1",
        )
        assert len(results) == 1
        assert results[0].id == "file:///vault1/Foo.md"

    def test_shortest_path_first(self, store: DocumentStore) -> None:
        store.upsert("default", "file:///vault/deep/nested/Foo.md", "deep", {})
        store.upsert("default", "file:///vault/Foo.md", "shallow", {})
        results = store.find_by_name("default", "Foo")
        assert results[0].id == "file:///vault/Foo.md"

    def test_no_match(self, store: DocumentStore) -> None:
        store.upsert("default", "file:///vault/Bar.md", "Bar", {})
        results = store.find_by_name("default", "Foo")
        assert results == []

    def test_no_partial_match(self, store: DocumentStore) -> None:
        """'Foo' should not match 'MyFoo.md'."""
        store.upsert("default", "file:///vault/MyFoo.md", "MyFoo", {})
        results = store.find_by_name("default", "Foo")
        assert results == []

# ---------------------------------------------------------------------------


class TestDistinctTagQueries:
    """Tests for list_distinct_tag_keys and list_distinct_tag_values using json_each."""

    @pytest.fixture
    def store(self):
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store

    def test_list_distinct_tag_keys_basic(self, store: DocumentStore) -> None:
        """Returns all user tag keys, sorted."""
        store.upsert("default", "d1", "S1", {"topic": "auth", "project": "web"})
        store.upsert("default", "d2", "S2", {"topic": "db", "status": "open"})

        keys = store.list_distinct_tag_keys("default")
        assert keys == ["project", "status", "topic"]

    def test_list_distinct_tag_keys_excludes_system(self, store: DocumentStore) -> None:
        """System tags (prefixed with _) are excluded."""
        store.upsert("default", "d1", "S1", {
            "topic": "auth",
            "_created": "2026-01-01",
            "_source": "inline",
        })

        keys = store.list_distinct_tag_keys("default")
        assert keys == ["topic"]
        assert "_created" not in keys
        assert "_source" not in keys

    def test_list_distinct_tag_keys_empty_collection(self, store: DocumentStore) -> None:
        """Empty collection returns empty list."""
        keys = store.list_distinct_tag_keys("default")
        assert keys == []

    def test_list_distinct_tag_keys_no_duplicates(self, store: DocumentStore) -> None:
        """Same key across multiple documents appears once."""
        store.upsert("default", "d1", "S1", {"topic": "a"})
        store.upsert("default", "d2", "S2", {"topic": "b"})
        store.upsert("default", "d3", "S3", {"topic": "c"})

        keys = store.list_distinct_tag_keys("default")
        assert keys.count("topic") == 1

    def test_list_distinct_tag_keys_collection_isolation(self, store: DocumentStore) -> None:
        """Keys from other collections are not included."""
        store.upsert("coll1", "d1", "S1", {"alpha": "1"})
        store.upsert("coll2", "d2", "S2", {"beta": "2"})

        assert store.list_distinct_tag_keys("coll1") == ["alpha"]
        assert store.list_distinct_tag_keys("coll2") == ["beta"]

    def test_list_distinct_tag_values_basic(self, store: DocumentStore) -> None:
        """Returns all distinct values for a key, sorted."""
        store.upsert("default", "d1", "S1", {"topic": "auth"})
        store.upsert("default", "d2", "S2", {"topic": "db"})
        store.upsert("default", "d3", "S3", {"topic": "auth"})  # duplicate

        values = store.list_distinct_tag_values("default", "topic")
        assert values == ["auth", "db"]

    def test_list_distinct_tag_values_missing_key(self, store: DocumentStore) -> None:
        """Key not present in any document returns empty list."""
        store.upsert("default", "d1", "S1", {"topic": "auth"})

        values = store.list_distinct_tag_values("default", "nonexistent")
        assert values == []

    def test_list_distinct_tag_values_partial_key(self, store: DocumentStore) -> None:
        """Only documents with the key contribute values."""
        store.upsert("default", "d1", "S1", {"topic": "auth", "status": "open"})
        store.upsert("default", "d2", "S2", {"topic": "db"})  # no status

        values = store.list_distinct_tag_values("default", "status")
        assert values == ["open"]

    def test_query_by_tag_key(self, store: DocumentStore) -> None:
        """query_by_tag_key returns documents having the specified key."""
        store.upsert("default", "d1", "S1", {"topic": "auth"})
        store.upsert("default", "d2", "S2", {"project": "web"})
        store.upsert("default", "d3", "S3", {"topic": "db", "project": "api"})

        results = store.query_by_tag_key("default", "topic")
        ids = {r.id for r in results}
        assert ids == {"d1", "d3"}

    def test_query_by_id_prefix_escapes_wildcards(self, store: DocumentStore) -> None:
        """LIKE wildcards in prefix are escaped, not treated as patterns."""
        store.upsert("default", "normal:1", "S1", {})
        store.upsert("default", "normal:2", "S2", {})
        store.upsert("default", "has%wild", "S3", {})
        store.upsert("default", "has_wild", "S4", {})

        # A prefix of "%" should NOT match everything
        results = store.query_by_id_prefix("default", "%")
        assert len(results) == 0  # no IDs start with literal %

        # A prefix of "_" should NOT match single-char wildcard
        results = store.query_by_id_prefix("default", "_")
        assert len(results) == 0

        # Literal prefix match works
        results = store.query_by_id_prefix("default", "normal:")
        assert len(results) == 2

        # Prefix with literal % matches the doc that has it
        results = store.query_by_id_prefix("default", "has%")
        assert len(results) == 1
        assert results[0].id == "has%wild"


class TestGetManyChunking:
    """Chunking of IN(...) queries over caller-supplied ID lists.

    These must stay under SQLite's variable limit (999 on older builds),
    via dedup + chunking.
    """

    @pytest.fixture
    def store(self):
        """Create a temporary document store."""
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "documents.db"
            with DocumentStore(db_path) as store:
                yield store

    def test_get_many_handles_ids_beyond_sqlite_variable_limit(
        self, store: DocumentStore,
    ) -> None:
        """Regression: >32k ids raised 'too many SQL variables'."""
        for i in range(3):
            store.upsert(collection="default", id=f"doc:{i}",
                         summary=f"summary {i}", tags={})

        # Well past both the legacy (999) and modern (32,766) limits,
        # with duplicates mixed in (find callers concatenate ID lists).
        ids = [f"missing:{i}" for i in range(40_000)]
        ids += [f"doc:{i}" for i in range(3)] * 2

        results = store.get_many("default", ids)
        assert set(results) == {"doc:0", "doc:1", "doc:2"}
        assert results["doc:1"].summary == "summary 1"

    def test_touch_many_handles_ids_beyond_sqlite_variable_limit(
        self, store: DocumentStore,
    ) -> None:
        """touch_many shares the same IN(...) shape and limit."""
        store.upsert(collection="default", id="doc:t",
                     summary="touch me", tags={})
        before = store.get("default", "doc:t").accessed_at

        ids = [f"missing:{i}" for i in range(40_000)] + ["doc:t", "doc:t"]
        store.touch_many("default", ids)

        after = store.get("default", "doc:t").accessed_at
        assert after is not None
        assert before is None or after >= before
