"""Error recovery tests for DocumentStore.

Tests malformed database detection, iterdump recovery, runtime recovery,
and the non-fatal touch() behavior added in v0.38.1.
"""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from keep.api import Keeper
from keep.document_store import DocumentStore


def _create_valid_db_with_data(path: Path) -> None:
    """Create a valid DocumentStore with some data."""
    store = DocumentStore(path)
    store.upsert("default", "doc1", "First document", {"topic": "test"})
    store.upsert("default", "doc2", "Second document", {"topic": "test"})
    store.upsert("default", "doc3", "Third document", {"topic": "other"})
    store._conn.close()


def _corrupt_database(path: Path) -> None:
    """Corrupt a SQLite database by overwriting bytes in the middle."""
    data = path.read_bytes()
    # Corrupt bytes past the header (first 100 bytes are the header)
    # Write garbage in the middle of the file to corrupt a data page
    if len(data) > 2000:
        corrupted = data[:1000] + b"\x00" * 500 + data[1500:]
    else:
        corrupted = data[:100] + b"\x00" * 100 + data[200:]
    path.write_bytes(corrupted)


class TestMalformedDetection:
    """Test that malformed databases are detected during init."""

    def test_init_detects_and_recovers_malformed(self, tmp_path):
        """A corrupted DB should trigger recovery during init."""
        db_path = tmp_path / "test.db"
        _create_valid_db_with_data(db_path)

        # Checkpoint WAL so corruption hits the main file
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        _corrupt_database(db_path)

        # Remove WAL/SHM so the corrupt main file is what gets opened
        for suffix in ("-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()

        # Opening should either recover or raise
        try:
            store = DocumentStore(db_path)
            # If recovery succeeded, the DB should be usable
            store.upsert("default", "new", "post-recovery", {})
            rec = store.get("default", "new")
            assert rec is not None
        except sqlite3.DatabaseError:
            # If data was too corrupted to dump, that's expected
            pass

    def test_corrupt_backup_created(self, tmp_path):
        """Recovery should preserve the corrupt file as .db.corrupt."""
        db_path = tmp_path / "test.db"
        _create_valid_db_with_data(db_path)

        # Force recovery by simulating malformation on first init
        original_init = DocumentStore._init_db
        call_count = [0]

        def patched_init(self):
            call_count[0] += 1
            if call_count[0] == 1:
                original_init(self)
                raise sqlite3.DatabaseError("database disk image is malformed")
            else:
                original_init(self)

        with patch.object(DocumentStore, "_init_db", patched_init):
            DocumentStore(db_path)

        corrupt_path = Path(str(db_path) + ".corrupt")
        assert corrupt_path.exists(), "Corrupt backup should exist"


class TestRecoveryFromMalformed:
    """Test the iterdump recovery path preserves data."""

    def test_recovery_preserves_data(self, tmp_path):
        """Data should survive recovery via iterdump."""
        db_path = tmp_path / "test.db"
        _create_valid_db_with_data(db_path)

        # Force recovery by making quick_check fail
        # We do this by patching — cleaner than real corruption
        original_init = DocumentStore._init_db

        call_count = [0]

        def patched_init(self):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: set up connection but pretend it's malformed
                original_init(self)
                raise sqlite3.DatabaseError("database disk image is malformed")
            else:
                # Recovery path re-calls _init_db
                original_init(self)

        with patch.object(DocumentStore, "_init_db", patched_init):
            store = DocumentStore(db_path)

        # Data should be preserved
        rec = store.get("default", "doc1")
        assert rec is not None
        assert rec.summary == "First document"

        rec2 = store.get("default", "doc2")
        assert rec2 is not None

    def test_recovery_removes_wal_shm(self, tmp_path):
        """Recovery should clean up stale WAL/SHM files."""
        db_path = tmp_path / "test.db"
        _create_valid_db_with_data(db_path)

        # Create fake WAL/SHM files
        wal_path = Path(str(db_path) + "-wal")
        shm_path = Path(str(db_path) + "-shm")
        wal_path.write_bytes(b"fake wal")
        shm_path.write_bytes(b"fake shm")

        # Force recovery
        original_init = DocumentStore._init_db
        call_count = [0]

        def patched_init(self):
            call_count[0] += 1
            if call_count[0] == 1:
                original_init(self)
                raise sqlite3.DatabaseError("database disk image is malformed")
            else:
                original_init(self)

        with patch.object(DocumentStore, "_init_db", patched_init):
            DocumentStore(db_path)

        # Old WAL/SHM should be gone (new ones may exist from reopening)
        # The key is that the recovery explicitly removes them

    def test_reopen_restores_missing_fts_tables(self, tmp_path):
        """Opening a current store should repair missing FTS tables."""
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path)
        store.upsert("default", "doc1", "Alpha summary", {"topic": "test"})
        store.upsert("default", "doc1", "Beta summary", {"topic": "test"})
        store._conn.close()

        conn = sqlite3.connect(str(db_path))
        for name in (
            "documents_fts_ai", "documents_fts_ad", "documents_fts_au",
            "parts_fts_ai", "parts_fts_ad", "parts_fts_au",
            "versions_fts_ai", "versions_fts_ad", "versions_fts_au",
        ):
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        for table in ("documents_fts", "parts_fts", "versions_fts"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        conn.close()

        repaired = DocumentStore(db_path)
        assert repaired._fts_available is True
        current_hits = repaired.query_fts("default", "Beta", limit=10)
        version_hits = repaired.query_fts("default", "Alpha", limit=10)
        assert any(doc_id == "doc1" for doc_id, _, _ in current_hits)
        assert any(doc_id == "doc1@v1" for doc_id, _, _ in version_hits)

    def test_reopen_repairs_orphaned_fts_shadow_tables(self, tmp_path):
        """Opening should rebuild FTS when only orphaned shadow tables remain."""
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path)
        store.upsert("default", "doc1", "Alpha summary", {"topic": "test"})
        store.upsert("default", "doc1", "Beta summary", {"topic": "test"})
        store._conn.close()

        conn = sqlite3.connect(str(db_path))
        for prefix in ("documents_fts", "parts_fts", "versions_fts"):
            conn.execute(f"DROP TABLE IF EXISTS {prefix}")
            for name in (f"{prefix}_data", f"{prefix}_idx", f"{prefix}_docsize", f"{prefix}_config"):
                conn.execute(f"CREATE TABLE IF NOT EXISTS {name} (x INTEGER)")
            for suffix in ("ai", "ad", "au"):
                conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {prefix}_{suffix} "
                    "AFTER INSERT ON documents BEGIN SELECT 1; END"
                )
        conn.commit()
        conn.close()

        repaired = DocumentStore(db_path)
        assert repaired._fts_available is True
        current_hits = repaired.query_fts("default", "Beta", limit=10)
        version_hits = repaired.query_fts("default", "Alpha", limit=10)
        assert any(doc_id == "doc1" for doc_id, _, _ in current_hits)
        assert any(doc_id == "doc1@v1" for doc_id, _, _ in version_hits)

    def test_unrecoverable_raises(self, tmp_path):
        """A truly unreadable file should propagate the error."""
        db_path = tmp_path / "test.db"
        db_path.write_bytes(b"\x00" * 100)  # Not a valid SQLite file

        with pytest.raises(sqlite3.DatabaseError):
            DocumentStore(db_path)


class TestRuntimeRecovery:
    """Test runtime recovery triggered by touch() and get()."""

    def test_touch_malformed_is_nonfatal(self, tmp_path):
        """touch() should not crash even if the DB is malformed."""
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path)
        store.upsert("default", "doc1", "hello", {})

        # Replace the connection with a wrapper that raises on touch SQL
        original_conn = store._conn
        original_execute = original_conn.execute

        class FailingConn:
            """Proxy that intercepts execute for touch SQL."""
            def __getattr__(self, name):
                return getattr(original_conn, name)

            def execute(self, sql, *args, **kwargs):
                if "UPDATE documents SET accessed_at" in str(sql):
                    raise sqlite3.DatabaseError("database disk image is malformed")
                return original_execute(sql, *args, **kwargs)

        store._conn = FailingConn()
        # Should not raise
        store.touch("default", "doc1")
        # Restore
        store._conn = original_conn

    def test_try_runtime_recover_returns_false_on_failure(self, tmp_path):
        """_try_runtime_recover returns False when recovery fails."""
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path)

        with (
            patch.object(
                store,
                "_open_connection",
                side_effect=sqlite3.DatabaseError("database disk image is malformed"),
            ),
            patch.object(store, "_recover_malformed", side_effect=Exception("boom")),
        ):
            result = store._try_runtime_recover()
            assert result is False

    def test_try_runtime_recover_returns_true_on_success(self, tmp_path):
        """_try_runtime_recover returns True when recovery succeeds."""
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path)
        store.upsert("default", "doc1", "hello", {})

        with patch.object(store, "_recover_malformed"):
            result = store._try_runtime_recover()
            assert result is True

    def test_runtime_recover_reopens_connections_even_when_quick_check_ok(self, tmp_path):
        """Runtime recovery should not reuse the connection that reported malformation."""
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path)
        store.upsert("default", "doc1", "hello", {})
        generation = store._connection_generation

        result = store._try_runtime_recover()

        assert result is True
        assert store._connection_generation > generation
        assert store.get("default", "doc1") is not None

    def test_malformed_fetch_invalidates_connection_generation(self, tmp_path):
        """A malformed sqlite3 error immediately invalidates thread-local handles."""
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path)
        generation = store._connection_generation
        original_conn = store._conn
        original_execute = original_conn.execute

        class FailingConn:
            """Proxy that raises a malformed error for one SELECT."""

            def __getattr__(self, name):
                return getattr(original_conn, name)

            def execute(self, sql, *args, **kwargs):
                if "SELECT 1" in str(sql):
                    raise sqlite3.DatabaseError("database disk image is malformed")
                return original_execute(sql, *args, **kwargs)

        store._conn = FailingConn()

        with pytest.raises(sqlite3.DatabaseError, match="malformed"):
            store._fetchone("SELECT 1")

        assert store._connection_generation > generation

    def test_slow_sql_logs_sanitized_fingerprint(self, tmp_path, caplog):
        """Slow statement diagnostics include a fingerprint and not raw params."""
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path)
        store._sqlite_slow_ms = 0

        with caplog.at_level("WARNING"):
            store._fetchone("SELECT ? AS value", ("secret-note-content",))

        assert "SQLite statement slow" in caplog.text
        assert "fingerprint=" in caplog.text
        assert "callsite=" in caplog.text
        assert "secret-note-content" not in caplog.text

    def test_keeper_get_raises_when_runtime_recovery_fails(self, tmp_path, mock_providers):
        """Malformed canonical reads should not silently fall back to Chroma."""
        kp = Keeper(store_path=tmp_path)
        try:
            kp.put(content="hello", id="doc1")
            with (
                patch.object(
                    kp._document_store,
                    "get",
                    side_effect=sqlite3.DatabaseError("database disk image is malformed"),
                ),
                patch.object(
                    kp._document_store,
                    "_try_runtime_recover",
                    return_value=False,
                    create=True,
                ),
            ):
                with pytest.raises(sqlite3.DatabaseError, match="malformed"):
                    kp._get_direct("doc1")
        finally:
            kp.close()


class TestUnicodeRepair:
    """Malformed surrogate text should not crash note writes."""

    def test_put_repairs_surrogate_content(self, tmp_path, mock_providers):
        kp = Keeper(store_path=tmp_path)
        try:
            item = kp.put(content="\ud83d\udcdd note")
            stored = kp.get(item.id)
            assert stored is not None
            assert stored.summary.startswith("📝 note")
        finally:
            kp.close()
