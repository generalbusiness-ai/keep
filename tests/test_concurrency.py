"""Concurrency tests for DocumentStore.

Verifies that multiple processes can safely access the same SQLite database
simultaneously — the real scenario when hooks fire concurrently.

Uses multiprocessing (not threading) to simulate separate keep processes.
"""

import json
import multiprocessing
import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest


# Worker functions must be top-level for multiprocessing spawn compatibility


def _worker_upsert_unique(db_path: str, worker_id: int, count: int):
    """Worker that writes unique doc IDs to the database."""
    from keep.document_store import DocumentStore
    store = DocumentStore(Path(db_path))
    for i in range(count):
        store.upsert(
            "default",
            f"worker{worker_id}:doc{i}",
            f"Summary from worker {worker_id} doc {i}",
            {"worker": str(worker_id)},
        )


def _worker_upsert_same_id(db_path: str, worker_id: int, count: int):
    """Worker that writes the same doc ID repeatedly."""
    from keep.document_store import DocumentStore
    store = DocumentStore(Path(db_path))
    for i in range(count):
        store.upsert(
            "default",
            "shared-doc",
            f"Version from worker {worker_id} iteration {i}",
            {"worker": str(worker_id), "iteration": str(i)},
        )


def _worker_reader(db_path: str, iterations: int):
    """Worker that reads from the database in a loop."""
    from keep.document_store import DocumentStore
    store = DocumentStore(Path(db_path))
    errors = []
    for _ in range(iterations):
        try:
            ids = store.list_ids("default")
            for doc_id in ids[:5]:
                rec = store.get("default", doc_id)
                if rec is not None and not rec.summary:
                    errors.append(f"Empty summary for {doc_id}")
        except Exception as e:
            errors.append(str(e))
    return errors


def _worker_open_and_migrate(db_path: str, worker_id: int):
    """Worker that opens a DocumentStore (triggering migration)."""
    from keep.document_store import DocumentStore
    store = DocumentStore(Path(db_path))
    version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    return version


class TestConcurrentWrites:
    """Test multiple processes writing to the same DocumentStore."""

    def test_threads_use_distinct_connections(self, tmp_path):
        """One DocumentStore must not share a sqlite connection across threads."""
        from keep.document_store import DocumentStore

        store = DocumentStore(tmp_path / "test.db")
        main_conn_id = id(store._conn)
        worker_conn_ids = []
        errors = []

        def worker(worker_id: int):
            try:
                worker_conn_ids.append(id(store._conn))
                store.upsert(
                    "default",
                    f"thread:{worker_id}",
                    f"thread summary {worker_id}",
                    {"worker": str(worker_id)},
                )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(set(worker_conn_ids)) == len(worker_conn_ids)
        assert main_conn_id not in worker_conn_ids
        assert store.count("default") == 4

    def test_release_thread_connection_closes_worker_handles(self, tmp_path):
        """Request-thread cleanup must remove closed worker handles from tracking."""
        from keep.document_store import DocumentStore

        store = DocumentStore(tmp_path / "test.db")
        store.upsert("default", "main", "main summary", {})
        with store._connections_lock:
            baseline = len(store._connections)
        errors = []

        def worker(worker_id: int):
            try:
                store.upsert(
                    "default",
                    f"thread-release:{worker_id}",
                    f"thread release summary {worker_id}",
                    {"worker": str(worker_id)},
                )
                store.release_thread_connection()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(12)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        with store._connections_lock:
            assert len(store._connections) == baseline
        assert store.count("default") == 13

    def test_readers_do_not_wait_for_store_write_lock(self, tmp_path):
        """Pure SELECT reads can proceed while another thread holds _lock."""
        from keep.document_store import DocumentStore

        store = DocumentStore(tmp_path / "test.db")
        store.upsert("default", "seed", "seed summary", {})

        finished = threading.Event()
        errors = []

        def reader():
            try:
                assert store.count("default") == 1
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()

        with store._lock:
            thread = threading.Thread(target=reader)
            thread.start()
            assert finished.wait(timeout=1.0)

        thread.join(timeout=1.0)
        assert errors == []

    def test_parallel_upserts_no_data_loss(self, tmp_path):
        """8 workers each write unique docs — all must be present after."""
        db_path = str(tmp_path / "test.db")
        num_workers = 8
        docs_per_worker = 10

        # Pre-create the database so migrations don't race with writes
        from keep.document_store import DocumentStore
        DocumentStore(Path(db_path))

        ctx = multiprocessing.get_context("spawn")
        processes = []
        for w in range(num_workers):
            p = ctx.Process(
                target=_worker_upsert_unique,
                args=(db_path, w, docs_per_worker),
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=30)

        # Verify all processes exited cleanly
        for p in processes:
            assert p.exitcode == 0, f"Worker exited with code {p.exitcode}"

        # Verify all docs are present
        store = DocumentStore(Path(db_path))
        all_ids = store.list_ids("default")
        expected = num_workers * docs_per_worker
        assert len(all_ids) == expected, (
            f"Expected {expected} docs, got {len(all_ids)}"
        )

    def test_parallel_upserts_same_id(self, tmp_path):
        """N workers upsert the same ID — one current + versions archived."""
        db_path = str(tmp_path / "test.db")
        num_workers = 4
        writes_per_worker = 5

        from keep.document_store import DocumentStore
        DocumentStore(Path(db_path))

        ctx = multiprocessing.get_context("spawn")
        processes = []
        for w in range(num_workers):
            p = ctx.Process(
                target=_worker_upsert_same_id,
                args=(db_path, w, writes_per_worker),
            )
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=30)

        for p in processes:
            assert p.exitcode == 0

        # Exactly one current version
        store = DocumentStore(Path(db_path))
        current = store.get("default", "shared-doc")
        assert current is not None
        assert current.summary.startswith("Version from worker")

        # Archived versions should exist (total writes - 1)
        total_writes = num_workers * writes_per_worker
        version_count = store.version_count("default", "shared-doc")
        assert version_count == total_writes - 1, (
            f"Expected {total_writes - 1} archived versions, got {version_count}"
        )

    def test_concurrent_read_during_write(self, tmp_path):
        """Writer + reader running in parallel — reader never sees corruption."""
        db_path = str(tmp_path / "test.db")

        from keep.document_store import DocumentStore
        DocumentStore(Path(db_path))

        ctx = multiprocessing.get_context("spawn")

        writer = ctx.Process(
            target=_worker_upsert_unique,
            args=(db_path, 0, 50),
        )
        # Reader uses a pool to get return value
        with ctx.Pool(1) as pool:
            writer.start()
            result = pool.apply_async(
                _worker_reader, (db_path, 100)
            )
            writer.join(timeout=30)
            errors = result.get(timeout=30)

        assert writer.exitcode == 0
        assert errors == [], f"Reader saw errors: {errors}"


class TestConcurrentMigration:
    """Test multiple processes migrating the same database simultaneously."""

    def test_parallel_migration_from_v0(self, tmp_path):
        """N processes open a v0 DB — schema must reach v3 exactly once."""
        db_path = str(tmp_path / "test.db")

        # Create a raw v0 database (just the documents table)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE documents (
                id TEXT NOT NULL,
                collection TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (id, collection)
            )
        """)
        conn.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
            ("test1", "default", "hello", "{}", "2025-01-01", "2025-01-01"),
        )
        conn.commit()
        conn.close()

        # Spawn N processes all opening DocumentStore simultaneously
        num_workers = 6
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(num_workers) as pool:
            results = pool.starmap(
                _worker_open_and_migrate,
                [(db_path, i) for i in range(num_workers)],
            )

        # All workers should see version 3
        from keep.document_store import SCHEMA_VERSION
        for version in results:
            assert version == SCHEMA_VERSION

        # Verify the schema is correct
        from keep.document_store import DocumentStore
        store = DocumentStore(Path(db_path))

        # Versions table exists
        tables = [
            r[0] for r in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        assert "document_versions" in tables

        # accessed_at column exists
        columns = [
            r[1] for r in
            store._conn.execute("PRAGMA table_info(documents)").fetchall()
        ]
        assert "accessed_at" in columns
        assert "content_hash" in columns

        # Original data preserved
        rec = store.get("default", "test1")
        assert rec is not None
        assert rec.summary == "hello"
