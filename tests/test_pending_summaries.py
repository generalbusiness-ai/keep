"""Tests for pending summaries queue."""

import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from keep.pending_summaries import PendingSummaryQueue


class TestPendingSummaryQueue:
    """Tests for the SQLite-backed pending summary queue."""

    def test_enqueue_and_count(self):
        """Should enqueue items and track count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            assert queue.count() == 0

            queue.enqueue("doc1", "default", "content one")
            assert queue.count() == 1

            queue.enqueue("doc2", "default", "content two")
            assert queue.count() == 2

            queue.close()

    def test_dequeue_returns_oldest_first(self):
        """Should return items in FIFO order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("first", "default", "content first")
            queue.enqueue("second", "default", "content second")
            queue.enqueue("third", "default", "content third")

            items = queue.dequeue(limit=2)
            assert len(items) == 2
            assert items[0].id == "first"
            assert items[1].id == "second"

            queue.close()

    def test_dequeue_claims_items(self):
        """Dequeued items should not appear in subsequent dequeue calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content one")
            queue.enqueue("doc2", "default", "content two")

            # First dequeue claims doc1
            items1 = queue.dequeue(limit=1)
            assert len(items1) == 1
            assert items1[0].id == "doc1"

            # Second dequeue should get doc2, not doc1 again
            items2 = queue.dequeue(limit=1)
            assert len(items2) == 1
            assert items2[0].id == "doc2"

            # Nothing left
            items3 = queue.dequeue(limit=1)
            assert len(items3) == 0

            queue.close()

    def test_dequeue_increments_attempts(self):
        """Should increment attempt counter on dequeue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")

            items = queue.dequeue(limit=1)
            assert items[0].attempts == 0  # Was 0 before dequeue

            # Release it back via fail()
            queue.fail("doc1", "default")

            # Clear retry_after so it's immediately available
            queue._conn.execute(
                "UPDATE pending_summaries SET retry_after = NULL WHERE id = 'doc1'"
            )
            queue._conn.commit()

            # Dequeue again — attempt counter should be incremented
            items = queue.dequeue(limit=1)
            assert items[0].attempts == 1

            queue.close()

    def test_complete_removes_item(self):
        """Should remove item from queue on complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")
            assert queue.count() == 1

            queue.complete("doc1", "default")
            assert queue.count() == 0

            queue.close()

    def test_fail_releases_item(self):
        """fail() should release a claimed item back to pending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")

            # Claim it
            items = queue.dequeue(limit=1)
            assert len(items) == 1
            assert queue.count() == 0  # Not pending anymore

            # Fail it — should be pending again (with retry backoff)
            queue.fail("doc1", "default")
            assert queue.count() == 1

            # Clear retry_after so it's immediately available
            queue._conn.execute(
                "UPDATE pending_summaries SET retry_after = NULL WHERE id = 'doc1'"
            )
            queue._conn.commit()

            # Can dequeue again
            items = queue.dequeue(limit=1)
            assert len(items) == 1
            assert items[0].id == "doc1"

            queue.close()

    def test_enqueue_replaces_existing(self):
        """Should replace existing item with same id+collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "original content")
            queue.enqueue("doc1", "default", "updated content")

            assert queue.count() == 1

            items = queue.dequeue(limit=1)
            assert items[0].content == "updated content"

            queue.close()

    def test_enqueue_resets_claimed_item(self):
        """Re-enqueueing a claimed item should reset it to pending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "original")
            queue.dequeue(limit=1)  # Claims it
            assert queue.count() == 0  # Processing, not pending

            # Re-enqueue resets to pending
            queue.enqueue("doc1", "default", "updated")
            assert queue.count() == 1

            items = queue.dequeue(limit=1)
            assert items[0].content == "updated"

            queue.close()

    def test_separate_collections(self):
        """Should treat same id in different collections as separate items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "collection_a", "content a")
            queue.enqueue("doc1", "collection_b", "content b")

            assert queue.count() == 2

            queue.close()

    def test_stats(self):
        """Should return queue statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "coll_a", "content")
            queue.enqueue("doc2", "coll_b", "content")

            stats = queue.stats()
            assert stats["pending"] == 2
            assert stats["collections"] == 2
            assert "queue_path" in stats

            queue.close()

    def test_clear(self):
        """Should clear all pending items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")
            queue.enqueue("doc2", "default", "content")

            cleared = queue.clear()
            assert cleared == 2
            assert queue.count() == 0

            queue.close()

    def test_concurrent_dequeue_no_overlap(self):
        """Two threads calling dequeue() should never get the same item.

        Uses small dequeue batches (limit=2) across multiple rounds to
        force actual thread interleaving — not just one thread winning
        all items in a single BEGIN IMMEDIATE.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            # Enqueue 20 items
            for i in range(20):
                queue.enqueue(f"doc{i}", "default", f"content {i}")

            results = [[], []]
            barrier = threading.Barrier(2)

            def dequeue_worker(idx):
                barrier.wait()  # Synchronize start
                # Multiple small dequeues to create interleaving
                for _ in range(10):
                    items = queue.dequeue(limit=2)
                    results[idx].extend(item.id for item in items)
                    # Complete items so they don't stay "processing"
                    for item in items:
                        queue.complete(item.id, item.collection, item.task_type)

            t1 = threading.Thread(target=dequeue_worker, args=(0,))
            t2 = threading.Thread(target=dequeue_worker, args=(1,))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            # No overlap: each item claimed by exactly one thread
            all_claimed = results[0] + results[1]
            assert len(all_claimed) == len(set(all_claimed)), \
                f"Overlap detected: {results[0]} vs {results[1]}"
            assert len(all_claimed) == 20

            queue.close()

    def test_get_status_reflects_claim(self):
        """get_status should show processing status for claimed items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")
            status = queue.get_status("doc1")
            assert status["status"] == "pending"

            queue.dequeue(limit=1)
            status = queue.get_status("doc1")
            assert status["status"] == "processing"

            queue.close()

    def test_count_excludes_processing(self):
        """count() should only count pending items, not processing ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content one")
            queue.enqueue("doc2", "default", "content two")
            assert queue.count() == 2

            queue.dequeue(limit=1)
            assert queue.count() == 1  # One still pending

            queue.close()

    def test_migration_adds_status_columns(self):
        """Opening an old database without status columns should migrate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "pending.db"

            # Create old-schema database manually
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE pending_summaries (
                    id TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    content TEXT NOT NULL,
                    queued_at TEXT NOT NULL,
                    attempts INTEGER DEFAULT 0,
                    task_type TEXT DEFAULT 'summarize',
                    metadata TEXT DEFAULT '{}',
                    PRIMARY KEY (id, collection, task_type)
                )
            """)
            conn.execute("""
                INSERT INTO pending_summaries (id, collection, content, queued_at)
                VALUES ('old_doc', 'default', 'old content', '2025-01-01T00:00:00')
            """)
            conn.commit()
            conn.close()

            # Open with new code — should migrate
            queue = PendingSummaryQueue(db_path)

            # Old item should be accessible and pending
            items = queue.dequeue(limit=1)
            assert len(items) == 1
            assert items[0].id == "old_doc"

            queue.close()

    def test_fail_sets_retry_backoff(self):
        """fail() should set retry_after in the future, blocking immediate dequeue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")
            queue.dequeue(limit=1)

            # Fail it — sets retry_after ~30s in the future
            queue.fail("doc1", "default", error="test error")

            # Item is pending but backoff hasn't elapsed
            assert queue.count() == 1

            # Immediate dequeue should return nothing (backoff active)
            items = queue.dequeue(limit=1)
            assert len(items) == 0

            queue.close()

    def test_fail_stores_error_message(self):
        """fail() should store the error message for diagnosis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")
            queue.dequeue(limit=1)
            queue.fail("doc1", "default", error="RuntimeError: model crashed")

            # Check error is stored
            cursor = queue._conn.execute(
                "SELECT last_error FROM pending_summaries WHERE id = 'doc1'"
            )
            row = cursor.fetchone()
            assert row[0] == "RuntimeError: model crashed"

            queue.close()

    def test_fail_backoff_increases_exponentially(self):
        """Successive failures should increase the retry delay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")

            retry_afters = []
            for i in range(3):
                # Clear backoff to allow dequeue
                queue._conn.execute(
                    "UPDATE pending_summaries SET retry_after = NULL WHERE id = 'doc1'"
                )
                queue._conn.commit()

                queue.dequeue(limit=1)
                queue.fail("doc1", "default", error=f"fail {i}")

                cursor = queue._conn.execute(
                    "SELECT retry_after FROM pending_summaries WHERE id = 'doc1'"
                )
                retry_afters.append(cursor.fetchone()[0])

            # Each retry_after should be further in the future
            # (30s, 60s, 120s from roughly the same "now")
            for i in range(1, len(retry_afters)):
                assert retry_afters[i] > retry_afters[i - 1], \
                    f"Backoff should increase: {retry_afters}"

            queue.close()

    def test_abandon_moves_to_failed_status(self):
        """abandon() should move item to 'failed' (dead letter)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")
            queue.dequeue(limit=1)

            queue.abandon("doc1", "default", error="Exhausted 5 attempts")

            # Not pending, not available for dequeue
            assert queue.count() == 0
            items = queue.dequeue(limit=1)
            assert len(items) == 0

            # But preserved in failed list
            failed = queue.list_failed()
            assert len(failed) == 1
            assert failed[0]["id"] == "doc1"
            assert failed[0]["last_error"] == "Exhausted 5 attempts"

            queue.close()

    def test_retry_failed_resets_to_pending(self):
        """retry_failed() should reset dead-letter items back to pending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")
            queue.dequeue(limit=1)
            queue.abandon("doc1", "default", error="gave up")

            assert queue.count() == 0
            assert len(queue.list_failed()) == 1

            # Retry resets them
            n = queue.retry_failed()
            assert n == 1
            assert queue.count() == 1
            assert len(queue.list_failed()) == 0

            # Can dequeue again
            items = queue.dequeue(limit=1)
            assert len(items) == 1
            assert items[0].id == "doc1"
            assert items[0].attempts == 0  # Reset

            queue.close()

    def test_stats_includes_status_breakdown(self):
        """stats() should include pending, processing, and failed counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")
            queue.enqueue("doc2", "default", "content")
            queue.enqueue("doc3", "default", "content")

            # doc1 → processing
            queue.dequeue(limit=1)
            # doc2 → failed
            queue._conn.execute(
                "UPDATE pending_summaries SET status = 'failed' WHERE id = 'doc2'"
            )
            queue._conn.commit()

            stats = queue.stats()
            assert stats["pending"] == 1     # doc3
            assert stats["processing"] == 1  # doc1
            assert stats["failed"] == 1      # doc2
            assert stats["total"] == 3

            queue.close()

    def test_stale_claim_recovery_respects_task_type(self):
        """Analyze tasks should get a longer stale claim threshold than embed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            # Enqueue two items with different task types
            queue.enqueue("doc1", "default", "c", task_type="embed")
            queue.enqueue("doc2", "default", "c", task_type="analyze")

            # Dequeue both (claims them with current timestamp)
            items = queue.dequeue(limit=2)
            assert len(items) == 2

            # Fake claimed_at to 15 minutes ago (> 10 min default, < 1 hour analyze)
            queue._conn.execute("""
                UPDATE pending_summaries SET claimed_at =
                    datetime('now', '-15 minutes')
                WHERE status = 'processing'
            """)
            queue._conn.commit()

            # Recover stale claims — should only recover embed (15 min > 10 min)
            # but NOT analyze (15 min < 1 hour)
            recovered = queue._recover_stale_claims()
            assert recovered == 1

            # embed should be back to pending, analyze still processing
            cursor = queue._conn.execute(
                "SELECT id, status FROM pending_summaries ORDER BY id"
            )
            rows = {r[0]: r[1] for r in cursor.fetchall()}
            assert rows["doc1"] == "pending"    # embed: recovered
            assert rows["doc2"] == "processing"  # analyze: still claimed

            queue.close()

    def test_mark_delegated(self):
        """mark_delegated should transition to 'delegated' status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content", task_type="summarize")
            items = queue.dequeue(limit=1)
            assert len(items) == 1

            queue.mark_delegated("doc1", "default", "summarize", "remote-task-abc")

            # Should not be pending or processing
            assert queue.count() == 0
            items = queue.dequeue(limit=1)
            assert len(items) == 0

            # Should appear in delegated list
            delegated = queue.list_delegated()
            assert len(delegated) == 1
            assert delegated[0].id == "doc1"
            assert delegated[0].metadata["_remote_task_id"] == "remote-task-abc"

            queue.close()

    def test_list_delegated_empty(self):
        """list_delegated should return empty list when nothing is delegated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content")

            assert queue.list_delegated() == []
            queue.close()

    def test_count_delegated(self):
        """count_delegated should count only delegated items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "c1", task_type="summarize")
            queue.enqueue("doc2", "default", "c2", task_type="ocr")
            assert queue.count_delegated() == 0

            items = queue.dequeue(limit=2)
            queue.mark_delegated("doc1", "default", "summarize", "rt-1")

            assert queue.count_delegated() == 1

            queue.mark_delegated("doc2", "default", "ocr", "rt-2")
            assert queue.count_delegated() == 2

            queue.close()

    def test_complete_delegated(self):
        """complete() should work on delegated items (same as processing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "content", task_type="summarize")
            queue.dequeue(limit=1)
            queue.mark_delegated("doc1", "default", "summarize", "rt-1")
            assert queue.count_delegated() == 1

            queue.complete("doc1", "default", "summarize")
            assert queue.count_delegated() == 0

            queue.close()

    def test_stats_includes_delegated(self):
        """stats() should include delegated count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")

            queue.enqueue("doc1", "default", "c1", task_type="summarize")
            queue.enqueue("doc2", "default", "c2", task_type="ocr")

            items = queue.dequeue(limit=1)
            queue.mark_delegated("doc1", "default", "summarize", "rt-1")

            stats = queue.stats()
            assert stats["delegated"] == 1
            assert stats["pending"] == 1

            queue.close()

    def test_migration_adds_delegation_columns(self):
        """Opening a pre-delegation database should add new columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "pending.db"

            # Create database with old schema (no remote_task_id/delegated_at)
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE pending_summaries (
                    id TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    content TEXT NOT NULL,
                    queued_at TEXT NOT NULL,
                    attempts INTEGER DEFAULT 0,
                    task_type TEXT DEFAULT 'summarize',
                    metadata TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'pending',
                    claimed_by TEXT,
                    claimed_at TEXT,
                    last_error TEXT,
                    retry_after TEXT,
                    PRIMARY KEY (id, collection, task_type)
                )
            """)
            conn.execute("""
                INSERT INTO pending_summaries (id, collection, content, queued_at)
                VALUES ('old_doc', 'default', 'old content', '2025-01-01T00:00:00')
            """)
            conn.commit()
            conn.close()

            # Open with new code — should migrate
            queue = PendingSummaryQueue(db_path)

            # Verify new columns exist by using delegation methods
            items = queue.dequeue(limit=1)
            assert len(items) == 1

            queue.mark_delegated("old_doc", "default", "summarize", "rt-1")
            delegated = queue.list_delegated()
            assert len(delegated) == 1

            queue.close()


class TestQueueThreadSafety:
    """The queue shares one sqlite connection across daemon threads.

    Regression tests for unlocked readers racing the processor thread's
    BEGIN IMMEDIATE transactions on the same connection (which raised
    "cannot start a transaction within a transaction" / recursive-cursor
    errors under load).
    """

    def test_readers_are_serialized_with_writers(self):
        """stats/list/count must not corrupt concurrent dequeue transactions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = PendingSummaryQueue(Path(tmpdir) / "pending.db")
            errors: list[Exception] = []

            def writer():
                try:
                    for i in range(200):
                        queue.enqueue(f"doc{i % 20}", "default", "content")
                        for item in queue.dequeue(limit=3):
                            queue.complete(item.id, item.collection,
                                           task_type=item.task_type)
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

            def reader():
                try:
                    for _ in range(200):
                        queue.stats()
                        queue.stats_by_type()
                        queue.list_pending(limit=5)
                        queue.list_failed()
                        queue.count()
                        queue.count_delegated()
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

            threads = [threading.Thread(target=writer)]
            threads += [threading.Thread(target=reader) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
            queue.close()

            assert errors == [], f"concurrent queue access raised: {errors}"
