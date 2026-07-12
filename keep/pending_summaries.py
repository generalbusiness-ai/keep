"""Pending work queue using SQLite.

Stores work items (summarization, analysis, etc.) for serial background
processing. This enables fast indexing with lazy summarization, and
ensures heavy ML work (MLX model loading) is serialized to prevent
memory exhaustion.

Dequeue is atomic: items transition from 'pending' to 'processing' with
a PID claim inside a single IMMEDIATE transaction. Concurrent processors
cannot grab the same items. Stale claims (crashed processors) are
recovered automatically.

Failed items use exponential backoff before retry (30s, 60s, 120s, ...
up to 1h). Items that exhaust MAX_ATTEMPTS are moved to 'failed' status
(dead letter) rather than deleted, preserving the error for diagnosis.
"""

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .const import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_MAX_SECONDS,
    SQLITE_BUSY_TIMEOUT_MS,
)

logger = logging.getLogger(__name__)

# Claims older than this are considered stale (processor crashed).
# analyze tasks get a longer timeout because ollama can be slow
# (20s per window × many windows on large documents).
STALE_CLAIM_SECONDS = 600  # 10 minutes (default)
STALE_CLAIM_SECONDS_BY_TYPE = {
    "analyze": 3600,   # 1 hour — large docs via ollama are slow
    "ocr": 1800,       # 30 min — multi-page PDF OCR can be slow
}


@dataclass
class PendingSummary:
    """A queued work item awaiting processing."""
    id: str
    collection: str
    content: str
    queued_at: str
    attempts: int = 0
    task_type: str = "summarize"
    metadata: dict = field(default_factory=dict)
    delegated_at: str | None = None


class PendingSummaryQueue:
    """SQLite-backed queue for pending background work.

    Items are added during fast indexing (with truncated placeholder summary)
    or by enqueue_analyze, and processed later by `keep daemon`
    or programmatically. All work is serialized to prevent concurrent
    ML model loading.
    """

    def __init__(self, queue_path: Path):
        """Initialize.

        Args:
        queue_path: Path to SQLite database file.
        """
        self._queue_path = queue_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None gives us manual transaction control
        # so we can use BEGIN IMMEDIATE for atomic dequeue
        self._conn = sqlite3.connect(
            str(self._queue_path), check_same_thread=False,
            isolation_level=None,
        )

        # Enable WAL mode for better concurrent access across processes
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Wait up to 5 seconds for locks instead of failing immediately
        self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_summaries (
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
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_queued_at
            ON pending_summaries(queued_at)
        """)

        # Migrate existing databases: add new columns if missing
        self._migrate()

        # Index on status (safe after migration ensures column exists)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status
            ON pending_summaries(status)
        """)

    def _migrate(self) -> None:
        """Migrate existing databases to current schema."""
        cursor = self._conn.execute("PRAGMA table_info(pending_summaries)")
        columns = {row[1] for row in cursor.fetchall()}

        if "task_type" not in columns:
            # Old schema: PK is (id, collection). Recreate with (id, collection, task_type).
            # Preserve any pending items during migration.
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_summaries_new (
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
            self._conn.execute("""
                INSERT OR IGNORE INTO pending_summaries_new
                    (id, collection, content, queued_at, attempts)
                SELECT id, collection, content, queued_at, attempts
                FROM pending_summaries
            """)
            self._conn.execute("DROP TABLE pending_summaries")
            self._conn.execute(
                "ALTER TABLE pending_summaries_new RENAME TO pending_summaries"
            )
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_queued_at
                ON pending_summaries(queued_at)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON pending_summaries(status)
            """)
            self._conn.commit()
            return  # Full recreate already has all columns

        # Check if task_type is in the primary key.
        # Earlier migrations added the column but not always the PK fix.
        pk_sql = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='pending_summaries'"
        ).fetchone()[0]
        if "task_type" in columns and "PRIMARY KEY (id, collection)" in pk_sql and "task_type)" not in pk_sql:
            # task_type column exists but isn't in PK — recreate table
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_summaries_new (
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
                    remote_task_id TEXT,
                    delegated_at TEXT,
                    PRIMARY KEY (id, collection, task_type)
                )
            """)
            # Copy data — pick only columns that exist in both tables
            copy_cols = sorted(columns & {
                "id", "collection", "content", "queued_at", "attempts",
                "task_type", "metadata", "status", "claimed_by", "claimed_at",
                "last_error", "retry_after", "remote_task_id", "delegated_at",
            })
            cols = ", ".join(copy_cols)
            self._conn.execute(f"""
                INSERT OR IGNORE INTO pending_summaries_new ({cols})
                SELECT {cols} FROM pending_summaries
            """)
            self._conn.execute("DROP TABLE pending_summaries")
            self._conn.execute(
                "ALTER TABLE pending_summaries_new RENAME TO pending_summaries"
            )
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_queued_at
                ON pending_summaries(queued_at)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON pending_summaries(status)
            """)
            self._conn.commit()
            return  # Full recreate — all columns present

        # Incremental migration: add columns if missing.
        # Use try/except to handle races where another process adds the
        # column between our column check and the ALTER TABLE.
        def _add_column(col: str, typedef: str = "TEXT") -> None:
            try:
                self._conn.execute(
                    f"ALTER TABLE pending_summaries ADD COLUMN {col} {typedef}"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e):
                    raise

        if "status" not in columns:
            _add_column("status", "TEXT DEFAULT 'pending'")
            _add_column("claimed_by")
            _add_column("claimed_at")
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON pending_summaries(status)
            """)
            self._conn.commit()

        # Add error tracking and retry backoff columns
        if "last_error" not in columns:
            _add_column("last_error")
            _add_column("retry_after")
            self._conn.commit()

        # Add delegation tracking columns (Phase 3)
        if "remote_task_id" not in columns:
            _add_column("remote_task_id")
            _add_column("delegated_at")
            self._conn.commit()

    def _recover_stale_claims(self) -> int:
        """Reset items claimed by crashed processors back to pending.

        Called at the start of dequeue. Items stuck in 'processing'
        longer than the stale threshold are assumed to be from crashed
        processors and are reset. The threshold varies by task_type
        (analyze tasks get longer because ollama can be slow).

        Returns count of recovered items.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        total = 0
        # Recover each task type with its own threshold
        for task_type, threshold in STALE_CLAIM_SECONDS_BY_TYPE.items():
            cursor = self._conn.execute("""
                UPDATE pending_summaries
                SET status = 'pending', claimed_by = NULL, claimed_at = NULL
                WHERE status = 'processing'
                  AND task_type = ?
                  AND claimed_at IS NOT NULL
                  AND julianday(?) - julianday(claimed_at) > ? / 86400.0
            """, (task_type, now_iso, threshold))
            total += cursor.rowcount
        # Default threshold for all other task types
        cursor = self._conn.execute("""
            UPDATE pending_summaries
            SET status = 'pending', claimed_by = NULL, claimed_at = NULL
            WHERE status = 'processing'
              AND task_type NOT IN ({})
              AND claimed_at IS NOT NULL
              AND julianday(?) - julianday(claimed_at) > ? / 86400.0
        """.format(",".join("?" for _ in STALE_CLAIM_SECONDS_BY_TYPE)),
            (*STALE_CLAIM_SECONDS_BY_TYPE.keys(), now_iso, STALE_CLAIM_SECONDS))
        total += cursor.rowcount
        if total:
            self._conn.commit()
            logger.info("Recovered %d stale claims from crashed processors", total)
        return total

    def enqueue(
        self,
        id: str,
        collection: str,
        content: str,
        *,
        task_type: str = "summarize",
        metadata: Optional[dict] = None,
    ) -> None:
        """Add an item to the pending queue.

        If the same (id, collection, task_type) already exists, replaces it
        (resets to pending status).
        Different task types for the same document coexist independently.
        """
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata) if metadata else "{}"
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO pending_summaries
                (id, collection, content, queued_at, attempts, task_type, metadata,
                 status, claimed_by, claimed_at)
                VALUES (?, ?, ?, ?, 0, ?, ?, 'pending', NULL, NULL)
            """, (id, collection, content, now, task_type, meta_json))
            self._conn.commit()

    def enqueue_batch(
        self,
        items: list[tuple[str, str, str, str, Optional[dict]]],
    ) -> int:
        """Add multiple items to the pending queue in a single transaction.

        Each item is a tuple of (id, collection, content, task_type, metadata).
        Returns the number of items enqueued.
        """
        if not items:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (id, collection, content, now, task_type, json.dumps(metadata) if metadata else "{}")
            for id, collection, content, task_type, metadata in items
        ]
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany("""
                    INSERT OR REPLACE INTO pending_summaries
                    (id, collection, content, queued_at, attempts, task_type, metadata,
                     status, claimed_by, claimed_at)
                    VALUES (?, ?, ?, ?, 0, ?, ?, 'pending', NULL, NULL)
                """, rows)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return len(rows)

    def dequeue(self, limit: int = 10) -> list[PendingSummary]:
        """Atomically claim the oldest pending items for processing.

        Uses BEGIN IMMEDIATE to get an exclusive write lock, then
        selects and claims items in a single transaction. Concurrent
        processors will block briefly, then see only unclaimed items.

        Items transition from 'pending' to 'processing'. Call complete()
        after success or fail() to return to pending.
        """
        pid = str(os.getpid())
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            # Recover stale claims from crashed processors
            self._recover_stale_claims()

            # BEGIN IMMEDIATE acquires a write lock immediately,
            # preventing any other writer from interleaving
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # Select oldest pending items, respecting retry backoff
                cursor = self._conn.execute("""
                    SELECT id, collection, content, queued_at, attempts,
                           task_type, metadata
                    FROM pending_summaries
                    WHERE status = 'pending'
                      AND (retry_after IS NULL OR retry_after <= ?)
                    ORDER BY queued_at ASC
                    LIMIT ?
                """, (now, limit))

                rows = cursor.fetchall()
                items = []
                for row in rows:
                    meta = {}
                    if row[6]:
                        try:
                            meta = json.loads(row[6])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    items.append(PendingSummary(
                        id=row[0],
                        collection=row[1],
                        content=row[2],
                        queued_at=row[3],
                        attempts=row[4],
                        task_type=row[5] or "summarize",
                        metadata=meta,
                    ))

                # Claim them atomically
                if items:
                    keys = [
                        (pid, now, item.id, item.collection, item.task_type)
                        for item in items
                    ]
                    self._conn.executemany("""
                        UPDATE pending_summaries
                        SET status = 'processing',
                            claimed_by = ?,
                            claimed_at = ?,
                            attempts = attempts + 1
                        WHERE id = ? AND collection = ? AND task_type = ?
                    """, keys)

                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return items

    def complete(
        self, id: str, collection: str, task_type: str = "summarize"
    ) -> None:
        """Remove an item from the queue after successful processing."""
        with self._lock:
            self._conn.execute("""
                DELETE FROM pending_summaries
                WHERE id = ? AND collection = ? AND task_type = ?
            """, (id, collection, task_type))
            self._conn.commit()

    def fail(
        self, id: str, collection: str, task_type: str = "summarize",
        error: str | None = None,
    ) -> None:
        """Release a claimed item back to pending with exponential backoff.

        The attempt counter (already incremented by dequeue) is preserved.
        Retry delay: min(BASE * 2^(attempts-1), MAX) seconds.
        Error message is stored for diagnosis.
        """
        with self._lock:
            # Read current attempt count to compute backoff
            cursor = self._conn.execute(
                "SELECT attempts FROM pending_summaries "
                "WHERE id = ? AND collection = ? AND task_type = ?",
                (id, collection, task_type),
            )
            row = cursor.fetchone()
            attempts = row[0] if row else 1

            delay = min(BACKOFF_BASE_SECONDS * (2 ** (attempts - 1)), BACKOFF_MAX_SECONDS)
            now = datetime.now(timezone.utc)
            retry_at = (now + timedelta(seconds=delay)).isoformat()

            self._conn.execute("""
                UPDATE pending_summaries
                SET status = 'pending', claimed_by = NULL, claimed_at = NULL,
                    last_error = ?, retry_after = ?
                WHERE id = ? AND collection = ? AND task_type = ?
            """, (error, retry_at, id, collection, task_type))
            self._conn.commit()

            logger.info(
                "Item %s failed (attempt %d), retry after %ds: %s",
                id, attempts, delay, error or "unknown",
            )

    def abandon(
        self, id: str, collection: str, task_type: str = "summarize",
        error: str | None = None,
    ) -> None:
        """Move an item to 'failed' status (dead letter).

        Called when an item has exhausted all retry attempts. The item
        is preserved with its error message for diagnosis, not deleted.
        """
        with self._lock:
            self._conn.execute("""
                UPDATE pending_summaries
                SET status = 'failed', claimed_by = NULL, claimed_at = NULL,
                    last_error = ?
                WHERE id = ? AND collection = ? AND task_type = ?
            """, (error, id, collection, task_type))
            self._conn.commit()
            logger.warning("Abandoned %s/%s (%s): %s", collection, id, task_type, error or "max attempts")

    def mark_delegated(
        self,
        id: str,
        collection: str,
        task_type: str,
        remote_task_id: str,
    ) -> None:
        """Mark an item as delegated to a remote service.

        Transitions from 'processing' to 'delegated' and records the
        remote task ID for polling.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute("""
                UPDATE pending_summaries
                SET status = 'delegated',
                    remote_task_id = ?,
                    delegated_at = ?
                WHERE id = ? AND collection = ? AND task_type = ?
            """, (remote_task_id, now, id, collection, task_type))
            self._conn.commit()

    def list_delegated(self) -> list[PendingSummary]:
        """List items delegated to a remote service.

        Returns PendingSummary objects with remote_task_id in metadata.
        """
        with self._lock:
            cursor = self._conn.execute("""
                SELECT id, collection, content, queued_at, attempts,
                       task_type, metadata, remote_task_id, delegated_at
                FROM pending_summaries
                WHERE status = 'delegated'
                ORDER BY delegated_at ASC
            """)
            items = []
            for row in cursor.fetchall():
                meta = {}
                if row[6]:
                    try:
                        meta = json.loads(row[6])
                    except (json.JSONDecodeError, TypeError):
                        pass
                meta["_remote_task_id"] = row[7]
                items.append(PendingSummary(
                    id=row[0],
                    collection=row[1],
                    content=row[2],
                    queued_at=row[3],
                    attempts=row[4],
                    task_type=row[5] or "summarize",
                    metadata=meta,
                    delegated_at=row[8],
                ))
        return items

    def count_delegated(self) -> int:
        """Count items currently delegated to a remote service."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM pending_summaries WHERE status = 'delegated'"
            )
            return cursor.fetchone()[0]

    def count(self) -> int:
        """Get count of pending items (excludes processing and failed)."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM pending_summaries WHERE status = 'pending'"
            )
            return cursor.fetchone()[0]

    def stats(self) -> dict:
        """Get queue statistics including status breakdown.

        All reads of the shared connection must hold ``self._lock`` — the
        daemon's HTTP threads call these while the processor thread is inside
        ``dequeue()``'s BEGIN IMMEDIATE transaction on the same connection.
        ``stats_by_type`` takes the lock itself, so call it before entering.
        """
        by_type = self.stats_by_type()
        with self._lock:
            cursor = self._conn.execute("""
                SELECT status, COUNT(*) as cnt
                FROM pending_summaries
                GROUP BY status
            """)
            by_status = {row[0] or "pending": row[1] for row in cursor.fetchall()}

            cursor = self._conn.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(DISTINCT collection) as collections,
                    MAX(attempts) as max_attempts,
                    MIN(queued_at) as oldest
                FROM pending_summaries
            """)
            row = cursor.fetchone()
        return {
            "pending": by_status.get("pending", 0),
            "processing": by_status.get("processing", 0),
            "delegated": by_status.get("delegated", 0),
            "failed": by_status.get("failed", 0),
            "total": row[0],
            "collections": row[1],
            "max_attempts": row[2] or 0,
            "oldest": row[3],
            "queue_path": str(self._queue_path),
            "by_type": by_type,
        }

    def stats_by_type(self) -> dict[str, int]:
        """Get count of active items grouped by task type."""
        with self._lock:
            cursor = self._conn.execute("""
                SELECT task_type, COUNT(*) as cnt
                FROM pending_summaries
                WHERE status IN ('pending', 'processing', 'delegated')
                GROUP BY task_type
                ORDER BY cnt DESC
            """)
            return {row[0]: row[1] for row in cursor.fetchall()}

    def list_pending(self, limit: int = 50) -> list[dict]:
        """List pending work items."""
        with self._lock:
            cursor = self._conn.execute("""
                SELECT id, task_type, queued_at
                FROM pending_summaries
                WHERE status IN ('pending', 'processing')
                ORDER BY queued_at ASC
                LIMIT ?
            """, (limit,))
            return [
                {"work_id": row[0], "task_type": row[1], "supersede_key": row[0],
                 "created_at": row[2], "retry_after": None}
                for row in cursor.fetchall()
            ]

    def list_failed(self) -> list[dict]:
        """List items in failed (dead letter) status.

        Returns list of dicts with id, collection, task_type, attempts,
        last_error, queued_at.
        """
        with self._lock:
            cursor = self._conn.execute("""
                SELECT id, collection, task_type, attempts, last_error, queued_at
                FROM pending_summaries
                WHERE status = 'failed'
                ORDER BY queued_at ASC
            """)
            return [
                {
                    "id": row[0], "collection": row[1], "task_type": row[2],
                    "attempts": row[3], "last_error": row[4], "queued_at": row[5],
                }
                for row in cursor.fetchall()
            ]

    def retry_failed(self) -> int:
        """Reset all failed items back to pending for retry.

        Resets attempt counters and clears backoff. Returns count of
        items moved back to pending.
        """
        with self._lock:
            cursor = self._conn.execute("""
                UPDATE pending_summaries
                SET status = 'pending', attempts = 0, claimed_by = NULL,
                    claimed_at = NULL, last_error = NULL, retry_after = NULL
                WHERE status = 'failed'
            """)
            self._conn.commit()
            count = cursor.rowcount
            if count:
                logger.info("Reset %d failed items back to pending", count)
            return count

    def get_status(self, id: str) -> dict | None:
        """Get pending task status for a specific note.

        Returns dict with id, task_type, status, queued_at if the note
        has pending work, or None if no work is pending.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, task_type, queued_at, status FROM pending_summaries WHERE id = ?",
                (id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "task_type": row[1],
            "status": row[3] or "queued",
            "queued_at": row[2],
        }

    def peek(
        self, id: str, collection: str, task_type: str = "summarize",
    ) -> PendingSummary | None:
        """Read a pending item without claiming it."""
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT id, collection, content, queued_at, attempts,
                       task_type, metadata, delegated_at
                FROM pending_summaries
                WHERE id = ? AND collection = ? AND task_type = ? AND status = 'pending'
                LIMIT 1
                """,
                (id, collection, task_type),
            )
            row = cursor.fetchone()
            if row is None:
                return None

        meta = {}
        if row[6]:
            try:
                meta = json.loads(row[6])
            except (json.JSONDecodeError, TypeError):
                pass

        return PendingSummary(
            id=row[0],
            collection=row[1],
            content=row[2],
            queued_at=row[3],
            attempts=row[4],
            task_type=row[5] or "summarize",
            metadata=meta,
            delegated_at=row[7],
        )

    def clear(self) -> int:
        """Clear all pending items. Returns count of items cleared."""
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM pending_summaries")
            count = cursor.fetchone()[0]
            self._conn.execute("DELETE FROM pending_summaries")
            self._conn.commit()
        return count

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        """Ensure connection is closed on garbage collection."""
        self.close()
