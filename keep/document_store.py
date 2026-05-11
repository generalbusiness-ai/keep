"""Document store using SQLite.

Stores canonical document records separate from embeddings.
This enables multiple embedding providers to index the same documents.

The document store is the source of truth for:
- Document identity (URI / custom ID)
- Summary text
- Tags (source + system)
- Timestamps

Embeddings are stored in ChromaDB collections, keyed by embedding provider.
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .const import SQLITE_BUSY_TIMEOUT_MS
from .recovery import is_malformed_db_error
from .tracing import get_tracer
from .types import (
    build_item_context, eval_when_predicate,
    normalize_id,
    normalize_tag_map,
    parse_ref,
    repair_surrogate_text,
    tag_values,
    utc_now,
)

logger = logging.getLogger(__name__)


# Schema version for migrations
SCHEMA_VERSION = 16
DEFAULT_SQLITE_SLOW_MS = 1000.0
DEFAULT_SQLITE_PROGRESS_MS = 5000.0


def _env_float(name: str, default: float) -> float:
    """Read a float environment setting without letting bad values break startup."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.1f", name, raw, default)
        return default


def _sql_operation(sql: str) -> str:
    """Return a low-cardinality SQL operation name for tracing."""
    stripped = sql.lstrip()
    if not stripped:
        return "UNKNOWN"
    return stripped.split(None, 1)[0].upper()


def _is_readonly_sql(sql: str) -> bool:
    """Return True for statements that can safely bypass the store write lock."""
    operation = _sql_operation(sql)
    return operation in {"SELECT", "EXPLAIN"}


def _sql_shape(sql: str) -> str:
    """Return a sanitized, stable SQL shape for diagnostics.

    Query diagnostics need to identify the statement without logging note
    content.  The helpers below collapse literal values and whitespace, leaving
    table/column names and placeholders visible enough to locate the callsite.
    """
    shape = re.sub(r"'(?:''|[^'])*'", "?", sql)
    shape = re.sub(r'"(?:""|[^"])*"', "?", shape)
    shape = re.sub(r"\b\d+(?:\.\d+)?\b", "?", shape)
    return re.sub(r"\s+", " ", shape).strip()


def _sql_fingerprint(sql: str) -> str:
    """Return a compact fingerprint of the sanitized SQL shape."""
    import hashlib

    return hashlib.sha256(_sql_shape(sql).encode("utf-8")).hexdigest()[:12]


def _sql_callsite() -> str:
    """Return the first non-DocumentStore Python frame for SQL diagnostics."""
    for frame in reversed(traceback.extract_stack(limit=16)[:-2]):
        if Path(frame.filename).name != "document_store.py":
            return f"{Path(frame.filename).name}:{frame.lineno}"
    frame = traceback.extract_stack(limit=3)[0]
    return f"{Path(frame.filename).name}:{frame.lineno}"


def _load_tags_json(
    raw: Any,
    *,
    table: str = "documents",
    doc_id: str = "",
    collection: str = "",
) -> dict[str, Any]:
    """Decode a stored ``tags_json`` value, tolerating NULL or malformed JSON.

    The schema declares ``tags_json TEXT NOT NULL DEFAULT '{}'`` for every
    table that stores tags, so the only way ``raw`` can be ``None`` or a
    non-JSON string is if a row predates a schema constraint, was written by
    something bypassing the helpers, or got corrupted.  Reads should not
    propagate a JSON decode error to callers because every read path treats a
    missing tag map as benign — it's better to log once and continue with an
    empty mapping than to break ``get`` for a single bad row.
    """
    if raw is None or raw == "":
        logger.warning(
            "%s.tags_json is empty for %s/%s; treating as empty tag map",
            table,
            collection or "?",
            doc_id or "?",
        )
        return {}
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "%s.tags_json is invalid JSON for %s/%s (%s); treating as empty",
            table,
            collection or "?",
            doc_id or "?",
            exc,
        )
        return {}
    if not isinstance(decoded, dict):
        logger.warning(
            "%s.tags_json is not an object for %s/%s; treating as empty",
            table,
            collection or "?",
            doc_id or "?",
        )
        return {}
    return decoded


@dataclass
class VersionInfo:
    """Information about a document version.

    Used for version navigation and history display.
    """
    version: int  # 1=oldest archived, increasing
    summary: str
    tags: dict[str, Any]
    created_at: str
    content_hash: Optional[str] = None


@dataclass
class PartInfo:
    """Information about a document part (structural section).

    Parts are produced by analyze() — an LLM-driven decomposition
    of content into meaningful sections. Each part has its own summary,
    tags, and embedding for targeted search.
    """
    part_num: int           # 1-indexed
    summary: str
    tags: dict[str, Any]
    created_at: str


@dataclass
class DocumentRecord:
    """A canonical document record.

    This is the source of truth, independent of any embedding index.
    """
    id: str
    collection: str
    summary: str
    tags: dict[str, Any]
    created_at: str
    updated_at: str
    content_hash: Optional[str] = None
    content_hash_full: Optional[str] = None
    accessed_at: Optional[str] = None


class DocumentStore:
    """SQLite-backed store for canonical document records.
    
    Separates document metadata from embedding storage, enabling:
    - Multiple embedding providers per document
    - Efficient tag/metadata queries without ChromaDB
    - Clear separation of concerns
    """
    
    def __init__(self, store_path: Path):
        """Initialize.

        Args:
        store_path: Path to SQLite database file.
        """
        self._db_path = store_path
        self._local = threading.local()
        self._connections: set[Any] = set()
        self._connections_lock = threading.RLock()
        self._connection_generation = 0
        self._closed = False
        self._lock = threading.RLock()
        self._fts_available = False
        self._stopwords: Optional[frozenset[str]] = None
        self.migrated_parts_for_reindex: list[dict[str, Any]] = []
        self._sqlite_slow_ms = _env_float("KEEP_SQLITE_SLOW_MS", DEFAULT_SQLITE_SLOW_MS)
        self._sqlite_progress_ms = _env_float(
            "KEEP_SQLITE_PROGRESS_MS", DEFAULT_SQLITE_PROGRESS_MS,
        )
        try:
            self._init_db()
        except sqlite3.DatabaseError as e:
            if is_malformed_db_error(e):
                logger.warning("Database malformed, attempting recovery: %s", self._db_path)
                self._recover_malformed()
            else:
                raise

    def _open_connection(self) -> sqlite3.Connection:
        """Open a SQLite connection for the current thread."""
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        """Thread-local SQLite connection.

        The daemon serves requests from multiple threads. Python's sqlite3
        connection object is not safe for concurrent use, even when
        ``check_same_thread`` is disabled, so each thread gets its own handle.
        """
        conn = getattr(self._local, "conn", None)
        generation = getattr(self._local, "conn_generation", -1)
        if conn is None or generation != self._connection_generation:
            if conn is not None:
                with self._connections_lock:
                    self._connections.discard(conn)
                try:
                    conn.close()
                except Exception:
                    logger.debug("Failed to close stale SQLite connection", exc_info=True)
            if self._closed:
                raise sqlite3.ProgrammingError("DocumentStore is closed")
            conn = self._open_connection()
            self._local.conn = conn
            self._local.conn_generation = self._connection_generation
            with self._connections_lock:
                self._connections.add(conn)
        return conn

    @_conn.setter
    def _conn(self, conn: Optional[sqlite3.Connection]) -> None:
        """Replace or discard only this thread's connection handle."""
        old = getattr(self._local, "conn", None)
        if old is not None:
            with self._connections_lock:
                self._connections.discard(old)
        self._local.conn = conn
        self._local.conn_generation = self._connection_generation
        if conn is not None:
            with self._connections_lock:
                self._connections.add(conn)

    def _close_all_connections(self, *, mark_closed: bool) -> None:
        with self._connections_lock:
            self._connection_generation += 1
            connections = list(self._connections)
            self._connections.clear()
        for conn in connections:
            try:
                conn.close()
            except Exception:
                logger.debug(
                    "Failed to close SQLite connection for %s",
                    self._db_path,
                    exc_info=True,
                )
        self._local.conn = None
        self._local.conn_generation = self._connection_generation
        if mark_closed:
            self._closed = True

    def release_thread_connection(self) -> None:
        """Close this thread's SQLite handle after a bounded unit of work.

        The daemon creates a short-lived thread for each HTTP request.  Keeping
        those request-thread connections in the store-wide connection set leaks
        file descriptors until SQLite can no longer open databases under a
        burst of requests.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            return
        with self._connections_lock:
            self._connections.discard(conn)
        self._local.conn = None
        self._local.conn_generation = self._connection_generation
        try:
            conn.close()
        except Exception:
            logger.debug(
                "Failed to close SQLite request-thread connection for %s",
                self._db_path,
                exc_info=True,
            )

    def _log_sqlite_runtime(
        self,
        *,
        message: str,
        level: int,
        sql: str,
        params: tuple,
        started: float,
        callsite: str,
    ) -> None:
        """Log a sanitized SQLite runtime diagnostic."""
        elapsed_ms = (time.monotonic() - started) * 1000.0
        logger.log(
            level,
            "%s elapsed_ms=%.1f op=%s fingerprint=%s params=%d callsite=%s shape=%s",
            message,
            elapsed_ms,
            _sql_operation(sql),
            _sql_fingerprint(sql),
            len(params),
            callsite,
            _sql_shape(sql)[:240],
        )

    def _run_sql(self, *, sql: str, params: tuple, materialize) -> Any:
        """Run one SQLite operation with slow-query and malformed-DB diagnostics."""
        conn = self._conn
        started = time.monotonic()
        callsite = _sql_callsite()
        progress_every = max(self._sqlite_progress_ms / 1000.0, 0.0)
        next_progress_log = started + progress_every if progress_every else 0.0

        def progress() -> int:
            nonlocal next_progress_log
            now = time.monotonic()
            if next_progress_log and now >= next_progress_log:
                self._log_sqlite_runtime(
                    message="SQLite statement still running",
                    level=logging.WARNING,
                    sql=sql,
                    params=params,
                    started=started,
                    callsite=callsite,
                )
                next_progress_log = now + progress_every
            return 0

        try:
            if progress_every:
                conn.set_progress_handler(progress, 10_000)
            return materialize(conn)
        except sqlite3.DatabaseError as exc:
            if is_malformed_db_error(exc):
                logger.error(
                    "SQLite malformed error; invalidating all document-store "
                    "connections op=%s fingerprint=%s callsite=%s: %s",
                    _sql_operation(sql),
                    _sql_fingerprint(sql),
                    callsite,
                    exc,
                )
                self._close_all_connections(mark_closed=False)
            raise
        finally:
            try:
                conn.set_progress_handler(None, 0)
            except Exception:
                logger.debug("Failed to clear SQLite progress handler", exc_info=True)
            if self._sqlite_slow_ms >= 0:
                elapsed_ms = (time.monotonic() - started) * 1000.0
                if elapsed_ms >= self._sqlite_slow_ms:
                    self._log_sqlite_runtime(
                        message="SQLite statement slow",
                        level=logging.WARNING,
                        sql=sql,
                        params=params,
                        started=started,
                        callsite=callsite,
                    )
    
    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL with thread-safety.

        Each thread uses its own sqlite3 connection. Mutating statements and
        transactions still serialize through ``self._lock``; simple reads can
        run concurrently on their thread-local connections.
        """
        # Keep SQL text out of traces; statements can contain note content.
        with get_tracer("document_store").start_as_current_span(
            "document_store.execute",
            attributes={
                "db.system": "sqlite",
                "db.operation": _sql_operation(sql),
                "db.params_count": len(params),
            },
        ):
            if _is_readonly_sql(sql):
                return self._run_sql(
                    sql=sql,
                    params=params,
                    materialize=lambda conn: conn.execute(sql, params),
                )
            with self._lock:
                return self._run_sql(
                    sql=sql,
                    params=params,
                    materialize=lambda conn: conn.execute(sql, params),
                )

    def _fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """Execute SQL and materialize one row within the chosen lock scope."""
        with get_tracer("document_store").start_as_current_span(
            "document_store.fetchone",
            attributes={
                "db.system": "sqlite",
                "db.operation": _sql_operation(sql),
                "db.params_count": len(params),
            },
        ):
            if _is_readonly_sql(sql):
                return self._run_sql(
                    sql=sql,
                    params=params,
                    materialize=lambda conn: conn.execute(sql, params).fetchone(),
                )
            with self._lock:
                return self._run_sql(
                    sql=sql,
                    params=params,
                    materialize=lambda conn: conn.execute(sql, params).fetchone(),
                )

    def _fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute SQL and materialize all rows within the chosen lock scope."""
        with get_tracer("document_store").start_as_current_span(
            "document_store.fetchall",
            attributes={
                "db.system": "sqlite",
                "db.operation": _sql_operation(sql),
                "db.params_count": len(params),
            },
        ):
            if _is_readonly_sql(sql):
                return self._run_sql(
                    sql=sql,
                    params=params,
                    materialize=lambda conn: conn.execute(sql, params).fetchall(),
                )
            with self._lock:
                return self._run_sql(
                    sql=sql,
                    params=params,
                    materialize=lambda conn: conn.execute(sql, params).fetchall(),
                )

    def _execute_write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a mutating statement under the store write lock."""
        with get_tracer("document_store").start_as_current_span(
            "document_store.execute_write",
            attributes={
                "db.system": "sqlite",
                "db.operation": _sql_operation(sql),
                "db.params_count": len(params),
            },
        ):
            with self._lock:
                return self._run_sql(
                    sql=sql,
                    params=params,
                    materialize=lambda conn: conn.execute(sql, params),
                )

    def _executemany(self, sql: str, params_seq) -> sqlite3.Cursor:
        """Like _execute but for executemany."""
        attributes: dict[str, str | int] = {
            "db.system": "sqlite",
            "db.operation": _sql_operation(sql),
        }
        try:
            attributes["db.batch_size"] = len(params_seq)
        except TypeError:
            pass
        with get_tracer("document_store").start_as_current_span(
            "document_store.executemany",
            attributes=attributes,
        ):
            with self._lock:
                return self._run_sql(
                    sql=sql,
                    params=(),
                    materialize=lambda conn: conn.executemany(sql, params_seq),
                )

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._conn = self._open_connection()

        self._execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT NOT NULL,
                collection TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                content_hash TEXT,
                content_hash_full TEXT,
                PRIMARY KEY (id, collection)
            )
        """)

        # Run schema migrations (serialized across processes)
        self._migrate_schema()
        self._ensure_fts_schema()

        # Quick integrity check for existing databases
        result = self._fetchone("PRAGMA quick_check")
        if result[0] != "ok":
            raise sqlite3.DatabaseError("database disk image is malformed")

    def _fts_tables_present(self) -> bool:
        """Return whether all FTS tables are available."""
        try:
            self._execute("SELECT 1 FROM documents_fts LIMIT 0")
            self._execute("SELECT 1 FROM parts_fts LIMIT 0")
            self._execute("SELECT 1 FROM versions_fts LIMIT 0")
            return True
        except sqlite3.OperationalError:
            return False

    def _fts_object_exists(self, name: str) -> bool:
        """Return whether a SQLite object exists."""
        row = self._fetchone(
            "SELECT 1 FROM sqlite_master WHERE name = ? LIMIT 1",
            (name,),
        )
        return row is not None

    def _drop_orphaned_fts_schema(self, prefix: str) -> None:
        """Drop orphaned FTS triggers and shadow tables when the virtual table is missing."""
        if self._fts_object_exists(prefix):
            return

        shadow_names = (
            f"{prefix}_data",
            f"{prefix}_idx",
            f"{prefix}_docsize",
            f"{prefix}_config",
        )
        trigger_names = (
            f"{prefix}_ai",
            f"{prefix}_ad",
            f"{prefix}_au",
        )

        if not any(self._fts_object_exists(name) for name in (*shadow_names, *trigger_names)):
            return

        for name in trigger_names:
            self._execute(f"DROP TRIGGER IF EXISTS {name}")
        for name in shadow_names:
            self._execute(f"DROP TABLE IF EXISTS {name}")

        logger.warning("Dropped orphaned FTS schema for %s", prefix)

    def _ensure_fts_schema(self) -> None:
        """Ensure the FTS schema exists even on already-migrated stores."""
        if self._fts_tables_present():
            self._fts_available = True
            return

        try:
            for prefix in ("documents_fts", "parts_fts", "versions_fts"):
                self._drop_orphaned_fts_schema(prefix)

            self._execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
                USING fts5(
                    summary,
                    content='documents',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                )
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS documents_fts_ai
                AFTER INSERT ON documents BEGIN
                    INSERT INTO documents_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS documents_fts_ad
                AFTER DELETE ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS documents_fts_au
                AFTER UPDATE OF summary ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                    INSERT INTO documents_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS parts_fts
                USING fts5(
                    summary,
                    content='document_parts',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                )
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS parts_fts_ai
                AFTER INSERT ON document_parts BEGIN
                    INSERT INTO parts_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS parts_fts_ad
                AFTER DELETE ON document_parts BEGIN
                    INSERT INTO parts_fts(parts_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS parts_fts_au
                AFTER UPDATE OF summary ON document_parts BEGIN
                    INSERT INTO parts_fts(parts_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                    INSERT INTO parts_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS versions_fts
                USING fts5(
                    summary,
                    content='document_versions',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                )
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS versions_fts_ai
                AFTER INSERT ON document_versions BEGIN
                    INSERT INTO versions_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS versions_fts_ad
                AFTER DELETE ON document_versions BEGIN
                    INSERT INTO versions_fts(versions_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS versions_fts_au
                AFTER UPDATE OF summary ON document_versions BEGIN
                    INSERT INTO versions_fts(versions_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                    INSERT INTO versions_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
            self._execute("INSERT INTO parts_fts(parts_fts) VALUES('rebuild')")
            self._execute("INSERT INTO versions_fts(versions_fts) VALUES('rebuild')")
        except sqlite3.OperationalError:
            logger.info("FTS5 not available, full-text search disabled")
            self._fts_available = False
            return

        self._fts_available = self._fts_tables_present()

    def _migrate_schema(self) -> None:
        """Run schema migrations using PRAGMA user_version.

        Uses BEGIN EXCLUSIVE to serialize migrations across concurrent
        processes (e.g. hooks firing simultaneously). Each migration method is
        idempotent so partially migrated stores can safely retry on startup.
        """
        current_version = self._execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        if current_version > SCHEMA_VERSION:
            raise sqlite3.DatabaseError(
                f"Store schema version {current_version} is newer than supported "
                f"({SCHEMA_VERSION}). Please upgrade keep."
            )

        if current_version >= SCHEMA_VERSION:
            return  # Already up to date — no writes needed

        # Exclusive lock prevents two processes from racing through migrations.
        self._execute("BEGIN EXCLUSIVE")
        try:
            # Re-read inside the lock (another process may have migrated).
            current_version = self._execute(
                "PRAGMA user_version"
            ).fetchone()[0]

            if current_version > SCHEMA_VERSION:
                self._conn.rollback()
                raise sqlite3.DatabaseError(
                    f"Store schema version {current_version} is newer than supported "
                    f"({SCHEMA_VERSION}). Please upgrade keep."
                )

            if current_version >= SCHEMA_VERSION:
                self._conn.rollback()
                return

            for target_version, migration in self._schema_migrations():
                if current_version < target_version:
                    migration()

            self._execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _schema_migrations(self) -> tuple[tuple[int, Callable[[], None]], ...]:
        """Return ordered schema migrations keyed by target user_version."""
        return (
            (1, self._migrate_to_v1),
            (2, self._migrate_to_v2),
            (3, self._migrate_to_v3),
            (4, self._migrate_to_v4),
            (5, self._migrate_to_v5),
            (6, self._migrate_to_v6),
            (7, self._migrate_to_v7),
            (8, self._migrate_to_v8),
            (9, self._migrate_to_v9),
            (10, self._migrate_to_v10),
            (11, self._migrate_to_v11),
            (12, self._migrate_to_v12),
            (13, self._migrate_to_v13),
            (14, self._migrate_to_v14),
            (15, self._migrate_to_v15),
            (16, self._migrate_to_v16),
        )

    def _migrate_to_v1(self) -> None:
        """Apply schema migration to user_version 1."""
        # Create versions table for document history
        self._execute("""
            CREATE TABLE IF NOT EXISTS document_versions (
                id TEXT NOT NULL,
                collection TEXT NOT NULL,
                version INTEGER NOT NULL,
                summary TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                content_hash TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (id, collection, version)
            )
        """)
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_versions_doc
            ON document_versions(id, collection, version DESC)
        """)

    def _migrate_to_v2(self) -> None:
        """Apply schema migration to user_version 2."""
        # Add accessed_at column for last-access tracking
        columns = {
            row[1] for row in
            self._execute("PRAGMA table_info(documents)").fetchall()
        }
        if "accessed_at" not in columns:
            self._execute(
                "ALTER TABLE documents ADD COLUMN accessed_at TEXT"
            )
            self._execute(
                "UPDATE documents SET accessed_at = updated_at "
                "WHERE accessed_at IS NULL"
            )
            self._execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_accessed
                ON documents(accessed_at)
            """)

    def _migrate_to_v3(self) -> None:
        """Apply schema migration to user_version 3."""
        # Add content_hash column if missing (very old databases)
        columns = {
            row[1] for row in
            self._execute("PRAGMA table_info(documents)").fetchall()
        }
        if "content_hash" not in columns:
            self._execute(
                "ALTER TABLE documents ADD COLUMN content_hash TEXT"
            )

        # One-time hash truncation (64-char → 10-char)
        self._execute("""
            UPDATE documents SET content_hash = SUBSTR(content_hash, -10)
            WHERE content_hash IS NOT NULL AND LENGTH(content_hash) > 10
        """)
        cursor = self._execute("""
            SELECT id, collection, tags_json FROM documents
            WHERE tags_json LIKE '%bundled_hash%'
        """)
        for row in cursor.fetchall():
            tags = _load_tags_json(row["tags_json"])
            bh = tags.get("bundled_hash")
            if bh and len(bh) > 10:
                tags["bundled_hash"] = bh[-10:]
                self._execute(
                    "UPDATE documents SET tags_json = ? "
                    "WHERE id = ? AND collection = ?",
                    (json.dumps(tags), row["id"], row["collection"])
                )

        # Create indexes (idempotent)
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_collection
            ON documents(collection)
        """)
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_updated
            ON documents(updated_at)
        """)

    def _migrate_to_v4(self) -> None:
        """Apply schema migration to user_version 4."""
        # Create parts table for structural decomposition
        self._execute("""
            CREATE TABLE IF NOT EXISTS document_parts (
                id TEXT NOT NULL,
                collection TEXT NOT NULL,
                part_num INTEGER NOT NULL,
                summary TEXT NOT NULL,
                tags_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                PRIMARY KEY (id, collection, part_num)
            )
        """)
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_parts_doc
            ON document_parts(id, collection, part_num)
        """)

    def _migrate_to_v5(self) -> None:
        """Apply schema migration to user_version 5."""
        # Index for content-hash dedup lookups
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_content_hash
            ON documents(collection, content_hash)
            WHERE content_hash IS NOT NULL
        """)

    def _migrate_to_v6(self) -> None:
        """Apply schema migration to user_version 6."""
        # Full SHA256 hash for dedup content verification
        columns = {
            row[1]
            for row in self._execute(
                "PRAGMA table_info(documents)"
            ).fetchall()
        }
        if "content_hash_full" not in columns:
            self._execute(
                "ALTER TABLE documents ADD COLUMN content_hash_full TEXT"
            )

    def _migrate_to_v7(self) -> None:
        """Apply schema migration to user_version 7."""
        # FTS5 full-text search index on document summaries
        try:
            self._execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
                USING fts5(
                    summary,
                    content='documents',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                )
            """)
            # Triggers: INSERT OR REPLACE fires DELETE then INSERT,
            # so both triggers cover the upsert case.
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS documents_fts_ai
                AFTER INSERT ON documents BEGIN
                    INSERT INTO documents_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS documents_fts_ad
                AFTER DELETE ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS documents_fts_au
                AFTER UPDATE OF summary ON documents BEGIN
                    INSERT INTO documents_fts(documents_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                    INSERT INTO documents_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute(
                "INSERT INTO documents_fts(documents_fts) VALUES('rebuild')"
            )
            self._fts_available = True
        except sqlite3.OperationalError:
            logger.info("FTS5 not available, full-text search disabled")

    def _migrate_to_v8(self) -> None:
        """Apply schema migration to user_version 8."""
        # FTS5 indexes for parts and versions
        try:
            # --- Parts FTS ---
            self._execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS parts_fts
                USING fts5(
                    summary,
                    content='document_parts',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                )
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS parts_fts_ai
                AFTER INSERT ON document_parts BEGIN
                    INSERT INTO parts_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS parts_fts_ad
                AFTER DELETE ON document_parts BEGIN
                    INSERT INTO parts_fts(parts_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS parts_fts_au
                AFTER UPDATE OF summary ON document_parts BEGIN
                    INSERT INTO parts_fts(parts_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                    INSERT INTO parts_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute(
                "INSERT INTO parts_fts(parts_fts) VALUES('rebuild')"
            )

            # --- Versions FTS ---
            self._execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS versions_fts
                USING fts5(
                    summary,
                    content='document_versions',
                    content_rowid='rowid',
                    tokenize='porter unicode61'
                )
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS versions_fts_ai
                AFTER INSERT ON document_versions BEGIN
                    INSERT INTO versions_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS versions_fts_ad
                AFTER DELETE ON document_versions BEGIN
                    INSERT INTO versions_fts(versions_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                END
            """)
            self._execute("""
                CREATE TRIGGER IF NOT EXISTS versions_fts_au
                AFTER UPDATE OF summary ON document_versions BEGIN
                    INSERT INTO versions_fts(versions_fts, rowid, summary)
                    VALUES('delete', old.rowid, old.summary);
                    INSERT INTO versions_fts(rowid, summary)
                    VALUES (new.rowid, new.summary);
                END
            """)
            self._execute(
                "INSERT INTO versions_fts(versions_fts) VALUES('rebuild')"
            )
            self._fts_available = True
        except sqlite3.OperationalError:
            logger.info("FTS5 not available, full-text search disabled")

    def _migrate_to_v9(self) -> None:
        """Version 8 → 9: edges and edge_backfill tables."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source_id   TEXT NOT NULL,
                collection  TEXT NOT NULL,
                predicate   TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                inverse     TEXT NOT NULL,
                created     TEXT NOT NULL,
                PRIMARY KEY (source_id, collection, predicate)
            )
        """)
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_target
            ON edges (target_id, collection, inverse, created)
        """)
        self._execute("""
            CREATE TABLE IF NOT EXISTS edge_backfill (
                collection  TEXT NOT NULL,
                predicate   TEXT NOT NULL,
                inverse     TEXT NOT NULL,
                completed   TEXT,
                PRIMARY KEY (collection, predicate)
            )
        """)

    def _migrate_to_v10(self) -> None:
        """Version 9 → 10: materialized version_edges."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS version_edges (
                collection  TEXT NOT NULL,
                source_id   TEXT NOT NULL,
                version     INTEGER NOT NULL,
                predicate   TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                inverse     TEXT NOT NULL,
                created     TEXT NOT NULL,
                PRIMARY KEY (collection, source_id, version, predicate)
            )
        """)
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_version_edges_target
            ON version_edges (target_id, collection, inverse, created)
        """)
        # One-time backfill from archived versions for currently known
        # edge predicates (prefer tagdocs; include backfill/edges for
        # compatibility with partially-migrated stores).
        self._execute("""
            WITH predicates AS (
                SELECT d.collection AS collection,
                       SUBSTR(d.id, 6) AS predicate,
                       CAST(json_extract(d.tags_json, '$._inverse') AS TEXT) AS inverse
                FROM documents d
                WHERE d.id LIKE '.tag/%'
                  AND INSTR(SUBSTR(d.id, 6), '/') = 0
                  AND json_extract(d.tags_json, '$._inverse') IS NOT NULL
                UNION
                SELECT collection, predicate, inverse
                FROM edge_backfill
                UNION
                SELECT DISTINCT collection, predicate, inverse
                FROM edges
            )
            INSERT OR REPLACE INTO version_edges
                (collection, source_id, version, predicate, target_id, inverse, created)
            SELECT
                v.collection,
                v.id,
                v.version,
                j.key,
                CAST(vv.value AS TEXT),
                p.inverse,
                v.created_at
            FROM document_versions v
            JOIN json_each(v.tags_json) j
            JOIN json_each(
                CASE
                    WHEN j.type = 'array' THEN j.value
                    ELSE json_array(j.value)
                END
            ) vv
            JOIN predicates p
              ON p.collection = v.collection
             AND p.predicate = j.key
            WHERE vv.value IS NOT NULL
              AND TRIM(CAST(vv.value AS TEXT)) != ''
              AND SUBSTR(CAST(vv.value AS TEXT), 1, 1) != '.'
            """)

    def _migrate_to_v11(self) -> None:
        """Version 10 → 11: allow multiple targets per predicate."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS edges_new (
                source_id   TEXT NOT NULL,
                collection  TEXT NOT NULL,
                predicate   TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                inverse     TEXT NOT NULL,
                created     TEXT NOT NULL,
                PRIMARY KEY (source_id, collection, predicate, target_id)
            )
        """)
        self._execute("""
            INSERT OR REPLACE INTO edges_new
                (source_id, collection, predicate, target_id, inverse, created)
            SELECT source_id, collection, predicate, target_id, inverse, created
            FROM edges
        """)
        self._execute("DROP TABLE IF EXISTS edges")
        self._execute("ALTER TABLE edges_new RENAME TO edges")
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_target
            ON edges (target_id, collection, inverse, created)
        """)

        self._execute("""
            CREATE TABLE IF NOT EXISTS version_edges_new (
                collection  TEXT NOT NULL,
                source_id   TEXT NOT NULL,
                version     INTEGER NOT NULL,
                predicate   TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                inverse     TEXT NOT NULL,
                created     TEXT NOT NULL,
                PRIMARY KEY (collection, source_id, version, predicate, target_id)
            )
        """)
        self._execute("""
            INSERT OR REPLACE INTO version_edges_new
                (collection, source_id, version, predicate, target_id, inverse, created)
            SELECT collection, source_id, version, predicate, target_id, inverse, created
            FROM version_edges
        """)
        self._execute("DROP TABLE IF EXISTS version_edges")
        self._execute("ALTER TABLE version_edges_new RENAME TO version_edges")
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_version_edges_target
            ON version_edges (target_id, collection, inverse, created)
        """)

    def _migrate_to_v12(self) -> None:
        """Version 11 → 12: planner outbox table + triggers."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS planner_outbox (
                outbox_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                mutation     TEXT NOT NULL,
                entity_id    TEXT NOT NULL,
                collection   TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now')),
                claimed_by   TEXT,
                claimed_at   TEXT,
                attempts     INTEGER DEFAULT 0
            )
        """)
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_planner_outbox_unclaimed
            ON planner_outbox (claimed_by) WHERE claimed_by IS NULL
        """)

        # --- Document triggers ---
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS planner_outbox_doc_ai
            AFTER INSERT ON documents BEGIN
                INSERT INTO planner_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('doc_insert', new.id, new.collection,
                        json_object('tags_json', new.tags_json));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS planner_outbox_doc_au
            AFTER UPDATE OF tags_json ON documents BEGIN
                INSERT INTO planner_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('doc_update', new.id, new.collection,
                        json_object('old_tags_json', old.tags_json,
                                    'new_tags_json', new.tags_json));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS planner_outbox_doc_ad
            AFTER DELETE ON documents BEGIN
                INSERT INTO planner_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('doc_delete', old.id, old.collection,
                        json_object('tags_json', old.tags_json));
            END
        """)

        # --- Edge triggers ---
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS planner_outbox_edge_ai
            AFTER INSERT ON edges BEGIN
                INSERT INTO planner_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('edge_insert', new.source_id, new.collection,
                        json_object('predicate', new.predicate,
                                    'target_id', new.target_id));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS planner_outbox_edge_ad
            AFTER DELETE ON edges BEGIN
                INSERT INTO planner_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('edge_delete', old.source_id, old.collection,
                        json_object('predicate', old.predicate,
                                    'target_id', old.target_id));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS planner_outbox_edge_au
            AFTER UPDATE ON edges BEGIN
                INSERT INTO planner_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('edge_update', new.source_id, new.collection,
                        json_object('predicate', new.predicate,
                                    'target_id', new.target_id,
                                    'old_target_id', old.target_id));
            END
        """)


    def _migrate_to_v13(self) -> None:
        """Reserved version; no schema changes were required."""
        return

    def _migrate_to_v14(self) -> None:
        """Apply schema migration to user_version 14."""
        columns = {
            row[1]
            for row in self._execute(
                "PRAGMA table_info(document_parts)"
            ).fetchall()
        }
        if "content" in columns:
            changed_rows = self._execute("""
                SELECT id, collection, part_num, summary, tags_json,
                       CASE
                           WHEN COALESCE(content, '') != '' THEN content
                           ELSE summary
                       END AS migrated_summary
                FROM document_parts
                WHERE CASE
                          WHEN COALESCE(content, '') != '' THEN content
                          ELSE summary
                      END != summary
            """).fetchall()
            self.migrated_parts_for_reindex = [
                {
                    "id": row["id"],
                    "collection": row["collection"],
                    "part_num": row["part_num"],
                    "summary": row["migrated_summary"],
                    "tags": _load_tags_json(row["tags_json"]),
                }
                for row in changed_rows
            ]

            self._execute("DROP TRIGGER IF EXISTS parts_fts_ai")
            self._execute("DROP TRIGGER IF EXISTS parts_fts_ad")
            self._execute("DROP TRIGGER IF EXISTS parts_fts_au")
            self._execute("DROP TABLE IF EXISTS parts_fts")

            self._execute("""
                CREATE TABLE document_parts_new (
                    id TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    part_num INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (id, collection, part_num)
                )
            """)
            self._execute("""
                INSERT INTO document_parts_new
                    (id, collection, part_num, summary, tags_json, created_at)
                SELECT
                    id,
                    collection,
                    part_num,
                    CASE
                        WHEN COALESCE(content, '') != '' THEN content
                        ELSE summary
                    END,
                    tags_json,
                    created_at
                FROM document_parts
            """)
            self._execute("DROP TABLE document_parts")
            self._execute(
                "ALTER TABLE document_parts_new RENAME TO document_parts"
            )
            self._execute("""
                CREATE INDEX IF NOT EXISTS idx_parts_doc
                ON document_parts(id, collection, part_num)
            """)

    def _migrate_to_v15(self) -> None:
        """Apply schema migration to user_version 15."""
        self._execute("""
            CREATE TABLE IF NOT EXISTS sync_outbox (
                outbox_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                mutation     TEXT NOT NULL,
                entity_id    TEXT NOT NULL,
                collection   TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now')),
                claimed_by   TEXT,
                claimed_at   TEXT,
                attempts     INTEGER DEFAULT 0
            )
        """)
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_sync_outbox_unclaimed
            ON sync_outbox (claimed_by) WHERE claimed_by IS NULL
        """)

        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_doc_ai
            AFTER INSERT ON documents
            WHEN new.id != '.markdown-mirrors'
            BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('doc_insert', new.id, new.collection,
                        json_object('tags_json', new.tags_json));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_doc_au
            AFTER UPDATE OF summary, tags_json, content_hash, content_hash_full, updated_at
            ON documents
            WHEN new.id != '.markdown-mirrors'
            BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('doc_update', new.id, new.collection,
                        json_object('old_tags_json', old.tags_json,
                                    'new_tags_json', new.tags_json,
                                    'old_summary', old.summary,
                                    'new_summary', new.summary,
                                    'old_content_hash', old.content_hash,
                                    'new_content_hash', new.content_hash,
                                    'old_content_hash_full', old.content_hash_full,
                                    'new_content_hash_full', new.content_hash_full));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_doc_ad
            AFTER DELETE ON documents
            WHEN old.id != '.markdown-mirrors'
            BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('doc_delete', old.id, old.collection,
                        json_object('tags_json', old.tags_json));
            END
        """)

        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_part_ai
            AFTER INSERT ON document_parts BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('part_insert', new.id, new.collection,
                        json_object('part_num', new.part_num));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_part_au
            AFTER UPDATE OF summary, tags_json ON document_parts BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('part_update', new.id, new.collection,
                        json_object('part_num', new.part_num));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_part_ad
            AFTER DELETE ON document_parts BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('part_delete', old.id, old.collection,
                        json_object('part_num', old.part_num));
            END
        """)

        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_version_ai
            AFTER INSERT ON document_versions BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('version_insert', new.id, new.collection,
                        json_object('version', new.version));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_version_au
            AFTER UPDATE OF summary, tags_json, content_hash ON document_versions BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('version_update', new.id, new.collection,
                        json_object('version', new.version));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_version_ad
            AFTER DELETE ON document_versions BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('version_delete', old.id, old.collection,
                        json_object('version', old.version));
            END
        """)

        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_edge_ai
            AFTER INSERT ON edges BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('edge_insert', new.source_id, new.collection,
                        json_object('predicate', new.predicate,
                                    'target_id', new.target_id));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_edge_au
            AFTER UPDATE ON edges BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('edge_update', new.source_id, new.collection,
                        json_object('predicate', new.predicate,
                                    'target_id', new.target_id,
                                    'old_target_id', old.target_id));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_edge_ad
            AFTER DELETE ON edges BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('edge_delete', old.source_id, old.collection,
                        json_object('predicate', old.predicate,
                                    'target_id', old.target_id));
            END
        """)

        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_version_edge_ai
            AFTER INSERT ON version_edges BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('version_edge_insert', new.source_id, new.collection,
                        json_object('version', new.version,
                                    'predicate', new.predicate,
                                    'target_id', new.target_id));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_version_edge_au
            AFTER UPDATE ON version_edges BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('version_edge_update', new.source_id, new.collection,
                        json_object('version', new.version,
                                    'predicate', new.predicate,
                                    'target_id', new.target_id,
                                    'old_target_id', old.target_id));
            END
        """)
        self._execute("""
            CREATE TRIGGER IF NOT EXISTS sync_outbox_version_edge_ad
            AFTER DELETE ON version_edges BEGIN
                INSERT INTO sync_outbox (mutation, entity_id, collection, payload_json)
                VALUES ('version_edge_delete', old.source_id, old.collection,
                        json_object('version', old.version,
                                    'predicate', old.predicate,
                                    'target_id', old.target_id));
            END
        """)

    def _migrate_to_v16(self) -> None:
        """Version 15 → 16: index for inverse-edge find queries."""
        self._execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_inverse_source
            ON edges (collection, inverse, source_id)
        """)

    def _recover_malformed(self) -> None:
        """Attempt to recover a malformed SQLite database.

        Strategy: dump all readable data, rebuild the database from scratch.
        The corrupt file is preserved as .db.corrupt for inspection.

        Raises the original error if recovery fails.
        """
        import shutil

        db_path = str(self._db_path)
        corrupt_path = db_path + ".corrupt"

        # Close any existing per-thread connections before moving files.
        self._close_all_connections(mark_closed=False)

        # Try to dump data from the corrupt database
        try:
            src = sqlite3.connect(db_path)
            dump = list(src.iterdump())
            src.close()
        except Exception as dump_err:
            logger.error("Cannot read data from malformed database: %s", dump_err)
            raise sqlite3.DatabaseError(
                f"Database is malformed and data is unreadable: {self._db_path}"
            ) from dump_err

        # Preserve the corrupt file
        shutil.move(db_path, corrupt_path)
        logger.info("Corrupt database saved to %s", corrupt_path)

        # Remove stale WAL/SHM files
        for suffix in ("-wal", "-shm"):
            p = Path(db_path + suffix)
            if p.exists():
                p.unlink()

        # Rebuild from dump
        dst = sqlite3.connect(db_path)
        for stmt in dump:
            try:
                dst.execute(stmt)
            except Exception:
                pass  # Skip errors from dump replay (e.g. duplicate CREATE)
        dst.commit()
        dst.close()

        logger.warning(
            "Database recovered from %d SQL statements. "
            "Corrupt file preserved at %s",
            len(dump), corrupt_path,
        )

        # Retry normal initialization
        self._init_db()

    def _try_runtime_recover(self) -> bool:
        """Attempt runtime recovery when a malformed error is detected mid-session.

        Returns True if recovery succeeded, False otherwise.
        """
        with self._lock:
            try:
                # A malformed error can be WAL/session-local.  Invalidate every
                # thread-local handle before quick_check so the retry does not
                # reuse the same poisoned sqlite3 connection that reported it.
                logger.warning(
                    "Runtime database malformation detected; reopening SQLite "
                    "connections before health check: %s",
                    self._db_path,
                )
                self._close_all_connections(mark_closed=False)
                try:
                    result = self._conn.execute("PRAGMA quick_check").fetchone()
                except sqlite3.DatabaseError as quick_check_err:
                    if not is_malformed_db_error(quick_check_err):
                        raise
                else:
                    if result and result[0] == "ok":
                        logger.info(
                            "Runtime recovery completed by reopening healthy database: %s",
                            self._db_path,
                        )
                        return True

                logger.warning(
                    "Runtime database still malformed after reopen, attempting recovery: %s",
                    self._db_path,
                )
                self._recover_malformed()
                logger.warning("Runtime recovery succeeded")
                return True
            except Exception as e:
                logger.error("Runtime recovery failed: %s", e)
                return False

    @staticmethod
    def _now() -> str:
        """Current timestamp in canonical UTC format."""
        return utc_now()

    def _get_unlocked(self, collection: str, id: str) -> Optional[DocumentRecord]:
        """Get a document by ID without acquiring the lock (for use within locked contexts).

        With RLock, this can safely use _execute (re-entrant).
        """
        row = self._fetchone("""
            SELECT id, collection, summary, tags_json, created_at, updated_at,
                   content_hash, content_hash_full, accessed_at
            FROM documents
            WHERE id = ? AND collection = ?
        """, (id, collection))
        if row is None:
            return None

        return DocumentRecord(
            id=row["id"],
            collection=row["collection"],
            summary=row["summary"],
            tags=_load_tags_json(row["tags_json"], doc_id=row["id"], collection=row["collection"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            content_hash=row["content_hash"],
            content_hash_full=row["content_hash_full"],
            accessed_at=row["accessed_at"],
        )

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------
    
    def upsert(
        self,
        collection: str,
        id: str,
        summary: str,
        tags: dict[str, Any],
        content_hash: Optional[str] = None,
        content_hash_full: Optional[str] = None,
        created_at: Optional[str] = None,
        archive: bool = True,
    ) -> tuple[DocumentRecord, bool]:
        """Insert or update a document record.

        Preserves created_at on update. Updates updated_at always.
        Archives the current version to history before updating (unless
        archive=False, which updates in-place without creating a version).

        Args:
            collection: Collection name
            id: Document identifier (URI or custom)
            summary: Document summary text
            tags: All tags (source + system)
            content_hash: Short SHA256 hash of content (for change detection)
            content_hash_full: Full SHA256 hash (for dedup verification)
            created_at: Optional override for created_at timestamp
                        (for importing historical data with original timestamps)
            archive: If True (default), archive the current version before
                     updating. If False, update in-place without creating
                     a version entry.

        Returns:
            Tuple of (stored DocumentRecord, content_changed bool).
            content_changed is True if content hash differs from previous,
            False if only tags/summary changed or if new document.
        """
        now = self._now()
        summary = repair_surrogate_text(summary)
        tags = normalize_tag_map(tags)
        tags_json = json.dumps(tags, ensure_ascii=False)

        with self._lock:
            # Use BEGIN IMMEDIATE for cross-process atomicity:
            # holds a write lock for the entire read-archive-replace sequence
            self._execute("BEGIN IMMEDIATE")
            try:
                # Check if exists to preserve created_at and archive
                existing = self._get_unlocked(collection, id)
                # If caller provides created_at, honour it (vstring ingest
                # passes per-version original dates).  Otherwise fall back to
                # the existing row's created_at, or now for brand-new docs.
                created_at = created_at or (existing.created_at if existing else now)
                content_changed = False

                if existing:
                    # Archive current version before updating
                    if archive:
                        self._archive_current_unlocked(collection, id, existing)
                    # Detect content change
                    content_changed = (
                        content_hash is not None
                        and existing.content_hash != content_hash
                    )

                self._execute("""
                    INSERT OR REPLACE INTO documents
                    (id, collection, summary, tags_json, created_at, updated_at,
                     content_hash, content_hash_full, accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (id, collection, summary, tags_json, created_at, now,
                      content_hash, content_hash_full, now))
                self._conn.commit()
                if id == ".stop":
                    self._stopwords = None
            except Exception:
                self._conn.rollback()
                raise

        return DocumentRecord(
            id=id,
            collection=collection,
            summary=summary,
            tags=tags,
            created_at=created_at,
            updated_at=now,
            content_hash=content_hash,
            content_hash_full=content_hash_full,
            accessed_at=now,
        ), content_changed

    def _archive_current_unlocked(
        self,
        collection: str,
        id: str,
        current: DocumentRecord,
    ) -> int:
        """Archive the current version to the versions table.

        Must be called within a lock context.

        Args:
            collection: Collection name
            id: Document identifier
            current: Current document record to archive

        Returns:
            The version number assigned to the archived version
        """
        # Get the next version number
        cursor = self._execute("""
            SELECT COALESCE(MAX(version), 0) + 1
            FROM document_versions
            WHERE id = ? AND collection = ?
        """, (id, collection))
        next_version = cursor.fetchone()[0]

        # Insert the current state as a version.
        # Inject _created/_updated into tags so version nav can display
        # accurate timestamps (these are normally synthesized by
        # _record_to_item but not stored in tags_json).
        version_tags = dict(current.tags)
        if current.created_at:
            version_tags.setdefault("_created", current.created_at)
        if current.updated_at:
            version_tags["_updated"] = current.updated_at
        version_tags = normalize_tag_map(version_tags)

        self._execute("""
            INSERT INTO document_versions
            (id, collection, version, summary, tags_json, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            id,
            collection,
            next_version,
            current.summary,
            json.dumps(version_tags, ensure_ascii=False),
            current.content_hash,
            current.updated_at,  # Use updated_at as the version's timestamp
        ))
        self._materialize_version_edges_for_version_unlocked(
            collection=collection,
            source_id=id,
            version=next_version,
            tags=version_tags,
            created=current.updated_at or self._now(),
        )

        return next_version

    def _edge_predicate_map_unlocked(self, collection: str) -> dict[str, str]:
        """Return predicate -> inverse for edge-enabled tags in *collection*."""
        mapping: dict[str, str] = {}

        # Source of truth: tagdocs with _inverse.
        rows = self._execute(
            """
            SELECT SUBSTR(id, 6) AS predicate,
                   CAST(json_extract(tags_json, '$._inverse') AS TEXT) AS inverse
            FROM documents
            WHERE collection = ?
              AND id LIKE '.tag/%'
              AND INSTR(SUBSTR(id, 6), '/') = 0
              AND json_extract(tags_json, '$._inverse') IS NOT NULL
            """,
            (collection,),
        ).fetchall()
        for row in rows:
            pred = row["predicate"]
            inv = row["inverse"]
            if pred and inv:
                mapping[pred] = inv

        # Compatibility: fall back to existing backfill/edge rows.
        rows = self._execute(
            """
            SELECT predicate, inverse
            FROM edge_backfill
            WHERE collection = ?
            """,
            (collection,),
        ).fetchall()
        for row in rows:
            pred = row["predicate"]
            inv = row["inverse"]
            if pred and inv and pred not in mapping:
                mapping[pred] = inv

        rows = self._execute(
            """
            SELECT DISTINCT predicate, inverse
            FROM edges
            WHERE collection = ?
            """,
            (collection,),
        ).fetchall()
        for row in rows:
            pred = row["predicate"]
            inv = row["inverse"]
            if pred and inv and pred not in mapping:
                mapping[pred] = inv

        return mapping

    def _version_edge_target_id(self, value: Any) -> str | None:
        """Resolve one version-edge tag value to its canonical target ID."""
        raw = str(value).strip()
        if not raw:
            return None
        target_id = parse_ref(raw)[0]
        if not target_id or target_id.startswith("."):
            return None
        try:
            return normalize_id(target_id)
        except ValueError:
            return None

    def _materialize_version_edges_for_version_unlocked(
        self,
        *,
        collection: str,
        source_id: str,
        version: int,
        tags: dict[str, Any],
        created: str,
    ) -> None:
        """Materialize version edge rows for one archived version."""
        predicate_map = self._edge_predicate_map_unlocked(collection)
        self._execute(
            """
            DELETE FROM version_edges
            WHERE collection = ? AND source_id = ? AND version = ?
            """,
            (collection, source_id, version),
        )
        if not predicate_map:
            return

        for key in tags:
            if key.startswith("_"):
                continue
            inverse = predicate_map.get(key)
            if not inverse:
                continue
            for value in tag_values(tags, key):
                target_id = self._version_edge_target_id(value)
                if not target_id:
                    continue
                self._execute(
                    """
                    INSERT OR REPLACE INTO version_edges
                        (collection, source_id, version, predicate, target_id, inverse, created)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (collection, source_id, version, key, target_id, inverse, created),
                )

    def _rebuild_version_edges_for_source_unlocked(
        self, collection: str, source_id: str
    ) -> None:
        """Rebuild all materialized version edges for one source document."""
        self._execute(
            "DELETE FROM version_edges WHERE collection = ? AND source_id = ?",
            (collection, source_id),
        )
        predicate_map = self._edge_predicate_map_unlocked(collection)
        if not predicate_map:
            return

        rows = self._execute(
            """
            SELECT version, tags_json, created_at
            FROM document_versions
            WHERE collection = ? AND id = ?
            """,
            (collection, source_id),
        ).fetchall()
        # This used to be a pure-SQL INSERT...SELECT path, but version-edge
        # targets can now be labeled refs. We have to parse_ref() each value in
        # Python before storing the canonical target_id.
        for row in rows:
            tags = _load_tags_json(row["tags_json"])
            for key in tags:
                if key.startswith("_"):
                    continue
                inverse = predicate_map.get(key)
                if not inverse:
                    continue
                for value in tag_values(tags, key):
                    target_id = self._version_edge_target_id(value)
                    if not target_id:
                        continue
                    self._execute(
                        """
                        INSERT OR REPLACE INTO version_edges
                            (collection, source_id, version, predicate, target_id, inverse, created)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            collection, source_id, row["version"], key,
                            target_id, inverse, row["created_at"],
                        ),
                    )
    
    def update_summary(self, collection: str, id: str, summary: str) -> bool:
        """Update just the summary of an existing document.
        
        Used by lazy summarization to replace placeholder summaries.
        
        Args:
            collection: Collection name
            id: Document identifier
            summary: New summary text
            
        Returns:
            True if document was found and updated, False otherwise
        """
        now = self._now()
        summary = repair_surrogate_text(summary)

        with self._lock:
            cursor = self._execute("""
                UPDATE documents
                SET summary = ?, updated_at = ?
                WHERE id = ? AND collection = ?
            """, (summary, now, id, collection))
            self._conn.commit()

        return cursor.rowcount > 0

    def update_content_hash(
        self,
        collection: str,
        id: str,
        content_hash: str,
        content_hash_full: str,
    ) -> bool:
        """Update content hashes in-place without archiving a version.

        Used by background OCR to replace the placeholder hash after
        real content has been extracted.
        """
        now = self._now()
        with self._lock:
            cursor = self._execute("""
                UPDATE documents
                SET content_hash = ?, content_hash_full = ?, updated_at = ?
                WHERE id = ? AND collection = ?
            """, (content_hash, content_hash_full, now, id, collection))
            self._conn.commit()
        return cursor.rowcount > 0

    def update_tags(
        self,
        collection: str,
        id: str,
        tags: dict[str, Any],
    ) -> bool:
        """Update tags of an existing document.

        Args:
            collection: Collection name
            id: Document identifier
            tags: New tags dict (replaces existing)

        Returns:
            True if document was found and updated, False otherwise
        """
        now = self._now()
        tags = normalize_tag_map(tags)
        tags_json = json.dumps(tags, ensure_ascii=False)

        with self._lock:
            cursor = self._execute("""
                UPDATE documents
                SET tags_json = ?, updated_at = ?
                WHERE id = ? AND collection = ?
            """, (tags_json, now, id, collection))
            self._conn.commit()

        return cursor.rowcount > 0

    def touch(self, collection: str, id: str) -> None:
        """Update accessed_at timestamp without changing updated_at.

        Non-fatal: logs errors instead of raising, since touch is a
        side-effect that should never prevent read operations.
        """
        now = self._now()
        try:
            with self._lock:
                self._execute("""
                    UPDATE documents SET accessed_at = ?
                    WHERE id = ? AND collection = ?
                """, (now, id, collection))
                self._conn.commit()
        except sqlite3.DatabaseError as e:
            logger.warning("touch(%s) failed (non-fatal): %s", id, e)
            if is_malformed_db_error(e):
                self._try_runtime_recover()

    def touch_many(self, collection: str, ids: list[str]) -> None:
        """Update accessed_at for multiple documents in one statement."""
        if not ids:
            return
        now = self._now()
        with self._lock:
            placeholders = ",".join("?" * len(ids))
            self._execute(f"""
                UPDATE documents SET accessed_at = ?
                WHERE collection = ? AND id IN ({placeholders})
            """, (now, collection, *ids))
            self._conn.commit()

    def restore_latest_version(self, collection: str, id: str) -> Optional[DocumentRecord]:
        """Restore the most recent archived version as current.

        Replaces the current document with the latest version from history,
        then deletes that version row.

        Returns:
            The restored DocumentRecord, or None if no versions exist.
        """
        with self._lock:
            self._execute("BEGIN IMMEDIATE")
            try:
                # Get the most recent archived version
                cursor = self._execute("""
                    SELECT version, summary, tags_json, content_hash, created_at
                    FROM document_versions
                    WHERE id = ? AND collection = ?
                    ORDER BY version DESC LIMIT 1
                """, (id, collection))
                row = cursor.fetchone()
                if row is None:
                    self._conn.rollback()
                    return None

                version = row["version"]
                summary = row["summary"]
                tags = normalize_tag_map(_load_tags_json(row["tags_json"]))
                content_hash = row["content_hash"]
                created_at = row["created_at"]

                # Get the original created_at from the current document
                existing = self._get_unlocked(collection, id)
                original_created_at = existing.created_at if existing else created_at

                now = self._now()
                # Replace current document with the archived version
                self._execute("""
                    INSERT OR REPLACE INTO documents
                    (id, collection, summary, tags_json, created_at, updated_at, content_hash, accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (id, collection, summary, json.dumps(tags, ensure_ascii=False),
                      original_created_at, created_at, content_hash, now))

                # Delete the version row we just restored
                self._execute("""
                    DELETE FROM document_versions
                    WHERE id = ? AND collection = ? AND version = ?
                """, (id, collection, version))
                self._execute(
                    """
                    DELETE FROM version_edges
                    WHERE collection = ? AND source_id = ? AND version = ?
                    """,
                    (collection, id, version),
                )

                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return DocumentRecord(
            id=id, collection=collection, summary=summary,
            tags=tags, created_at=original_created_at,
            updated_at=created_at, content_hash=content_hash,
            accessed_at=now,
        )

    def delete_version(self, collection: str, id: str, version: int) -> bool:
        """Delete a specific archived version by version number.

        Other versions are unaffected; gaps in version numbering are
        handled naturally by offset-based queries.

        Args:
            collection: Collection name
            id: Document identifier
            version: Internal version number (from VersionInfo.version)

        Returns:
            True if the version existed and was deleted
        """
        with self._lock:
            cursor = self._execute(
                "DELETE FROM document_versions"
                " WHERE id = ? AND collection = ? AND version = ?",
                (id, collection, version),
            )
            self._execute(
                """
                DELETE FROM version_edges
                WHERE collection = ? AND source_id = ? AND version = ?
                """,
                (collection, id, version),
            )
            return cursor.rowcount > 0

    def replace_version_content(
        self,
        collection: str,
        id: str,
        version: int,
        summary: str,
        tags: dict[str, Any],
        content_hash: Optional[str] = None,
    ) -> bool:
        """Replace the content of a specific archived version in-place.

        Used to update the bundled base version of system docs without
        affecting user override versions above it.

        Returns:
            True if the version existed and was updated.
        """
        summary = repair_surrogate_text(summary)
        tags = normalize_tag_map(tags)
        tags_json = json.dumps(tags, ensure_ascii=False)
        with self._lock:
            cursor = self._execute("""
                UPDATE document_versions
                SET summary = ?, tags_json = ?, content_hash = ?
                WHERE id = ? AND collection = ? AND version = ?
            """, (summary, tags_json, content_hash, id, collection, version))
            return cursor.rowcount > 0

    def find_version_by_content_hash(
        self,
        collection: str,
        id: str,
        content_hash: str,
    ) -> Optional[int]:
        """Find the oldest archived version with a given content_hash.

        Returns:
            The version number, or None if not found.
        """
        row = self._fetchone("""
            SELECT version FROM document_versions
            WHERE id = ? AND collection = ? AND content_hash = ?
            ORDER BY version ASC LIMIT 1
        """, (id, collection, content_hash))
        return row["version"] if row else None

    def get_latest_version_info_by_content_hash(
        self,
        collection: str,
        id: str,
        content_hash: str,
    ) -> Optional[VersionInfo]:
        """Return the newest archived version matching *content_hash*."""
        row = self._fetchone("""
            SELECT version, summary, tags_json, content_hash, created_at
            FROM document_versions
            WHERE id = ? AND collection = ? AND content_hash = ?
            ORDER BY version DESC
            LIMIT 1
        """, (id, collection, content_hash))
        if row is None:
            return None
        return VersionInfo(
            version=row["version"],
            summary=row["summary"],
            tags=_load_tags_json(row["tags_json"]),
            created_at=row["created_at"],
            content_hash=row["content_hash"],
        )

    def delete_all_versions(self, collection: str, id: str) -> int:
        """Delete all archived versions for a document.

        Returns:
            Number of versions deleted.
        """
        with self._lock:
            cursor = self._execute("""
                DELETE FROM document_versions
                WHERE id = ? AND collection = ?
            """, (id, collection))
            self._execute("""
                DELETE FROM version_edges
                WHERE collection = ? AND source_id = ?
            """, (collection, id))
            return cursor.rowcount

    def patch_head_tags(
        self,
        collection: str,
        id: str,
        patch: dict[str, Any],
    ) -> bool:
        """Merge patch into the head document's tags without creating a version.

        Only updates tags; content and timestamps are unchanged.

        Returns:
            True if the document existed and was updated.
        """
        with self._lock:
            existing = self._get_unlocked(collection, id)
            if not existing:
                return False
            tags = dict(existing.tags)
            tags.update(patch)
            tags = normalize_tag_map(tags)
            self._execute("""
                UPDATE documents SET tags_json = ?
                WHERE id = ? AND collection = ?
            """, (json.dumps(tags, ensure_ascii=False), id, collection))
            return True

    def delete(self, collection: str, id: str, delete_versions: bool = True) -> bool:
        """Delete a document record and optionally its version history.

        Note: this does NOT clean up edge rows. Callers that need edge
        consistency (e.g. Keeper.delete) must call delete_edges_for_source
        and delete_edges_for_target separately.

        Args:
            collection: Collection name
            id: Document identifier
            delete_versions: If True, also delete version history

        Returns:
            True if document existed and was deleted
        """
        with self._lock:
            self._execute("BEGIN IMMEDIATE")
            try:
                cursor = self._execute("""
                    DELETE FROM documents
                    WHERE id = ? AND collection = ?
                """, (id, collection))

                if delete_versions:
                    self._execute("""
                        DELETE FROM document_versions
                        WHERE id = ? AND collection = ?
                    """, (id, collection))
                    self._execute("""
                        DELETE FROM version_edges
                        WHERE collection = ? AND source_id = ?
                    """, (collection, id))

                # Always clean up parts (structural decomposition)
                self._execute("""
                    DELETE FROM document_parts
                    WHERE id = ? AND collection = ?
                """, (id, collection))

                self._conn.commit()
                if id == ".stop":
                    self._stopwords = None
            except Exception:
                self._conn.rollback()
                raise

        return cursor.rowcount > 0
    
    def extract_versions(
        self,
        collection: str,
        source_id: str,
        target_id: str,
        tag_filter: Optional[dict[str, str]] = None,
        only_current: bool = False,
    ) -> tuple[list[VersionInfo], Optional[DocumentRecord], int]:
        """Extract matching versions from source into a target document.

        Moves matching archived versions (and optionally the current document)
        from source_id to target_id. If target already exists, its current is
        archived and the extracted versions are appended on top. Source retains
        non-matching versions (gaps are tolerated).

        Args:
            collection: Collection name
            source_id: Document to extract from
            target_id: Document to create or extend
            tag_filter: If provided, only extract versions whose tags
                        contain all specified key=value pairs.
                        If None, extract everything.
            only_current: If True, only extract the current (tip) version,
                        not any archived history.

        Returns:
            Tuple of (extracted_versions, new_source_current_or_None, base_version).
            extracted_versions: list of VersionInfo that were moved to target.
            new_source_current: the new current state of the source document
                after extraction, or None if source was fully emptied.
            base_version: the starting version number used for the extracted
                history in the target (1 for new targets, higher for appends).

        Raises:
            ValueError: If source_id doesn't exist or no versions match.
        """
        def _tags_match(tags: dict, filt: dict) -> bool:
            for key in filt:
                wanted = set(tag_values(filt, key))
                if not wanted:
                    continue
                stored = set(tag_values(tags, key))
                if not wanted.issubset(stored):
                    return False
            return True

        with self._lock:
            self._execute("BEGIN IMMEDIATE")
            try:
                # Validate source
                source = self._get_unlocked(collection, source_id)
                if source is None:
                    raise ValueError(f"Source document '{source_id}' not found")

                # Check if target already exists (append mode)
                existing_target = self._get_unlocked(collection, target_id)

                # Get all archived versions (oldest first for sequential renumbering)
                cursor = self._execute("""
                    SELECT version, summary, tags_json, content_hash, created_at
                    FROM document_versions
                    WHERE id = ? AND collection = ?
                    ORDER BY version ASC
                """, (source_id, collection))
                all_versions = []
                for row in cursor:
                    all_versions.append(VersionInfo(
                        version=row["version"],
                        summary=row["summary"],
                        tags=_load_tags_json(row["tags_json"]),
                        created_at=row["created_at"],
                        content_hash=row["content_hash"],
                    ))

                # Partition: matching vs remaining
                if only_current:
                    # Only extract the current (tip) version, skip all history
                    matching_versions = []
                    if tag_filter:
                        current_matches = _tags_match(source.tags, tag_filter)
                    else:
                        current_matches = True
                elif tag_filter:
                    matching_versions = [v for v in all_versions if _tags_match(v.tags, tag_filter)]
                    current_matches = _tags_match(source.tags, tag_filter)
                else:
                    matching_versions = list(all_versions)
                    current_matches = True

                # Build the full list of extracted items (versions + possibly current)
                extracted: list[VersionInfo] = list(matching_versions)
                if current_matches:
                    # Current becomes the newest extracted item
                    extracted.append(VersionInfo(
                        version=0,  # placeholder, will be renumbered
                        summary=source.summary,
                        tags=source.tags,
                        created_at=source.updated_at,
                        content_hash=source.content_hash,
                    ))

                if not extracted:
                    raise ValueError("No versions match the tag filter")

                # Delete matching archived versions from source
                if matching_versions:
                    version_nums = [v.version for v in matching_versions]
                    placeholders = ",".join("?" * len(version_nums))
                    self._execute(f"""
                        DELETE FROM document_versions
                        WHERE id = ? AND collection = ? AND version IN ({placeholders})
                    """, (source_id, collection, *version_nums))
                    self._execute(f"""
                        DELETE FROM version_edges
                        WHERE collection = ? AND source_id = ? AND version IN ({placeholders})
                    """, (collection, source_id, *version_nums))

                # Determine base version for target history
                now = self._now()
                if existing_target:
                    # Archive existing target's current into its history
                    self._archive_current_unlocked(collection, target_id, existing_target)
                    # Get the next version number after archiving
                    cursor = self._execute("""
                        SELECT COALESCE(MAX(version), 0) + 1
                        FROM document_versions
                        WHERE id = ? AND collection = ?
                    """, (target_id, collection))
                    base_version = cursor.fetchone()[0]
                else:
                    base_version = 1

                # extracted is in chronological order (oldest first)
                target_current = extracted[-1]  # newest
                target_history = extracted[:-1]  # older ones

                # Insert or update target current in documents table
                self._execute("""
                    INSERT OR REPLACE INTO documents
                    (id, collection, summary, tags_json, created_at, updated_at, content_hash, accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    target_id, collection, target_current.summary,
                    json.dumps(normalize_tag_map(target_current.tags), ensure_ascii=False),
                    existing_target.created_at if existing_target else target_current.created_at,
                    now, target_current.content_hash, now,
                ))

                # Insert target version history with sequential numbering
                for seq, vi in enumerate(target_history, start=base_version):
                    self._execute("""
                        INSERT INTO document_versions
                        (id, collection, version, summary, tags_json, content_hash, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        target_id, collection, seq, vi.summary,
                        json.dumps(normalize_tag_map(vi.tags), ensure_ascii=False),
                        vi.content_hash, vi.created_at,
                    ))

                # Handle source after extraction
                new_source: Optional[DocumentRecord] = None
                if current_matches:
                    # Source's current was extracted — need to promote or delete
                    remaining_versions = [v for v in all_versions if v not in matching_versions]
                    if remaining_versions:
                        # Promote newest remaining to current
                        promote = remaining_versions[-1]  # already sorted ASC
                        self._execute("""
                            UPDATE documents
                            SET summary = ?, tags_json = ?, updated_at = ?,
                                content_hash = ?, accessed_at = ?
                            WHERE id = ? AND collection = ?
                        """, (
                            promote.summary,
                            json.dumps(normalize_tag_map(promote.tags), ensure_ascii=False),
                            promote.created_at, promote.content_hash, now,
                            source_id, collection,
                        ))
                        # Delete the promoted version from history
                        self._execute("""
                            DELETE FROM document_versions
                            WHERE id = ? AND collection = ? AND version = ?
                        """, (source_id, collection, promote.version))
                        self._execute("""
                            DELETE FROM version_edges
                            WHERE collection = ? AND source_id = ? AND version = ?
                        """, (collection, source_id, promote.version))
                        new_source = self._get_unlocked(collection, source_id)
                    else:
                        # Nothing remains — delete source
                        self._execute("""
                            DELETE FROM documents WHERE id = ? AND collection = ?
                        """, (source_id, collection))
                else:
                    # Source current was not extracted — it stays
                    new_source = self._get_unlocked(collection, source_id)

                # Rebuild materialized version edges after renumbering/moves.
                self._rebuild_version_edges_for_source_unlocked(collection, target_id)
                self._rebuild_version_edges_for_source_unlocked(collection, source_id)

                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return extracted, new_source, base_version

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------

    def get(self, collection: str, id: str) -> Optional[DocumentRecord]:
        """Get a document by ID.

        Args:
            collection: Collection name
            id: Document identifier

        Returns:
            DocumentRecord if found, None otherwise
        """
        row = self._fetchone("""
            SELECT id, collection, summary, tags_json, created_at, updated_at,
                   content_hash, content_hash_full, accessed_at
            FROM documents
            WHERE id = ? AND collection = ?
        """, (id, collection))
        if row is None:
            return None

        return DocumentRecord(
            id=row["id"],
            collection=row["collection"],
            summary=row["summary"],
            tags=_load_tags_json(row["tags_json"], doc_id=row["id"], collection=row["collection"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            content_hash=row["content_hash"],
            content_hash_full=row["content_hash_full"],
            accessed_at=row["accessed_at"],
        )

    def get_version(
        self,
        collection: str,
        id: str,
        offset: int = 0,
    ) -> Optional[VersionInfo]:
        """Get a specific version of a document by offset.

        Offset semantics:
        - 0 = current version (returns None, use get() instead)
        - 1 = previous version (most recent archived)
        - 2 = two versions ago
        - etc.

        Args:
            collection: Collection name
            id: Document identifier
            offset: Version offset (0=current, 1=previous, etc.)

        Returns:
            VersionInfo if found, None if offset 0 or version doesn't exist
        """
        if offset == 0:
            # Offset 0 means current - caller should use get()
            return None

        # Use OFFSET query to handle gaps in version numbering.
        # offset=1 → OFFSET 0 (newest archived), offset=2 → OFFSET 1, etc.
        row = self._fetchone("""
            SELECT version, summary, tags_json, content_hash, created_at
            FROM document_versions
            WHERE id = ? AND collection = ?
            ORDER BY version DESC
            LIMIT 1 OFFSET ?
        """, (id, collection, offset - 1))
        if row is None:
            return None

        return VersionInfo(
            version=row["version"],
            summary=row["summary"],
            tags=_load_tags_json(row["tags_json"]),
            created_at=row["created_at"],
            content_hash=row["content_hash"],
        )

    def get_version_by_number(
        self,
        collection: str,
        id: str,
        version: int,
    ) -> Optional[VersionInfo]:
        """Get a specific archived version by version number."""
        row = self._fetchone("""
            SELECT version, summary, tags_json, content_hash, created_at
            FROM document_versions
            WHERE id = ? AND collection = ? AND version = ?
        """, (id, collection, version))
        if row is None:
            return None

        return VersionInfo(
            version=row["version"],
            summary=row["summary"],
            tags=_load_tags_json(row["tags_json"]),
            created_at=row["created_at"],
            content_hash=row["content_hash"],
        )

    def list_versions(
        self,
        collection: str,
        id: str,
        limit: int = 10,
    ) -> list[VersionInfo]:
        """List version history for a document.

        Returns versions in reverse chronological order (newest first).

        Args:
            collection: Collection name
            id: Document identifier
            limit: Maximum versions to return

        Returns:
            List of VersionInfo, newest archived first
        """
        # limit <= 0 means no limit (SQLite LIMIT -1 = unlimited)
        effective_limit = limit if limit > 0 else -1
        rows = self._fetchall("""
            SELECT version, summary, tags_json, content_hash, created_at
            FROM document_versions
            WHERE id = ? AND collection = ?
            ORDER BY version DESC
            LIMIT ?
        """, (id, collection, effective_limit))

        versions = []
        for row in rows:
            versions.append(VersionInfo(
                version=row["version"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
                content_hash=row["content_hash"],
            ))

        return versions

    def list_versions_many(
        self,
        collection: str,
        ids: list[str],
    ) -> dict[str, list[VersionInfo]]:
        """List all archived versions for multiple documents.

        Returns versions grouped by document ID, newest archived first within
        each document.
        """
        if not ids:
            return {}

        placeholders = ",".join("?" * len(ids))
        rows = self._fetchall(f"""
            SELECT id, version, summary, tags_json, content_hash, created_at
            FROM document_versions
            WHERE collection = ? AND id IN ({placeholders})
            ORDER BY id, version DESC
        """, (collection, *ids))

        versions_by_id: dict[str, list[VersionInfo]] = {}
        for row in rows:
            versions_by_id.setdefault(row["id"], []).append(VersionInfo(
                version=row["version"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
                content_hash=row["content_hash"],
            ))

        return versions_by_id

    def list_versions_around(
        self,
        collection: str,
        id: str,
        version: int,
        radius: int = 2,
    ) -> list[VersionInfo]:
        """Return versions within `radius` of `version`, in chronological order.

        Fetches up to `radius` versions before and after the specified version
        number, useful for showing surrounding context when a version is hit
        during search.
        """
        rows = self._fetchall("""
            SELECT version, summary, tags_json, content_hash, created_at
            FROM document_versions
            WHERE id = ? AND collection = ? AND version BETWEEN ? AND ?
            ORDER BY version ASC
        """, (id, collection, version - radius, version + radius))

        return [VersionInfo(
            version=row["version"],
            summary=row["summary"],
            tags=_load_tags_json(row["tags_json"]),
            created_at=row["created_at"],
            content_hash=row["content_hash"],
        ) for row in rows]

    def get_version_nav(
        self,
        collection: str,
        id: str,
        current_version: Optional[int] = None,
        limit: int = 3,
    ) -> dict[str, list[VersionInfo]]:
        """Get version navigation info (prev/next) for display.

        Args:
            collection: Collection name
            id: Document identifier
            current_version: The version being viewed (None = current/live version)
            limit: Max previous versions to return when viewing current

        Returns:
            Dict with 'prev' and optionally 'next' lists of VersionInfo.
            When viewing current (None): {'prev': [up to limit versions]}
            When viewing old version N: {'prev': [N-1 if exists], 'next': [N+1 if exists]}
        """
        result: dict[str, list[VersionInfo]] = {"prev": []}

        if current_version is None:
            # Viewing current version: get up to `limit` previous versions
            versions = self.list_versions(collection, id, limit=limit)
            result["prev"] = versions
        else:
            # Viewing an old version: get prev (N-1) and next (N+1)
            # Previous version (older)
            if current_version > 1:
                row = self._fetchone("""
                    SELECT version, summary, tags_json, content_hash, created_at
                    FROM document_versions
                    WHERE id = ? AND collection = ? AND version = ?
                """, (id, collection, current_version - 1))
                if row:
                    result["prev"] = [VersionInfo(
                        version=row["version"],
                        summary=row["summary"],
                        tags=_load_tags_json(row["tags_json"]),
                        created_at=row["created_at"],
                        content_hash=row["content_hash"],
                    )]

            # Next version (newer)
            row = self._fetchone("""
                SELECT version, summary, tags_json, content_hash, created_at
                FROM document_versions
                WHERE id = ? AND collection = ? AND version = ?
            """, (id, collection, current_version + 1))
            if row:
                result["next"] = [VersionInfo(
                    version=row["version"],
                    summary=row["summary"],
                    tags=_load_tags_json(row["tags_json"]),
                    created_at=row["created_at"],
                    content_hash=row["content_hash"],
                )]
            else:
                # Check if there's a current version (meaning we're at newest archived)
                if self.exists(collection, id):
                    # Next is "current" - indicate this with empty next
                    # (caller knows to check current doc)
                    result["next"] = []

        return result

    def version_count(self, collection: str, id: str) -> int:
        """Count archived versions for a document."""
        row = self._fetchone("""
            SELECT COUNT(*) FROM document_versions
            WHERE id = ? AND collection = ?
        """, (id, collection))
        return row[0]

    def max_version(self, collection: str, id: str) -> int:
        """Return the highest archived version number, or 0 if none."""
        row = self._fetchone("""
            SELECT COALESCE(MAX(version), 0) FROM document_versions
            WHERE id = ? AND collection = ?
        """, (id, collection))
        return row[0]

    def count_versions_from(
        self, collection: str, id: str, from_version: int
    ) -> int:
        """Count archived versions with version >= from_version."""
        row = self._fetchone("""
            SELECT COUNT(*) FROM document_versions
            WHERE id = ? AND collection = ? AND version >= ?
        """, (id, collection, from_version))
        return row[0]

    def copy_record(
        self, collection: str, from_id: str, to_id: str
    ) -> Optional["DocumentRecord"]:
        """Copy a document record to a new ID, preserving all fields including timestamps.

        Returns the new DocumentRecord, or None if source not found.
        Does nothing if to_id already exists.
        """
        with self._lock:
            # Check source exists
            source = self.get(collection, from_id)
            if source is None:
                return None
            # Check target doesn't exist
            if self.get(collection, to_id) is not None:
                return self.get(collection, to_id)
            # Copy with original timestamps
            import json
            tags_json = json.dumps(normalize_tag_map(source.tags), ensure_ascii=False)
            self._execute("""
                INSERT OR REPLACE INTO documents
                (id, collection, summary, tags_json, created_at, updated_at,
                 content_hash, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (to_id, collection, source.summary, tags_json,
                  source.created_at, source.updated_at,
                  source.content_hash, source.accessed_at))
            self._conn.commit()
            return self.get(collection, to_id)

    def get_many(
        self,
        collection: str,
        ids: list[str],
    ) -> dict[str, DocumentRecord]:
        """Get multiple documents by ID.

        Args:
            collection: Collection name
            ids: List of document identifiers

        Returns:
            Dict mapping id → DocumentRecord (missing IDs omitted)
        """
        if not ids:
            return {}

        placeholders = ",".join("?" * len(ids))
        cursor = self._execute(f"""
            SELECT id, collection, summary, tags_json, created_at, updated_at, content_hash, accessed_at
            FROM documents
            WHERE collection = ? AND id IN ({placeholders})
        """, (collection, *ids))

        results = {}
        for row in cursor:
            results[row["id"]] = DocumentRecord(
                id=row["id"],
                collection=row["collection"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                content_hash=row["content_hash"],
                accessed_at=row["accessed_at"],
            )

        return results

    def exists(self, collection: str, id: str) -> bool:
        """Check if a document exists."""
        row = self._fetchone("""
            SELECT 1 FROM documents
            WHERE id = ? AND collection = ?
        """, (id, collection))
        return row is not None

    def insert_if_absent(
        self,
        collection: str,
        id: str,
        summary: str,
        tags: dict[str, Any],
        created_at: Optional[str] = None,
    ) -> bool:
        """Insert a document only if it doesn't already exist.

        Atomic INSERT OR IGNORE — avoids the TOCTOU race of
        exists() + upsert() where a concurrent writer could create
        the real document between check and write.

        Returns:
            True if a new row was inserted, False if it already existed.
        """
        now = self._now()
        tags = normalize_tag_map(tags)
        tags_json = json.dumps(tags, ensure_ascii=False)
        ts = created_at or now

        with self._lock:
            cursor = self._conn.execute("""
                INSERT OR IGNORE INTO documents
                    (id, collection, summary, tags_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (id, collection, summary, tags_json, ts, now))
            self._conn.commit()
        return cursor.rowcount > 0

    def find_by_content_hash(
        self,
        collection: str,
        content_hash: str,
        *,
        content_hash_full: str = "",
        exclude_id: str = "",
        limit: int = 1,
    ) -> Optional[DocumentRecord] | list[DocumentRecord]:
        """Find documents with matching content hash.

        Uses short hash for indexed lookup, then verifies via full hash
        to avoid 40-bit collision false positives.

        When limit=1 (default), returns a single DocumentRecord or None
        for backwards compatibility.  When limit>1, returns a list.
        """
        rows = self._fetchall("""
            SELECT id, collection, summary, tags_json, created_at,
                   updated_at, content_hash, content_hash_full, accessed_at
            FROM documents
            WHERE collection = ? AND content_hash = ? AND id != ?
            LIMIT ?
        """, (collection, content_hash, exclude_id, limit))
        results: list[DocumentRecord] = []
        for row in rows:
            # Verify full hash if both sides have one
            if content_hash_full and row["content_hash_full"]:
                if content_hash_full != row["content_hash_full"]:
                    continue
            results.append(DocumentRecord(
                id=row["id"],
                collection=row["collection"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                content_hash=row["content_hash"],
                content_hash_full=row["content_hash_full"],
                accessed_at=row["accessed_at"],
            ))
        if limit == 1:
            return results[0] if results else None
        return results

    def list_ids(
        self,
        collection: str,
        limit: Optional[int] = None,
    ) -> list[str]:
        """List document IDs in a collection.
        
        Args:
            collection: Collection name
            limit: Maximum number to return (None for all)
            
        Returns:
            List of document IDs
        """
        if limit:
            cursor = self._execute("""
                SELECT id FROM documents
                WHERE collection = ?
                ORDER BY updated_at DESC
                LIMIT ?
            """, (collection, limit))
        else:
            cursor = self._execute("""
                SELECT id FROM documents
                WHERE collection = ?
                ORDER BY updated_at DESC
            """, (collection,))

        return [row["id"] for row in cursor]

    def list_recent(
        self,
        collection: str,
        limit: int = 10,
        order_by: str = "updated",
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """List recent documents ordered by timestamp.

        Args:
            collection: Collection name
            limit: Maximum number to return
            order_by: Sort column - "updated" (default) or "accessed"
            offset: Number of rows to skip (for pagination)

        Returns:
            List of DocumentRecords, most recent first
        """
        allowed_order = {
            "updated": ("updated_at", "DESC"),
            "accessed": ("accessed_at", "DESC"),
            "created": ("created_at", "DESC"),
            "id": ("id", "ASC"),
        }
        entry = allowed_order.get(order_by)
        if entry is None:
            raise ValueError(f"Invalid order_by: {order_by!r} (expected 'updated', 'accessed', 'created', or 'id')")
        order_col, order_dir = entry
        cursor = self._execute(f"""
            SELECT id, collection, summary, tags_json, created_at, updated_at, content_hash, accessed_at
            FROM documents
            WHERE collection = ?
            ORDER BY {order_col} {order_dir}
            LIMIT ? OFFSET ?
        """, (collection, limit, offset))

        return [
            DocumentRecord(
                id=row["id"],
                collection=row["collection"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                content_hash=row["content_hash"],
                accessed_at=row["accessed_at"],
            )
            for row in cursor
        ]

    def list_recent_with_history(
        self,
        collection: str,
        limit: int = 10,
        order_by: str = "updated",
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """List recent documents including archived versions.

        Returns DocumentRecords sorted by timestamp. Archived versions
        have '_version' tag set to their offset (1=previous, 2=two ago...).
        Current versions have no '_version' tag (equivalent to offset 0).
        """
        allowed_order = {
            "updated": ("updated_at", "DESC"),
            "accessed": ("accessed_at", "DESC"),
            "created": ("created_at", "DESC"),
            "id": ("id", "ASC"),
        }
        entry = allowed_order.get(order_by)
        if entry is None:
            raise ValueError(f"Invalid order_by: {order_by!r} (expected 'updated', 'accessed', 'created', or 'id')")
        order_col, order_dir = entry

        cursor = self._execute(f"""
            SELECT id, summary, tags_json, {order_col} as sort_ts,
                   0 as version_offset, content_hash, accessed_at
            FROM documents
            WHERE collection = ?

            UNION ALL

            SELECT dv.id, dv.summary, dv.tags_json, dv.created_at as sort_ts,
                   ROW_NUMBER() OVER (PARTITION BY dv.id ORDER BY dv.version DESC) as version_offset,
                   dv.content_hash, NULL as accessed_at
            FROM document_versions dv
            WHERE dv.collection = ?

            ORDER BY sort_ts {order_dir}
            LIMIT ? OFFSET ?
        """, (collection, collection, limit, offset))

        records = []
        for row in cursor:
            tags = _load_tags_json(row["tags_json"])
            offset = row["version_offset"]
            if offset > 0:
                tags["_version"] = str(offset)
            records.append(DocumentRecord(
                id=row["id"],
                collection=collection,
                summary=row["summary"],
                tags=tags,
                created_at=row["sort_ts"],
                updated_at=row["sort_ts"],
                content_hash=row["content_hash"],
                accessed_at=row["accessed_at"],
            ))

        return records

    def count(self, collection: str) -> int:
        """Count documents in a collection."""
        row = self._fetchone("""
            SELECT COUNT(*) FROM documents
            WHERE collection = ?
        """, (collection,))
        return row[0]
    
    def count_all(self) -> int:
        """Count total documents across all collections."""
        row = self._fetchone("SELECT COUNT(*) FROM documents")
        return row[0]

    def count_versions(self, collection: str) -> int:
        """Count archived versions in a collection."""
        row = self._fetchone("""
            SELECT COUNT(*) FROM document_versions
            WHERE collection = ?
        """, (collection,))
        return row[0]

    def query_by_id_prefix(
        self,
        collection: str,
        prefix: str,
        limit: int = 0,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """Query documents by ID prefix.

        Args:
            collection: Collection name
            prefix: ID prefix to match (e.g., ".")
            limit: Max results (0 = unlimited)
            offset: Number of rows to skip (for pagination)

        Returns:
            List of matching DocumentRecords
        """
        # Escape LIKE wildcards in the prefix to prevent injection
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql = """
            SELECT id, collection, summary, tags_json, created_at, updated_at, content_hash, accessed_at
            FROM documents
            WHERE collection = ? AND id LIKE ? ESCAPE '\\'
            ORDER BY id
        """
        params: tuple = (collection, f"{escaped}%")
        if limit > 0:
            sql += " LIMIT ?"
            params += (limit,)
        if offset > 0:
            if limit == 0:
                sql += " LIMIT -1"  # SQLite requires LIMIT before OFFSET
            sql += " OFFSET ?"
            params += (offset,)
        rows = self._fetchall(sql, params)

        results = []
        for row in rows:
            results.append(DocumentRecord(
                id=row["id"],
                collection=row["collection"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                content_hash=row["content_hash"],
                accessed_at=row["accessed_at"],
            ))
        return results

    def _get_stopwords(self) -> frozenset[str]:
        """Load stopwords, checking for a `.stop` override in the store."""
        if self._stopwords is not None:
            return self._stopwords
        # Check for user override in the store
        try:
            row = self._fetchone(
                "SELECT summary FROM documents WHERE id = '.stop' LIMIT 1"
            )
            if row and row[0].strip():
                words = set()
                for line in row[0].splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        words.add(line.lower())
                self._stopwords = frozenset(words)
                return self._stopwords
        except Exception:
            pass
        # Fall back to bundled stop list
        from importlib.resources import files
        stop_text = (
            files("keep.data.system").joinpath("stop.md").read_text()
        )
        words = set()
        in_frontmatter = False
        for line in stop_text.splitlines():
            stripped = line.strip()
            if stripped == "---":
                in_frontmatter = not in_frontmatter
                continue
            if in_frontmatter:
                continue
            if stripped and not stripped.startswith("#"):
                words.add(stripped.lower())
        self._stopwords = frozenset(words)
        return self._stopwords

    def get_stopwords(self) -> frozenset[str]:
        """Return the active stopword set used for FTS and deep query scoring."""
        return self._get_stopwords()

    def _build_fts_query(self, query: str) -> Optional[str]:
        """Tokenize a natural-language query into an FTS5 MATCH expression.

        Returns ``None`` when no usable tokens remain after stripping
        stopwords and special characters.
        """
        stopwords = self._get_stopwords()
        tokens = [t for t in query.split() if t.lower() not in stopwords]
        if not tokens:
            return None
        safe = [t.replace('"', '').replace("'", "") for t in tokens]
        safe = [t for t in safe if t]
        if not safe:
            return None
        return " OR ".join(f'"{t}"' for t in safe)

    def query_fts(
        self,
        collection: str,
        query: str,
        limit: int = 10,
        tags: Optional[dict[str, str]] = None,
        part_tag_join_base_ids: Optional[list[str]] = None,
    ) -> list[tuple[str, str, float]]:
        """Full-text search using FTS5 indexes (documents + parts).

        Searches document summaries and part summaries.
        Part results are returned with ``id@p{N}`` IDs so the caller's
        part-to-parent uplift logic can merge them.

        Args:
            collection: Collection name
            query: Natural-language search query
            limit: Max results
            tags: Optional tag filter — only return items matching all tags
                  (keys and values should be casefolded)
            part_tag_join_base_ids: Optional list of base document IDs
                  whose parts should bypass the ``tags`` filter. Parts
                  do not inherit parent tags (see analyze.py), so when
                  a tag filter is applied the caller precomputes the
                  set of parents whose own tags match and passes them
                  here; parts belonging to any of those parents are
                  then accepted regardless of their own tag markers.

        Returns:
            List of (id, summary, bm25_rank) tuples ordered by relevance.
            bm25_rank is negative (more negative = better match).
            Returns empty list if FTS5 is not available.
        """
        if not self._fts_available:
            return []
        fts_query = self._build_fts_query(query)
        if fts_query is None:
            return []

        with get_tracer("doc_store").start_as_current_span(
            "doc_store.query_fts",
            attributes={
                "collection": collection,
                "limit": limit,
                "query": query,
                "tag_count": len(tags or {}),
                "part_join_size": len(part_tag_join_base_ids or []),
            },
        ) as span:
            # --- Search documents ---
            doc_sql = """
                SELECT d.id, d.summary, f.rank
                FROM documents_fts f
                JOIN documents d ON d.rowid = f.rowid
                WHERE documents_fts MATCH ?
                AND d.collection = ?
            """
            doc_params: list[Any] = [fts_query, collection]
            if tags:
                for k in tags:
                    for v in tag_values(tags, k):
                        doc_sql += (
                            " AND EXISTS ("
                            "SELECT 1 FROM json_each(d.tags_json, ?) jv "
                            "WHERE CAST(jv.value AS TEXT) = ?)"
                        )
                        doc_params.extend([f"$.{k}", v])
            doc_sql += " ORDER BY f.rank LIMIT ?"
            doc_params.append(limit)
            doc_rows = self._fetchall(doc_sql, tuple(doc_params))

            # --- Search parts (summary only) ---
            part_sql = """
                SELECT p.id || '@p' || p.part_num, p.summary, f.rank
                FROM parts_fts f
                JOIN document_parts p ON p.rowid = f.rowid
                WHERE parts_fts MATCH ?
                AND p.collection = ?
            """
            part_params: list[Any] = [fts_query, collection]
            if tags:
                # Build the per-tag EXISTS clauses for the part's own tags.
                tag_clauses: list[str] = []
                tag_params: list[Any] = []
                for k in tags:
                    for v in tag_values(tags, k):
                        tag_clauses.append(
                            "EXISTS ("
                            "SELECT 1 FROM json_each(p.tags_json, ?) jv "
                            "WHERE CAST(jv.value AS TEXT) = ?)"
                        )
                        tag_params.extend([f"$.{k}", v])
                own_match = " AND ".join(tag_clauses) if tag_clauses else ""
                if part_tag_join_base_ids and own_match:
                    # Part is accepted if its own tags match all filters
                    # OR its _base_id (p.id) is in the precomputed set of
                    # parents whose own tags match. See analyze.py for why
                    # parts don't inherit parent tags.
                    placeholders = ",".join("?" * len(part_tag_join_base_ids))
                    part_sql += (
                        f" AND (({own_match}) OR p.id IN ({placeholders}))"
                    )
                    part_params.extend(tag_params)
                    part_params.extend(part_tag_join_base_ids)
                elif part_tag_join_base_ids:
                    placeholders = ",".join("?" * len(part_tag_join_base_ids))
                    part_sql += f" AND p.id IN ({placeholders})"
                    part_params.extend(part_tag_join_base_ids)
                elif own_match:
                    part_sql += f" AND {own_match}"
                    part_params.extend(tag_params)
            part_sql += " ORDER BY f.rank LIMIT ?"
            part_params.append(limit)
            part_rows = self._fetchall(part_sql, tuple(part_params))

            # --- Search versions ---
            ver_sql = """
                SELECT v.id || '@v' || v.version, v.summary, f.rank
                FROM versions_fts f
                JOIN document_versions v ON v.rowid = f.rowid
                WHERE versions_fts MATCH ?
                AND v.collection = ?
            """
            ver_params: list[Any] = [fts_query, collection]
            if tags:
                for k in tags:
                    for v in tag_values(tags, k):
                        ver_sql += (
                            " AND EXISTS ("
                            "SELECT 1 FROM json_each(v.tags_json, ?) jv "
                            "WHERE CAST(jv.value AS TEXT) = ?)"
                        )
                        ver_params.extend([f"$.{k}", v])
            ver_sql += " ORDER BY f.rank LIMIT ?"
            ver_params.append(limit)
            ver_rows = self._fetchall(ver_sql, tuple(ver_params))

            # Merge by BM25 rank (more negative = better), take top `limit`
            combined = [(row[0], row[1], row[2]) for row in doc_rows]
            combined.extend((row[0], row[1], row[2]) for row in part_rows)
            combined.extend((row[0], row[1], row[2]) for row in ver_rows)
            combined.sort(key=lambda r: r[2])  # sort by rank ascending (best first)
            limited = combined[:limit]
            span.set_attribute("result_count", len(limited))
            return limited

    def query_fts_scoped(
        self,
        collection: str,
        query: str,
        ids: list[str],
        limit: int = 10,
        tags: Optional[dict[str, str]] = None,
        part_tag_join_base_ids: Optional[list[str]] = None,
    ) -> list[tuple[str, str, float]]:
        """Full-text search scoped to a whitelist of base document IDs.

        Same as :meth:`query_fts` but adds ``AND <table>.id IN (...)``
        to each sub-query so only documents, parts, and versions belonging
        to *ids* are considered.  Used by edge-following deep search.

        The ``part_tag_join_base_ids`` parameter has the same meaning
        as on :meth:`query_fts`: parts whose ``p.id`` is in this list
        bypass the ``tags`` filter, since parts intentionally do not
        inherit parent tags (see analyze.py).
        """
        if not self._fts_available or not ids:
            return []
        fts_query = self._build_fts_query(query)
        if fts_query is None:
            return []

        with get_tracer("doc_store").start_as_current_span(
            "doc_store.query_fts_scoped",
            attributes={
                "collection": collection,
                "limit": limit,
                "query": query,
                "scope_size": len(ids),
                "tag_count": len(tags or {}),
                "part_join_size": len(part_tag_join_base_ids or []),
            },
        ) as span:
            placeholders = ",".join("?" * len(ids))

            # --- Search documents ---
            doc_sql = f"""
                SELECT d.id, d.summary, f.rank
                FROM documents_fts f
                JOIN documents d ON d.rowid = f.rowid
                WHERE documents_fts MATCH ?
                AND d.collection = ?
                AND d.id IN ({placeholders})
            """
            doc_params: list[Any] = [fts_query, collection, *ids]
            if tags:
                for k in tags:
                    for v in tag_values(tags, k):
                        doc_sql += (
                            " AND EXISTS ("
                            "SELECT 1 FROM json_each(d.tags_json, ?) jv "
                            "WHERE CAST(jv.value AS TEXT) = ?)"
                        )
                        doc_params.extend([f"$.{k}", v])
            doc_sql += " ORDER BY f.rank LIMIT ?"
            doc_params.append(limit)
            doc_rows = self._fetchall(doc_sql, tuple(doc_params))

            # --- Search parts ---
            part_sql = f"""
                SELECT p.id || '@p' || p.part_num, p.summary, f.rank
                FROM parts_fts f
                JOIN document_parts p ON p.rowid = f.rowid
                WHERE parts_fts MATCH ?
                AND p.collection = ?
                AND p.id IN ({placeholders})
            """
            part_params: list[Any] = [fts_query, collection, *ids]
            if tags:
                tag_clauses: list[str] = []
                tag_params: list[Any] = []
                for k in tags:
                    for v in tag_values(tags, k):
                        tag_clauses.append(
                            "EXISTS ("
                            "SELECT 1 FROM json_each(p.tags_json, ?) jv "
                            "WHERE CAST(jv.value AS TEXT) = ?)"
                        )
                        tag_params.extend([f"$.{k}", v])
                own_match = " AND ".join(tag_clauses) if tag_clauses else ""
                if part_tag_join_base_ids and own_match:
                    join_ph = ",".join("?" * len(part_tag_join_base_ids))
                    part_sql += (
                        f" AND (({own_match}) OR p.id IN ({join_ph}))"
                    )
                    part_params.extend(tag_params)
                    part_params.extend(part_tag_join_base_ids)
                elif part_tag_join_base_ids:
                    join_ph = ",".join("?" * len(part_tag_join_base_ids))
                    part_sql += f" AND p.id IN ({join_ph})"
                    part_params.extend(part_tag_join_base_ids)
                elif own_match:
                    part_sql += f" AND {own_match}"
                    part_params.extend(tag_params)
            part_sql += " ORDER BY f.rank LIMIT ?"
            part_params.append(limit)
            part_rows = self._fetchall(part_sql, tuple(part_params))

            # --- Search versions ---
            ver_sql = f"""
                SELECT v.id || '@v' || v.version, v.summary, f.rank
                FROM versions_fts f
                JOIN document_versions v ON v.rowid = f.rowid
                WHERE versions_fts MATCH ?
                AND v.collection = ?
                AND v.id IN ({placeholders})
            """
            ver_params: list[Any] = [fts_query, collection, *ids]
            if tags:
                for k in tags:
                    for v in tag_values(tags, k):
                        ver_sql += (
                            " AND EXISTS ("
                            "SELECT 1 FROM json_each(v.tags_json, ?) jv "
                            "WHERE CAST(jv.value AS TEXT) = ?)"
                        )
                        ver_params.extend([f"$.{k}", v])
            ver_sql += " ORDER BY f.rank LIMIT ?"
            ver_params.append(limit)
            ver_rows = self._fetchall(ver_sql, tuple(ver_params))

            combined = [(row[0], row[1], row[2]) for row in doc_rows]
            combined.extend((row[0], row[1], row[2]) for row in part_rows)
            combined.extend((row[0], row[1], row[2]) for row in ver_rows)
            combined.sort(key=lambda r: r[2])
            limited = combined[:limit]
            span.set_attribute("result_count", len(limited))
            return limited

    def query_by_id_glob(
        self,
        collection: str,
        pattern: str,
        limit: int = 0,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """Query documents by ID glob pattern.

        Supports * (any chars) and ? (single char).

        Args:
            collection: Collection name
            pattern: Glob pattern (e.g., "session-*", "*auth*")
            limit: Max results (0 = unlimited)
            offset: Number of rows to skip (for pagination)

        Returns:
            List of matching DocumentRecords
        """
        # Escape SQL LIKE special chars first, then convert glob to LIKE
        escaped = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like_pattern = escaped.replace("*", "%").replace("?", "_")
        sql = """
            SELECT id, collection, summary, tags_json, created_at, updated_at, content_hash, accessed_at
            FROM documents
            WHERE collection = ? AND id LIKE ? ESCAPE '\\'
            ORDER BY id
        """
        params: tuple = (collection, like_pattern)
        if limit > 0:
            sql += " LIMIT ?"
            params += (limit,)
        if offset > 0:
            if limit == 0:
                sql += " LIMIT -1"
            sql += " OFFSET ?"
            params += (offset,)
        rows = self._fetchall(sql, params)

        results = []
        for row in rows:
            results.append(DocumentRecord(
                id=row["id"],
                collection=row["collection"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                content_hash=row["content_hash"],
                accessed_at=row["accessed_at"],
            ))

        return results

    def find_by_name(
        self,
        collection: str,
        stem: str,
        *,
        id_prefix: str | None = None,
        limit: int = 5,
    ) -> list[DocumentRecord]:
        """Find documents whose ID ends with ``/{stem}`` or ``/{stem}.md``.

        Used for vault-wide wikilink resolution: given a note name like
        ``Foo``, find any item whose path ends with ``/Foo.md`` or ``/Foo``.

        Args:
            collection: Collection name
            stem: Bare note name (no extension)
            id_prefix: Optional URI prefix to scope results (e.g. vault root)
            limit: Max results
        """
        escaped = stem.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        # Match /{stem}.md or /{stem} at end of ID
        conditions = [
            "collection = ?",
            "(id LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\')",
        ]
        params: list[Any] = [
            collection,
            f"%/{escaped}.md",
            f"%/{escaped}",
        ]

        if id_prefix:
            esc_prefix = id_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("id LIKE ? ESCAPE '\\'")
            params.append(f"{esc_prefix}%")

        sql = f"""
            SELECT id, collection, summary, tags_json, created_at, updated_at, content_hash, accessed_at
            FROM documents
            WHERE {' AND '.join(conditions)}
            ORDER BY length(id), id
            LIMIT ?
        """
        params.append(limit)
        rows = self._fetchall(sql, tuple(params))

        results = []
        for row in rows:
            results.append(DocumentRecord(
                id=row["id"],
                collection=row["collection"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                content_hash=row["content_hash"],
                accessed_at=row["accessed_at"],
            ))
        return results

    # -------------------------------------------------------------------------
    # Part Operations (structural decomposition)
    # -------------------------------------------------------------------------

    def upsert_parts(
        self,
        collection: str,
        id: str,
        parts: list[PartInfo],
    ) -> int:
        """Replace all parts for a document atomically.

        Re-analysis produces a fresh decomposition — old parts are deleted
        and new ones inserted in a single transaction.

        Args:
            collection: Collection name
            id: Document identifier
            parts: List of PartInfo to store

        Returns:
            Number of parts stored
        """
        with self._lock:
            self._execute("BEGIN IMMEDIATE")
            try:
                # Delete existing parts
                self._execute("""
                    DELETE FROM document_parts
                    WHERE id = ? AND collection = ?
                """, (id, collection))

                # Insert new parts
                for part in parts:
                    self._execute("""
                        INSERT INTO document_parts
                        (id, collection, part_num, summary, tags_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        id, collection, part.part_num, part.summary,
                        json.dumps(normalize_tag_map(part.tags), ensure_ascii=False),
                        part.created_at,
                    ))

                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return len(parts)

    def upsert_single_part(
        self,
        collection: str,
        id: str,
        part: PartInfo,
    ) -> None:
        """Insert or replace a single part without affecting other parts.

        Args:
            collection: Collection name
            id: Document identifier
            part: PartInfo to store
        """
        with self._lock:
            self._execute("""
                INSERT OR REPLACE INTO document_parts
                (id, collection, part_num, summary, tags_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                id, collection, part.part_num, part.summary,
                json.dumps(normalize_tag_map(part.tags), ensure_ascii=False),
                part.created_at,
            ))
            self._conn.commit()

    def list_part_doc_ids(self, collection: str, part_num: int) -> list[str]:
        """List document IDs that have a part with the given part number."""
        cursor = self._execute("""
            SELECT id
            FROM document_parts
            WHERE collection = ? AND part_num = ?
            ORDER BY id
        """, (collection, part_num))
        return [row["id"] for row in cursor]

    def get_part(
        self,
        collection: str,
        id: str,
        part_num: int,
    ) -> Optional[PartInfo]:
        """Get a specific part by number.

        Args:
            collection: Collection name
            id: Document identifier
            part_num: Part number (1-indexed)

        Returns:
            PartInfo if found, None otherwise
        """
        row = self._fetchone("""
            SELECT part_num, summary, tags_json, created_at
            FROM document_parts
            WHERE id = ? AND collection = ? AND part_num = ?
        """, (id, collection, part_num))
        if row is None:
            return None

        return PartInfo(
            part_num=row["part_num"],
            summary=row["summary"],
            tags=_load_tags_json(row["tags_json"]),
            created_at=row["created_at"],
        )

    def list_parts(
        self,
        collection: str,
        id: str,
    ) -> list[PartInfo]:
        """List all parts for a document, ordered by part number.

        Args:
            collection: Collection name
            id: Document identifier

        Returns:
            List of PartInfo, ordered by part_num
        """
        rows = self._fetchall("""
            SELECT part_num, summary, tags_json, created_at
            FROM document_parts
            WHERE id = ? AND collection = ?
            ORDER BY part_num
        """, (id, collection))

        return [
            PartInfo(
                part_num=row["part_num"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def list_parts_many(
        self,
        collection: str,
        ids: list[str],
    ) -> dict[str, list[PartInfo]]:
        """List all parts for multiple documents, ordered by part number."""
        if not ids:
            return {}

        placeholders = ",".join("?" * len(ids))
        rows = self._fetchall(f"""
            SELECT id, part_num, summary, tags_json, created_at
            FROM document_parts
            WHERE collection = ? AND id IN ({placeholders})
            ORDER BY id, part_num
        """, (collection, *ids))

        parts_by_id: dict[str, list[PartInfo]] = {}
        for row in rows:
            parts_by_id.setdefault(row["id"], []).append(PartInfo(
                part_num=row["part_num"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
            ))

        return parts_by_id

    def part_count(self, collection: str, id: str) -> int:
        """Count parts for a document."""
        row = self._fetchone("""
            SELECT COUNT(*) FROM document_parts
            WHERE id = ? AND collection = ?
        """, (id, collection))
        return row[0]

    def max_part_num(self, collection: str, id: str) -> int:
        """Return the highest part_num for a document, or 0 if none."""
        row = self._fetchone("""
            SELECT COALESCE(MAX(part_num), 0) FROM document_parts
            WHERE id = ? AND collection = ?
        """, (id, collection))
        return row[0]

    def delete_parts(self, collection: str, id: str) -> int:
        """Delete all parts for a document.

        Args:
            collection: Collection name
            id: Document identifier

        Returns:
            Number of parts deleted
        """
        with self._lock:
            cursor = self._execute("""
                DELETE FROM document_parts
                WHERE id = ? AND collection = ?
            """, (id, collection))
            self._conn.commit()
            return cursor.rowcount

    def delete_part(self, collection: str, id: str, part_num: int) -> int:
        """Delete a single part row."""
        with self._lock:
            cursor = self._execute("""
                DELETE FROM document_parts
                WHERE id = ? AND collection = ? AND part_num = ?
            """, (id, collection, part_num))
            self._conn.commit()
            return cursor.rowcount

    def update_part_tags(
        self,
        collection: str,
        id: str,
        part_num: int,
        tags: dict[str, Any],
    ) -> bool:
        """Update tags on a single part.

        Args:
            collection: Collection name
            id: Parent document identifier
            part_num: Part number (1-indexed)
            tags: Complete merged tag dict to store

        Returns:
            True if the part was found and updated
        """
        with self._lock:
            cursor = self._execute("""
                UPDATE document_parts
                SET tags_json = ?
                WHERE id = ? AND collection = ? AND part_num = ?
            """, (json.dumps(normalize_tag_map(tags), ensure_ascii=False), id, collection, part_num))
            self._conn.commit()
        return cursor.rowcount > 0

    # -------------------------------------------------------------------------
    # Tag Queries
    # -------------------------------------------------------------------------

    def list_distinct_tag_keys(self, collection: str) -> list[str]:
        """List all distinct tag keys used in the collection.

        Excludes system tags (prefixed with _).

        Returns:
            Sorted list of distinct tag keys
        """
        cursor = self._execute("""
            SELECT DISTINCT j.key FROM documents, json_each(tags_json) AS j
            WHERE collection = ? AND j.key NOT LIKE '\\_%' ESCAPE '\\'
            ORDER BY j.key
        """, (collection,))

        return [row[0] for row in cursor]

    def list_distinct_tag_values(self, collection: str, key: str) -> list[str]:
        """List all distinct values for a given tag key.

        Args:
            collection: Collection name
            key: Tag key to get values for

        Returns:
            Sorted list of distinct values
        """
        cursor = self._execute("""
            SELECT DISTINCT CAST(jv.value AS TEXT) AS val
            FROM documents d
            JOIN json_each(d.tags_json, '$.' || ?) jv
            WHERE d.collection = ?
            ORDER BY val
        """, (key, collection))

        return [row[0] for row in cursor]

    def tag_pair_counts(self, collection: str) -> dict[tuple[str, str], int]:
        """Count documents per (key, value) tag pair, excluding system tags.

        Used for IDF weighting in deep tag-follow scoring.
        """
        cursor = self._execute("""
            SELECT j.key, CAST(v.value AS TEXT) AS val, COUNT(DISTINCT d.id) as cnt
            FROM documents d
            JOIN json_each(d.tags_json) AS j
            JOIN json_each(
                CASE
                    WHEN j.type = 'array' THEN j.value
                    ELSE json_array(j.value)
                END
            ) AS v
            WHERE d.collection = ?
              AND j.key NOT LIKE '\\_%' ESCAPE '\\'
            GROUP BY j.key, val
        """, (collection,))
        return {(row[0], row[1]): row[2] for row in cursor}

    def query_by_tag_key(
        self,
        collection: str,
        key: str,
        limit: int = 100,
        since_date: Optional[str] = None,
        until_date: Optional[str] = None,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """Find documents that have a specific tag key (any value).

        Args:
            collection: Collection name
            key: Tag key to search for
            limit: Maximum results
            since_date: Only include items updated on or after this date (YYYY-MM-DD)
            until_date: Only include items updated before this date (YYYY-MM-DD)
            offset: Number of rows to skip (for pagination)

        Returns:
            List of matching DocumentRecords
        """
        # SQLite JSON functions for tag key existence
        # json_extract returns NULL if key doesn't exist
        params: list[Any] = [collection, f"$.{key}"]

        sql = """
            SELECT id, collection, summary, tags_json, created_at, updated_at,
                   content_hash, content_hash_full, accessed_at
            FROM documents
            WHERE collection = ?
              AND json_extract(tags_json, ?) IS NOT NULL
        """

        if since_date is not None:
            # Compare against the date portion of updated_at
            sql += "  AND updated_at >= ?\n"
            params.append(since_date)

        if until_date is not None:
            sql += "  AND updated_at < ?\n"
            params.append(until_date)

        sql += "ORDER BY updated_at DESC\nLIMIT ?"
        params.append(limit)
        if offset > 0:
            sql += " OFFSET ?"
            params.append(offset)

        cursor = self._execute(sql, params)

        results = []
        for row in cursor:
            results.append(DocumentRecord(
                id=row["id"],
                collection=row["collection"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                content_hash=row["content_hash"],
                content_hash_full=row["content_hash_full"],
                accessed_at=row["accessed_at"],
            ))

        return results

    def query_by_tag_value(
        self,
        collection: str,
        key: str,
        value: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DocumentRecord]:
        """Find documents where tag *key* contains *value*.

        Handles both scalar (``"val"``) and array (``["a","b"]``) storage
        using SQLite JSON functions.
        """
        cursor = self._execute(
            """
            SELECT id, collection, summary, tags_json, created_at, updated_at,
                   content_hash, content_hash_full, accessed_at
            FROM documents
            WHERE collection = ?
              AND (
                  json_extract(tags_json, ?) = ?
                  OR EXISTS (
                      SELECT 1 FROM json_each(
                          CASE
                              WHEN json_type(tags_json, ?) = 'array'
                              THEN json_extract(tags_json, ?)
                              ELSE '[]'
                          END
                      ) WHERE value = ?
                  )
              )
            ORDER BY updated_at DESC
            LIMIT ?
            OFFSET ?
            """,
            (collection, f"$.{key}", value, f"$.{key}", f"$.{key}", value, limit, offset),
        )
        results = []
        for row in cursor:
            results.append(DocumentRecord(
                id=row["id"],
                collection=row["collection"],
                summary=row["summary"],
                tags=_load_tags_json(row["tags_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                content_hash=row["content_hash"],
                content_hash_full=row["content_hash_full"],
                accessed_at=row["accessed_at"],
            ))
        return results

    # -------------------------------------------------------------------------
    # Collection Management
    # -------------------------------------------------------------------------
    
    def list_collections(self) -> list[str]:
        """List all collection names."""
        cursor = self._execute("""
            SELECT DISTINCT collection FROM documents
            ORDER BY collection
        """)
        return [row["collection"] for row in cursor]
    
    def delete_collection(self, collection: str) -> int:
        """Delete all documents in a collection.

        Args:
            collection: Collection name

        Returns:
            Number of documents deleted
        """
        with self._lock:
            had_stop = self.exists(collection, ".stop")
            cursor = self._execute("""
                DELETE FROM documents
                WHERE collection = ?
            """, (collection,))
            self._conn.commit()
            if had_stop:
                self._stopwords = None
        return cursor.rowcount
    
    # -------------------------------------------------------------------------
    # Bulk Import
    # -------------------------------------------------------------------------

    def delete_collection_all(self, collection: str) -> int:
        """Delete all documents, versions, and parts in a collection.

        Unlike delete_collection(), this also clears version history
        and parts tables.

        Args:
            collection: Collection name

        Returns:
            Number of documents deleted
        """
        with self._lock:
            had_stop = self.exists(collection, ".stop")
            self._execute("BEGIN IMMEDIATE")
            try:
                cursor = self._execute(
                    "DELETE FROM documents WHERE collection = ?", (collection,))
                doc_count = cursor.rowcount
                self._execute(
                    "DELETE FROM document_versions WHERE collection = ?", (collection,))
                self._execute(
                    "DELETE FROM document_parts WHERE collection = ?", (collection,))
                self._execute(
                    "DELETE FROM version_edges WHERE collection = ?", (collection,))
                self._execute(
                    "DELETE FROM edges WHERE collection = ?", (collection,))
                self._execute(
                    "DELETE FROM edge_backfill WHERE collection = ?", (collection,))
                self._conn.commit()
                if had_stop:
                    self._stopwords = None
            except Exception:
                self._conn.rollback()
                raise
        return doc_count

    def import_batch(
        self,
        collection: str,
        documents: list[dict],
    ) -> dict:
        """Bulk-insert documents with versions and parts in a single transaction.

        Bypasses the normal upsert/archive logic — no auto-timestamping,
        no version archiving. Intended for data import/restore.

        Each document dict must have:
            id, summary, tags (dict), created_at, updated_at, accessed_at,
            content_hash (optional), content_hash_full (optional),
            versions (list of version dicts), parts (list of part dicts).

        Version dicts: version, summary, tags, content_hash, created_at.
        Part dicts: part_num, summary, tags, created_at.

        Args:
            collection: Target collection name
            documents: List of document dicts in export format

        Returns:
            Dict with stats: {documents: int, versions: int, parts: int}
        """
        doc_count = 0
        ver_count = 0
        part_count = 0

        with self._lock:
            self._execute("BEGIN IMMEDIATE")
            try:
                touched_with_versions: set[str] = set()
                for doc in documents:
                    tags_json = json.dumps(
                        normalize_tag_map(doc.get("tags", {})),
                        ensure_ascii=False,
                    )
                    self._execute("""
                        INSERT OR REPLACE INTO documents
                        (id, collection, summary, tags_json, created_at,
                         updated_at, content_hash, content_hash_full, accessed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        doc["id"], collection, doc.get("summary", ""),
                        tags_json, doc["created_at"], doc["updated_at"],
                        doc.get("content_hash"), doc.get("content_hash_full"),
                        doc.get("accessed_at", doc["updated_at"]),
                    ))
                    doc_count += 1

                    for ver in doc.get("versions", []):
                        ver_tags_json = json.dumps(
                            normalize_tag_map(ver.get("tags", {})),
                            ensure_ascii=False,
                        )
                        self._execute("""
                            INSERT OR REPLACE INTO document_versions
                            (id, collection, version, summary, tags_json,
                             content_hash, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            doc["id"], collection, ver["version"],
                            ver.get("summary", ""), ver_tags_json,
                            ver.get("content_hash"), ver["created_at"],
                        ))
                        ver_count += 1
                        touched_with_versions.add(doc["id"])

                    for part in doc.get("parts", []):
                        part_tags_json = json.dumps(
                            normalize_tag_map(part.get("tags", {})),
                            ensure_ascii=False,
                        )
                        self._execute("""
                            INSERT OR REPLACE INTO document_parts
                            (id, collection, part_num, summary, tags_json, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            doc["id"], collection, part["part_num"],
                            part.get("summary", ""), part_tags_json,
                            part["created_at"],
                        ))
                        part_count += 1

                for source_id in touched_with_versions:
                    self._rebuild_version_edges_for_source_unlocked(
                        collection, source_id,
                    )

                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

        return {"documents": doc_count, "versions": ver_count, "parts": part_count}

    # -------------------------------------------------------------------------
    # Edges
    # -------------------------------------------------------------------------

    def upsert_edge(
        self,
        collection: str,
        source_id: str,
        predicate: str,
        target_id: str,
        inverse: str,
        created: str,
    ) -> None:
        """Insert or replace an edge row."""
        self._execute_write(
            """
            INSERT OR REPLACE INTO edges
                (source_id, collection, predicate, target_id, inverse, created)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (source_id, collection, predicate, target_id, inverse, created),
        )
        self._conn.commit()

    def upsert_edges_batch(
        self,
        collection: str,
        edges: list[tuple[str, str, str, str, str]],
    ) -> int:
        """Batch insert/replace edge rows in a single transaction.

        Each tuple is (source_id, predicate, target_id, inverse, created).
        """
        if not edges:
            return 0
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.executemany(
                    """
                    INSERT OR REPLACE INTO edges
                        (source_id, collection, predicate, target_id, inverse, created)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [(s, collection, p, t, inv, c) for s, p, t, inv, c in edges],
                )
                self._conn.commit()
                return len(edges)
            except Exception:
                self._conn.rollback()
                raise

    def delete_edges_batch(
        self,
        collection: str,
        edges: list[tuple[str, str, str]],
    ) -> int:
        """Batch delete edge rows in a single transaction.

        Each tuple is (source_id, predicate, target_id).
        """
        if not edges:
            return 0
        count = 0
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for source_id, predicate, target_id in edges:
                    cur = self._conn.execute(
                        """
                        DELETE FROM edges
                        WHERE source_id = ? AND collection = ? AND predicate = ? AND target_id = ?
                        """,
                        (source_id, collection, predicate, target_id),
                    )
                    count += cur.rowcount
                self._conn.commit()
                return count
            except Exception:
                self._conn.rollback()
                raise

    def delete_edge(
        self,
        collection: str,
        source_id: str,
        predicate: str,
        target_id: Optional[str] = None,
    ) -> int:
        """Delete edge rows for source/predicate, optionally one target."""
        if target_id is None:
            cur = self._execute_write(
                "DELETE FROM edges WHERE source_id = ? AND collection = ? AND predicate = ?",
                (source_id, collection, predicate),
            )
        else:
            cur = self._execute_write(
                """
                DELETE FROM edges
                WHERE source_id = ? AND collection = ? AND predicate = ? AND target_id = ?
                """,
                (source_id, collection, predicate, target_id),
            )
        self._conn.commit()
        return cur.rowcount

    def delete_edges_for_source(self, collection: str, source_id: str) -> int:
        """Delete all edges originating from *source_id*."""
        cur = self._execute_write(
            "DELETE FROM edges WHERE collection = ? AND source_id = ?",
            (collection, source_id),
        )
        self._conn.commit()
        return cur.rowcount

    def delete_edges_for_target(self, collection: str, target_id: str) -> int:
        """Delete all edges pointing at *target_id*."""
        cur = self._execute_write(
            "DELETE FROM edges WHERE collection = ? AND target_id = ?",
            (collection, target_id),
        )
        self._conn.commit()
        return cur.rowcount

    def delete_edges_for_predicate(self, collection: str, predicate: str) -> int:
        """Delete all edges with a given predicate (used when _inverse is removed)."""
        cur = self._execute_write(
            "DELETE FROM edges WHERE collection = ? AND predicate = ?",
            (collection, predicate),
        )
        self._conn.commit()
        return cur.rowcount

    def delete_version_edge(
        self, collection: str, source_id: str, version: int, predicate: str,
    ) -> int:
        """Delete one materialized version edge row."""
        cur = self._execute_write(
            """
            DELETE FROM version_edges
            WHERE collection = ? AND source_id = ? AND version = ? AND predicate = ?
            """,
            (collection, source_id, version, predicate),
        )
        self._conn.commit()
        return cur.rowcount

    def delete_version_edges_for_source(self, collection: str, source_id: str) -> int:
        """Delete all materialized version edges from *source_id*."""
        cur = self._execute_write(
            """
            DELETE FROM version_edges
            WHERE collection = ? AND source_id = ?
            """,
            (collection, source_id),
        )
        self._conn.commit()
        return cur.rowcount

    def rebuild_version_edges_for_source(self, collection: str, source_id: str) -> None:
        """Rebuild all materialized archived-version edges for one source."""
        with self._lock:
            self._execute("BEGIN IMMEDIATE")
            try:
                self._rebuild_version_edges_for_source_unlocked(collection, source_id)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def delete_version_edges_for_target(self, collection: str, target_id: str) -> int:
        """Delete all materialized version edges pointing at *target_id*."""
        cur = self._execute_write(
            """
            DELETE FROM version_edges
            WHERE collection = ? AND target_id = ?
            """,
            (collection, target_id),
        )
        self._conn.commit()
        return cur.rowcount

    def delete_version_edges_for_predicate(self, collection: str, predicate: str) -> int:
        """Delete all materialized version edges with a given predicate."""
        cur = self._execute_write(
            """
            DELETE FROM version_edges
            WHERE collection = ? AND predicate = ?
            """,
            (collection, predicate),
        )
        self._conn.commit()
        return cur.rowcount

    def backfill_version_edges_for_predicate(
        self,
        collection: str,
        predicate: str,
        inverse: str,
        *,
        when_source: str = "",
    ) -> int:
        """Rebuild materialized version edges for one predicate across a collection.

        If *when_source* is a non-empty CEL expression, each version's tags
        are evaluated against it and only matching versions produce edge rows.
        """
        if not predicate or not inverse:
            return 0

        with self._lock:
            self._execute("BEGIN IMMEDIATE")
            try:
                self._execute(
                    """
                    DELETE FROM version_edges
                    WHERE collection = ? AND predicate = ?
                    """,
                    (collection, predicate),
                )
                rows = self._execute(
                    """
                    SELECT id, version, tags_json, created_at, summary
                    FROM document_versions
                    WHERE collection = ?
                    """,
                    (collection,),
                ).fetchall()
                inserted = 0
                for row in rows:
                    tags = _load_tags_json(row["tags_json"])

                    # Evaluate _when condition against this version's tags
                    if when_source:
                        item_ctx = build_item_context(
                            id=row["id"],
                            tags=tags,
                            summary=row["summary"] or "",
                            content_type=tags.get("_content_type", ""),
                            uri=tags.get("_source_uri", ""),
                        )
                        if not eval_when_predicate(when_source, item_ctx):
                            continue

                    for value in tag_values(tags, predicate):
                        target_id = self._version_edge_target_id(value)
                        if not target_id:
                            continue
                        self._execute(
                            """
                            INSERT OR REPLACE INTO version_edges
                                (collection, source_id, version, predicate, target_id, inverse, created)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                collection,
                                row["id"],
                                row["version"],
                                predicate,
                                target_id,
                                inverse,
                                row["created_at"],
                            ),
                        )
                        inserted += 1
                self._conn.commit()
                return inserted
            except Exception:
                self._conn.rollback()
                raise

    def get_inverse_edges(
        self, collection: str, target_id: str,
    ) -> list[tuple[str, str, str]]:
        """Return inverse edges pointing at *target_id*.

        Returns list of (inverse, source_id, created) ordered by inverse
        then created descending.
        """
        rows = self._fetchall(
            """
            SELECT inverse, source_id, created
            FROM edges
            WHERE collection = ? AND target_id = ?
            ORDER BY inverse, created DESC
            """,
            (collection, target_id),
        )
        return [(r["inverse"], r["source_id"], r["created"]) for r in rows]

    def find_by_inverse_edge(
        self,
        collection: str,
        inverse: str,
        source_id: str,
        *,
        limit: int = 200,
    ) -> list[str]:
        """Return target IDs reachable via an inverse edge predicate from *source_id*.

        For example, if source A has ``cites: B``, then
        ``find_by_inverse_edge(coll, "cited_by", "A")`` returns ``["B"]``
        (targets where A's edge points, queried from the inverse direction).

        Uses idx_edges_inverse_source for efficient lookup.
        """
        rows = self._fetchall(
            """
            SELECT DISTINCT target_id
            FROM edges
            WHERE collection = ? AND inverse = ? AND source_id = ?
            LIMIT ?
            """,
            (collection, inverse, source_id, limit),
        )
        return [r["target_id"] for r in rows]

    def get_inverse_version_edges(
        self,
        collection: str,
        target_id: str,
        *,
        limit: int = 200,
    ) -> list[tuple[str, str, str]]:
        """Return inverse edges from materialized archived-version edge rows."""
        if not target_id:
            return []
        rows = self._fetchall(
            """
            SELECT inverse, source_id, MAX(created) AS created
            FROM version_edges
            WHERE collection = ?
              AND target_id = ?
            GROUP BY inverse, source_id
            ORDER BY inverse, created DESC
            LIMIT ?
            """,
            (collection, target_id, limit),
        )
        return [(r["inverse"], r["source_id"], r["created"]) for r in rows]

    def get_forward_version_edges(
        self,
        collection: str,
        source_id: str,
        *,
        limit: int = 1000,
    ) -> list[tuple[int, str, str, str]]:
        """Return archived-version edges originating from *source_id*.

        Returns list of ``(version, predicate, target_id, created)`` ordered by
        version descending, then predicate, then created descending.
        """
        if not source_id:
            return []
        rows = self._fetchall(
            """
            SELECT version, predicate, target_id, created
            FROM version_edges
            WHERE collection = ?
              AND source_id = ?
            ORDER BY version DESC, predicate, created DESC
            LIMIT ?
            """,
            (collection, source_id, limit),
        )
        return [
            (r["version"], r["predicate"], r["target_id"], r["created"])
            for r in rows
        ]

    def get_forward_edges(
        self, collection: str, source_id: str,
    ) -> list[tuple[str, str, str]]:
        """Return forward edges originating from *source_id*.

        Returns list of (predicate, target_id, created) ordered by predicate
        then created descending.
        """
        rows = self._fetchall(
            """
            SELECT predicate, target_id, created
            FROM edges
            WHERE collection = ? AND source_id = ?
            ORDER BY predicate, created DESC
            """,
            (collection, source_id),
        )
        return [(r["predicate"], r["target_id"], r["created"]) for r in rows]

    def find_supernode_candidates(
        self,
        collection: str,
        *,
        min_fan_in: int = 5,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find items with high inbound edge counts (supernodes).

        Returns items scored by ``fan_in × (1 + new_refs)`` where
        ``new_refs`` is the count of edges created after the item's
        ``_supernode_reviewed`` timestamp (or all edges if never reviewed).

        Each result dict has: ``id``, ``fan_in``, ``new_refs``, ``score``,
        ``last_reviewed`` (or None if never reviewed).
        """
        # Single query: join edges with documents to get fan-in counts
        # and the _supernode_reviewed timestamp from tags.
        rows = self._fetchall(
            """
            SELECT
                e.target_id AS id,
                COUNT(1) AS fan_in,
                d.tags_json
            FROM edges e
            JOIN documents d
                ON d.collection = e.collection AND d.id = e.target_id
            WHERE e.collection = ?
            GROUP BY e.target_id
            HAVING fan_in >= ?
            ORDER BY fan_in DESC
            LIMIT ?
            """,
            (collection, min_fan_in, limit * 3),  # overfetch for scoring
        )

        import json as _json

        results: list[dict[str, Any]] = []
        for row in rows:
            item_id = row["id"]
            fan_in = int(row["fan_in"])
            tags_raw = row["tags_json"]
            tags: dict[str, Any] = {}
            if tags_raw:
                try:
                    tags = _json.loads(tags_raw)
                except Exception:
                    pass

            last_reviewed = tags.get("_supernode_reviewed")
            if isinstance(last_reviewed, list):
                last_reviewed = last_reviewed[0] if last_reviewed else None

            # Count new refs since last review
            if last_reviewed:
                new_row = self._fetchone(
                    """
                    SELECT COUNT(1) AS c FROM edges
                    WHERE collection = ? AND target_id = ? AND created > ?
                    """,
                    (collection, item_id, str(last_reviewed)),
                )
                new_refs = int(new_row["c"]) if new_row else 0
            else:
                new_refs = fan_in  # never reviewed — all refs are new

            if new_refs == 0:
                continue  # nothing new since last review

            score = fan_in * (1 + new_refs)
            results.append({
                "id": item_id,
                "fan_in": fan_in,
                "new_refs": new_refs,
                "score": score,
                "last_reviewed": last_reviewed,
                "source": tags.get("_source"),
            })

        # Sort by score descending, limit
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def find_edge_targets(
        self, collection: str, query: str,
    ) -> list[str]:
        """Return edge target IDs whose names appear in *query*.

        Uses word-boundary matching so multi-word names work and
        partial matches are avoided (e.g. "Sam" won't match "Sample").

        Used for entity injection: surface entities the user mentioned
        by name so their edges get traversed.
        """
        import re

        if not query:
            return []
        query_lower = query.lower()
        rows = self._fetchall(
            "SELECT DISTINCT target_id FROM edges WHERE collection = ?",
            (collection,),
        )
        hits = []
        for (target_id,) in rows:
            # Match target_id as a standalone token in the query.
            # Use \b where the boundary is a word character, otherwise
            # use lookaround for whitespace/string boundary to handle
            # IDs like "C++" that end with non-word chars.
            escaped = re.escape(target_id.lower())
            left = r'\b' if re.match(r'\w', target_id) else r'(?:^|(?<=\s))'
            right = r'\b' if re.search(r'\w$', target_id) else r'(?=\s|$)'
            pattern = left + escaped + right
            if re.search(pattern, query_lower):
                hits.append(target_id)
        return hits

    def has_edges(self, collection: str) -> bool:
        """Return True if *collection* has any edges at all."""
        row = self._fetchone(
            "SELECT 1 FROM edges WHERE collection = ? LIMIT 1",
            (collection,),
        )
        return row is not None

    def backfill_exists(self, collection: str, predicate: str) -> bool:
        """Return True if a backfill record exists (pending or completed)."""
        row = self._fetchone(
            "SELECT 1 FROM edge_backfill WHERE collection = ? AND predicate = ?",
            (collection, predicate),
        )
        return row is not None

    def get_backfill_status(
        self, collection: str, predicate: str,
    ) -> Optional[str]:
        """Return the completed timestamp for a backfill, or None if not found."""
        row = self._fetchone(
            """
            SELECT completed FROM edge_backfill
            WHERE collection = ? AND predicate = ?
            """,
            (collection, predicate),
        )
        if row is None:
            return None
        return row["completed"]

    def upsert_backfill(
        self,
        collection: str,
        predicate: str,
        inverse: str,
        completed: Optional[str] = None,
    ) -> None:
        """Insert or update a backfill tracking record."""
        self._execute_write(
            """
            INSERT OR REPLACE INTO edge_backfill
                (collection, predicate, inverse, completed)
            VALUES (?, ?, ?, ?)
            """,
            (collection, predicate, inverse, completed),
        )
        self._conn.commit()

    def delete_backfill(self, collection: str, predicate: str) -> None:
        """Delete a backfill record (used when _inverse is removed from tagdoc)."""
        self._execute_write(
            "DELETE FROM edge_backfill WHERE collection = ? AND predicate = ?",
            (collection, predicate),
        )
        self._conn.commit()

    # -------------------------------------------------------------------------
    # Planner outbox
    # -------------------------------------------------------------------------

    _OUTBOX_STALE_SECONDS = 300  # 5 minutes

    def _outbox_depth(self, table: str) -> int:
        row = self._fetchone(
            f"SELECT COUNT(*) FROM {table} WHERE claimed_by IS NULL"
        )
        return row[0] if row else 0

    def _dequeue_named_outbox(
        self,
        table: str,
        *,
        limit: int = 50,
        claim_id: str | None = None,
    ) -> list[dict]:
        import os

        if claim_id is None:
            claim_id = str(os.getpid())
        now = utc_now()

        with self._lock:
            self._execute(
                f"""
                UPDATE {table}
                SET claimed_by = NULL, claimed_at = NULL,
                    attempts = attempts + 1
                WHERE claimed_by IS NOT NULL
                  AND julianday(?) - julianday(claimed_at)
                      > ? / 86400.0
                """,
                (now, self._OUTBOX_STALE_SECONDS),
            )

            rows = self._fetchall(
                f"""
                SELECT outbox_id, mutation, entity_id, collection,
                       payload_json, created_at
                FROM {table}
                WHERE claimed_by IS NULL
                ORDER BY outbox_id ASC
                LIMIT ?
                """,
                (limit,),
            )

            if not rows:
                self._conn.commit()
                return []

            ids = [r[0] for r in rows]
            placeholders = ",".join("?" for _ in ids)
            self._execute(
                f"""
                UPDATE {table}
                SET claimed_by = ?, claimed_at = ?
                WHERE outbox_id IN ({placeholders})
                """,
                [claim_id, now] + ids,
            )
            self._conn.commit()

        return [
            {
                "outbox_id": r[0],
                "mutation": r[1],
                "entity_id": r[2],
                "collection": r[3],
                "payload_json": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    def _complete_named_outbox(self, table: str, outbox_ids: list[int]) -> None:
        if not outbox_ids:
            return
        placeholders = ",".join("?" for _ in outbox_ids)
        with self._lock:
            self._execute(
                f"DELETE FROM {table} WHERE outbox_id IN ({placeholders})",
                outbox_ids,
            )
            self._conn.commit()

    def _fail_named_outbox(self, table: str, outbox_ids: list[int]) -> None:
        if not outbox_ids:
            return
        placeholders = ",".join("?" for _ in outbox_ids)
        with self._lock:
            self._execute(
                f"""
                UPDATE {table}
                SET claimed_by = NULL, claimed_at = NULL,
                    attempts = attempts + 1
                WHERE outbox_id IN ({placeholders})
                """,
                outbox_ids,
            )
            self._conn.commit()

    def outbox_depth(self) -> int:
        """Count unclaimed outbox rows."""
        return self._outbox_depth("planner_outbox")

    def dequeue_outbox(
        self, limit: int = 50, claim_id: str | None = None,
    ) -> list[dict]:
        """Claim and return unclaimed outbox rows.

        Returns list of dicts with keys: outbox_id, mutation, entity_id,
        collection, payload_json, created_at.
        """
        return self._dequeue_named_outbox(
            "planner_outbox", limit=limit, claim_id=claim_id,
        )

    def complete_outbox(self, outbox_ids: list[int]) -> None:
        """Delete completed outbox rows."""
        self._complete_named_outbox("planner_outbox", outbox_ids)

    def fail_outbox(self, outbox_ids: list[int]) -> None:
        """Release failed outbox rows for retry."""
        self._fail_named_outbox("planner_outbox", outbox_ids)

    def sync_outbox_depth(self) -> int:
        """Count unclaimed markdown-sync outbox rows."""
        return self._outbox_depth("sync_outbox")

    def sync_outbox_bounds(self) -> tuple[int | None, int | None]:
        """Return the oldest and newest markdown-sync outbox IDs."""
        row = self._fetchone(
            "SELECT MIN(outbox_id), MAX(outbox_id) FROM sync_outbox",
        )
        if row is None:
            return None, None
        oldest, newest = row
        return oldest, newest

    def list_sync_outbox_since(
        self,
        *,
        after_outbox_id: int = 0,
        limit: int = 50,
    ) -> list[dict]:
        """List markdown-sync outbox rows newer than ``after_outbox_id``."""
        if limit <= 0:
            return []
        rows = self._fetchall(
            """
            SELECT outbox_id, mutation, entity_id, collection,
                   payload_json, created_at
            FROM sync_outbox
            WHERE outbox_id > ?
            ORDER BY outbox_id ASC
            LIMIT ?
            """,
            (after_outbox_id, limit),
        )
        return [
            {
                "outbox_id": r[0],
                "mutation": r[1],
                "entity_id": r[2],
                "collection": r[3],
                "payload_json": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    def dequeue_sync_outbox(
        self, limit: int = 50, claim_id: str | None = None,
    ) -> list[dict]:
        """Claim and return unclaimed markdown-sync outbox rows."""
        return self._dequeue_named_outbox(
            "sync_outbox", limit=limit, claim_id=claim_id,
        )

    def complete_sync_outbox(self, outbox_ids: list[int]) -> None:
        """Delete completed markdown-sync outbox rows."""
        self._complete_named_outbox("sync_outbox", outbox_ids)

    def fail_sync_outbox(self, outbox_ids: list[int]) -> None:
        """Release failed markdown-sync outbox rows for retry."""
        self._fail_named_outbox("sync_outbox", outbox_ids)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    
    def close(self) -> None:
        """Close all thread-local database connections owned by this store."""
        with self._lock:
            self._close_all_connections(mark_closed=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __del__(self):
        self.close()
