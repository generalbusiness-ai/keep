"""Embedding cache using SQLite.

Wraps any EmbeddingProvider to cache embeddings by content hash,
avoiding redundant embedding calls for unchanged content.
"""

import hashlib
import json
import logging
import sqlite3
import struct
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..compute_context import apply_context_attrs, current_counters
from ..const import SQLITE_BUSY_TIMEOUT_MS
from ..provider_identity import provider_model_name
from ..tracing import get_tracer
from .base import EmbedTask, EmbeddingProvider

logger = logging.getLogger(__name__)

SUMMARY_INTERVAL_S = 600.0  # 10 minutes — per-flow `compute:` lines now carry near-term detail.
SUMMARY_TEXT_THRESHOLD = 100
WORKING_SET_SHORT_WINDOW_S = 5 * 60.0
WORKING_SET_LONG_WINDOW_S = 30 * 60.0


class EmbeddingCache:
    """SQLite-based embedding cache.
    
    Cache key is SHA256(model_name + content), so different models
    don't share cached embeddings.
    """
    
    def __init__(self, cache_path: Path, max_entries: int = 50000):
        """Initialize.

        Args:
        cache_path: Path to SQLite database file
        max_entries: Maximum cache entries (LRU eviction when exceeded).
        """
        self._cache_path = cache_path
        self._max_entries = max_entries
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._pending_touches: set[str] = set()
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._cache_path), check_same_thread=False)

        # Enable WAL mode for better concurrent access across processes
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Wait up to 5 seconds for locks instead of failing immediately
        self._conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_accessed 
            ON embedding_cache(last_accessed)
        """)
        self._conn.commit()
    
    def _hash_key(self, model_name: str, content: str) -> str:
        """Generate cache key from model name and content."""
        key_input = f"{model_name}:{content}"
        return hashlib.sha256(key_input.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_embedding(embedding: list[float]) -> bytes:
        """Serialize embedding to binary format (little-endian float32)."""
        return struct.pack(f"<{len(embedding)}f", *embedding)

    @staticmethod
    def _deserialize_embedding(data: bytes | str) -> list[float]:
        """Deserialize embedding from binary or legacy JSON format."""
        if isinstance(data, bytes):
            n = len(data) // 4
            return list(struct.unpack(f"<{n}f", data))
        # Legacy JSON format
        return json.loads(data)

    def get(self, model_name: str, content: str) -> Optional[list[float]]:
        """Get cached embedding if it exists.

        Defers last_accessed update to the next write (put/flush) to avoid
        a SQLite write transaction on every cache hit.
        """
        content_hash = self._hash_key(model_name, content)

        with self._lock:
            if self._conn is None:
                return None
            cursor = self._conn.execute(
                "SELECT embedding FROM embedding_cache WHERE content_hash = ?",
                (content_hash,)
            )
            row = cursor.fetchone()

            if row is not None:
                self._pending_touches.add(content_hash)
                return self._deserialize_embedding(row[0])

        return None
    
    def _flush_touches(self) -> None:
        """Batch-update last_accessed for recently read entries.

        Must be called with self._lock held.
        """
        if not self._pending_touches or self._conn is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        hashes = list(self._pending_touches)
        self._pending_touches.clear()
        placeholders = ",".join("?" for _ in hashes)
        self._conn.execute(
            f"UPDATE embedding_cache SET last_accessed = ? "
            f"WHERE content_hash IN ({placeholders})",
            [now, *hashes],
        )

    def put(
        self,
        model_name: str,
        content: str,
        embedding: list[float]
    ) -> int:
        """Cache an embedding.

        Also flushes deferred last_accessed updates from get() hits.
        Evicts oldest entries if cache exceeds max_entries.

        Returns number of entries evicted during this write.
        """
        content_hash = self._hash_key(model_name, content)
        now = datetime.now(timezone.utc).isoformat()
        embedding_blob = self._serialize_embedding(embedding)

        with self._lock:
            if self._conn is None:
                return 0
            self._flush_touches()
            self._conn.execute("""
                INSERT OR REPLACE INTO embedding_cache
                (content_hash, model_name, embedding, dimension, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (content_hash, model_name, embedding_blob, len(embedding), now, now))
            self._conn.commit()

            # Evict old entries if needed
            return self._maybe_evict()
    
    def _maybe_evict(self) -> int:
        """Evict oldest entries if cache exceeds max size.

        Returns number of entries evicted.
        """
        with self._lock:
            if self._conn is None:
                return 0
            cursor = self._conn.execute("SELECT COUNT(*) FROM embedding_cache")
            count = cursor.fetchone()[0]

            if count > self._max_entries:
                # Delete oldest 10% by last_accessed
                evict_count = max(1, count // 10)
                self._conn.execute("""
                    DELETE FROM embedding_cache
                    WHERE content_hash IN (
                        SELECT content_hash FROM embedding_cache
                        ORDER BY last_accessed ASC
                        LIMIT ?
                    )
                """, (evict_count,))
                self._conn.commit()
                return evict_count
            return 0
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            if self._conn is None:
                return {
                    "entries": 0,
                    "models": 0,
                    "max_entries": self._max_entries,
                    "cache_path": str(self._cache_path),
                }
            cursor = self._conn.execute("""
                SELECT COUNT(*), COUNT(DISTINCT model_name)
                FROM embedding_cache
            """)
            count, models = cursor.fetchone()
            result = {
                "entries": count,
                "models": models,
                "max_entries": self._max_entries,
                "cache_path": str(self._cache_path),
            }
            try:
                result["db_bytes"] = self._cache_path.stat().st_size
            except OSError:
                result["db_bytes"] = 0
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM embedding_cache WHERE typeof(embedding) = 'text'"
            )
            legacy_count = cursor.fetchone()[0]
            if legacy_count > 0:
                result["legacy_json_entries"] = legacy_count
            return result
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute("DELETE FROM embedding_cache")
            self._conn.commit()

    def migrate(self) -> int:
        """Bulk-convert legacy JSON embeddings to binary format.

        Returns number of entries migrated.
        """
        migrated = 0
        with self._lock:
            if self._conn is None:
                return 0
            cursor = self._conn.execute(
                "SELECT content_hash, embedding FROM embedding_cache"
            )
            for content_hash, data in cursor.fetchall():
                if isinstance(data, str):
                    embedding = json.loads(data)
                    binary = self._serialize_embedding(embedding)
                    self._conn.execute(
                        "UPDATE embedding_cache SET embedding = ? WHERE content_hash = ?",
                        (binary, content_hash)
                    )
                    migrated += 1
            if migrated:
                self._conn.commit()
        return migrated

    def close(self) -> None:
        """Close the database connection, flushing pending touches."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._flush_touches()
                    self._conn.commit()
                except Exception:
                    pass
                self._conn.close()
                self._conn = None

    def __del__(self) -> None:
        """Ensure connection is closed on cleanup."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()
        return False


class CachingEmbeddingProvider:
    """Wrapper that adds caching to any EmbeddingProvider.
    
    Usage:
        base_provider = SentenceTransformerEmbedding(model="all-MiniLM-L6-v2")
        cached = CachingEmbeddingProvider(base_provider, cache_path)
    """
    
    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_path: Path,
        max_entries: int = 50000,
        provider_name: str | None = None,
    ):
        self._provider = provider
        self._cache = EmbeddingCache(cache_path, max_entries)
        self._provider_name = provider_name or type(provider).__name__.lower()
        self._hits = 0
        self._misses = 0
        self._requests = 0
        self._batch_requests = 0
        self._texts = 0
        self._inserts = 0
        self._evictions = 0
        self._stats_lock = threading.Lock()
        self._started_at = time.monotonic()
        self._last_summary_at = self._started_at
        self._last_summary_texts = 0
        # Track recent requested keys in memory so we can estimate the hot
        # working set without persisting any content-derived identifiers.
        self._recent_keys: deque[tuple[float, str]] = deque()

    @property
    def provider_name(self) -> str:
        """Configured provider family name."""
        return self._provider_name
    
    @property
    def model_name(self) -> str:
        """Get the underlying provider's model name."""
        return provider_model_name(self._provider)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension from the wrapped provider."""
        return self._provider.dimension
    
    def _cache_key_model(self, task: EmbedTask) -> str:
        """Cache key model name, scoped by task so document/query don't collide."""
        return f"{self.model_name}:{task.value}"

    def _record_request(self, cache_model: str, texts: list[str], *, batch: bool) -> None:
        """Track request volume and recent key activity for cache summaries."""
        now = time.monotonic()
        with self._stats_lock:
            self._requests += 1
            if batch:
                self._batch_requests += 1
            self._texts += len(texts)
            for text in texts:
                key = self._cache._hash_key(cache_model, text)
                self._recent_keys.append((now, key))
            self._prune_recent_keys_locked(now)

    def _prune_recent_keys_locked(self, now: float) -> None:
        """Drop working-set observations older than the long summary window."""
        cutoff = now - WORKING_SET_LONG_WINDOW_S
        while self._recent_keys and self._recent_keys[0][0] < cutoff:
            self._recent_keys.popleft()

    def _working_set_sizes_locked(self, now: float) -> tuple[int, int]:
        """Return distinct keys touched in the recent 5m and 30m windows."""
        self._prune_recent_keys_locked(now)
        short_cutoff = now - WORKING_SET_SHORT_WINDOW_S
        short_keys: set[str] = set()
        long_keys: set[str] = set()
        for seen_at, key in self._recent_keys:
            long_keys.add(key)
            if seen_at >= short_cutoff:
                short_keys.add(key)
        return len(short_keys), len(long_keys)

    def snapshot(self) -> dict:
        """Return a structured summary for investigation and status output."""
        cache_stats = self._cache.stats()
        now = time.monotonic()
        with self._stats_lock:
            hits = self._hits
            misses = self._misses
            requests = self._requests
            batch_requests = self._batch_requests
            texts = self._texts
            inserts = self._inserts
            evictions = self._evictions
            working_set_5m, working_set_30m = self._working_set_sizes_locked(now)

        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0
        return {
            **cache_stats,
            "provider": self.provider_name,
            "model": self.model_name,
            "requests": requests,
            "batch_requests": batch_requests,
            "texts": texts,
            "hits": hits,
            "misses": misses,
            "hit_rate": f"{hit_rate:.1%}",
            "hit_rate_value": hit_rate,
            "inserts": inserts,
            "evictions": evictions,
            "uptime_s": round(now - self._started_at, 1),
            "working_set_5m": working_set_5m,
            "working_set_30m": working_set_30m,
        }

    def log_summary(self, reason: str, *, force: bool = False) -> None:
        """Emit a bounded info log with the cache working-set summary."""
        now = time.monotonic()
        with self._stats_lock:
            texts = self._texts
            if not force:
                if texts == 0:
                    return
                enough_time = (now - self._last_summary_at) >= SUMMARY_INTERVAL_S
                enough_texts = (texts - self._last_summary_texts) >= SUMMARY_TEXT_THRESHOLD
                if not (enough_time or enough_texts):
                    return
            self._last_summary_at = now
            self._last_summary_texts = texts

        snapshot = self.snapshot()

        logger.info(
            "Embedding cache summary "
            "reason=%s provider=%s model=%s instance_requests=%d "
            "instance_batch_requests=%d instance_texts=%d instance_hits=%d "
            "instance_misses=%d hit_rate=%s instance_inserts=%d "
            "instance_evictions=%d working_set_5m=%d working_set_30m=%d "
            "db_entries=%d cache_capacity=%d db_models=%d db_bytes=%d "
            "uptime_s=%.1f",
            reason,
            snapshot["provider"],
            snapshot["model"],
            snapshot["requests"],
            snapshot["batch_requests"],
            snapshot["texts"],
            snapshot["hits"],
            snapshot["misses"],
            snapshot["hit_rate"],
            snapshot["inserts"],
            snapshot["evictions"],
            snapshot["working_set_5m"],
            snapshot["working_set_30m"],
            snapshot["entries"],
            snapshot["max_entries"],
            snapshot["models"],
            snapshot.get("db_bytes", 0),
            snapshot["uptime_s"],
        )

    def _span_attrs(self, *, task: EmbedTask, batch_size: int) -> dict[str, object]:
        """Stable request attributes for embed tracing."""
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "task": task.value,
            "batch_size": batch_size,
        }

    def _set_request_summary_attrs(
        self,
        span,
        *,
        hit_count: int,
        miss_count: int,
    ) -> None:
        """Attach investigation attributes to a request span."""
        now = time.monotonic()
        with self._stats_lock:
            working_set_5m, working_set_30m = self._working_set_sizes_locked(now)
        uptime_s = round(now - self._started_at, 1)
        source = "cached" if miss_count == 0 else "computed" if hit_count == 0 else "mixed"
        span.set_attribute("source", source)
        span.set_attribute("hit_count", hit_count)
        span.set_attribute("miss_count", miss_count)
        span.set_attribute("process_uptime_s", uptime_s)
        span.set_attribute("working_set_5m", working_set_5m)
        span.set_attribute("working_set_30m", working_set_30m)
        span.set_attribute("max_entries", self._cache._max_entries)

    def embed(self, text: str, *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[float]:
        """Get embedding, using cache when available.

        Cache failures are non-fatal — falls through to the real provider.
        """
        _tracer = get_tracer("embed")

        cache_model = self._cache_key_model(task)
        self._record_request(cache_model, [text], batch=False)

        with _tracer.start_as_current_span(
            "embed.request",
            attributes=self._span_attrs(task=task, batch_size=1),
        ) as span:
            apply_context_attrs(span)
            counters = current_counters()
            # Check cache (fail-safe)
            try:
                cached = self._cache.get(cache_model, text)
                if cached is not None:
                    with self._stats_lock:
                        self._hits += 1
                    if counters is not None:
                        counters.add_embed(hit=True, ms=0.0)
                    self._set_request_summary_attrs(span, hit_count=1, miss_count=0)
                    self.log_summary("interval")
                    return cached
            except Exception as e:
                logger.debug("Embedding cache read failed: %s", e)

            # Cache miss - compute embedding
            with self._stats_lock:
                self._misses += 1
            with _tracer.start_as_current_span(
                "embed.compute",
                attributes=self._span_attrs(task=task, batch_size=1),
            ) as compute_span:
                apply_context_attrs(compute_span)
                _t0 = time.monotonic()
                embedding = self._provider.embed(text, task=task)
                _compute_ms = (time.monotonic() - _t0) * 1000.0
            if counters is not None:
                counters.add_embed(hit=False, ms=_compute_ms)

            # Store in cache (fail-safe)
            try:
                evicted = self._cache.put(cache_model, text, embedding)
                with self._stats_lock:
                    self._inserts += 1
                    self._evictions += evicted
            except Exception as e:
                logger.debug("Embedding cache write failed: %s", e)

            self._set_request_summary_attrs(span, hit_count=0, miss_count=1)
            self.log_summary("interval")
            return embedding

    def embed_batch(self, texts: list[str], *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[list[float]]:
        """Get embeddings for batch, using cache where available.

        Only computes embeddings for cache misses. Cache failures
        are non-fatal — falls through to the real provider.
        """
        _tracer = get_tracer("embed")
        results: list[Optional[list[float]]] = [None] * len(texts)
        to_embed: list[tuple[int, str]] = []

        cache_model = self._cache_key_model(task)
        self._record_request(cache_model, texts, batch=True)

        with _tracer.start_as_current_span(
            "embed.batch",
            attributes=self._span_attrs(task=task, batch_size=len(texts)),
        ) as span:
            apply_context_attrs(span)
            counters = current_counters()
            # Check cache for each text (fail-safe)
            for i, text in enumerate(texts):
                try:
                    cached = self._cache.get(cache_model, text)
                    if cached is not None:
                        with self._stats_lock:
                            self._hits += 1
                        if counters is not None:
                            counters.add_embed(hit=True, ms=0.0)
                        results[i] = cached
                        continue
                except Exception as e:
                    logger.debug("Embedding cache read failed: %s", e)
                with self._stats_lock:
                    self._misses += 1
                to_embed.append((i, text))

            # Batch embed cache misses
            if to_embed:
                indices, texts_to_embed = zip(*to_embed)
                with _tracer.start_as_current_span(
                    "embed.compute",
                    attributes={
                        **self._span_attrs(task=task, batch_size=len(texts)),
                        "count": len(texts_to_embed),
                    },
                ) as compute_span:
                    apply_context_attrs(compute_span)
                    _t0 = time.monotonic()
                    embeddings = self._provider.embed_batch(list(texts_to_embed), task=task)
                    _compute_ms = (time.monotonic() - _t0) * 1000.0
                if counters is not None:
                    # Distribute the batch wall time across the misses so per-flow
                    # totals stay comparable to single-call timing.
                    per_call = _compute_ms / max(1, len(to_embed))
                    for _ in to_embed:
                        counters.add_embed(hit=False, ms=per_call)

                for idx, text, embedding in zip(indices, texts_to_embed, embeddings):
                    results[idx] = embedding
                    try:
                        evicted = self._cache.put(cache_model, text, embedding)
                        with self._stats_lock:
                            self._inserts += 1
                            self._evictions += evicted
                    except Exception as e:
                        logger.debug("Embedding cache write failed: %s", e)

            self._set_request_summary_attrs(
                span,
                hit_count=len(texts) - len(to_embed),
                miss_count=len(to_embed),
            )
            self.log_summary("interval")
            return results  # type: ignore
    
    def stats(self) -> dict:
        """Get cache and hit/miss statistics."""
        return self.snapshot()

    def close(self) -> None:
        """Flush a final summary and close the underlying cache."""
        self.log_summary("close", force=True)
        self._cache.close()
