"""Vector store implementation using ChromaDb.

This is the first concrete store implementation. The interface is designed
to be extractable to a Protocol when additional backends are needed.

For now, ChromaDb is the only implementation — and that's fine.
"""

import logging
import threading
import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .model_lock import ModelLock
from .tracing import get_tracer
from .types import Item, SYSTEM_TAG_PREFIX, tag_values, utc_now

logger = logging.getLogger(__name__)


@dataclass
class StoreResult:
    """Result from a store query with raw data before Item conversion."""
    id: str
    summary: str
    tags: dict[str, str]
    distance: float | None = None  # Lower is more similar in Chroma
    
    def to_item(self) -> Item:
        """Convert to Item, transforming distance to similarity score."""
        # Chroma cosine distance is in [0, 2]; convert to 0-1 similarity
        # score = 1 - distance/2 gives 1.0 for identical, 0.0 for orthogonal
        score = None
        if self.distance is not None:
            score = 1.0 - self.distance / 2.0
        return Item(id=self.id, summary=self.summary, tags=self.tags, score=score)


class ChromaStore:
    """Persistent vector store using ChromaDb.
    
    Each collection maps to a ChromaDb collection. Items are stored with:
    - id: The item's URI or custom identifier
    - embedding: Vector representation for similarity search
    - document: The item's summary (stored for retrieval, searchable)
    - metadata: All tags (flattened to strings for Chroma compatibility)
    
    The store is initialized at a specific path and persists across sessions.
    
    Future: This class's public interface could become a Protocol for
    pluggable backends (SQLite+faiss, Postgres+pgvector, etc.)
    """
    
    def __init__(self, store_path: Path, embedding_dimension: Optional[int] = None):
        """Initialize or open a ChromaDb store.

        Args:
            store_path: Directory for persistent storage
            embedding_dimension: Expected dimension of embeddings (for validation).
                Can be None for read-only access; will be set on first write.
        """
        import chromadb
        from chromadb.config import Settings

        self._store_path = store_path
        self._embedding_dimension = embedding_dimension

        # Ensure store directory exists
        store_path.mkdir(parents=True, exist_ok=True)

        # Disable Posthog telemetry completely.  ChromaDB sets posthog.disabled
        # but the default client is created with send=posthog.send (default True),
        # which starts a background consumer thread and registers an atexit handler.
        # That consumer can deadlock on shutdown in lock.acquire().
        try:
            import posthog
            posthog.send = False
            posthog.disabled = True
        except ImportError:
            pass

        # Initialize persistent client
        self._client = chromadb.PersistentClient(
            path=str(store_path / "chroma"),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Cache of collection handles
        self._collections: dict[str, Any] = {}
        # Set during L2→cosine migration so caller can trigger reindex
        self.migrated_to_cosine = False

        # Cross-process write serialization (fcntl file lock)
        self._chroma_lock = ModelLock(store_path / ".chroma.lock")
        # In-process serialization for client/cache state.
        self._state_lock = threading.RLock()
        # Per-thread write-lock depth (enables safe re-entry on nested paths).
        self._write_lock_state = threading.local()
        # Epoch sentinel for staleness detection
        self._epoch_path = store_path / ".chroma.epoch"
        self._last_epoch: float = self._read_epoch()
    
    @property
    def embedding_dimension(self) -> Optional[int]:
        """Current expected embedding dimension (may be None before first write)."""
        return self._embedding_dimension

    def reset_embedding_dimension(self, dimension: int) -> None:
        """Update expected embedding dimension (for provider changes)."""
        self._embedding_dimension = dimension

    @contextmanager
    def _write_guard(self):
        """Acquire cross-process write lock with thread-local re-entrancy."""
        depth = getattr(self._write_lock_state, "depth", 0)
        acquired = depth == 0
        if acquired:
            self._chroma_lock.acquire()
        self._write_lock_state.depth = depth + 1
        try:
            yield
        finally:
            new_depth = getattr(self._write_lock_state, "depth", 1) - 1
            if new_depth <= 0:
                if hasattr(self._write_lock_state, "depth"):
                    delattr(self._write_lock_state, "depth")
                if acquired:
                    self._chroma_lock.release()
            else:
                self._write_lock_state.depth = new_depth

    # -------------------------------------------------------------------------
    # Cross-process coordination
    # -------------------------------------------------------------------------

    def _read_epoch(self) -> float:
        """Read the sentinel file's mtime, or 0.0 if it doesn't exist yet."""
        try:
            return self._epoch_path.stat().st_mtime
        except (FileNotFoundError, OSError):
            return 0.0

    def _bump_epoch(self) -> None:
        """Touch sentinel file and cache the new mtime.

        Called after every ChromaDB write while the write lock is held.
        Other processes detect the mtime change and reload their client.
        """
        self._epoch_path.touch()
        self._last_epoch = self._epoch_path.stat().st_mtime

    def _check_freshness(self) -> None:
        """If another process has written since our last observation, reload.

        Compares the sentinel file mtime to our cached value. A mismatch
        means another process bumped the epoch, so our in-memory hnswlib
        index is stale. Recreating the PersistentClient forces a fresh
        load from disk.

        Caller must hold ``self._state_lock``.
        """
        if self._client is None:
            return  # Store is closed
        current_epoch = self._read_epoch()
        if current_epoch != self._last_epoch:
            self._reload_client()
            self._last_epoch = current_epoch

    def _reload_client(self) -> None:
        """Recreate PersistentClient to pick up on-disk changes.

        Clears the collection handle cache and evicts the shared system cache
        so ChromaDB builds a fresh System that reloads hnswlib from disk.

        Caller must hold ``self._state_lock``.
        """
        import chromadb
        from chromadb.config import Settings

        logger.debug("Reloading ChromaDB client (epoch changed)")
        self._collections.clear()
        # Evict the cached System so PersistentClient creates a fresh one.
        # Without this, the SharedSystemClient returns the stale system.
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        self._client = chromadb.PersistentClient(
            path=str(self._store_path / "chroma"),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

    # -------------------------------------------------------------------------
    # Collection management helpers
    # -------------------------------------------------------------------------

    def _get_collection(self, name: str) -> Any:
        """Get or create a collection by name, migrating L2→cosine if needed.

        Caller must hold ``self._state_lock``.
        """
        if self._client is None:
            raise RuntimeError("Store is closed")
        if name not in self._collections:
            coll = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            # Migrate: if existing collection uses L2, recreate with cosine.
            # This is a write operation — take write lock re-entrantly.
            if coll.metadata.get("hnsw:space") != "cosine":
                with self._write_guard():
                    # Another process may have changed epoch while we waited.
                    self._check_freshness()
                    coll = self._client.get_or_create_collection(
                        name=name,
                        metadata={"hnsw:space": "cosine"},
                    )
                    if coll.metadata.get("hnsw:space") != "cosine":
                        logger.info(
                            "Migrating collection %s from %s to cosine (requires reindex)",
                            name, coll.metadata.get("hnsw:space"),
                        )
                        self._client.delete_collection(name)
                        coll = self._client.create_collection(
                            name=name,
                            metadata={"hnsw:space": "cosine"},
                        )
                        self.migrated_to_cosine = True
                        self._bump_epoch()
            self._collections[name] = coll
        return self._collections[name]
    
    def _tags_to_metadata(self, tags: dict[str, Any]) -> dict[str, Any]:
        """Convert tags to Chroma metadata format.
        
        User tags are encoded as marker booleans so multivalue tags can be
        queried without list metadata support in Chroma.
        """
        metadata: dict[str, Any] = {}
        for raw_key in tags:
            key = str(raw_key)
            values = tag_values(tags, raw_key)
            if not values:
                continue
            if key.startswith(SYSTEM_TAG_PREFIX):
                metadata[key] = str(values[-1])
                continue
            key_cf = key.casefold()
            if len(values) == 1:
                metadata[key_cf] = values[0]
            metadata[self._tag_key_marker(key_cf)] = True
            for value in values:
                metadata[self._tag_value_marker(key_cf, value)] = True
        return metadata
    
    def _metadata_to_tags(self, metadata: dict[str, Any] | None) -> dict[str, str]:
        """Convert Chroma metadata back to tags."""
        if metadata is None:
            return {}
        out: dict[str, str] = {}
        for k, v in metadata.items():
            if k.startswith(self._TAG_KEY_MARKER_PREFIX) or k.startswith(self._TAG_VALUE_MARKER_PREFIX):
                continue
            out[k] = str(v)
        return out
    
    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------
    
    def upsert(
        self,
        collection: str,
        id: str,
        embedding: list[float],
        summary: str,
        tags: dict[str, Any],
    ) -> None:
        """Insert or update an item in the store.

        Args:
            collection: Collection name
            id: Item identifier (URI or custom)
            embedding: Vector embedding
            summary: Human-readable summary (stored as document)
            tags: All tags (source + system + generated)
        """
        # Validate or set embedding dimension
        if self._embedding_dimension is None:
            self._embedding_dimension = len(embedding)
        elif len(embedding) != self._embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dimension}, "
                f"got {len(embedding)}"
            )

        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                coll = self._get_collection(collection)

                # Add timestamp if not present
                now = utc_now()
                if "_updated" not in tags:
                    tags = {**tags, "_updated": now}
                if "_created" not in tags:
                    # Check if item exists to preserve original created time
                    existing = coll.get(ids=[id], include=["metadatas"])
                    if existing["ids"]:
                        old_created = existing["metadatas"][0].get("_created")
                        if old_created:
                            tags = {**tags, "_created": old_created}
                        else:
                            tags = {**tags, "_created": now}
                    else:
                        tags = {**tags, "_created": now}

                # Add date portion for easier date queries
                tags = {**tags, "_updated_date": now[:10]}

                coll.upsert(
                    ids=[id],
                    embeddings=[embedding],
                    documents=[summary],
                    metadatas=[self._tags_to_metadata(tags)],
                )
                self._bump_epoch()

    def upsert_version(
        self,
        collection: str,
        id: str,
        version: int,
        embedding: list[float],
        summary: str,
        tags: dict[str, Any],
    ) -> None:
        """Store an archived version with a versioned ID.

        The versioned ID format is: {id}@v{version}
        Metadata includes _version and _base_id for filtering/navigation.

        Args:
            collection: Collection name
            id: Base item identifier (not versioned)
            version: Version number (1=oldest archived)
            embedding: Vector embedding
            summary: Human-readable summary
            tags: All tags from the archived version
        """
        if self._embedding_dimension is None:
            self._embedding_dimension = len(embedding)
        elif len(embedding) != self._embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dimension}, "
                f"got {len(embedding)}"
            )

        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                coll = self._get_collection(collection)

                # Versioned ID format
                versioned_id = f"{id}@v{version}"

                # Add version metadata
                versioned_tags = dict(tags)
                versioned_tags["_version"] = str(version)
                versioned_tags["_base_id"] = id

                coll.upsert(
                    ids=[versioned_id],
                    embeddings=[embedding],
                    documents=[summary],
                    metadatas=[self._tags_to_metadata(versioned_tags)],
                )
                self._bump_epoch()

    def upsert_with_version(
        self,
        collection: str,
        id: str,
        embedding: list[float],
        summary: str,
        tags: dict[str, Any],
        version_id: str,
        version: int,
        version_embedding: list[float],
        version_summary: str,
        version_tags: dict[str, Any],
    ) -> None:
        """Upsert a document and its archived version in one batch.

        Single lock acquire, single ChromaDB write, single epoch bump.
        """
        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                coll = self._get_collection(collection)

                now = utc_now()
                if "_updated" not in tags:
                    tags = {**tags, "_updated": now}
                if "_created" not in tags:
                    existing = coll.get(ids=[id], include=["metadatas"])
                    if existing["ids"]:
                        old_created = existing["metadatas"][0].get("_created")
                        tags = {**tags, "_created": old_created or now}
                    else:
                        tags = {**tags, "_created": now}
                tags = {**tags, "_updated_date": now[:10]}

                v_tags = dict(version_tags)
                v_tags["_version"] = str(version)
                v_tags["_base_id"] = id

                coll.upsert(
                    ids=[id, version_id],
                    embeddings=[embedding, version_embedding],
                    documents=[summary, version_summary],
                    metadatas=[
                        self._tags_to_metadata(tags),
                        self._tags_to_metadata(v_tags),
                    ],
                )
                self._bump_epoch()

    def upsert_part(
        self,
        collection: str,
        id: str,
        part_num: int,
        embedding: list[float],
        summary: str,
        tags: dict[str, Any],
    ) -> None:
        """Store a document part with a part-numbered ID.

        The part ID format is: {id}@p{part_num}
        Metadata includes _part_num and _base_id for filtering/navigation.

        Args:
            collection: Collection name
            id: Base item identifier (not part-numbered)
            part_num: Part number (1-indexed)
            embedding: Vector embedding
            summary: Human-readable summary of the part
            tags: All tags for this part
        """
        if self._embedding_dimension is None:
            self._embedding_dimension = len(embedding)
        elif len(embedding) != self._embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dimension}, "
                f"got {len(embedding)}"
            )

        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                coll = self._get_collection(collection)

                # Part ID format
                part_id = f"{id}@p{part_num}"

                # Add part metadata
                part_tags = dict(tags)
                part_tags["_part_num"] = str(part_num)
                part_tags["_base_id"] = id

                coll.upsert(
                    ids=[part_id],
                    embeddings=[embedding],
                    documents=[summary],
                    metadatas=[self._tags_to_metadata(part_tags)],
                )
                self._bump_epoch()

    def delete_parts(self, collection: str, id: str) -> int:
        """Delete all parts for a document from the vector store.

        Args:
            collection: Collection name
            id: Base item identifier

        Returns:
            Number of parts deleted
        """
        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                coll = self._get_collection(collection)
                try:
                    parts = coll.get(
                        where={"_base_id": id},
                        include=[],
                    )
                    part_ids = [pid for pid in parts["ids"] if "@p" in pid]
                    if part_ids:
                        coll.delete(ids=part_ids)
                    count = len(part_ids)
                except ValueError:
                    count = 0  # No parts exist
                self._bump_epoch()
                return count

    def delete(self, collection: str, id: str, delete_versions: bool = True) -> bool:
        """Delete an item from the store.

        Args:
            collection: Collection name
            id: Item identifier
            delete_versions: If True, also delete versioned copies ({id}@v{N})

        Returns:
            True if item existed and was deleted, False if not found
        """
        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                coll = self._get_collection(collection)

                # Check existence first
                existing = coll.get(ids=[id])
                if not existing["ids"]:
                    return False

                coll.delete(ids=[id])

                if delete_versions:
                    # Delete all versioned copies and parts
                    # Query by _base_id metadata to find all versions (@v{N}) and parts (@p{N})
                    try:
                        related = coll.get(
                            where={"_base_id": id},
                            include=[],
                        )
                        if related["ids"]:
                            coll.delete(ids=related["ids"])
                    except ValueError:
                        pass  # Metadata filter may fail if no related entries exist

                self._bump_epoch()
                return True

    def update_summary(self, collection: str, id: str, summary: str) -> bool:
        """Update just the summary of an existing item.

        Used by lazy summarization to replace placeholder summaries
        with real generated summaries.

        Args:
            collection: Collection name
            id: Item identifier
            summary: New summary text

        Returns:
            True if item was updated, False if not found
        """
        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                coll = self._get_collection(collection)

                # Get existing item — include embeddings so we can preserve them.
                # ChromaDB re-embeds when documents= is passed, using its default
                # embedding function (384d all-MiniLM-L6-v2) which would corrupt
                # embeddings produced by the configured provider (e.g. 768d nomic).
                existing = coll.get(ids=[id], include=["metadatas", "embeddings"])
                if not existing["ids"]:
                    return False

                # Update metadata with new timestamp
                metadata = existing["metadatas"][0] or {}
                now = utc_now()
                metadata["_updated"] = now
                metadata["_updated_date"] = now[:10]

                # Update document text + metadata, passing existing embeddings
                # to prevent ChromaDB from re-embedding with its default function.
                coll.update(
                    ids=[id],
                    documents=[summary],
                    embeddings=[existing["embeddings"][0]],
                    metadatas=[metadata],
                )
                self._bump_epoch()
                return True

    def update_tags(self, collection: str, id: str, tags: dict[str, Any]) -> bool:
        """Update tags of an existing item without changing embedding or summary.

        Args:
            collection: Collection name
            id: Item identifier
            tags: New tags dict (replaces existing)

        Returns:
            True if item was updated, False if not found
        """
        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                coll = self._get_collection(collection)

                # Get existing item
                existing = coll.get(ids=[id], include=["metadatas"])
                if not existing["ids"]:
                    return False

                # Update timestamp
                now = utc_now()
                tags = dict(tags)  # Copy to avoid mutating input
                tags["_updated"] = now
                tags["_updated_date"] = now[:10]

                # Convert to metadata format
                metadata = self._tags_to_metadata(tags)

                coll.update(
                    ids=[id],
                    metadatas=[metadata],
                )
                self._bump_epoch()
                return True

    def rewrite_tags(self, collection: str, id: str, tags: dict[str, Any]) -> bool:
        """Rewrite metadata tags in-place without mutating timestamps.

        Used for metadata-shape migrations (e.g., legacy key/value metadata
        to marker-based tag metadata) where embeddings and summary stay the same.
        """
        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                coll = self._get_collection(collection)

                existing = coll.get(ids=[id], include=["metadatas"])
                if not existing["ids"]:
                    return False

                coll.update(
                    ids=[id],
                    metadatas=[self._tags_to_metadata(tags)],
                )
                self._bump_epoch()
                return True

    # -------------------------------------------------------------------------
    # Read Operations
    # -------------------------------------------------------------------------
    
    def get(self, collection: str, id: str) -> StoreResult | None:
        """Retrieve a specific item by ID.
        
        Args:
            collection: Collection name
            id: Item identifier
            
        Returns:
            StoreResult if found, None otherwise
        """
        with self._state_lock:
            self._check_freshness()
            coll = self._get_collection(collection)
            result = coll.get(
                ids=[id],
                include=["documents", "metadatas"],
            )

            if not result["ids"]:
                return None

            return StoreResult(
                id=result["ids"][0],
                summary=result["documents"][0] or "",
                tags=self._metadata_to_tags(result["metadatas"][0]),
            )

    def exists(self, collection: str, id: str) -> bool:
        """Check if an item exists in the store."""
        with self._state_lock:
            self._check_freshness()
            coll = self._get_collection(collection)
            result = coll.get(ids=[id], include=[])
            return bool(result["ids"])

    def get_embedding(self, collection: str, id: str) -> list[float] | None:
        """Retrieve the stored embedding for a document.

        Args:
            collection: Collection name
            id: Item identifier

        Returns:
            Embedding vector if found, None otherwise
        """
        with self._state_lock:
            self._check_freshness()
            coll = self._get_collection(collection)
            result = coll.get(ids=[id], include=["embeddings"])
            if not result["ids"] or result["embeddings"] is None or len(result["embeddings"]) == 0:
                return None
            return list(result["embeddings"][0])

    _LIST_IDS_PAGE = 5000

    def list_ids(self, collection: str) -> list[str]:
        """List all document IDs in a collection.

        Paginates internally to avoid loading all IDs in a single
        ChromaDB call, which can be slow or OOM on large collections.

        Args:
            collection: Collection name

        Returns:
            List of document IDs
        """
        with self._state_lock:
            self._check_freshness()
            coll = self._get_collection(collection)
            all_ids: list[str] = []
            offset = 0
            while True:
                result = coll.get(include=[], limit=self._LIST_IDS_PAGE, offset=offset)
                batch = result["ids"]
                if not batch:
                    break
                all_ids.extend(batch)
                if len(batch) < self._LIST_IDS_PAGE:
                    break
                offset += len(batch)
            return all_ids

    def find_missing_ids(self, collection: str, ids: list[str]) -> set[str]:
        """Given a list of IDs, return those not present in ChromaDB.

        Checks in batches to avoid oversized requests.

        Args:
            collection: Collection name
            ids: IDs to check

        Returns:
            Set of IDs not found in ChromaDB
        """
        with self._state_lock:
            self._check_freshness()
            coll = self._get_collection(collection)
            missing: set[str] = set()
            batch_size = self._LIST_IDS_PAGE
            for i in range(0, len(ids), batch_size):
                batch = ids[i:i + batch_size]
                result = coll.get(ids=batch, include=[])
                found = set(result["ids"])
                missing.update(id for id in batch if id not in found)
            return missing

    def query_embedding(
        self,
        collection: str,
        embedding: list[float],
        limit: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[StoreResult]:
        """Query by embedding similarity.

        Args:
            collection: Collection name
            embedding: Query embedding vector
            limit: Maximum results to return
            where: Optional metadata filter (Chroma where clause)

        Returns:
            List of results ordered by similarity (most similar first)
        """
        with get_tracer("store").start_as_current_span(
            "store.query_embedding",
            attributes={
                "collection": collection,
                "limit": limit,
                "where": bool(where),
            },
        ) as span:
            with self._state_lock:
                self._check_freshness()
                coll = self._get_collection(collection)

                query_params = {
                    "query_embeddings": [embedding],
                    "n_results": limit,
                    "include": ["documents", "metadatas", "distances"],
                }
                if where:
                    normalized_where = self._normalize_where(where)
                    if normalized_where:
                        query_params["where"] = normalized_where

                result = coll.query(**query_params)

                results = []
                # Chroma normally returns nested lists, but some embedding/store
                # combinations return NumPy arrays.  Do not use array truthiness:
                # it raises when a query has more than one result.
                distances = result.get("distances")
                has_distances = distances is not None and len(distances) > 0
                for i, id in enumerate(result["ids"][0]):
                    results.append(StoreResult(
                        id=id,
                        summary=result["documents"][0][i] or "",
                        tags=self._metadata_to_tags(result["metadatas"][0][i]),
                        distance=distances[0][i] if has_distances else None,
                    ))

                span.set_attribute("result_count", len(results))
                return results
    
    def query_metadata(
        self,
        collection: str,
        where: dict[str, Any],
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoreResult]:
        """Query by metadata filter (tag query).

        Args:
            collection: Collection name
            where: Chroma where clause for metadata filtering
            limit: Maximum results to return
            offset: Number of results to skip (for pagination)

        Returns:
            List of matching results (no particular order)
        """
        with get_tracer("store").start_as_current_span(
            "store.query_metadata",
            attributes={
                "collection": collection,
                "limit": limit,
                "offset": offset,
                "where": bool(where),
            },
        ) as span:
            with self._state_lock:
                self._check_freshness()
                coll = self._get_collection(collection)

                normalized_where = self._normalize_where(where)
                result = coll.get(
                    where=normalized_where,
                    limit=limit,
                    offset=offset,
                    include=["documents", "metadatas"],
                )

                results = []
                for i, id in enumerate(result["ids"]):
                    results.append(StoreResult(
                        id=id,
                        summary=result["documents"][i] or "",
                        tags=self._metadata_to_tags(result["metadatas"][i]),
                    ))

                span.set_attribute("result_count", len(results))
                return results
    
    # -------------------------------------------------------------------------
    # Collection Management
    # -------------------------------------------------------------------------
    
    def list_collections(self) -> list[str]:
        """List all collection names in the store."""
        with self._state_lock:
            self._check_freshness()
            collections = self._client.list_collections()
            return [c.name for c in collections]

    def delete_collection(self, name: str) -> bool:
        """Delete an entire collection.

        Args:
            name: Collection name

        Returns:
            True if collection existed and was deleted
        """
        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                try:
                    self._client.delete_collection(name)
                    self._collections.pop(name, None)
                    self._bump_epoch()
                    return True
                except ValueError:
                    return False

    def count(self, collection: str) -> int:
        """Return the number of items in a collection."""
        with self._state_lock:
            self._check_freshness()
            coll = self._get_collection(collection)
            return coll.count()

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    def get_entries_full(
        self, collection: str, ids: list[str]
    ) -> list[dict[str, Any]]:
        """Batch get entries with embeddings, summaries, and metadata.

        Returns list of dicts with keys: id, embedding, summary, tags.
        Only entries that exist are returned (missing IDs are silently skipped).
        """
        if not ids:
            return []
        with self._state_lock:
            self._check_freshness()
            coll = self._get_collection(collection)
            result = coll.get(
                ids=ids,
                include=["embeddings", "documents", "metadatas"],
            )
            entries = []
            if result["ids"]:
                for i, entry_id in enumerate(result["ids"]):
                    embedding = None
                    if result["embeddings"] is not None and i < len(result["embeddings"]):
                        emb = result["embeddings"][i]
                        embedding = list(emb) if emb is not None else None
                    summary = ""
                    if result["documents"] is not None and i < len(result["documents"]):
                        summary = result["documents"][i] or ""
                    tags = {}
                    if result["metadatas"] is not None and i < len(result["metadatas"]):
                        tags = self._metadata_to_tags(result["metadatas"][i])
                    entries.append({
                        "id": entry_id,
                        "embedding": embedding,
                        "summary": summary,
                        "tags": tags,
                    })
            return entries

    def upsert_batch(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        summaries: list[str],
        tags: list[dict[str, Any]],
    ) -> None:
        """Batch upsert entries with embeddings.

        All lists must have the same length. Tags are converted to store
        metadata format internally.
        """
        if not ids:
            return
        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                coll = self._get_collection(collection)
                metadatas = [self._tags_to_metadata(t) for t in tags]
                coll.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=summaries,
                    metadatas=metadatas,
                )
                self._bump_epoch()

    def delete_entries(self, collection: str, ids: list[str]) -> None:
        """Delete specific entries by ID.

        Unlike delete(), this does not expand version IDs — it deletes
        exactly the IDs given. Silently ignores IDs that don't exist.
        """
        if not ids:
            return
        with self._state_lock:
            with self._write_guard():
                self._check_freshness()
                coll = self._get_collection(collection)
                try:
                    coll.delete(ids=ids)
                except ValueError:
                    pass  # Some IDs may not exist
                self._bump_epoch()

    # -------------------------------------------------------------------------
    # Resource Management
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Close ChromaDB client and release resources.

        Evicts the shared system cache so the underlying System (hnswlib
        indices, SQLite connections) can be garbage-collected.
        """
        with self._state_lock:
            self._collections.clear()
            if self._client is not None:
                import chromadb.api.client
                chromadb.api.client.SharedSystemClient.clear_system_cache()
                self._client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close resources."""
        self.close()
        return False

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass  # Suppress errors during garbage collection
    _TAG_KEY_MARKER_PREFIX = "_tk::"
    _TAG_VALUE_MARKER_PREFIX = "_tv::"

    @classmethod
    def _tag_key_marker(cls, key: str) -> str:
        return f"{cls._TAG_KEY_MARKER_PREFIX}{key}"

    @classmethod
    def _tag_value_marker(cls, key: str, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
        return f"{cls._TAG_VALUE_MARKER_PREFIX}{key}::{digest}"

    def build_tag_where(self, tags: dict[str, Any]) -> dict[str, Any] | None:
        """Build Chroma where-clause from multivalue tag filters."""
        conditions: list[dict[str, Any]] = []
        for raw_key in tags:
            raw_key_str = str(raw_key)
            key = raw_key_str.casefold()
            values = tag_values(tags, raw_key)
            if not values:
                continue
            if raw_key_str.startswith(SYSTEM_TAG_PREFIX):
                for value in values:
                    conditions.append({raw_key_str: str(value)})
            else:
                conditions.append({self._tag_key_marker(key): True})
                for value in values:
                    conditions.append({self._tag_value_marker(key, value): True})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def _normalize_where(self, where: dict[str, Any] | None) -> dict[str, Any] | None:
        """Translate legacy key=value filters to marker-based filters."""
        if where is None:
            return None
        if "$and" in where:
            clauses = [self._normalize_where(clause) for clause in where["$and"]]
            clauses = [c for c in clauses if c]
            if not clauses:
                return None
            return clauses[0] if len(clauses) == 1 else {"$and": clauses}
        if "$or" in where:
            clauses = [self._normalize_where(clause) for clause in where["$or"]]
            clauses = [c for c in clauses if c]
            if not clauses:
                return None
            return clauses[0] if len(clauses) == 1 else {"$or": clauses}
        if len(where) != 1:
            return where

        key, value = next(iter(where.items()))
        if key.startswith(SYSTEM_TAG_PREFIX):
            return {key: value}
        if key.startswith(self._TAG_KEY_MARKER_PREFIX) or key.startswith(self._TAG_VALUE_MARKER_PREFIX):
            return {key: value}
        if isinstance(value, dict):
            return {key: value}

        key_cf = key.casefold()
        val = str(value)
        return {
            "$and": [
                {self._tag_key_marker(key_cf): True},
                {self._tag_value_marker(key_cf, val): True},
            ]
        }

    def has_tag_markers(self, collection: str, id: str) -> bool:
        """Check whether an indexed item's metadata includes tag markers."""
        with self._state_lock:
            self._check_freshness()
            coll = self._get_collection(collection)
            result = coll.get(ids=[id], include=["metadatas"])
            if not result["ids"]:
                return False
            metadata = result["metadatas"][0] or {}
            return any(
                k.startswith(self._TAG_KEY_MARKER_PREFIX)
                or k.startswith(self._TAG_VALUE_MARKER_PREFIX)
                for k in metadata
            )
