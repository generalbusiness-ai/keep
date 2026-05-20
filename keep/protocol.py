"""Protocol definitions for keep's flow host and storage backends.

Defines interface contracts at three levels:
- FlowHostProtocol: the stable hosted/local execution boundary
- KeeperProtocol: legacy richer object API used during migration
- VectorStoreProtocol / DocumentStoreProtocol: internal storage backends
"""

from typing import Any, Callable, Iterator, Optional, Protocol, runtime_checkable

from .document_store import DocumentRecord, PartInfo, VersionInfo
from .pending_summaries import PendingSummary
from .store import StoreResult
from .types import Item, ItemContext, TagMap


@runtime_checkable
class FlowHostProtocol(Protocol):
    """Minimal backend-neutral flow execution interface.

    This is the stable semantic boundary shared by local and hosted stores.
    Higher-level helpers like get/put/find live above this interface.
    """

    def run_flow(
        self,
        state: str,
        *,
        params: dict[str, Any] | None = None,
        budget: int | None = None,
        cursor_token: str | None = None,
        state_doc_yaml: str | None = None,
        writable: bool = True,
    ) -> Any:
        """Run a flow and return a FlowResult-like object."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class KeeperProtocol(FlowHostProtocol, Protocol):
    """The public interface for reflective memory operations.

    Implemented by:
    - Keeper (local backend)
    - RemoteKeeper (hosted keep.generalbusiness.ai API)
    """

    # -- Write operations --

    def put(
        self,
        content: Optional[str] = None,
        *,
        uri: Optional[str] = None,
        id: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[TagMap] = None,
        created_at: Optional[str] = None,
    ) -> Item: ...

    def set_now(
        self,
        content: str,
        *,
        scope: Optional[str] = None,
        tags: Optional[TagMap] = None,
    ) -> Item: ...

    def tag(
        self,
        id: str,
        tags: Optional[TagMap] = None,
    ) -> Optional[Item]: ...

    def delete(
        self,
        id: str,
        *,
        delete_versions: bool = True,
    ) -> bool: ...

    def revert(self, id: str) -> Optional[Item]: ...

    def move(
        self,
        name: str,
        *,
        source_id: str = "now",
        tags: Optional[TagMap] = None,
        only_current: bool = False,
    ) -> Item: ...

    # -- Query operations --

    def find(
        self,
        query: Optional[str] = None,
        *,
        tags: Optional[TagMap] = None,
        similar_to: Optional[str] = None,
        limit: int = 10,
        since: Optional[str] = None,
        until: Optional[str] = None,
        include_self: bool = False,
        include_hidden: bool = False,
        deep: bool = False,
    ) -> list[Item]: ...

    def get_similar_for_display(
        self,
        id: str,
        *,
        limit: int = 3,
    ) -> list[Item]: ...

    def list_tags(
        self,
        key: Optional[str] = None,
    ) -> list[str]: ...

    def resolve_meta(
        self,
        item_id: str,
        *,
        limit_per_doc: int = 3,
    ) -> dict[str, list[Item]]: ...

    def resolve_inline_meta(
        self,
        item_id: str,
        queries: list[dict[str, str]],
        context_keys: list[str] | None = None,
        prereq_keys: list[str] | None = None,
        *,
        limit: int = 3,
    ) -> list[Item]: ...

    def list_items(
        self,
        *,
        prefix: Optional[str] = None,
        tags: Optional[TagMap] = None,
        tag_keys: Optional[list[str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        order_by: str = "updated",
        include_hidden: bool = False,
        include_history: bool = False,
        limit: int = 10,
    ) -> list[Item]: ...

    # -- Display context --

    def get_context(
        self,
        id: str,
        *,
        version: int | None = None,
        similar_limit: int = 3,
        meta_limit: int = 3,
        include_similar: bool = True,
        include_meta: bool = True,
        include_parts: bool = True,
        include_versions: bool = True,
    ) -> ItemContext | None: ...

    # -- Direct access --

    def get(self, id: str) -> Optional[Item]: ...

    def get_now(self, *, scope: Optional[str] = None) -> Item: ...

    def get_version(
        self,
        id: str,
        offset: int = 0,
    ) -> Optional[Item]: ...

    def list_versions(
        self,
        id: str,
        limit: int = 10,
    ) -> list[VersionInfo]: ...

    def list_versions_around(
        self,
        id: str,
        version: int,
        radius: int = 2,
    ) -> list[VersionInfo]: ...

    def get_version_nav(
        self,
        id: str,
        current_version: Optional[int] = None,
        limit: int = 3,
    ) -> dict: ...

    def get_version_offset(self, item: Item) -> int: ...

    def exists(self, id: str) -> bool: ...

    # -- Collection management --

    def list_collections(self) -> list[str]: ...

    def count(self) -> int: ...

    # -- Flows --

    def run_flow_command(
        self,
        state: str,
        *,
        params: dict[str, Any] | None = None,
        budget: int | None = None,
        cursor_token: str | None = None,
        state_doc_yaml: str | None = None,
        writable: bool = True,
    ) -> Any:
        """Run a state-doc flow synchronously.

        Returns a FlowResult with status, bindings, cursor (if stopped).
        """
        ...

    # -- Data export / import --

    def export_iter(self, *, include_system: bool = True) -> Iterator[dict]:
        """Stream-export documents.

        Yields header dict first, then one self-contained dict per document
        (versions and parts inline).
        """
        ...

    def export_data(self, *, include_system: bool = True) -> dict: ...

    def export_bundle(
        self,
        id: str,
        *,
        include_system: bool = True,
        include_parts: bool = True,
        include_versions: bool = True,
    ) -> dict | None: ...

    def export_changes(
        self,
        *,
        cursor: str | None = None,
        limit: int = 1000,
    ) -> dict: ...

    def import_data(self, data: dict, *, mode: str = "merge") -> dict: ...

    def import_markdown(
        self,
        source: str,
        *,
        mode: str = "merge",
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Storage backend protocols — internal to Keeper
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Abstract vector search backend.

    Provides embedding storage, similarity search, and metadata queries.
    """

    # -- Embedding dimension --

    @property
    def embedding_dimension(self) -> Optional[int]: ...

    def reset_embedding_dimension(self, dimension: int) -> None: ...

    # -- Single-item operations --

    def upsert(
        self,
        collection: str,
        id: str,
        embedding: list[float],
        summary: str,
        tags: dict[str, str],
    ) -> None: ...

    def upsert_version(
        self,
        collection: str,
        id: str,
        version: int,
        embedding: list[float],
        summary: str,
        tags: dict[str, str],
    ) -> None: ...

    def upsert_part(
        self,
        collection: str,
        id: str,
        part_num: int,
        embedding: list[float],
        summary: str,
        tags: dict[str, str],
    ) -> None: ...

    def delete_parts(self, collection: str, id: str) -> int: ...

    def get(self, collection: str, id: str) -> Optional[StoreResult]: ...

    def get_embedding(self, collection: str, id: str) -> Optional[list[float]]: ...

    def exists(self, collection: str, id: str) -> bool: ...

    def delete(
        self, collection: str, id: str, delete_versions: bool = True
    ) -> bool: ...

    def update_summary(self, collection: str, id: str, summary: str) -> bool: ...

    def update_tags(
        self, collection: str, id: str, tags: dict[str, str]
    ) -> bool: ...

    # -- Batch operations --

    def get_entries_full(
        self, collection: str, ids: list[str]
    ) -> list[dict[str, Any]]: ...

    def upsert_batch(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        summaries: list[str],
        tags: list[dict[str, str]],
    ) -> None: ...

    def delete_entries(self, collection: str, ids: list[str]) -> None: ...

    # -- Search --

    def query_embedding(
        self,
        collection: str,
        embedding: list[float],
        limit: int = 10,
        where: Optional[dict[str, Any]] = None,
    ) -> list[StoreResult]: ...

    def query_metadata(
        self,
        collection: str,
        where: dict[str, Any],
        limit: int = 100,
        offset: int = 0,
    ) -> list[StoreResult]: ...

    # -- Collection management --

    def list_ids(self, collection: str) -> list[str]: ...

    def find_missing_ids(
        self, collection: str, ids: list[str]
    ) -> set[str]: ...

    def list_collections(self) -> list[str]: ...

    def delete_collection(self, name: str) -> bool: ...

    def count(self, collection: str) -> int: ...

    def close(self) -> None: ...


@runtime_checkable
class DocumentStoreProtocol(Protocol):
    """Abstract document metadata backend.

    Provides document storage, versioning, and tag-based queries.
    """

    # -- Write --

    def upsert(
        self,
        collection: str,
        id: str,
        summary: str,
        tags: dict[str, str],
        content_hash: Optional[str] = None,
        content_hash_full: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> tuple[DocumentRecord, bool]: ...

    def update_summary(
        self, collection: str, id: str, summary: str
    ) -> bool: ...

    def update_tags(
        self, collection: str, id: str, tags: dict[str, str]
    ) -> bool: ...

    def touch(self, collection: str, id: str) -> None: ...

    def touch_many(self, collection: str, ids: list[str]) -> None: ...

    def delete(
        self, collection: str, id: str, delete_versions: bool = True
    ) -> bool: ...

    def restore_latest_version(
        self, collection: str, id: str
    ) -> Optional[DocumentRecord]: ...

    def copy_record(
        self, collection: str, from_id: str, to_id: str
    ) -> Optional[DocumentRecord]: ...

    def count_versions_from(
        self, collection: str, id: str, from_version: int
    ) -> int: ...

    # -- Parts --

    def upsert_parts(
        self,
        collection: str,
        id: str,
        parts: list[PartInfo],
    ) -> int: ...

    def get_part(
        self,
        collection: str,
        id: str,
        part_num: int,
    ) -> Optional[PartInfo]: ...

    def list_parts(
        self, collection: str, id: str
    ) -> list[PartInfo]: ...

    def list_parts_many(
        self, collection: str, ids: list[str]
    ) -> dict[str, list[PartInfo]]: ...

    def part_count(self, collection: str, id: str) -> int: ...

    def delete_parts(self, collection: str, id: str) -> int: ...

    def update_part_tags(
        self, collection: str, id: str, part_num: int,
        tags: dict[str, str],
    ) -> bool: ...

    # -- Read --

    def get(self, collection: str, id: str) -> Optional[DocumentRecord]: ...

    def get_many(
        self, collection: str, ids: list[str]
    ) -> dict[str, DocumentRecord]: ...

    def exists(self, collection: str, id: str) -> bool: ...

    def find_by_content_hash(
        self, collection: str, content_hash: str, *,
        content_hash_full: str = "", exclude_id: str = "",
    ) -> Optional[DocumentRecord]: ...

    def get_version(
        self, collection: str, id: str, offset: int = 0
    ) -> Optional[VersionInfo]: ...

    def get_version_by_number(
        self, collection: str, id: str, version: int
    ) -> Optional[VersionInfo]: ...

    def list_versions(
        self, collection: str, id: str, limit: int = 10
    ) -> list[VersionInfo]: ...

    def list_versions_many(
        self, collection: str, ids: list[str]
    ) -> dict[str, list[VersionInfo]]: ...

    def list_versions_around(
        self, collection: str, id: str, version: int, radius: int = 2,
    ) -> list[VersionInfo]: ...

    def get_version_nav(
        self, collection: str, id: str,
        current_version: Optional[int] = None,
        limit: int = 3,
    ) -> dict: ...

    def version_count(self, collection: str, id: str) -> int: ...

    def max_version(self, collection: str, id: str) -> int: ...

    def count_versions(self, collection: str) -> int: ...

    # -- Query --

    def list_ids(
        self, collection: str, limit: Optional[int] = None
    ) -> list[str]: ...

    def list_recent(
        self,
        collection: str,
        limit: int = 10,
        order_by: str = "updated",
        offset: int = 0,
    ) -> list[DocumentRecord]: ...

    def list_recent_with_history(
        self, collection: str, limit: int = 10,
        order_by: str = "updated",
        offset: int = 0,
    ) -> list[DocumentRecord]: ...

    def count(self, collection: str) -> int: ...

    def count_all(self) -> int: ...

    def query_by_id_prefix(
        self, collection: str, prefix: str, limit: int = 0,
        offset: int = 0,
    ) -> list[DocumentRecord]: ...

    def query_by_id_glob(
        self, collection: str, pattern: str, limit: int = 0,
        offset: int = 0,
    ) -> list[DocumentRecord]: ...

    def list_distinct_tag_keys(self, collection: str) -> list[str]: ...

    def list_distinct_tag_values(
        self, collection: str, key: str
    ) -> list[str]: ...

    def query_by_tag_key(
        self, collection: str, key: str,
        limit: int = 100, since_date: Optional[str] = None,
        until_date: Optional[str] = None,
        offset: int = 0,
    ) -> list[DocumentRecord]: ...

    # -- Version extraction (for move) --

    def extract_versions(
        self,
        collection: str,
        source_id: str,
        target_id: str,
        tag_filter: Optional[dict[str, str]] = None,
        only_current: bool = False,
    ) -> tuple[list[VersionInfo], Optional[DocumentRecord], int]: ...

    # -- Collection management --

    def list_collections(self) -> list[str]: ...

    def delete_collection(self, collection: str) -> int: ...

    # -- Bulk import --

    def delete_collection_all(self, collection: str) -> int: ...

    def import_batch(
        self, collection: str, documents: list[dict]
    ) -> dict: ...

    def close(self) -> None: ...


@runtime_checkable
class PendingQueueProtocol(Protocol):
    """Abstract pending work queue.

    Manages background tasks: summarization, embedding, and analysis.
    """

    def enqueue(
        self, id: str, collection: str, content: str,
        *, task_type: str = "summarize", metadata: Optional[dict] = None,
    ) -> None: ...

    def dequeue(self, limit: int = 10) -> list[PendingSummary]: ...

    def complete(self, id: str, collection: str, task_type: str = "summarize") -> None: ...

    def fail(
        self, id: str, collection: str, task_type: str = "summarize",
        error: Optional[str] = None,
    ) -> None: ...

    def abandon(
        self, id: str, collection: str, task_type: str = "summarize",
        error: Optional[str] = None,
    ) -> None: ...

    def count(self) -> int: ...

    def stats(self) -> dict: ...

    def stats_by_type(self) -> dict[str, int]: ...

    def list_failed(self) -> list[dict]: ...

    def retry_failed(self) -> int: ...

    def clear(self) -> int: ...

    def get_status(self, id: str) -> dict | None: ...

    def close(self) -> None: ...


@runtime_checkable
class WorkQueueProtocol(Protocol):
    """Abstract work queue for background task processing.

    Manages prioritized, lease-based work items with supersede semantics
    and exponential backoff on failure.
    """

    def enqueue(
        self, kind: str, input_data: dict[str, Any],
        *, supersede_key: Optional[str] = None, priority: int = 5,
    ) -> str: ...

    def claim(
        self, worker_id: str, *, limit: int = 10, lease_seconds: int = 120,
    ) -> list: ...

    def complete(self, work_id: str, result: dict[str, Any] | None = None) -> None: ...

    def fail(
        self, work_id: str, worker_id: str, error: Optional[str] = None,
        *, backoff_base: int = 30, backoff_max: int = 3600,
    ) -> None: ...

    def has_superseding(self, work_id: str, supersede_key: str, created_at: str) -> bool: ...

    def cancel_by_item_ids(self, item_ids: set[str]) -> int: ...

    def count(self, *, claimable_only: bool = False) -> int: ...

    def count_by_kind(self) -> dict[str, int]: ...

    def list_pending(self, limit: int = 50) -> list[dict]: ...

    def release_stale_leases(self, current_worker: str) -> int: ...

    def purge(self) -> int: ...

    def prune(self, keep_hours: int = 24) -> int: ...

    def enable_auto_vacuum(self) -> bool: ...

    def close(self) -> None: ...
