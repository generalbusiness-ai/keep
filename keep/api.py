"""Core API for reflective memory.

This is the minimal working implementation focused on:
- put(): fetch/embed → summarize → store
- find(): embed query → search
- get(): retrieve by ID
"""

import difflib
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone, timedelta
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)
_MAX_EXPORT_CHANGES_LIMIT = 10_000

from .utils import (
    _parse_date_param,
    _filter_by_date,
    _enrich_updated_date,
    _is_hidden,
    _record_to_item,
    _extract_markdown_frontmatter,
    _parse_meta_doc,
    _get_env_tags,
    _user_tags_changed,
    _merge_tags_additive,
    _split_tag_additions,
    _normalize_remove_keys,
    _apply_tag_mutations,
    _text_content_id,
    _MARKDOWN_EXTENSIONS,
)

from .analyzers import (
    TagClassifier,
    _estimate_tokens,
    _extract_line_ranges,
    _find_best_passage,
    _parse_parts,
    extract_prompt_section,
)
from .backend import NullPendingQueue, create_stores
from .config import load_or_create_config, save_config, StoreConfig, EmbeddingIdentity
from .console_support import expand_prompt
from .markdown_import import load_markdown_import
from .markdown_sync import (
    DOC_STRUCTURAL_MUTATIONS,
    DOC_UPDATE_MUTATION,
    EDGE_MUTATIONS,
    PART_MUTATIONS,
    VERSION_EDGE_MUTATIONS,
    VERSION_MUTATIONS,
    decode_sync_event_payload,
)
from .body_policy import (
    BODY_AUTHORITY_DERIVED,
    BODY_AUTHORITY_MARKDOWN,
    BODY_AUTHORITY_TAG,
    BodyWriteIntent,
    body_write_allowed,
    normalize_body_authority,
    resolve_body_authority,
)
from .paths import get_config_dir, get_default_store_path
from .planner_stats import PlannerStatsStore, build_scope_key
from .provider_identity import provider_model_name
from .protocol import DocumentStoreProtocol, VectorStoreProtocol, PendingQueueProtocol
from .providers import get_registry
from .providers.base import (
    AnalysisChunk,
    Document,
    DocumentProvider,
    EmbedTask,
    EmbeddingProvider,
    MediaDescriber,
    SummarizationProvider,
)
from .providers.documents import CompositeDocumentProvider, FileDocumentProvider
from .providers.embedding_cache import CachingEmbeddingProvider
from .recovery import is_malformed_db_error
from .document_store import DocumentStore, PartInfo, VersionInfo
from .types import (
    Item, ItemContext, EdgeRef, TagMap,
    build_item_context, eval_when_predicate,
    casefold_tags, casefold_tags_for_index, filter_non_system_tags,
    iter_tag_pairs, note_display_name, normalize_edge_value, set_tag_values, tag_values, parse_ref, format_ref,
    SYSTEM_TAG_PREFIX, local_date, utc_now,
    parse_utc_timestamp, validate_tag_key, validate_id, normalize_id, is_part_id,
    is_system_id, file_uri_to_path,
    MAX_TAG_VALUE_LENGTH,
    repair_surrogate_text,
)
from .context_cache import ContextCache
from .const import STATE_PROMPT
from .flow_client import (
    delete_item as flow_delete_item,
    find_items as flow_find_items,
    get_item as flow_get_item,
    get_now_item as flow_get_now_item,
    move_item as flow_move_item,
    put_item as flow_put_item,
    set_now_item as flow_set_now_item,
    stub_item as flow_stub_item,
    tag_item as flow_tag_item,
)
from .flow_env import LocalFlowEnvironment
from .ignore import match_ignore, parse_ignore_patterns, uri_pattern_prefixes
from .logging_config import configure_ops_log
from .markdown_mirrors import path_inside_markdown_mirror
from .tracing import get_tracer as _get_tracer
from .tracing import init_tracing as _init_tracing
from ._background_processing import BackgroundProcessingMixin
from ._provider_lifecycle import ProviderLifecycleMixin
from ._search_augmentation import SearchAugmentationMixin
from ._context_resolution import ContextResolutionMixin


class FindResults(list):
    """List of Items with optional deep-follow groups.

    Subclasses list for backward compatibility.  When ``deep=True`` is
    used, ``deep_groups`` maps each primary item ID to the bridge items
    discovered via its tags.
    """

    def __init__(self, items, deep_groups=None):
        super().__init__(items)
        self.deep_groups: dict[str, list[Item]] = deep_groups or {}


@dataclass
class _PutContext:
    """Explicit body-resolution inputs for one put() mutation."""

    id: str
    content: str
    summary: str | None
    tags: dict[str, Any]
    existing_doc: Any
    restored_version: Any
    content_unchanged: bool
    tags_changed: bool
    new_hash: str


@dataclass(frozen=True)
class _ResolvedBodySummary:
    """Resolved body text plus the authority that selected it."""

    summary: str
    body_authority: str


# Default max length for truncated placeholder summaries
TRUNCATE_LENGTH = 500

# Fixed ID for the current working context (singleton)
NOWDOC_ID = "now"


from .system_docs import (
    SYSTEM_DOC_DIR,
    SYSTEM_DOC_IDS,
    _load_frontmatter,
    migrate_system_documents,
    reset_system_documents,
    system_doc_migration_needed,
)

from .dependencies import NoteDependencyService
from .markdown_export import _get_export_bundle
from .processors import _content_hash, _content_hash_full, process_analyze
from .state_doc import parse_state_doc
from .state_doc_runtime import (
    FlowResult,
    decode_cursor,
    make_action_runner,
    make_state_doc_loader,
    run_flow,
)
from .task_client import TaskClient
from .task_workflows import _apply_mutations


# -------------------------------------------------------------------------
# Decomposition helpers (module-level, used by Keeper.analyze)
# -------------------------------------------------------------------------

class Keeper(ProviderLifecycleMixin, BackgroundProcessingMixin, SearchAugmentationMixin, ContextResolutionMixin):
    """Reflective memory keeper - persistent storage with similarity search.

    Example:
        kp = Keeper()
        kp.put(uri="file:///path/to/readme.md")
        results = kp.find("installation instructions")
    """
    
    def __init__(
        self,
        store_path: Optional[str | Path] = None,
        decay_half_life_days: float = 30.0,
        *,
        config: Optional[StoreConfig] = None,
        doc_store: Optional["DocumentStoreProtocol"] = None,
        vector_store: Optional["VectorStoreProtocol"] = None,
        pending_queue: Optional["PendingQueueProtocol"] = None,
        defer_startup_maintenance: bool = False,
    ) -> None:
        """Initialize or open an existing reflective memory store.

        Args:
            store_path: Path to store directory. Uses default if not specified.
                       Overrides any store.path setting in config.
            decay_half_life_days: Memory decay half-life in days (ACT-R model).
                After this many days, an item's effective relevance is halved.
                Set to 0 or negative to disable decay.
            config: Pre-loaded StoreConfig (skips filesystem config discovery).
            doc_store: Injected document store (skips default backend creation).
            vector_store: Injected vector store (skips default backend creation).
            pending_queue: Injected summary queue (skips default backend creation).
            defer_startup_maintenance: If True, skip expensive startup scans and
                background-reconcile checks during initialization. Call
                ``start_deferred_startup_maintenance()`` after the daemon is
                ready to serve requests.
        """
        self._decay_half_life_days = decay_half_life_days
        _init_tracing()

        # --- Config resolution ---
        if config is not None:
            # Injected config — skip filesystem discovery
            self._config: StoreConfig = config
            self._store_path = config.path if config.path else Path(".")
        else:
            # Resolve config and store paths from filesystem.
            # KEEP_CONFIG takes priority for config dir; otherwise
            # store_path doubles as config dir (backwards compat).
            explicit_config = os.environ.get("KEEP_CONFIG")
            if store_path is not None:
                self._store_path = Path(store_path).resolve()
                config_dir = Path(explicit_config).expanduser().resolve() if explicit_config else self._store_path
            else:
                config_dir = get_config_dir()

            self._config = load_or_create_config(config_dir)

            if store_path is None:
                self._store_path = get_default_store_path(self._config)

        # --- Document provider ---
        registry = get_registry()
        self._document_provider: DocumentProvider = registry.create_document(
            self._config.document.name,
            self._config.document.params,
        )
        self._apply_file_size_limit(self._document_provider)

        # Lazy-loaded providers (created on first use to avoid network access for read-only ops)
        self._embedding_provider: Optional[EmbeddingProvider] = None
        self._summarization_provider: Optional[SummarizationProvider] = None
        self._media_describer: Optional[MediaDescriber] = None
        self._content_extractor = None  # ContentExtractor, lazy-loaded
        self._analyzer = None  # AnalyzerProvider, lazy-loaded

        # Cache and validate config/env tags once per Keeper instance.
        self._default_tags = self._validate_tag_map(
            self._config.default_tags,
            source="Config default tags",
            check_constraints=False,
        )
        self._env_tags = self._validate_tag_map(
            _get_env_tags(),
            source="KEEP_TAG_* environment tags",
            check_constraints=False,
        )

        # Ensure the store directory exists before any file I/O against it.
        # Previously this was a side-effect of save_config() mkdir'ing the
        # config dir when config_dir == store_path; making it explicit avoids
        # breaking when KEEP_CONFIG points elsewhere (e.g. test isolation).
        self._store_path.mkdir(parents=True, exist_ok=True)

        # --- Persistent operations log ---
        self._ops_log_handler = configure_ops_log(self._store_path)

        # --- Storage backends (injected or factory-created) ---
        if doc_store is not None and vector_store is not None:
            # Fully injected (tests, custom setups)
            self._document_store = doc_store
            self._store = vector_store
            self._pending_queue = pending_queue or NullPendingQueue()
            self._is_local = False
        else:
            # Factory-based creation from config
            bundle = create_stores(self._config)
            self._document_store = bundle.doc_store
            self._store = bundle.vector_store
            self._pending_queue = bundle.pending_queue
            self._is_local = bundle.is_local
            if getattr(bundle, 'work_queue', None) is not None:
                self._work_queue = bundle.work_queue

        # Guard against concurrent background reconciliation
        self._reconcile_lock = threading.Lock()
        self._reconcile_done = threading.Event()
        self._closing = threading.Event()  # signals reconcile to abort
        self._closed = False
        self._provider_init_lock = threading.RLock()
        self._startup_maintenance_deferred = bool(defer_startup_maintenance)
        self._startup_maintenance_started = threading.Event()
        self._startup_maintenance_thread: Optional[threading.Thread] = None
        self._last_spawn_time: float = 0.0
        self._tagdoc_cache: dict[str, Optional[dict[str, str]]] = {}
        self._cel_cache: dict[str, Any] = {}  # compiled CEL programs keyed by source string
        self._ignore_patterns: Optional[list[str]] = None
        self._ignore_patterns_ts: float = 0.0
        self._context_cache = ContextCache()
        # Some startup maintenance paths can route through flow-backed writes
        # and therefore call ensure_sysdocs(). Initialize this sentinel before
        # any migration step that might create stubs or otherwise re-enter the
        # public state-doc surface.
        self._needs_sysdoc_migration = system_doc_migration_needed(self)

        # If cosine migration fired (L2→cosine), auto-enqueue reindex
        if getattr(self._store, "migrated_to_cosine", False):
            logger.info("Cosine migration detected — enqueuing reindex")
            try:
                stats = self.enqueue_reindex()
                print(
                    f"Search index migrated to cosine similarity.\n"
                    f"Search is unavailable until reindex completes.\n"
                    f"Run: keep daemon",
                    file=sys.stderr,
                )
            except Exception as e:
                logger.error("Failed to enqueue reindex after cosine migration: %s", e)
                print(
                    f"ERROR: Search index migration failed. Search may not work.\n"
                    f"Try: keep daemon --reindex\n"
                    f"Details: {e}",
                    file=sys.stderr,
                )
        chroma_coll = self._resolve_chroma_collection()
        doc_coll = self._resolve_doc_collection()
        if not self._startup_maintenance_deferred:
            self._run_labeled_ref_format_migration(doc_coll)

        if not self._startup_maintenance_deferred:
            self._enqueue_migrated_part_reindex()
        self._run_legacy_overview_cleanup(chroma_coll, doc_coll)

        # Check store consistency and reconcile in background if needed
        # (safe for all backends — uses abstract store interface)
        needs_reconcile = False
        if not self._startup_maintenance_deferred:
            needs_reconcile = (
                self._check_store_consistency() and self._config.embedding is not None
            )
            if getattr(self._store, "migrated_to_cosine", False):
                needs_reconcile = False  # reindex will handle everything

        # Legacy metadata migration: old key=value Chroma metadata does not
        # satisfy marker-based tag filters. Rewrite metadata in-place.
        #
        # The detection scan is O(number of indexed rows), so persist a
        # per-store "verified" flag after a successful check/migration.
        if not self._startup_maintenance_deferred:
            self._run_tag_marker_startup_check(chroma_coll, doc_coll)

        # Embed-task-type migration: re-embed all items with correct
        # document/query task prefixes (nomic search_document:, Voyage
        # input_type, Gemini task_type).  Runs once, low-priority background.
        if (
            not self._startup_maintenance_deferred
            and self._config.embedding is not None
            and not self._config.embed_task_reindex_done
        ):
            self._enqueue_embed_task_reindex()

        if needs_reconcile:
            self._reconcile_thread = threading.Thread(
                target=self._auto_reconcile_safe, args=(chroma_coll, doc_coll), daemon=True,
            )
            self._reconcile_thread.start()
        else:
            self._reconcile_thread = None
            self._reconcile_done.set()

        # --- Task delegation client (for hosted processing) ---
        self._task_client = None
        if self._config.remote:
            try:
                self._task_client = TaskClient(
                    self._config.remote.api_url,
                    self._config.remote.api_key,
                    project=self._config.remote.project,
                )
            except Exception as e:
                logger.warning("Failed to initialize TaskClient: %s", e)

        # --- Planner stats (precomputed priors for flow discriminators) ---
        self._planner_stats = None
        if self._is_local:
            try:
                self._planner_stats = PlannerStatsStore(
                    self._store_path / "planner_stats.db"
                )
                # Bootstrap: enqueue rebuild task if stats have never been built.
                # The daemon will pick this up on its next tick.
                doc_coll = self._resolve_doc_collection()
                if self._planner_stats.needs_rebuild(doc_coll):
                    self._pending_queue.enqueue(
                        ".planner-rebuild", doc_coll, "",
                        task_type="planner-rebuild",
                    )
            except Exception as e:
                logger.warning("Failed to initialize PlannerStatsStore: %s", e)

        # Direct work queue (replaces FlowEngine for background tasks).
        self._work_queue = None
        self._work_queue_lock = threading.Lock()
        self._write_context_by_id: dict[str, dict[str, Any]] = {}
        self._write_context_lock = threading.Lock()

    def _apply_file_size_limit(self, provider: DocumentProvider) -> None:
        """Apply max_file_size config to file-based providers."""
        max_size = self._config.max_file_size
        if isinstance(provider, FileDocumentProvider):
            provider.max_size = max_size
        elif isinstance(provider, CompositeDocumentProvider):
            for p in provider._providers:
                if isinstance(p, FileDocumentProvider):
                    p.max_size = max_size

    def _check_store_consistency(
        self,
        *,
        _doc_store: Optional["DocumentStoreProtocol"] = None,
    ) -> bool:
        """Check if document store and vector store ID sets match.

        Returns True if reconciliation is needed. Does not fix —
        that is deferred to the first _upsert call when the
        embedding provider is available.
        """
        try:
            result = self.reconcile(fix=False, _doc_store=_doc_store)
            if result["missing_from_index"] or result["orphaned_in_index"]:
                logger.info(
                    "Store inconsistency: %d missing from search index, %d orphaned (will auto-reconcile)",
                    result["missing_from_index"], result["orphaned_in_index"],
                )
                return True
        except Exception as e:
            logger.debug("Store consistency check failed: %s", e)
        return False

    def _mark_chroma_tag_markers_verified(self) -> None:
        """Persist that this store no longer needs startup marker scans."""
        if self._config.chroma_tag_markers_verified:
            return
        self._config.chroma_tag_markers_verified = True
        try:
            save_config(self._config)
        except Exception as e:
            logger.debug("Failed to persist chroma_tag_markers_verified: %s", e)

    def _mark_legacy_overview_parts_cleaned(self) -> None:
        """Persist that legacy @P{0} overview cleanup has run for this store."""
        if self._config.legacy_overview_parts_cleaned:
            return
        self._config.legacy_overview_parts_cleaned = True
        try:
            save_config(self._config)
        except Exception as e:
            logger.debug("Failed to persist legacy_overview_parts_cleaned: %s", e)

    def _mark_labeled_ref_format_verified(self) -> None:
        """Persist that labeled refs are canonicalized in this store."""
        if self._config.labeled_ref_format_verified:
            return
        self._config.labeled_ref_format_verified = True
        try:
            save_config(self._config)
        except Exception as e:
            logger.debug("Failed to persist labeled_ref_format_verified: %s", e)

    def _run_legacy_overview_cleanup(self, chroma_coll: str, doc_coll: str) -> int:
        """Run @P{0} overview cleanup at most once per store."""
        if self._config.legacy_overview_parts_cleaned:
            return 0
        deleted = self._cleanup_legacy_overview_parts(chroma_coll, doc_coll)
        self._mark_legacy_overview_parts_cleaned()
        return deleted

    def _run_labeled_ref_format_migration(self, doc_coll: str) -> dict[str, int]:
        """Canonicalize stored labeled refs once per store."""
        if self._config.labeled_ref_format_verified:
            return {"documents": 0, "versions": 0, "parts": 0}
        if not self._document_store.list_ids(doc_coll, limit=1):
            self._mark_labeled_ref_format_verified()
            return {"documents": 0, "versions": 0, "parts": 0}

        try:
            logger.info(
                "Canonicalizing labeled refs to MediaWiki-style [[target|label]]"
            )
            stats = self._migrate_labeled_ref_format(doc_coll)
            self._mark_labeled_ref_format_verified()
            logger.info(
                "Labeled-ref canonicalization complete (%d docs, %d versions, %d parts)",
                stats["documents"], stats["versions"], stats["parts"],
            )
            return stats
        except Exception as e:
            logger.warning(
                "Labeled-ref canonicalization failed; exports may contain mixed ref "
                "syntax until it succeeds: %s",
                e,
            )
            return {"documents": 0, "versions": 0, "parts": 0}

    def _merge_auto_vivified_target_names(
        self,
        doc_coll: str,
        target_name_updates: dict[str, list[str]],
    ) -> None:
        """Merge observed labels into auto-vivified target notes only."""
        if not target_name_updates:
            return
        chroma_coll = self._resolve_chroma_collection()
        targets = self._document_store.get_many(doc_coll, list(target_name_updates))
        for target_id, incoming_names in target_name_updates.items():
            target_doc = targets.get(target_id)
            if target_doc is None:
                continue
            if target_doc.tags.get("_source") != "auto-vivify":
                continue
            merged_name_values = tag_values(target_doc.tags, "name")
            changed = False
            for name in incoming_names:
                if name not in merged_name_values:
                    merged_name_values.append(name)
                    changed = True
            if not changed:
                continue
            updated_tags = dict(target_doc.tags)
            set_tag_values(updated_tags, "name", merged_name_values)
            self._document_store.update_tags(doc_coll, target_id, updated_tags)
            self._store.update_tags(
                chroma_coll, target_id, casefold_tags_for_index(updated_tags)
            )

    def _restore_current_edges_without_backfill(
        self,
        id: str,
        merged_tags: dict[str, str],
        doc_coll: str,
        *,
        summary: str = "",
    ) -> None:
        """Upsert current edge rows and target labels without delete/backfill churn."""
        if id.startswith("."):
            return

        edges_to_add: list[tuple[str, str, str, str, str]] = []
        target_name_updates: dict[str, list[str]] = {}

        def _queue_target_name(target_id: str, label: str | None) -> None:
            label = (label or "").strip()
            if not label:
                return
            names = target_name_updates.setdefault(target_id, [])
            if label not in names:
                names.append(label)

        for key in merged_tags:
            if key.startswith(SYSTEM_TAG_PREFIX):
                continue
            if key not in self._tagdoc_cache:
                parent = self._document_store.get(doc_coll, f".tag/{key}")
                self._tagdoc_cache[key] = parent.tags if parent else None
            td_tags = self._tagdoc_cache[key]
            if td_tags is None or not td_tags.get("_inverse"):
                continue
            inverse = td_tags.get("_inverse")
            if isinstance(inverse, list):
                inverse = inverse[0]
            if not inverse:
                continue

            # Evaluate _when condition against source note
            when_source = td_tags.get("_when", "")
            if when_source:
                item_ctx = build_item_context(
                    id=id,
                    tags=merged_tags,
                    summary=summary,
                    content_type=merged_tags.get("_content_type", ""),
                    uri=merged_tags.get("_source_uri", ""),
                )
                if not eval_when_predicate(when_source, item_ctx, cache=self._cel_cache):
                    continue  # condition not met — skip this edge tag

            for current_value in tag_values(merged_tags, key):
                if not current_value:
                    continue
                raw_target_id, target_label = parse_ref(current_value)
                try:
                    target_id = normalize_id(raw_target_id)
                except ValueError:
                    logger.debug(
                        "Skipping invalid edge target for tag %r during migration: %r",
                        key, current_value,
                    )
                    continue
                if target_id.startswith("."):
                    continue
                _queue_target_name(target_id, target_label)

                reference_created = (
                    merged_tags.get("_created")
                    or merged_tags.get("_updated")
                    or utc_now()
                )
                now = utc_now()
                target_tags = {
                    "_created": reference_created,
                    "_updated": now,
                    "_source": "auto-vivify",
                }
                if target_id in target_name_updates:
                    set_tag_values(target_tags, "name", target_name_updates[target_id])
                self._stub_via_flow(
                    id=target_id,
                    content="",
                    summary="",
                    tags=target_tags,
                    created_at=reference_created,
                    queue_background_tasks=True,
                )

                created = (
                    merged_tags.get("_created")
                    or merged_tags.get("_updated")
                    or utc_now()
                )
                edges_to_add.append((id, key, target_id, inverse, created))

        self._merge_auto_vivified_target_names(doc_coll, target_name_updates)
        if edges_to_add:
            self._document_store.upsert_edges_batch(doc_coll, edges_to_add)

    def _rewrite_index_tags_without_timestamp(
        self,
        chroma_coll: str,
        id: str,
        tags: dict[str, Any],
    ) -> bool:
        """Rewrite vector-store metadata tags without mutating timestamps."""
        rewrite_tags = getattr(self._store, "rewrite_tags", None)
        if callable(rewrite_tags):
            return bool(rewrite_tags(chroma_coll, id, tags))
        return bool(self._store.update_tags(chroma_coll, id, tags))

    def _migrate_labeled_ref_format(self, doc_coll: str) -> dict[str, int]:
        """Rewrite stored edge-tag refs into canonical syntax in-place."""
        changed_docs = 0
        changed_versions = 0
        changed_parts = 0
        touched_version_sources: set[str] = set()
        chroma_coll = self._resolve_chroma_collection()

        for doc_id in self._document_store.list_ids(doc_coll):
            record = self._document_store.get(doc_coll, doc_id)
            if record is None:
                continue

            current_tags = dict(record.tags)
            migrated_tags = dict(record.tags)
            self._normalize_edge_tag_values(migrated_tags, doc_coll)
            if migrated_tags != current_tags:
                self._document_store.patch_head_tags(doc_coll, doc_id, migrated_tags)
                self._rewrite_index_tags_without_timestamp(
                    chroma_coll, doc_id, casefold_tags_for_index(migrated_tags),
                )
                self._restore_current_edges_without_backfill(
                    doc_id, migrated_tags, doc_coll,
                    summary=record.summary,
                )
                changed_docs += 1

            for version in self._document_store.list_versions(doc_coll, doc_id, limit=0):
                version_tags = dict(version.tags)
                migrated_version_tags = dict(version.tags)
                self._normalize_edge_tag_values(migrated_version_tags, doc_coll)
                if migrated_version_tags != version_tags:
                    self._document_store.replace_version_content(
                        doc_coll,
                        doc_id,
                        version.version,
                        version.summary,
                        migrated_version_tags,
                        version.content_hash,
                    )
                    indexed_version_tags = dict(migrated_version_tags)
                    indexed_version_tags["_version"] = str(version.version)
                    indexed_version_tags["_base_id"] = doc_id
                    self._rewrite_index_tags_without_timestamp(
                        chroma_coll,
                        f"{doc_id}@v{version.version}",
                        casefold_tags_for_index(indexed_version_tags),
                    )
                    touched_version_sources.add(doc_id)
                    changed_versions += 1

            for part in self._document_store.list_parts(doc_coll, doc_id):
                part_tags = dict(part.tags)
                migrated_part_tags = dict(part.tags)
                self._normalize_edge_tag_values(migrated_part_tags, doc_coll)
                if migrated_part_tags != part_tags:
                    self._document_store.update_part_tags(
                        doc_coll, doc_id, part.part_num, migrated_part_tags,
                    )
                    indexed_part_tags = dict(migrated_part_tags)
                    indexed_part_tags["_part_num"] = str(part.part_num)
                    indexed_part_tags["_base_id"] = doc_id
                    self._rewrite_index_tags_without_timestamp(
                        chroma_coll,
                        f"{doc_id}@p{part.part_num}",
                        casefold_tags_for_index(indexed_part_tags),
                    )
                    changed_parts += 1

        for source_id in touched_version_sources:
            self._document_store.rebuild_version_edges_for_source(doc_coll, source_id)

        return {
            "documents": changed_docs,
            "versions": changed_versions,
            "parts": changed_parts,
        }

    def _filter_tag_specs_by_when(
        self,
        specs: list[dict],
        item_tags: dict[str, Any],
        item_id: str = "",
        item_summary: str = "",
    ) -> list[dict]:
        """Remove tag specs whose ``_when`` condition is not met by the item."""
        result = []
        item_ctx = build_item_context(
            id=item_id,
            tags=item_tags,
            summary=item_summary,
            content_type=item_tags.get("_content_type", ""),
            uri=item_tags.get("_source_uri", ""),
        )
        for spec in specs:
            when_source = spec.get("_when", "")
            if not when_source or eval_when_predicate(
                when_source, item_ctx, cache=self._cel_cache,
            ):
                result.append(spec)
        return result

    # Content-kind values that should live under ``kind`` instead of ``type``.
    # Entity-type values (conversation, paper, vulnerability, …) stay in ``type``.
    _TYPE_TO_KIND_VALUES: frozenset[str] = frozenset({
        "learning", "breakdown", "gotcha", "reference", "teaching",
        "meeting", "pattern", "possibility", "decision",
    })

    def _migrate_type_to_kind(self, doc_coll: str) -> dict[str, int]:
        """Move content-kind values from ``type`` to ``kind`` tag in-place.

        After this migration ``type`` holds only entity-type values
        (conversation, paper, …) and ``kind`` holds content-kind values
        (learning, breakdown, …).  If a note's ``type`` contains both
        entity-type and content-kind values, only the content-kind ones
        are moved; the entity-type values remain.
        """
        changed_docs = 0
        changed_versions = 0
        changed_parts = 0
        chroma_coll = self._resolve_chroma_collection()

        for doc_id in self._document_store.list_ids(doc_coll):
            record = self._document_store.get(doc_coll, doc_id)
            if record is None:
                continue

            # --- head document ---
            migrated_tags = self._retag_type_to_kind(dict(record.tags))
            if migrated_tags is not None:
                self._document_store.update_tags(doc_coll, doc_id, migrated_tags)
                self._rewrite_index_tags_without_timestamp(
                    chroma_coll, doc_id, casefold_tags_for_index(migrated_tags),
                )
                changed_docs += 1

            # --- versions ---
            for version in self._document_store.list_versions(
                doc_coll, doc_id, limit=0,
            ):
                migrated_vtags = self._retag_type_to_kind(dict(version.tags))
                if migrated_vtags is not None:
                    self._document_store.replace_version_content(
                        doc_coll, doc_id, version.version,
                        version.summary, migrated_vtags, version.content_hash,
                    )
                    indexed = dict(migrated_vtags)
                    indexed["_version"] = str(version.version)
                    indexed["_base_id"] = doc_id
                    self._rewrite_index_tags_without_timestamp(
                        chroma_coll,
                        f"{doc_id}@v{version.version}",
                        casefold_tags_for_index(indexed),
                    )
                    changed_versions += 1

            # --- parts ---
            for part in self._document_store.list_parts(doc_coll, doc_id):
                migrated_ptags = self._retag_type_to_kind(dict(part.tags))
                if migrated_ptags is not None:
                    self._document_store.update_part_tags(
                        doc_coll, doc_id, part.part_num, migrated_ptags,
                    )
                    indexed = dict(migrated_ptags)
                    indexed["_part_num"] = str(part.part_num)
                    indexed["_base_id"] = doc_id
                    self._rewrite_index_tags_without_timestamp(
                        chroma_coll,
                        f"{doc_id}@p{part.part_num}",
                        casefold_tags_for_index(indexed),
                    )
                    changed_parts += 1

        return {
            "documents": changed_docs,
            "versions": changed_versions,
            "parts": changed_parts,
        }

    def _retag_type_to_kind(self, tags: dict[str, Any]) -> dict[str, Any] | None:
        """Move content-kind values from ``type`` to ``kind``.

        Returns the mutated *tags* dict if any values were moved, else ``None``
        (no change needed).
        """
        type_vals = tag_values(tags, "type")
        if not type_vals:
            return None

        to_move = [v for v in type_vals if v in self._TYPE_TO_KIND_VALUES]
        if not to_move:
            return None

        # Values that stay in type (entity-type values)
        keep_in_type = [v for v in type_vals if v not in self._TYPE_TO_KIND_VALUES]
        set_tag_values(tags, "type", keep_in_type)

        # Merge into existing kind values (hermes may already have set kind)
        existing_kind = tag_values(tags, "kind")
        merged_kind = list(dict.fromkeys(existing_kind + to_move))
        set_tag_values(tags, "kind", merged_kind)

        return tags

    def _run_type_to_kind_migration(self, doc_coll: str) -> dict[str, int]:
        """Rename content-kind values once per store."""
        if self._config.type_to_kind_migrated:
            return {"documents": 0, "versions": 0, "parts": 0}
        if not self._document_store.list_ids(doc_coll, limit=1):
            self._config.type_to_kind_migrated = True
            try:
                save_config(self._config)
            except Exception as e:
                logger.debug("Failed to persist type_to_kind_migrated: %s", e)
            return {"documents": 0, "versions": 0, "parts": 0}

        try:
            logger.info("Migrating content-kind values from 'type' to 'kind' tag")
            stats = self._migrate_type_to_kind(doc_coll)
            self._config.type_to_kind_migrated = True
            try:
                save_config(self._config)
            except Exception as e:
                logger.debug("Failed to persist type_to_kind_migrated: %s", e)
            logger.info(
                "type→kind migration complete (%d docs, %d versions, %d parts)",
                stats["documents"], stats["versions"], stats["parts"],
            )
            return stats
        except Exception as e:
            logger.warning(
                "type→kind migration failed; content-kind values remain in 'type' "
                "until it succeeds: %s", e,
            )
            return {"documents": 0, "versions": 0, "parts": 0}

    # Provider+model combinations known to be symmetric (task param is a no-op).
    # Key: provider name, Value: set of model prefixes that are symmetric,
    # or None meaning the entire provider family is symmetric.
    _SYMMETRIC_PROVIDERS: dict[str, set[str] | None] = {
        "openai": None,         # all OpenAI models are symmetric
        "mistral": None,        # all Mistral models are symmetric
    }
    # Providers where only specific models are asymmetric.
    _ASYMMETRIC_MODEL_PREFIXES: dict[str, list[str]] = {
        "ollama": ["nomic-embed"],
        "sentence-transformers": ["nomic"],
        "mlx": ["nomic"],
    }

    def _config_uses_embed_task(self) -> bool:
        """Check from config whether the embedding provider uses task types.

        Avoids instantiating the provider (which loads the model).
        Uses provider name and model params to decide statically.
        """
        if self._config.embedding is None:
            return False
        name = self._config.embedding.name
        params = self._config.embedding.params

        # Entirely symmetric provider families
        if name in self._SYMMETRIC_PROVIDERS:
            return False

        # Always asymmetric (Voyage, Gemini, OpenRouter)
        if name in ("voyage", "gemini", "openrouter"):
            return True

        # Model-specific: check if model name matches known asymmetric prefixes
        prefixes = self._ASYMMETRIC_MODEL_PREFIXES.get(name)
        if prefixes is not None:
            model = str(params.get("model", ""))
            return any(model.startswith(p) for p in prefixes)

        # Unknown provider — conservatively skip reindex
        return False

    def _enqueue_embed_task_reindex(self) -> None:
        """One-time migration: re-embed all items with task-type prefixes.

        Only runs for providers whose embed() actually changes behavior
        based on the task parameter.  Symmetric providers and models
        without task-prompt support are skipped.
        """
        if not self._config_uses_embed_task():
            logger.debug("Skipping embed-task reindex: provider is symmetric")
            self._config.embed_task_reindex_done = True
            try:
                save_config(self._config)
            except Exception as exc:
                logger.warning(
                    "Failed to persist embed_task_reindex_done after skipping reindex: %s",
                    exc,
                    exc_info=True,
                )
            return

        stats = self.enqueue_reindex()
        enqueued = stats.get("enqueued", 0)
        versions = stats.get("versions", 0)
        parts = stats.get("parts", 0)
        if enqueued:
            print(
                f"Queued embedding task-type migration for "
                f"{enqueued} items + {versions} versions + {parts} parts "
                f"(background)...",
                file=sys.stderr,
            )
        self._config.embed_task_reindex_done = True
        try:
            save_config(self._config)
        except Exception as exc:
            logger.warning(
                "Failed to persist embed_task_reindex_done after queuing reindex: %s",
                exc,
                exc_info=True,
            )

    def _detect_chroma_tag_marker_migration_need(
        self,
        chroma_coll: str,
        doc_coll: str,
        *,
        _doc_store: Optional["DocumentStoreProtocol"] = None,
    ) -> Optional[bool]:
        """Detect legacy Chroma metadata without multivalue tag markers.

        Returns:
            True: legacy metadata detected, migration needed
            False: scan completed and migration not needed
            None: check failed (caller should avoid persisting "verified")
        """
        has_tag_markers = getattr(self._store, "has_tag_markers", None)
        if not callable(has_tag_markers):
            return False

        def _has_user_tags(tags: dict[str, Any]) -> bool:
            return any(
                not key.startswith(SYSTEM_TAG_PREFIX) and tag_values(tags, key)
                for key in tags
            )

        doc_store = _doc_store or self._document_store
        try:
            indexed_ids = set(self._store.list_ids(chroma_coll))
            if not indexed_ids:
                return False
            for doc_id in doc_store.list_ids(doc_coll):
                doc = doc_store.get(doc_coll, doc_id)
                if doc is not None and doc_id in indexed_ids:
                    if _has_user_tags(doc.tags) and not has_tag_markers(chroma_coll, doc_id):
                        return True

                for vi in doc_store.list_versions(
                    doc_coll, doc_id, limit=1_000_000,
                ):
                    version_id = f"{doc_id}@v{vi.version}"
                    if version_id not in indexed_ids:
                        continue
                    ver_tags = dict(vi.tags)
                    ver_tags["_version"] = str(vi.version)
                    ver_tags["_base_id"] = doc_id
                    if _has_user_tags(ver_tags) and not has_tag_markers(chroma_coll, version_id):
                        return True

                for part in doc_store.list_parts(doc_coll, doc_id):
                    part_id = f"{doc_id}@p{part.part_num}"
                    if part_id not in indexed_ids:
                        continue
                    part_tags = dict(part.tags)
                    part_tags["_part_num"] = str(part.part_num)
                    part_tags["_base_id"] = doc_id
                    if _has_user_tags(part_tags) and not has_tag_markers(chroma_coll, part_id):
                        return True
        except Exception as e:
            logger.debug("Tag marker migration check failed: %s", e)
            return None
        return False

    def _migrate_chroma_tag_markers(
        self,
        chroma_coll: str,
        doc_coll: str,
        *,
        _doc_store: Optional["DocumentStoreProtocol"] = None,
    ) -> dict[str, int]:
        """Rewrite legacy Chroma metadata to marker-based tag encoding."""
        rewrite_tags = getattr(self._store, "rewrite_tags", None)
        if not callable(rewrite_tags):
            return {"docs": 0, "versions": 0, "parts": 0}

        doc_store = _doc_store or self._document_store
        indexed_ids = set(self._store.list_ids(chroma_coll))
        if not indexed_ids:
            return {"docs": 0, "versions": 0, "parts": 0}

        docs = versions = parts = 0
        for doc_id in doc_store.list_ids(doc_coll):
            if doc_id in indexed_ids:
                doc = doc_store.get(doc_coll, doc_id)
                if doc is not None and rewrite_tags(
                    chroma_coll, doc_id, casefold_tags_for_index(doc.tags),
                ):
                    docs += 1

            for vi in doc_store.list_versions(
                doc_coll, doc_id, limit=1_000_000,
            ):
                version_id = f"{doc_id}@v{vi.version}"
                if version_id not in indexed_ids:
                    continue
                ver_tags = dict(vi.tags)
                ver_tags["_version"] = str(vi.version)
                ver_tags["_base_id"] = doc_id
                if rewrite_tags(
                    chroma_coll, version_id, casefold_tags_for_index(ver_tags),
                ):
                    versions += 1

            for part in doc_store.list_parts(doc_coll, doc_id):
                part_id = f"{doc_id}@p{part.part_num}"
                if part_id not in indexed_ids:
                    continue
                part_tags = dict(part.tags)
                part_tags["_part_num"] = str(part.part_num)
                part_tags["_base_id"] = doc_id
                if rewrite_tags(
                    chroma_coll, part_id, casefold_tags_for_index(part_tags),
                ):
                    parts += 1

        return {"docs": docs, "versions": versions, "parts": parts}

    def _run_tag_marker_startup_check(
        self,
        chroma_coll: str,
        doc_coll: str,
        *,
        _doc_store: Optional["DocumentStoreProtocol"] = None,
    ) -> None:
        """Run the legacy tag-marker migration check during startup."""
        def _call_with_optional_doc_store(fn):
            if _doc_store is None:
                return fn(chroma_coll, doc_coll)
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                params = {}
            if "_doc_store" in params:
                return fn(chroma_coll, doc_coll, _doc_store=_doc_store)
            return fn(chroma_coll, doc_coll)

        if self._config.chroma_tag_markers_verified:
            return

        marker_migration_state = _call_with_optional_doc_store(
            self._detect_chroma_tag_marker_migration_need,
        )
        if marker_migration_state is True:
            try:
                print(
                    "Migrating search metadata to multivalue tag markers "
                    "(this may take a while on larger stores)...",
                    file=sys.stderr,
                    flush=True,
                )
                stats = _call_with_optional_doc_store(
                    self._migrate_chroma_tag_markers,
                )
                logger.info(
                    "Tag marker migration complete: %d docs, %d versions, %d parts",
                    stats["docs"], stats["versions"], stats["parts"],
                )
                print(
                    "Search metadata migrated to multivalue tag markers "
                    f"({stats['docs']} docs, {stats['versions']} versions, {stats['parts']} parts).",
                    file=sys.stderr,
                )
                self._mark_chroma_tag_markers_verified()
            except Exception as e:
                logger.warning("Tag marker migration failed: %s", e)
                print(
                    "WARNING: tag metadata migration failed; "
                    "tag-filtered semantic search may be incomplete.\n"
                    "Run: keep daemon --reindex",
                    file=sys.stderr,
                )
        elif marker_migration_state is False:
            self._mark_chroma_tag_markers_verified()

    def _run_deferred_startup_maintenance(self) -> None:
        """Run deferred startup scans after the daemon is already reachable."""
        startup_doc_store = None
        try:
            if self._closing.is_set():
                return

            if self._is_local and hasattr(self._document_store, "_db_path"):
                startup_doc_store = DocumentStore(self._document_store._db_path)

            chroma_coll = self._resolve_chroma_collection()
            doc_coll = self._resolve_doc_collection()
            self._run_labeled_ref_format_migration(doc_coll)
            self._enqueue_migrated_part_reindex()
            self._run_tag_marker_startup_check(
                chroma_coll, doc_coll, _doc_store=startup_doc_store,
            )

            if self._closing.is_set():
                return

            # Move content-kind values from ``type`` to ``kind`` tag
            self._run_type_to_kind_migration(doc_coll)

            if self._closing.is_set():
                return

            # Embed-task-type migration (deferred path)
            if (
                self._config.embedding is not None
                and not self._config.embed_task_reindex_done
            ):
                self._enqueue_embed_task_reindex()

            if self._closing.is_set():
                return

            needs_reconcile = (
                self._check_store_consistency(_doc_store=startup_doc_store)
                and self._config.embedding is not None
            )
            if needs_reconcile:
                self._reconcile_done.clear()
                self._reconcile_thread = threading.Thread(
                    target=self._auto_reconcile_safe,
                    args=(chroma_coll, doc_coll),
                    daemon=True,
                    name="startup-reconcile",
                )
                self._reconcile_thread.start()
            else:
                self._reconcile_done.set()
        except Exception as e:
            logger.warning("Deferred startup maintenance failed: %s", e, exc_info=True)
            self._reconcile_done.set()
        finally:
            self._startup_maintenance_deferred = False
            if startup_doc_store is not None:
                startup_doc_store.close()

    def start_deferred_startup_maintenance(self) -> bool:
        """Start deferred startup scans once the daemon is ready to serve."""
        if not self._startup_maintenance_deferred:
            return False
        if self._startup_maintenance_started.is_set():
            return False
        self._startup_maintenance_started.set()
        self._startup_maintenance_thread = threading.Thread(
            target=self._run_deferred_startup_maintenance,
            daemon=True,
            name="startup-maintenance",
        )
        self._startup_maintenance_thread.start()
        return True

    def _auto_reconcile_safe(self, chroma_coll: str, doc_coll: str) -> None:
        """Background-safe wrapper for auto-reconcile. Logs failures."""
        if not self._reconcile_lock.acquire(blocking=False):
            logger.info("Reconciliation already in progress, skipping")
            return
        try:
            self._auto_reconcile(chroma_coll, doc_coll)
        except Exception as e:
            logger.warning("Auto-reconcile failed: %s", e)
        finally:
            self._reconcile_lock.release()
            self._reconcile_done.set()

    def _auto_reconcile(self, chroma_coll: str, doc_coll: str) -> None:
        """Fix store divergence using summaries (no content re-fetch needed).

        Validates embedding provider first, then fixes missing/orphaned items.
        Uses its own DocumentStore connection to avoid concurrent-access
        segfaults with the main thread's SQLite connection.
        """
        if self._config.embedding is None:
            logger.info("Skipping reconciliation: no embedding provider configured")
            return

        try:
            self._get_embedding_provider()
        except Exception as e:
            logger.warning("Skipping reconciliation: provider unavailable: %s", e)
            return

        # Open a separate SQLite connection for thread-safe reads.
        # The main thread may be running find() concurrently, and Python's
        # sqlite3 module crashes (segfault) on concurrent access to the
        # same Connection object from multiple threads.
        recon_ds = DocumentStore(self._document_store._db_path)
        try:
            result = self.reconcile(fix=True, _doc_store=recon_ds)
            logger.info(
                "Auto-reconcile complete: %d fixed, %d orphans removed, %d missing",
                result["fixed"], result["removed"], result["missing_from_index"],
            )
        finally:
            recon_ds.close()

    def ensure_sysdocs(self) -> None:
        """Run deferred system-doc migration if needed (idempotent, best-effort)."""
        if not self._needs_sysdoc_migration:
            return
        with _get_tracer("keeper").start_as_current_span("ensure_sysdocs"):
            try:
                self._migrate_system_documents()
                self._needs_sysdoc_migration = False  # clear AFTER success only
            except Exception as e:
                logger.warning("System doc migration deferred: %s", e, exc_info=True)

    def _migrate_system_documents(self, progress=None) -> dict:
        """Migrate system documents to stable IDs and current version."""
        result = migrate_system_documents(self, progress=progress)
        self._tagdoc_cache.clear()  # tagdocs may have changed
        self._cel_cache.clear()
        self._cel_cache.clear()
        self._context_cache.clear()
        self._scan_tagdoc_backfills()
        return result

    def _scan_tagdoc_backfills(self) -> None:
        """Ensure backfill is queued for every tagdoc with _inverse.

        Called once at startup (after system doc migration) to catch
        tagdocs that were created outside the normal put() path or
        from a previous version that didn't support edges.

        Also materializes any missing inverse tagdocs for existing
        stores that were created before inverse materialization existed.
        """
        doc_coll = self._resolve_doc_collection()
        tagdocs = self._document_store.query_by_id_prefix(doc_coll, ".tag/")
        for td in tagdocs:
            # Only top-level tagdocs (.tag/KEY, not .tag/KEY/VALUE)
            if "/" in td.id[5:]:
                continue
            inverse = td.tags.get("_inverse")
            if inverse:
                # Coerce to str in case stored as list (defensive)
                if isinstance(inverse, list):
                    inverse = inverse[0]
                self._check_edge_backfill(td.id[5:], inverse, doc_coll)
                self._ensure_inverse_tagdoc(td.id[5:], inverse, doc_coll)

    # -- Provider lifecycle methods are in ProviderLifecycleMixin --

    def _resolve_prompt_doc(
        self,
        prefix: str,
        doc_tags: dict[str, str],
        *,
        item_id: str | None = None,
        item_summary: str = "",
        required: bool = False,
    ) -> str | None:
        """Find a .prompt/* doc matching the given tags and return its prompt text.

        Scans .prompt/{prefix}/* documents. For each, parses match rules
        from content (same DSL as .meta/* docs) and checks against doc_tags.
        Returns the ## Prompt section of the best match (most specific).

        When ``item_id`` is provided, prompt docs with a ``scope`` tag
        are matched as a glob against the item ID (e.g., ``scope: *@*``
        matches email addresses).  More specific scopes win over less
        specific ones.

        Args:
            prefix: "analyze", "summarize", "supernode", etc.
            doc_tags: Tags of the document being processed
            item_id: Optional item ID for scope-glob matching
            item_summary: Summary of the item being processed (for _when CEL)
            required: When True, raise if no matching prompt doc resolves.

        Returns:
            Prompt text from the best-matching doc, or None.
        """
        doc_coll = self._resolve_doc_collection()
        prompt_docs = self._document_store.query_by_id_prefix(
            doc_coll, f".prompt/{prefix}/"
        )
        if not prompt_docs:
            if required:
                raise ValueError(f"missing prompt docs for .prompt/{prefix}/*")
            return None

        best_prompt = None
        best_specificity = -1  # more match rules = more specific

        for rec in prompt_docs:
            content = rec.summary if hasattr(rec, 'summary') else ""

            prompt_text = extract_prompt_section(content)
            if not prompt_text:
                continue

            # Check _when condition (CEL expression against item context)
            rec_tags = rec.tags if hasattr(rec, 'tags') else {}
            when_source = rec_tags.get("_when", "")
            when_matched = False
            if when_source:
                item_ctx = build_item_context(
                    id=item_id or "",
                    tags=doc_tags,
                    summary=item_summary,
                    content_type=doc_tags.get("_content_type", ""),
                    uri=doc_tags.get("_source_uri", ""),
                )
                if not eval_when_predicate(when_source, item_ctx, cache=self._cel_cache):
                    continue  # item doesn't match this prompt's condition
                when_matched = True

            # Check scope tag (glob against item ID)
            scope = rec_tags.get("scope")
            if isinstance(scope, list):
                scope = scope[0] if scope else None
            if scope and item_id:
                if not fnmatch(item_id, str(scope)):
                    continue  # scope doesn't match this item
                # Scope specificity: non-wildcard chars count as more specific
                scope_specificity = len(str(scope).replace("*", "").replace("?", ""))
                if scope_specificity > best_specificity:
                    best_specificity = scope_specificity
                    best_prompt = prompt_text
                continue
            elif scope and not item_id:
                continue  # scope requires item_id

            # If _when matched, it takes precedence over body match rules.
            # Specificity 1 beats the bare default (0).
            if when_matched:
                if 1 > best_specificity:
                    best_specificity = 1
                    best_prompt = prompt_text
                continue

            # Parse match rules from body (before ## Prompt)
            query_lines, _, _ = _parse_meta_doc(content)

            if not query_lines:
                # No match rules = default/fallback (specificity 0)
                if best_specificity < 0:
                    best_prompt = prompt_text
                    best_specificity = 0
                continue

            # Check if any query line fully matches doc_tags
            for query in query_lines:
                if all(v in tag_values(doc_tags, k) for k, v in query.items()):
                    specificity = len(query)
                    if specificity > best_specificity:
                        best_specificity = specificity
                        best_prompt = prompt_text
                    break

        if best_prompt is None and required:
            raise ValueError(f"no matching prompt doc for .prompt/{prefix}/*")
        return best_prompt

    def _load_prompt_doc(
        self,
        doc_id: str,
        *,
        required: bool = False,
    ) -> str | None:
        """Load a prompt doc by exact ID and return its ``## Prompt`` section."""
        doc_coll = self._resolve_doc_collection()
        rec = self._document_store.get(doc_coll, doc_id)
        if rec is None or not getattr(rec, "summary", None):
            if required:
                raise ValueError(f"missing prompt doc: {doc_id}")
            return None
        prompt_text = extract_prompt_section(rec.summary)
        if prompt_text:
            return prompt_text
        if required:
            raise ValueError(f"prompt doc has no ## Prompt section: {doc_id}")
        return None

    def _gather_context(
        self,
        id: str,
        tags: dict[str, str],
    ) -> str | None:
        """Gather related item summaries that share any user tag.

        Uses OR union (any tag matches), not AND intersection.
        Boosts score when multiple tags match.

        Args:
            id: ID of the item being summarized (to exclude from results)
            tags: User tags from the item being summarized

        Returns:
            Formatted context string, or None if no related items found
        """
        if not tags:
            return None

        # Get similar items (broader search, we'll filter by tags)
        try:
            similar = self.find(similar_to=id, limit=20)
        except KeyError:
            # Item not found yet (first indexing) - no context available
            return None

        # Score each item: similarity * (1 + matching_tag_count * boost)
        TAG_BOOST = 0.2  # 20% boost per matching tag
        scored: list[tuple[float, int, Item]] = []

        for item in similar:
            if item.id == id:
                continue

            # Count matching tags (OR: at least one must match)
            matching = sum(
                1 for k, v in iter_tag_pairs(tags, include_system=False)
                if v in tag_values(item.tags, k)
            )
            if matching == 0:
                continue  # No tag overlap, skip

            # Boost score by number of matching tags
            base_score = item.score if item.score is not None else 0.5
            boosted_score = base_score * (1 + matching * TAG_BOOST)
            scored.append((boosted_score, matching, item))

        if not scored:
            return None

        # Sort by boosted score, take top 5
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:5]

        # Format context as topic keywords only (not summaries).
        # Including raw summary text causes small models to parrot
        # phrases from context into the new summary (contamination).
        topic_values: set[str] = set()
        for _, _, item in top:
            for key in filter_non_system_tags(item.tags):
                for value in tag_values(item.tags, key):
                    topic_values.add(value)

        if not topic_values:
            return None

        return "Related topics: " + ", ".join(sorted(topic_values))

    def _validate_embedding_identity(self, provider: EmbeddingProvider) -> None:
        """Validate embedding provider matches stored identity, or record it.

        On first use, records the embedding identity to config.
        On subsequent uses, if the provider changed, silently updates config
        and enqueues reindex tasks into the pending queue.
        """
        # Get current provider's identity
        current = EmbeddingIdentity(
            provider=self._config.embedding.name,
            model=provider_model_name(provider),
            dimension=provider.dimension,
        )

        stored = self._config.embedding_identity

        if stored is None:
            # First use: record the identity and set store dimension
            logger.info(
                "Recording embedding identity: %s/%s (%dd)",
                current.provider, current.model, current.dimension,
            )
            self._config.embedding_identity = current
            save_config(self._config)
            self._store.reset_embedding_dimension(current.dimension)
        else:
            if stored.compatibility_key == current.compatibility_key:
                if (stored.provider != current.provider or
                    stored.model != current.model):
                    logger.info(
                        "Embedding provider changed compatibly: %s/%s (%dd) → %s/%s (%dd)",
                        stored.provider, stored.model, stored.dimension,
                        current.provider, current.model, current.dimension,
                    )
                    self._config.embedding_identity = current
                    save_config(self._config)
                return

            # Check for provider change requiring a reindex
            if (stored.provider != current.provider or
                stored.model != current.model or
                stored.dimension != current.dimension):
                logger.info(
                    "Embedding provider changed: %s/%s (%dd) → %s/%s (%dd)",
                    stored.provider, stored.model, stored.dimension,
                    current.provider, current.model, current.dimension,
                )
                self._config.embedding_identity = current
                save_config(self._config)
                # Update store dimension for new model
                self._store.reset_embedding_dimension(current.dimension)
                # If dimension changed, drop ChromaDB collection so it's
                # recreated with the new dimension on first write.
                # Without this, ChromaDB rejects new-dimension vectors.
                if stored.dimension != current.dimension:
                    chroma_coll = self._resolve_chroma_collection()
                    try:
                        self._store.delete_collection(chroma_coll)
                        logger.info(
                            "Dropped search index (dimension %d → %d)",
                            stored.dimension, current.dimension,
                        )
                    except Exception as e:
                        logger.warning("Could not drop search index: %s", e)
                # Enqueue reindex tasks for pending queue
                try:
                    stats = self.enqueue_reindex()
                    logger.info(
                        "Enqueued %d items (+%d versions) for reindex",
                        stats["enqueued"], stats["versions"],
                    )
                    dim_msg = (
                        f" Dimension changed ({stored.dimension}→{current.dimension});"
                        f" search index was cleared."
                        if stored.dimension != current.dimension else ""
                    )
                    print(
                        f"Embedding model changed.{dim_msg}\n"
                        f"Enqueued {stats['enqueued']} items for reindex.\n"
                        f"Search is unavailable until reindex completes.\n"
                        f"Run: keep daemon",
                        file=sys.stderr,
                    )
                except Exception as e:
                    logger.error("Failed to enqueue reindex after model change: %s", e)
                    print(
                        f"ERROR: Embedding model changed but reindex failed.\n"
                        f"Search will not work until reindex completes.\n"
                        f"Try: keep daemon --reindex\n"
                        f"Details: {e}",
                        file=sys.stderr,
                    )
            else:
                logger.debug(
                    "Embedding identity unchanged: %s/%s (%dd)",
                    stored.provider, stored.model, stored.dimension,
                )

    def enqueue_reindex(self) -> dict:
        """Enqueue embed tasks for all docs, versions, and parts.

        Returns:
            Dict with stats: enqueued (int), versions (int), parts (int)
        """
        doc_coll = self._resolve_doc_collection()
        doc_ids = self._document_store.list_ids(doc_coll)
        enqueued = 0
        versions = 0
        parts = 0
        batch: list[tuple[str, str, str, str, dict | None]] = []
        BATCH_SIZE = 200

        doc_ids = [doc_id for doc_id in doc_ids if not doc_id.startswith(".")]
        for i in range(0, len(doc_ids), BATCH_SIZE):
            chunk_ids = doc_ids[i:i + BATCH_SIZE]
            records = self._document_store.get_many(doc_coll, chunk_ids)
            versions_by_id = self._document_store.list_versions_many(doc_coll, chunk_ids)
            parts_by_id = self._document_store.list_parts_many(doc_coll, chunk_ids)

            for doc_id in chunk_ids:
                record = records.get(doc_id)
                if record is None:
                    continue
                batch.append((doc_id, doc_coll, record.summary, "reindex",
                              {"tags": dict(record.tags)}))
                enqueued += 1

                # Enqueue version reindex
                for vi in versions_by_id.get(doc_id, []):
                    version_id = f"{doc_id}@v{vi.version}"
                    batch.append((version_id, doc_coll, vi.summary, "reindex",
                                  {"version": vi.version, "base_id": doc_id,
                                   "tags": dict(vi.tags)}))
                    versions += 1

                # Enqueue part reindex
                for pi in parts_by_id.get(doc_id, []):
                    if not pi.summary:
                        continue
                    part_id = f"{doc_id}@p{pi.part_num}"
                    batch.append((part_id, doc_coll, pi.summary, "reindex",
                                  {"part_num": pi.part_num, "base_id": doc_id,
                                   "tags": dict(pi.tags)}))
                    parts += 1

                if len(batch) >= BATCH_SIZE:
                    self._pending_queue.enqueue_batch(batch)
                    batch.clear()

        if batch:
            self._pending_queue.enqueue_batch(batch)

        logger.info(
            "Enqueue reindex: %d items + %d versions + %d parts",
            enqueued, versions, parts,
        )
        return {"enqueued": enqueued, "versions": versions, "parts": parts}

    # -------------------------------------------------------------------------
    # Data Export / Import
    # -------------------------------------------------------------------------

    def export_iter(self, *, include_system: bool = True) -> Iterator[dict]:
        """Stream-export all documents as an iterator of dicts.

        Yields dicts one at a time so that arbitrarily large stores can be
        exported without loading everything into memory.

        **First yield** — header dict::

            {"format": "keep-export", "version": 3, "exported_at": "...",
             "store_info": {"document_count": N, "version_count": N,
                            "part_count": N, "collection": "..."}}

        **Subsequent yields** — one dict per document. Each document is
        self-contained: its ``versions`` and ``parts`` lists are included
        inline (not yielded separately)::

            {"id": "...", "summary": "...", "tags": {...},
             "created_at": "...", "updated_at": "...", "accessed_at": "...",
             "versions": [...], "parts": [...]}

        Embeddings are excluded (model-dependent; regenerated on import).

        Args:
            include_system: If False, skip system documents (dot-prefix IDs)
        """
        doc_coll = self._resolve_doc_collection()
        doc_ids = self._document_store.list_ids(doc_coll)

        if not include_system:
            doc_ids = [d for d in doc_ids if not d.startswith(".")]

        # Header with document count; version/part counts filled per-document
        # as we stream (header store_info is best-effort for streaming)
        yield {
            "format": "keep-export",
            "version": 3,
            "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "store_info": {
                "document_count": len(doc_ids),
                "version_count": sum(
                    self._document_store.version_count(doc_coll, d)
                    for d in doc_ids
                ),
                "part_count": sum(
                    self._document_store.part_count(doc_coll, d)
                    for d in doc_ids
                ),
                "collection": doc_coll,
            },
        }

        for doc_id in doc_ids:
            record = self._document_store.get(doc_coll, doc_id)
            if record is None:
                continue

            doc_dict: dict = {
                "id": record.id,
                "summary": record.summary,
                "tags": dict(record.tags),
                "content_hash": record.content_hash,
                "content_hash_full": record.content_hash_full,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "accessed_at": record.accessed_at,
            }

            # Versions — inline within the document dict
            versions = []
            for vi in self._document_store.list_versions(doc_coll, doc_id, limit=10000):
                versions.append({
                    "version": vi.version,
                    "summary": vi.summary,
                    "tags": dict(vi.tags),
                    "content_hash": vi.content_hash,
                    "created_at": vi.created_at,
                })
            if versions:
                doc_dict["versions"] = versions

            # Parts — inline within the document dict
            parts = []
            for pi in self._document_store.list_parts(doc_coll, doc_id):
                parts.append({
                    "part_num": pi.part_num,
                    "summary": pi.summary,
                    "tags": dict(pi.tags),
                    "created_at": pi.created_at,
                })
            if parts:
                doc_dict["parts"] = parts

            yield doc_dict

    def export_data(self, *, include_system: bool = True) -> dict:
        """Export all documents, versions, and parts as a single dict.

        Convenience wrapper around :meth:`export_iter` that collects everything
        into memory.  Fine for small/medium stores; for large stores use
        ``export_iter()`` directly to stream documents.

        Returns:
            Dict in keep-export format (version 3) with a ``documents`` list.
        """
        it = self.export_iter(include_system=include_system)
        header = next(it)
        header["documents"] = list(it)
        return header

    def export_bundle(
        self,
        id: str,
        *,
        include_system: bool = True,
        include_parts: bool = True,
        include_versions: bool = True,
    ) -> dict | None:
        """Export one note bundle with metadata needed for markdown rendering."""
        return _get_export_bundle(
            self,
            id,
            include_system=include_system,
            include_parts=include_parts,
            include_versions=include_versions,
        )

    def export_changes(
        self,
        *,
        cursor: str | None = None,
        limit: int = 1000,
    ) -> dict:
        """Return a non-destructive cursor-based sync change feed."""
        try:
            after_outbox_id = int(str(cursor or "0"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid export change cursor: {cursor!r}") from exc
        if after_outbox_id < 0:
            raise ValueError(f"invalid export change cursor: {cursor!r}")
        limit = max(0, min(int(limit), _MAX_EXPORT_CHANGES_LIMIT))

        oldest_id, newest_id = self._document_store.sync_outbox_bounds()
        compacted = (
            after_outbox_id > 0
            and oldest_id is not None
            and oldest_id > after_outbox_id + 1
        )
        events = self._document_store.list_sync_outbox_since(
            after_outbox_id=after_outbox_id,
            limit=limit,
        )
        if events:
            doc_coll = self._resolve_doc_collection()
            dependencies = NoteDependencyService(self._document_store, doc_coll)
            _expanded_targets: dict[str, list[str]] = {}

            def _dedupe_note_ids(values: list[str]) -> list[str]:
                seen: set[str] = set()
                deduped: list[str] = []
                for value in values:
                    if not value or value in seen:
                        continue
                    seen.add(value)
                    deduped.append(value)
                return deduped

            for row in events:
                mutation = str(row.get("mutation") or "")
                doc_id = str(row.get("entity_id") or "")
                payload = decode_sync_event_payload(row.get("payload_json"))
                affected_note_ids: list[str] = []

                if mutation == DOC_UPDATE_MUTATION:
                    affected_note_ids = [doc_id]
                    if doc_id not in _expanded_targets:
                        _expanded_targets[doc_id] = dependencies.all_target_ids(doc_id)
                    affected_note_ids.extend(_expanded_targets[doc_id])
                elif mutation in PART_MUTATIONS:
                    affected_note_ids = [doc_id]
                elif mutation in VERSION_MUTATIONS:
                    affected_note_ids = [doc_id]
                elif mutation in EDGE_MUTATIONS:
                    affected_note_ids = [
                        doc_id,
                        str(payload.get("target_id") or ""),
                        str(payload.get("old_target_id") or ""),
                    ]
                elif mutation in VERSION_EDGE_MUTATIONS:
                    affected_note_ids = [
                        doc_id,
                        str(payload.get("target_id") or ""),
                        str(payload.get("old_target_id") or ""),
                    ]
                elif mutation in DOC_STRUCTURAL_MUTATIONS:
                    affected_note_ids = [doc_id]

                row["affected_note_ids"] = _dedupe_note_ids(affected_note_ids)
        head_cursor = newest_id or after_outbox_id
        next_cursor = events[-1]["outbox_id"] if events else head_cursor
        truncated = (
            bool(events)
            and newest_id is not None
            and events[-1]["outbox_id"] < newest_id
        )
        return {
            "format": "keep-export-changes",
            "version": 1,
            "cursor": str(next_cursor),
            "head_cursor": str(head_cursor),
            "compacted": compacted,
            "truncated": truncated,
            "events": events,
        }

    def import_data(self, data: dict, *, mode: str = "merge") -> dict:
        """Import documents from an export dict.

        All SQLite writes happen in a single transaction for speed.
        Re-embedding is queued for background processing (not done inline).

        Args:
            data: Dict in keep-export format
            mode: "merge" (skip existing IDs) or "replace" (clear first)

        Returns:
            Dict with stats: {imported, skipped, versions, parts, queued}
        """
        if data.get("format") != "keep-export":
            raise ValueError("Invalid export format (expected 'keep-export')")
        if data.get("version", 0) > 3:
            raise ValueError(
                f"Export format version {data['version']} is not supported "
                f"(this version supports up to 3)"
            )
        return self._import_documents(data.get("documents", []), mode=mode)

    def _collect_import_edge_keys(self, documents: list[dict[str, Any]]) -> set[str]:
        """Return edge-tag keys declared either in the batch or existing store."""
        incoming_edge_keys: set[str] = set()
        for doc in documents:
            doc_id = str(doc.get("id", ""))
            if not doc_id.startswith(".tag/"):
                continue
            key = doc_id[5:]
            if not key or "/" in key:
                continue
            raw_tags = doc.get("tags", {})
            if isinstance(raw_tags, dict) and raw_tags.get("_inverse"):
                incoming_edge_keys.add(key)
        return incoming_edge_keys

    @staticmethod
    def _rewrite_import_edge_ref(value: str, ref_map: dict[str, str] | None) -> str:
        """Map exported vault-local refs back to canonical keep ids."""
        if not ref_map or "[[" not in value:
            return value
        stripped = value.strip()
        target, alias = parse_ref(stripped)
        mapped = ref_map.get(target)
        if mapped is None:
            return stripped
        return format_ref(mapped, alias)

    def _normalize_import_edge_tags(
        self,
        tags: dict[str, Any],
        *,
        doc_coll: str,
        incoming_edge_keys: set[str],
        ref_map: dict[str, str] | None = None,
    ) -> None:
        """Normalize import-time edge tags, optionally remapping export refs."""
        if not tags:
            return
        for key in list(tags.keys()):
            if key.startswith(SYSTEM_TAG_PREFIX):
                continue
            is_edge_key = key in incoming_edge_keys
            if not is_edge_key:
                td_tags = self._get_tagdoc_tags(key, doc_coll)
                is_edge_key = bool(td_tags and td_tags.get("_inverse"))
            if not is_edge_key:
                continue

            value = tags[key]
            if isinstance(value, list):
                tags[key] = [
                    normalize_edge_value(self._rewrite_import_edge_ref(v, ref_map))
                    if isinstance(v, str)
                    else v
                    for v in value
                ]
            elif isinstance(value, str):
                rewritten = self._rewrite_import_edge_ref(value, ref_map)
                tags[key] = normalize_edge_value(rewritten)

    def _import_documents(
        self,
        documents: list[dict[str, Any]],
        *,
        mode: str = "merge",
        ref_map: dict[str, str] | None = None,
    ) -> dict:
        """Import already-parsed documents into the store."""
        doc_coll = self._resolve_doc_collection()
        incoming_edge_keys = self._collect_import_edge_keys(documents)

        if mode == "replace":
            self._document_store.delete_collection_all(doc_coll)
            chroma_coll = self._resolve_chroma_collection()
            try:
                self._store.delete_collection(chroma_coll)
            except Exception:
                pass  # ChromaDB collection may not exist yet

        # In merge mode, get existing IDs to skip
        existing_ids: set[str] = set()
        if mode == "merge":
            existing_ids = set(self._document_store.list_ids(doc_coll))

        # Validate imported tag keys and normalize case to match normal writes.
        for doc in documents:
            doc_id = doc.get("id", "<unknown>")
            doc["tags"] = self._validate_tag_map(
                doc.get("tags", {}),
                source=f"Import document tags ({doc_id})",
                check_constraints=False,
            )
            self._normalize_import_edge_tags(
                doc["tags"],
                doc_coll=doc_coll,
                incoming_edge_keys=incoming_edge_keys,
                ref_map=ref_map,
            )
            for ver in doc.get("versions", []):
                ver_num = ver.get("version", "?")
                ver["tags"] = self._validate_tag_map(
                    ver.get("tags", {}),
                    source=f"Import version tags ({doc_id}@v{ver_num})",
                    check_constraints=False,
                )
                self._normalize_import_edge_tags(
                    ver["tags"],
                    doc_coll=doc_coll,
                    incoming_edge_keys=incoming_edge_keys,
                    ref_map=ref_map,
                )
            for part in doc.get("parts", []):
                part_num = part.get("part_num", "?")
                part["tags"] = self._validate_tag_map(
                    part.get("tags", {}),
                    source=f"Import part tags ({doc_id}@p{part_num})",
                    check_constraints=False,
                )
                self._normalize_import_edge_tags(
                    part["tags"],
                    doc_coll=doc_coll,
                    incoming_edge_keys=incoming_edge_keys,
                    ref_map=ref_map,
                )
                part["summary"] = str(part.get("content") or part.get("summary") or "")

        # Filter to importable documents
        to_import = []
        skipped = 0
        for doc in documents:
            if doc["id"] in existing_ids:
                skipped += 1
            else:
                to_import.append(doc)

        # Batch-insert all documents, versions, parts in one transaction
        stats = self._document_store.import_batch(doc_coll, to_import)

        # Bulk-enqueue reindex tasks for all imported documents
        queued = 0
        for doc in to_import:
            doc_id = doc["id"]
            self._pending_queue.enqueue(
                doc_id, doc_coll, doc.get("summary", ""),
                task_type="reindex",
                metadata={"tags": doc.get("tags", {})},
            )
            queued += 1

            # Enqueue version reindex
            for ver in doc.get("versions", []):
                version_id = f"{doc_id}@v{ver['version']}"
                self._pending_queue.enqueue(
                    version_id, doc_coll, ver.get("summary", ""),
                    task_type="reindex",
                    metadata={
                        "version": ver["version"],
                        "base_id": doc_id,
                        "tags": ver.get("tags", {}),
                    },
                )
                queued += 1

            for part in doc.get("parts", []):
                summary = str(part.get("summary") or "")
                if not summary:
                    continue
                part_id = f"{doc_id}@p{part['part_num']}"
                self._pending_queue.enqueue(
                    part_id, doc_coll, summary,
                    task_type="reindex",
                    metadata={
                        "part_num": part["part_num"],
                        "base_id": doc_id,
                        "tags": part.get("tags", {}),
                    },
                )
                queued += 1

        return {
            "imported": stats["documents"],
            "skipped": skipped,
            "versions": stats["versions"],
            "parts": stats["parts"],
            "queued": queued,
        }

    def import_markdown(
        self,
        source: str | Path,
        *,
        mode: str = "merge",
        progress: Any | None = None,
    ) -> dict:
        """Import one markdown file or a recursive markdown directory tree."""
        documents, ref_map = load_markdown_import(source, progress=progress)
        return self._import_documents(documents, mode=mode, ref_map=ref_map)

    @property
    def embedding_identity(self) -> EmbeddingIdentity | None:
        """Current embedding identity (provider, model, dimension)."""
        return self._config.embedding_identity
    
    def _resolve_chroma_collection(self) -> str:
        """Vector collection name derived from embedding identity."""
        if self._config.embedding_identity:
            return self._config.embedding_identity.key
        return "default"

    def _resolve_doc_collection(self) -> str:
        """DocumentStore collection — always 'default'."""
        return "default"

    # ------------------------------------------------------------------
    # .ignore — global store-level ignore patterns
    # ------------------------------------------------------------------

    def _load_ignore_patterns(self) -> list[str]:
        """Load and cache ``.ignore`` patterns from the store (60s TTL)."""
        now = time.monotonic()
        if self._ignore_patterns is not None and (now - self._ignore_patterns_ts) < 60:
            return self._ignore_patterns
        doc_coll = self._resolve_doc_collection()
        rec = self._document_store.get(doc_coll, ".ignore")
        if rec and rec.summary:
            self._ignore_patterns = parse_ignore_patterns(rec.summary)
        else:
            self._ignore_patterns = []
        self._ignore_patterns_ts = now
        return self._ignore_patterns

    def _invalidate_ignore_cache(self) -> None:
        """Clear cached ``.ignore`` patterns."""
        self._ignore_patterns = None
        self._ignore_patterns_ts = 0.0

    def _purge_ignored_items(self, patterns: list[str]) -> dict:
        """Delete items matching ignore patterns + cancel queued work.

        Called automatically when ``.ignore`` gains new patterns.
        Handles both file-path patterns (matched against ``file://`` items)
        and URI-scheme patterns like ``git://x-access-token/*``.
        Returns ``{"deleted": int, "cancelled": int}``.
        """
        doc_coll = self._resolve_doc_collection()

        # Always query file:// items (for path-glob patterns).
        # Also query any URI-scheme prefixes derived from URI patterns.
        prefixes = ["file://"]
        for p in uri_pattern_prefixes(patterns):
            if p not in prefixes:
                prefixes.append(p)

        seen_ids: set[str] = set()
        matched_ids: set[str] = set()
        for prefix in prefixes:
            records = self._document_store.query_by_id_prefix(
                doc_coll, prefix, limit=0,
            )
            for rec in records:
                if rec.id in seen_ids:
                    continue
                seen_ids.add(rec.id)
                if match_ignore(rec.id, patterns):
                    matched_ids.add(rec.id)

        if not matched_ids:
            return {"deleted": 0, "cancelled": 0}

        for mid in matched_ids:
            try:
                self.delete(mid)
            except Exception as e:
                logger.warning("Failed to purge %s: %s", mid, e)

        cancelled = 0
        try:
            wq = self._get_work_queue()
            cancelled = wq.cancel_by_item_ids(matched_ids)
        except Exception as e:
            logger.warning("Failed to cancel work items: %s", e)

        logger.info(
            "Purged %d items and cancelled %d work items matching .ignore patterns",
            len(matched_ids), cancelled,
        )
        return {"deleted": len(matched_ids), "cancelled": cancelled}

    def _build_tag_where(self, tags: dict) -> dict | None:
        """Build backend where-clause for tag filters."""
        builder = getattr(self._store, "build_tag_where", None)
        if callable(builder):
            return builder(tags)
        conditions: list[dict[str, str]] = []
        for key in tags:
            for value in tag_values(tags, key):
                conditions.append({key: value})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    # Cap for the _base_id expansion in tag-filtered find. Large enough
    # to cover realistic "papers tagged topic=X" fan-outs without
    # producing an unwieldy Chroma where clause.
    _FIND_PART_JOIN_LIMIT = 200
    # Hard ceiling on rows scanned when paginating the first-tag query
    # for multi-tag conjunctions. Prevents pathological "first tag
    # matches everything, second tag narrows it" queries from scanning
    # the entire collection while still being large enough to find
    # realistic matches that sit outside the _FIND_PART_JOIN_LIMIT
    # recency window.
    _FIND_PART_JOIN_SCAN_BUDGET = 10_000

    def _ids_matching_tags(
        self, doc_coll: str, casefolded_tags: dict, limit: int,
    ) -> set[str]:
        """IDs of documents whose own stored tags satisfy every filter.

        Used by tag-filtered find() to expand the where clause so that
        parts reach their matching parents via ``_base_id`` — parts
        intentionally don't inherit parent tags (see analyze.py), so
        without this expansion a query like
        ``find(query=X, tags={topic: Y})`` would silently stop
        surfacing parts of topic=Y papers.

        For single-tag filters this is a direct ``query_by_tag_value``
        call capped at ``limit``. For multi-tag filters it paginates
        the first tag's rows and post-filters the rest in Python,
        accumulating matches until either ``limit`` is reached or the
        ``_FIND_PART_JOIN_SCAN_BUDGET`` is exhausted. The scan budget
        prevents a broad first tag from producing false negatives
        when the true match lies outside the first ``limit`` rows.
        """
        tag_pairs = list(iter_tag_pairs(casefolded_tags))
        if not tag_pairs:
            return set()

        first_key, first_value = tag_pairs[0]
        if len(tag_pairs) == 1:
            docs = self._document_store.query_by_tag_value(
                doc_coll, first_key, first_value, limit=limit,
            )
            return {doc.id for doc in docs}

        # Multi-tag conjunction: paginate first tag and post-filter.
        result: set[str] = set()
        offset = 0
        page_size = max(limit, 500)
        while len(result) < limit and offset < self._FIND_PART_JOIN_SCAN_BUDGET:
            docs = self._document_store.query_by_tag_value(
                doc_coll, first_key, first_value,
                limit=page_size, offset=offset,
            )
            if not docs:
                break
            for doc in docs:
                all_match = True
                for extra_key, extra_value in tag_pairs[1:]:
                    if extra_value not in tag_values(doc.tags, extra_key):
                        all_match = False
                        break
                if all_match:
                    result.add(doc.id)
                    if len(result) >= limit:
                        break
            offset += len(docs)
            if len(docs) < page_size:
                break  # source exhausted
        return result
    
    # -------------------------------------------------------------------------
    # Tag Validation
    # -------------------------------------------------------------------------

    def _validate_tag_map(
        self,
        tags: dict,
        *,
        source: str,
        check_constraints: bool,
    ) -> dict:
        """Casefold and validate tag keys/values for non-system tags."""
        try:
            normalized = casefold_tags(tags)
        except ValueError as e:
            raise ValueError(f"{source}: {e}") from e
        for key in normalized:
            if key.startswith(SYSTEM_TAG_PREFIX):
                continue
            try:
                validate_tag_key(key)
            except ValueError as e:
                raise ValueError(f"{source}: {e}") from e
            for value in tag_values(normalized, key):
                if len(value) > MAX_TAG_VALUE_LENGTH:
                    raise ValueError(
                        f"{source}: Tag value too long (max {MAX_TAG_VALUE_LENGTH}): {key!r}"
                    )
        if check_constraints:
            self._validate_tagdoc_value_constraints(normalized)
        return normalized

    def _validate_write_tags(self, tags: dict) -> dict:
        """Casefold, validate keys/lengths, and check tagdoc value constraints.

        Returns the casefolded tags dict.  Used by put(), tag(), tag_part().
        """
        return self._validate_tag_map(tags, source="Tags", check_constraints=True)

    def _get_tagdoc_tags(
        self, key: str, doc_coll: str,
    ) -> Optional[dict[str, Any]]:
        """Return cached parent tagdoc tags for one key, if present."""
        if key not in self._tagdoc_cache:
            parent = self._document_store.get(doc_coll, f".tag/{key}")
            tags = parent.tags if parent else None
            if tags is None:
                try:
                    tagdoc_id = f".tag/{key}"
                    rel_path = next(
                        (rel for rel, doc_id in SYSTEM_DOC_IDS.items() if doc_id == tagdoc_id),
                        None,
                    )
                    if rel_path is not None:
                        _, tags = _load_frontmatter(SYSTEM_DOC_DIR / rel_path)
                except Exception:
                    tags = None
            self._tagdoc_cache[key] = tags
        return self._tagdoc_cache[key]

    def _normalize_valid_edge_target(
        self, value: str,
    ) -> tuple[str, str | None]:
        """Return the canonical target ID for an edge-tag value.

        Raises ValueError when the parsed target is not a valid ordinary-note
        edge target. This is the single owner of edge-target validity for
        write-time validation and edge creation.
        """
        target_id, target_label = parse_ref(value)
        normalized_target = normalize_id(target_id)
        if normalized_target.startswith("."):
            raise ValueError(
                f"Edge target {normalized_target!r} is a system document and cannot be used as an edge target"
            )
        return normalized_target, target_label

    def _validate_tagdoc_value_constraints(
        self, tags: dict, existing_tags: dict | None = None,
    ) -> None:
        """Check tag values against tagdoc-defined write-time constraints.

        For each user tag, looks up `.tag/KEY`. If that doc exists and has
        `_constrained=true`, checks that `.tag/KEY/VALUE` exists. If it has
        `_value_regex`, validates the assigned values against that regex.

        If the tag doc has ``_requires``, the constraint only applies when
        the required tag is present in either *tags* or *existing_tags*.
        """
        doc_coll = self._resolve_doc_collection()
        for key in tags:
            if key.startswith(SYSTEM_TAG_PREFIX):
                continue
            values = tag_values(tags, key)
            if not values:
                continue
            td_tags = self._get_tagdoc_tags(key, doc_coll)
            if td_tags is None:
                continue  # no tag doc → unconstrained
            constrained = td_tags.get("_constrained") == "true"
            raw_value_regex = td_tags.get("_value_regex")
            has_value_regex = raw_value_regex is not None and str(raw_value_regex).strip() != ""
            if not constrained and not has_value_regex:
                continue
            if constrained and has_value_regex:
                raise ValueError(
                    f"Tagdoc '.tag/{key}' cannot declare both _constrained: true and _value_regex"
                )
            # _requires: only enforce constraint when the required tag is present
            requires = td_tags.get("_requires")
            if requires:
                in_new = requires in tags
                in_existing = bool(existing_tags and requires in existing_tags)
                if not in_new and not in_existing:
                    continue
            value_regex: re.Pattern[str] | None = None
            if has_value_regex:
                if not isinstance(raw_value_regex, str):
                    raise ValueError(
                        f"Tagdoc '.tag/{key}' has invalid _value_regex: expected non-empty string"
                    )
                try:
                    value_regex = re.compile(raw_value_regex)
                except re.error as exc:
                    raise ValueError(
                        f"Tagdoc '.tag/{key}' has invalid _value_regex: {exc}"
                    ) from exc
            is_edge_tag = bool(td_tags.get("_inverse"))
            for value in values:
                if value == "":
                    continue  # deletion, no validation needed
                if constrained:
                    value_id = f".tag/{key}/{value}"
                    if not self._document_store.get(doc_coll, value_id):
                        valid = self._list_constrained_values(key)
                        raise ValueError(
                            f"Invalid value for constrained tag '{key}': {value!r}. "
                            f"Valid values: {', '.join(sorted(valid))}"
                        )
                if value_regex is None:
                    continue
                validation_value = value
                if is_edge_tag:
                    try:
                        validation_value, _ = self._normalize_valid_edge_target(value)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid value for tag '{key}': {value!r}. {exc}"
                        ) from exc
                if not value_regex.fullmatch(validation_value):
                    raise ValueError(
                        f"Invalid value for tag '{key}': {validation_value!r}. "
                        f"Value must match regex {value_regex.pattern!r}"
                    )

    def _list_constrained_values(self, key: str) -> list[str]:
        """List valid values for a constrained tag by finding sub-docs."""
        doc_coll = self._resolve_doc_collection()
        prefix = f".tag/{key}/"
        docs = self._document_store.query_by_id_prefix(doc_coll, prefix)
        return [doc.id[len(prefix):] for doc in docs]

    def _get_singular_keys(self, keys: Iterable[str]) -> set[str]:
        """Return the subset of *keys* whose tagdoc has ``_singular=true``.

        For each non-system key, looks up ``.tag/{key}`` in the document
        store.  If the tagdoc exists and carries ``_singular: "true"``,
        the key is included in the result set.
        """
        doc_coll = self._resolve_doc_collection()
        singular: set[str] = set()
        for key in keys:
            if key.startswith(SYSTEM_TAG_PREFIX):
                continue
            parent = self._document_store.get(doc_coll, f".tag/{key}")
            if parent is not None and parent.tags.get("_singular") == "true":
                singular.add(key)
        return singular

    def _validate_singular_tags(self, add_changes: dict, singular_keys: set[str]) -> None:
        """Raise if any singular key has more than one incoming value."""
        for key in singular_keys:
            values = tag_values(add_changes, key)
            if len(values) > 1:
                raise ValueError(
                    f"Tag '{key}' is singular (at most one value allowed), "
                    f"but got {len(values)} values: {values!r}"
                )

    # -------------------------------------------------------------------------
    # Edge processing (tag-driven relationship edges)
    # -------------------------------------------------------------------------

    def _normalize_edge_tag_values(
        self, merged_tags: dict[str, Any], doc_coll: str,
    ) -> None:
        """Rewrite edge-tag values into canonical labeled-ref form.

        Mutates ``merged_tags`` in place. For each non-system key whose
        ``.tag/{key}`` tagdoc declares an ``_inverse`` (i.e. it's an edge
        tag), values are normalized into canonical MediaWiki-style
        ``[[target|label]]`` refs. Legacy ``target[[label]]`` refs and
        markdown links ``[Title](URL)`` are both accepted on input.
        Non-edge tags are left alone.

        This is the single normalization point for both ``put`` and
        ``set_tags`` mutation paths, so the canonical storage form stays
        stable regardless of which writer produced the value.
        """
        if not merged_tags:
            return
        for key in list(merged_tags.keys()):
            if key.startswith(SYSTEM_TAG_PREFIX):
                continue
            value = merged_tags[key]
            # Only proceed if this key is actually an edge tag.
            td_tags = self._get_tagdoc_tags(key, doc_coll)
            if td_tags is None or not td_tags.get("_inverse"):
                continue
            if isinstance(value, list):
                rewritten = [
                    normalize_edge_value(v) if isinstance(v, str) else v
                    for v in value
                ]
                if rewritten != value:
                    merged_tags[key] = rewritten
            elif isinstance(value, str):
                rewritten_str = normalize_edge_value(value)
                if rewritten_str != value:
                    merged_tags[key] = rewritten_str

    def _process_edge_tags(
        self,
        id: str,
        merged_tags: dict[str, str],
        existing_tags: dict[str, str],
        doc_coll: str,
        *,
        summary: str = "",
    ) -> None:
        """Create/update/delete edges based on tagdoc _inverse declarations.

        For each non-system tag on the document, checks whether the
        corresponding `.tag/{key}` tagdoc has an `_inverse` value.
        If so, the tag represents an edge: source=id, target=value,
        predicate=key, inverse=_inverse value.

        Also handles tag removal: if a key that was an edge-tag is no
        longer present, the corresponding edge row is deleted.
        """
        # System docs never participate as edge sources.
        # Hosted RBAC invariant: non-system writes (writer role) must not
        # trigger writes to dot-prefixed system documents.
        if id.startswith("."):
            return

        # Determine which keys to check: preserve first-seen key order from the
        # incoming write, then any keys that only existed on the prior version.
        current_keys = [
            k for k in merged_tags
            if not k.startswith(SYSTEM_TAG_PREFIX) and tag_values(merged_tags, k)
        ]
        previous_keys = [
            k for k in existing_tags
            if not k.startswith(SYSTEM_TAG_PREFIX) and tag_values(existing_tags, k)
        ]
        all_keys = list(dict.fromkeys(current_keys + previous_keys))
        if not all_keys:
            return  # no user tags → no edges possible

        # Collect batch operations across all edge-tag keys
        edges_to_delete: list[tuple[str, str, str]] = []  # (source_id, predicate, target_id)
        edges_to_add: list[tuple[str, str, str, str, str]] = []  # (source_id, predicate, target_id, inverse, created)
        backfill_checks: list[tuple[str, str]] = []  # (predicate, inverse)
        target_name_updates: dict[str, list[str]] = {}

        def _queue_target_name(target_id: str, label: str | None) -> None:
            label = (label or "").strip()
            if not label:
                return
            names = target_name_updates.setdefault(target_id, [])
            if label not in names:
                names.append(label)

        for key in all_keys:
            td_tags = self._get_tagdoc_tags(key, doc_coll)
            if td_tags is None:
                continue
            inverse = td_tags.get("_inverse")
            if not inverse:
                continue
            if isinstance(inverse, list):
                inverse = inverse[0]

            # Evaluate _when condition from tagdoc against source note.
            # When false, edge *creation* is suppressed but edge *removal*
            # must still run — a previously-matching source may have stopped
            # matching after a tag change.
            when_allows_creation = True
            when_source = td_tags.get("_when", "")
            if when_source:
                item_ctx = build_item_context(
                    id=id,
                    tags=merged_tags,
                    summary=summary,
                    content_type=merged_tags.get("_content_type", ""),
                    uri=merged_tags.get("_source_uri", ""),
                )
                if not eval_when_predicate(when_source, item_ctx, cache=self._cel_cache):
                    when_allows_creation = False

            current_values = tag_values(merged_tags, key)
            previous_values = tag_values(existing_tags, key)
            previous_value_set = set(previous_values)
            current_value_set = set(current_values)

            if when_allows_creation:
                removed_values = [v for v in previous_values if v not in current_value_set]
                added_values = [v for v in current_values if v not in previous_value_set]
            else:
                # _when is false: delete ALL edges for this key (previous
                # AND current values) and create nothing.
                removed_values = list(dict.fromkeys(previous_values + current_values))
                added_values = []

            # Tag values removed → queue edge deletions.
            for removed in removed_values:
                target_id = parse_ref(removed)[0]
                try:
                    target_id = normalize_id(target_id)
                except ValueError:
                    pass
                edges_to_delete.append((id, key, target_id))

            # Tag present → queue edge upsert + auto-vivify target.
            for current_value in added_values:
                if not current_value:
                    continue
                try:
                    target_id, target_label = self._normalize_valid_edge_target(current_value)
                except ValueError:
                    logger.debug("Skipping invalid edge target for tag %r: %r", key, current_value)
                    continue
                _queue_target_name(target_id, target_label)
                # Auto-vivify: create target as empty doc if it doesn't exist.
                # Uses atomic INSERT OR IGNORE to avoid TOCTOU race where a
                # concurrent writer creates the real document between check
                # and write (upsert would overwrite it with the empty stub).
                reference_created = (
                    merged_tags.get("_created")
                    or merged_tags.get("_updated")
                    or utc_now()
                )
                now = utc_now()
                target_tags = {
                    "_created": reference_created,
                    "_updated": now,
                    "_source": "auto-vivify",
                }
                if target_id in target_name_updates:
                    set_tag_values(target_tags, "name", target_name_updates[target_id])
                self._stub_via_flow(
                    id=target_id,
                    content="",
                    summary="",
                    tags=target_tags,
                    created_at=reference_created,
                    queue_background_tasks=True,
                )

                created = merged_tags.get("_created") or merged_tags.get("_updated") or utc_now()
                edges_to_add.append((id, key, target_id, inverse, created))
                backfill_checks.append((key, inverse))

        self._merge_auto_vivified_target_names(doc_coll, target_name_updates)

        # Execute batched edge mutations
        if edges_to_delete:
            self._document_store.delete_edges_batch(doc_coll, edges_to_delete)
        if edges_to_add:
            self._document_store.upsert_edges_batch(doc_coll, edges_to_add)

        # Trigger backfill checks (deduplicated by predicate)
        seen_predicates: set[str] = set()
        for predicate, inverse in backfill_checks:
            if predicate not in seen_predicates:
                seen_predicates.add(predicate)
                self._check_edge_backfill(predicate, inverse, doc_coll)

    def _check_edge_backfill(
        self, predicate: str, inverse: str, doc_coll: str,
    ) -> None:
        """Ensure edges are backfilled for a predicate that has _inverse.

        Only enqueues a backfill task if no record exists at all.
        A pending record (completed=NULL) means a task is already queued.
        """
        if self._document_store.backfill_exists(doc_coll, predicate):
            return  # already pending or completed
        self._document_store.upsert_backfill(doc_coll, predicate, inverse)
        self._pending_queue.enqueue(
            id=f".backfill/{predicate}",
            collection=doc_coll,
            content="",
            task_type="backfill-edges",
            metadata={"predicate": predicate, "inverse": inverse},
        )

    def _process_tagdoc_inverse_change(
        self,
        id: str,
        new_tags: dict[str, str],
        old_tags: dict[str, str],
        doc_coll: str,
    ) -> None:
        """Handle _inverse changes on a .tag/* document.

        Called *before* storage so old_tags reflect the previous state.
        Detects whether _inverse was added, changed, or removed, and
        adjusts edges/backfill accordingly.
        """
        # Extract predicate from tagdoc ID: .tag/KEY → KEY.
        # Explicit boundary: only system-tagdoc writes may mutate edge schema
        # and materialize inverse tagdocs. Non-system writes cannot enter here.
        if not id.startswith(".tag/") or "/" in id[5:]:
            return  # not a top-level tagdoc
        predicate = id[5:]

        old_inverse = old_tags.get("_inverse") if old_tags else None
        new_inverse = new_tags.get("_inverse") if new_tags else None

        if old_inverse == new_inverse:
            return  # no change

        if old_inverse and not new_inverse:
            # _inverse removed → clean up all edges and backfill for this predicate
            self._document_store.delete_edges_for_predicate(doc_coll, predicate)
            self._document_store.delete_version_edges_for_predicate(doc_coll, predicate)
            self._document_store.delete_backfill(doc_coll, predicate)
        elif new_inverse and old_inverse != new_inverse:
            # _inverse added or changed → delete old edges, enqueue new backfill
            if old_inverse:
                self._document_store.delete_edges_for_predicate(doc_coll, predicate)
                self._document_store.delete_version_edges_for_predicate(doc_coll, predicate)
                self._document_store.delete_backfill(doc_coll, predicate)
            self._check_edge_backfill(predicate, new_inverse, doc_coll)
            # Materialize the inverse tagdoc synchronously
            self._ensure_inverse_tagdoc(predicate, new_inverse, doc_coll)

    def _ensure_inverse_tagdoc(
        self, predicate: str, inverse: str, doc_coll: str,
    ) -> None:
        """Ensure .tag/{inverse} exists with _inverse={predicate}.

        Called synchronously when .tag/{predicate} declares _inverse={inverse}.
        Creates the inverse tagdoc if missing, or verifies consistency.

        Raises ValueError if .tag/{inverse} already has a different _inverse.
        """
        inverse_tagdoc_id = f".tag/{inverse}"
        existing = self._document_store.get(doc_coll, inverse_tagdoc_id)

        if existing:
            existing_inverse = existing.tags.get("_inverse")
            if existing_inverse == predicate:
                return  # already correct
            if existing_inverse and existing_inverse != predicate:
                raise ValueError(
                    f"Inverse conflict: .tag/{inverse} already declares "
                    f"_inverse={existing_inverse!r}, cannot set to {predicate!r} "
                    f"(required by .tag/{predicate} _inverse={inverse})"
                )
            # Exists without _inverse → add it
            tags = dict(existing.tags)
            tags["_inverse"] = predicate
            tags["_updated"] = utc_now()
            self._document_store.upsert(
                doc_coll, inverse_tagdoc_id,
                summary=existing.summary, tags=tags,
            )
        else:
            # Create minimal inverse tagdoc with a useful summary
            # so it can be embedded and found via search.
            now = utc_now()
            summary = f"Inverse edge tag for `{predicate}` (auto-generated)"
            self._document_store.upsert(
                doc_coll, inverse_tagdoc_id,
                summary=summary,
                tags={
                    "_inverse": predicate,
                    "_created": now,
                    "_updated": now,
                    "_source": "auto-vivify",
                    "category": "system",
                    "context": "tag-description",
                },
            )

        # Backfill edges for the inverse direction too
        self._check_edge_backfill(inverse, predicate, doc_coll)

    # -------------------------------------------------------------------------
    # Write Operations
    # -------------------------------------------------------------------------

    def _restored_processing_tags(
        self,
        restored_tags: dict[str, Any],
    ) -> dict[str, Any]:
        tags: dict[str, Any] = {}
        for key in ("_summarized_hash", "_tagged_hash", "_tagged_summary_hash"):
            if key in restored_tags:
                tags[key] = restored_tags[key]
        return tags

    # -- Task dispatch methods are in BackgroundProcessingMixin --
    # (_task_idempotency_key, _enqueue_*_background, _dispatch_after_write_flow,
    #  _load_after_write_state_doc, _store_write_context, _consume_write_context)

    def _upsert(
        self,
        id: str,
        content: str,
        *,
        tags: Optional[dict] = None,
        summary: Optional[str] = None,
        system_tags: dict[str, str],
        created_at: Optional[str] = None,
        force: bool = False,
        queue_summarize: bool = True,
        _body_authority: str | None = None,
    ) -> Item:
        """Core upsert logic used by put()."""
        with _get_tracer("keeper").start_as_current_span(
            "keeper.put", attributes={"item_id": id},
        ):
            return self.__upsert_impl(
                id, content, tags=tags, summary=summary,
                system_tags=system_tags, created_at=created_at,
                force=force, queue_summarize=queue_summarize,
                _body_authority=_body_authority,
            )

    def _resolve_note_body_summary(
        self,
        ctx: _PutContext,
    ) -> _ResolvedBodySummary:
        """Choose the stored note body/summary from the centralized policy.

        Mutates ``ctx.tags`` in place when the chosen body means the note
        should be treated as already summarized.  Keeping these inputs in a
        named context makes that mutation and its authority explicit at the
        call site.
        """
        max_len = self._config.max_summary_length
        is_system_doc = is_system_id(ctx.id)
        body_authority = resolve_body_authority(ctx.tags)

        if ctx.summary is not None:
            summary = ctx.summary
            if (
                body_authority != BODY_AUTHORITY_MARKDOWN
                and not is_system_doc
                and len(summary) > max_len
            ):
                warnings.warn(
                    f"Summary exceeds max_summary_length ({len(summary)} > {max_len}), truncating",
                    UserWarning,
                    stacklevel=3,
                )
                summary = summary[:max_len]
            return _ResolvedBodySummary(summary, body_authority)

        if is_system_doc:
            return _ResolvedBodySummary(ctx.content, body_authority)

        if ctx.restored_version is not None:
            return _ResolvedBodySummary(ctx.restored_version.summary, body_authority)

        if ctx.content_unchanged and ctx.tags_changed:
            logger.debug("Tags changed for %s", ctx.id)
            return _ResolvedBodySummary(ctx.existing_doc.summary, body_authority)

        if body_authority == BODY_AUTHORITY_MARKDOWN:
            ctx.tags["_summarized_hash"] = ctx.new_hash
            return _ResolvedBodySummary(ctx.content, body_authority)

        if len(ctx.content) <= max_len:
            ctx.tags["_summarized_hash"] = ctx.new_hash
            return _ResolvedBodySummary(ctx.content, body_authority)

        return _ResolvedBodySummary(ctx.content[:max_len] + "...", body_authority)

    def _apply_summary_mutation(
        self,
        collection: str,
        target: str,
        summary: str,
        *,
        intent: BodyWriteIntent,
        embed: bool = False,
        content_hash: str = "",
        content_hash_full: str = "",
    ) -> None:
        """Apply a body mutation through the centralized authority policy.

        A rejected mutation is a no-op for the body itself but does not inhibit
        sibling mutations in the same batch. Callers that need transactional
        semantics across ``set_summary`` and related tag updates must
        coordinate that explicitly.
        """
        existing = self._document_store.get(collection, target)
        existing_tags = dict(existing.tags or {}) if existing else {}
        authority = resolve_body_authority(existing_tags)
        if not body_write_allowed(authority, intent):
            logger.warning(
                "Skipped %s body mutation for markdown-authored note %s",
                intent,
                target,
            )
            return

        updated = self._document_store.update_summary(collection, target, summary)
        if not updated:
            return

        if content_hash:
            self._document_store.update_content_hash(
                collection,
                target,
                content_hash=content_hash,
                content_hash_full=content_hash_full,
            )

        if embed:
            chroma_coll = self._resolve_chroma_collection()
            embedding = self._get_embedding_provider().embed(summary)
            self._store.upsert(
                collection=chroma_coll,
                id=target,
                embedding=embedding,
                summary=summary,
                tags=casefold_tags_for_index(existing_tags),
            )
        else:
            self._store.update_summary(collection, target, summary)

    def __upsert_impl(
        self,
        id: str,
        content: str,
        *,
        tags: Optional[dict] = None,
        summary: Optional[str] = None,
        system_tags: dict[str, str],
        created_at: Optional[str] = None,
        force: bool = False,
        queue_summarize: bool = True,
        _body_authority: str | None = None,
    ) -> Item:
        content = repair_surrogate_text(content)
        if summary is not None:
            summary = repair_surrogate_text(summary)

        # Wait for background reconciliation to finish before writing.
        # The reconcile thread and main thread both access the embedding
        # provider, ChromaDB, and SQLite — concurrent access causes hangs
        # with ChromaDB's Rust bindings and double model loading.
        self._reconcile_done.wait(timeout=120)

        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()

        # Deferred init tasks (best-effort — don't block user writes)
        self.ensure_sysdocs()

        # Get existing item to preserve tags (check document store first, fall back to ChromaDB)
        existing_tags = {}
        existing_doc = self._document_store.get(doc_coll, id)
        if existing_doc:
            existing_tags = filter_non_system_tags(existing_doc.tags)
        else:
            existing = self._store.get(chroma_coll, id)
            if existing:
                existing_tags = filter_non_system_tags(existing.tags)

        # Preserve analysis watermark across puts (incremental analysis needs
        # the version baseline even when content changes).
        _prev_analyzed_version = None
        _prev_analyzed_hash = None
        if existing_doc:
            _prev_analyzed_version = existing_doc.tags.get("_analyzed_version")
            _prev_analyzed_hash = existing_doc.tags.get("_analyzed_hash")

        # Compute content hash for change detection
        new_hash = _content_hash(content)
        new_hash_full = _content_hash_full(content)

        restored_version = None
        restored_embedding = None
        if existing_doc is not None and existing_doc.content_hash != new_hash:
            restored_version = self._document_store.get_latest_version_info_by_content_hash(
                doc_coll, id, new_hash,
            )
            if restored_version is not None:
                if self._config.embedding is not None and not is_system_id(id):
                    restored_embedding = self._store.get_embedding(
                        chroma_coll,
                        f"{id}@v{restored_version.version}",
                    )

        # Build tags: existing + config + env + user, then replace system tags.
        if restored_version is not None:
            merged_tags = {
                **filter_non_system_tags(restored_version.tags),
            }
        else:
            merged_tags = {**existing_tags}

        if self._default_tags:
            _merge_tags_additive(merged_tags, self._default_tags)

        _merge_tags_additive(merged_tags, self._env_tags)

        if tags:
            user_tags = casefold_tags(filter_non_system_tags(tags))
            # Validate tagdoc-defined constraints against user-provided values
            # and already-present tags (_requires may depend on existing state).
            self._validate_tagdoc_value_constraints(user_tags, existing_tags=merged_tags)
            singular_keys = self._get_singular_keys(
                k for k in user_tags if tag_values(user_tags, k) != [""]
            )
            if singular_keys:
                self._validate_singular_tags(user_tags, singular_keys)
            # Replace semantics: new tag values replace existing, not accumulate
            for key in user_tags:
                values = tag_values(user_tags, key)
                if values == [""]:
                    merged_tags.pop(key, None)
                else:
                    set_tag_values(merged_tags, key, values)

        # Track content length as a system tag (always updated)
        system_tags["_content_length"] = str(len(content))

        _merge_tags_additive(merged_tags, system_tags, replace_system=True)

        if restored_version is not None:
            merged_tags.update(
                self._restored_processing_tags(restored_version.tags)
            )
            merged_tags["_restored_hash"] = new_hash
            merged_tags["_restored_from_version"] = str(restored_version.version)
            if _prev_analyzed_hash:
                merged_tags["_analyzed_hash"] = _prev_analyzed_hash
            if _prev_analyzed_version:
                merged_tags["_analyzed_version"] = _prev_analyzed_version

        # Restore analysis version watermark (dropped by filter_non_system_tags)
        if (
            restored_version is None
            and _prev_analyzed_version
            and "_analyzed_version" not in merged_tags
        ):
            merged_tags["_analyzed_version"] = _prev_analyzed_version

        requested_body_authority = (
            normalize_body_authority(_body_authority)
            if _body_authority is not None
            else None
        )
        if requested_body_authority == BODY_AUTHORITY_MARKDOWN:
            merged_tags[BODY_AUTHORITY_TAG] = BODY_AUTHORITY_MARKDOWN
        elif requested_body_authority == BODY_AUTHORITY_DERIVED:
            merged_tags.pop(BODY_AUTHORITY_TAG, None)
        elif existing_doc is not None:
            existing_authority = resolve_body_authority(existing_doc.tags)
            if existing_authority == BODY_AUTHORITY_MARKDOWN:
                merged_tags[BODY_AUTHORITY_TAG] = BODY_AUTHORITY_MARKDOWN

        # Normalize edge-tag values to canonical labeled-ref form before
        # change detection and storage. Done here so a single
        # call covers both cloud and local code paths below, and so the
        # change-detection diff sees the canonical form.
        self._normalize_edge_tag_values(merged_tags, doc_coll)

        # Change detection (before embedding to allow early return)
        content_unchanged = (
            existing_doc is not None
            and existing_doc.content_hash == new_hash
        )
        tags_changed = (
            existing_doc is not None
            and _user_tags_changed(existing_doc.tags, merged_tags)
        )

        # Backfill system tags for items stored before they were tracked
        if existing_doc is not None:
            backfill_tags = {}
            for sys_key in ("_content_type", "_content_length"):
                if sys_key in merged_tags and sys_key not in existing_doc.tags:
                    backfill_tags[sys_key] = merged_tags[sys_key]
            if backfill_tags:
                self._document_store.patch_head_tags(doc_coll, id, backfill_tags)

        # Early return: content and tags unchanged.
        # Still dispatch after-write flow — it will no-op if processing
        # is already complete, but re-enqueues if a prior purge dropped
        # unfinished work.
        if content_unchanged and not tags_changed and summary is None and not force:
            logger.debug("Content and tags unchanged for %s", id)
            if queue_summarize and not is_system_id(id):
                self._dispatch_after_write_flow(
                    item_id=id,
                    content=content,
                    tags=dict(existing_doc.tags),
                )
            return _record_to_item(existing_doc, changed=False)

        body_context = _PutContext(
            id=id,
            content=content,
            summary=summary,
            tags=merged_tags,
            existing_doc=existing_doc,
            restored_version=restored_version,
            content_unchanged=content_unchanged,
            tags_changed=tags_changed,
            new_hash=new_hash,
        )
        resolved_body = self._resolve_note_body_summary(body_context)
        final_summary = resolved_body.summary
        effective_body_authority = resolved_body.body_authority
        is_system_doc = is_system_id(id)
        max_len = self._config.max_summary_length

        _has_embeddings = self._config.embedding is not None
        should_index = _has_embeddings and not is_system_doc

        # Cloud mode: defer embedding to background worker for faster response.
        # The doc store write happens immediately; the note is findable by
        # tags/FTS/ID right away. Similarity search works once the
        # background worker computes and stores the embedding.
        if not self._is_local:
            result, content_changed = self._document_store.upsert(
                collection=doc_coll,
                id=id,
                summary=final_summary,
                tags=merged_tags,
                content_hash=new_hash,
                content_hash_full=new_hash_full,
                created_at=created_at,
            )
            if is_system_doc:
                self._store.delete(chroma_coll, id, delete_versions=True)
                return _record_to_item(result, changed=not content_unchanged)
            # Try embedding dedup before enqueueing (saves network round-trip)
            if should_index and not content_unchanged:
                if restored_embedding is not None:
                    self._store.upsert(
                        collection=chroma_coll, id=id,
                        embedding=restored_embedding,
                        summary=final_summary,
                        tags=casefold_tags_for_index(merged_tags),
                    )
                    return _record_to_item(result, changed=True)
                donor_embedding = self._try_dedup_embedding(
                    doc_coll, chroma_coll, new_hash, id, content,
                )
                if donor_embedding is not None:
                    self._store.upsert(
                        collection=chroma_coll, id=id,
                        embedding=donor_embedding,
                        summary=final_summary,
                        tags=casefold_tags_for_index(merged_tags),
                    )
                    return _record_to_item(result, changed=True)
            if not should_index:
                return _record_to_item(result, changed=not content_unchanged)
            # Enqueue embedding task (content needed for embedding computation)
            embed_meta = {}
            if existing_doc is not None and not content_unchanged:
                embed_meta["content_changed"] = True
            self._pending_queue.enqueue(
                id, doc_coll, content,
                task_type="embed",
                metadata=embed_meta,
            )
            return _record_to_item(result, changed=not content_unchanged)

        # Local mode: compute embedding synchronously
        # If no embedding provider, write to document store only (data is safe;
        # embeddings are filled in by reconciliation when a provider appears).
        _tracer = _get_tracer("keeper")
        embedding = None
        if should_index:
            with _tracer.start_as_current_span("embed", attributes={"item_id": id}) as _embed_span:
                _embed_source = "compute"
                if content_unchanged:
                    embedding = self._store.get_embedding(chroma_coll, id)
                    if embedding is not None:
                        _embed_source = "existing"
                    if embedding is None:
                        embedding = self._try_dedup_embedding(
                            doc_coll, chroma_coll, new_hash, id, content,
                        )
                        if embedding is not None:
                            _embed_source = "dedup"
                    if embedding is None:
                        embedding = self._get_embedding_provider().embed(final_summary)
                else:
                    if restored_embedding is not None:
                        embedding = restored_embedding
                        _embed_source = "restored_version"
                    else:
                        embedding = self._try_dedup_embedding(doc_coll, chroma_coll, new_hash, id, content)
                        if embedding is not None:
                            _embed_source = "dedup"
                    if embedding is None:
                        embedding = self._get_embedding_provider().embed(final_summary)
                _embed_span.set_attribute("source", _embed_source)

        # Detect _inverse changes on tagdocs BEFORE storage overwrites old state
        if id.startswith(".tag/"):
            old_tagdoc_tags = existing_doc.tags if existing_doc else {}
            self._process_tagdoc_inverse_change(id, merged_tags, old_tagdoc_tags, doc_coll)
            tag_key = id.removeprefix(".tag/").split("/")[0]
            self._tagdoc_cache.pop(tag_key, None)
            self._cel_cache.clear()  # tagdoc _when may have changed

        # Save old embedding before ChromaDB upsert overwrites it (for version archival)
        old_embedding = None
        if should_index and existing_doc is not None and not content_unchanged:
            old_embedding = self._store.get_embedding(chroma_coll, id)

        # Dual-write: document store (canonical) + ChromaDB (embedding index)
        with _tracer.start_as_current_span("doc_store.upsert"):
            result, content_changed = self._document_store.upsert(
                collection=doc_coll,
                id=id,
                summary=final_summary,
                tags=merged_tags,
                content_hash=new_hash,
                content_hash_full=new_hash_full,
                created_at=created_at,
            )

        if should_index:
            # If content changed and we have a version to archive, batch both
            # ChromaDB writes into a single call (one lock, one epoch bump).
            max_ver = (
                self._document_store.max_version(doc_coll, id)
                if existing_doc is not None and content_changed else 0
            )
            if max_ver > 0:
                with _tracer.start_as_current_span("chroma.upsert_batch"):
                    if old_embedding is None:
                        old_embedding = self._get_embedding_provider().embed(existing_doc.summary)
                    self._store.upsert_with_version(
                        collection=chroma_coll,
                        id=id,
                        embedding=embedding,
                        summary=final_summary,
                        tags=casefold_tags_for_index(merged_tags),
                        version_id=f"{id}@v{max_ver}",
                        version=max_ver,
                        version_embedding=old_embedding,
                        version_summary=existing_doc.summary,
                        version_tags=casefold_tags_for_index(existing_doc.tags),
                    )
            else:
                with _tracer.start_as_current_span("chroma.upsert"):
                    self._store.upsert(
                        collection=chroma_coll,
                        id=id,
                        embedding=embedding,
                        summary=final_summary,
                        tags=casefold_tags_for_index(merged_tags),
                    )
        elif is_system_doc:
            self._store.delete(chroma_coll, id, delete_versions=True)

        with _tracer.start_as_current_span("edge_tags"):
            self._process_edge_tags(
                id, merged_tags, existing_tags, doc_coll,
                summary=final_summary,
            )

        # .ignore update: purge items matching current patterns
        if id == ".ignore":
            self._invalidate_ignore_cache()
            pats = self._load_ignore_patterns()
            if pats:
                self._purge_ignored_items(pats)

        # Spawn background processor if needed (local only — uses filesystem locks)
        if (
            queue_summarize
            and summary is None
            and effective_body_authority != BODY_AUTHORITY_MARKDOWN
            and len(content) > max_len
            and (not content_unchanged or tags_changed or force)
        ):
            self._spawn_processor()

        self._context_cache.notify_write(
            id,
            old_tags=existing_tags,
            new_tags=merged_tags,
        )

        return _record_to_item(result, changed=not content_unchanged)

    def _put_direct(
        self,
        content: Optional[str] = None,
        *,
        uri: Optional[str] = None,
        id: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[TagMap] = None,
        created_at: Optional[str] = None,
        force: bool = False,
        queue_background_tasks: bool = True,
        capture_write_context: bool = False,
        _body_authority: str | None = None,
    ) -> Item:
        """Store content in the memory.

        Provide either inline content or a URI to fetch — not both.

        **Inline mode** (content provided):
        - Stores text directly. Auto-generates an ID if not provided.
        - Short content is used verbatim as summary. Large content gets
          async summarization (truncated placeholder stored immediately).

        **URI mode** (uri provided):
        - Fetches the document, extracts text, generates embeddings.

        Args:
            content: Inline text content to store.
            uri: URI of a document to fetch and index.
            id: Explicit item ID (auto-generated if omitted).
            summary: Pre-computed summary (skips auto-summarization).
            tags: Tag map to attach to the item.
            created_at: Override creation timestamp (ISO 8601).
            force: Re-process even if content is unchanged.
            queue_background_tasks: Enqueue summarization/analysis after write.
            capture_write_context: Attach write-time context as tags.
            _body_authority: Private override for authored-markdown semantics.
        """
        if content is not None and uri is not None:
            raise ValueError("Provide content or uri, not both")
        if content is None and uri is None:
            raise ValueError("Either content or uri is required")

        with _get_tracer("keeper").start_as_current_span(
            "put.request",
            attributes={
                "has_content": bool(content is not None),
                "has_uri": bool(uri is not None),
                "requested_id": id or "",
                "queue_background_tasks": bool(queue_background_tasks),
                "capture_write_context": bool(capture_write_context),
            },
        ):
            return self.__put_direct_impl(
                content=content,
                uri=uri,
                id=id,
                summary=summary,
                tags=tags,
                created_at=created_at,
                force=force,
                queue_background_tasks=queue_background_tasks,
                capture_write_context=capture_write_context,
                _body_authority=_body_authority,
            )

    def _stub_direct(
        self,
        *,
        id: str,
        content: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[TagMap] = None,
        created_at: Optional[str] = None,
        queue_background_tasks: bool = True,
    ) -> Item:
        """Insert a stub note only if it does not already exist."""
        if not str(id or "").strip():
            raise ValueError("stub requires id")

        normalized_id = normalize_id(id)
        if is_part_id(normalized_id):
            raise ValueError(
                f"Cannot create stub for part directly: {normalized_id!r}. "
                "Parts are managed by analyze()."
            )

        if tags:
            tags = self._validate_write_tags(tags)
        if content is not None:
            content = repair_surrogate_text(content)
        if summary is not None:
            summary = repair_surrogate_text(summary)

        with _get_tracer("keeper").start_as_current_span(
            "stub.request",
            attributes={
                "item_id": normalized_id,
                "queue_background_tasks": bool(queue_background_tasks),
            },
        ):
            return self.__stub_direct_impl(
                id=normalized_id,
                content=content,
                summary=summary,
                tags=tags,
                created_at=created_at,
                queue_background_tasks=queue_background_tasks,
            )

    def _stub_via_flow(
        self,
        *,
        id: str,
        content: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[TagMap] = None,
        created_at: Optional[str] = None,
        queue_background_tasks: bool = True,
    ) -> Item:
        """Create a stub through the flow layer so assessment policy applies."""
        if self._needs_sysdoc_migration:
            doc_coll = self._resolve_doc_collection()
            # Startup maintenance can materialize stubs before bundled state
            # docs have been migrated into the store for this Keeper. Only use
            # the flow path once the stub state note is actually present;
            # otherwise fall back to direct insertion to avoid re-entering
            # ensure_sysdocs() from bootstrap maintenance.
            if self._document_store.get(doc_coll, ".state/stub") is None:
                return self._stub_direct(
                    id=id,
                    content=content,
                    summary=summary,
                    tags=tags,
                    created_at=created_at,
                    queue_background_tasks=queue_background_tasks,
                )
        return flow_stub_item(
            self,
            id=id,
            content=content,
            summary=summary,
            tags=tags,
            created_at=created_at,
            queue_background_tasks=queue_background_tasks,
        )

    def __stub_direct_impl(
        self,
        *,
        id: str,
        content: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[TagMap] = None,
        created_at: Optional[str] = None,
        queue_background_tasks: bool = True,
    ) -> Item:
        doc_coll = self._resolve_doc_collection()
        reference_created = created_at or utc_now()
        target_summary = summary if summary is not None else (content or "")
        target_tags = dict(tags or {})
        target_tags.setdefault("_source", "auto-vivify")
        target_tags.setdefault("_created", reference_created)
        target_tags.setdefault("_updated", utc_now())

        inserted = self._document_store.insert_if_absent(
            doc_coll,
            id,
            summary=target_summary,
            tags=target_tags,
            created_at=reference_created,
        )

        result = self._document_store.get(doc_coll, id)
        if result is None:
            raise RuntimeError(f"stub insert failed unexpectedly for {id!r}")

        if inserted:
            self._context_cache.notify_write(
                id,
                old_tags={},
                new_tags=dict(result.tags),
            )
            if queue_background_tasks and not is_system_id(id):
                self._pending_queue.enqueue(
                    id,
                    doc_coll,
                    content or target_summary,
                    task_type="reindex",
                    metadata={"tags": dict(result.tags)},
                )

        return _record_to_item(result, changed=inserted)

    def __put_direct_impl(
        self,
        content: Optional[str] = None,
        *,
        uri: Optional[str] = None,
        id: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[TagMap] = None,
        created_at: Optional[str] = None,
        force: bool = False,
        queue_background_tasks: bool = True,
        capture_write_context: bool = False,
        _body_authority: str | None = None,
    ) -> Item:
        if tags:
            tags = self._validate_write_tags(tags)

        # Parts are immutable — block put() with part-like IDs
        effective_id = id or uri or ""
        if is_part_id(effective_id):
            raise ValueError(
                f"Cannot modify part directly: {effective_id!r}. "
                "Parts are managed by analyze()."
            )

        if uri is not None and (uri.startswith("file://") or uri.startswith("/")):
            file_path = Path(file_uri_to_path(uri)).resolve()
            if path_inside_markdown_mirror(self, file_path):
                raise ValueError(
                    f"Cannot put files from a synced markdown mirror root: {file_path}"
                )

        # Enforce required tags (skip for system docs with dot-prefix IDs)
        if self._config.required_tags and not is_system_id(effective_id):
            user_tags = {k: v for k, v in (tags or {}).items()
                         if not k.startswith(SYSTEM_TAG_PREFIX)} if tags else {}
            missing = [t for t in self._config.required_tags if t not in user_tags]
            if missing:
                raise ValueError(f"Required tags missing: {', '.join(missing)}")

        if uri is not None:
            # URI mode: fetch document, extract content, store
            uri = normalize_id(uri)
            # When --id is provided, use it as the document ID; otherwise the URI is the ID
            doc_id = normalize_id(id) if id else uri

            # Fast path for local files: skip expensive read if stat unchanged
            is_file_uri = uri.startswith("file://") or uri.startswith("/")
            if is_file_uri and summary is None and not force:
                try:
                    fpath = Path(file_uri_to_path(uri)).resolve()
                    st = fpath.stat()
                    doc_coll = self._resolve_doc_collection()
                    existing = self._document_store.get(doc_coll, doc_id)
                    if (existing
                            and existing.tags.get("_file_mtime_ns") == str(st.st_mtime_ns)
                            and existing.tags.get("_file_size") == str(st.st_size)):
                        # Backfill _content_type for items stored before it was tracked
                        if "_content_type" not in existing.tags:
                            ct = FileDocumentProvider.EXTENSION_TYPES.get(fpath.suffix.lower())
                            if ct:
                                self._document_store.patch_head_tags(
                                    doc_coll, doc_id, {"_content_type": ct},
                                )
                        # File stat unchanged — check if tags would also be unchanged
                        if not tags or not _user_tags_changed(
                                existing.tags,
                                {**filter_non_system_tags(existing.tags),
                                 **casefold_tags(tags)}):
                            logger.debug("File stat unchanged, skipping read for %s", doc_id)
                            return _record_to_item(existing, changed=False)
                except OSError:
                    pass  # Fall through to normal fetch

            doc = self._document_provider.fetch(uri)

            # Extract frontmatter from markdown files
            _uri_lower = uri.lower()
            _is_markdown = any(_uri_lower.endswith(ext) for ext in _MARKDOWN_EXTENSIONS)
            if _is_markdown and doc.content:
                body, fm_tags = _extract_markdown_frontmatter(doc.content)
                if fm_tags:
                    fm_tags = self._validate_tag_map(
                        fm_tags,
                        source=f"Frontmatter tags in {uri}",
                        check_constraints=False,
                    )
                if body != doc.content or fm_tags:
                    doc = Document(
                        uri=doc.uri,
                        content=body,
                        content_type=doc.content_type,
                        metadata=doc.metadata,
                        tags={**(doc.tags or {}), **fm_tags},
                    )

            # Merge provider-extracted tags with user tags (user wins on collision)
            merged_tags: dict[str, str] | None = None
            if doc.tags or tags:
                merged_tags = {}
                if doc.tags:
                    merged_tags.update(
                        self._validate_tag_map(
                            filter_non_system_tags(doc.tags),
                            source=f"Document-derived tags from {uri}",
                            check_constraints=False,
                        )
                    )
                if tags:
                    merged_tags.update(tags)

            system_tags = {"_source": "uri"}
            if doc.content_type:
                system_tags["_content_type"] = doc.content_type

            # Store file stat for fast-path change detection on next put
            if is_file_uri and doc.metadata:
                try:
                    fpath = Path(file_uri_to_path(uri)).resolve()
                    st = fpath.stat()
                    system_tags["_file_mtime_ns"] = str(st.st_mtime_ns)
                    system_tags["_file_size"] = str(st.st_size)
                except OSError:
                    pass

            # Use file birthtime as created_at for new items
            if created_at is None and is_file_uri and doc.metadata:
                birthtime = doc.metadata.get("birthtime")
                if birthtime is not None:
                    created_at = datetime.fromtimestamp(
                        birthtime, tz=timezone.utc
                    ).isoformat()

            # Email threading: if the email has a thread ID, use it as the
            # item ID.  Each message becomes a version of the thread item.
            # Use the email Date header as created_at for version ordering.
            thread_id = (doc.tags or {}).get("_thread_id")
            if thread_id and doc.content_type == "message/rfc822" and id is None:
                # Strip angle brackets from Message-ID for valid keep ID
                clean_thread_id = thread_id.strip("<>")
                doc_id = normalize_id(f"thread:{clean_thread_id}")
                # Use email Date as created_at for chronological version ordering
                email_date = (merged_tags or {}).get("date") or (doc.tags or {}).get("date")
                if email_date and created_at is None:
                    created_at = email_date
                # No special tag handling needed — replace semantics in _upsert
                # means each message's tags replace the previous, and the
                # version archive preserves per-message tags independently.

            # Store source URI as system tag when using custom ID
            if doc_id != uri:
                system_tags["_source_uri"] = uri

            result = self._upsert(
                doc_id, doc.content,
                tags=merged_tags, summary=summary,
                system_tags=system_tags,
                created_at=created_at,
                force=force,
                queue_summarize=queue_background_tasks,
                _body_authority=_body_authority,
            )

            ocr_pages = (doc.metadata or {}).get("_ocr_pages")
            doc_links = (doc.metadata or {}).get("_links")

            if capture_write_context:
                self._store_write_context(
                    result.id,
                    {
                        "content": doc.content,
                        "uri": uri,
                        "ocr_pages": list(ocr_pages or []),
                        "content_type": doc.content_type or "",
                    },
                )

            # Post-write background tasks are driven by the after-write
            # state doc — do NOT hardcode task enqueues here. See
            # _dispatch_after_write_flow() and the bundled `.state/*`
            # system notes under keep/data/system/.
            if queue_background_tasks:
                self._dispatch_after_write_flow(
                    item_id=result.id,
                    content=doc.content,
                    uri=uri,
                    content_type=doc.content_type or "",
                    tags=merged_tags,
                    summary=summary,
                    ocr_pages=ocr_pages,
                    doc_links=doc_links,
                )

            # Process email attachments as child items with edges
            attachments = (doc.metadata or {}).get("_attachments")
            if attachments:
                self._put_email_attachments(
                    parent_id=result.id,
                    attachments=attachments,
                    parent_tags=merged_tags,
                    created_at=created_at,
                    queue_background_tasks=queue_background_tasks,
                )

            return result
        else:
            # Inline mode: store content directly
            # Enforce inline length limit at the API level so all paths
            # (CLI, MCP, direct API) are bounded identically.
            is_system = is_system_id(id)
            if not is_system and len(content) > self._config.max_inline_length:
                raise ValueError(
                    f"Inline content too long ({len(content)} chars, "
                    f"max {self._config.max_inline_length}). "
                    "Use a file URI instead: keep put file:///path/to/file"
                )

            if id is None:
                # Match CLI/MCP behavior: default inline IDs are content-addressed.
                id = _text_content_id(content)
            else:
                id = normalize_id(id)

            inline_content_type = "text/markdown"
            result = self._upsert(
                id, content,
                tags=tags, summary=summary,
                system_tags={
                    "_source": "inline",
                    "_content_type": inline_content_type,
                },
                created_at=created_at,
                force=force,
                queue_summarize=queue_background_tasks,
                _body_authority=_body_authority,
            )
            if capture_write_context:
                self._store_write_context(
                    result.id,
                    {
                        "content": str(content or ""),
                        "uri": "",
                        "ocr_pages": [],
                        "content_type": inline_content_type,
                    },
                )
            # Post-write background tasks are driven by the after-write
            # state doc — do NOT hardcode task enqueues here. See
            # _dispatch_after_write_flow() and the bundled `.state/*`
            # system notes under keep/data/system/.
            if queue_background_tasks:
                self._dispatch_after_write_flow(
                    item_id=result.id,
                    content=str(content or ""),
                    content_type=inline_content_type,
                    tags=tags,
                    summary=summary,
                )
            return result

    def _put_email_attachments(
        self,
        parent_id: str,
        attachments: list[dict],
        parent_tags: dict,
        created_at: Optional[str] = None,
        queue_background_tasks: bool = True,
    ) -> list[Item]:
        """Store email attachments as child items with edges to the parent.

        Each attachment is put via the normal URI path (so it gets text
        extraction, OCR, etc.) with an ``attachment`` edge-tag pointing
        to the parent email.  The child ID uses a fragment suffix:
        ``{parent_id}#att-{N}``.

        Temp files in ~/.cache/keep/email-att/ are NOT cleaned up here
        because background tasks (OCR, describe) need the files later.
        The daemon sweeps this directory daily, deleting dirs older than 24h.
        """
        results = []

        for i, att in enumerate(attachments, 1):
            att_path = att.get("path")
            if not att_path:
                continue

            # Use MIME Content-ID as fragment if available, else att-{N}
            content_id = att.get("content_id")
            fragment = content_id if content_id else f"att-{i}"
            child_id = f"{parent_id}#{fragment}"
            filename = att.get("filename", f"attachment-{i}")

            # Inherit key context tags from the parent email
            child_tags: dict = {}
            for key in ("from", "to", "cc", "bcc", "date", "subject"):
                if key in parent_tags:
                    child_tags[key] = parent_tags[key]
            child_tags["attachment"] = parent_id
            child_tags["filename"] = filename

            try:
                result = self._put_direct(
                    uri=att_path,
                    id=child_id,
                    tags=child_tags,
                    created_at=created_at,
                    queue_background_tasks=queue_background_tasks,
                )
                results.append(result)
                logger.info(
                    "Stored email attachment %s (%s, %s)",
                    child_id, filename, att.get("content_type", ""),
                )
            except Exception as e:
                logger.warning(
                    "Failed to store email attachment %s (%s): %s",
                    child_id, filename, e,
                )

        return results

    def put(
        self,
        content: Optional[str] = None,
        *,
        uri: Optional[str] = None,
        id: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[TagMap] = None,
        created_at: Optional[str] = None,
        force: bool = False,
    ) -> Item:
        """Store content in memory via the flow host interface."""
        logger.debug(
            "Keeper.put id=%s uri=%s content=%s summary=%s tags=%d force=%s",
            id or "",
            uri or "",
            content is not None,
            summary is not None,
            len(tags or {}),
            force,
        )
        return flow_put_item(
            self,
            content=content,
            uri=uri,
            id=id,
            summary=summary,
            tags=tags,
            created_at=created_at,
            force=force,
        )

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------
    # _apply_recency_decay, _rrf_fuse, _deep_tag_follow, _deep_edge_follow,
    # _deep_follow_via_flow are provided by SearchAugmentationMixin.

    def _find_direct(
        self,
        query: Optional[str] = None,
        *,
        tags: Optional[TagMap] = None,
        similar_to: Optional[str] = None,
        stored_only: bool = False,
        limit: int = 10,
        since: Optional[str] = None,
        until: Optional[str] = None,
        include_self: bool = False,
        include_hidden: bool = False,
        deep: bool = False,
        scope: Optional[str] = None,
    ) -> list[Item]:
        """Find items by hybrid search (semantic + FTS5) or similarity to an existing note.

        When an embedding provider is configured, runs both semantic and full-text
        search and fuses results with Reciprocal Rank Fusion (RRF). Falls back to
        FTS-only when no embedding provider is available.

        Exactly one of `query` or `similar_to` must be provided.

        Args:
            query: Search query text
            tags: Optional tag filter — only return items matching all specified tags
            similar_to: Find items similar to this note ID
            stored_only: In ``similar_to`` mode, use only the stored anchor
                embedding. If the anchor is not indexed yet, skip the semantic
                fallback instead of loading an embedding provider on the read path.
            limit: Maximum results to return
            since: Only include items updated since (ISO duration like P3D, or date)
            until: Only include items updated before (ISO duration like P3D, or date)
            include_self: Include the queried item in results (only with similar_to)
            include_hidden: Include system notes (dot-prefix IDs)
            deep: Follow tags from results to discover related items
            scope: ID glob pattern to constrain results (e.g. ``file:///path/to/dir*``).
                   Search may traverse items outside the scope, but only items whose
                   base ID matches the glob are returned.
        """
        if query and similar_to:
            raise ValueError("Specify either query or similar_to, not both")
        if not query and not similar_to:
            raise ValueError("Specify either query or similar_to")

        chroma_coll = self._resolve_chroma_collection()
        doc_coll = self._resolve_doc_collection()

        # Resolve scope glob to a set of base IDs.  Search may traverse
        # items outside the scope but only scoped items are returned.
        scope_ids: Optional[set[str]] = None
        if scope:
            scope_records = self._document_store.query_by_id_glob(
                doc_coll, scope, limit=0,
            )
            scope_ids = {r.id for r in scope_records}
            if not scope_ids:
                return FindResults([])

        # Deep search needs edges (created by tag definitions with
        # _inverse).  Ensure system-doc migration has run and, if it
        # enqueued edge-backfill tasks, process them synchronously so
        # edges are available for this query.
        if deep and self._needs_sysdoc_migration:
            self.ensure_sysdocs()
            if not self._needs_sysdoc_migration:
                # Migration succeeded — drain edge-backfill tasks so
                # edges are ready for this search.
                self._flush_edge_backfill(doc_coll)

        embedding = None  # Set in semantic/similar_to branches

        # Build where clause from tags filter
        where = None
        casefolded_tags: Optional[dict] = None
        # Parents whose own tags satisfy the filter — used to let parts
        # of those parents bypass the tag filter in both the semantic
        # (Chroma) and lexical (FTS) search branches. Parts intentionally
        # don't inherit parent tags (see analyze.py), so without this
        # expansion a tag filter would never reach them.
        tag_join_base_ids: list[str] = []
        # IDs from inverse-edge tag filters (e.g. cited_by=source_id)
        inverse_edge_ids: Optional[set[str]] = None
        if tags:
            casefolded_tags = casefold_tags(tags)
            for k in casefolded_tags:
                if not k.startswith(SYSTEM_TAG_PREFIX):
                    validate_tag_key(k)

            # Separate inverse-edge filters from regular tag filters.
            # An inverse-edge filter is a tag key that matches an inverse
            # predicate (i.e., .tag/KEY exists with _inverse pointing back).
            regular_tags: dict[str, Any] = {}
            for k, v in casefolded_tags.items():
                td = self._get_tagdoc_tags(k, doc_coll)
                if td is not None and td.get("_inverse"):
                    # This key is itself an inverse predicate — resolve
                    # via the edges table instead of tag matching.
                    for source_id in tag_values(casefolded_tags, k):
                        if not source_id:
                            continue
                        found = self._document_store.find_by_inverse_edge(
                            doc_coll, k, source_id,
                        )
                        if inverse_edge_ids is None:
                            inverse_edge_ids = set(found)
                        else:
                            inverse_edge_ids &= set(found)
                else:
                    regular_tags[k] = v

            casefolded_tags = regular_tags
            where = self._build_tag_where(casefolded_tags) if casefolded_tags else None

            if where is not None:
                parent_ids_set = self._ids_matching_tags(
                    doc_coll, casefolded_tags,
                    limit=self._FIND_PART_JOIN_LIMIT,
                )
                # Drop any parent IDs that are themselves parts — we
                # only want real base IDs for the _base_id join.
                parent_ids_set = {
                    pid for pid in parent_ids_set if not is_part_id(pid)
                }
                if parent_ids_set:
                    tag_join_base_ids = sorted(parent_ids_set)
                    # Semantic branch: $or expansion on the where clause.
                    where = {
                        "$or": [
                            where,
                            {"_base_id": {"$in": tag_join_base_ids}},
                        ],
                    }

        if similar_to:
            # Similar-to mode: resolve the anchor from the canonical
            # document store first. Fresh notes can exist there before
            # the embedding index catches up, so ChromaDB alone is not
            # a safe existence check.
            similar_to = normalize_id(similar_to)
            anchor = self._document_store.get(doc_coll, similar_to)
            item = self._store.get(chroma_coll, similar_to)
            if anchor is None and item is None:
                raise KeyError(f"Item not found: {similar_to}")

            embedding = self._store.get_embedding(chroma_coll, similar_to)
            if embedding is None:
                if stored_only:
                    if casefolded_tags or since or until:
                        items = self.list_items(
                            tags=casefolded_tags if casefolded_tags else None,
                            since=since,
                            until=until,
                            include_hidden=include_hidden,
                            limit=limit,
                        )
                        if not include_self:
                            items = [candidate for candidate in items if candidate.id != similar_to]
                        items = self._apply_recency_decay(items)
                    else:
                        items = []
                else:
                    anchor_summary = (
                        anchor.summary
                        if anchor is not None
                        else (item.summary if item is not None else "")
                    )
                    with _get_tracer("keeper").start_as_current_span("embed"):
                        embedding = self._get_embedding_provider().embed(
                            anchor_summary,
                            task=EmbedTask.QUERY,
                        )
            if embedding is not None:
                actual_limit = (limit + 1 if not include_self else limit) * 3
                if deep:
                    actual_limit = max(actual_limit, 30)
                if scope_ids is not None:
                    actual_limit = max(actual_limit, len(scope_ids))
                with _get_tracer("keeper").start_as_current_span("chroma.query"):
                    results = self._store.query_embedding(
                        chroma_coll, embedding, limit=actual_limit, where=where,
                    )

                if not include_self:
                    results = [r for r in results if r.id != similar_to]

                items = [r.to_item() for r in results]
                items = self._apply_recency_decay(items)

        elif self._config.embedding is not None:
            # Hybrid search: semantic + FTS5, fused with RRF.
            # Each list over-fetches independently so RRF can discover
            # items that rank well in one signal but poorly in the other.
            with _get_tracer("keeper").start_as_current_span("embed"):
                embedding = self._get_embedding_provider().embed(query, task=EmbedTask.QUERY)
            sem_fetch = max(limit * 10, 200)
            fts_fetch = max(limit * 10, 100)
            if deep:
                sem_fetch = max(sem_fetch, 30)
            if scope_ids is not None:
                sem_fetch = max(sem_fetch, len(scope_ids))

            with _get_tracer("keeper").start_as_current_span("chroma.query"):
                sem_results = self._store.query_embedding(
                    chroma_coll, embedding, limit=sem_fetch, where=where,
                )
            sem_items = [r.to_item() for r in sem_results]
            sem_items = self._apply_recency_decay(sem_items)

            with _get_tracer("keeper").start_as_current_span("fts.query"):
                if scope_ids is not None:
                    fts_rows = self._document_store.query_fts_scoped(
                        doc_coll, query, list(scope_ids),
                        limit=fts_fetch, tags=casefolded_tags,
                        part_tag_join_base_ids=tag_join_base_ids or None,
                    )
                else:
                    fts_rows = self._document_store.query_fts(
                        doc_coll, query, limit=fts_fetch, tags=casefolded_tags,
                        part_tag_join_base_ids=tag_join_base_ids or None,
                    )
            fts_items = [Item(id=r[0], summary=r[1]) for r in fts_rows]

            if fts_items:
                with _get_tracer("keeper").start_as_current_span("rrf"):
                    items = self._rrf_fuse(sem_items, fts_items)
            else:
                # FTS unavailable or no matches — use semantic results as-is
                items = sem_items

        else:
            # No embedding provider — FTS only
            fetch_limit = limit * 3
            with _get_tracer("keeper").start_as_current_span("fts.query"):
                if scope_ids is not None:
                    fts_rows = self._document_store.query_fts_scoped(
                        doc_coll, query, list(scope_ids),
                        limit=fetch_limit, tags=casefolded_tags,
                        part_tag_join_base_ids=tag_join_base_ids or None,
                    )
                else:
                    fts_rows = self._document_store.query_fts(
                        doc_coll, query, limit=fetch_limit, tags=casefolded_tags,
                        part_tag_join_base_ids=tag_join_base_ids or None,
                    )
            items = [Item(id=r[0], summary=r[1]) for r in fts_rows]

        # Hydrate search hits from canonical SQLite tags so user tags remain
        # available even when Chroma metadata stores marker fields only.
        with _get_tracer("keeper").start_as_current_span("hydrate"):
            hydrated: list[Item] = []
            for item in items:
                base_id = item.tags.get(
                    "_base_id",
                    item.id.split("@")[0] if "@" in item.id else item.id,
                )
                head = self._document_store.get(doc_coll, base_id)
                if head is None:
                    hydrated.append(item)
                    continue
                head_item = _record_to_item(head, score=item.score)
                merged_tags = dict(head_item.tags)
                merged_tags.update(item.tags or {})
                hydrated.append(Item(
                    id=item.id,
                    summary=item.summary or head_item.summary,
                    tags=merged_tags,
                    score=item.score,
                ))
            items = hydrated

        # Scope filter: keep only items whose base ID is in the scope set.
        # Applied before deep follow so traversal can still discover edges
        # through out-of-scope items; deep group results are filtered later.
        if scope_ids is not None:
            items = [i for i in items
                     if (i.tags.get("_base_id") or
                         (i.id.split("@")[0] if "@" in i.id else i.id))
                     in scope_ids]

        # Filter by inverse-edge results (e.g. find -t cited_by=source)
        if inverse_edge_ids is not None:
            if not items:
                # No semantic/FTS results — return inverse-edge targets directly
                for tid in sorted(inverse_edge_ids)[:limit]:
                    doc = self._document_store.get(doc_coll, tid)
                    if doc is not None:
                        items.append(Item(id=doc.id, summary=doc.summary, tags=dict(doc.tags)))
            else:
                base_ids = inverse_edge_ids
                items = [i for i in items
                         if (i.tags.get("_base_id") or
                             (i.id.split("@")[0] if "@" in i.id else i.id))
                         in base_ids]

        # Deep follow: prefer edge-following when edges exist in the store,
        # fall back to tag-following for stores without edges.
        deep_groups: dict[str, list[Item]] = {}
        injected_entity_ids: set[str] = set()
        if deep and embedding is not None:
            with _get_tracer("keeper").start_as_current_span("deep.follow"):
                if self._document_store.has_edges(doc_coll):
                    # For similar_to mode, use the anchor item's summary as FTS query
                    deep_query = query if query else ""

                    # Entity injection: if query mentions known edge targets
                    # by name, inject them as synthetic primaries so their
                    # edges get traversed even if they didn't rank in search.
                    deep_items = list(items)
                    entity_hits: list[str] = []
                    if deep_query:
                        # Only consider the top display window as "already present"
                        _ENTITY_WINDOW = 10
                        top_ids = {(i.id.split("@")[0] if "@" in i.id else i.id)
                                   for i in items[:_ENTITY_WINDOW]}
                        entity_hits = self._document_store.find_edge_targets(
                            doc_coll, deep_query)
                        # Insert at front so entities are within top_k window
                        inject_pos = 0
                        for eid in entity_hits:
                            if eid not in top_ids:
                                deep_items.insert(inject_pos, Item(id=eid, summary="", tags={}, score=0.5))
                                injected_entity_ids.add(eid)
                                inject_pos += 1
                        # Remove matched entity phrases from deep FTS query so
                        # activity/content terms dominate deep evidence. This
                        # avoids over-blocking standalone content words.
                        if entity_hits:
                            cleaned = deep_query
                            for eid in sorted(entity_hits, key=len, reverse=True):
                                parts = re.findall(r"[a-z0-9]+", eid.lower())
                                if not parts:
                                    continue
                                # Match phrase tokens with flexible separators.
                                pattern = r"\b" + r"[^a-z0-9]+".join(
                                    re.escape(tok) for tok in parts
                                ) + r"\b"
                                cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
                            kept = re.findall(r"[a-z0-9]+", cleaned.lower())
                            if kept:
                                deep_query = " ".join(kept)

                    # Exclude only a few top primaries from deep results.
                    # With deep_primary_cap, most primaries get dropped so
                    # they should remain available as deep sub-items.
                    # Entities are never excluded (they're the group hubs).
                    _DEEP_EXCLUDE = 3
                    exclude = set()
                    for i in items[:_DEEP_EXCLUDE]:
                        pid = i.id.split("@")[0] if "@" in i.id else i.id
                        if pid not in injected_entity_ids:
                            exclude.add(pid)
                    deep_groups = self._deep_edge_follow(
                        deep_items, chroma_coll, doc_coll,
                        query=deep_query,
                        embedding=embedding,
                        exclude_ids=exclude,
                    )
                    # Inject entities that produced deep groups into the
                    # primary list so they appear as result items and
                    # pass the final_ids filter in the remapping step.
                    if deep_groups:
                        item_ids = {(i.id.split("@")[0] if "@" in i.id else i.id)
                                    for i in items}
                        for gk in deep_groups:
                            if gk not in item_ids:
                                head = self._document_store.get(doc_coll, gk)
                                if head:
                                    items.append(_record_to_item(head, score=0.5))
                                else:
                                    items.append(Item(
                                        id=gk, summary="", tags={}, score=0.5,
                                    ))
                else:
                    # For similar_to mode, use the anchor item's summary
                    # as the flow query since the find action requires one.
                    flow_query = query or ""
                    if not flow_query and similar_to:
                        anchor = self._document_store.get(doc_coll, similar_to)
                        if anchor:
                            flow_query = getattr(anchor, "summary", "") or ""
                    deep_groups = self._deep_follow_via_flow(
                        query=flow_query,
                        limit=limit,
                        embedding=embedding,
                    )

        # Apply common filters
        if since is not None or until is not None:
            # Enrich _updated_date from SQLite for items missing it
            # (e.g. version refs whose ChromaDB metadata lacks the tag)
            all_items = list(items)
            for g in deep_groups.values():
                all_items.extend(g)
            _enrich_updated_date(all_items, self._document_store, doc_coll)
            items = _filter_by_date(items, since=since, until=until)
            deep_groups = {pid: _filter_by_date(g, since=since, until=until)
                          for pid, g in deep_groups.items()}
        if not include_hidden:
            items = [i for i in items if not _is_hidden(i)]
            deep_groups = {pid: [i for i in g if not _is_hidden(i)]
                          for pid, g in deep_groups.items()}
        deep_groups = {pid: g for pid, g in deep_groups.items() if g}

        # Scope filter for deep groups: keep only scoped items within groups.
        if scope_ids is not None and deep_groups:
            deep_groups = {
                pid: [i for i in g
                      if (i.tags.get("_base_id") or
                          (i.id.split("@")[0] if "@" in i.id else i.id))
                      in scope_ids]
                for pid, g in deep_groups.items()
            }
            deep_groups = {pid: g for pid, g in deep_groups.items() if g}

        # Part-to-parent uplift: replace part hits with their parent
        # documents, carrying _focus_part so the formatter can window
        # the parts manifest around the hit.  Dedup: keep the highest-
        # scoring part when multiple parts of the same parent match.
        uplifted: list[Item] = []
        seen_parents: dict[str, int] = {}  # parent_id -> index in uplifted
        for item in items:
            if is_part_id(item.id):
                parent_id = item.tags.get("_base_id", item.id.split("@")[0])
                part_num = item.tags.get("_part_num")
                # FTS-originated items lack tags — parse part_num from ID
                if not part_num:
                    suffix = item.id.rsplit("@p", 1)
                    if len(suffix) == 2 and suffix[1].isdigit():
                        part_num = suffix[1]
                if parent_id in seen_parents:
                    # Already have this parent — skip (first hit had higher score)
                    continue
                parent_doc = self._document_store.get(doc_coll, parent_id)
                if parent_doc:
                    parent_item = _record_to_item(parent_doc)
                    parent_tags = dict(parent_item.tags)
                    if part_num:
                        parent_tags["_focus_part"] = part_num
                        parent_tags["_focus_summary"] = item.summary
                        # Propagate line range tags if present
                        if "_start_line" in item.tags:
                            parent_tags["_focus_start_line"] = item.tags["_start_line"]
                        if "_end_line" in item.tags:
                            parent_tags["_focus_end_line"] = item.tags["_end_line"]
                    uplifted.append(Item(
                        id=parent_id, summary=parent_item.summary,
                        tags=parent_tags, score=item.score,
                    ))
                    seen_parents[parent_id] = len(uplifted) - 1
                else:
                    uplifted.append(item)  # Parent gone — keep raw part
            elif "@v" in item.id:
                # Version hit — uplift to parent, preserving hit version
                parent_id = item.id.rsplit("@v", 1)[0]
                version_str = item.id.rsplit("@v", 1)[1]
                if parent_id in seen_parents:
                    continue
                parent_doc = self._document_store.get(doc_coll, parent_id)
                if parent_doc:
                    parent_item = _record_to_item(parent_doc)
                    parent_tags = dict(parent_item.tags)
                    if version_str.isdigit():
                        parent_tags["_focus_version"] = version_str
                        parent_tags["_focus_summary"] = item.summary
                    uplifted.append(Item(
                        id=parent_id, summary=parent_item.summary,
                        tags=parent_tags, score=item.score,
                    ))
                    seen_parents[parent_id] = len(uplifted) - 1
                else:
                    uplifted.append(item)  # Parent gone — keep raw version
            else:
                # Regular document — dedup against uplifted parents
                if item.id in seen_parents:
                    continue
                uplifted.append(item)
                seen_parents[item.id] = len(uplifted) - 1
        items = uplifted

        # Keyword passage fallback: for file-backed items that matched
        # semantically (no _focus_part from FTS), find the best passage
        # by keyword overlap in the source file. Only runs for URI items
        # when a query string is available.
        if query:
            for idx, item in enumerate(items):
                if item.tags.get("_focus_part"):
                    continue  # already has a part match
                if item.tags.get("_source") != "uri":
                    continue  # not file-backed
                if not item.id.startswith("file://"):
                    continue

                fpath = Path(file_uri_to_path(item.id))
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                passage = _find_best_passage(content, query)
                if passage:
                    enriched_tags = dict(item.tags)
                    enriched_tags["_focus_summary"] = passage["snippet"]
                    enriched_tags["_focus_start_line"] = passage["start_line"]
                    enriched_tags["_focus_end_line"] = passage["end_line"]
                    items[idx] = Item(
                        id=item.id, summary=item.summary,
                        tags=enriched_tags, score=item.score,
                    )

        # Remap deep_groups: uplift keys AND items (parts → parents), dedup.
        # Only exclude deep items that appear in the top primary results
        # the user will actually see — not the full over-fetched pool
        # (which may swallow all deep candidates when fetch_limit >> display).
        _DEEP_EXCLUDE_WINDOW = 10
        if deep_groups:
            final_ids = set()
            for i in items[:min(limit, _DEEP_EXCLUDE_WINDOW)]:
                final_ids.add(i.id)
                # Include parent ID so version hits match deep group keys
                if "@" in i.id:
                    final_ids.add(i.id.split("@")[0])
            # Ensure injected entities pass the filter even if they're
            # beyond the limit window (they were appended, not ranked)
            final_ids.update(injected_entity_ids & deep_groups.keys())
            remapped: dict[str, list[Item]] = {}
            remap_keys: dict[str, set[str]] = {}
            for source_id, group in deep_groups.items():
                # Uplift the group key (source primary) to parent
                key_parent = source_id.split("@")[0] if "@" in source_id else source_id
                is_entity_group = key_parent in injected_entity_ids
                bucket = remapped.setdefault(key_parent, [])
                seen_keys = remap_keys.setdefault(key_parent, set())
                # Uplift each deep item to its parent too
                for item in group:
                    deep_parent_id = item.id.split("@")[0] if "@" in item.id else item.id
                    # Skip if this deep item IS a final primary result —
                    # UNLESS this is an entity group where overlap is
                    # intentional (deep_primary_cap handles dedup at render)
                    if deep_parent_id in final_ids and not is_entity_group:
                        continue
                    anchor_key = (
                        item.tags.get("_anchor_id")
                        or f"{item.id}|{item.tags.get('_focus_version', '')}|"
                           f"{item.tags.get('_focus_part', '')}|"
                           f"{item.tags.get('_focus_summary', '')}"
                    )
                    if anchor_key in seen_keys:
                        continue
                    seen_keys.add(anchor_key)
                    if "@" in item.id:
                        parent_doc = self._document_store.get(doc_coll, deep_parent_id)
                        if parent_doc:
                            pi = _record_to_item(parent_doc)
                            tags = dict(pi.tags)
                            for k, v in (item.tags or {}).items():
                                if k.startswith("_focus_") or k.startswith("_anchor_") or k == "_lane":
                                    tags[k] = v
                            bucket.append(
                                Item(
                                    id=item.id,
                                    summary=pi.summary,
                                    tags=tags,
                                    score=item.score,
                                )
                            )
                        else:
                            bucket.append(item)
                    else:
                        bucket.append(item)
            deep_groups = {
                pid: sorted(bucket, key=lambda x: x.score or 0, reverse=True)
                for pid, bucket in remapped.items()
                if pid in final_ids
            }

        final = items[:limit]
        # Promote injected entities into the final results so their
        # deep groups can be rendered.  Replace the lowest-scored
        # non-entity item to stay within the limit.
        if deep_groups and injected_entity_ids:
            final_base_ids = {(i.id.split("@")[0] if "@" in i.id else i.id)
                              for i in final}
            for eid in injected_entity_ids:
                if eid in deep_groups and eid not in final_base_ids:
                    entity_item = next(
                        (i for i in items if i.id == eid), None)
                    if entity_item and final:
                        # Mark as entity so renderer can prioritize
                        entity_tags = dict(entity_item.tags) if entity_item.tags else {}
                        entity_tags["_entity"] = "true"
                        entity_item = Item(
                            id=entity_item.id, summary=entity_item.summary,
                            tags=entity_tags, score=entity_item.score,
                        )
                        # Replace the lowest-scored item that has no
                        # deep group (preserve items that do)
                        worst_idx = None
                        worst_score = float('inf')
                        for idx, fi in enumerate(final):
                            fi_base = fi.id.split("@")[0] if "@" in fi.id else fi.id
                            if fi_base not in deep_groups and fi.id not in deep_groups:
                                if (fi.score or 0) < worst_score:
                                    worst_score = fi.score or 0
                                    worst_idx = idx
                        if worst_idx is not None:
                            final[worst_idx] = entity_item
                        # else: all items have deep groups — skip to stay within limit
        # Enrich tags from SQLite (ChromaDB stores casefolded values;
        # SQLite has the canonical original-case values for display)
        def _enrich_from_sqlite(items_to_enrich):
            enriched = []
            for item in items_to_enrich:
                doc = self._document_store.get(doc_coll, item.id)
                if not doc and "@" in item.id:
                    base_id = item.id.split("@")[0]
                    doc = self._document_store.get(doc_coll, base_id)
                if doc:
                    enriched_item = _record_to_item(doc, score=item.score)
                    tags = enriched_item.tags
                    focus = item.tags.get("_focus_part")
                    if focus:
                        tags["_focus_part"] = focus
                    focus_summary = item.tags.get("_focus_summary")
                    if focus_summary:
                        tags["_focus_summary"] = focus_summary
                    focus_version = item.tags.get("_focus_version")
                    if focus_version:
                        tags["_focus_version"] = focus_version
                    focus_start = item.tags.get("_focus_start_line")
                    if focus_start:
                        tags["_focus_start_line"] = focus_start
                    focus_end = item.tags.get("_focus_end_line")
                    if focus_end:
                        tags["_focus_end_line"] = focus_end
                    entity_marker = item.tags.get("_entity")
                    if entity_marker:
                        tags["_entity"] = entity_marker
                    enriched.append(Item(
                        id=item.id, summary=item.summary,
                        tags=tags, score=item.score,
                    ))
                else:
                    enriched.append(item)
            return enriched

        with _get_tracer("keeper").start_as_current_span("enrich"):
            if final:
                self._document_store.touch_many(doc_coll, [i.id for i in final])
                final = _enrich_from_sqlite(final)
            # Enrich deep group items too (same casefolded-tag issue)
            if deep_groups:
                deep_groups = {
                    pid: _enrich_from_sqlite(group)
                    for pid, group in deep_groups.items()
                }
        return FindResults(final, deep_groups=deep_groups)

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
        scope: Optional[str] = None,
    ) -> list[Item]:
        """Find items via the flow host interface."""
        logger.debug(
            "Keeper.find query=%s tags=%d similar_to=%s limit=%s deep=%s scope=%s",
            bool(query),
            len(tags or {}),
            similar_to or "",
            limit,
            deep,
            scope or "",
        )
        return flow_find_items(
            self,
            query,
            tags=tags,
            similar_to=similar_to,
            limit=limit,
            since=since,
            until=until,
            include_self=include_self,
            include_hidden=include_hidden,
            deep=deep,
            scope=scope,
        )

    # -------------------------------------------------------------------------
    # State-doc flow binding mappers for get_context
    # -------------------------------------------------------------------------

    def _resolve_edge_refs(self, item: "Item", item_id: str) -> dict[str, list["EdgeRef"]]:
        """Resolve structural edge references (explicit + inverse) for display.

        Uses direct database queries rather than the generic traverse action,
        because edges require inverse-edge table lookups and explicit edge-tag
        resolution — operations the traverse action doesn't support.
        """
        doc_coll = self._resolve_doc_collection()
        edge_ref_index: dict[str, dict[str, EdgeRef]] = {}

        def _upsert(key: str, ref: EdgeRef, *, prefer_new: bool = False) -> None:
            refs_for_key = edge_ref_index.setdefault(key, {})
            existing = refs_for_key.get(ref.source_id)
            if existing is None:
                refs_for_key[ref.source_id] = ref
                return
            winner_date = (ref.date if prefer_new else existing.date) or \
                          (existing.date if prefer_new else ref.date)
            winner_summary = (ref.summary if prefer_new else existing.summary) or \
                             (existing.summary if prefer_new else ref.summary)
            refs_for_key[ref.source_id] = EdgeRef(
                source_id=ref.source_id, date=winner_date, summary=winner_summary,
            )

        # Explicit edge refs from current item's edge-tag values
        explicit_targets_by_key: dict[str, list[str]] = {}
        user_keys = [
            k for k in item.tags
            if not k.startswith(SYSTEM_TAG_PREFIX) and tag_values(item.tags, k)
        ]
        if user_keys:
            tagdoc_ids = [f".tag/{k}" for k in user_keys]
            tagdocs = self._document_store.get_many(doc_coll, tagdoc_ids)
            for key in user_keys:
                td = tagdocs.get(f".tag/{key}")
                if td is None or not td.tags.get("_inverse"):
                    continue
                seen_targets: set[str] = set()
                resolved_targets: list[str] = []
                for raw_target in tag_values(item.tags, key):
                    try:
                        target_id = normalize_id(parse_ref(raw_target)[0])
                    except ValueError:
                        continue
                    if target_id.startswith(".") or target_id in seen_targets:
                        continue
                    seen_targets.add(target_id)
                    resolved_targets.append(target_id)
                if resolved_targets:
                    explicit_targets_by_key[key] = resolved_targets

        if explicit_targets_by_key:
            target_ids = {
                tid for tids in explicit_targets_by_key.values() for tid in tids
            }
            target_docs = self._document_store.get_many(doc_coll, list(target_ids))
            for key, target_ids_for_key in explicit_targets_by_key.items():
                for target_id in target_ids_for_key:
                    doc = target_docs.get(target_id)
                    created = ""
                    summary = ""
                    if doc is not None:
                        created = local_date(
                            doc.tags.get("_created") or doc.tags.get("_updated") or ""
                        )
                        summary = note_display_name(doc.tags, doc.summary)
                    _upsert(key, EdgeRef(
                        source_id=target_id, date=created, summary=summary,
                    ))

        raw_edges = self._document_store.get_inverse_edges(doc_coll, item_id)
        if raw_edges:
            source_ids = list({src for _, src, _ in raw_edges})
            source_docs = self._document_store.get_many(doc_coll, source_ids)
            for inverse, source_id, created in raw_edges:
                doc = source_docs.get(source_id)
                _upsert(inverse, EdgeRef(
                    source_id=source_id,
                    date=local_date(created),
                    summary=note_display_name(doc.tags, doc.summary) if doc else "",
                ), prefer_new=True)

        return {
            key: list(refs_by_source.values())
            for key, refs_by_source in edge_ref_index.items()
        }


    # get_context, resolve_version_offset, render_prompt, list_prompts,
    # get_similar_for_display, get_version_offset, resolve_meta,
    # resolve_inline_meta, _resolve_meta_queries, _rank_by_relevance,
    # _map_flow_similar, _map_flow_meta, _map_flow_parts
    # are provided by ContextResolutionMixin.

    def list_tags(
        self,
        key: Optional[str] = None,
    ) -> list[str]:
        """List distinct tag keys or values.

        Args:
            key: If provided, list distinct values for this key.
                 If None, list distinct tag keys.

        Returns:
            Sorted list of distinct keys or values
        """
        if key is not None:
            validate_tag_key(key)
        doc_coll = self._resolve_doc_collection()

        if key is None:
            return self._document_store.list_distinct_tag_keys(doc_coll)
        else:
            return self._document_store.list_distinct_tag_values(doc_coll, key)
    
    # -------------------------------------------------------------------------
    # Direct Access
    # -------------------------------------------------------------------------
    
    def _get_direct(self, id: str) -> Optional[Item]:
        """Retrieve a specific item by ID.

        Reads from document store (canonical), falls back to vector store for legacy data.
        Touches accessed_at on successful retrieval.
        """
        id = normalize_id(id)
        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()

        with _get_tracer("keeper").start_as_current_span(
            "get",
            attributes={"item_id": id},
        ) as span:
            # Try document store first (canonical)
            try:
                with _get_tracer("keeper").start_as_current_span("doc_store.get"):
                    doc_record = self._document_store.get(doc_coll, id)
            except Exception as e:
                logger.warning("DocumentStore.get(%s) failed: %s", id, e)
                if self._is_local and is_malformed_db_error(e):
                    span.set_attribute("recovered", True)
                    # SQLite-specific recovery — only for local backends
                    recovered = False
                    if hasattr(self._document_store, '_try_runtime_recover'):
                        recovered = bool(self._document_store._try_runtime_recover())
                    if not recovered:
                        # Do not hide a malformed canonical store behind the
                        # Chroma fallback.  Serving stale vector data after
                        # SQLite malformation makes daemon state harder to
                        # diagnose and can leave request threads piling up.
                        raise
                    # Retry once after recovery
                    try:
                        with _get_tracer("keeper").start_as_current_span("doc_store.get.retry"):
                            doc_record = self._document_store.get(doc_coll, id)
                    except Exception:
                        logger.exception("DocumentStore.get(%s) failed after recovery", id)
                        raise
                else:
                    doc_record = None
            if doc_record:
                self._document_store.touch(doc_coll, id)
                span.set_attribute("source", "sqlite")
                span.set_attribute("found", True)
                return _record_to_item(doc_record)

            # Fall back to ChromaDB for legacy data
            with _get_tracer("keeper").start_as_current_span("chroma.get"):
                result = self._store.get(chroma_coll, id)
            span.set_attribute("source", "chroma_fallback")
            span.set_attribute("found", result is not None)
            if result is None:
                return None
            return result.to_item()

    def get(self, id: str) -> Optional[Item]:
        """Retrieve a specific item via the flow host interface."""
        logger.debug("Keeper.get id=%s", id)
        return flow_get_item(self, id)

    def peek(self, id: str) -> Optional[Item]:
        """Read an item without updating accessed_at.

        Used for cache hydration where context items shouldn't count
        as direct accesses.
        """
        id = normalize_id(id)
        doc_coll = self._resolve_doc_collection()
        doc_record = self._document_store.get(doc_coll, id)
        if doc_record:
            return _record_to_item(doc_record)
        # Fall back to ChromaDB for legacy data
        result = self._store.get(self._resolve_chroma_collection(), id)
        if result is None:
            return None
        return result.to_item()

    def get_version(
        self,
        id: str,
        offset: int = 0,
    ) -> Optional[Item]:
        """Get a specific version of a document by public selector.

        Selector semantics:
        - 0 = current version
        - 1 = previous version
        - 2 = two versions ago
        - ...
        - -1 = oldest archived version
        - -2 = second oldest archived version
        - ...

        Args:
            id: Document identifier
            offset: Public version selector

        Returns:
            Item if found, None if version doesn't exist
        """
        id = normalize_id(id)
        resolved_offset = self.resolve_version_offset(id, offset)
        if resolved_offset is None:
            return None
        doc_coll = self._resolve_doc_collection()

        if resolved_offset == 0:
            # Current version
            return self.get(id)

        # Get archived version
        version_info = self._document_store.get_version(doc_coll, id, resolved_offset)
        if version_info is None:
            return None

        return Item(
            id=id,
            summary=version_info.summary,
            tags=version_info.tags,
        )

    def list_versions(
        self,
        id: str,
        limit: int = 10,
    ) -> list[VersionInfo]:
        """List version history for a document.

        Returns versions in reverse chronological order (newest archived first).
        Does not include the current version.

        Args:
            id: Document identifier
            limit: Maximum versions to return

        Returns:
            List of VersionInfo, newest archived first
        """
        id = normalize_id(id)
        doc_coll = self._resolve_doc_collection()
        return self._document_store.list_versions(doc_coll, id, limit)

    def list_versions_around(
        self,
        id: str,
        version: int,
        radius: int = 2,
    ) -> list[VersionInfo]:
        """Return versions within `radius` of `version`, in chronological order.

        Useful for showing surrounding context when a specific version was
        matched during search.
        """
        id = normalize_id(id)
        doc_coll = self._resolve_doc_collection()
        return self._document_store.list_versions_around(
            doc_coll, id, version, radius,
        )

    def get_version_nav(
        self,
        id: str,
        current_version: Optional[int] = None,
        limit: int = 3,
    ) -> dict[str, list[VersionInfo]]:
        """Get version navigation info (prev/next) for display.

        Args:
            id: Document identifier
            current_version: The version being viewed (None = current/live version)
            limit: Max previous versions to return when viewing current

        Returns:
            Dict with 'prev' and optionally 'next' lists of VersionInfo.
        """
        doc_coll = self._resolve_doc_collection()
        return self._document_store.get_version_nav(doc_coll, id, current_version, limit)

    def exists(self, id: str) -> bool:
        """Check if an item exists in the store."""
        id = normalize_id(id)
        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()
        # Check document store first, then ChromaDB
        return self._document_store.exists(doc_coll, id) or self._store.exists(chroma_coll, id)
    
    def _delete_direct(
        self,
        id: str,
        *,
        delete_versions: bool = True,
    ) -> bool:
        """Delete an item from both stores.

        Args:
            id: Document identifier
            delete_versions: If True, also delete version history

        Returns:
            True if item existed and was deleted.
        """
        id = normalize_id(id)
        if is_part_id(id):
            raise ValueError(
                f"Cannot delete part directly: {id!r}. "
                "Use analyze() to replace parts, or delete the parent document."
            )
        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()
        existing = self._get_direct(id)
        old_tags = dict(existing.tags) if existing is not None else None

        # Delete from both stores (including versions)
        doc_deleted = self._document_store.delete(doc_coll, id, delete_versions=delete_versions)
        chroma_deleted = self._store.delete(chroma_coll, id, delete_versions=delete_versions)
        # Clean up edges (both directions)
        self._document_store.delete_edges_for_source(doc_coll, id)
        self._document_store.delete_edges_for_target(doc_coll, id)
        self._document_store.delete_version_edges_for_source(doc_coll, id)
        self._document_store.delete_version_edges_for_target(doc_coll, id)
        self._context_cache.notify_delete(id, old_tags=old_tags)
        return doc_deleted or chroma_deleted

    def delete(
        self,
        id: str,
        *,
        delete_versions: bool = True,
    ) -> bool:
        """Delete an item via the flow host interface."""
        return flow_delete_item(self, id, delete_versions=delete_versions)

    def revert(self, id: str) -> Optional[Item]:
        """Revert to the previous version, or delete if no versions exist.

        Returns the restored item, or None if the item was fully deleted.
        """
        id = normalize_id(id)
        if is_part_id(id):
            raise ValueError(
                f"Cannot revert part directly: {id!r}. "
                "Use analyze() to replace parts, or delete the parent document."
            )
        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()

        max_ver = self._document_store.max_version(doc_coll, id)

        if max_ver == 0:
            # No history — full delete
            self.delete(id)
            return None

        # Get the versioned ChromaDB ID we need to promote
        versioned_chroma_id = f"{id}@v{max_ver}"

        # Get the archived embedding from ChromaDB
        archived_embedding = self._store.get_embedding(chroma_coll, versioned_chroma_id)

        # Restore in DocumentStore (promotes latest version to current)
        restored = self._document_store.restore_latest_version(doc_coll, id)

        if restored is None:
            # Shouldn't happen given version_count > 0, but be safe
            self.delete(id)
            return None

        # Update ChromaDB: replace current with restored version's data
        if archived_embedding:
            self._store.upsert(
                collection=chroma_coll, id=id,
                embedding=archived_embedding,
                summary=restored.summary,
                tags=casefold_tags_for_index(restored.tags),
            )

        # Delete the versioned entry from ChromaDB
        self._store.delete_entries(chroma_coll, [versioned_chroma_id])

        # Clean up stale parts (structural decomposition of old content)
        self._store.delete_parts(chroma_coll, id)
        self._document_store.delete_parts(doc_coll, id)

        return self.get(id)

    def delete_version(self, id: str, offset: int) -> bool:
        """Delete a specific archived version by public selector.

        Selector semantics match get_version:
        - 1=previous, 2=two ago, ...
        - -1=oldest archived, -2=second oldest archived, ...
        - 0=current (not allowed here; use revert()).

        Returns True if the version was found and deleted.
        """
        id = normalize_id(id)
        resolved_offset = self.resolve_version_offset(id, offset)
        if resolved_offset is None:
            return False
        if resolved_offset < 1:
            raise ValueError("Use revert() to delete the current version (offset 0)")
        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()

        # Resolve offset → internal version number
        version_info = self._document_store.get_version(doc_coll, id, resolved_offset)
        if version_info is None:
            return False

        # Delete from SQLite
        self._document_store.delete_version(doc_coll, id, version_info.version)

        # Delete from ChromaDB
        versioned_chroma_id = f"{id}@v{version_info.version}"
        self._store.delete_entries(chroma_coll, [versioned_chroma_id])

        return True

    # -------------------------------------------------------------------------
    # Current Working Context (Now)
    # -------------------------------------------------------------------------

    def _get_now_direct(self, *, scope: Optional[str] = None) -> Item:
        """Get the current working intentions.

        A singleton document representing what you're currently working on.
        If it doesn't exist, creates one with default content and tags from
        the bundled system now.md file.

        Args:
            scope: Optional scope for multi-user isolation (e.g. user ID).
                   When set, uses ``now:{scope}`` instead of the singleton ``now``.

        Returns:
            The current intentions Item (never None - auto-creates if missing)
        """
        doc_id = f"now:{scope}" if scope else NOWDOC_ID
        item = self._get_direct(doc_id)
        if item is None:
            if scope:
                # Scoped now: initialize with minimal content
                item = self._set_now_direct(f"# Now ({scope})\n\nWorking context.", scope=scope)
            else:
                # Singleton now: use bundled system doc
                try:
                    default_content, default_tags = _load_frontmatter(SYSTEM_DOC_DIR / "now.md")
                except FileNotFoundError:
                    default_content = "# Now\n\nYour working context."
                    default_tags = {}
                item = self._set_now_direct(default_content, tags=default_tags)
        return item

    def get_now(self, *, scope: Optional[str] = None) -> Item:
        """Get the current working intentions via the flow host interface."""
        return flow_get_now_item(self, scope=scope)

    def _set_now_direct(
        self,
        content: str,
        *,
        scope: Optional[str] = None,
        tags: Optional[TagMap] = None,
    ) -> Item:
        """Set the current working intentions.

        Updates the singleton intentions with new content. Uses put()
        internally with the fixed NOWDOC_ID.

        Args:
            content: New content for the current intentions
            scope: Optional scope for multi-user isolation (e.g. user ID).
                   When set, uses ``now:{scope}`` and auto-tags with ``user={scope}``.
            tags: Optional additional tags to apply

        Returns:
            The updated context Item
        """
        doc_id = f"now:{scope}" if scope else NOWDOC_ID
        merged_tags = dict(tags or {})
        if scope:
            merged_tags.setdefault("user", scope)
        return self._put_direct(content, id=doc_id, tags=merged_tags or None)

    def set_now(
        self,
        content: str,
        *,
        scope: Optional[str] = None,
        tags: Optional[TagMap] = None,
    ) -> Item:
        """Set the current working intentions via the flow host interface."""
        return flow_set_now_item(self, content, scope=scope, tags=tags)

    def move(
        self,
        name: str,
        *,
        source_id: str = NOWDOC_ID,
        tags: Optional[TagMap] = None,
        only_current: bool = False,
    ) -> Item:
        """Move versions from a source document into a named item.

        Moves matching versions (filtered by tags if provided) from source_id
        to a named item. If the target already exists, extracted versions are
        appended to its history. The source retains non-matching versions;
        if fully emptied and source is 'now', it resets to default.

        Args:
            name: ID for the target item (created if new, extended if exists)
            source_id: Document to extract from (default: now)
            tags: If provided, only extract versions whose tags contain
                  all specified key=value pairs. If None, extract all.
            only_current: If True, only extract the current (tip) version,
                        not any archived history.

        Returns:
            The moved Item.

        Raises:
            ValueError: If name is empty, source doesn't exist,
                        or no versions match the filter.
        """
        return flow_move_item(
            self,
            name,
            source_id=source_id,
            tags=tags,
            only_current=only_current,
        )

    def _move_direct(
        self,
        name: str,
        *,
        source_id: str = NOWDOC_ID,
        tags: Optional[TagMap] = None,
        only_current: bool = False,
    ) -> Item:
        if not name:
            raise ValueError("Name cannot be empty")
        name = normalize_id(name)
        source_id = normalize_id(source_id)
        if is_part_id(name):
            raise ValueError(
                f"Cannot move to a part ID: {name!r}. "
                "Parts are managed by analyze()."
            )

        # Casefold tag filters so they match casefolded storage
        if tags:
            tags = casefold_tags(tags)
            for key in tags:
                validate_tag_key(key)

        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()

        # Get the source's archived version numbers before extraction
        # (needed to map to ChromaDB versioned IDs)
        source_versions = self._document_store.list_versions(
            doc_coll, source_id, limit=10000
        )
        source_current = self._document_store.get(doc_coll, source_id)
        if source_current is None:
            raise ValueError(f"Source document '{source_id}' not found")

        # Identify which versions will be extracted (for ChromaDB cleanup)
        def _tags_match(item_tags: dict, filt: dict) -> bool:
            for k in filt:
                wanted = set(tag_values(filt, k))
                if not wanted:
                    continue
                stored_values = set(tag_values(item_tags, k))
                if not wanted.issubset(stored_values):
                    return False
            return True

        if only_current:
            # Only extract the tip — no archived versions
            matched_version_nums = []
            if tags:
                current_matches = _tags_match(source_current.tags, tags)
            else:
                current_matches = True
        elif tags:
            matched_version_nums = [
                v.version for v in source_versions
                if _tags_match(v.tags, tags)
            ]
            current_matches = _tags_match(source_current.tags, tags)
        else:
            matched_version_nums = [v.version for v in source_versions]
            current_matches = True

        # Extract in DocumentStore (SQLite side)
        extracted, new_source, base_version = self._document_store.extract_versions(
            doc_coll, source_id, name, tag_filter=tags,
            only_current=only_current,
        )

        # ChromaDB side: move embeddings

        # 1. Collect source ChromaDB IDs for matched versions
        source_chroma_ids = [f"{source_id}@v{n}" for n in matched_version_nums]
        if current_matches:
            source_chroma_ids.append(source_id)

        # 2. If target already exists, archive its current embedding
        # (extract_versions already archived the SQLite side; we mirror in ChromaDB)
        if base_version > 1:
            archive_version = base_version - 1  # version assigned by _archive_current_unlocked
            existing = self._store.get_entries_full(chroma_coll, [name])
            if existing and existing[0].get("embedding") is not None:
                entry = existing[0]
                archived_vid = f"{name}@v{archive_version}"
                archived_tags = dict(entry["tags"])
                archived_tags["_version"] = str(archive_version)
                archived_tags["_base_id"] = name
                self._store.upsert_batch(
                    chroma_coll,
                    [archived_vid],
                    [entry["embedding"]],
                    [entry["summary"] or ""],
                    [casefold_tags_for_index(archived_tags)],
                )

        # 3. Batch-get embeddings from source
        embedding_map: dict[str, list[float]] = {}
        if source_chroma_ids:
            source_entries = self._store.get_entries_full(chroma_coll, source_chroma_ids)
            for entry in source_entries:
                if entry.get("embedding") is not None:
                    embedding_map[entry["id"]] = entry["embedding"]

            # 4. Batch-delete source entries
            found_ids = [entry["id"] for entry in source_entries]
            if found_ids:
                self._store.delete_entries(chroma_coll, found_ids)

        # 5. Insert target entries with new IDs
        # The extracted list is chronological (oldest first),
        # last one is current, rest are history
        target_ids = []
        target_embeddings = []
        target_summaries = []
        target_tags = []

        # History versions (sequential from base_version)
        for seq, vi in enumerate(extracted[:-1], start=base_version):
            target_vid = f"{name}@v{seq}"
            # Find the embedding for this version
            source_vid = f"{source_id}@v{vi.version}" if vi.version > 0 else source_id
            emb = embedding_map.get(source_vid)
            if emb is not None:
                ver_tags = dict(vi.tags)
                ver_tags["_version"] = str(seq)
                ver_tags["_base_id"] = name
                target_ids.append(target_vid)
                target_embeddings.append(emb)
                target_summaries.append(vi.summary)
                target_tags.append(ver_tags)

        # Current (newest extracted)
        newest = extracted[-1]
        source_cur_id = f"{source_id}@v{newest.version}" if newest.version > 0 else source_id
        cur_emb = embedding_map.get(source_cur_id)
        if cur_emb is not None:
            cur_tags = dict(newest.tags)
            cur_tags["_saved_from"] = source_id
            cur_tags["_saved_at"] = utc_now()
            target_ids.append(name)
            target_embeddings.append(cur_emb)
            target_summaries.append(newest.summary)
            target_tags.append(cur_tags)

        # 6. Batch-insert/update into ChromaDB
        if target_ids:
            self._store.upsert_batch(
                chroma_coll,
                target_ids,
                target_embeddings,
                target_summaries,
                [casefold_tags_for_index(t) for t in target_tags],
            )

        # Add system tags to the saved document in DocumentStore too
        saved_doc = self._document_store.get(doc_coll, name)
        if saved_doc:
            saved_tags = dict(saved_doc.tags)
            saved_tags["_saved_from"] = source_id
            saved_tags["_saved_at"] = utc_now()
            self._document_store.update_tags(doc_coll, name, saved_tags)

        # If source was fully emptied and is 'now', recreate with defaults
        if new_source is None and source_id == NOWDOC_ID:
            try:
                default_content, default_tags = _load_frontmatter(
                    SYSTEM_DOC_DIR / "now.md"
                )
            except FileNotFoundError:
                default_content = "# Now\n\nYour working context."
                default_tags = {}
            self._set_now_direct(default_content, tags=default_tags)

        return self._get_direct(name)

    def reset_system_documents(self) -> dict:
        """Force reload all system documents from bundled content."""
        return reset_system_documents(self)

    def _tag_direct(
        self,
        id: str,
        tags: Optional[dict[str, Any]] = None,
        remove: Optional[list[str]] = None,
        remove_values: Optional[dict[str, Any]] = None,
    ) -> Optional[Item]:
        """Update tags on an existing document without re-processing.

        Does NOT re-fetch, re-embed, or re-summarize. Only updates tags.

        Tag behavior:
        - Provided tags are merged with existing user tags
        - `remove=["key"]` removes full keys
        - `remove_values={"key": "value"}` removes specific values
        - Empty string value ("") still deletes that key (legacy compatibility)
        - System tags (_prefixed) cannot be modified via this method

        Args:
            id: Document identifier
            tags: Tags to add/update
            remove: Tag keys to delete
            remove_values: Specific tag values to remove per key

        Returns:
            Updated Item if found, None if document doesn't exist
        """
        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()

        # Deferred system doc migration (normally runs on first _upsert)
        self.ensure_sysdocs()

        # Validate inputs
        id = normalize_id(id)
        add_changes: dict[str, Any] = {}
        legacy_delete_keys: set[str] = set()
        if tags:
            # For tag mutations, constrained-tag validation applies only to
            # additive values, not removals.
            tags = self._validate_tag_map(
                tags, source="Tags", check_constraints=False,
            )
            user_changes = {
                k: tags[k] for k in tags if not k.startswith(SYSTEM_TAG_PREFIX)
            }
            legacy_delete_keys, add_changes = _split_tag_additions(user_changes)
        remove_keys = _normalize_remove_keys(remove) | legacy_delete_keys
        remove_value_changes: dict[str, Any] = {}
        if remove_values:
            remove_values = self._validate_tag_map(
                remove_values, source="Tags", check_constraints=False,
            )
            remove_value_changes = {
                k: remove_values[k]
                for k in remove_values
                if not k.startswith(SYSTEM_TAG_PREFIX)
            }

        # Get existing item (prefer document store, fall back to ChromaDB)
        existing = self._get_direct(id)
        if existing is None:
            return None

        # Start with existing tags, separate system from user
        current_tags = dict(existing.tags)
        system_tags = {k: v for k, v in current_tags.items()
                       if k.startswith(SYSTEM_TAG_PREFIX)}
        user_tags = {k: v for k, v in current_tags.items()
                     if not k.startswith(SYSTEM_TAG_PREFIX)}

        # Apply tag changes (filter out system tags from input)
        if add_changes:
            self._validate_tagdoc_value_constraints(add_changes, existing_tags=current_tags)
            singular_keys = self._get_singular_keys(add_changes)
            if singular_keys:
                self._validate_singular_tags(add_changes, singular_keys)
                for sk in singular_keys:
                    user_tags.pop(sk, None)
        if add_changes or remove_keys or remove_value_changes:
            user_tags = _apply_tag_mutations(
                user_tags,
                add_changes,
                remove_keys=remove_keys,
                remove_by_key=remove_value_changes,
            )

        # Merge back: user tags + system tags
        final_tags = {**user_tags, **system_tags}
        self._normalize_edge_tag_values(final_tags, doc_coll)

        # Dual-write: SQLite gets original values, ChromaDB gets casefolded
        self._document_store.update_tags(doc_coll, id, final_tags)
        self._store.update_tags(chroma_coll, id, casefold_tags_for_index(final_tags))

        self._process_edge_tags(
            id, final_tags, current_tags, doc_coll,
            summary=existing.summary,
        )

        # Tag changes can affect meta-doc resolution (tag-based queries)
        self._context_cache.notify_write(
            id,
            old_tags=current_tags,
            new_tags=final_tags,
        )

        # Return updated item
        return self._get_direct(id)

    def tag(
        self,
        id: str,
        tags: Optional[dict[str, Any]] = None,
        remove: Optional[list[str]] = None,
        remove_values: Optional[dict[str, Any]] = None,
    ) -> Optional[Item]:
        """Update tags via the flow host interface when possible."""
        if remove or remove_values:
            return self._tag_direct(
                id,
                tags=tags,
                remove=remove,
                remove_values=remove_values,
            )
        return flow_tag_item(self, id, tags)

    def tag_part(
        self,
        id: str,
        part_num: int,
        tags: Optional[dict[str, Any]] = None,
        remove: Optional[list[str]] = None,
        remove_values: Optional[dict[str, Any]] = None,
    ) -> Optional[PartInfo]:
        """Update user tags on a part without re-analyzing.

        Parts are machine-generated, so summaries and content are immutable.
        Tags can be edited to correct or override analyzer tagging decisions.

        System tags (_prefixed) cannot be modified.
        `remove=["key"]` removes full keys.
        `remove_values={"key": "value"}` removes specific values.
        Empty string still deletes a key (legacy compatibility).

        Args:
            id: Parent document ID (not the part ID)
            part_num: Part number (1-indexed)
            tags: Tags to add/update
            remove: Tag keys to delete
            remove_values: Specific tag values to remove per key

        Returns:
            Updated PartInfo if found, None if part doesn't exist
        """
        id = normalize_id(id)
        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()

        part = self._document_store.get_part(doc_coll, id, part_num)
        if part is None:
            return None

        # Merge: existing tags + new tags, empty string = delete
        merged = dict(part.tags)
        add_changes: dict[str, Any] = {}
        legacy_delete_keys: set[str] = set()
        if tags:
            tags = self._validate_tag_map(
                tags, source="Tags", check_constraints=False,
            )
            user_changes = {
                k: tags[k] for k in tags if not k.startswith(SYSTEM_TAG_PREFIX)
            }
            legacy_delete_keys, add_changes = _split_tag_additions(user_changes)
        remove_keys = _normalize_remove_keys(remove) | legacy_delete_keys
        remove_value_changes: dict[str, Any] = {}
        if remove_values:
            remove_values = self._validate_tag_map(
                remove_values, source="Tags", check_constraints=False,
            )
            remove_value_changes = {
                k: remove_values[k]
                for k in remove_values
                if not k.startswith(SYSTEM_TAG_PREFIX)
            }
        if add_changes:
            self._validate_tagdoc_value_constraints(add_changes, existing_tags=merged)
            singular_keys = self._get_singular_keys(add_changes)
            if singular_keys:
                self._validate_singular_tags(add_changes, singular_keys)
                for sk in singular_keys:
                    merged.pop(sk, None)
        if add_changes or remove_keys or remove_value_changes:
            merged = _apply_tag_mutations(
                merged,
                add_changes,
                remove_keys=remove_keys,
                remove_by_key=remove_value_changes,
            )

        # Dual-write: SQLite gets original values, ChromaDB gets casefolded
        self._document_store.update_part_tags(doc_coll, id, part_num, merged)
        self._store.update_tags(chroma_coll, f"{id}@p{part_num}",
                                casefold_tags_for_index(merged))

        return self._document_store.get_part(doc_coll, id, part_num)

    # -------------------------------------------------------------------------
    # Parts (structural decomposition)
    # -------------------------------------------------------------------------

    def _gather_analyze_chunks(
        self, id: str, doc_record, *, since_version: int | None = None,
    ) -> list[dict] | dict[str, list[dict]]:
        """Gather content chunks for analysis as serializable dicts.

        For URI sources: re-fetch the document content.
        For inline notes: assemble version history chronologically.

        When *since_version* is set and the item is not URI-sourced, returns
        ``{"context": [...], "targets": [...]}`` with overlap context chunks
        and new-version target chunks.  Returns ``{"context": [], "targets": []}``
        if there are no new versions.
        """
        doc_coll = self._resolve_doc_collection()
        source = doc_record.tags.get("_source")
        parent_user_tags = {
            k: v for k, v in doc_record.tags.items()
            if not k.startswith(SYSTEM_TAG_PREFIX)
        }
        chunks: list[dict] = []

        if source == "uri":
            source_uri = doc_record.tags.get("_source_uri") or id
            if isinstance(source_uri, list):
                source_uri = source_uri[0] if source_uri else id
            try:
                doc = self._document_provider.fetch(str(source_uri))
                chunks = [{"content": doc.content, "tags": parent_user_tags, "index": 0}]
            except Exception as e:
                logger.warning("Could not re-fetch %s: %s, using summary", source_uri, e)
            if chunks:
                return chunks  # URI sources don't support incremental

        versions = self._document_store.list_versions(doc_coll, id, limit=100)

        if since_version is not None and versions:
            # Incremental: split versions into context (already analyzed) and targets (new)
            chronological = list(reversed(versions))
            context_versions = [v for v in chronological if v.version <= since_version]
            new_versions = [v for v in chronological if v.version > since_version]

            if not new_versions:
                return {"context": [], "targets": []}

            # Take last N context versions as overlap
            overlap = context_versions[-self._config.incremental_context:]

            context_chunks: list[dict] = []
            for i, v in enumerate(overlap):
                date_str = v.created_at[:10] if v.created_at else ""
                context_chunks.append({
                    "content": f"[{date_str}]\n{v.summary}",
                    "tags": parent_user_tags,
                    "index": i,
                })

            target_chunks: list[dict] = []
            idx = len(context_chunks)
            for v in new_versions:
                date_str = v.created_at[:10] if v.created_at else ""
                target_chunks.append({
                    "content": f"[{date_str}]\n{v.summary}",
                    "tags": parent_user_tags,
                    "index": idx,
                })
                idx += 1
            # Current doc as final target
            target_chunks.append({
                "content": f"[current]\n{doc_record.summary}",
                "tags": parent_user_tags,
                "index": idx,
            })

            return {"context": context_chunks, "targets": target_chunks}

        # Full analysis (no since_version or no versions)
        if versions:
            for i, v in enumerate(reversed(versions)):
                date_str = v.created_at[:10] if v.created_at else ""
                chunks.append({
                    "content": f"[{date_str}]\n{v.summary}",
                    "tags": parent_user_tags,
                    "index": i,
                })
            chunks.append({
                "content": f"[current]\n{doc_record.summary}",
                "tags": parent_user_tags,
                "index": len(chunks),
            })
        else:
            chunks = [{"content": doc_record.summary, "tags": parent_user_tags, "index": 0}]

        return chunks

    def _gather_guide_context(self, tags: list[str]) -> str:
        """Build guide context string from tag descriptions."""
        doc_coll = self._resolve_doc_collection()
        guide_parts = []
        for tag_key in tags:
            tag_doc = self._document_store.get(doc_coll, f".tag/{tag_key}")
            if tag_doc:
                guide_parts.append(f"## Tag: {tag_key}\n{tag_doc.summary}")
        return "\n\n".join(guide_parts)

    def analyze(
        self,
        id: str,
        *,
        analyzer=None,
        tags: Optional[list[str]] = None,
        force: bool = False,
    ) -> list[PartInfo]:
        """Decompose a note or string into meaningful parts.

        For URI-sourced documents: decomposes the document content structurally.
        For inline notes (strings): assembles the version history and decomposes
        the temporal sequence into episodic parts.

        Uses a pluggable AnalyzerProvider to identify sections with summaries
        and tags. Re-analysis replaces all previous parts atomically.

        Skips analysis if the document's content_hash matches the stored
        _analyzed_hash tag (parts are already current). Use force=True
        to override.

        Args:
            id: Document or string to analyze
            analyzer: Override AnalyzerProvider for decomposition
                (default: use configured analyzer or SlidingWindowAnalyzer)
            tags: Guidance tag keys (e.g., ["topic", "type"]) —
                fetches .tag/xxx descriptions as decomposition context
            force: Skip the _analyzed_hash check and re-analyze regardless

        Returns:
            List of PartInfo for the created parts (empty list if skipped)
        """
        id = normalize_id(id)
        doc_coll = self._resolve_doc_collection()

        # Get the document
        doc_record = self._document_store.get(doc_coll, id)
        if doc_record is None:
            raise ValueError(f"Document not found: {id}")

        # Skip if parts are already current (content unchanged since last analysis)
        if not force and doc_record.content_hash:
            analyzed_hash = doc_record.tags.get("_analyzed_hash")
            if analyzed_hash and analyzed_hash == doc_record.content_hash:
                logger.info("Skipping analysis for %s: parts already current", id)
                return self.list_parts(id)

        # Detect incremental vstring analysis
        analyzed_version_str = doc_record.tags.get("_analyzed_version")
        is_vstring = doc_record.tags.get("_source") != "uri"
        incremental = (
            not force
            and bool(analyzed_version_str)
            and is_vstring
            and analyzer is None  # custom analyzers always get full
        )
        since_version: int | None = None
        if incremental:
            try:
                since_version = int(analyzed_version_str)
            except (ValueError, TypeError):
                incremental = False

        # Skip when content is untruncated and there's no version thread
        # to synthesize — parts would be redundant copies of already-
        # searchable content.
        if not force and not incremental and is_vstring and not analyzed_version_str:
            content_length = int(
                doc_record.tags.get("_content_length", "0") or "0"
            )
            if content_length <= self._config.max_summary_length:
                logger.info(
                    "Skipping analysis for %s: single version, "
                    "untruncated (%d <= %d chars)",
                    id, content_length, self._config.max_summary_length,
                )
                self._record_analyzed_tags(doc_coll, id, doc_record)
                return []

        # Phase 1: Gather — assemble chunks, guide context, tag specs (local)
        gather_result = self._gather_analyze_chunks(
            id, doc_record, since_version=since_version if incremental else None,
        )

        # Handle incremental gather result
        if isinstance(gather_result, dict):
            context_chunks = gather_result["context"]
            target_chunks = gather_result["targets"]
            if not target_chunks:
                logger.info("Skipping incremental analysis for %s: no new versions", id)
                self._record_analyzed_tags(doc_coll, id, doc_record)
                return self.list_parts(id)
            # Diff-ratio skip: if latest target is near-identical to last
            # context version, the change is too trivial to re-analyze.
            diff_threshold = self._config.analyze_diff_threshold
            if diff_threshold < 1.0 and context_chunks and target_chunks:
                prev_text = context_chunks[-1].get("content", "")
                curr_text = target_chunks[-1].get("content", "")
                ratio = difflib.SequenceMatcher(
                    None, prev_text, curr_text,
                ).ratio()
                if ratio >= diff_threshold:
                    logger.info(
                        "Skipping analysis for %s: trivial diff (%.1f%% similar)",
                        id, ratio * 100,
                    )
                    self._record_analyzed_tags(doc_coll, id, doc_record)
                    return self.list_parts(id)
            chunk_dicts = context_chunks + target_chunks
        else:
            context_chunks = None
            target_chunks = None
            chunk_dicts = gather_result
            incremental = False  # gather returned flat list (URI or full)

        total_content = "".join(c["content"] for c in chunk_dicts)
        min_len = self._config.min_analyze_length
        if len(total_content.strip()) < min_len:
            logger.info(
                "Skipping analysis for %s: content too short (%d < %d)",
                id, len(total_content.strip()), min_len,
            )
            return []

        guide_context = self._gather_guide_context(tags) if tags else ""

        tag_specs = None
        try:
            classifier = TagClassifier(
                provider=self._get_summarization_provider(),
            )
            tag_specs = classifier.load_specs(self) or None
        except Exception as e:
            logger.warning("Could not load tag specs: %s", e)

        # Resolve analysis prompt from store-backed .prompt/analyze/* docs
        analysis_prompt = self._resolve_prompt_doc(
            "analyze", doc_record.tags, required=True,
        )

        # Phase 2: Compute — pure processor (local or remote)
        # Wait for any background reconciliation to finish first — both
        # sentence-transformers (embedding) and mlx-lm (summarization)
        # import the `transformers` package, and concurrent imports
        # corrupt module state (Python import lock is per-module).
        if analyzer is None:
            self._reconcile_done.wait(timeout=30)
            analyzer_provider = self._get_summarization_provider()
        else:
            analyzer_provider = None  # custom analyzer passed directly

        if incremental and context_chunks is not None and target_chunks is not None:
            # Incremental path: single-window LLM call with context + targets
            incremental_prompt = self._load_prompt_doc(
                ".prompt/analyze/incremental", required=True,
            )

            total_tokens = sum(
                _estimate_tokens(c["content"])
                for c in context_chunks + target_chunks
            )
            # If too large for single window, fall back to full analysis
            if total_tokens > 12000:
                logger.info(
                    "Incremental content too large (%d tokens), falling back to full analysis: %s",
                    total_tokens, id,
                )
                incremental = False
                # Re-gather as full
                chunk_dicts = self._gather_analyze_chunks(id, doc_record)
                # Fall through to full path below
            else:
                # Build single-window prompt with <analyze> marking
                prompt_parts = ["<content>"]
                for c in context_chunks:
                    prompt_parts.append(c["content"])
                prompt_parts.append("<analyze>")
                for c in target_chunks:
                    prompt_parts.append(c["content"])
                prompt_parts.append("</analyze>")
                prompt_parts.append("</content>")
                user_prompt = "\n\n".join(prompt_parts)

                if guide_context:
                    user_prompt = f"{guide_context}\n\n---\n\n{user_prompt}"

                provider = analyzer_provider or self._get_summarization_provider()
                # Unwrap caching wrapper if present
                raw_provider = provider
                if hasattr(raw_provider, '_provider') and raw_provider._provider is not None:
                    raw_provider = raw_provider._provider

                prompt_text = incremental_prompt
                try:
                    result_text = raw_provider.generate(
                        prompt_text, user_prompt, max_tokens=4096,
                    )
                    new_parts = _parse_parts(result_text) if result_text else []
                except Exception as e:
                    logger.warning("Incremental analysis LLM call failed: %s", e)
                    new_parts = []

                # Classify new parts with tag specs (filtered by _when)
                if tag_specs and new_parts:
                    filtered_specs = self._filter_tag_specs_by_when(
                        tag_specs, dict(doc_record.tags), id,
                        doc_record.summary,
                    )
                    if filtered_specs:
                        try:
                            classifier.classify(new_parts, filtered_specs)
                        except Exception as e:
                            logger.warning("Tag classification skipped: %s", e)

                # Append new parts (don't delete old ones)
                if new_parts:
                    self._append_incremental_parts(
                        id, doc_coll, new_parts,
                    )

                self._record_analyzed_tags(doc_coll, id, doc_record)
                return self.list_parts(id)

        # Full analysis path
        if analyzer is not None:
            # Custom analyzer: call directly (not through process_analyze)
            analysis_chunks = [
                AnalysisChunk(
                    content=c["content"], tags=c.get("tags", {}),
                    index=c.get("index", i),
                )
                for i, c in enumerate(chunk_dicts)
            ]
            raw_parts = analyzer.analyze(
                analysis_chunks, guide_context,
                prompt_override=analysis_prompt,
            )
            if tag_specs and raw_parts:
                filtered_specs = self._filter_tag_specs_by_when(
                    tag_specs, dict(doc_record.tags), id,
                    doc_record.summary,
                )
                if filtered_specs:
                    try:
                        classifier.classify(raw_parts, filtered_specs)
                    except Exception as e:
                        logger.warning("Tag classification skipped: %s", e)
            analyze_result = {"parts": raw_parts}
        else:
            analyze_result = process_analyze(
                chunk_dicts, guide_context, tag_specs,
                analyzer_provider=analyzer_provider,
                prompt_override=analysis_prompt,
                classifier_provider=analyzer_provider,
            )

        # Extract line ranges for URI-sourced documents
        raw_parts = analyze_result.get("parts") or []
        if (raw_parts 
            and doc_record.tags.get("_source") == "uri" 
            and chunk_dicts 
            and len(chunk_dicts) == 1):
            try:
                source_content = chunk_dicts[0]["content"]
                raw_parts = _extract_line_ranges(source_content, raw_parts)
                analyze_result["parts"] = raw_parts
            except Exception as e:
                logger.warning("Line range extraction failed: %s", e)

        # Content not decomposable — single section is redundant with the note
        if not raw_parts or len(raw_parts) <= 1:
            logger.info("Content not decomposable into multiple parts: %s", id)
            self._record_analyzed_tags(doc_coll, id, doc_record)
            return []

        # Phase 3: Apply — write parts + embeddings to stores (local)
        # Build action-style output and apply via the shared mutation path.
        output = self._task_result_to_output(
            id,
            doc_coll,
            "analyze",
            analyze_result,
            existing_tags=doc_record.tags,
        )
        if output and output.get("mutations"):
            _apply_mutations(self, doc_coll, output)

        return self.list_parts(id)

    def _record_analyzed_tags(self, doc_coll: str, id: str, doc_record) -> None:
        """Update _analyzed_hash and _analyzed_version tags after analysis."""
        chroma_coll = self._resolve_chroma_collection()
        updated_tags = dict(doc_record.tags)
        if doc_record.content_hash:
            updated_tags["_analyzed_hash"] = doc_record.content_hash
        versions = self._document_store.list_versions(doc_coll, id, limit=1)
        if versions:
            updated_tags["_analyzed_version"] = str(versions[0].version)
        self._document_store.update_tags(doc_coll, id, updated_tags)
        self._store.update_tags(chroma_coll, id, casefold_tags_for_index(updated_tags))

    def _append_incremental_parts(
        self,
        id: str,
        doc_coll: str,
        new_parts: list[dict],
    ) -> None:
        """Append new analysis parts without deleting existing ones."""
        chroma_coll = self._resolve_chroma_collection()
        max_part = self._document_store.max_part_num(doc_coll, id)
        embed = self._get_embedding_provider()
        now = utc_now()

        # Parts do NOT inherit parent tags — see analyze.py for the
        # rationale. Each part carries only its analyzer-assigned tags
        # plus _base_id/_part_num bookkeeping.
        for i, raw in enumerate(new_parts, start=max_part + 1):
            part_tags: dict[str, Any] = {}
            if raw.get("tags"):
                part_tags.update(raw["tags"])
            part_tags["_base_id"] = id
            part_tags["_part_num"] = str(i)
            summary = str(raw.get("summary") or "")

            part = PartInfo(
                part_num=i,
                summary=summary,
                tags=part_tags,
                created_at=now,
            )
            self._document_store.upsert_single_part(doc_coll, id, part)

            if summary:
                embedding = embed.embed(summary)
                self._store.upsert_part(
                    chroma_coll, id, i,
                    embedding, summary,
                    casefold_tags_for_index(part_tags),
                )


    def get_part(self, id: str, part_num: int) -> Optional[Item]:
        """Get a specific part of a document.

        Returns the part as an Item with _part_num, _base_id, and
        _total_parts metadata tags. Parts are numbered 1..N.

        Args:
            id: Document identifier
            part_num: Part number (1+ for decomposed parts)

        Returns:
            Item if found, None otherwise
        """
        doc_coll = self._resolve_doc_collection()
        part = self._document_store.get_part(doc_coll, id, part_num)
        if part is None:
            return None

        total = self._document_store.part_count(doc_coll, id)
        tags = dict(part.tags)
        tags["_part_num"] = str(part.part_num)
        tags["_base_id"] = id
        tags["_total_parts"] = str(total)

        return Item(
            id=id,
            summary=part.summary,
            tags=tags,
        )

    def list_parts(self, id: str) -> list[PartInfo]:
        """List all parts for a document.

        Args:
            id: Document identifier

        Returns:
            List of PartInfo, ordered by part_num
        """
        doc_coll = self._resolve_doc_collection()
        return self._document_store.list_parts(doc_coll, id)

    def _cleanup_legacy_overview_parts(
        self,
        chroma_coll: str,
        doc_coll: str,
    ) -> int:
        """Delete legacy @P{0} overview parts from both stores."""
        overview_ids = self._document_store.list_part_doc_ids(doc_coll, part_num=0)
        if not overview_ids:
            return 0

        self._store.delete_entries(
            chroma_coll,
            [f"{doc_id}@p0" for doc_id in overview_ids],
        )
        deleted = 0
        for doc_id in overview_ids:
            deleted += self._document_store.delete_part(doc_coll, doc_id, 0)
        logger.info("Removed %d legacy overview part(s)", deleted)
        return deleted

    def _enqueue_migrated_part_reindex(self) -> int:
        """Queue part reindex tasks for rows rewritten by schema migration."""
        migrated = getattr(self._document_store, "migrated_parts_for_reindex", [])
        if not migrated:
            return 0

        queued = len(migrated)
        for part in migrated:
            self._pending_queue.enqueue(
                f"{part['id']}@p{part['part_num']}",
                part["collection"],
                part["summary"],
                task_type="reindex",
                metadata={
                    "part_num": part["part_num"],
                    "base_id": part["id"],
                    "tags": dict(part.get("tags") or {}),
                },
            )
        migrated.clear()
        logger.info("Queued %d migrated part(s) for reindex", queued)
        return queued

    # -------------------------------------------------------------------------
    # Collection Management
    # -------------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """List all collections in the store."""
        # Merge collections from both stores
        doc_collections = set(self._document_store.list_collections())
        chroma_collections = set(self._store.list_collections())
        return sorted(doc_collections | chroma_collections)
    
    def count(self) -> int:
        """Count items in a collection.

        Returns count from document store if available, else vector store.
        """
        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()
        doc_count = self._document_store.count(doc_coll)
        if doc_count > 0:
            return doc_count
        return self._store.count(chroma_coll)

    def count_versions(self) -> int:
        """Count archived versions in the collection."""
        doc_coll = self._resolve_doc_collection()
        return self._document_store.count_versions(doc_coll)

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
    ) -> list[Item]:
        """List items with composable filters.

        All filters are AND'd together. Prefix narrows by ID, tags narrow
        by metadata, date narrows by time.

        Args:
            prefix: ID prefix filter (e.g. ".tag/act").
            tags: Tag key=value filters (all must match).
            tag_keys: Tag key-only filters (item must have key, any value).
            since: Only items updated since (ISO duration like P3D, or date).
            until: Only items updated before (ISO duration or date).
            order_by: Sort order - "updated" (default), "accessed", or "created".
            include_hidden: Include system notes (dot-prefix IDs).
            include_history: Include previous versions alongside current items.
            limit: Maximum results to return.

        Returns:
            List of Items, most recent first.
        """
        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()

        # Casefold tag filters to match casefolded storage
        if tags:
            tags = casefold_tags(tags)
            for k in tags:
                validate_tag_key(k)
        if tag_keys:
            tag_keys = [k.casefold() for k in tag_keys]
            for k in tag_keys:
                validate_tag_key(k)

        # ------------------------------------------------------------------
        # Paginated fetch-and-filter: fetches pages of results, applies
        # post-filters, and accumulates until we have `limit` results or
        # the data source is exhausted.
        # ------------------------------------------------------------------
        page_size = min(limit * 3, 200)
        max_rows = max(limit * 20, 200)
        offset = 0
        items: list[Item] = []

        while True:
            # --------------------------------------------------------------
            # Primary data source: pick the most selective query
            # --------------------------------------------------------------
            if tags:
                # Key=value tags: query the canonical document store so
                # listing semantics do not depend on embedding/index state.
                tag_pairs = list(iter_tag_pairs(tags))
                if not tag_pairs:
                    batch = []
                    raw_count = 0
                else:
                    first_key, first_value = tag_pairs[0]
                    docs = self._document_store.query_by_tag_value(
                        doc_coll,
                        first_key,
                        first_value,
                        limit=page_size,
                        offset=offset,
                    )
                    batch = [_record_to_item(doc) for doc in docs]
                    raw_count = len(docs)
                    for extra_key, extra_value in tag_pairs[1:]:
                        batch = [
                            item for item in batch
                            if extra_value in tag_values(item.tags, extra_key)
                        ]

            elif tag_keys:
                # Key-only: use SQLite tag key query (first key as primary,
                # additional keys as post-filter)
                since_date = _parse_date_param(since) if since else None
                until_date = _parse_date_param(until) if until else None
                docs = self._document_store.query_by_tag_key(
                    doc_coll, tag_keys[0], limit=page_size,
                    since_date=since_date, until_date=until_date,
                    offset=offset,
                )
                batch = [_record_to_item(d) for d in docs]
                raw_count = len(docs)
                # Post-filter additional key-only tags
                for extra_key in tag_keys[1:]:
                    batch = [i for i in batch if extra_key in i.tags]

            elif prefix is not None:
                # ID pattern query — glob if pattern contains * or ?, else prefix.
                if "*" in prefix or "?" in prefix:
                    records = self._document_store.query_by_id_glob(
                        doc_coll, prefix, limit=page_size, offset=offset,
                    )
                else:
                    # Prefix match — also finds children (e.g. ".tag" finds ".tag/foo").
                    records = self._document_store.query_by_id_prefix(
                        doc_coll, prefix, limit=page_size, offset=offset,
                    )
                batch = [_record_to_item(rec) for rec in records]
                raw_count = len(records)

            else:
                # Default: recent items
                if include_history:
                    records = self._document_store.list_recent_with_history(
                        doc_coll, page_size, order_by=order_by, offset=offset,
                    )
                else:
                    records = self._document_store.list_recent(
                        doc_coll, page_size, order_by=order_by, offset=offset,
                    )
                batch = [_record_to_item(rec) for rec in records]
                raw_count = len(records)

            # --------------------------------------------------------------
            # Post-filters (apply remaining predicates to this page)
            # --------------------------------------------------------------

            # Prefix filter (if primary query wasn't prefix-based)
            if prefix is not None and tags:
                batch = [i for i in batch if i.id.startswith(prefix)]

            # Tag-key filter (if primary query was something else)
            if tag_keys and tags:
                for k in tag_keys:
                    batch = [i for i in batch if k in i.tags]

            # Date filter (skip if already applied in SQL via tag_keys path)
            if (since is not None or until is not None) and not tag_keys:
                batch = _filter_by_date(batch, since=since, until=until)

            # Hidden filter (prefix queries always include hidden)
            if not include_hidden and prefix is None:
                batch = [i for i in batch if not _is_hidden(i)]

            items.extend(batch)
            offset += raw_count

            # Enough results, or data source exhausted?
            if len(items) >= limit or raw_count < page_size or offset >= max_rows:
                break

        return items[:limit]

    # -------------------------------------------------------------------------
    # Pending Work Queue (summaries + analysis)
    # -------------------------------------------------------------------------

    def enqueue_analyze(
        self,
        id: str,
        tags: Optional[list[str]] = None,
        force: bool = False,
    ) -> bool:
        """Enqueue a note for background analysis (decomposition into parts).

        Validates the document exists, then adds it to the pending work
        queue for serial processing by the background daemon.

        Skips enqueueing if the document's _analyzed_hash matches its
        content_hash (parts are already current). Use force=True to
        override.

        Args:
            id: Document ID to analyze
            tags: Guidance tag keys for decomposition
            force: Enqueue even if parts are already current

        Returns:
            True if enqueued, False if skipped (parts already current)
        """
        id = normalize_id(id)
        doc_coll = self._resolve_doc_collection()
        doc = self._document_store.get(doc_coll, id)
        if doc is None:
            raise ValueError(f"Document not found: {id}")

        # Skip if parts are already current
        if not force and doc.content_hash:
            analyzed_hash = doc.tags.get("_analyzed_hash")
            if analyzed_hash and analyzed_hash == doc.content_hash:
                logger.info("Skipping enqueue for %s: parts already current", id)
                return False

        metadata: dict[str, Any] = {}
        if tags:
            metadata["tags"] = list(tags)
        if force:
            metadata["force"] = True
        self._enqueue_task_background(
            task_type="analyze",
            id=id,
            doc_coll=doc_coll,
            content="",
            metadata=metadata,
        )
        self._spawn_processor()
        return True

    # -- Pending processing and delegation methods are in BackgroundProcessingMixin --
    # (_emit_processing_breakdown, process_pending, _delegate_task,
    #  _poll_delegated, _is_delegation_stale, _revert_stale_delegation)

    # -------------------------------------------------------------------------
    # Planner priors
    # -------------------------------------------------------------------------

    def get_planner_priors(
        self,
        scope_key: str | None = None,
        candidates: list[str] | None = None,
    ) -> dict:
        """Return minimal planner priors for flow discriminators.

        Shape:
        {
            "planner_priors": {
                "fanout": {...},
                "selectivity": {...},
                "cardinality": {...}
            },
            "staleness": {"stats_age_s": 14, "fallback_mode": false}
        }
        """
        if not self._planner_stats:
            return {
                "planner_priors": {},
                "staleness": {"stats_age_s": None, "fallback_mode": True},
            }

        if scope_key is None:
            scope_key = build_scope_key()

        raw_priors = self._planner_stats.get_priors(
            scope_key,
            metric_families=[
                "expansion.fanout",
                "expansion.selectivity",
                "facet.cardinality",
            ],
            subject_keys=candidates,
        )
        staleness = self._planner_stats.get_staleness(scope_key)

        priors = {
            "fanout": raw_priors.get("expansion.fanout", {}),
            "selectivity": raw_priors.get("expansion.selectivity", {}),
            "cardinality": raw_priors.get("facet.cardinality", {}),
        }

        return {
            "planner_priors": priors,
            "staleness": staleness,
        }

    def rebuild_planner_stats(self) -> dict:
        """Full rebuild of planner statistics from canonical data.

        Returns dict with count of stats upserted per metric family.
        """
        if not self._planner_stats:
            return {}
        doc_coll = self._resolve_doc_collection()
        return self._planner_stats.rebuild(self._document_store, doc_coll)

    # -- Processing pipeline methods are in BackgroundProcessingMixin --
    # (apply_result, _run_local_task_workflow, _process_pending_embed,
    #  _process_pending_reindex, _flush_edge_backfill, _process_pending_backfill_edges,
    #  _ocr_image, _ocr_pdf, pending_count, pending_work_count, process_pending_work,
    #  pending_stats, pending_stats_by_type, pending_status, _get_work_queue)


    def _run_read_flow(
        self,
        state: str,
        params: dict[str, Any],
        *,
        budget: int = 5,
        query_embedding: Any = None,
    ) -> "FlowResult":
        """Run a synchronous state-doc flow for the read/query path.

        Creates a read-only environment, wires up a state doc loader
        and action runner, then evaluates the flow to completion.
        State docs are loaded from ``.state/*`` notes in the store
        (seeded by system doc migration).

        Args:
            state: Name of the starting state doc (e.g. "get").
            params: Caller-supplied parameters.
            budget: Maximum ticks before forced stop.
            query_embedding: Optional embedding vector for semantic
                tiebreaking in traverse actions.

        Returns:
            FlowResult with terminal status and accumulated bindings.
        """
        self.ensure_sysdocs()
        env = LocalFlowEnvironment(self)
        if query_embedding is not None:
            env._query_embedding = query_embedding
        loader = make_state_doc_loader(env)
        runner = make_action_runner(env, context_cache=self._context_cache)
        result = run_flow(state, params, budget=budget, load_state_doc=loader, run_action=runner)

        # If a foreground flow hit an async action, enqueue the cursor
        # for daemon execution and return what we have so far.
        if result.status == "async" and result.cursor:
            self._enqueue_flow_cursor(
                state=state, cursor_token=result.cursor, params=params,
            )

        return result

    def run_flow(
        self,
        state: str,
        *,
        params: dict[str, Any] | None = None,
        budget: int | None = None,
        cursor_token: str | None = None,
        state_doc_yaml: str | None = None,
        writable: bool = True,
    ) -> "FlowResult":
        return self.run_flow_command(
            state,
            params=params,
            budget=budget,
            cursor_token=cursor_token,
            state_doc_yaml=state_doc_yaml,
            writable=writable,
        )

    def run_flow_command(
        self,
        state: str,
        *,
        params: dict[str, Any] | None = None,
        budget: int | None = None,
        cursor_token: str | None = None,
        state_doc_yaml: str | None = None,
        writable: bool = True,
    ) -> "FlowResult":
        """Run a state-doc flow synchronously.

        This is the public API behind ``keep flow``. Supports starting
        new flows, resuming stopped flows via cursor, and loading state
        docs from YAML or the store.

        Args:
            state: State doc name (e.g. "after-write", "query-resolve").
            params: Caller-supplied parameters.
            budget: Max ticks this invocation. Defaults to config.budget_per_flow.
            cursor_token: Opaque cursor from a previous stopped flow.
            state_doc_yaml: If provided, parse this YAML as the state doc
                instead of loading from the store.
            writable: If True, enable write actions (summarize, tag, etc.).

        Returns:
            FlowResult with status, bindings, cursor (if stopped), etc.
        """
        # Prompt rendering: render_prompt + expand_prompt as a flow command
        if state == STATE_PROMPT:
            p = params or {}
            name = p.get("name", "")
            # list mode: return available prompts
            if not name or p.get("list"):
                prompts = self.list_prompts()
                return FlowResult(status="done", data={
                    "prompts": [
                        {
                            "name": pr.name,
                            "summary": pr.summary,
                            **({"mcp_arguments": list(pr.mcp_arguments)} if pr.mcp_arguments else {}),
                        }
                        for pr in prompts
                    ],
                }, ticks=1)
            # render mode
            try:
                result = self.render_prompt(
                    name, p.get("text"),
                    id=p.get("id"), since=p.get("since"), until=p.get("until"),
                    tags=p.get("tags"), deep=p.get("deep", False),
                    scope=p.get("scope"), token_budget=p.get("token_budget"),
                )
            except Exception as e:
                return FlowResult(status="error", data={"error": str(e)})
            if result is None:
                return FlowResult(status="error", data={"error": f"prompt not found: {name}"})
            expanded = expand_prompt(result, self)
            data: dict[str, Any] = {"text": expanded}
            if result.context:
                data["context"] = result.context.to_dict()
            return FlowResult(status="done", data=data, ticks=1)

        if budget is None:
            budget = self._config.budget_per_flow

        if state_doc_yaml is None and state != STATE_PROMPT:
            self.ensure_sysdocs()

        env = LocalFlowEnvironment(self)
        base_runner = make_action_runner(env, writable=writable)
        if writable:
            collection = self._resolve_doc_collection()

            def runner(action_name: str, action_params: dict[str, Any]) -> dict[str, Any]:
                output = base_runner(action_name, action_params)
                if isinstance(output, dict) and output.get("mutations"):
                    _apply_mutations(self, collection, output)
                return output
        else:
            runner = base_runner

        # Build loader: inline YAML overrides store lookup
        if state_doc_yaml is not None:
            inline_doc = parse_state_doc(state, state_doc_yaml)

            def _loader(name: str):
                if name == state:
                    return inline_doc
                # Fall through to store for transitions to other states
                return make_state_doc_loader(env)(name)
        else:
            _loader = make_state_doc_loader(env)

        # Decode cursor if resuming
        cursor = None
        if cursor_token:
            # Try server-side cursor first (short ID)
            cursor = self._load_cursor(cursor_token)
            if cursor is None:
                # Fall back to self-contained base64 cursor (backward compat)
                cursor = decode_cursor(cursor_token)

        # Inject defaults for query flow params
        merged_params: dict[str, Any] = {
            "limit": 10,
            "margin_high": 0.15,
            "margin_low": 0.03,
            "entropy_high": 0.8,
            "entropy_low": 0.3,
            "lineage_strong": 0.5,
            "explore_limit": 10,
            "explore_limit_wide": 15,
            "pivot_limit": 5,
            "bridge_limit": 5,
            "deep_limit": 10,
            "max_summary_length": 200,
            "similar_limit": 3,
            "meta_limit": 3,
            "parts_limit": 5,
            "versions_limit": 3,
            "edges_limit": 5,
        }
        merged_params.update(params or {})

        result = run_flow(
            state,
            merged_params,
            budget=budget,
            load_state_doc=_loader,
            run_action=runner,
            cursor=cursor,
        )

        # If a foreground flow hit an async action, enqueue the cursor
        # for daemon execution.
        if result.status == "async" and result.cursor:
            self._enqueue_flow_cursor(
                state=state, cursor_token=result.cursor, params=merged_params,
            )

        # Store cursor server-side, replace with short ID
        if result.cursor:
            cursor_id = self._store_cursor(result.cursor)
            result.cursor = cursor_id

        return result

    def _store_cursor(self, cursor_token: str) -> str:
        """Store a self-contained cursor as a system note, return short ID."""
        cursor_id = hashlib.sha256(cursor_token.encode()).hexdigest()[:12]
        note_id = f".cursor/{cursor_id}"
        self.put(cursor_token, id=note_id)
        return cursor_id

    def _load_cursor(self, cursor_id: str) -> "Optional[FlowCursor]":
        """Load a server-side cursor by short ID, delete after loading."""
        note_id = f".cursor/{cursor_id}"
        item = self.get(note_id)
        if item is None:
            return None
        try:
            self.delete(note_id)
        except Exception:
            pass
        content = getattr(item, "summary", None)
        if not content:
            return None
        return decode_cursor(str(content))

    # -- Spawn/processor methods are in BackgroundProcessingMixin --

    def reconcile(
        self,
        fix: bool = False,
        _doc_store: "Optional[DocumentStore]" = None,
    ) -> dict:
        """Check and optionally fix consistency between document store and vector store.

        Detects:
        - Documents in document store missing from vector store (not searchable)
        - Documents in vector store missing from document store (orphaned embeddings)

        Args:
            fix: If True, re-index documents missing from vector store
            _doc_store: Optional separate DocumentStore for thread-safe reads
                (used by background reconcile to avoid concurrent SQLite access)

        Returns:
            Dict with 'missing_from_index', 'orphaned_in_index', 'fixed' counts
        """
        ds = _doc_store or self._document_store
        doc_coll = self._resolve_doc_collection()
        chroma_coll = self._resolve_chroma_collection()

        # Find mismatches between stores. Dot-prefixed system notes are not
        # indexed in Chroma, so they are excluded from "missing" checks.
        doc_ids = ds.list_ids(doc_coll)
        indexed_doc_ids = [doc_id for doc_id in doc_ids if not is_system_id(doc_id)]
        missing_from_chroma_raw = self._store.find_missing_ids(chroma_coll, indexed_doc_ids)
        missing_records: dict[str, Any] = {}
        for doc_id in missing_from_chroma_raw:
            rec = ds.get(doc_coll, doc_id)
            if rec and rec.summary:
                missing_records[doc_id] = rec

        def _vector_base_id(chroma_id: str) -> str:
            for marker in ("@v", "@V", "@p", "@P"):
                if marker in chroma_id:
                    return chroma_id.rsplit(marker, 1)[0]
            return chroma_id

        # Indexed system-note rows are stale by definition and should be
        # removed during reconcile. For non-system notes, skip versioned and
        # part IDs here because they are tracked with their base document.
        chroma_ids = self._store.list_ids(chroma_coll)
        doc_id_set = set(indexed_doc_ids)
        stale_system_ids = {
            cid for cid in chroma_ids
            if is_system_id(_vector_base_id(cid))
        }
        orphaned_in_chroma = {
            cid for cid in chroma_ids
            if cid not in doc_id_set and "@v" not in cid and "@p" not in cid
        }
        orphaned_in_chroma.update(stale_system_ids)

        fixed = 0
        removed = 0
        if fix:
            # Re-index items missing from ChromaDB using stored summary
            for doc_id, doc_record in missing_records.items():
                if self._closing.is_set():
                    break
                try:
                    if doc_record:
                        embed_text = doc_record.summary or doc_id
                        embedding = self._get_embedding_provider().embed(embed_text)
                        self._store.upsert(
                            collection=chroma_coll,
                            id=doc_id,
                            embedding=embedding,
                            summary=doc_record.summary or doc_id,
                            tags=casefold_tags_for_index(doc_record.tags),
                        )
                        fixed += 1
                        logger.info("Reconciled: %s", doc_id)
                except Exception as e:
                    logger.warning("Failed to reconcile %s: %s", doc_id, e)

            # Remove orphaned ChromaDB entries
            for orphan_id in orphaned_in_chroma:
                if self._closing.is_set():
                    break
                try:
                    self._store.delete(chroma_coll, orphan_id)
                    removed += 1
                    logger.info("Removed orphan: %s", orphan_id)
                except Exception as e:
                    logger.warning("Failed to remove orphan %s: %s", orphan_id, e)

        return {
            "missing_from_index": len(missing_records),
            "orphaned_in_index": len(orphaned_in_chroma),
            "fixed": fixed,
            "removed": removed,
            "missing_ids": list(missing_records) if missing_records else [],
            "orphaned_ids": list(orphaned_in_chroma) if orphaned_in_chroma else [],
        }

    @property
    def config(self) -> "StoreConfig":
        """Public access to store configuration."""
        return self._config

    def close(self) -> None:
        """Close resources (stores, caches, queues).

        Waits for background reconcile to finish before tearing down stores,
        then releases model locks (freeing GPU memory) before file locks.

        Wraps each step in try/except because close() may be called from
        an atexit handler during interpreter shutdown, when C extension
        modules (sqlite3, chromadb) may already be partially finalized.
        """
        if getattr(self, '_closed', False):
            return
        self._closed = True

        # Signal reconcile thread to stop and wait for it
        if hasattr(self, '_closing'):
            self._closing.set()
        if hasattr(self, '_reconcile_thread') and self._reconcile_thread is not None:
            self._reconcile_thread.join(timeout=10)

        # Release locked model providers (frees GPU memory + gc)
        self._safe_close_step("embedding provider", self._release_embedding_provider)
        self._safe_close_step("summarization provider", self._release_summarization_provider)

        if self._media_describer is not None:
            if hasattr(self._media_describer, 'release'):
                self._safe_close_step("media describer", self._media_describer.release)
            self._media_describer = None

        if self._content_extractor is not None:
            if hasattr(self._content_extractor, 'release'):
                self._safe_close_step("content extractor", self._content_extractor.release)
            self._content_extractor = None

        # Close ChromaDB store
        if hasattr(self, '_store') and self._store is not None:
            self._safe_close_step("vector store", self._store.close)

        # Close document store (SQLite)
        if hasattr(self, '_document_store') and self._document_store is not None:
            self._safe_close_step("document store", self._document_store.close)

        # Close pending summary queue
        if hasattr(self, '_pending_queue'):
            self._safe_close_step("pending queue", self._pending_queue.close)

        # Close task delegation client
        if hasattr(self, '_task_client') and self._task_client is not None:
            self._safe_close_step("task client", self._task_client.close)
            self._task_client = None

        # Close work queue
        if hasattr(self, "_work_queue") and self._work_queue is not None:
            self._safe_close_step("work queue", self._work_queue.close)

        # Remove ops log handler to avoid handler accumulation
        if hasattr(self, '_ops_log_handler') and self._ops_log_handler:
            logging.getLogger("keep").removeHandler(self._ops_log_handler)
            self._ops_log_handler.close()
            self._ops_log_handler = None

    def _safe_close_step(self, label: str, close_fn) -> None:
        """Run one close/release step without hiding failures from debug logs."""
        try:
            close_fn()
        except Exception:
            logger.debug("close: failed to release %s", label, exc_info=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close resources."""
        self.close()
        return False

    def __del__(self):
        """Release file handles during GC — no logging, no I/O beyond close."""
        if getattr(self, '_closed', False):
            return
        self._closed = True
        # Release resources that hold OS file descriptors.
        # Skip perf logging, model releases, reconcile waits — the
        # filesystem may already be gone (temp stores, test teardown).
        for attr in ('_document_store', '_store', '_pending_queue', '_work_queue'):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        # Remove log handler to avoid handler accumulation
        handler = getattr(self, '_ops_log_handler', None)
        if handler:
            try:
                logging.getLogger("keep").removeHandler(handler)
                handler.close()
            except Exception:
                pass
