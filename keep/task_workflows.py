"""Local task workflow dispatch.

Actions implement a single ``run(params, context)`` method.  This module
provides the ``TaskRequest`` / ``TaskRunResult`` types, a lightweight
adapter that presents a Keeper as an ``ActionContext``, a generic
mutation applier, and the ``run_local_task`` dispatcher that ties them
together.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .recovery import run_with_document_store_recovery
from .tracing import get_tracer

if TYPE_CHECKING:
    from .api import Keeper

logger = logging.getLogger(__name__)


@dataclass
class TaskRequest:
    """Minimal task request shape shared by queue and flow paths."""

    task_type: str
    id: str
    collection: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRunResult:
    """Outcome of a local task workflow."""

    status: str  # "applied" | "skipped"
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter: Keeper → ActionContext
# ---------------------------------------------------------------------------

class _KeeperActionContext:
    """Present a Keeper as an ActionContext for background task execution."""

    def __init__(
        self,
        keeper: "Keeper",
        *,
        collection: str,
        item_id: str | None = None,
        item_text: str | None = None,
    ) -> None:
        self._keeper = keeper
        self._collection = collection
        self.item_id = item_id
        self.item_text = item_text
        self.item_content = item_text

    def get(self, id: str) -> Any:
        return self._keeper.get(id)

    def put(
        self,
        *,
        content: str | None = None,
        uri: str | None = None,
        id: str | None = None,
        tags: dict[str, Any] | None = None,
        summary: str | None = None,
        created_at: str | None = None,
        force: bool = False,
        queue_background_tasks: bool = True,
    ) -> Any:
        return self._keeper._put_direct(
            content=content,
            uri=uri,
            id=id,
            tags=tags,
            summary=summary,
            created_at=created_at,
            force=force,
            queue_background_tasks=queue_background_tasks,
        )

    def stub(
        self,
        *,
        id: str,
        content: str | None = None,
        tags: dict[str, Any] | None = None,
        summary: str | None = None,
        created_at: str | None = None,
        queue_background_tasks: bool = True,
    ) -> Any:
        return self._keeper._stub_direct(
            id=id,
            content=content,
            tags=tags,
            summary=summary,
            created_at=created_at,
            queue_background_tasks=queue_background_tasks,
        )

    def tag(
        self,
        id: str,
        tags: dict[str, Any] | None = None,
        *,
        remove: list[str] | None = None,
        remove_values: dict[str, Any] | None = None,
    ) -> Any:
        return self._keeper._tag_direct(
            id,
            tags=tags,
            remove=remove,
            remove_values=remove_values,
        )

    def delete(self, id: str, *, delete_versions: bool = True) -> None:
        self._keeper._delete_direct(id, delete_versions=delete_versions)

    def find(
        self,
        query: str | None = None,
        *,
        tags: dict[str, Any] | None = None,
        similar_to: str | None = None,
        stored_only: bool = False,
        limit: int = 10,
        since: str | None = None,
        until: str | None = None,
        include_self: bool = False,
        include_hidden: bool = False,
        deep: bool = False,
        scope: str | None = None,
    ) -> list[Any]:
        return self._keeper.find(
            query, tags=tags, similar_to=similar_to, limit=limit,
            stored_only=stored_only, since=since, until=until, include_self=include_self,
            include_hidden=include_hidden, deep=deep, scope=scope,
        )

    def list_items(
        self,
        *,
        prefix: str | None = None,
        tags: dict[str, Any] | None = None,
        since: str | None = None,
        until: str | None = None,
        order_by: str = "updated",
        include_hidden: bool = False,
        limit: int = 10,
    ) -> list[Any]:
        return self._keeper.list_items(
            prefix=prefix, tags=tags, since=since, until=until,
            order_by=order_by, include_hidden=include_hidden, limit=limit,
        )

    def list_parts(self, id: str) -> list[Any]:
        return self._keeper.list_parts(id)

    def list_versions(self, id: str, *, limit: int = 3) -> list[Any]:
        return self._keeper.list_versions(id, limit=limit)

    def resolve_edges(self, id: str, *, limit: int = 5) -> dict[str, Any]:
        item = self._keeper.get(id)
        if item is None:
            return {"edges": {}, "count": 0}
        edge_refs = self._keeper._resolve_edge_refs(item, id)
        result: dict[str, list[dict[str, Any]]] = {}
        total = 0
        for key, refs in edge_refs.items():
            entries = []
            for ref in refs[:limit]:
                entries.append({
                    "id": ref.source_id,
                    "summary": ref.summary or "",
                    "predicate": key,
                    "date": ref.date or "",
                })
                total += 1
            if entries:
                result[key] = entries
        return {"edges": result, "count": total}

    def get_document(self, id: str) -> Any:
        return self._keeper._document_store.get(self._collection, id)

    def get_document_store(self) -> Any:
        return self._keeper._document_store

    def get_collection(self) -> str:
        return self._collection

    def get_db_connection(self) -> Any:
        return self._keeper._document_store._conn

    def find_by_name(self, stem: str, *, vault: str | None = None) -> Any:
        results = self._keeper._document_store.find_by_name(
            self._collection, stem, id_prefix=vault, limit=1,
        )
        if results:
            from .types import Item
            rec = results[0]
            return Item(id=rec.id, summary=rec.summary, tags=rec.tags)
        return None

    def find_referencing(self, target_id: str, tag_key: str = "references", limit: int = 50) -> list[Any]:
        """Find items that reference *target_id* via edge-tag *tag_key*."""
        from .dependencies import NoteDependencyService

        doc_coll = self._keeper._resolve_doc_collection()
        dependencies = NoteDependencyService(
            self._keeper._document_store, doc_coll,
        )
        source_ids = dependencies.all_source_ids(
            target_id, include_archived=False,
        )[:limit]
        if not source_ids:
            return []
        docs = self._keeper._document_store.get_many(doc_coll, source_ids)
        from .types import Item
        return [
            Item(id=rec.id, summary=rec.summary, tags=rec.tags)
            for rec in docs.values()
        ]

    def find_by_content_hash(
        self, content_hash: str, *, content_hash_full: str = "",
        exclude_id: str = "", limit: int = 10,
    ) -> list[Any]:
        """Find documents with matching content hash."""
        results = self._keeper._document_store.find_by_content_hash(
            self._collection, content_hash,
            content_hash_full=content_hash_full,
            exclude_id=exclude_id, limit=limit,
        )
        if isinstance(results, list):
            return results
        return [results] if results else []

    def resolve_meta(self, id: str, limit_per_doc: int = 3) -> dict[str, list[Any]]:
        return self._keeper.resolve_meta(item_id=id, limit_per_doc=limit_per_doc)

    def gather_context(self, item_id: str, tags: dict[str, Any]) -> str:
        return self._keeper._gather_context(item_id, tags)

    def gather_analyze_chunks(
        self, item_id: str, item: Any, *, since_version: int | None = None,
    ) -> Any:
        return self._keeper._gather_analyze_chunks(
            item_id, item, since_version=since_version,
        )

    def gather_guide_context(self, tags: list[str]) -> str:
        return self._keeper._gather_guide_context(tags)

    def load_prompt_doc(self, doc_id: str, *, required: bool = False) -> str | None:
        """Load a prompt doc by exact ID and return its ``## Prompt`` section."""
        return self._keeper._load_prompt_doc(doc_id, required=required)

    def max_part_num(self, item_id: str) -> int:
        """Highest existing part number for *item_id* (0 if none)."""
        return self._keeper._document_store.max_part_num(
            self._collection, item_id,
        ) or 0

    def resolve_provider(self, kind: str, name: str | None = None) -> Any:
        _PROVIDER_MAP = {
            "summarization": self._keeper._get_summarization_provider,
            "content_extractor": self._keeper._get_content_extractor,
            "analyzer": self._keeper._get_analyzer,
            "media": self._keeper._get_media_describer,
        }
        method = _PROVIDER_MAP.get(kind)
        if method is None:
            raise ValueError(f"unknown provider kind: {kind!r}")
        return method()

    def resolve_prompt(
        self, prefix: str, doc_tags: dict[str, Any] | None = None,
        *, item_id: str | None = None, item_summary: str = "",
    ) -> str | None:
        return self._keeper._resolve_prompt_doc(
            prefix, doc_tags or {}, item_id=item_id, item_summary=item_summary,
        )

    def move(
        self, name: str, *, source_id: str = "now",
        tags: dict[str, Any] | None = None, only_current: bool = False,
    ) -> Any:
        return self._keeper._move_direct(
            name, source_id=source_id, tags=tags, only_current=only_current,
        )

    def traverse(self, source_ids: list[str], *, limit: int = 5) -> dict[str, list[Any]]:
        traverse = getattr(self._keeper, "traverse_related", None)
        if callable(traverse):
            return traverse(source_ids, limit_per_source=limit)
        return {}


# ---------------------------------------------------------------------------
# Generic mutation applier
# ---------------------------------------------------------------------------

def _resolve_ref(value: Any, output: dict[str, Any]) -> Any:
    """Resolve ``$output.X`` references to concrete values."""
    if isinstance(value, str) and value.startswith("$output."):
        key = value[len("$output."):]
        return output.get(key, value)
    return value


def _apply_mutations(
    keeper: "Keeper",
    collection: str,
    output: dict[str, Any],
) -> None:
    """Process mutations returned by an action's ``run()`` method."""
    from .types import casefold_tags_for_index

    mutations = output.get("mutations")
    if not isinstance(mutations, list):
        return

    for mut in mutations:
        if not isinstance(mut, dict):
            continue
        op = mut.get("op", "")

        if op == "set_summary":
            target = str(mut["target"])
            summary = str(_resolve_ref(mut["summary"], output))
            intent = str(mut.get("intent") or "derived_summary_replace")
            # Strip LLM preambles ("Here is a summary...") as a safety net
            from keep.providers.base import strip_summary_preamble
            summary = strip_summary_preamble(summary)
            keeper._apply_summary_mutation(
                collection,
                target,
                summary,
                intent=intent,
                embed=bool(mut.get("embed")),
                content_hash=str(mut.get("content_hash", "")),
                content_hash_full=str(mut.get("content_hash_full", "")),
            )

        elif op == "put_item":
            from .types import is_part_id, parse_part_id
            item_id = mut.get("id", "")
            if is_part_id(item_id):
                # Parts go through part-specific storage, not _put_direct
                base_id, part_num = parse_part_id(item_id)
                from .document_store import PartInfo
                from .types import utc_now
                part = PartInfo(
                    part_num=part_num,
                    summary=str(mut.get("summary", "")),
                    tags=dict(mut.get("tags") or {}),
                    created_at=utc_now(),
                )
                keeper._document_store.upsert_single_part(collection, base_id, part)
                chroma_coll = keeper._resolve_chroma_collection()
                embedding = keeper._get_embedding_provider().embed(part.summary)
                keeper._store.upsert_part(
                    chroma_coll, base_id, part_num,
                    embedding, part.summary, casefold_tags_for_index(part.tags),
                )
            else:
                keeper._put_direct(
                    content=str(mut.get("content", "")),
                    id=item_id,
                    summary=str(mut.get("summary", "")),
                    tags=mut.get("tags"),
                    queue_background_tasks=mut.get("queue_background_tasks", False),
                )

        elif op == "stub_item":
            keeper._stub_via_flow(
                id=str(mut.get("id", "")),
                content=str(mut.get("content", "")),
                summary=str(mut.get("summary", "")),
                tags=mut.get("tags"),
                created_at=mut.get("created_at"),
                queue_background_tasks=mut.get("queue_background_tasks", True),
            )

        elif op == "set_tags":
            target = str(mut["target"])
            tags = _resolve_ref(mut["tags"], output)
            if isinstance(tags, dict):
                # Fetch existing tags for edge diff
                existing_doc = keeper._document_store.get(collection, target)
                existing_tags = existing_doc.tags if existing_doc else {}

                # Merge into existing tags (don't replace — actions only
                # return the tags they want to set, not the full tag map)
                merged = dict(existing_tags)
                merged.update(tags)

                # Normalize edge-tag values into canonical labeled-ref
                # form before storage. Mirrors the put path.
                keeper._normalize_edge_tag_values(merged, collection)

                keeper._document_store.update_tags(collection, target, merged)
                chroma_coll = keeper._resolve_chroma_collection()
                keeper._store.update_tags(
                    chroma_coll, target, casefold_tags_for_index(merged),
                )

                # Sync edge table for any edge-tag changes
                keeper._process_edge_tags(target, merged, existing_tags, collection)

        elif op == "delete_prefix":
            prefix = str(mut["prefix"])
            chroma_coll = keeper._resolve_chroma_collection()
            keeper._store.delete_parts(chroma_coll, prefix.rstrip("@p"))
            keeper._document_store.delete_parts(collection, prefix.rstrip("@p"))

        else:
            logger.warning("Unknown mutation op: %r", op)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_local_task(keeper: "Keeper", req: TaskRequest) -> TaskRunResult:
    """Run a background task by calling the action's ``run()`` method."""
    from .actions import prepare_action_params

    task_type = str(req.task_type or "").strip()

    ctx = _KeeperActionContext(
        keeper,
        collection=req.collection,
        item_id=req.id,
        item_text=req.content or None,
    )

    params: dict[str, Any] = {"item_id": req.id}
    params.update(req.metadata)
    action, params = prepare_action_params(task_type, params, ctx)

    def _execute_once() -> TaskRunResult:
        with get_tracer("work_queue").start_as_current_span(
            "task.run",
            attributes={
                "item_id": req.id,
                "collection": req.collection,
                "task_type": task_type,
            },
        ):
            output = action.run(params, ctx)

        if not isinstance(output, dict):
            return TaskRunResult(status="skipped", details={"reason": "no_output"})

        if output.get("skipped"):
            return TaskRunResult(status="skipped", details=output)

        mutations = output.get("mutations")
        if not mutations:
            return TaskRunResult(status="skipped", details=output)

        _apply_mutations(keeper, req.collection, output)
        return TaskRunResult(status="applied", details=output)

    return run_with_document_store_recovery(
        _execute_once,
        get_document_store=ctx.get_document_store,
        operation=f"Background task {task_type} for {req.id}",
    )
