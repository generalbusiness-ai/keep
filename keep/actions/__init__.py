"""Action registry and base contracts for state-doc style actions."""

from __future__ import annotations

import importlib
import pkgutil
import re
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Action",
    "ActionContext",
    "PreparedAction",
    "DelegatableAction",
    "action",
    "build_delegated_payload",
    "get_action",
    "get_action_priority",
    "is_async_action",
    "list_actions",
    "item_to_result",
    "coerce_item_id",
    "prepare_action_params",
]

# Default action priority (0 = highest, 9 = lowest).
DEFAULT_ACTION_PRIORITY = 5


@runtime_checkable
class Action(Protocol):
    """Protocol for a named action that transforms params into results."""

    def run(self, params: dict[str, Any], context: "ActionContext") -> dict[str, Any]: ...


@runtime_checkable
class PreparedAction(Protocol):
    """Optional hook for action-owned parameter preparation."""

    def prepare(self, params: dict[str, Any], context: "ActionContext") -> dict[str, Any]: ...


@runtime_checkable
class DelegatableAction(Protocol):
    """Optional hook for action-owned remote payload shaping."""

    def build_delegated_payload(
        self, params: dict[str, Any], content: str,
    ) -> tuple[str, dict[str, Any] | None]: ...


@runtime_checkable
class ActionContext(Protocol):
    """Protocol providing store operations available to actions during execution."""

    def get(self, id: str) -> Any | None: ...

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
    ) -> Any: ...

    def stub(
        self,
        *,
        id: str,
        content: str | None = None,
        tags: dict[str, Any] | None = None,
        summary: str | None = None,
        created_at: str | None = None,
        queue_background_tasks: bool = True,
    ) -> Any: ...

    def tag(
        self,
        id: str,
        tags: dict[str, Any] | None = None,
        *,
        remove: list[str] | None = None,
        remove_values: dict[str, Any] | None = None,
    ) -> Any: ...

    def delete(self, id: str, *, delete_versions: bool = True) -> Any: ...

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
    ) -> list[Any]: ...

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
    ) -> list[Any]: ...

    def list_parts(self, id: str) -> list[Any]: ...
    def list_versions(self, id: str, *, limit: int = 3) -> list[Any]: ...
    def resolve_edges(self, id: str, *, limit: int = 5) -> dict[str, Any]: ...

    def get_document(self, id: str) -> Any | None: ...
    def get_document_store(self) -> Any: ...
    def get_collection(self) -> str: ...
    def get_db_connection(self) -> Any: ...
    def find_by_name(self, stem: str, *, vault: str | None = None) -> Any | None: ...
    def find_by_content_hash(
        self, content_hash: str, *, content_hash_full: str = "",
        exclude_id: str = "", limit: int = 20,
    ) -> list[Any]: ...
    def resolve_meta(self, id: str, limit_per_doc: int = 3) -> dict[str, list[Any]]: ...
    def resolve_prompt(
        self, prefix: str, doc_tags: dict[str, Any] | None = None,
        *, item_id: str | None = None, item_summary: str = "",
    ) -> str | None: ...
    def resolve_provider(self, kind: str, name: str | None = None) -> Any: ...
    def gather_context(self, item_id: str, tags: dict[str, Any]) -> str: ...
    def gather_analyze_chunks(
        self, item_id: str, item: Any, *, since_version: int | None = None,
    ) -> Any: ...
    def gather_guide_context(self, tags: list[str]) -> str: ...
    def load_prompt_doc(self, doc_id: str, *, required: bool = False) -> str | None: ...
    def max_part_num(self, item_id: str) -> int: ...
    def move(
        self, name: str, *, source_id: str = "now",
        tags: dict[str, Any] | None = None, only_current: bool = False,
    ) -> Any: ...


_ACTION_REGISTRY: dict[str, type] = {}
_ACTIONS_DISCOVERED = False


def _snake_case(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def action(
    cls: type | None = None,
    *,
    id: str | None = None,
    name: str | None = None,
    priority: int = DEFAULT_ACTION_PRIORITY,
    async_action: bool = False,
):
    """Register a class as a named action.

    Use ``@action(id="summarize")`` to declare the action's canonical ID.
    The ``name`` kwarg is accepted as a legacy alias for ``id``.
    If neither is provided, the ID is derived from the class name via snake_case.

    ``priority`` controls work-queue ordering (0 = highest, 9 = lowest).

    ``async_action`` marks actions that require daemon context (e.g. LLM
    calls, media processing).  When a foreground flow encounters an async
    action it stops, produces a cursor, and delegates the remainder to the
    work queue.
    """
    effective_id = id or name

    def _register(target: type) -> type:
        action_id = str(effective_id or _snake_case(target.__name__)).strip()
        if not action_id:
            raise ValueError("action id cannot be empty")
        _ACTION_REGISTRY[action_id] = target
        target.ACTION_ID = action_id
        target.ACTION_PRIORITY = max(0, min(9, int(priority)))
        target.ACTION_ASYNC = bool(async_action)
        return target

    if cls is None:
        return _register
    return _register(cls)


def _discover_actions() -> None:
    global _ACTIONS_DISCOVERED
    if _ACTIONS_DISCOVERED:
        return
    for module in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        mod_name = module.name
        if mod_name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{mod_name}")
    _ACTIONS_DISCOVERED = True


def get_action(name: str) -> Action:
    _discover_actions()
    key = str(name or "").strip()
    cls = _ACTION_REGISTRY.get(key)
    if cls is None:
        known = ", ".join(sorted(_ACTION_REGISTRY.keys()))
        raise ValueError(f"unknown action: {key!r} (known: {known})")
    inst = cls()
    if not isinstance(inst, Action):
        raise TypeError(f"action {key!r} does not implement Action protocol")
    return inst


def get_action_priority(name: str) -> int:
    """Return the declared priority for an action (0-9, default 5)."""
    _discover_actions()
    cls = _ACTION_REGISTRY.get(str(name or "").strip())
    if cls is None:
        return DEFAULT_ACTION_PRIORITY
    return getattr(cls, "ACTION_PRIORITY", DEFAULT_ACTION_PRIORITY)


def is_async_action(name: str) -> bool:
    """Return whether an action requires daemon context."""
    _discover_actions()
    cls = _ACTION_REGISTRY.get(str(name or "").strip())
    if cls is None:
        return False
    return getattr(cls, "ACTION_ASYNC", False)


def list_actions() -> list[str]:
    _discover_actions()
    return sorted(_ACTION_REGISTRY.keys())


def prepare_action_params(
    name: str,
    params: dict[str, Any],
    context: Any,
) -> tuple[Action, dict[str, Any]]:
    """Return action instance plus prepared params for local/remote execution."""
    act = get_action(name)
    prepared = dict(params)
    if isinstance(act, PreparedAction):
        updated = act.prepare(dict(prepared), context)
        if isinstance(updated, dict):
            prepared = updated
    return act, prepared


def build_delegated_payload(
    action_name: str,
    params: dict[str, Any],
    content: str,
) -> tuple[str, dict[str, Any] | None]:
    """Return ``(content, metadata)`` for the current hosted task API."""
    act = get_action(action_name)
    if isinstance(act, DelegatableAction):
        built = act.build_delegated_payload(dict(params), content)
        if (
            isinstance(built, tuple)
            and len(built) == 2
            and (built[1] is None or isinstance(built[1], dict))
        ):
            delegated_content = built[0]
            return str(delegated_content or ""), built[1]
    metadata = dict(params)
    metadata.pop("item_id", None)
    return content, metadata or None


def coerce_item_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, dict):
        raw = value.get("id")
        if raw is None:
            return None
        return str(raw).strip() or None
    raw = getattr(value, "id", None)
    if raw is None:
        return None
    return str(raw).strip() or None


def item_to_result(item: Any) -> dict[str, Any]:
    tags = getattr(item, "tags", None)
    score = getattr(item, "score", None)
    out = {
        "id": str(getattr(item, "id", "")),
        "summary": str(getattr(item, "summary", "")),
        "tags": dict(tags) if isinstance(tags, dict) else {},
        "score": float(score) if isinstance(score, (int, float)) else None,
    }
    return out
