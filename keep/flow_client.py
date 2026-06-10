"""Shared flow-backed client wrappers.

The stable execution boundary is ``run_flow``. Public convenience helpers
live here and invoke named state docs where those names already exist.
For semantics that do not yet have a stable named state doc, these helpers
use inline compatibility state docs rather than inventing new public names.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Optional

from .const import (
    STATE_COMPAT_FIND,
    STATE_COMPAT_GET_ITEM,
    STATE_DELETE,
    STATE_FIND_DEEP,
    STATE_GET_CONTEXT,
    STATE_LIST,
    STATE_MOVE,
    STATE_PUT,
    STATE_STUB,
    STATE_TAG,
)
from .protocol import FlowHostProtocol
from .types import Item, TagMap

_COMPAT_GET_ITEM = dedent(
    """
    match: sequence
    rules:
      - id: item
        do: get
        with:
          id: "{params.id}"
      - return:
          status: done
          with:
            item: "{item}"
    """
).strip()

_COMPAT_FIND = dedent(
    """
    match: sequence
    rules:
      - id: results
        do: find
        with:
          query: "{params.query}"
          tags: "{params.tags}"
          similar_to: "{params.similar_to}"
          limit: "{params.limit}"
          since: "{params.since}"
          until: "{params.until}"
          include_self: "{params.include_self}"
          include_hidden: "{params.include_hidden}"
          deep: "{params.deep}"
          deep_follow_depth: "{params.deep_follow_depth}"
          scope: "{params.scope}"
      - return:
          status: done
          with:
            items: "{results.results}"
            deep_groups: "{results.deep_groups}"
    """
).strip()


def _expect_done(result: Any, state: str) -> Any:
    status = getattr(result, "status", None)
    if status != "done":
        data = getattr(result, "data", None)
        raise ValueError(f"flow {state!r} failed with status {status!r}: {data!r}")
    return result


def _coerce_item(data: Any) -> Item:
    if not isinstance(data, dict):
        raise ValueError(f"Expected item dict, got {type(data).__name__}")
    item_id = data.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(f"Missing item id in {data!r}")
    tags = data.get("tags", {})
    if not isinstance(tags, dict):
        tags = {}
    score = data.get("score")
    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
    changed = data.get("changed")
    if changed is not None:
        changed = bool(changed)
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)
    return Item(
        id=item_id,
        summary=summary,
        tags={str(k): v for k, v in tags.items()},
        score=score,
        changed=changed,
    )


def _coerce_item_list(items: Any) -> list[Item]:
    if not isinstance(items, list):
        raise ValueError(f"Expected list of items, got {type(items).__name__}")
    return [_coerce_item(item) for item in items]


def _raise_binding_error(binding: Any, state: str) -> None:
    if not isinstance(binding, dict):
        return
    error = binding.get("error")
    if isinstance(error, str) and error:
        raise ValueError(error)


def _run(host: FlowHostProtocol, state: str, *, params: dict[str, Any], writable: bool, state_doc_yaml: str | None = None) -> Any:
    return _expect_done(
        host.run_flow(
            state,
            params=params,
            writable=writable,
            state_doc_yaml=state_doc_yaml,
        ),
        state,
    )


def get_item(host: FlowHostProtocol, id: str) -> Optional[Item]:
    result = _run(
        host,
        STATE_COMPAT_GET_ITEM,
        params={"id": id},
        writable=False,
        state_doc_yaml=_COMPAT_GET_ITEM,
    )
    binding = getattr(result, "bindings", {}).get("item", {})
    _raise_binding_error(binding, STATE_COMPAT_GET_ITEM)
    data = getattr(result, "data", None) or {}
    item = data.get("item")
    if item is None:
        return None
    if isinstance(item, dict) and not item.get("id"):
        return None
    return _coerce_item(item)


def put_item(
    host: FlowHostProtocol,
    content: Optional[str] = None,
    *,
    uri: Optional[str] = None,
    id: Optional[str] = None,
    summary: Optional[str] = None,
    tags: Optional[TagMap] = None,
    created_at: Optional[str] = None,
    force: bool = False,
) -> Item:
    result = _run(
        host,
        STATE_PUT,
        params={
            "content": content,
            "uri": uri,
            "id": id,
            "summary": summary,
            "tags": tags,
            "created_at": created_at,
            "force": force,
        },
        writable=True,
    )
    binding = getattr(result, "bindings", {}).get("stored", {})
    _raise_binding_error(binding, STATE_PUT)
    if isinstance(binding, dict) and binding.get("id"):
        item = dict(binding)
        if tags is not None and "tags" not in item:
            item["tags"] = tags
        return _coerce_item(item)
    data = getattr(result, "data", None) or {}
    stored = data.get("stored")
    _raise_binding_error(stored, STATE_PUT)
    if isinstance(stored, dict) and stored.get("id"):
        item = dict(stored)
        if tags is not None and "tags" not in item:
            item["tags"] = tags
        return _coerce_item(item)
    item = data.get("item")
    _raise_binding_error(item, STATE_PUT)
    if isinstance(item, dict) and item.get("id"):
        return _coerce_item(item)
    raise ValueError("put flow completed without a stored item")


def stub_item(
    host: FlowHostProtocol,
    *,
    id: str,
    content: Optional[str] = None,
    summary: Optional[str] = None,
    tags: Optional[TagMap] = None,
    created_at: Optional[str] = None,
    queue_background_tasks: bool = True,
) -> Item:
    result = _run(
        host,
        STATE_STUB,
        params={
            "id": id,
            "content": content,
            "summary": summary,
            "tags": tags,
            "created_at": created_at,
            "queue_background_tasks": queue_background_tasks,
        },
        writable=True,
    )
    binding = getattr(result, "bindings", {}).get("stored", {})
    _raise_binding_error(binding, STATE_STUB)
    if isinstance(binding, dict) and binding.get("id"):
        item = dict(binding)
        if tags is not None and "tags" not in item:
            item["tags"] = tags
        return _coerce_item(item)
    data = getattr(result, "data", None) or {}
    stored = data.get("stored")
    _raise_binding_error(stored, STATE_STUB)
    if isinstance(stored, dict) and stored.get("id"):
        item = dict(stored)
        if tags is not None and "tags" not in item:
            item["tags"] = tags
        return _coerce_item(item)
    raise ValueError("stub flow completed without a stored item")


def find_items(
    host: FlowHostProtocol,
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
    deep_follow_depth: int = 10,
    scope: Optional[str] = None,
) -> list[Item]:
    if not query and not similar_to and not deep:
        result = _run(
            host,
            STATE_LIST,
            params={
                "prefix": None,
                "tags": tags,
                "tag_keys": None,
                "since": since,
                "until": until,
                "order_by": "updated",
                "include_hidden": include_hidden,
                "limit": limit,
            },
            writable=False,
        )
        bindings = getattr(result, "bindings", {})
        results = bindings.get("results", {}) if isinstance(bindings, dict) else {}
        _raise_binding_error(results, STATE_LIST)
        items = _coerce_item_list(results.get("results", []))
        deep_groups_raw = {}
    else:
        result = _run(
            host,
            STATE_COMPAT_FIND,
            params={
                "query": query,
                "tags": tags,
                "similar_to": similar_to,
                "stored_only": stored_only,
                "limit": limit,
                "since": since,
                "until": until,
                "include_self": include_self,
                "include_hidden": include_hidden,
                "scope": scope,
                "deep": deep,
                "deep_follow_depth": deep_follow_depth,
            },
            writable=False,
            state_doc_yaml=_COMPAT_FIND,
        )
        binding = getattr(result, "bindings", {}).get("results", {})
        _raise_binding_error(binding, STATE_COMPAT_FIND)
        data = getattr(result, "data", None) or {}
        items = _coerce_item_list(data.get("items", []))
        deep_groups_raw = data.get("deep_groups", {}) if deep else {}

    deep_groups: dict[str, list[Item]] = {}
    if isinstance(deep_groups_raw, dict):
        for key, values in deep_groups_raw.items():
            group = _coerce_item_list(values)
            if group:
                deep_groups[str(key)] = group
    try:
        from .api import FindResults
        return FindResults(items, deep_groups=deep_groups)
    except Exception:
        return items


def tag_item(
    host: FlowHostProtocol,
    id: str,
    tags: Optional[TagMap] = None,
) -> Optional[Item]:
    if tags is None:
        return get_item(host, id)
    result = _run(
        host,
        STATE_TAG,
        params={"id": id, "tags": tags},
        writable=True,
    )
    binding = getattr(result, "bindings", {}).get("tagged", {})
    _raise_binding_error(binding, STATE_TAG)
    return get_item(host, id)


def delete_item(
    host: FlowHostProtocol,
    id: str,
    *,
    delete_versions: bool = True,
) -> bool:
    result = _run(
        host,
        STATE_DELETE,
        params={"id": id, "delete_versions": delete_versions},
        writable=True,
    )
    binding = getattr(result, "bindings", {}).get("result", {})
    _raise_binding_error(binding, STATE_DELETE)
    if isinstance(binding, dict) and binding.get("deleted"):
        return True
    data = getattr(result, "data", None) or {}
    return bool(data.get("deleted", False))


def move_item(
    host: FlowHostProtocol,
    name: str,
    *,
    source_id: str = "now",
    tags: Optional[TagMap] = None,
    only_current: bool = False,
) -> Item:
    result = _run(
        host,
        STATE_MOVE,
        params={
            "name": name,
            "source": source_id,
            "tags": tags,
            "only_current": only_current,
        },
        writable=True,
    )
    binding = getattr(result, "bindings", {}).get("moved", {})
    _raise_binding_error(binding, STATE_MOVE)
    if isinstance(binding, dict) and binding.get("id"):
        return _coerce_item(binding)
    data = getattr(result, "data", None) or {}
    if "item" in data:
        return _coerce_item(data["item"])
    fetched = get_item(host, name)
    if fetched is None:
        raise ValueError(f"move flow did not produce target item {name!r}")
    return fetched


def get_now_item(host: FlowHostProtocol, *, scope: Optional[str] = None) -> Item:
    doc_id = f"now:{scope}" if scope else "now"
    item = get_item(host, doc_id)
    if item is not None:
        return item
    if scope:
        return set_now_item(host, f"# Now ({scope})\n\nWorking context.", scope=scope)
    from .system_docs import SYSTEM_DOC_DIR, _load_frontmatter
    try:
        default_content, default_tags = _load_frontmatter(SYSTEM_DOC_DIR / "now.md")
    except FileNotFoundError:
        default_content = "# Now\n\nYour working context."
        default_tags = {}
    return set_now_item(host, default_content, tags=default_tags)


def set_now_item(
    host: FlowHostProtocol,
    content: str,
    *,
    scope: Optional[str] = None,
    tags: Optional[TagMap] = None,
) -> Item:
    doc_id = f"now:{scope}" if scope else "now"
    merged_tags = dict(tags or {})
    if scope:
        merged_tags.setdefault("user", scope)
    return put_item(host, content, id=doc_id, tags=merged_tags or None)
