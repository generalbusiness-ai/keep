from __future__ import annotations

from typing import Any

from . import action, coerce_item_id, item_to_result


def _updated_key(item: Any) -> str:
    tags = getattr(item, "tags", None)
    if not isinstance(tags, dict):
        return ""
    return str(tags.get("_updated") or "")


@action(id="traverse")
class Traverse:
    """Follow edges from a set of items to discover related items."""

    def run(self, params: dict[str, Any], context) -> dict[str, Any]:
        raw_items = params.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("traverse requires non-empty items list")
        limit = max(int(params.get("limit", 5)), 1)
        # deep_follow_depth controls how many primary items contribute tags to
        # the Tier-2 tag-follow fallback.  Defaults to 10; clamped in the env.
        deep_follow_depth = int(params.get("deep_follow_depth", 10))

        source_ids: list[str] = []
        for raw in raw_items:
            sid = coerce_item_id(raw)
            if sid:
                source_ids.append(sid)
        if not source_ids:
            return {"groups": {}, "count": 0}

        source_set = set(source_ids)
        groups: dict[str, list[dict[str, Any]]] = {}
        total = 0

        traverse = getattr(context, "traverse", None)
        related_groups = (
            traverse(source_ids, limit=limit, deep_follow_depth=deep_follow_depth)
            if callable(traverse)
            else {}
        )
        if not isinstance(related_groups, dict):
            related_groups = {}

        for source_id in source_ids:
            raw_group = related_groups.get(source_id)
            if isinstance(raw_group, list) and raw_group:
                related = sorted(raw_group, key=_updated_key, reverse=True)
            else:
                related = []
            dedup: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in related:
                item_id = coerce_item_id(item)
                if not item_id or item_id in source_set or item_id in seen:
                    continue
                seen.add(item_id)
                dedup.append(item_to_result(item))
                if len(dedup) >= limit:
                    break
            groups[source_id] = dedup
            total += len(dedup)

        return {"groups": groups, "count": total}
