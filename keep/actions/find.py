from __future__ import annotations

from typing import Any

from . import action, item_to_result


def _part_to_result(base_id: str, part) -> dict[str, Any]:
    """Convert a PartInfo to a find-result dict with a part ID."""
    tags = dict(getattr(part, "tags", None) or {})
    tags["_base_id"] = base_id
    tags["_part_num"] = str(part.part_num)
    return {
        "id": f"{base_id}@p{part.part_num}",
        "summary": getattr(part, "summary", "") or "",
        "tags": tags,
        "score": None,
    }


@action(id="find")
class Find:
    """Search items by query, tags, prefix, or similarity."""

    def run(self, params: dict[str, Any], context) -> dict[str, Any]:
        query = params.get("query")
        similar_to = params.get("similar_to")
        stored_only = bool(params.get("stored_only", False))
        tags = params.get("tags") if isinstance(params.get("tags"), dict) else None
        tag_keys = params.get("tag_keys") if isinstance(params.get("tag_keys"), list) else None
        prefix = params.get("prefix")
        since = params.get("since")
        until = params.get("until")
        offset = int(params.get("offset") or 0)
        include_hidden = bool(params.get("include_hidden", False))
        include_self = bool(params.get("include_self", False))
        deep = bool(params.get("deep", False))
        # deep_follow_depth: how many primary items contribute tags/edges to the
        # deep-follow step.  Clamp: negative/zero → 1; > 100 → 100 (also
        # enforced downstream in _find_direct and traverse_related).
        _raw_depth = params.get("deep_follow_depth")
        if _raw_depth is not None:
            try:
                deep_follow_depth = max(1, min(int(_raw_depth), 100))
            except (TypeError, ValueError):
                deep_follow_depth = 10
        else:
            deep_follow_depth = 10
        order_by = str(params.get("order_by") or "updated")
        limit = int(params.get("limit", 10))
        if limit <= 0:
            return {"results": [], "count": 0}
        limit = max(limit, 1)

        # Bias: {id: weight} — score multiplier. 0=exclude, <1=demote, 1=neutral, >1=boost
        bias = params.get("bias")
        if isinstance(bias, dict):
            bias = {str(k): float(v) for k, v in bias.items()
                    if isinstance(v, (int, float))}
        else:
            bias = None
        n_excluded = sum(1 for v in (bias or {}).values() if v == 0)

        scope = params.get("scope")
        if isinstance(scope, str) and scope:
            scope = scope
        else:
            scope = None

        list_all = bool(params.get("list_all"))
        has_selector = any([
            bool(query),
            bool(similar_to),
            bool(tags),
            bool(tag_keys),
            bool(prefix),
            bool(since),
            list_all,
        ])
        if not has_selector:
            raise ValueError("find requires one of query, similar_to, tags, prefix, or since")
        if query and similar_to:
            raise ValueError("find.query and find.similar_to are mutually exclusive")

        fetch_limit = limit + n_excluded + offset

        # Parts prefix query: prefix ending with @p targets the parts
        # table (document_parts), not the documents table.
        if prefix and str(prefix).endswith("@p"):
            list_parts = getattr(context, "list_parts", None)
            if callable(list_parts):
                base_id = str(prefix)[:-2]  # strip @p suffix
                all_parts = list_parts(base_id)
                results = [_part_to_result(base_id, p) for p in all_parts]
                if offset > 0:
                    results = results[offset:]
                results = results[:limit]
                return {"results": results, "count": len(results)}

        if query or similar_to:
            rows = context.find(
                str(query) if query is not None else None,
                tags=tags,
                similar_to=str(similar_to) if similar_to is not None else None,
                stored_only=stored_only,
                limit=fetch_limit,
                since=str(since) if since is not None else None,
                until=str(until) if until is not None else None,
                include_self=include_self,
                include_hidden=include_hidden,
                deep=deep,
                deep_follow_depth=deep_follow_depth,
                scope=scope,
            )
        else:
            list_kwargs: dict[str, Any] = {
                "prefix": str(prefix) if prefix is not None else None,
                "tags": tags,
                "since": str(since) if since is not None else None,
                "until": str(until) if until is not None else None,
                "order_by": order_by,
                "include_hidden": include_hidden,
                "limit": fetch_limit,
            }
            if tag_keys:
                list_kwargs["tag_keys"] = tag_keys
            rows = context.list_items(**list_kwargs)

        # Apply bias exclusions (weight=0) before converting to result dicts
        if bias:
            rows = [r for r in rows
                    if bias.get(getattr(r, "id", None), 1) != 0]

        # Apply offset
        if offset > 0:
            rows = rows[offset:]

        deep_groups_raw = getattr(rows, "deep_groups", {}) if deep else {}
        rows = rows[:limit]
        results = [item_to_result(row) for row in rows]
        deep_groups: dict[str, list[dict[str, Any]]] = {}
        if isinstance(deep_groups_raw, dict):
            for key, values in deep_groups_raw.items():
                if not isinstance(values, list):
                    continue
                rendered = [item_to_result(value) for value in values]
                if rendered:
                    deep_groups[str(key)] = rendered

        # Apply bias score multipliers on result dicts (Items are frozen)
        if bias:
            for r in results:
                rid = r.get("id")
                w = bias.get(rid) if rid else None
                if w is not None and w != 0 and isinstance(r.get("score"), (int, float)):
                    r["score"] = r["score"] * w
            results.sort(key=lambda r: -(r.get("score") or 0))

        return {
            "results": results,
            "count": len(results),
            "deep_groups": deep_groups,
        }
