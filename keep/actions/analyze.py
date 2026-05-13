from __future__ import annotations

"""Item-scoped decomposition action for generating structured parts."""

import logging
from datetime import datetime, timezone
from typing import Any

from ..processors import process_analyze
from ..providers.base import AnalysisChunk
from ..tracing import get_tracer
from ..types import SYSTEM_TAG_PREFIX, parse_utc_timestamp, utc_now
from . import action
from ._item_scope import check_content_hash, resolve_item_text
from ._tagging import classify_parts_with_specs, _filter_specs_by_when
from ._item_scope import resolve_item
from ._tagging import load_tag_specs

tracer = get_tracer("flow")
logger = logging.getLogger(__name__)

# Minimum interval between successful analyze runs on the same item, in
# seconds. Rapid edits within this window coalesce — the analyze action
# skips with reason "throttled" and lets the next post-throttle edit (or
# an explicit `force=True`) drive the next decomposition. Tunable via the
# `KEEP_ANALYZE_MIN_INTERVAL_S` env var. Mostly aimed at watched files
# (test files, design docs) where one save can fire several rapid
# re-imports without anything actually changing meaningfully.
DEFAULT_MIN_ANALYZE_INTERVAL_S = 300.0  # 5 minutes


def _normalize_part(raw: Any) -> dict[str, Any]:
    """Normalize provider output into a stable part shape."""
    if not isinstance(raw, dict):
        return {"summary": "", "tags": {}}
    tags = raw.get("tags")
    return {
        "summary": str(raw.get("summary") or ""),
        "tags": dict(tags) if isinstance(tags, dict) else {},
    }


def _min_analyze_interval_s() -> float:
    """Read the throttle window from env or fall back to the default.

    Set ``KEEP_ANALYZE_MIN_INTERVAL_S=0`` to disable throttling entirely.
    """
    import os
    raw = os.environ.get("KEEP_ANALYZE_MIN_INTERVAL_S")
    if raw is None:
        return DEFAULT_MIN_ANALYZE_INTERVAL_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_MIN_ANALYZE_INTERVAL_S


def _params_force(params: dict[str, Any]) -> bool:
    """`force=True` (or `"true"`) bypasses every analyze guard."""
    raw = params.get("force")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1", "yes")
    return False


def _throttle_skip_reason(item_tags: dict[str, Any]) -> str | None:
    """If `_analyzed_at` is within the throttle window, return a skip reason.

    Returns ``None`` when there's no recorded prior analyze, the timestamp is
    unparseable, or enough time has passed for the next decomposition to run.
    """
    interval = _min_analyze_interval_s()
    if interval <= 0:
        return None
    last_at = item_tags.get("_analyzed_at")
    if not last_at:
        return None
    try:
        last_dt = parse_utc_timestamp(str(last_at))
    except (ValueError, TypeError):
        return None
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
    if elapsed < 0:
        # Clock skew or future timestamp — don't pretend to know what to do.
        return None
    if elapsed >= interval:
        return None
    return f"throttled (last analyzed {int(elapsed)}s ago, min {int(interval)}s)"


@action(id="analyze", priority=7, async_action=True)
class Analyze:
    """Decompose item content into parts and emit part `put_item` mutations."""

    def prepare(self, params: dict[str, Any], context) -> dict[str, Any]:
        """Populate analyze inputs shared by local and delegated execution."""
        prepared = dict(params)
        item_id, item = resolve_item(prepared, context)
        item_tags = dict(getattr(item, "tags", None) or {})
        item_summary = str(getattr(item, "summary", "") or "")

        if prepared.get("chunks") is None:
            gather_chunks = getattr(context, "gather_analyze_chunks", None)
            if callable(gather_chunks):
                with tracer.start_as_current_span(
                    "analyze.prepare.chunks",
                    attributes={"item_id": item_id},
                ):
                    chunk_data = gather_chunks(item_id, item)
                if isinstance(chunk_data, dict):
                    prepared["chunks"] = list(chunk_data.get("context", [])) + list(chunk_data.get("targets", []))
                elif isinstance(chunk_data, list):
                    prepared["chunks"] = chunk_data

        if prepared.get("guide_context") in (None, ""):
            raw_tags = prepared.get("tags")
            if isinstance(raw_tags, list) and raw_tags:
                gather_guide = getattr(context, "gather_guide_context", None)
                if callable(gather_guide):
                    with tracer.start_as_current_span(
                        "analyze.prepare.guide_context",
                        attributes={"item_id": item_id, "tag_count": len(raw_tags)},
                    ):
                        prepared["guide_context"] = gather_guide(raw_tags)

        if prepared.get("tag_specs") is None:
            with tracer.start_as_current_span(
                "analyze.prepare.tag_specs",
                attributes={"item_id": item_id},
            ):
                specs = load_tag_specs(context)
            if specs:
                prepared["tag_specs"] = specs

        if prepared.get("prompt_override") is None and hasattr(context, "resolve_prompt"):
            with tracer.start_as_current_span(
                "analyze.prepare.prompt",
                attributes={"item_id": item_id, "tag_count": len(item_tags)},
            ):
                prompt_text = context.resolve_prompt(
                    "analyze", item_tags,
                    item_id=item_id, item_summary=item_summary,
                )
            if prompt_text is not None:
                prepared["prompt_override"] = prompt_text

        return prepared

    def build_delegated_payload(
        self, params: dict[str, Any], content: str,
    ) -> tuple[str, dict[str, Any] | None]:
        metadata: dict[str, Any] = {}
        for key in ("chunks", "guide_context", "tag_specs", "prompt_override"):
            value = params.get(key)
            if value:
                metadata[key] = value
        if isinstance(params.get("tags"), list):
            metadata["tags"] = list(params["tags"])
        return "", metadata or None

    def run(self, params: dict[str, Any], context) -> dict[str, Any]:
        """Analyze content, classify parts, and build storage mutations."""
        item_id, _item = resolve_item(params, context)
        item_tags = dict(getattr(_item, "tags", None) or {})
        item_summary = str(getattr(_item, "summary", "") or "")

        if check_content_hash(params, context, item_id, "_analyzed_hash"):
            return {"skipped": True, "reason": "content unchanged"}

        # Throttle rapid re-analyze on the same item. The full analyze
        # pipeline can spend tens of seconds of LLM time per call; watched
        # files that get edited several times a minute would otherwise
        # queue back-to-back decompositions. The next post-throttle edit
        # (or `force=True`) gets through and catches up to current content.
        if not _params_force(params):
            throttled = _throttle_skip_reason(item_tags)
            if throttled is not None:
                return {"skipped": True, "reason": throttled}
        with tracer.start_as_current_span(
            "analyze.prepare",
            attributes={"item_id": item_id},
        ):
            prepared = self.prepare(params, context)
        guide_context = str(prepared.get("guide_context") or "")
        prompt_text = prepared.get("prompt_override")
        if prompt_text is None:
            raise ValueError("missing prompt doc for analyze")

        raw_chunks = prepared.get("chunks")
        if isinstance(raw_chunks, list) and raw_chunks:
            chunk_dicts = raw_chunks
        else:
            _item_id, _item_again, content = resolve_item_text(params, context)
            chunk_dicts = [{"content": str(content), "tags": {}, "index": 0}]

        with tracer.start_as_current_span(
            "analyze.normalize_chunks",
            attributes={"item_id": item_id, "chunk_count": len(chunk_dicts)},
        ):
            analysis_chunks = [
                AnalysisChunk(
                    content=str(chunk.get("content", "")),
                    tags=dict(chunk.get("tags") or {}),
                    index=int(chunk.get("index", idx)),
                )
                for idx, chunk in enumerate(chunk_dicts)
                if isinstance(chunk, dict)
            ]

        raw_parts: list[dict[str, Any]]
        with tracer.start_as_current_span(
            "analyze.resolve_provider",
            attributes={"item_id": item_id},
        ):
            analyzer = context.resolve_provider("analyzer")
        analyze_fn = getattr(analyzer, "analyze", None)
        if callable(analyze_fn):
            with tracer.start_as_current_span(
                "analyze.provider",
                attributes={
                    "item_id": item_id,
                    "chunk_count": len(analysis_chunks),
                    "guide_chars": len(guide_context),
                    "has_prompt": bool(prompt_text),
                },
            ):
                result = analyze_fn(analysis_chunks, guide_context, prompt_override=prompt_text)
                raw_parts = result if isinstance(result, list) else []
        else:
            with tracer.start_as_current_span(
                "analyze.resolve_fallback_provider",
                attributes={"item_id": item_id},
            ):
                summarizer = context.resolve_provider("summarization")
            with tracer.start_as_current_span(
                "analyze.fallback",
                attributes={"item_id": item_id, "chunk_count": len(chunk_dicts)},
            ):
                proc = process_analyze(
                    chunk_dicts,
                    guide_context,
                    None,
                    analyzer_provider=summarizer,
                    classifier_provider=summarizer,
                    prompt_override=prompt_text,
                )
                raw_parts = proc.get("parts") or []

        with tracer.start_as_current_span(
            "analyze.normalize_parts",
            attributes={"item_id": item_id, "raw_part_count": len(raw_parts)},
        ):
            parts = [_normalize_part(part) for part in raw_parts]
        for idx, part in enumerate(parts, start=1):
            part["part_num"] = idx
        tag_specs = prepared.get("tag_specs")
        if isinstance(tag_specs, list) and tag_specs:
            # Filter specs by _when conditions against the item
            tag_specs = _filter_specs_by_when(tag_specs, item_tags, item_id, item_summary)
            if not tag_specs:
                pass  # all specs filtered out — skip classification
            else:
                try:
                    with tracer.start_as_current_span(
                        "analyze.classify",
                        attributes={"item_id": item_id, "part_count": len(parts), "spec_count": len(tag_specs)},
                    ):
                        from ..analyzers import TagClassifier
                        provider = context.resolve_provider("summarization")
                        classifier = TagClassifier(provider=provider)
                        parts = classifier.classify(parts, specs=tag_specs)
                except Exception:
                    with tracer.start_as_current_span(
                        "analyze.classify_fallback",
                        attributes={"item_id": item_id, "part_count": len(parts)},
                    ):
                        parts = classify_parts_with_specs(
                            parts, context, item_tags=item_tags, item_id=item_id,
                            item_summary=item_summary,
                        )
        else:
            with tracer.start_as_current_span(
                "analyze.classify_fallback",
                attributes={"item_id": item_id, "part_count": len(parts)},
            ):
                parts = classify_parts_with_specs(
                    parts, context, item_tags=item_tags, item_id=item_id,
                )
        out: dict[str, Any] = {"parts": parts}

        if not parts:
            return out

        with tracer.start_as_current_span(
            "analyze.mutations",
            attributes={"item_id": item_id, "part_count": len(parts)},
        ):
            mutations: list[dict[str, Any]] = []

            # Delete old parts before inserting new ones
            mutations.append({"op": "delete_prefix", "prefix": f"{item_id}@p"})

            doc = context.get_document(item_id) if hasattr(context, "get_document") else None
            existing_tags = dict(getattr(doc, "tags", None) or {}) if doc else {}

            # Parts do NOT inherit parent tags — neither edge tags (which
            # would clone the parent's relationship graph onto every
            # fragment) nor content tags (which drift when the parent is
            # re-tagged). Each part carries only what the analyzer
            # assigned plus _base_id/_part_num bookkeeping. Search/find
            # can recover parent-tag filtering by joining through
            # _base_id when needed.
            for idx, part in enumerate(parts, start=1):
                part_id = f"{item_id}@p{idx}"
                tags = dict(part.get("tags") or {})
                tags["_base_id"] = item_id
                tags["_part_num"] = str(idx)
                mutations.append(
                    {
                        "op": "put_item",
                        "id": part_id,
                        "summary": str(part.get("summary") or ""),
                        "tags": tags,
                        "queue_background_tasks": False,
                    }
                )

            # Record _analyzed_hash so we don't re-analyze unchanged content,
            # and _analyzed_at so the throttle in `run()` can see how
            # recently this item was decomposed.
            content_hash = getattr(doc, "content_hash", None) if doc else None
            if content_hash:
                existing_tags["_analyzed_hash"] = content_hash
                existing_tags["_analyzed_at"] = utc_now()
                list_versions = getattr(context, "list_versions", None)
                if callable(list_versions):
                    versions = list_versions(item_id, limit=1)
                    if versions:
                        version = getattr(versions[0], "version", None)
                        if version is not None:
                            existing_tags["_analyzed_version"] = str(version)
                mutations.append(
                    {
                        "op": "set_tags",
                        "target": item_id,
                        "tags": existing_tags,
                    }
                )

        out["mutations"] = mutations
        return out
