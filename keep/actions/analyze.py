from __future__ import annotations

"""Item-scoped decomposition action for generating structured parts."""

import logging
from datetime import datetime, timezone
from typing import Any

from ..analyzers import DEFAULT_CONTEXT_BUDGET, _estimate_tokens, _parse_parts
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


def _resolve_since_version(item_tags: dict[str, Any]) -> int | None:
    """Return the last analyzed version for incremental gather, if vstring.

    URI-backed items don't have an analyzable version thread — they are
    re-fetched as a single chunk each time, so incremental gather makes
    no sense for them. A missing or unparseable ``_analyzed_version``
    means this is the first analyze for the item; fall back to full.
    """
    if item_tags.get("_source") == "uri":
        return None
    raw = item_tags.get("_analyzed_version")
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


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

        # Pick incremental gather when the item is a vstring (not a URI-backed
        # source) and we already analyzed an earlier version. Forced runs go
        # back to a full pass so a re-analyze can rebuild parts from scratch.
        force = _params_force(prepared)
        since_version = _resolve_since_version(item_tags) if not force else None

        # Force must wipe any caller-prepared chunks so we actually re-gather.
        # Two callers leave chunks pre-populated: (a) run_local_task() invokes
        # prepare() once up front, then passes the prepared params straight
        # into run(); (b) _run_incremental()'s over-budget fallback recurses
        # via self.run({**params, "force": True}, context). Without this
        # clear, the gather condition below sees chunks_targets and skips —
        # the fallback then loops back into _run_incremental() and recurses
        # until RecursionError.
        if force:
            for key in (
                "chunks", "chunks_context", "chunks_targets",
                "incremental_prompt",
            ):
                prepared.pop(key, None)

        if prepared.get("chunks") is None and prepared.get("chunks_targets") is None:
            gather_chunks = getattr(context, "gather_analyze_chunks", None)
            if callable(gather_chunks):
                with tracer.start_as_current_span(
                    "analyze.prepare.chunks",
                    attributes={
                        "item_id": item_id,
                        "since_version": since_version if since_version is not None else -1,
                    },
                ):
                    chunk_data = gather_chunks(
                        item_id, item, since_version=since_version,
                    )
                if isinstance(chunk_data, dict):
                    # Incremental shape — keep context and targets separate so
                    # run() can build the <analyze>-marked single-window prompt.
                    prepared["chunks_context"] = list(chunk_data.get("context", []))
                    prepared["chunks_targets"] = list(chunk_data.get("targets", []))
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

        # When we're going to take the incremental path, also pre-load the
        # `.prompt/analyze/incremental` doc. It's required for that path —
        # surfacing the missing-doc error here keeps run() simpler.
        if (
            prepared.get("chunks_targets")
            and prepared.get("incremental_prompt") is None
            and hasattr(context, "load_prompt_doc")
        ):
            with tracer.start_as_current_span(
                "analyze.prepare.incremental_prompt",
                attributes={"item_id": item_id},
            ):
                prepared["incremental_prompt"] = context.load_prompt_doc(
                    ".prompt/analyze/incremental", required=True,
                )

        return prepared

    def build_delegated_payload(
        self, params: dict[str, Any], content: str,
    ) -> tuple[str, dict[str, Any] | None]:
        metadata: dict[str, Any] = {}
        for key in (
            "chunks", "chunks_context", "chunks_targets",
            "guide_context", "tag_specs",
            "prompt_override", "incremental_prompt",
        ):
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

        # Incremental path: prepare() supplied {context, targets} separately.
        # The whole point is to look only at the recent versions (the targets)
        # against a small overlap of already-analyzed context — one LLM call
        # for the new trajectory, append-only parts on disk.
        target_chunks = prepared.get("chunks_targets")
        if isinstance(target_chunks, list) and target_chunks:
            context_chunks = prepared.get("chunks_context") or []
            return self._run_incremental(
                item_id, item_tags, item_summary,
                context_chunks=list(context_chunks),
                target_chunks=list(target_chunks),
                guide_context=guide_context,
                incremental_prompt=prepared.get("incremental_prompt"),
                tag_specs=prepared.get("tag_specs"),
                context=context,
                params=params,
            )

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

    def _run_incremental(
        self,
        item_id: str,
        item_tags: dict[str, Any],
        item_summary: str,
        *,
        context_chunks: list[dict[str, Any]],
        target_chunks: list[dict[str, Any]],
        guide_context: str,
        incremental_prompt: str | None,
        tag_specs: Any,
        context: Any,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """One-shot incremental analysis of new versions over an overlap.

        Prior context (already-analyzed versions) goes outside ``<analyze>``;
        the new versions plus the current note state go inside. Parts emitted
        here are appended to whatever already exists — we don't ``delete_prefix``
        the way the full path does, so the previously-analyzed trajectory stays
        intact and the LLM only narrates what's genuinely new.

        Falls back to the full path when the assembled window would exceed
        ``DEFAULT_CONTEXT_BUDGET`` — at that size the single-LLM-call savings
        disappear and we'd rather rebuild the decomposition.
        """
        if not incremental_prompt:
            raise ValueError("missing prompt doc for incremental analyze")

        total_tokens = sum(
            _estimate_tokens(str(c.get("content", "")))
            for c in context_chunks + target_chunks
        )
        if total_tokens > DEFAULT_CONTEXT_BUDGET:
            logger.info(
                "Incremental analyze content too large (%d tokens > %d budget) "
                "for %s — falling back to full analysis",
                total_tokens, DEFAULT_CONTEXT_BUDGET, item_id,
            )
            # Strip the incremental keys before recursing so prepare() actually
            # re-gathers as a full pass. force=True alone is not enough — the
            # `if force: prepared.pop(...)` block in prepare() does the work,
            # but we drop them here too as belt-and-braces in case a future
            # refactor changes that contract.
            fallback_params = {
                k: v for k, v in params.items()
                if k not in {
                    "chunks", "chunks_context", "chunks_targets",
                    "incremental_prompt",
                }
            }
            fallback_params["force"] = True
            return self.run(fallback_params, context)

        prompt_parts: list[str] = ["<content>"]
        for c in context_chunks:
            prompt_parts.append(str(c.get("content", "")))
        prompt_parts.append("<analyze>")
        for c in target_chunks:
            prompt_parts.append(str(c.get("content", "")))
        prompt_parts.append("</analyze>")
        prompt_parts.append("</content>")
        user_prompt = "\n\n".join(prompt_parts)
        if guide_context:
            user_prompt = f"{guide_context}\n\n---\n\n{user_prompt}"

        # Mirror Keeper.analyze()'s provider unwrapping — the caching wrapper
        # exposes a `_provider` that hands back the raw generate() callable.
        with tracer.start_as_current_span(
            "analyze.incremental.resolve_provider",
            attributes={"item_id": item_id},
        ):
            provider = context.resolve_provider("summarization")
        raw_provider = provider
        if hasattr(raw_provider, "_provider") and raw_provider._provider is not None:
            raw_provider = raw_provider._provider

        with tracer.start_as_current_span(
            "analyze.incremental.provider",
            attributes={
                "item_id": item_id,
                "context_chunks": len(context_chunks),
                "target_chunks": len(target_chunks),
                "prompt_chars": len(user_prompt),
                "estimated_tokens": total_tokens,
            },
        ):
            try:
                result_text = raw_provider.generate(
                    incremental_prompt, user_prompt, max_tokens=4096,
                )
            except Exception as e:
                logger.warning(
                    "Incremental analyze LLM call failed for %s: %s", item_id, e,
                )
                result_text = None

        raw_parts = _parse_parts(result_text) if result_text else []
        parts = [_normalize_part(part) for part in raw_parts]

        if isinstance(tag_specs, list) and tag_specs and parts:
            filtered_specs = _filter_specs_by_when(
                tag_specs, item_tags, item_id, item_summary,
            )
            if filtered_specs:
                try:
                    with tracer.start_as_current_span(
                        "analyze.incremental.classify",
                        attributes={
                            "item_id": item_id,
                            "part_count": len(parts),
                            "spec_count": len(filtered_specs),
                        },
                    ):
                        from ..analyzers import TagClassifier
                        classifier = TagClassifier(
                            provider=context.resolve_provider("summarization"),
                        )
                        parts = classifier.classify(parts, specs=filtered_specs)
                except Exception:
                    with tracer.start_as_current_span(
                        "analyze.incremental.classify_fallback",
                        attributes={"item_id": item_id, "part_count": len(parts)},
                    ):
                        parts = classify_parts_with_specs(
                            parts, context, item_tags=item_tags,
                            item_id=item_id, item_summary=item_summary,
                        )

        # Record _analyzed_version even when the LLM found nothing new — that
        # is the correct signal that we have already considered these versions
        # and should not analyze them again. Without this, every subsequent
        # write would re-gather the same context/target pair.
        mutations: list[dict[str, Any]] = []
        doc = context.get_document(item_id) if hasattr(context, "get_document") else None

        if parts:
            # Continue part numbering from whatever's on disk so we don't
            # collide with the existing parts (which we deliberately don't
            # delete in the incremental path).
            max_part = 0
            get_max_part = getattr(context, "max_part_num", None)
            if callable(get_max_part):
                try:
                    max_part = int(get_max_part(item_id) or 0)
                except Exception:
                    max_part = 0
            for offset, part in enumerate(parts, start=1):
                idx = max_part + offset
                part["part_num"] = idx
                tags = dict(part.get("tags") or {})
                tags["_base_id"] = item_id
                tags["_part_num"] = str(idx)
                mutations.append(
                    {
                        "op": "put_item",
                        "id": f"{item_id}@p{idx}",
                        "summary": str(part.get("summary") or ""),
                        "tags": tags,
                        "queue_background_tasks": False,
                    }
                )

        existing_tags = dict(getattr(doc, "tags", None) or {}) if doc else {}
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

        return {"parts": parts, "mutations": mutations, "incremental": True}
