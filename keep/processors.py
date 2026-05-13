"""Pure processing helpers for keep.

These functions encapsulate the "compute" portion of background processing
(summarization, OCR, analysis) without any store reads or writes. They return
plain dicts so callers can compose them into action outputs without relying on
an intermediate result object contract.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from .compute_context import apply_context_attrs, current_counters
from .tracing import get_tracer

logger = logging.getLogger(__name__)

_tracer = get_tracer("processors")


# --- Hash functions (used by OCR processing and document dedup) ---

def _content_hash(content: str) -> str:
    """Short SHA256 hash of content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[-10:]


def _content_hash_full(content: str) -> str:
    """Full SHA256 hash of content for dedup verification."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# --- Task type constants ---

# Task types that can be delegated to the hosted service
DELEGATABLE_TASK_TYPES = ("summarize", "ocr", "analyze")

# Task types that must run locally (need local store access)
LOCAL_ONLY_TASK_TYPES = ("embed", "reindex")

# MIME type → file extension for OCR temp files
MIME_TO_EXTENSION = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}

def _llm_summarize(
    content: str,
    provider,
    *,
    context: str | None = None,
    system_prompt_override: str | None = None,
) -> str | None:
    """Build prompts and call provider.generate().  Returns None for non-LLM providers."""
    from .providers.base import (
        build_summarization_prompt,
        SUMMARIZATION_SYSTEM_PROMPT,
        strip_summary_preamble,
    )

    gen = getattr(provider, "generate", None)
    if gen is None:
        return None

    truncated = content[:50000] if len(content) > 50000 else content
    user_prompt = build_summarization_prompt(truncated, context)
    system = system_prompt_override or SUMMARIZATION_SYSTEM_PROMPT
    if context and not system_prompt_override:
        system = (
            "You are a helpful assistant that summarizes content. "
            "Follow the instructions in the user message."
        )
    with _tracer.start_as_current_span(
        "summarize.request",
        attributes={
            "provider": type(provider).__name__,
            "content_len": len(content),
            "truncated_len": len(truncated),
            "kind": "llm",
        },
    ) as span:
        apply_context_attrs(span)
        t0 = time.monotonic()
        result = gen(system, user_prompt)
        ms = (time.monotonic() - t0) * 1000.0
    if (c := current_counters()) is not None:
        c.add_summarize(ms=ms)
    if result is None:
        return None
    return strip_summary_preamble(result)


def process_summarize(
    content: str,
    *,
    context: str | None = None,
    summarization_provider,
    system_prompt_override: str | None = None,
) -> dict[str, str]:
    """Summarize content.  Pure function — no store access."""
    summary = _llm_summarize(
        content, summarization_provider,
        context=context, system_prompt_override=system_prompt_override,
    )
    if summary is None:
        # Non-LLM fallback (truncate, first_paragraph) — record as a summarize call
        # too, since it's the same operation from the caller's POV.
        with _tracer.start_as_current_span(
            "summarize.request",
            attributes={
                "provider": type(summarization_provider).__name__,
                "content_len": len(content),
                "kind": "fallback",
            },
        ) as span:
            apply_context_attrs(span)
            t0 = time.monotonic()
            summary = summarization_provider.summarize(content, context=context)
            ms = (time.monotonic() - t0) * 1000.0
        if (c := current_counters()) is not None:
            c.add_summarize(ms=ms)
    return {"summary": summary}


def ocr_image(path: Path, content_type: str, extractor) -> str | None:
    """OCR a single image file.  Returns cleaned text or None."""
    from .providers.documents import FileDocumentProvider

    with _tracer.start_as_current_span(
        "extract.image",
        attributes={
            "extractor": type(extractor).__name__,
            "content_type": content_type,
        },
    ) as span:
        apply_context_attrs(span)
        t0 = time.monotonic()
        text = extractor.extract(str(path), content_type)
        ms = (time.monotonic() - t0) * 1000.0
    if (c := current_counters()) is not None:
        c.add_extract(ms=ms)
    if not text:
        return None
    cleaned = FileDocumentProvider._clean_ocr_text(text)
    confidence = FileDocumentProvider._estimate_ocr_confidence(cleaned)
    if confidence < 0.3 or len(cleaned) <= 10:
        logger.info("Image OCR low confidence (%.2f) for %s", confidence, path.name)
        return None
    return cleaned


def ocr_pdf(path: Path, ocr_pages: list[int], extractor) -> str | None:
    """OCR scanned PDF pages and merge with text-layer pages.  Returns text or None."""
    from .providers.documents import FileDocumentProvider

    file_provider = FileDocumentProvider()
    with _tracer.start_as_current_span(
        "extract.pdf",
        attributes={
            "extractor": type(extractor).__name__,
            "page_count": len(ocr_pages),
        },
    ) as span:
        apply_context_attrs(span)
        t0 = time.monotonic()
        ocr_results = file_provider._ocr_pdf_pages(path, ocr_pages, extractor=extractor)
        ms = (time.monotonic() - t0) * 1000.0
    if (c := current_counters()) is not None:
        c.add_extract(ms=ms)

    if not ocr_results:
        return None

    # Re-extract text pages (fast) and merge with OCR
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text_parts: list[tuple[int, str]] = []
    ocr_set = set(ocr_pages)
    for i, page in enumerate(reader.pages):
        if i not in ocr_set:
            text = page.extract_text()
            if text and text.strip():
                text_parts.append((i, text))
    text_parts.extend(ocr_results)
    text_parts.sort(key=lambda t: t[0])
    return "\n\n".join(text for _, text in text_parts)


def process_ocr(
    full_content: str,
    *,
    max_summary_length: int,
    context: str | None = None,
    summarization_provider=None,
) -> dict[str, str]:
    """Process OCR'd text: summarize if needed, compute hashes.

    Pure function — no store access.
    """
    if len(full_content) <= max_summary_length:
        summary = full_content
    elif summarization_provider:
        summary = _llm_summarize(full_content, summarization_provider, context=context)
        if summary is None:
            summary = summarization_provider.summarize(full_content, context=context)
    else:
        summary = full_content[:max_summary_length] + "..."

    return {
        "summary": summary,
        "content": full_content,
        "content_hash": _content_hash(full_content),
        "content_hash_full": _content_hash_full(full_content),
    }


def process_analyze(
    chunks: list[dict],
    guide_context: str = "",
    tag_specs: list[dict] | None = None,
    *,
    analyzer_provider=None,
    classifier_provider=None,
    prompt_override: str | None = None,
) -> dict[str, list[dict]]:
    """Analyze + classify content into parts.  Pure function — no store access.

    Args:
        chunks: Serializable chunk dicts [{"content": str, "tags": dict, "index": int}].
        guide_context: Tag guidance descriptions for the analyzer.
        tag_specs: Tag taxonomy specs for classification (from TagClassifier.load_specs).
        analyzer_provider: SummarizationProvider for the analyzer LLM.
        classifier_provider: SummarizationProvider for the classifier LLM
            (defaults to analyzer_provider if not set).
        prompt_override: Analysis prompt text from .prompt/analyze/* docs.

    Returns:
        Dict with parts=[{"summary": str, "tags": dict}, ...]
    """
    from .analyzers import SlidingWindowAnalyzer, TagClassifier
    from .providers.base import AnalysisChunk

    # Reconstruct AnalysisChunk objects from serializable dicts
    analysis_chunks = [
        AnalysisChunk(
            content=c.get("content", ""),
            tags=c.get("tags", {}),
            index=c.get("index", i),
        )
        for i, c in enumerate(chunks)
    ]

    # Run analyzer
    analyzer = SlidingWindowAnalyzer(provider=analyzer_provider)
    raw_parts = analyzer.analyze(
        analysis_chunks, guide_context, prompt_override=prompt_override,
    )

    # Run classifier if tag specs provided
    if tag_specs and raw_parts:
        try:
            classifier = TagClassifier(
                provider=classifier_provider or analyzer_provider,
            )
            classifier.classify(raw_parts, tag_specs)
        except Exception as e:
            logger.warning("Tag classification skipped: %s", e)

    return {"parts": raw_parts}
