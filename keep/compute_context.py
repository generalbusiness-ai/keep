"""Compute-attribution context.

Threads `work_id`, `item_id`, `flow_id` through provider calls (embed,
summarize, extract, analyze) via ContextVars, and aggregates per-flow
counters so the ops log can answer: "this flow caused N embeddings on
the GPU, and which note was responsible".

Activation: counters are tracked whenever a `counter_scope()` is open on
the current ContextVar stack. Providers call `current_counters()` to
bump; outside any scope, calls are no-ops with zero overhead beyond a
ContextVar read.

Wire-in points (see callers):
- `state_doc_runtime.run_flow` opens a scope per top-level flow.
- `work_processor` pushes work_id/item_id attribution around each work
  item, and snapshots counters into the work-queue result.
- Embedding cache, processors.summarize, processors.extract, and the
  analyzer's inner generate() each bump the appropriate counter.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class ComputeCounters:
    """Per-scope tally of provider calls that consume CPU/GPU/network.

    Thread-safe — provider calls may happen from multiple threads sharing
    a ContextVar (e.g. work queue dispatchers).
    """

    embed_hit: int = 0
    embed_miss: int = 0
    embed_ms: float = 0.0
    summarize_count: int = 0
    summarize_ms: float = 0.0
    extract_count: int = 0
    extract_ms: float = 0.0
    analyze_count: int = 0
    analyze_ms: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add_embed(self, *, hit: bool, ms: float) -> None:
        with self._lock:
            if hit:
                self.embed_hit += 1
            else:
                self.embed_miss += 1
                # Only the miss path actually runs the model — cache hits are free.
                self.embed_ms += ms

    def add_summarize(self, *, ms: float) -> None:
        with self._lock:
            self.summarize_count += 1
            self.summarize_ms += ms

    def add_extract(self, *, ms: float) -> None:
        with self._lock:
            self.extract_count += 1
            self.extract_ms += ms

    def add_analyze(self, *, ms: float) -> None:
        with self._lock:
            self.analyze_count += 1
            self.analyze_ms += ms

    def to_dict(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "embed_hit": self.embed_hit,
                "embed_miss": self.embed_miss,
                "embed_ms": round(self.embed_ms, 1),
                "summarize_count": self.summarize_count,
                "summarize_ms": round(self.summarize_ms, 1),
                "extract_count": self.extract_count,
                "extract_ms": round(self.extract_ms, 1),
                "analyze_count": self.analyze_count,
                "analyze_ms": round(self.analyze_ms, 1),
            }

    def is_empty(self) -> bool:
        with self._lock:
            return (
                self.embed_hit == 0
                and self.embed_miss == 0
                and self.summarize_count == 0
                and self.extract_count == 0
                and self.analyze_count == 0
            )


_counters: contextvars.ContextVar[ComputeCounters | None] = contextvars.ContextVar(
    "keep.compute_counters", default=None,
)
_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "keep.compute_context", default={},
)


def current_counters() -> ComputeCounters | None:
    """Return the active counters, or None when no scope is open."""
    return _counters.get()


def current_context() -> dict[str, Any]:
    """Return the active attribution context (work_id, item_id, flow_id, ...)."""
    return _context.get()


@contextmanager
def attribution(**ctx: Any) -> Iterator[dict[str, Any]]:
    """Push attribution keys onto the current context.

    Merges with any existing context (work_processor sets work_id; an inner
    run_flow can layer flow_id on top). None values are dropped so callers
    can pass optional ids without padding the log line.
    """
    merged = {**_context.get(), **{k: v for k, v in ctx.items() if v is not None}}
    token = _context.set(merged)
    try:
        yield merged
    finally:
        _context.reset(token)


@contextmanager
def counter_scope(*, label: str) -> Iterator[ComputeCounters]:
    """Begin a compute-attribution scope.

    If a scope is already active in this context (e.g. a sub-flow inside
    an outer flow), reuse the existing counters silently — the outermost
    scope emits a single rollup line so we don't double-count.

    On exit, if the scope is the outermost and any compute happened,
    emit one INFO line:
        ``compute: <label> work_id=W item_id=I wall=Nms embed=K(hit=H/miss=M,Tms) ...``
    """
    existing = _counters.get()
    if existing is not None:
        # Inner scope — let the outer one own the rollup.
        yield existing
        return

    counters = ComputeCounters()
    token = _counters.set(counters)
    t0 = time.monotonic()
    try:
        yield counters
    finally:
        # Reset the ContextVar first so failures in the logging path can't
        # leak the scope. Never `return` from inside a finally — that would
        # swallow exceptions propagating out of the `yield`.
        _counters.reset(token)
        if not counters.is_empty():
            wall_ms = (time.monotonic() - t0) * 1000.0
            ctx = _context.get()
            ctx_parts = " ".join(f"{k}={v}" for k, v in ctx.items() if v != "")
            d = counters.to_dict()
            embed_total = d["embed_hit"] + d["embed_miss"]
            logger.info(
                "compute: %s %s wall=%.0fms "
                "embed=%d(hit=%d/miss=%d,%.0fms) "
                "summarize=%d(%.0fms) extract=%d(%.0fms) analyze=%d(%.0fms)",
                label,
                ctx_parts,
                wall_ms,
                embed_total,
                d["embed_hit"],
                d["embed_miss"],
                d["embed_ms"],
                d["summarize_count"],
                d["summarize_ms"],
                d["extract_count"],
                d["extract_ms"],
                d["analyze_count"],
                d["analyze_ms"],
            )


def apply_context_attrs(span: Any) -> None:
    """Copy the active attribution context onto an OTel span.

    Lets `KEEP_TRACE=1` correlate spans back to the triggering work/flow.
    Safe to call with a no-op span.
    """
    setter = getattr(span, "set_attribute", None)
    if not callable(setter):
        return
    for k, v in _context.get().items():
        if v is None or v == "":
            continue
        try:
            setter(k, v)
        except Exception:
            # Span may reject non-primitive values; don't break the caller.
            pass
