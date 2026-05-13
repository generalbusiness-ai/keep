"""Tests for keep.compute_context."""

from __future__ import annotations

import logging
import re

import pytest

from keep.compute_context import (
    ComputeCounters,
    apply_context_attrs,
    attribution,
    counter_scope,
    current_context,
    current_counters,
)


class TestComputeCounters:
    def test_starts_empty(self):
        c = ComputeCounters()
        assert c.is_empty()
        d = c.to_dict()
        assert d["embed_hit"] == 0 and d["embed_miss"] == 0
        assert d["summarize_count"] == 0 and d["extract_count"] == 0
        assert d["analyze_count"] == 0

    def test_add_embed_hit_does_not_charge_time(self):
        c = ComputeCounters()
        c.add_embed(hit=True, ms=123.0)
        d = c.to_dict()
        assert d["embed_hit"] == 1
        assert d["embed_miss"] == 0
        # Cache hits are free — only misses should accumulate compute ms.
        assert d["embed_ms"] == 0.0
        assert not c.is_empty()

    def test_add_embed_miss_charges_time(self):
        c = ComputeCounters()
        c.add_embed(hit=False, ms=50.0)
        c.add_embed(hit=False, ms=70.0)
        d = c.to_dict()
        assert d["embed_miss"] == 2
        assert d["embed_ms"] == 120.0

    def test_summarize_extract_analyze(self):
        c = ComputeCounters()
        c.add_summarize(ms=1500.0)
        c.add_extract(ms=200.0)
        c.add_analyze(ms=800.0)
        d = c.to_dict()
        assert d["summarize_count"] == 1 and d["summarize_ms"] == 1500.0
        assert d["extract_count"] == 1 and d["extract_ms"] == 200.0
        assert d["analyze_count"] == 1 and d["analyze_ms"] == 800.0


class TestContextVars:
    def test_no_scope_returns_none(self):
        assert current_counters() is None
        assert current_context() == {}

    def test_attribution_pushes_and_pops(self):
        with attribution(work_id="w1", item_id="x"):
            ctx = current_context()
            assert ctx["work_id"] == "w1"
            assert ctx["item_id"] == "x"
        assert current_context() == {}

    def test_attribution_drops_none_values(self):
        with attribution(work_id="w1", item_id=None, flow_id="f"):
            ctx = current_context()
            assert ctx == {"work_id": "w1", "flow_id": "f"}

    def test_nested_attribution_merges(self):
        with attribution(work_id="w1"):
            with attribution(flow_id="put"):
                ctx = current_context()
                assert ctx == {"work_id": "w1", "flow_id": "put"}
            # Inner scope reset; outer still active.
            assert current_context() == {"work_id": "w1"}


class TestCounterScope:
    def test_creates_counters_inside_scope(self):
        with counter_scope(label="test") as counters:
            assert current_counters() is counters
            counters.add_embed(hit=True, ms=0.0)
        # Scope exited — ContextVar reset.
        assert current_counters() is None

    def test_nested_scope_shares_outer_counters(self):
        with counter_scope(label="outer") as outer:
            with counter_scope(label="inner") as inner:
                # Inner reuses the outer counters; otherwise inner's compute
                # would not show up in the outer's rollup.
                assert inner is outer
                inner.add_embed(hit=False, ms=10.0)
            # Outer still in scope and sees the inner mutation.
            assert current_counters() is outer
            assert outer.to_dict()["embed_miss"] == 1

    def test_emits_rollup_log_on_exit(self, caplog):
        caplog.set_level(logging.INFO, logger="keep.compute_context")
        with counter_scope(label="flow:put") as c:
            c.add_embed(hit=False, ms=42.5)
            c.add_summarize(ms=1500.0)
        rollup = [r for r in caplog.records if "compute:" in r.getMessage()]
        assert len(rollup) == 1
        msg = rollup[0].getMessage()
        assert "flow:put" in msg
        assert "embed=1(hit=0/miss=1" in msg
        assert "summarize=1" in msg
        # wall= must be present and look like a number followed by ms.
        assert re.search(r"wall=\d+ms", msg)

    def test_empty_scope_emits_no_log(self, caplog):
        caplog.set_level(logging.INFO, logger="keep.compute_context")
        with counter_scope(label="flow:get"):
            pass
        assert not [r for r in caplog.records if "compute:" in r.getMessage()]

    def test_exception_propagates_through_scope(self, caplog):
        caplog.set_level(logging.INFO, logger="keep.compute_context")
        with pytest.raises(RuntimeError, match="boom"):
            with counter_scope(label="flow:put") as c:
                c.add_embed(hit=False, ms=10.0)
                raise RuntimeError("boom")
        # Counter scope must not swallow the exception, but the rollup line
        # should still have been emitted before propagation.
        rollup = [r for r in caplog.records if "compute:" in r.getMessage()]
        assert len(rollup) == 1
        assert "embed=1" in rollup[0].getMessage()

    def test_attribution_attrs_appear_in_rollup(self, caplog):
        caplog.set_level(logging.INFO, logger="keep.compute_context")
        with attribution(work_id="w-123", item_id="now"):
            with counter_scope(label="flow:put") as c:
                c.add_summarize(ms=10.0)
        msg = next(r.getMessage() for r in caplog.records if "compute:" in r.getMessage())
        assert "work_id=w-123" in msg
        assert "item_id=now" in msg


class TestApplyContextAttrs:
    class _FakeSpan:
        def __init__(self):
            self.attrs: dict[str, object] = {}

        def set_attribute(self, k, v):
            self.attrs[k] = v

    def test_copies_active_context_onto_span(self):
        span = self._FakeSpan()
        with attribution(work_id="w1", item_id="now", flow_id="put"):
            apply_context_attrs(span)
        assert span.attrs == {"work_id": "w1", "item_id": "now", "flow_id": "put"}

    def test_skips_empty_values(self):
        span = self._FakeSpan()
        with attribution(work_id="w1", flow_id=""):
            apply_context_attrs(span)
        assert span.attrs == {"work_id": "w1"}

    def test_no_op_on_object_without_set_attribute(self):
        # Should not raise — used against potentially-no-op OTel spans.
        apply_context_attrs(object())
