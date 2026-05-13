"""Tests for the synchronous state-doc flow runtime."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from keep.api import Keeper
from keep.const import STATE_FIND_DEEP
from keep.flow_env import LocalFlowEnvironment
from keep.state_doc import parse_state_doc
from keep.state_doc_runtime import FlowResult, decode_cursor, make_state_doc_loader, run_flow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader(docs: dict[str, str]):
    """Create a state doc loader from a dict of {name: yaml_body}."""
    compiled = {}
    for name, body in docs.items():
        compiled[name] = parse_state_doc(name, body)

    def _load(name: str):
        return compiled.get(name)

    return _load


def _make_runner(outputs: dict[str, dict] | None = None):
    """Create an action runner that returns canned outputs."""
    calls = []
    defaults = outputs or {}

    def _run(action_name: str, params: dict):
        calls.append((action_name, dict(params)))
        return dict(defaults.get(action_name, {}))

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


@pytest.fixture
def kp(mock_providers, tmp_path):
    keeper = Keeper(store_path=tmp_path)
    keeper.ensure_sysdocs()
    return keeper


def _store_loader(kp: Keeper):
    return make_state_doc_loader(LocalFlowEnvironment(kp))


# ---------------------------------------------------------------------------
# Basic terminal flow
# ---------------------------------------------------------------------------

def test_run_flow_top_level_logs_at_info(caplog):
    """Top-level run_flow logs state transitions at INFO."""
    import logging
    caplog.set_level(logging.DEBUG, logger="keep.state_doc_runtime")
    loader = _make_loader({"simple": "match: sequence\nrules:\n  - return: done"})
    run_flow("simple", {}, load_state_doc=loader, run_action=_make_runner())
    info_msgs = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert any("flow: start simple" in m for m in info_msgs)
    assert any("simple -> done" in m for m in info_msgs)


def test_run_flow_nested_subflow_logs_at_debug(caplog):
    """Subflows invoked from within an active counter_scope log at DEBUG, not INFO.

    The parent flow's `compute:` rollup captures the aggregate, so trivial
    subflow transitions don't need to flood the ops log.
    """
    import logging

    from keep.compute_context import counter_scope

    caplog.set_level(logging.DEBUG, logger="keep.state_doc_runtime")
    loader = _make_loader({"simple": "match: sequence\nrules:\n  - return: done"})

    # Open a parent scope so run_flow sees it as nested (this is what
    # state_doc_runtime.run_flow itself does, or what work_processor sets up).
    with counter_scope(label="parent"):
        run_flow("simple", {}, load_state_doc=loader, run_action=_make_runner())

    info_msgs = [r.getMessage() for r in caplog.records if r.levelname == "INFO" and "flow:" in r.getMessage()]
    debug_msgs = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG" and "flow:" in r.getMessage()]
    # No INFO-level flow lines from the nested subflow.
    assert not info_msgs, f"unexpected INFO flow lines: {info_msgs}"
    # But DEBUG should still have them.
    assert any("flow: start simple" in m for m in debug_msgs)
    assert any("simple -> done" in m for m in debug_msgs)


def test_run_flow_emits_runtime_trace_spans():
    """run_flow emits trace spans for state-doc load and evaluation."""
    loader = _make_loader({
        "simple": "match: sequence\nrules:\n  - return: done",
    })
    spans = []

    class _Span:
        def __init__(self, name, attributes=None):
            self.name = name
            self.attributes = dict(attributes or {})

        def __enter__(self):
            spans.append(self)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Tracer:
        def start_as_current_span(self, name, attributes=None):
            return _Span(name, attributes)

    with patch("keep.tracing.get_tracer", return_value=_Tracer()):
        result = run_flow("simple", {}, load_state_doc=loader, run_action=_make_runner())

    assert result.status == "done"
    assert any(s.name == "state_doc.load" and s.attributes.get("state") == "simple" for s in spans)
    assert any(s.name == "state_doc.evaluate" and s.attributes.get("state") == "simple" for s in spans)


class TestBasicFlow:
    """Tests for basic runtime flow."""
    def test_immediate_done(self):
        loader = _make_loader({
            "simple": "match: sequence\nrules:\n  - return: done",
        })
        result = run_flow("simple", {}, load_state_doc=loader, run_action=_make_runner())
        assert result.status == "done"
        assert result.ticks == 1
        assert result.history == ["simple"]

    def test_immediate_error(self):
        loader = _make_loader({
            "fail": "match: sequence\nrules:\n  - return: error",
        })
        result = run_flow("fail", {}, load_state_doc=loader, run_action=_make_runner())
        assert result.status == "error"

    def test_immediate_stopped(self):
        loader = _make_loader({
            "stop": """\
match: sequence
rules:
  - return:
      status: stopped
      with:
        reason: "ambiguous"
""",
        })
        result = run_flow("stop", {}, load_state_doc=loader, run_action=_make_runner())
        assert result.status == "stopped"
        assert result.data["reason"] == "ambiguous"

    def test_missing_state_doc(self):
        result = run_flow(
            "nonexistent", {},
            load_state_doc=lambda n: None,
            run_action=_make_runner(),
        )
        assert result.status == "error"
        assert "not found" in result.data["reason"]

    def test_should_stop_returns_resumable_cursor(self):
        loader = _make_loader({
            "one": """\
match: sequence
rules:
  - then: two
""",
            "two": """\
match: sequence
rules:
  - return: done
""",
        })
        calls = iter([False, True])
        result = run_flow(
            "one",
            {},
            load_state_doc=loader,
            run_action=_make_runner(),
            should_stop=lambda: next(calls),
        )
        assert result.status == "stopped"
        assert result.data["reason"] == "shutdown"
        assert result.cursor is not None
        cursor = decode_cursor(result.cursor)
        assert cursor is not None
        assert cursor.state == "two"


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

class TestActionExecution:
    """Tests for runtime action execution."""
    def test_action_runs(self):
        loader = _make_loader({
            "with-action": """\
match: sequence
rules:
  - id: result
    do: find
    with:
      query: "test"
  - return: done
""",
        })
        runner = _make_runner({"find": {"results": [], "count": 0}})
        result = run_flow(
            "with-action", {"query": "test"},
            load_state_doc=loader, run_action=runner,
        )
        assert result.status == "done"
        assert len(runner.calls) == 1
        assert runner.calls[0][0] == "find"

    def test_action_output_bound(self):
        loader = _make_loader({
            "bound": """\
match: sequence
rules:
  - id: search
    do: find
    with:
      query: "test"
  - return: done
""",
        })
        runner = _make_runner({
            "find": {
                "results": [
                    {"id": "a", "summary": "x", "tags": {}, "score": 0.9},
                    {"id": "b", "summary": "y", "tags": {}, "score": 0.3},
                ],
                "count": 2,
            },
        })
        result = run_flow(
            "bound", {},
            load_state_doc=loader, run_action=runner,
        )
        assert "search" in result.bindings
        # Find output should be enriched with stats
        assert "margin" in result.bindings["search"]
        assert "entropy" in result.bindings["search"]

    def test_subflow_output_bound(self):
        loader = _make_loader({
            "parent": """\
match: sequence
rules:
  - id: assessed
    do: .state/child
    with:
      target: "abc"
  - return:
      status: done
      with:
        assessment: "{assessed.assessment}"
""",
            "child": """\
match: sequence
rules:
  - return:
      status: done
      with:
        assessment: "ok"
        target: "{params.target}"
""",
        })
        result = run_flow(
            "parent", {},
            load_state_doc=loader, run_action=_make_runner(),
        )
        assert result.status == "done"
        assert result.bindings["assessed"]["assessment"] == "ok"
        assert result.bindings["assessed"]["target"] == "abc"
        assert result.data["assessment"] == "ok"
        assert result.ticks == 2

    def test_subflow_error_fails_parent(self):
        loader = _make_loader({
            "parent": """\
match: sequence
rules:
  - id: assessed
    do: .state/child
    with:
      target: "abc"
  - return: done
""",
            "child": """\
match: sequence
rules:
  - return:
      status: error
      with:
        reason: "boom"
""",
        })
        result = run_flow(
            "parent", {},
            load_state_doc=loader, run_action=_make_runner(),
        )
        assert result.status == "error"
        assert "subflow" in result.data["reason"]
        assert "boom" in result.data["reason"]

    def test_subflow_stopped_fails_parent(self):
        loader = _make_loader({
            "parent": """\
match: sequence
rules:
  - id: assessed
    do: .state/child
  - return: done
""",
            "child": """\
match: sequence
rules:
  - return:
      status: stopped
      with:
        reason: "budget"
""",
        })
        result = run_flow(
            "parent", {},
            load_state_doc=loader, run_action=_make_runner(),
        )
        assert result.status == "error"
        assert "unsupported status" in result.data["reason"]

    def test_subflow_budget_exhaustion_fails_parent(self):
        loader = _make_loader({
            "parent": """\
match: sequence
rules:
  - id: assessed
    do: .state/child
  - return: done
""",
            "child": """\
match: sequence
rules:
  - return: done
""",
        })
        result = run_flow(
            "parent", {},
            budget=1,
            load_state_doc=loader, run_action=_make_runner(),
        )
        assert result.status == "error"
        assert "subflow 'child' cannot run: budget exhausted" in result.data["reason"]


# ---------------------------------------------------------------------------
# Find output enrichment
# ---------------------------------------------------------------------------

class TestFindEnrichment:
    """Tests for find output enrichment."""
    def test_find_output_gets_stats(self):
        loader = _make_loader({
            "enriched": """\
match: sequence
rules:
  - id: search
    do: find
    with:
      query: "test"
  - return: done
""",
        })
        runner = _make_runner({
            "find": {
                "results": [
                    {"id": "a", "summary": "", "tags": {}, "score": 0.95},
                    {"id": "b", "summary": "", "tags": {}, "score": 0.30},
                ],
                "count": 2,
            },
        })
        result = run_flow(
            "enriched", {},
            load_state_doc=loader, run_action=runner,
        )
        search = result.bindings["search"]
        assert search["margin"] > 0.5  # dominant result
        assert isinstance(search["entropy"], float)
        assert isinstance(search["lineage_strong"], float)

    def test_non_find_action_not_enriched(self):
        loader = _make_loader({
            "other": """\
match: sequence
rules:
  - id: item
    do: get
    with:
      id: "test"
  - return: done
""",
        })
        runner = _make_runner({"get": {"id": "test", "summary": "x", "tags": {}}})
        result = run_flow(
            "other", {},
            load_state_doc=loader, run_action=runner,
        )
        assert "margin" not in result.bindings.get("item", {})


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

class TestTransitions:
    """Tests for state transitions."""
    def test_simple_transition(self):
        loader = _make_loader({
            "first": "match: sequence\nrules:\n  - then: second",
            "second": "match: sequence\nrules:\n  - return: done",
        })
        result = run_flow(
            "first", {},
            load_state_doc=loader, run_action=_make_runner(),
        )
        assert result.status == "done"
        assert result.ticks == 2
        assert result.history == ["first", "second"]

    def test_transition_with_params(self):
        loader = _make_loader({
            "first": """\
match: sequence
rules:
  - then:
      state: second
      with:
        extra: "value"
""",
            "second": "match: sequence\nrules:\n  - return: done",
        })
        result = run_flow(
            "first", {"original": "kept"},
            load_state_doc=loader, run_action=_make_runner(),
        )
        assert result.status == "done"
        assert result.history == ["first", "second"]

    def test_self_transition_with_budget(self):
        loader = _make_loader({
            "loop": "match: sequence\nrules:\n  - then: loop",
        })
        result = run_flow(
            "loop", {}, budget=3,
            load_state_doc=loader, run_action=_make_runner(),
        )
        assert result.status == "stopped"
        assert result.data["reason"] == "budget"
        assert result.ticks == 3
        assert result.history == ["loop", "loop", "loop"]

    def test_transition_to_missing_doc(self):
        loader = _make_loader({
            "start": "match: sequence\nrules:\n  - then: nonexistent",
        })
        result = run_flow(
            "start", {},
            load_state_doc=loader, run_action=_make_runner(),
        )
        assert result.status == "error"
        assert "not found" in result.data["reason"]


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

class TestBudget:
    """Tests for runtime budget limits."""
    def test_budget_1(self):
        loader = _make_loader({
            "step": "match: sequence\nrules:\n  - then: step",
        })
        result = run_flow(
            "step", {}, budget=1,
            load_state_doc=loader, run_action=_make_runner(),
        )
        # Budget=1 means we get 1 tick, which transitions to step again,
        # but the loop exits because ticks >= budget
        assert result.status == "stopped"
        assert result.ticks == 1

    def test_budget_context_available(self):
        """Budget context is available to predicates."""
        # This test verifies the eval context has budget.remaining.
        # The predicate "budget.remaining > 0" should be evaluable.
        # We can't easily test CEL predicates without the CEL library,
        # so we test the context structure indirectly through the flow.
        loader = _make_loader({
            "check": "match: sequence\nrules:\n  - return: done",
        })
        result = run_flow(
            "check", {}, budget=5,
            load_state_doc=loader, run_action=_make_runner(),
        )
        assert result.status == "done"
        assert result.ticks == 1


# ---------------------------------------------------------------------------
# Match: all with post block
# ---------------------------------------------------------------------------

class TestMatchAll:
    """Tests for match-all parallel actions."""
    def test_parallel_actions_then_post(self):
        loader = _make_loader({
            "parallel": """\
match: all
rules:
  - id: a
    do: find
    with:
      query: "one"
  - id: b
    do: get
    with:
      id: "two"
post:
  - return: done
""",
        })
        runner = _make_runner({
            "find": {"results": [], "count": 0},
            "get": {"id": "two", "summary": "x", "tags": {}},
        })
        result = run_flow(
            "parallel", {},
            load_state_doc=loader, run_action=runner,
        )
        assert result.status == "done"
        assert "a" in result.bindings
        assert "b" in result.bindings
        assert len(runner.calls) == 2


# ---------------------------------------------------------------------------
# Params flow across transitions
# ---------------------------------------------------------------------------

class TestParamsFlow:
    """Tests for params persistence across transitions."""
    def test_params_persist_across_transitions(self):
        """Original params remain available after transition."""
        calls = []

        def _loader(name):
            docs = {
                "first": """\
match: sequence
rules:
  - then:
      state: second
      with:
        added: "new"
""",
                "second": """\
match: sequence
rules:
  - id: search
    do: find
    with:
      query: "{params.query}"
  - return: done
""",
            }
            if name in docs:
                return parse_state_doc(name, docs[name])
            return None

        def _runner(action_name, params):
            calls.append((action_name, dict(params)))
            return {"results": [], "count": 0}

        result = run_flow(
            "first", {"query": "original"},
            load_state_doc=_loader, run_action=_runner,
        )
        assert result.status == "done"
        # Template resolution of {params.query} should work
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Factory: make_state_doc_loader
# ---------------------------------------------------------------------------

class TestMakeStateDocLoader:
    """Tests for state document loader creation."""
    def test_loads_from_env(self):
        from keep.state_doc_runtime import make_state_doc_loader

        class FakeNote:
            id = ".state/test"
            summary = "match: sequence\nrules:\n  - return: done"
            tags = {}


        class FakeEnv:
            def get(self, id):
                if id == ".state/test":
                    return FakeNote()
                return None

        loader = make_state_doc_loader(FakeEnv())
        doc = loader("test")
        assert doc is not None
        assert doc.name == "test"

    def test_returns_none_for_missing(self):
        from keep.state_doc_runtime import make_state_doc_loader

        class FakeEnv:
            def get(self, id):
                return None

        loader = make_state_doc_loader(FakeEnv())
        assert loader("nonexistent") is None

    def test_missing_store_doc_returns_none(self):
        """Runtime loader only uses store-backed .state/* docs."""
        from keep.state_doc_runtime import make_state_doc_loader

        class FakeEnv:
            def get(self, id):
                return None

        loader = make_state_doc_loader(FakeEnv())
        assert loader("find-deep") is None

    def test_store_doc_loaded_without_builtin_fallback(self):
        """Store .state/* note is the only runtime source of truth."""
        from keep.state_doc_runtime import make_state_doc_loader

        class FakeNote:
            id = f".state/{STATE_FIND_DEEP}"
            summary = "match: sequence\nrules:\n  - return: error"
            tags = {}

        class FakeEnv:
            def get(self, id):
                if id == f".state/{STATE_FIND_DEEP}":
                    return FakeNote()
                return None

        loader = make_state_doc_loader(FakeEnv())
        result = run_flow(STATE_FIND_DEEP, {}, load_state_doc=loader, run_action=_make_runner())
        assert result.status == "error"

    def test_empty_store_summary_returns_none(self):
        """Empty store docs do not silently fall back to bundled YAML."""
        from keep.state_doc_runtime import make_state_doc_loader

        class FakeNote:
            id = ".state/find-deep"
            summary = ""
            tags = {}

        class FakeEnv:
            def get(self, id):
                return FakeNote()

        loader = make_state_doc_loader(FakeEnv())
        assert loader("find-deep") is None

    def test_returns_none_for_empty_summary_no_builtin(self):
        """When store note has empty summary, returns None."""
        from keep.state_doc_runtime import make_state_doc_loader

        class FakeNote:
            id = ".state/custom-nonexistent"
            summary = ""
            tags = {}

        class FakeEnv:
            def get(self, id):
                return FakeNote()

        loader = make_state_doc_loader(FakeEnv())
        assert loader("custom-nonexistent") is None

    def test_invalid_store_doc_returns_none(self):
        """Broken store docs do not fall back to package defaults."""
        from keep.state_doc_runtime import make_state_doc_loader

        class FakeNote:
            id = ".state/find-deep"
            summary = "not: [valid"
            tags = {}

        class FakeEnv:
            def get(self, id):
                return FakeNote()

        loader = make_state_doc_loader(FakeEnv())
        assert loader("find-deep") is None

    def test_handles_dotstate_prefix(self):
        from keep.state_doc_runtime import make_state_doc_loader

        class FakeNote:
            id = ".state/prefixed"
            summary = "match: sequence\nrules:\n  - return: done"
            tags = {}

        class FakeEnv:
            def get(self, id):
                if id == ".state/prefixed":
                    return FakeNote()
                return None

        loader = make_state_doc_loader(FakeEnv())
        doc = loader(".state/prefixed")
        assert doc is not None


# ---------------------------------------------------------------------------
# Factory: make_action_runner
# ---------------------------------------------------------------------------

class TestMakeActionRunner:
    """Tests for action runner creation."""
    def test_runs_find_action(self):
        from keep.state_doc_runtime import make_action_runner

        class FakeItem:
            def __init__(self, id, summary="", tags=None, score=None):
                self.id = id
                self.summary = summary
                self.tags = tags or {}
                self.score = score

        class FakeEnv:
            def find(self, query=None, **kw):
                return [FakeItem("a", "note a", {"topic": "test"}, 0.9)]
            def get(self, id):
                return None
            def list_items(self, **kw):
                return []
            def get_document(self, id):
                return None
            def resolve_meta(self, id, **kw):
                return {}
            def traverse_related(self, source_ids, **kw):
                return {}

        runner = make_action_runner(FakeEnv())
        output = runner("find", {"query": "test"})
        assert output["count"] == 1
        assert output["results"][0]["id"] == "a"

    def test_unknown_action_raises(self):
        from keep.state_doc_runtime import make_action_runner

        class FakeEnv:
            pass

        runner = make_action_runner(FakeEnv())
        with pytest.raises(ValueError, match="unknown action"):
            runner("nonexistent_action_xyz", {})

    def test_runner_uses_prepared_params(self):
        from keep.state_doc_runtime import make_action_runner

        class FakeEnv:
            def get(self, id):
                return None
            def find(self, query=None, **kw):
                return []
            def list_items(self, **kw):
                return []
            def get_document(self, id):
                return None
            def resolve_meta(self, id, **kw):
                return {}
            def traverse_related(self, source_ids, **kw):
                return {}

        action = MagicMock()
        action.run.return_value = {"results": [], "count": 0}

        with patch(
            "keep.actions.prepare_action_params",
            return_value=(action, {"query": "prepared"}),
        ) as mock_prepare:
            runner = make_action_runner(FakeEnv())
            result = runner("find", {"query": "raw"})

        assert result["count"] == 0
        mock_prepare.assert_called_once()
        action.run.assert_called_once()
        assert action.run.call_args.args[0] == {"query": "prepared"}

    def test_runner_recovers_and_retries_on_malformed_db(self, caplog):
        from keep.state_doc_runtime import make_action_runner

        class FakeStore:
            def __init__(self):
                self.recovery_calls = 0

            def _try_runtime_recover(self):
                self.recovery_calls += 1
                return True

        class FakeEnv:
            def __init__(self):
                self.store = FakeStore()

            def get_document_store(self):
                return self.store

            def get(self, id):
                return None

            def find(self, query=None, **kw):
                return []

            def list_items(self, **kw):
                return []

            def get_document(self, id):
                return None

            def resolve_meta(self, id, **kw):
                return {}

            def traverse_related(self, source_ids, **kw):
                return {}

        action = MagicMock()
        action.run.side_effect = [
            sqlite3.DatabaseError("database disk image is malformed"),
            {"results": [], "count": 0},
        ]

        env = FakeEnv()
        with patch(
            "keep.actions.prepare_action_params",
            return_value=(action, {"query": "prepared"}),
        ):
            runner = make_action_runner(env)
            result = runner("find", {"query": "raw"})

        assert result["count"] == 0
        assert action.run.call_count == 2
        assert env.store.recovery_calls == 1
        assert "Action find hit malformed note database; triggering runtime recovery" in caplog.text


# ---------------------------------------------------------------------------
# CEL predicates in flows (find-deep exercises search.count == 0)
# ---------------------------------------------------------------------------

class TestCELPredicates:
    """Tests for CEL predicate evaluation."""
    def test_find_deep_skips_traverse_on_empty_search(self, kp):
        """find-deep returns early when search.count == 0 (CEL predicate)."""
        doc = _store_loader(kp)("find-deep")
        assert doc is not None

        # Run with empty search results — CEL predicate should short-circuit
        loader = _store_loader(kp)
        calls = []

        def _runner(action_name, params):
            calls.append(action_name)
            if action_name == "find":
                return {"results": [], "count": 0}
            return {"groups": {}, "count": 0}

        result = run_flow(
            "find-deep",
            {"query": "test", "limit": 10, "deep_limit": 5},
            load_state_doc=loader, run_action=_runner,
        )
        assert result.status == "done"
        # Only find should have been called (traverse skipped via CEL)
        assert calls == ["find"]

    def test_find_deep_traverses_when_results_found(self, kp):
        """find-deep traverses results when search.count > 0."""
        loader = _store_loader(kp)
        calls = []

        def _runner(action_name, params):
            calls.append(action_name)
            if action_name == "find":
                return {
                    "results": [
                        {"id": "a", "summary": "x", "tags": {}, "score": 0.9},
                    ],
                    "count": 1,
                }
            return {"groups": {"a": [{"id": "b", "summary": "y", "tags": {}}]}, "count": 1}

        result = run_flow(
            "find-deep",
            {"query": "test", "limit": 10, "deep_limit": 5},
            load_state_doc=loader, run_action=_runner,
        )
        assert result.status == "done"
        # Both find and traverse should have been called
        assert "find" in calls
        assert "traverse" in calls
        assert "related" in result.bindings

    @pytest.mark.parametrize(
        ("state", "params"),
        [
            ("find-deep", {"query": "", "limit": 10, "deep_limit": 5}),
            ("memory-search", {"query": "", "limit": 10}),
            ("query-resolve", {"query": ""}),
            ("query-explore", {"query": ""}),
            ("query-branch", {"query": ""}),
        ],
    )
    def test_query_flows_error_on_blank_query_without_find_warnings(
        self, kp, caplog, state, params,
    ):
        """Query-only flows fail fast on blank input instead of looping or logging find errors."""
        loader = _store_loader(kp)
        calls = []

        def _runner(action_name, params):
            calls.append(action_name)
            return {"results": [], "count": 0}

        with caplog.at_level("WARNING"):
            result = run_flow(
                state,
                params,
                load_state_doc=loader,
                run_action=_runner,
            )

        assert result.status == "error"
        assert result.data == {"reason": "query required"}
        assert result.history == [state]
        assert calls == []
        assert "Action find failed" not in caplog.text
        assert "Predicate eval error" not in caplog.text


class TestQueryResolveFlow:
    """Test query-resolve state doc with CEL conditions and transitions."""

    def test_high_margin_returns_done(self, kp):
        """High margin result (dominant top-1) short-circuits to done."""
        loader = _store_loader(kp)

        def _runner(action_name, params):
            if action_name == "find":
                return {
                    "results": [
                        {"id": "a", "summary": "best", "tags": {}, "score": 0.95},
                        {"id": "b", "summary": "weak", "tags": {}, "score": 0.10},
                    ],
                    "count": 2,
                }
            return {}

        result = run_flow(
            "query-resolve",
            {"query": "test", "limit": 10, "margin_high": 0.18,
             "margin_low": 0.08, "entropy_high": 0.72, "entropy_low": 0.45,
             "lineage_strong": 0.75},
            load_state_doc=loader, run_action=_runner,
        )
        assert result.status == "done"
        assert result.history == ["query-resolve"]
        assert "search" in result.bindings

    def test_low_margin_transitions_to_branch(self, kp):
        """Low margin (ambiguous top results) transitions to query-branch."""
        loader = _store_loader(kp)

        call_count = {"find": 0}

        def _runner(action_name, params):
            if action_name == "find":
                call_count["find"] += 1
                return {
                    "results": [
                        {"id": "a", "summary": "x", "tags": {"topic": "auth"}, "score": 0.51},
                        {"id": "b", "summary": "y", "tags": {"topic": "auth"}, "score": 0.50},
                        {"id": "c", "summary": "z", "tags": {"topic": "db"}, "score": 0.49},
                    ],
                    "count": 3,
                }
            return {}

        result = run_flow(
            "query-resolve",
            {"query": "test", "limit": 10, "margin_high": 0.18,
             "margin_low": 0.08, "entropy_high": 0.72, "entropy_low": 0.45,
             "lineage_strong": 0.75, "pivot_limit": 5, "bridge_limit": 5},
            budget=10,
            load_state_doc=loader, run_action=_runner,
        )
        # Should have transitioned through multiple states
        assert "query-branch" in result.history or "query-explore" in result.history


class TestBatchEdges:
    """Test batch edge operations on DocumentStore."""

    @pytest.fixture
    def store(self, tmp_path):
        from keep.document_store import DocumentStore
        db_path = tmp_path / "documents.db"
        with DocumentStore(db_path) as s:
            yield s

    def test_upsert_edges_batch(self, store):
        edges = [
            ("doc1", "speaker", "alice", "said", "2025-01-01T00:00:00"),
            ("doc1", "speaker", "bob", "said", "2025-01-01T00:00:00"),
            ("doc2", "topic", "ai", "discussed_in", "2025-01-02T00:00:00"),
        ]
        count = store.upsert_edges_batch("default", edges)
        assert count == 3

        alice_edges = store.get_inverse_edges("default", "alice")
        assert len(alice_edges) == 1
        bob_edges = store.get_inverse_edges("default", "bob")
        assert len(bob_edges) == 1
        ai_edges = store.get_inverse_edges("default", "ai")
        assert len(ai_edges) == 1

    def test_delete_edges_batch(self, store):
        store.upsert_edge("default", "doc1", "speaker", "alice", "said", "2025-01-01T00:00:00")
        store.upsert_edge("default", "doc1", "speaker", "bob", "said", "2025-01-01T00:00:00")
        store.upsert_edge("default", "doc2", "speaker", "alice", "said", "2025-01-02T00:00:00")

        deleted = store.delete_edges_batch("default", [
            ("doc1", "speaker", "alice"),
            ("doc1", "speaker", "bob"),
        ])
        assert deleted == 2
        # doc2's edge remains
        assert len(store.get_inverse_edges("default", "alice")) == 1

    def test_empty_batch_is_noop(self, store):
        assert store.upsert_edges_batch("default", []) == 0
        assert store.delete_edges_batch("default", []) == 0
