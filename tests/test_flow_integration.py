"""End-to-end behavioral tests for the flow system.

These tests verify that the state-doc flow paths are actually invoked
through the public API, not just that individual components work in
isolation.
"""

import json
import os

import pytest
from unittest.mock import patch

from keep.api import Keeper, FindResults
from keep.const import STATE_ASSESS, STATE_DELETE, STATE_MOVE, STATE_PUT, STATE_STUB, STATE_TAG
from keep.state_doc import parse_state_doc
from keep.state_doc_runtime import run_flow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kp(mock_providers, tmp_path):
    """Keeper instance with embedding provider initialized."""
    kp = Keeper(store_path=tmp_path)
    kp._get_embedding_provider()
    return kp


# ---------------------------------------------------------------------------
# Write path: put() routes through state-doc flow
# ---------------------------------------------------------------------------

class TestWritePathFlow:
    """Verify put() dispatches background tasks via the after-write state doc.

    Task decisions are driven by evaluating the after-write state doc rules,
    NOT hardcoded in _put_direct.  See _dispatch_after_write_flow().
    """

    def test_put_enqueues_after_write_flow(self, kp):
        """put() enqueues a single after-write flow work item."""
        # Drain any migration-enqueued tasks first
        queue = kp._get_work_queue()
        queue.claim("drain", limit=200)
        kp.put("Short note", id="s1")
        claimed = queue.claim("test", limit=20)
        kinds = {t.kind for t in claimed}
        assert kinds == {"flow"}, f"Expected single flow item, got {kinds}"
        # Verify the flow is for after-write with the right item
        flow_item = claimed[0]
        assert flow_item.input.get("state") == "after-write"
        assert flow_item.input.get("item_id") == "s1"

    def test_put_system_note_skips_background(self, kp):
        """System notes: no background work enqueued at all."""
        # Drain any migration-enqueued tasks first
        queue = kp._get_work_queue()
        queue.claim("drain", limit=200)
        kp.put("System data", id=".sys/test")
        claimed = queue.claim("test", limit=20)
        assert len(claimed) == 0, f"Expected no work items for system note, got {len(claimed)}"

    def test_markdown_authored_body_stores_full_content_and_persists(self, kp):
        """Markdown-authored notes keep exact body text across later writes."""
        first = "a" * min(kp._config.max_inline_length - 10, kp._config.max_summary_length + 50)
        second = "b" * min(kp._config.max_inline_length - 5, kp._config.max_summary_length + 75)

        item = kp._put_direct(
            content=first,
            id="sync/md-note",
            queue_background_tasks=False,
            _body_authority="markdown",
        )
        assert item.summary == first
        assert item.tags["_body_authority"] == "markdown"

        updated = kp._put_direct(
            content=second,
            id="sync/md-note",
            queue_background_tasks=False,
        )
        assert updated.summary == second
        assert updated.tags["_body_authority"] == "markdown"

    def test_assess_builtin_returns_noop_payload(self, kp):
        """With VT bundled but no API key, assess returns 'disabled' (no-op)."""
        result = kp.run_flow_command(STATE_ASSESS, params={"target_id": "x"}, budget=2)

        assert result.status == "done"
        # VT fragment fires but has no API key → assessment: disabled.
        # All caller params pass through unchanged.
        assert result.data["assessment"] == "disabled"
        assert result.data["stop_processing"] is False
        assert result.data["skip_fetch"] is False

    def test_put_flow_uses_assessed_directives(self, kp):
        """A custom assess fragment can rewrite the final put directives."""
        kp._put_direct(
            """\
order: before:default
rules:
  - return:
      status: done
      with:
        assessment: "malicious"
        id: "blocked-note"
        uri: null
        content: "Blocked by assessment"
        tags:
          assessment_test: "blocked"
        summary: "Blocked summary"
        created_at: null
        force: false
        queue_background_tasks: false
        stop_processing: true
        skip_fetch: true
""",
            id=".state/assess/aaa-test",
            queue_background_tasks=False,
        )
        queue = kp._get_work_queue()
        queue.claim("drain", limit=200)

        result = kp.run_flow_command(
            STATE_PUT,
            params={"uri": "https://example.com"},
            budget=4,
        )

        assert result.status == "done"
        stored = kp.get("blocked-note")
        assert stored is not None
        assert stored.summary == "Blocked summary"
        assert stored.tags.get("assessment_test") == "blocked"
        assert queue.claim("test", limit=20) == []

    def test_stub_flow_uses_assessed_directives(self, kp):
        """A custom assess fragment can rewrite the final stub directives."""
        kp._put_direct(
            """\
order: before:default
rules:
  - return:
      status: done
      with:
        assessment: "suspicious"
        id: "{params.id}"
        uri: null
        content: "Assessed stub"
        tags:
          assessment_test: "stubbed"
        summary: "Stub summary"
        created_at: null
        force: false
        queue_background_tasks: false
        stop_processing: false
        skip_fetch: false
""",
            id=".state/assess/aaa-test",
            queue_background_tasks=False,
        )
        queue = kp._get_work_queue()
        queue.claim("drain", limit=200)

        result = kp.run_flow_command(
            STATE_STUB,
            params={"id": "assessed-stub", "content": "Ignored placeholder"},
            budget=4,
        )

        assert result.status == "done"
        stored = kp.get("assessed-stub")
        assert stored is not None
        assert stored.summary == "Stub summary"
        assert stored.tags.get("assessment_test") == "stubbed"
        assert queue.claim("test", limit=20) == []

    def test_stub_flow_inserts_if_absent_with_default_source(self, kp):
        """The builtin stub flow inserts a stub note without clobbering later content."""
        result = kp.run_flow_command(
            STATE_STUB,
            params={"id": "stub-flow", "content": "Placeholder"},
            budget=4,
        )

        assert result.status == "done"
        stored = kp.get("stub-flow")
        assert stored is not None
        assert stored.summary == "Placeholder"
        assert stored.tags.get("_source") == "auto-vivify"
        assert result.data["stored"]["changed"] is True

        kp._put_direct("Real content", id="stub-flow", queue_background_tasks=False)
        second = kp.run_flow_command(
            STATE_STUB,
            params={"id": "stub-flow", "content": "Ignored placeholder"},
            budget=4,
        )

        assert second.status == "done"
        assert second.data["stored"]["changed"] is False
        preserved = kp.get("stub-flow")
        assert preserved is not None
        assert preserved.summary == "Real content"

    def test_stub_flow_preserves_explicit_source_and_queue_flag(self, kp):
        """Stub flow preserves caller provenance and can suppress follow-up queue work."""
        queue = kp._get_work_queue()
        queue.claim("drain", limit=200)

        result = kp.run_flow_command(
            STATE_STUB,
            params={
                "id": "stub-link",
                "content": "https://example.com",
                "tags": {"_source": "link"},
                "queue_background_tasks": False,
            },
            budget=4,
        )

        assert result.status == "done"
        stored = kp.get("stub-link")
        assert stored is not None
        assert stored.tags.get("_source") == "link"
        assert queue.claim("test", limit=20) == []

    def test_stub_via_flow_uses_flow_when_state_doc_is_present(self, kp):
        """Bootstrap fallback should only apply while .state/stub is absent."""
        flow_item = object()
        original_get = kp._document_store.get

        def fake_get(collection, item_id):
            if item_id == ".state/stub":
                return {"id": ".state/stub"}
            return original_get(collection, item_id)

        with (
            patch.object(kp, "_stub_direct") as stub_direct,
            patch("keep.api.flow_stub_item", return_value=flow_item) as flow_stub,
            patch.object(kp._document_store, "get", side_effect=fake_get),
        ):
            kp._needs_sysdoc_migration = True
            result = kp._stub_via_flow(
                id="flow-eligible-stub",
                content="Placeholder",
                tags={"_source": "link"},
                queue_background_tasks=False,
            )

        assert result is flow_item
        stub_direct.assert_not_called()
        flow_stub.assert_called_once_with(
            kp,
            id="flow-eligible-stub",
            content="Placeholder",
            summary=None,
            tags={"_source": "link"},
            created_at=None,
            queue_background_tasks=False,
        )


# ---------------------------------------------------------------------------
# Read path: get_context() uses state-doc flow
# ---------------------------------------------------------------------------

class TestGetContextFlow:
    """Tests for get-context flow invocation."""

    def test_get_context_invokes_read_flow(self, kp):
        """get_context() calls _run_read_flow('get-context')."""
        kp.put("Test note about architecture", id="arch1")
        kp.put("Related note about design", id="design1")

        calls = []
        original = kp._run_read_flow

        def tracking_flow(state, params, **kwargs):
            calls.append(state)
            return original(state, params, **kwargs)

        kp._run_read_flow = tracking_flow
        kp.get_context("arch1")

        assert "get" in calls, \
            "get_context() should invoke the get-context state-doc flow"

    def test_get_context_system_doc_skips_read_flow(self, kp):
        """System docs return the document only, without read-flow context."""
        calls = []
        original = kp._run_read_flow

        def tracking_flow(state, params, **kwargs):
            calls.append(state)
            return original(state, params, **kwargs)

        kp._run_read_flow = tracking_flow
        ctx = kp.get_context(".meta/todo")

        assert ctx is not None
        assert ctx.item.id == ".meta/todo"
        assert ctx.similar == []
        assert ctx.meta == {}
        assert ctx.edges == {}
        assert ctx.parts == []
        assert ctx.prev == []
        assert ctx.next == []
        assert "get" not in calls, \
            "system docs should not invoke the get-context state-doc flow"

    def test_get_context_returns_similar_items(self, kp):
        """get_context assembles similar items via the flow."""
        kp.put("Python async patterns", id="py1")
        kp.put("Python concurrency guide", id="py2")
        kp.put("Unrelated cooking recipe", id="cook1")

        ctx = kp.get_context("py1")
        assert ctx is not None
        # Should have similar items populated
        similar = getattr(ctx, "similar", None)
        if similar is not None:
            similar_ids = {s.id for s in similar}
            assert "py2" in similar_ids or len(similar) > 0

    def test_get_context_skips_similar_reembed_when_anchor_not_indexed(self, kp):
        """Display context should not load an embedding model for fresh anchors."""
        kp.put("Anchor note about auth flows", id="anchor-display")
        kp.put("Related note about auth flows", id="related-display")

        chroma_coll = kp._resolve_chroma_collection()
        kp._store.delete(chroma_coll, "anchor-display")

        def _unexpected_embed(*args, **kwargs):
            raise AssertionError("get_context() should not re-embed for display similarity")

        kp._embedding_provider.embed = _unexpected_embed

        ctx = kp.get_context(
            "anchor-display",
            meta_limit=0,
            parts_limit=0,
            edges_limit=0,
            versions_limit=0,
        )

        assert ctx is not None
        assert ctx.item.id == "anchor-display"
        assert ctx.similar == []

    def test_similar_limit_exact_conformance(self, kp):
        """similar_limit=2 returns exactly 2 when 8 similar items exist."""
        for i in range(8):
            kp.put(f"Python concurrency pattern number {i}", id=f"pycon-{i}")

        # Unconstrained: should find several
        ctx_all = kp.get_context("pycon-0", similar_limit=8)
        assert ctx_all is not None
        all_count = len(ctx_all.similar or [])
        assert all_count >= 4, f"Need at least 4 similar items to test, got {all_count}"

        # Constrained: exactly 2
        ctx = kp.get_context("pycon-0", similar_limit=2)
        assert ctx is not None
        assert len(ctx.similar) == 2, f"Expected exactly 2 similar, got {len(ctx.similar)}"

    def test_meta_limit_exact_conformance(self, kp):
        """meta_limit=1 returns exactly 1 item per section even when multiple rules match.

        .meta/todo has 4 find rules (commitments, requests, offers, blocked).
        With enough data each rule finds items. meta_limit must cap the
        *total* per section, not per rule.
        """
        anchor = kp.put("Working on auth flow", tags={"project": "test"})
        # Create items matching different todo rules
        for i in range(4):
            kp.put(f"I will fix bug {i}", tags={
                "act": "commitment", "status": "open",
            })
        for i in range(3):
            kp.put(f"Please review PR {i}", tags={
                "act": "request", "status": "open",
            })

        # Verify unconstrained has multiple (rules produce independent results)
        ctx_all = kp.get_context(anchor.id, meta_limit=10)
        assert ctx_all is not None
        todo_all = ctx_all.meta.get("todo", []) if ctx_all.meta else []
        assert len(todo_all) >= 3, f"Need at least 3 todo items to test, got {len(todo_all)}"

        # Constrained: exactly 1 per section (not per rule!)
        ctx = kp.get_context(anchor.id, meta_limit=1)
        for section_name, items in (ctx.meta or {}).items():
            assert len(items) <= 1, \
                f"meta/{section_name} has {len(items)} items, expected <= 1"

    def test_edges_limit_exact_conformance(self, kp):
        """edges_limit=2 returns exactly 2 refs when 6 inbound edges exist."""
        kp.put("Alice", id="alice-edges")
        for i in range(6):
            kp.put(f"Message from Alice #{i}", id=f"msg-a-{i}",
                   tags={"from": "alice-edges"})

        # Verify unconstrained has all 6
        ctx_all = kp.get_context("alice-edges", edges_limit=10)
        assert ctx_all is not None
        all_refs = []
        for refs in (ctx_all.edges or {}).values():
            all_refs.extend(refs)
        assert len(all_refs) >= 5, f"Need at least 5 edge refs to test, got {len(all_refs)}"

        # Constrained: exactly 2 per predicate
        ctx = kp.get_context("alice-edges", edges_limit=2)
        for pred, refs in (ctx.edges or {}).items():
            assert len(refs) == 2, \
                f"Expected exactly 2 refs for '{pred}', got {len(refs)}"

    def test_parts_limit_exact_conformance(self, kp):
        """parts_limit=2 returns exactly 2 when 5 parts exist."""
        kp.put("Document with many structural parts " * 50, id="parts-lim")
        chroma_coll = kp._resolve_chroma_collection()
        embed = kp._get_embedding_provider()
        doc_coll = kp._resolve_doc_collection()
        for i in range(5):
            from keep.document_store import PartInfo
            from keep.types import utc_now
            part = PartInfo(
                part_num=i + 1,
                summary=f"Part {i + 1} about topic {i + 1}",
                tags={"_base_id": "parts-lim"},
                created_at=utc_now(),
            )
            kp._document_store.upsert_single_part(doc_coll, "parts-lim", part)
            embedding = embed.embed(part.summary)
            kp._store.upsert_part(
                chroma_coll, "parts-lim", i + 1, embedding,
                part.summary, {"_base_id": "parts-lim", "_part_num": str(i + 1)},
            )

        # Verify unconstrained returns all 5
        ctx_all = kp.get_context("parts-lim", parts_limit=10)
        assert ctx_all is not None
        assert len(ctx_all.parts or []) >= 4, \
            f"Need at least 4 parts to test, got {len(ctx_all.parts or [])}"

        # Constrained: exactly 2
        ctx = kp.get_context("parts-lim", parts_limit=2)
        assert len(ctx.parts or []) == 2, \
            f"Expected exactly 2 parts, got {len(ctx.parts or [])}"

    def test_versions_limit_exact_conformance(self, tmp_path):
        """versions_limit=2 returns exactly 2 prev refs when 6 versions exist.

        Uses a real DocumentStore (not mocked) because versioning requires
        actual SQLite version tracking.  Only mocks providers to avoid
        hitting real LLM APIs.
        """
        from unittest.mock import patch as _patch, MagicMock
        from tests.conftest import MockEmbeddingProvider, MockSummarizationProvider, MockDocumentProvider

        mock_reg = MagicMock()
        mock_reg.create_embedding.return_value = MockEmbeddingProvider()
        mock_reg.create_summarization.return_value = MockSummarizationProvider()
        mock_reg.create_document.return_value = MockDocumentProvider()

        with _patch("keep.api.get_registry", return_value=mock_reg), \
             _patch("keep._provider_lifecycle.get_registry", return_value=mock_reg), \
             _patch("keep.api.CachingEmbeddingProvider", side_effect=lambda p, **kw: p), \
             _patch("keep._provider_lifecycle.CachingEmbeddingProvider", side_effect=lambda p, **kw: p), \
             _patch.object(Keeper, "_spawn_processor", return_value=False):
            kp = Keeper(store_path=tmp_path / "ver-test-store")
            kp._get_embedding_provider()

            for i in range(6):
                kp.put(f"Version {i} of the evolving document", id="ver-lim")

            # Verify unconstrained returns multiple prev refs
            ctx_all = kp.get_context("ver-lim", versions_limit=5)
            assert ctx_all is not None
            prev_all = ctx_all.prev or []
            assert len(prev_all) >= 3, \
                f"Need at least 3 prev versions to test, got {len(prev_all)}"

            # Constrained: exactly 2
            ctx = kp.get_context("ver-lim", versions_limit=2)
            prev = ctx.prev or []
            assert len(prev) == 2, f"Expected exactly 2 prev versions, got {len(prev)}"

            kp.close()

    def test_zero_limit_suppresses_all(self, kp):
        """Limit of 0 produces no results in each section."""
        # Create enough data to have results without limits
        anchor = kp.put("Zero limit test anchor", id="zero-lim")
        for i in range(5):
            kp.put(f"Related to zero limit {i}", id=f"zero-r-{i}")
        for i in range(3):
            kp.put(f"Commitment {i}", tags={
                "act": "commitment", "status": "open",
            })

        ctx = kp.get_context(
            "zero-lim",
            similar_limit=0, meta_limit=0, edges_limit=0,
        )
        assert ctx is not None
        assert len(ctx.similar or []) == 0, f"similar should be empty, got {len(ctx.similar or [])}"
        assert not ctx.meta or all(len(v) == 0 for v in ctx.meta.values()), \
            f"meta should be empty, got {ctx.meta}"


# ---------------------------------------------------------------------------
# Read path: find(deep=True) uses state-doc flow
# ---------------------------------------------------------------------------

class TestFindDeepFlow:
    """Tests for find-deep flow invocation."""
    def test_find_deep_invokes_flow_when_no_edges(self, kp):
        """find(deep=True) dispatches through the flow-host compat path."""
        kp.put("OAuth2 token design for project X", id="a",
               tags={"project": "x", "topic": "auth"})
        for i in range(35):
            kp.put(f"Filler note {i}", id=f"f-{i}", tags={"filler": "yes"})
        kp.put("Project X performance report", id="b",
               tags={"project": "x"})

        calls = []
        original = kp.run_flow

        def tracking_flow(state, **kwargs):
            calls.append(state)
            return original(state, **kwargs)

        kp.run_flow = tracking_flow  # type: ignore[method-assign]
        results = kp.find("OAuth2 auth token design", deep=True, limit=5)

        assert "compat-find" in calls, \
            "find(deep=True) should invoke the flow-backed compat path"

    def test_find_deep_flow_discovers_bridge_items(self, kp):
        """The flow path discovers bridge items via tag-follow."""
        kp.put("OAuth2 design for project X", id="a",
               tags={"project": "x", "topic": "auth"})
        for i in range(35):
            kp.put(f"Filler note {i}", id=f"f-{i}", tags={"filler": "yes"})
        kp.put("Project X latency report", id="b",
               tags={"project": "x"})

        results = kp.find("OAuth2 auth design", deep=True, limit=5)
        all_ids = {r.id for r in results}
        deep_ids = set()
        for group in results.deep_groups.values():
            deep_ids.update(item.id for item in group)
        all_found = all_ids | deep_ids

        assert "b" in all_found, \
            "Bridge item 'b' should be found via project=x tag-follow"

    def test_find_deep_similar_to_uses_anchor_summary(self, kp):
        """find(similar_to=..., deep=True) falls back to anchor summary as query."""
        kp.put("OAuth2 token design", id="a", tags={"project": "x"})
        kp.put("Project X perf", id="b", tags={"project": "x"})

        # Should not raise — similar_to mode uses anchor summary
        results = kp.find(similar_to="a", deep=True, limit=5)
        assert isinstance(results, FindResults)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Budget enforcement in read flows
# ---------------------------------------------------------------------------

class TestFlowBudget:
    """Tests for flow budget exhaustion."""
    def test_budget_exhaustion_returns_stopped(self):
        """Flow that transitions indefinitely stops at budget limit."""
        loader_docs = {
            "loop": """\
match: sequence
rules:
  - id: search
    do: find
    with:
      query: "test"
      limit: 5
  - then: loop
""",
        }
        compiled = {}
        for name, body in loader_docs.items():
            compiled[name] = parse_state_doc(name, body)

        def loader(name):
            return compiled.get(name)

        calls = []
        def runner(action_name, params):
            calls.append(action_name)
            return {"results": [], "count": 0}

        result = run_flow("loop", {}, budget=3,
                         load_state_doc=loader, run_action=runner)
        assert result.status == "stopped"
        assert result.ticks == 3
        assert result.data["reason"] == "budget"
        assert len(result.history) == 3

    def test_budget_one_allows_single_tick(self):
        """Budget=1 allows exactly one state-doc evaluation."""
        compiled = {
            "one": parse_state_doc("one", """\
match: sequence
rules:
  - id: s
    do: find
    with:
      query: "test"
      limit: 1
  - return: done
"""),
        }

        result = run_flow(
            "one", {}, budget=1,
            load_state_doc=lambda n: compiled.get(n),
            run_action=lambda n, p: {"results": [], "count": 0},
        )
        assert result.status == "done"
        assert result.ticks == 1


# ---------------------------------------------------------------------------
# Traversal: Tier 1 (edges) → Tier 2 (tag-follow) fallback
# ---------------------------------------------------------------------------

class TestTraversalFallback:
    """Tests for traversal fallback to tag follow."""
    def test_no_edges_falls_to_tag_follow(self, kp):
        """When store has no edges, traverse_related uses tag-follow."""
        kp.put("Topic A note 1", id="a1", tags={"topic": "alpha"})
        kp.put("Topic A note 2", id="a2", tags={"topic": "alpha"})
        kp.put("Topic B note", id="b1", tags={"topic": "beta"})

        from keep.flow_env import LocalFlowEnvironment
        env = LocalFlowEnvironment(kp)

        groups = env.traverse_related(["a1"], limit_per_source=5)
        assert "a1" in groups
        # a2 shares topic=alpha with a1, should be discovered via tag-follow
        group_ids = {item.id for item in groups.get("a1", [])}
        assert "a2" in group_ids, \
            "a2 should be found via tag-follow on topic=alpha"

    def test_traverse_does_not_touch_accessed_at(self, kp):
        """Internal traversal should not update accessed_at timestamps."""
        kp.put("Note for traversal", id="t1", tags={"topic": "traverse"})

        doc_coll = kp._resolve_doc_collection()
        before = kp._document_store.get(doc_coll, "t1")
        before_accessed = before.accessed_at

        from keep.flow_env import LocalFlowEnvironment
        env = LocalFlowEnvironment(kp)
        env.traverse_related(["t1"], limit_per_source=5)

        after = kp._document_store.get(doc_coll, "t1")
        assert after.accessed_at == before_accessed, \
            "traverse_related should not update accessed_at"


# ---------------------------------------------------------------------------
# Auto-vivify atomicity
# ---------------------------------------------------------------------------

class TestAutoVivify:
    """Tests for auto-vivify insert-if-absent."""
    def test_insert_if_absent_does_not_overwrite(self, kp):
        """Auto-vivify uses insert_if_absent — doesn't clobber existing docs."""
        doc_coll = kp._resolve_doc_collection()

        # Pre-create the target with real content
        kp.put("Real document about Alice", id="alice",
               tags={"type": "person"})

        # Now simulate auto-vivify for the same ID
        inserted = kp._document_store.insert_if_absent(
            doc_coll, "alice",
            summary="",
            tags={"_source": "auto-vivify"},
        )

        assert not inserted, "insert_if_absent should return False for existing doc"

        # Verify original content preserved
        doc = kp._document_store.get(doc_coll, "alice")
        assert doc.summary == "Real document about Alice"
        assert doc.tags.get("type") == "person"


# ---------------------------------------------------------------------------
# Cursor encoding/decoding and flow resumption
# ---------------------------------------------------------------------------

class TestFlowCursor:
    """Tests for flow cursor round-trip."""
    def test_cursor_round_trip(self):
        """encode_cursor / decode_cursor are symmetric."""
        from keep.state_doc_runtime import encode_cursor, decode_cursor

        token = encode_cursor("query-explore", 3, {"search": {"count": 5}})
        assert isinstance(token, str)
        assert len(token) > 0

        decoded = decode_cursor(token)
        assert decoded is not None
        assert decoded.state == "query-explore"
        assert decoded.ticks == 3
        assert decoded.bindings == {"search": {"count": 5}}

    def test_decode_invalid_cursor(self):
        """decode_cursor returns None for garbage input."""
        from keep.state_doc_runtime import decode_cursor

        assert decode_cursor("") is None
        assert decode_cursor("not-valid-base64!!!") is None
        assert decode_cursor(None) is None

    def test_stopped_flow_returns_cursor(self):
        """A budget-exhausted flow returns a cursor for resumption."""
        compiled = {
            "loop": parse_state_doc("loop", """\
match: sequence
rules:
  - id: search
    do: find
    with:
      query: "test"
      limit: 5
  - then: loop
"""),
        }

        result = run_flow(
            "loop", {"query": "test"}, budget=2,
            load_state_doc=lambda n: compiled.get(n),
            run_action=lambda n, p: {"results": [], "count": 0},
        )
        assert result.status == "stopped"
        assert result.cursor is not None
        assert result.ticks == 2

    def test_resume_from_cursor(self):
        """Resuming a stopped flow continues from the checkpoint."""
        from keep.state_doc_runtime import decode_cursor

        compiled = {
            "counter": parse_state_doc("counter", """\
match: sequence
rules:
  - id: s
    do: find
    with:
      query: "test"
  - then: counter
"""),
        }

        tick_counts = []
        def runner(name, params):
            tick_counts.append(1)
            return {"results": [], "count": 0}

        # First run: budget=2
        r1 = run_flow(
            "counter", {}, budget=2,
            load_state_doc=lambda n: compiled.get(n),
            run_action=runner,
        )
        assert r1.status == "stopped"
        assert r1.ticks == 2
        assert r1.cursor is not None

        # Resume with budget=3
        cursor = decode_cursor(r1.cursor)
        r2 = run_flow(
            "counter", {}, budget=3,
            load_state_doc=lambda n: compiled.get(n),
            run_action=runner,
            cursor=cursor,
        )
        assert r2.status == "stopped"
        assert r2.ticks == 5  # 2 prior + 3 new
        assert len(tick_counts) == 5  # total action calls

    def test_done_flow_no_cursor(self):
        """A completed flow does not include a cursor."""
        compiled = {
            "simple": parse_state_doc("simple", """\
match: sequence
rules:
  - id: s
    do: find
    with:
      query: "test"
  - return: done
"""),
        }

        result = run_flow(
            "simple", {}, budget=5,
            load_state_doc=lambda n: compiled.get(n),
            run_action=lambda n, p: {"results": [], "count": 0},
        )
        assert result.status == "done"
        assert result.cursor is None

    def test_bindings_accumulate_across_resume(self):
        """Bindings from previous ticks are preserved on resume."""
        from keep.state_doc_runtime import decode_cursor, FlowCursor

        compiled = {
            "accum": parse_state_doc("accum", """\
match: sequence
rules:
  - id: search
    do: find
    with:
      query: "test"
  - then: accum
"""),
        }

        call_count = [0]
        def runner(name, params):
            call_count[0] += 1
            return {"results": [f"r{call_count[0]}"], "count": 1}

        r1 = run_flow(
            "accum", {}, budget=1,
            load_state_doc=lambda n: compiled.get(n),
            run_action=runner,
        )
        assert r1.status == "stopped"
        assert "search" in r1.bindings

        cursor = decode_cursor(r1.cursor)
        assert cursor.bindings.get("search") is not None

        r2 = run_flow(
            "accum", {}, budget=1,
            load_state_doc=lambda n: compiled.get(n),
            run_action=runner,
            cursor=cursor,
        )
        # Bindings should have the latest search result
        assert "search" in r2.bindings


# ---------------------------------------------------------------------------
# CLI: keep flow
# ---------------------------------------------------------------------------

class TestFlowCLI:
    """Tests for flow CLI commands."""
    @pytest.fixture
    def cli(self, mock_providers, tmp_path):
        """CLI runner with in-process daemon (no subprocess needed)."""
        from keep.api import Keeper
        from keep.daemon_server import DaemonServer
        from keep.cli_app import app
        from typer.testing import CliRunner

        kp = Keeper(store_path=tmp_path)
        server = DaemonServer(kp, port=0)
        port = server.start()

        # Write daemon files so the command app connects to our in-process server
        (tmp_path / ".daemon.port").write_text(str(port))
        (tmp_path / ".daemon.token").write_text(server.auth_token)

        runner = CliRunner()
        def invoke(*args):
            env = {
                "KEEP_STORE_PATH": str(tmp_path),
                "KEEP_CONFIG": str(tmp_path),
            }
            return runner.invoke(app, list(args), env=env, catch_exceptions=False)
        yield invoke
        server.stop()
        kp.close()

    def test_flow_help(self, cli):
        result = cli("flow", "--help")
        assert result.exit_code == 0
        assert "state-doc flow" in result.stdout.lower() or "state doc" in result.stdout.lower()

    def test_flow_requires_argument(self, cli):
        result = cli("flow")
        assert result.exit_code != 0

    def test_flow_inline_yaml(self, cli, tmp_path):
        """Run a flow from an inline YAML file."""
        state_doc = tmp_path / "test.yaml"
        state_doc.write_text("""\
match: sequence
rules:
  - id: s
    do: find
    with:
      query: "hello"
  - return: done
""")
        result = cli("flow", "--file", str(state_doc))
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["status"] == "done"

    def test_flow_with_budget(self, cli, tmp_path):
        """--budget flag limits ticks."""
        state_doc = tmp_path / "loop.yaml"
        state_doc.write_text("""\
match: sequence
rules:
  - id: s
    do: find
    with:
      query: "test"
  - then: inline
""")
        result = cli("flow", "--file", str(state_doc), "--budget", "2")
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["status"] == "stopped"
        assert output["ticks"] == 2
        assert "cursor" in output

    def test_flow_with_token_budget_renders_text(self, cli):
        """--token-budget controls rendering, independently of tick budget."""
        result = cli(
            "flow", "get", "-p", "item_id=.library/mn61",
            "--token-budget", "200",
        )

        assert result.exit_code == 0
        assert "id: .library/mn61" in result.stdout
        assert not result.stdout.lstrip().startswith("{")

    def test_flow_tokens_alias_with_json_preserves_envelope(self, cli):
        """--json remains machine-readable when rendered output is requested."""
        result = cli(
            "flow", "get", "-p", "item_id=.library/mn61",
            "--tokens", "200", "--json",
        )

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["status"] == "done"
        assert "rendered" in output


# ---------------------------------------------------------------------------
# API: run_flow_command
# ---------------------------------------------------------------------------

class TestRunFlowCommand:
    """Tests for run-flow command."""
    @pytest.fixture
    def kp(self, mock_providers, tmp_path):
        kp = Keeper(store_path=tmp_path)
        kp._get_embedding_provider()
        return kp

    def test_run_stored_state_doc(self, kp):
        """run_flow_command with a built-in state doc name."""
        result = kp.run_flow_command(
            "get",
            params={"id": "nonexistent"},
            budget=1,
        )
        # get-context on nonexistent ID still completes
        assert result.status in ("done", "error")

    def test_run_inline_yaml(self, kp):
        """run_flow_command with inline YAML state doc."""
        yaml_doc = """\
match: sequence
rules:
  - id: s
    do: find
    with:
      query: "hello"
  - return: done
"""
        result = kp.run_flow_command(
            "test",
            params={},
            state_doc_yaml=yaml_doc,
        )
        assert result.status == "done"
        assert result.ticks == 1

    def test_run_with_cursor_resume(self, kp):
        """run_flow_command can resume via cursor."""
        yaml_loop = """\
match: sequence
rules:
  - id: s
    do: find
    with:
      query: "test"
  - then: test
"""
        r1 = kp.run_flow_command("test", params={}, budget=1, state_doc_yaml=yaml_loop)
        assert r1.status == "stopped"
        assert r1.cursor is not None

        r2 = kp.run_flow_command("test", params={}, budget=2,
                                  cursor_token=r1.cursor, state_doc_yaml=yaml_loop)
        assert r2.status == "stopped"
        assert r2.ticks == 3  # 1 prior + 2 new


# ---------------------------------------------------------------------------
# Validation: missing params surface actionable errors
# ---------------------------------------------------------------------------

class TestFlowValidation:
    """Verify that missing required params produce error bindings, not silent nulls."""

    def test_put_missing_content(self, kp):
        # put now calls assess first, so the validation path needs 3 ticks.
        r = kp.run_flow_command(STATE_PUT, params={}, budget=3)
        assert r.data.get("stored", {}).get("error")
        assert "content" in r.data["stored"]["error"]

    def test_tag_missing_tags(self, kp):
        r = kp.run_flow_command(STATE_TAG, params={"id": "x"}, budget=1)
        tagged = r.data.get("tagged", {})
        # Action skips gracefully when no tags provided
        assert tagged.get("skipped") or tagged.get("error")

    def test_delete_missing_id(self, kp):
        r = kp.run_flow_command(STATE_DELETE, params={}, budget=1)
        assert r.data.get("result", {}).get("error")
        assert "id" in r.data["result"]["error"]

    def test_move_missing_name(self, kp):
        r = kp.run_flow_command(STATE_MOVE, params={}, budget=1)
        assert r.data.get("moved", {}).get("error")
        assert "name" in r.data["moved"]["error"]

    def test_nonexistent_state_doc(self, kp):
        r = kp.run_flow_command("nonexistent", params={}, budget=1)
        assert r.status == "error"
        assert "not found" in r.data["reason"]


# ---------------------------------------------------------------------------
# Async action flag and delegation
# ---------------------------------------------------------------------------

class TestAsyncActionFlag:
    """Verify async_action decorator flag and is_async_action() lookup."""

    def test_async_actions_marked(self):
        from keep.actions import is_async_action
        assert is_async_action("summarize") is True
        assert is_async_action("generate") is True
        assert is_async_action("analyze") is True
        assert is_async_action("auto_tag") is True
        assert is_async_action("ocr") is True
        assert is_async_action("describe") is True

    def test_sync_actions_not_marked(self):
        from keep.actions import is_async_action
        assert is_async_action("find") is False
        assert is_async_action("get") is False
        assert is_async_action("put") is False
        assert is_async_action("traverse") is False
        assert is_async_action("resolve_meta") is False
        assert is_async_action("extract_links") is False
        assert is_async_action("tag") is False

    def test_unknown_action_returns_false(self):
        from keep.actions import is_async_action
        assert is_async_action("nonexistent") is False
        assert is_async_action("") is False


class TestAsyncDelegation:
    """Verify that foreground flows delegate async actions via cursor."""

    def test_foreground_flow_stops_at_async_action(self):
        """Foreground flow hitting an async action → status='async' + cursor."""
        compiled = {
            "mixed": parse_state_doc("mixed", """\
match: sequence
rules:
  - id: search
    do: find
    with:
      query: "test"
  - id: summary
    do: summarize
    with:
      item_id: "test-item"
  - return: done
"""),
        }

        calls = []
        def runner(name, params):
            calls.append(name)
            return {"results": [], "count": 0}

        result = run_flow(
            "mixed", {}, budget=5,
            load_state_doc=lambda n: compiled.get(n),
            run_action=runner,
            foreground=True,  # default, but explicit
        )

        assert result.status == "async"
        assert result.cursor is not None
        # find executed, summarize did not
        assert calls == ["find"]
        assert result.data["action"] == "summarize"

    def test_daemon_flow_executes_async_actions_inline(self):
        """Daemon flow (foreground=False) executes async actions inline."""
        compiled = {
            "mixed": parse_state_doc("mixed", """\
match: sequence
rules:
  - id: search
    do: find
    with:
      query: "test"
  - id: summary
    do: summarize
    with:
      item_id: "test-item"
  - return: done
"""),
        }

        calls = []
        def runner(name, params):
            calls.append(name)
            return {"results": [], "count": 0}

        result = run_flow(
            "mixed", {}, budget=5,
            load_state_doc=lambda n: compiled.get(n),
            run_action=runner,
            foreground=False,
        )

        assert result.status == "done"
        assert result.cursor is None
        # Both actions executed
        assert calls == ["find", "summarize"]

    def test_async_cursor_carries_state(self):
        """Cursor from async delegation encodes state for resume."""
        from keep.state_doc_runtime import decode_cursor

        compiled = {
            "two-step": parse_state_doc("two-step", """\
match: sequence
rules:
  - id: prep
    do: find
    with:
      query: "prep"
  - id: gen
    do: generate
    with:
      system: "test"
      user: "test"
  - return: done
"""),
        }

        result = run_flow(
            "two-step", {"context": "test"}, budget=5,
            load_state_doc=lambda n: compiled.get(n),
            run_action=lambda n, p: {"results": [], "count": 0, "text": "ok"},
            foreground=True,
        )

        assert result.status == "async"
        cursor = decode_cursor(result.cursor)
        assert cursor is not None
        assert cursor.state == "two-step"

    def test_async_resume_completes_flow(self):
        """Resume from async cursor in daemon context completes the flow."""
        from keep.state_doc_runtime import decode_cursor

        compiled = {
            "async-resume": parse_state_doc("async-resume", """\
match: sequence
rules:
  - id: search
    do: find
    with:
      query: "test"
  - id: gen
    do: generate
    with:
      system: "test"
      user: "test"
  - return: done
"""),
        }

        calls = []
        def runner(name, params):
            calls.append(name)
            return {"results": [], "count": 0, "text": "generated"}

        # Step 1: foreground stops at generate
        r1 = run_flow(
            "async-resume", {}, budget=5,
            load_state_doc=lambda n: compiled.get(n),
            run_action=runner,
            foreground=True,
        )
        assert r1.status == "async"
        assert calls == ["find"]

        # Step 2: resume in daemon context
        cursor = decode_cursor(r1.cursor)
        calls.clear()
        r2 = run_flow(
            "async-resume", {}, budget=5,
            load_state_doc=lambda n: compiled.get(n),
            run_action=runner,
            cursor=cursor,
            foreground=False,
        )
        assert r2.status == "done"
        # Re-evaluates from top: find runs again, then generate completes
        assert "find" in calls
        assert "generate" in calls

    def test_sync_only_flow_unaffected_by_foreground(self):
        """Flow with only sync actions completes normally in foreground."""
        compiled = {
            "sync-only": parse_state_doc("sync-only", """\
match: sequence
rules:
  - id: s1
    do: find
    with:
      query: "a"
  - id: s2
    do: get
    with:
      id: "test"
  - return: done
"""),
        }

        calls = []
        def runner(name, params):
            calls.append(name)
            return {"results": [], "count": 0, "id": "test", "summary": "x"}

        result = run_flow(
            "sync-only", {}, budget=5,
            load_state_doc=lambda n: compiled.get(n),
            run_action=runner,
            foreground=True,
        )

        assert result.status == "done"
        assert calls == ["find", "get"]


# ---------------------------------------------------------------------------
# Flow work item execution (daemon)
# ---------------------------------------------------------------------------

class TestFlowWorkItemExecution:
    """Verify _execute_flow_item handles fresh and cursor-based flows."""

    def test_fresh_flow_execution(self, kp):
        """_execute_flow_item runs a fresh flow (no cursor)."""
        from keep.work_processor import _execute_flow_item

        kp.put("Test content for flow", id="flow-test-1")

        result = _execute_flow_item(kp, {
            "state": "get",
            "params": {
                "item_id": "flow-test-1",
                "similar_limit": 3,
                "meta_limit": 3,
                "parts_limit": 5,
                "versions_limit": 3,
                "edges_limit": 5,
            },
        })

        assert result["status"] in ("done", "error")

    def test_flow_execution_repairs_missing_sysdocs_before_subflow(self, kp):
        """Daemon flow execution repairs bundled state docs before loading subflows."""
        from keep.work_processor import _execute_flow_item

        collection = kp._resolve_doc_collection()
        kp._document_store.delete(collection, ".state/assess")
        kp._needs_sysdoc_migration = True

        result = _execute_flow_item(kp, {
            "state": STATE_PUT,
            "params": {
                "id": "flow-sysdoc-repair",
                "content": "stored after sysdoc repair",
                "queue_background_tasks": False,
            },
            "budget": 4,
        })

        assert result["status"] == "done"
        assert kp.get(".state/assess") is not None
        assert kp.get("flow-sysdoc-repair").summary == "stored after sysdoc repair"

    def test_flow_item_requires_state(self):
        """Flow work item without state name raises ValueError."""
        from keep.work_processor import _execute_flow_item
        from keep.api import Keeper

        with pytest.raises(ValueError, match="missing state name"):
            _execute_flow_item(None, {"params": {}})

    def test_after_write_flow_item_has_correct_shape(self, kp):
        """After-write enqueues a flow item with expected fields."""
        queue = kp._get_work_queue()
        queue.claim("drain", limit=200)

        kp.put("Content for shape test " * 30, id="shape-1")

        claimed = queue.claim("test", limit=10)
        flow_items = [i for i in claimed if i.kind == "flow"]
        assert len(flow_items) == 1

        item = flow_items[0]
        assert item.input["state"] == "after-write"
        assert item.input["item_id"] == "shape-1"
        assert "content" in item.input
        params = item.input["params"]
        assert params["item_id"] == "shape-1"
        assert "item" in params
        assert params["item"]["id"] == "shape-1"
        assert params["item"]["content_length"] > 0
        assert "summary" in params["item"]
        assert "tags" in params["item"]
        assert "system" in params
        assert "max_summary_length" in params


class TestEnqueueFlowCursor:
    """Verify _enqueue_flow_cursor puts cursor into work queue."""

    def test_enqueue_and_claim_cursor(self, kp):
        """Enqueued cursor appears as a 'flow' work item."""
        from keep.state_doc_runtime import encode_cursor

        queue = kp._get_work_queue()
        queue.claim("drain", limit=200)

        cursor_token = encode_cursor("test-state", 1, {"s": {"count": 5}})
        kp._enqueue_flow_cursor(
            state="test-state",
            cursor_token=cursor_token,
            params={"item_id": "x"},
            priority=5,
        )

        claimed = queue.claim("test", limit=10)
        flow_items = [i for i in claimed if i.kind == "flow"]
        assert len(flow_items) == 1
        assert flow_items[0].input["state"] == "test-state"
        assert flow_items[0].input["cursor"] == cursor_token

class TestVersionContextNavigation:
    """Regression tests for offset-based version navigation in get_context()."""

    def test_get_context_old_version_uses_offset_neighbors(self, mock_providers, tmp_path):
        from keep.api import Keeper
        from keep.types import Item

        kp = Keeper(store_path=tmp_path)

        # Current exists but should not be used for offset>1 next link.
        kp.get = lambda _id: Item(id="doc", summary="current", tags={"_created": "2026-01-04T00:00:00+00:00"})  # type: ignore[method-assign]

        # Simulate sparse internal version numbering behind a stable offset API.
        by_offset = {
            2: Item(id="doc", summary="viewed", tags={"_created": "2026-01-03T00:00:00+00:00", "_version": "100"}),
            3: Item(id="doc", summary="older", tags={"_created": "2026-01-02T00:00:00+00:00", "_version": "42"}),
            1: Item(id="doc", summary="newer", tags={"_created": "2026-01-04T00:00:00+00:00", "_version": "777"}),
        }
        kp.get_version = lambda _id, off=0: by_offset.get(off)  # type: ignore[method-assign]

        # Guard against regressions that pass offset into internal-version nav.
        kp.get_version_nav = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not call get_version_nav for old-version context"))  # type: ignore[method-assign]

        ctx = kp.get_context(
            "doc",
            version=2,
            include_similar=False,
            include_meta=False,
            include_parts=False,
            include_versions=True,
        )

        assert ctx is not None
        assert [v.offset for v in ctx.prev] == [3]
        assert [v.summary for v in ctx.prev] == ["older"]
        assert [v.offset for v in ctx.next] == [1]
        assert [v.summary for v in ctx.next] == ["newer"]


class TestNegativeVersionSelectors:
    """Public negative selectors resolve by oldest-ordinal semantics."""

    def test_resolve_version_offset_negative_from_oldest(self, mock_providers, tmp_path):
        from keep.api import Keeper

        kp = Keeper(store_path=tmp_path)
        kp._document_store.version_count = lambda _coll, _id: 5  # type: ignore[method-assign]

        assert kp.resolve_version_offset("doc", -1) == 5  # oldest
        assert kp.resolve_version_offset("doc", -2) == 4  # second-oldest
        assert kp.resolve_version_offset("doc", -5) == 1  # newest archived
        assert kp.resolve_version_offset("doc", -6) is None  # out of range

    def test_get_context_accepts_negative_selector(self, mock_providers, tmp_path):
        from keep.api import Keeper
        from keep.types import Item

        kp = Keeper(store_path=tmp_path)
        kp._document_store.version_count = lambda _coll, _id: 3  # type: ignore[method-assign]

        seen_offsets: list[int] = []

        def _get_version(_id: str, off: int = 0):
            seen_offsets.append(off)
            if off == 3:
                return Item(id="doc", summary="oldest", tags={"_created": "2026-01-01T00:00:00+00:00"})
            return None

        kp.get_version = _get_version  # type: ignore[method-assign]

        ctx = kp.get_context(
            "doc",
            version=-1,
            include_similar=False,
            include_meta=False,
            include_parts=False,
            include_versions=False,
        )
        assert ctx is not None
        assert ctx.item.summary == "oldest"
        assert seen_offsets == [3]
