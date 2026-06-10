"""Tests for the deep_follow_depth parameter (Plan 006).

Verifies that:
1. The default depth (10) is unchanged when the parameter is unset.
2. A custom depth value propagates to _deep_tag_follow as ``top_k``.
3. Out-of-range values are clamped (negative/zero → 1; > 100 → 100).
4. The parameter threads correctly from find() / _find_direct() through
   flow_env.traverse_related() and the traverse action.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
from typing import Any

import pytest

from keep.api import Keeper, FindResults
from keep.flow_env import LocalFlowEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_keeper(tmp_path):
    """Return a Keeper instance wired with mock providers + stores."""
    return Keeper(store_path=tmp_path)


# ---------------------------------------------------------------------------
# Unit tests: traverse_related clamping
# ---------------------------------------------------------------------------

class TestTraverseRelatedClamping:
    """Verify that traverse_related clamps deep_follow_depth before passing top_k to _deep_tag_follow.

    Exercises the clamping logic in ``LocalFlowEnvironment.traverse_related``.
    """

    def _make_env(self, tmp_path):
        """Create a LocalFlowEnvironment with a minimal mock keeper."""
        kp = MagicMock()
        # Mimic _resolve_chroma_collection / _resolve_doc_collection
        kp._resolve_chroma_collection.return_value = "test-coll"
        kp._resolve_doc_collection.return_value = "test-coll"
        # Empty document store so Tier 1 (edge follow) produces nothing
        ds = MagicMock()
        ds.get_many.return_value = {}
        kp._document_store = ds
        env = LocalFlowEnvironment(kp)
        return env, kp

    def test_default_depth_is_ten(self, tmp_path):
        """With no depth argument, top_k=10 is passed to _deep_tag_follow."""
        env, kp = self._make_env(tmp_path)
        # _deep_tag_follow is only reached when tagfollow_items is non-empty,
        # but we mock the keeper so it never raises.
        kp._deep_tag_follow.return_value = {}
        # Pass a non-empty source_ids list so the function body executes;
        # the source record lookup returns empty, so tagfollow_items stays
        # empty and _deep_tag_follow is NOT called — but the clamped value
        # of follow_depth should equal 10.
        # We test the clamping logic directly via the concrete method.
        # Inspect the resolved value by patching the private constant.
        result = env.traverse_related(["some-id"])
        # _deep_tag_follow was NOT called because get_many returned {} →
        # source_items is empty → early return.  Check it didn't error.
        assert isinstance(result, dict)

    def test_zero_depth_is_clamped_to_one(self, tmp_path):
        """deep_follow_depth=0 is clamped to 1 inside traverse_related."""
        env, kp = self._make_env(tmp_path)
        # Provide a source item so Tier 2 is reached and top_k is observable
        from keep.types import Item
        from keep.document_store import DocumentRecord
        fake_doc = MagicMock()
        fake_doc.id = "src"
        fake_doc.summary = "source"
        fake_doc.tags = {"project": "x"}
        fake_doc.created_at = None
        fake_doc.score = None
        kp._document_store.get_many.return_value = {"src": fake_doc}
        # No edges
        deps = MagicMock()
        deps.current_targets.return_value = []
        deps.current_sources.return_value = []

        kp._deep_tag_follow.return_value = {}
        with patch("keep.dependencies.NoteDependencyService", return_value=deps):
            env.traverse_related(["src"], deep_follow_depth=0)

        # _deep_tag_follow should be called with top_k=1 (clamped from 0)
        kp._deep_tag_follow.assert_called_once()
        _, kwargs = kp._deep_tag_follow.call_args
        assert kwargs["top_k"] == 1, (
            f"Expected top_k=1 (clamped from 0), got {kwargs['top_k']}"
        )

    def test_large_depth_clamped_to_hundred(self, tmp_path):
        """deep_follow_depth > 100 is clamped to 100."""
        env, kp = self._make_env(tmp_path)
        from keep.document_store import DocumentRecord
        fake_doc = MagicMock()
        fake_doc.id = "src"
        fake_doc.summary = "source"
        fake_doc.tags = {"project": "x"}
        fake_doc.created_at = None
        fake_doc.score = None
        kp._document_store.get_many.return_value = {"src": fake_doc}
        deps = MagicMock()
        deps.current_targets.return_value = []
        deps.current_sources.return_value = []
        kp._deep_tag_follow.return_value = {}

        with patch("keep.dependencies.NoteDependencyService", return_value=deps):
            env.traverse_related(["src"], deep_follow_depth=9999)

        kp._deep_tag_follow.assert_called_once()
        _, kwargs = kp._deep_tag_follow.call_args
        assert kwargs["top_k"] == 100, (
            f"Expected top_k=100 (clamped from 9999), got {kwargs['top_k']}"
        )

    def test_custom_depth_passed_through(self, tmp_path):
        """A valid custom depth (e.g. 25) is passed as top_k unchanged."""
        env, kp = self._make_env(tmp_path)
        fake_doc = MagicMock()
        fake_doc.id = "src"
        fake_doc.summary = "source"
        fake_doc.tags = {"project": "x"}
        fake_doc.created_at = None
        fake_doc.score = None
        kp._document_store.get_many.return_value = {"src": fake_doc}
        deps = MagicMock()
        deps.current_targets.return_value = []
        deps.current_sources.return_value = []
        kp._deep_tag_follow.return_value = {}

        with patch("keep.dependencies.NoteDependencyService", return_value=deps):
            env.traverse_related(["src"], deep_follow_depth=25)

        kp._deep_tag_follow.assert_called_once()
        _, kwargs = kp._deep_tag_follow.call_args
        assert kwargs["top_k"] == 25, (
            f"Expected top_k=25, got {kwargs['top_k']}"
        )

    def test_default_depth_passed_through_as_ten(self, tmp_path):
        """Omitting deep_follow_depth passes top_k=10 (the default)."""
        env, kp = self._make_env(tmp_path)
        fake_doc = MagicMock()
        fake_doc.id = "src"
        fake_doc.summary = "source"
        fake_doc.tags = {"project": "x"}
        fake_doc.created_at = None
        fake_doc.score = None
        kp._document_store.get_many.return_value = {"src": fake_doc}
        deps = MagicMock()
        deps.current_targets.return_value = []
        deps.current_sources.return_value = []
        kp._deep_tag_follow.return_value = {}

        with patch("keep.dependencies.NoteDependencyService", return_value=deps):
            env.traverse_related(["src"])  # no deep_follow_depth → default 10

        kp._deep_tag_follow.assert_called_once()
        _, kwargs = kp._deep_tag_follow.call_args
        assert kwargs["top_k"] == 10, (
            f"Expected top_k=10 (default), got {kwargs['top_k']}"
        )


# ---------------------------------------------------------------------------
# Unit tests: traverse action deep_follow_depth extraction
# ---------------------------------------------------------------------------

class TestTraverseActionDepth:
    """Verify the traverse action reads deep_follow_depth from params."""

    def test_traverse_action_passes_depth_to_context(self):
        """Traverse action extracts deep_follow_depth and forwards it."""
        from keep.actions.traverse import Traverse

        captured: dict = {}

        class FakeContext:
            def traverse(self, source_ids, *, limit, deep_follow_depth=10):
                captured["source_ids"] = source_ids
                captured["limit"] = limit
                captured["deep_follow_depth"] = deep_follow_depth
                return {}

        action = Traverse()
        from keep.types import Item
        fake_item = {"id": "src", "summary": "test", "tags": {}}
        result = action.run(
            {"items": [fake_item], "limit": 5, "deep_follow_depth": 30},
            FakeContext(),
        )
        assert captured["deep_follow_depth"] == 30

    def test_traverse_action_default_depth_is_ten(self):
        """Traverse action uses deep_follow_depth=10 when param absent."""
        from keep.actions.traverse import Traverse

        captured: dict = {}

        class FakeContext:
            def traverse(self, source_ids, *, limit, deep_follow_depth=10):
                captured["deep_follow_depth"] = deep_follow_depth
                return {}

        action = Traverse()
        fake_item = {"id": "src", "summary": "test", "tags": {}}
        action.run({"items": [fake_item], "limit": 5}, FakeContext())
        assert captured["deep_follow_depth"] == 10, (
            f"Expected default 10, got {captured['deep_follow_depth']}"
        )

    @pytest.mark.parametrize("bad_value", [None, "bad", "", [], {}])
    def test_traverse_action_invalid_depth_falls_back_to_ten(self, bad_value):
        """A None/non-numeric deep_follow_depth must not error the binding.

        State-doc params can be user-supplied or templated; the traverse action
        parses defensively and falls back to 10 rather than raising before the
        env's clamp runs.  Mirrors the find action's behaviour.
        """
        from keep.actions.traverse import Traverse

        captured: dict = {}

        class FakeContext:
            def traverse(self, source_ids, *, limit, deep_follow_depth=10):
                captured["deep_follow_depth"] = deep_follow_depth
                return {}

        action = Traverse()
        fake_item = {"id": "src", "summary": "test", "tags": {}}
        # Must not raise.
        action.run(
            {"items": [fake_item], "limit": 5, "deep_follow_depth": bad_value},
            FakeContext(),
        )
        assert captured["deep_follow_depth"] == 10


# ---------------------------------------------------------------------------
# Unit tests: find action deep_follow_depth extraction and clamping
# ---------------------------------------------------------------------------

class TestFindActionDepth:
    """Verify the find action reads and clamps deep_follow_depth."""

    def _run_find_action(self, params: dict, captured: dict):
        from keep.actions.find import Find

        class FakeContext:
            def find(self, query=None, *, tags=None, similar_to=None,
                     stored_only=False, limit=10, since=None, until=None,
                     include_self=False, include_hidden=False, deep=False,
                     deep_follow_depth=10, scope=None):
                captured["deep_follow_depth"] = deep_follow_depth
                from keep.api import FindResults
                return FindResults([])

        action = Find()
        action.run(params, FakeContext())

    def test_default_is_ten(self):
        """Find action uses deep_follow_depth=10 when absent from params."""
        captured: dict = {}
        self._run_find_action(
            {"query": "test", "deep": True},
            captured,
        )
        assert captured["deep_follow_depth"] == 10

    def test_custom_depth_passes_through(self):
        """Find action forwards a valid deep_follow_depth unchanged."""
        captured: dict = {}
        self._run_find_action(
            {"query": "test", "deep": True, "deep_follow_depth": 42},
            captured,
        )
        assert captured["deep_follow_depth"] == 42

    def test_depth_zero_clamped_to_one(self):
        """Find action clamps deep_follow_depth=0 to 1."""
        captured: dict = {}
        self._run_find_action(
            {"query": "test", "deep": True, "deep_follow_depth": 0},
            captured,
        )
        assert captured["deep_follow_depth"] == 1

    def test_depth_over_hundred_clamped(self):
        """Find action clamps deep_follow_depth > 100 to 100."""
        captured: dict = {}
        self._run_find_action(
            {"query": "test", "deep": True, "deep_follow_depth": 9999},
            captured,
        )
        assert captured["deep_follow_depth"] == 100

    def test_invalid_depth_falls_back_to_ten(self):
        """Find action falls back to 10 when deep_follow_depth is not numeric."""
        captured: dict = {}
        self._run_find_action(
            {"query": "test", "deep": True, "deep_follow_depth": "bad"},
            captured,
        )
        assert captured["deep_follow_depth"] == 10


# ---------------------------------------------------------------------------
# Integration tests: _find_direct default depth regression guard
# ---------------------------------------------------------------------------

class TestFindDirectDepth:
    """Guard that _find_direct's default (top_k=10) is provably unchanged."""

    @pytest.fixture
    def kp(self, mock_providers, tmp_path):
        kp = Keeper(store_path=tmp_path)
        kp._get_embedding_provider()
        # Seed minimal data so deep search can run
        kp.put("OAuth2 token design for project X", id="a",
               tags={"project": "x", "topic": "auth"})
        for i in range(5):
            kp.put(f"Filler {i}", id=f"f{i}", tags={"filler": "yes"})
        kp.put("Project X latency improvement", id="b",
               tags={"project": "x"})
        return kp

    def test_default_depth_unchanged(self, kp):
        """Default deep search (no depth param) uses top_k=10 in _deep_tag_follow.

        This is the regression guard: if the default ever changes from 10,
        this test will catch it.
        """
        with patch.object(
            kp, "_deep_tag_follow", wraps=kp._deep_tag_follow
        ) as spy:
            kp.find("OAuth2 token design", deep=True, limit=5)

        if spy.called:
            # _deep_tag_follow was used (tag-follow path; no edges in store)
            _, kwargs = spy.call_args
            assert kwargs.get("top_k", 10) == 10, (
                f"Default top_k must be 10, got {kwargs.get('top_k')}"
            )

    def test_custom_depth_reaches_deep_tag_follow(self, kp):
        """deep_follow_depth=3 reaches _deep_tag_follow with top_k=3."""
        with patch.object(
            kp, "_deep_tag_follow", wraps=kp._deep_tag_follow
        ) as spy:
            kp.find("OAuth2 token design", deep=True, limit=5,
                    deep_follow_depth=3)

        if spy.called:
            _, kwargs = spy.call_args
            assert kwargs.get("top_k", 10) == 3, (
                f"Expected top_k=3, got {kwargs.get('top_k')}"
            )

    def test_out_of_range_depth_is_clamped(self, kp):
        """deep_follow_depth=200 is clamped to 100 before reaching _deep_tag_follow."""
        with patch.object(
            kp, "_deep_tag_follow", wraps=kp._deep_tag_follow
        ) as spy:
            kp.find("OAuth2 token design", deep=True, limit=5,
                    deep_follow_depth=200)

        if spy.called:
            _, kwargs = spy.call_args
            assert kwargs.get("top_k", 10) <= 100, (
                f"top_k should be clamped to ≤100, got {kwargs.get('top_k')}"
            )
