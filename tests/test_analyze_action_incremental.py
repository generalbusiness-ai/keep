"""Tests for the analyze flow action's incremental path.

The incremental path is what protects vstring items like ``now`` from a
full sliding-window decomposition on every write. When ``_analyzed_version``
is recorded on the item, ``prepare()`` asks for a context+targets split
from the gather function and ``run()`` produces a single LLM call over the
overlap rather than dozens of windows over the whole version history.

Regression target: a previous bug routed daemon analyze through the
flow action's full path, which on a 1200-version ``now`` note generated
13-14 LLM calls and ~100 seconds of GPU work per fire.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from keep.actions.analyze import Analyze, _resolve_since_version


# ---------------------------------------------------------------------------
# _resolve_since_version: choosing when to take the incremental gather path
# ---------------------------------------------------------------------------

class TestResolveSinceVersion:
    def test_returns_int_when_vstring_has_analyzed_version(self):
        assert _resolve_since_version({"_analyzed_version": "42"}) == 42

    def test_none_when_no_analyzed_version(self):
        # First analyze on the item: nothing to overlap against, so the
        # action should take the full path and seed _analyzed_version.
        assert _resolve_since_version({}) is None

    def test_none_for_uri_sourced(self):
        # URI-backed items don't have a meaningful version thread —
        # they're re-fetched whole each time. Incremental makes no sense.
        assert _resolve_since_version({
            "_source": "uri",
            "_analyzed_version": "42",
        }) is None

    def test_none_for_unparseable_version(self):
        assert _resolve_since_version({"_analyzed_version": "v1.2.3"}) is None

    def test_none_for_empty_string(self):
        assert _resolve_since_version({"_analyzed_version": ""}) is None


# ---------------------------------------------------------------------------
# prepare(): incremental gather pulls context+targets when _analyzed_version
# is set, so run() can emit a single-window analyze instead of fanning out.
# ---------------------------------------------------------------------------

def _ctx(*, gather_result, item, **overrides):
    """Build a minimal action context with the surface the action uses.

    Defaults give a healthy item (no _analyzed_hash collision, no tags),
    a stub gather function that returns ``gather_result``, and stubs for
    every helper the action calls. Override individual fields to assert
    behavior under specific conditions.
    """
    ctx = MagicMock()
    ctx.get.return_value = item
    ctx.get_document.return_value = MagicMock(
        tags=dict(item.tags or {}),
        content_hash="hash-xyz",
    )
    ctx.list_versions.return_value = [MagicMock(version=99)]
    ctx.list_items.return_value = []  # load_tag_specs short-circuits
    ctx.gather_analyze_chunks = MagicMock(return_value=gather_result)
    ctx.gather_guide_context = MagicMock(return_value="")
    ctx.resolve_prompt = MagicMock(return_value="analyze prompt")
    ctx.load_prompt_doc = MagicMock(return_value="incremental prompt")
    ctx.max_part_num = MagicMock(return_value=0)
    ctx.resolve_provider = MagicMock()
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def _item(tags):
    item = MagicMock()
    item.tags = tags
    item.summary = "current state"
    return item


class TestPrepareIncrementalGather:
    def test_passes_since_version_from_analyzed_version_tag(self):
        item = _item({"_analyzed_version": "10"})
        ctx = _ctx(
            item=item,
            gather_result={"context": [], "targets": [{"content": "new"}]},
        )

        Analyze().prepare({"item_id": "now"}, ctx)

        ctx.gather_analyze_chunks.assert_called_once()
        kwargs = ctx.gather_analyze_chunks.call_args.kwargs
        assert kwargs["since_version"] == 10

    def test_full_path_when_no_analyzed_version(self):
        # First analyze: since_version=None means the gather returns a
        # flat list, which run() feeds to the sliding-window analyzer.
        item = _item({})
        ctx = _ctx(
            item=item,
            gather_result=[{"content": "v1"}, {"content": "v2"}],
        )

        prepared = Analyze().prepare({"item_id": "now"}, ctx)

        assert ctx.gather_analyze_chunks.call_args.kwargs["since_version"] is None
        assert "chunks_targets" not in prepared
        assert prepared.get("chunks") == [{"content": "v1"}, {"content": "v2"}]

    def test_force_disables_incremental(self):
        # `force=True` is the explicit "rebuild from scratch" lever — it
        # should bypass the incremental overlap so re-analyze rebuilds
        # parts even when _analyzed_version is set.
        item = _item({"_analyzed_version": "10"})
        ctx = _ctx(item=item, gather_result=[{"content": "full"}])

        Analyze().prepare({"item_id": "now", "force": True}, ctx)

        assert ctx.gather_analyze_chunks.call_args.kwargs["since_version"] is None

    def test_splits_dict_gather_into_chunks_context_and_targets(self):
        # The incremental gather shape `{context: [...], targets: [...]}`
        # must NOT be flattened into prepared["chunks"]; run() needs to
        # build the <analyze>-marked prompt and so requires them separate.
        item = _item({"_analyzed_version": "10"})
        ctx = _ctx(
            item=item,
            gather_result={
                "context": [{"content": "overlap"}],
                "targets": [{"content": "new1"}, {"content": "new2"}],
            },
        )

        prepared = Analyze().prepare({"item_id": "now"}, ctx)

        assert prepared["chunks_context"] == [{"content": "overlap"}]
        assert prepared["chunks_targets"] == [
            {"content": "new1"}, {"content": "new2"},
        ]
        assert "chunks" not in prepared

    def test_loads_incremental_prompt_when_incremental(self):
        item = _item({"_analyzed_version": "10"})
        ctx = _ctx(
            item=item,
            gather_result={"context": [], "targets": [{"content": "new"}]},
        )

        prepared = Analyze().prepare({"item_id": "now"}, ctx)

        ctx.load_prompt_doc.assert_called_once_with(
            ".prompt/analyze/incremental", required=True,
        )
        assert prepared.get("incremental_prompt") == "incremental prompt"


# ---------------------------------------------------------------------------
# run(): incremental path makes ONE LLM call and emits APPEND-ONLY mutations.
# ---------------------------------------------------------------------------

class TestRunIncrementalPath:
    def _setup(self, *, max_part=3, generate_result="New theme appeared"):
        item = _item({
            "_analyzed_version": "10",
            "_source": "inline",  # vstring
        })
        ctx = _ctx(
            item=item,
            gather_result={
                "context": [{"content": "[2026-05-01]\nprior version body"}],
                "targets": [
                    {"content": "[2026-05-02]\nnew version body 1"},
                    {"content": "[current]\ncurrent state"},
                ],
            },
        )
        ctx.max_part_num.return_value = max_part
        provider = MagicMock()
        provider._provider = None  # no caching wrapper to unwrap
        provider.generate.return_value = generate_result
        ctx.resolve_provider.return_value = provider
        ctx.get_document.return_value.tags = dict(item.tags)
        return ctx, provider

    def test_makes_exactly_one_provider_call(self):
        # The whole point of incremental: collapse the sliding-window
        # fan-out into one LLM call. Multiple calls here would mean we
        # regressed back to the old behavior.
        ctx, provider = self._setup()

        Analyze().run({"item_id": "now"}, ctx)

        assert provider.generate.call_count == 1

    def test_prompt_marks_targets_with_analyze_tags(self):
        # The context overlap is what gives the incremental prompt the
        # narrative-change signal the user explicitly cares about. If we
        # ever stop wrapping targets in <analyze> the model would treat
        # everything as new.
        ctx, provider = self._setup()

        Analyze().run({"item_id": "now"}, ctx)

        system_prompt, user_prompt = provider.generate.call_args.args[:2]
        assert system_prompt == "incremental prompt"
        assert "<content>" in user_prompt
        assert "<analyze>" in user_prompt
        assert "</analyze>" in user_prompt
        # Context must appear BEFORE the <analyze> marker.
        analyze_idx = user_prompt.index("<analyze>")
        assert "prior version body" in user_prompt[:analyze_idx]
        # Targets must appear INSIDE the <analyze> block.
        analyze_block = user_prompt[analyze_idx : user_prompt.index("</analyze>")]
        assert "new version body 1" in analyze_block
        assert "current state" in analyze_block

    def test_emits_no_delete_prefix_mutation(self):
        # Append-only semantics: the previously-analyzed parts must
        # survive. delete_prefix would wipe the trajectory we built up.
        ctx, _ = self._setup()

        out = Analyze().run({"item_id": "now"}, ctx)

        ops = [m["op"] for m in out["mutations"]]
        assert "delete_prefix" not in ops

    def test_new_parts_continue_numbering_from_max_part_num(self):
        # max_part_num was 3 → new parts must be p4, p5, ... so they
        # don't overwrite existing parts on disk. Summaries must be
        # ≥20 chars to survive _parse_parts' filter.
        ctx, _ = self._setup(
            max_part=3,
            generate_result=(
                "New theme appearing in this iteration\n"
                "Second distinct shift observed downstream"
            ),
        )

        out = Analyze().run({"item_id": "now"}, ctx)

        put_ops = [m for m in out["mutations"] if m["op"] == "put_item"]
        ids = [m["id"] for m in put_ops]
        assert ids == ["now@p4", "now@p5"]
        part_nums = [m["tags"]["_part_num"] for m in put_ops]
        assert part_nums == ["4", "5"]

    def test_records_analyzed_version_even_when_no_new_parts(self):
        # If the LLM finds nothing new, we still must bump
        # _analyzed_version. Otherwise the next write would gather the
        # same targets again and re-run the LLM call indefinitely.
        ctx, provider = self._setup(generate_result="EMPTY")

        out = Analyze().run({"item_id": "now"}, ctx)

        set_tags = [m for m in out["mutations"] if m["op"] == "set_tags"]
        assert len(set_tags) == 1
        # The version bump comes from list_versions(limit=1) → version=99.
        assert set_tags[0]["tags"]["_analyzed_version"] == "99"


# ---------------------------------------------------------------------------
# Regression: the over-budget incremental fallback used to loop forever.
# run_local_task pre-populates `chunks_targets` via prepare(), then passes
# the prepared params into run(). _run_incremental's fallback recursed via
# self.run({**params, "force": True}, context); but prepare()'s gather
# condition saw chunks_targets and skipped re-gathering, so run() re-
# entered _run_incremental with the same over-budget data — RecursionError.
# ---------------------------------------------------------------------------

class TestOverBudgetFallbackDoesNotRecurse:
    def test_force_clears_caller_prepared_incremental_chunks(self):
        # When prepare() is called with chunks_targets already in params AND
        # force=True, it must wipe them so the gather actually re-runs as
        # full. Without this, _run_incremental's over-budget fallback would
        # recurse with the same incremental shape and loop until stack overflow.
        item = _item({"_analyzed_version": "10"})
        ctx = _ctx(
            item=item,
            gather_result=[{"content": "fresh full"}],
        )

        prepared = Analyze().prepare(
            {
                "item_id": "now",
                "force": True,
                # Stale incremental params left over from an upstream
                # prepare() call — simulates run_local_task feeding back.
                "chunks_targets": [{"content": "stale target"}],
                "chunks_context": [{"content": "stale context"}],
                "incremental_prompt": "stale prompt",
            },
            ctx,
        )

        # Gather must have been called (incremental params were cleared
        # despite being pre-populated).
        ctx.gather_analyze_chunks.assert_called_once()
        # And it returned a flat list, so prepared now holds chunks (not
        # chunks_targets).
        assert prepared.get("chunks") == [{"content": "fresh full"}]
        assert "chunks_targets" not in prepared
        assert "incremental_prompt" not in prepared

    def test_over_budget_incremental_falls_back_without_recursing(self):
        # End-to-end: incremental run with content over DEFAULT_CONTEXT_BUDGET
        # falls back to full once and terminates — not RecursionError.
        from keep.analyzers import DEFAULT_CONTEXT_BUDGET

        # Build target chunks that exceed the budget. _estimate_tokens uses
        # len/4, so each ~80,000-char chunk is ~20,000 tokens — well past the
        # 12,000-token budget.
        huge_chunk = {"content": "x" * (DEFAULT_CONTEXT_BUDGET * 4 + 1000)}

        item = _item({"_analyzed_version": "10", "_source": "inline"})
        ctx = _ctx(
            item=item,
            gather_result={"context": [], "targets": [huge_chunk]},
        )

        # The fallback re-runs the action with force=True; on the second
        # pass the gather should return a flat (full) list so we don't
        # re-enter the incremental path. We use a side_effect to return
        # different shapes on first vs second call.
        full_chunks = [{"content": "smaller full content", "index": 0}]
        ctx.gather_analyze_chunks.side_effect = [
            {"context": [], "targets": [huge_chunk]},  # initial incremental
            full_chunks,                                # forced full re-gather
        ]
        # Resolve a full-path analyzer so run() can complete the full pass.
        analyzer = MagicMock()
        analyzer.analyze.return_value = [
            {"summary": "Full pass part one summary text", "tags": {}},
        ]
        # resolve_provider is called with both "analyzer" and "summarization";
        # return the same mock for both.
        ctx.resolve_provider.return_value = analyzer

        # This used to recurse until RecursionError. After the fix, it should
        # return normally.
        result = Analyze().run({"item_id": "now"}, ctx)

        # We must have re-gathered exactly twice (incremental, then full).
        assert ctx.gather_analyze_chunks.call_count == 2
        # And the result is from the full path — incremental flag is absent.
        assert result.get("incremental") is not True


# ---------------------------------------------------------------------------
# Backlog handling: rebase-to-newest semantic.
#
# `list_versions(limit=100)` returns only the newest 100 versions. When more
# than 100 versions have accumulated since the cursor, the gap between
# `since_version` and the oldest fetched version CANNOT be reconstructed —
# those versions are gone from the analyze window. This is the *intentional*
# semantic for sliding-window analyze of vstrings like `now`: we care about
# the current trajectory, not exhaustive coverage of historical churn. The
# gather switches to the full path, the analyzer rebuilds parts from the
# visible window, and `_record_analyzed_tags` advances `_analyzed_version`
# to the latest — the newest window becomes the new baseline.
#
# These tests document and lock that intent. If you want exhaustive-coverage
# semantics ("never skip an unanalyzed version") you must change the design
# AND these tests.
# ---------------------------------------------------------------------------

class TestGatherBacklogRebasesToNewest:
    def test_backlog_returns_flat_list_not_incremental_dict(self, tmp_path):
        # When since_version=10 but oldest fetched=100 (89-version gap),
        # gather must drop the dict shape so the caller switches to the
        # full path. Returning the dict would attribute "newness" to
        # versions 100..199 and silently lose 11..99 from the analysis.
        from keep.api import Keeper
        from keep.document_store import VersionInfo
        from unittest.mock import patch

        kp = Keeper(store_path=tmp_path)
        kp.put("Some content for testing the gather backlog path", id="doc-backlog")

        doc_coll = kp._resolve_doc_collection()
        doc = kp._document_store.get(doc_coll, "doc-backlog")

        backlog_versions = [
            VersionInfo(
                version=v,
                summary=f"Version {v} content body",
                tags={},
                created_at=f"2026-05-{(v % 28) + 1:02d}T00:00:00",
                content_hash=f"hash_{v}",
            )
            for v in range(199, 99, -1)  # newest-first
        ]

        with patch.object(
            kp._document_store, "list_versions", return_value=backlog_versions,
        ):
            result = kp._gather_analyze_chunks(
                "doc-backlog", doc, since_version=10,
            )

        assert isinstance(result, list)

    def test_backlog_logs_warning_so_drop_is_visible(self, tmp_path, caplog):
        # The dropped gap is a real loss of analysis coverage even though
        # it's intentional. Operators need to see it in logs so they can
        # tell "daemon was offline for a while" from "everything's fine".
        import logging
        from keep.api import Keeper
        from keep.document_store import VersionInfo
        from unittest.mock import patch

        kp = Keeper(store_path=tmp_path)
        kp.put("doc body", id="doc-warn")
        doc_coll = kp._resolve_doc_collection()
        doc = kp._document_store.get(doc_coll, "doc-warn")

        versions = [
            VersionInfo(
                version=v, summary=f"v{v}", tags={},
                created_at=f"2026-05-{(v % 28) + 1:02d}T00:00:00",
                content_hash=f"hash_{v}",
            )
            for v in range(199, 99, -1)
        ]

        with caplog.at_level(logging.WARNING, logger="keep.api"):
            with patch.object(
                kp._document_store, "list_versions", return_value=versions,
            ):
                kp._gather_analyze_chunks(
                    "doc-warn", doc, since_version=10,
                )

        # Look for the rebase warning specifically. It must name the gap
        # size so the operator can tell whether the drop matters.
        rebase_logs = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "Rebasing to the newest window" in r.message
        ]
        assert rebase_logs, (
            "Expected a WARNING-level log naming the rebase; got: "
            f"{[r.message for r in caplog.records]}"
        )
        assert "89 versions in the gap" in rebase_logs[0].message

    def test_backlog_rebase_advances_cursor_to_latest_intentionally(
        self, tmp_path, mock_providers,
    ):
        # This test locks in the *intent* of the rebase semantic: after
        # the full-path analyze runs on the visible window, the cursor
        # advances to the latest version even though the gap was not
        # analyzed. If this assertion ever fails, someone is mid-way
        # through changing the semantic — either commit to that change or
        # restore the rebase behavior.
        from keep.api import Keeper
        from keep.document_store import VersionInfo
        from unittest.mock import MagicMock, patch

        kp = Keeper(store_path=tmp_path)
        kp.put("Some content for rebase cursor test", id="doc-cursor")
        doc_coll = kp._resolve_doc_collection()

        # Seed _analyzed_version=10 so the next analyze takes the
        # incremental path that triggers the backlog guard.
        doc = kp._document_store.get(doc_coll, "doc-cursor")
        updated_tags = dict(doc.tags)
        updated_tags["_analyzed_version"] = "10"
        kp._document_store.update_tags(doc_coll, "doc-cursor", updated_tags)

        # Pretend the document has 100 versions starting from 100, leaving
        # 11..99 as the unanalyzed gap.
        versions = [
            VersionInfo(
                version=v, summary=f"version {v} body content here",
                tags={},
                created_at=f"2026-05-{(v % 28) + 1:02d}T00:00:00",
                content_hash=f"hash_{v}",
            )
            for v in range(199, 99, -1)
        ]
        # `_record_analyzed_tags` calls `list_versions(..., limit=1)` and
        # uses the first row's version. Patch that to also return our
        # synthetic "latest" so the cursor lands on 199.
        latest_only = [versions[0]]

        def fake_list_versions(_coll, _id, limit=10):
            return latest_only if limit == 1 else versions

        with patch.object(
            kp._document_store, "list_versions",
            side_effect=fake_list_versions,
        ):
            # Stub the analyzer so the test doesn't depend on real LLMs.
            with patch("keep.analyzers.SlidingWindowAnalyzer.analyze") as mock_llm:
                mock_llm.return_value = [
                    {"summary": "Some analyzed part summary text", "tags": {}},
                    {"summary": "Another analyzed part summary", "tags": {}},
                ]
                kp.analyze("doc-cursor")

        doc_after = kp._document_store.get(doc_coll, "doc-cursor")
        assert doc_after.tags.get("_analyzed_version") == "199", (
            "Rebase semantic requires the cursor to advance to latest "
            "after the full pass on the visible window — even though "
            "versions 11..99 were never analyzed. If you intended to "
            "preserve a 'never skip' guarantee, that's a design change."
        )

    def test_contiguous_window_still_does_incremental(self, tmp_path):
        # Sanity check: when since_version is contiguous with the fetched
        # window (oldest fetched = since_version + 1, or there's overlap),
        # incremental still kicks in. Otherwise the backlog guard would be
        # too aggressive and disable incremental for every healthy case.
        from keep.api import Keeper
        from keep.document_store import VersionInfo
        from unittest.mock import patch

        kp = Keeper(store_path=tmp_path)
        kp.put("Some content", id="doc-contig")

        doc_coll = kp._resolve_doc_collection()
        doc = kp._document_store.get(doc_coll, "doc-contig")

        # since_version=50, fetched 51..150 (oldest = 51 = since_version + 1).
        contiguous_versions = [
            VersionInfo(
                version=v, summary=f"v{v}", tags={},
                created_at=f"2026-05-{(v % 28) + 1:02d}T00:00:00",
                content_hash=f"hash_{v}",
            )
            for v in range(150, 50, -1)
        ]

        with patch.object(
            kp._document_store, "list_versions", return_value=contiguous_versions,
        ):
            result = kp._gather_analyze_chunks(
                "doc-contig", doc, since_version=50,
            )

        # Contiguous → incremental shape returned.
        assert isinstance(result, dict)
        assert "context" in result
        assert "targets" in result
