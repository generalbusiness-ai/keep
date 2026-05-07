"""Tests for work_processor and task_workflows — the background task execution path.

Covers:
- process_work_batch: claim → execute → complete/fail lifecycle
- _execute_work_item: input parsing, outcome propagation
- Supersede skipping
- Error handling and stats
- run_local_task dispatch via action registry
- Action run() skip conditions
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from keep.work_queue import WorkQueue
from keep.work_processor import process_work_batch, _execute_work_item
from keep.task_workflows import (
    TaskRequest,
    TaskRunResult,
    run_local_task,
    _apply_mutations,
    _resolve_ref,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def queue(tmp_path):
    q = WorkQueue(tmp_path / "work.db")
    yield q
    q.close()


@pytest.fixture
def mock_keeper():
    """Minimal mock Keeper for work processor tests."""
    kp = MagicMock()
    kp._resolve_doc_collection.return_value = "default"
    kp._run_local_task_workflow.return_value = {"status": "applied", "details": {}}
    return kp


# ---------------------------------------------------------------------------
# process_work_batch
# ---------------------------------------------------------------------------

class TestProcessWorkBatch:
    """Tests for work batch processing."""
    def test_empty_queue_returns_zero_stats(self, mock_keeper, queue):
        stats = process_work_batch(mock_keeper, queue, limit=5)
        assert stats["claimed"] == 0
        assert stats["processed"] == 0
        assert stats["failed"] == 0

    def test_processes_single_item(self, mock_keeper, queue):
        queue.enqueue("analyze", {"task_type": "analyze", "item_id": "doc1", "content": "x"})
        stats = process_work_batch(mock_keeper, queue, limit=5, worker_id="w1")
        assert stats["claimed"] == 1
        assert stats["processed"] == 1
        assert stats["failed"] == 0
        mock_keeper._run_local_task_workflow.assert_called_once()

    def test_processes_multiple_items(self, mock_keeper, queue):
        for i in range(3):
            queue.enqueue("tag", {"task_type": "tag", "item_id": f"d{i}", "content": "x"})
        stats = process_work_batch(mock_keeper, queue, limit=10, worker_id="w1")
        assert stats["claimed"] == 3
        assert stats["processed"] == 3

    def test_failure_increments_failed_count(self, mock_keeper, queue):
        mock_keeper._run_local_task_workflow.side_effect = RuntimeError("boom")
        queue.enqueue("analyze", {"task_type": "analyze", "item_id": "doc1", "content": "x"})
        stats = process_work_batch(mock_keeper, queue, limit=5, worker_id="w1")
        assert stats["failed"] == 1
        assert stats["processed"] == 0
        assert len(stats["errors"]) == 1
        assert "boom" in stats["errors"][0]["error"]

    def test_mixed_success_and_failure(self, mock_keeper, queue):
        queue.enqueue("tag", {"task_type": "tag", "item_id": "ok1", "content": "x"})
        queue.enqueue("tag", {"task_type": "tag", "item_id": "bad", "content": "x"})
        queue.enqueue("tag", {"task_type": "tag", "item_id": "ok2", "content": "x"})

        call_count = [0]
        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("bad item")
            return {"status": "applied", "details": {}}
        mock_keeper._run_local_task_workflow.side_effect = side_effect

        stats = process_work_batch(mock_keeper, queue, limit=10, worker_id="w1")
        assert stats["processed"] == 2
        assert stats["failed"] == 1

    def test_superseded_item_completed_not_executed(self, mock_keeper, queue):
        # Enqueue two items with same supersede key.
        # The second enqueue marks the first as 'superseded' at enqueue time,
        # so only the newer item is claimable. Verify the newer one executes.
        queue.enqueue("analyze", {"task_type": "analyze", "item_id": "doc1", "content": "v1"},
                       supersede_key="analyze:doc1")
        queue.enqueue("analyze", {"task_type": "analyze", "item_id": "doc1", "content": "v2"},
                       supersede_key="analyze:doc1")
        stats = process_work_batch(mock_keeper, queue, limit=10, worker_id="w1")
        # Only the newer item is claimed and executed
        assert stats["claimed"] == 1
        assert stats["processed"] == 1
        assert stats["failed"] == 0
        assert mock_keeper._run_local_task_workflow.call_count == 1

    def test_skipped_outcome_logged_as_processed(self, mock_keeper, queue):
        mock_keeper._run_local_task_workflow.return_value = {
            "status": "skipped", "details": {"reason": "content_too_short"},
        }
        queue.enqueue("analyze", {"task_type": "analyze", "item_id": "doc1", "content": "x"})
        stats = process_work_batch(mock_keeper, queue, limit=5, worker_id="w1")
        assert stats["processed"] == 1
        assert stats["failed"] == 0

    def test_limit_respected(self, mock_keeper, queue):
        for i in range(5):
            queue.enqueue("tag", {"task_type": "tag", "item_id": f"d{i}", "content": "x"})
        stats = process_work_batch(mock_keeper, queue, limit=2, worker_id="w1")
        assert stats["claimed"] == 2
        assert stats["processed"] == 2

    def test_stopped_flow_is_requeued_with_cursor(self, mock_keeper, queue):
        queue.enqueue(
            "flow",
            {"state": "query-resolve", "params": {"query": "test"}, "item_id": "doc1"},
            supersede_key="flow:doc1",
            priority=2,
        )
        with patch("keep.work_processor._execute_work_item", return_value={
            "status": "stopped",
            "cursor": "cursor-123",
            "state": "query-resolve",
            "details": {"data": {"reason": "shutdown"}},
        }):
            stats = process_work_batch(mock_keeper, queue, limit=5, worker_id="w1")

        assert stats["processed"] == 1
        resumed = queue.claim("w2", limit=5)
        assert len(resumed) == 1
        assert resumed[0].kind == "flow"
        assert resumed[0].priority == 2
        assert resumed[0].supersede_key == "flow:doc1"
        assert resumed[0].input["cursor"] == "cursor-123"

    def test_error_flow_is_failed_for_retry_backoff(self, mock_keeper, queue):
        queue.enqueue(
            "flow",
            {"state": "after-write", "params": {}, "item_id": "doc1"},
            supersede_key="flow:doc1",
            priority=3,
        )
        with patch("keep.work_processor._execute_work_item", return_value={
            "status": "error",
            "state": "after-write",
            "details": {"data": {"reason": "Cannot reach Ollama at http://localhost:11434"}},
        }):
            stats = process_work_batch(mock_keeper, queue, limit=5, worker_id="w1")

        assert stats["failed"] == 1
        assert stats["processed"] == 0
        row = queue._conn.execute(
            "SELECT status, retry_after, last_error FROM continue_work"
        ).fetchone()
        assert row["status"] == "requested"
        assert row["retry_after"]
        assert "Cannot reach Ollama" in row["last_error"]

    def test_done_flow_with_action_error_is_failed_for_retry_backoff(self, mock_keeper, queue):
        queue.enqueue(
            "flow",
            {"state": "after-write", "params": {}, "item_id": "doc1"},
            supersede_key="flow:doc1",
            priority=3,
        )
        with patch("keep.work_processor._execute_work_item", return_value={
            "status": "done",
            "state": "after-write",
            "details": {"data": {"summary": {"error": "Cannot reach Ollama at http://localhost:11434"}}},
        }):
            stats = process_work_batch(mock_keeper, queue, limit=5, worker_id="w1")

        assert stats["failed"] == 1
        row = queue._conn.execute(
            "SELECT status, retry_after, last_error FROM continue_work"
        ).fetchone()
        assert row["status"] == "requested"
        assert row["retry_after"]
        assert "summary: Cannot reach Ollama" in row["last_error"]

    def test_done_flow_with_validation_error_completes(self, mock_keeper, queue):
        queue.enqueue(
            "flow",
            {"state": "after-write", "params": {}, "item_id": "doc1"},
            supersede_key="flow:doc1",
            priority=3,
        )
        with patch("keep.work_processor._execute_work_item", return_value={
            "status": "done",
            "state": "after-write",
            "details": {"data": {"summary": {"error": "item content unavailable: doc1"}}},
        }):
            stats = process_work_batch(mock_keeper, queue, limit=5, worker_id="w1")

        assert stats["processed"] == 1
        assert stats["failed"] == 0
        row = queue._conn.execute(
            "SELECT status, retry_after, last_error FROM continue_work"
        ).fetchone()
        assert row["status"] == "completed"
        assert row["retry_after"] is None


# ---------------------------------------------------------------------------
# _execute_work_item
# ---------------------------------------------------------------------------

class TestExecuteWorkItem:
    """Tests for work item execution."""
    def test_extracts_fields_from_input(self, mock_keeper):
        _execute_work_item(mock_keeper, "analyze", {
            "task_type": "analyze",
            "item_id": "doc1",
            "collection": "coll1",
            "content": "hello",
            "metadata": {"force": True},
        })
        mock_keeper._run_local_task_workflow.assert_called_once_with(
            task_type="analyze",
            item_id="doc1",
            collection="coll1",
            content="hello",
            metadata={"force": True},
        )

    def test_missing_item_id_raises(self, mock_keeper):
        with pytest.raises(ValueError, match="missing item_id"):
            _execute_work_item(mock_keeper, "tag", {"content": "x"})

    def test_falls_back_to_kind_for_task_type(self, mock_keeper):
        _execute_work_item(mock_keeper, "tag", {"item_id": "d1", "content": "x"})
        call_kwargs = mock_keeper._run_local_task_workflow.call_args.kwargs
        assert call_kwargs["task_type"] == "tag"

    def test_falls_back_to_resolve_collection(self, mock_keeper):
        mock_keeper._resolve_doc_collection.return_value = "fallback_coll"
        _execute_work_item(mock_keeper, "tag", {"item_id": "d1", "content": "x"})
        call_kwargs = mock_keeper._run_local_task_workflow.call_args.kwargs
        assert call_kwargs["collection"] == "fallback_coll"

    def test_non_dict_metadata_becomes_empty_dict(self, mock_keeper):
        _execute_work_item(mock_keeper, "tag", {
            "item_id": "d1", "content": "x", "metadata": "not-a-dict",
        })
        call_kwargs = mock_keeper._run_local_task_workflow.call_args.kwargs
        assert call_kwargs["metadata"] == {}

    def test_returns_outcome(self, mock_keeper):
        mock_keeper._run_local_task_workflow.return_value = {"status": "skipped", "details": {"reason": "x"}}
        result = _execute_work_item(mock_keeper, "tag", {"item_id": "d1", "content": "x"})
        assert result["status"] == "skipped"

    def test_none_outcome_defaults_to_applied(self, mock_keeper):
        mock_keeper._run_local_task_workflow.return_value = None
        result = _execute_work_item(mock_keeper, "tag", {"item_id": "d1", "content": "x"})
        assert result["status"] == "applied"


# ---------------------------------------------------------------------------
# run_local_task dispatch
# ---------------------------------------------------------------------------

class TestRunLocalTask:
    """Tests for local task execution."""
    def test_unsupported_task_type_raises(self):
        kp = MagicMock()
        req = TaskRequest(task_type="bogus", id="d1", collection="c", content="x")
        with pytest.raises(ValueError, match="unknown action"):
            run_local_task(kp, req)

    def test_dispatches_to_analyze(self):
        """Analyze action runs and produces parts via run_local_task."""
        kp = MagicMock()
        mock_item = MagicMock(content="Long content for analysis " * 10, summary="", tags={})
        kp.get.return_value = mock_item
        # Mock analyzer to produce parts
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [
            {"summary": "Part 1", "tags": {}},
        ]
        kp._get_analyzer.return_value = mock_analyzer
        kp.list_items.return_value = []  # no tag specs
        kp._document_store.get.return_value = None  # no doc for _analyzed_hash

        req = TaskRequest(task_type="analyze", id="d1", collection="c", content="x")
        result = run_local_task(kp, req)
        assert result.status == "applied"

    def test_summarize_uses_prepared_context_and_prompt(self):
        """run_local_task should use summarize.prepare() inputs before execution."""
        kp = MagicMock()
        mock_item = MagicMock(
            content="Long content for summary",
            summary="",
            tags={"topic": "auth", "_source": "inline"},
        )
        kp.get.return_value = mock_item
        kp._gather_context.return_value = "related auth context"
        kp._resolve_prompt_doc.return_value = "Summarize with auth framing"
        provider = MagicMock()
        provider.summarize.return_value = "summary result"
        kp._get_summarization_provider.return_value = provider
        kp._document_store.get.return_value = None

        req = TaskRequest(task_type="summarize", id="d1", collection="c", content="Long content for summary")
        result = run_local_task(kp, req)

        assert result.status == "applied"
        provider.summarize.assert_called_once_with(
            "Long content for summary",
            max_length=500,
            context="related auth context",
            system_prompt="Summarize with auth framing",
        )

    def test_analyze_uses_prepared_chunks_and_guide_context(self):
        """run_local_task should use analyze.prepare() inputs before execution."""
        kp = MagicMock()
        mock_item = MagicMock(content="", summary="Current summary", tags={"type": "note"})
        kp.get.return_value = mock_item
        kp._gather_analyze_chunks.return_value = [
            {"content": "context chunk", "tags": {"kind": "context"}, "index": 0},
            {"content": "target chunk", "tags": {"kind": "target"}, "index": 1},
        ]
        kp._gather_guide_context.return_value = "guide text"
        kp._resolve_prompt_doc.return_value = "Analyze with prompt"
        kp.list_items.return_value = []
        kp._document_store.get.return_value = None
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [
            {"summary": "Part 1", "tags": {}},
        ]
        kp._get_analyzer.return_value = mock_analyzer

        req = TaskRequest(
            task_type="analyze",
            id="d1",
            collection="c",
            content="ignored",
            metadata={"tags": ["topic"]},
        )
        result = run_local_task(kp, req)

        assert result.status == "applied"
        kp._gather_analyze_chunks.assert_called_once_with("d1", mock_item)
        kp._gather_guide_context.assert_called_once_with(["topic"])
        analyze_args = mock_analyzer.analyze.call_args.args
        assert len(analyze_args[0]) == 2
        assert analyze_args[1] == "guide text"
        assert mock_analyzer.analyze.call_args.kwargs["prompt_override"] == "Analyze with prompt"

    def test_dispatches_to_tag(self):
        """Auto-tag action runs through run_local_task."""
        kp = MagicMock()
        mock_item = MagicMock(content="some content", summary="some content", tags={})
        kp.get.return_value = mock_item
        kp.list_items.return_value = []  # no tag specs → no mutations
        kp._get_summarization_provider.return_value = MagicMock()

        req = TaskRequest(
            task_type="auto_tag", id="d1", collection="c", content="some content",
        )
        result = run_local_task(kp, req)
        # No tag specs → no mutations → skipped
        assert result.status == "skipped"

    def test_run_local_task_recovers_and_retries_on_malformed_db(self, caplog):
        kp = MagicMock()
        kp.get.return_value = MagicMock(content="hello", summary="", tags={})
        kp._document_store.get.return_value = None
        kp._document_store._try_runtime_recover.return_value = True

        action = MagicMock()
        action.run.side_effect = [
            sqlite3.DatabaseError("database disk image is malformed"),
            {"skipped": True, "reason": "recovered"},
        ]

        with patch(
            "keep.actions.prepare_action_params",
            return_value=(action, {"item_id": "d1"}),
        ):
            req = TaskRequest(task_type="summarize", id="d1", collection="c", content="x")
            result = run_local_task(kp, req)

        assert result.status == "skipped"
        assert action.run.call_count == 2
        kp._document_store._try_runtime_recover.assert_called_once_with()
        assert "Background task summarize for d1 hit malformed note database; triggering runtime recovery" in caplog.text


# ---------------------------------------------------------------------------
# Action run() skip conditions
# ---------------------------------------------------------------------------

class TestAnalyzeSkipConditions:
    """Tests for analyze skip conditions."""
    def test_empty_content_raises(self):
        """Analyze raises when item has no content."""
        from keep.actions.analyze import Analyze
        ctx = MagicMock()
        ctx.get.return_value = MagicMock(content="", summary="")
        with pytest.raises(ValueError, match="content unavailable"):
            Analyze().run({"item_id": "d1"}, ctx)

    def test_no_parts_returns_empty(self):
        """Analyze returns empty parts when analyzer produces nothing."""
        from keep.actions.analyze import Analyze
        ctx = MagicMock()
        ctx.get.return_value = MagicMock(summary="some text content here")
        ctx.resolve_provider.return_value = MagicMock(analyze=MagicMock(return_value=[]))
        ctx.list_items.return_value = []
        result = Analyze().run({"item_id": "d1"}, ctx)
        assert result["parts"] == []
        assert "mutations" not in result


class TestAutoTagSkipConditions:
    """Tests for auto-tag skip conditions."""
    def test_empty_content_raises(self):
        """AutoTag raises when item has no content."""
        from keep.actions.auto_tag import AutoTag
        ctx = MagicMock()
        ctx.get.return_value = MagicMock(content="", summary="")
        with pytest.raises(ValueError, match="content unavailable"):
            AutoTag().run({"item_id": "d1"}, ctx)

    def test_no_tag_specs_returns_no_mutations(self):
        """AutoTag with no .tag/* specs returns no mutations."""
        from keep.actions.auto_tag import AutoTag
        ctx = MagicMock()
        ctx.get.return_value = MagicMock(content="hello world", summary="hello")
        ctx.list_items.return_value = []  # no tag specs
        ctx.resolve_provider.return_value = MagicMock()
        result = AutoTag().run({"item_id": "d1"}, ctx)
        assert "mutations" not in result or not result.get("mutations")


# ---------------------------------------------------------------------------
# _apply_mutations
# ---------------------------------------------------------------------------

class TestApplyMutations:
    """Tests for mutation application."""
    def test_set_summary_without_embed(self):
        kp = MagicMock()
        output = {"mutations": [{"op": "set_summary", "target": "d1", "summary": "hello"}]}
        _apply_mutations(kp, "coll", output)
        kp._apply_summary_mutation.assert_called_once_with(
            "coll",
            "d1",
            "hello",
            intent="derived_summary_replace",
            embed=False,
            content_hash="",
            content_hash_full="",
        )

    def test_set_summary_with_embed(self):
        kp = MagicMock()
        output = {"mutations": [{"op": "set_summary", "target": "d1", "summary": "hello", "embed": True}]}
        _apply_mutations(kp, "coll", output)
        kp._apply_summary_mutation.assert_called_once_with(
            "coll",
            "d1",
            "hello",
            intent="derived_summary_replace",
            embed=True,
            content_hash="",
            content_hash_full="",
        )

    def test_put_item_part_uses_part_storage(self):
        kp = MagicMock()
        output = {"mutations": [
            {"op": "put_item", "id": "d1@p1", "summary": "s",
             "tags": {"_base_id": "d1"}, "queue_background_tasks": False},
        ]}
        _apply_mutations(kp, "coll", output)
        # Part IDs go through part-specific storage, not _put_direct
        kp._put_direct.assert_not_called()
        kp._document_store.upsert_single_part.assert_called_once()
        kp._store.upsert_part.assert_called_once()

    def test_put_item_non_part_calls_put_direct(self):
        kp = MagicMock()
        output = {"mutations": [
            {"op": "put_item", "id": "child1", "content": "c", "summary": "s",
             "tags": {"topic": "test"}, "queue_background_tasks": False},
        ]}
        _apply_mutations(kp, "coll", output)
        kp._put_direct.assert_called_once_with(
            content="c", id="child1", summary="s",
            tags={"topic": "test"}, queue_background_tasks=False,
        )

    def test_stub_item_calls_stub_via_flow(self):
        kp = MagicMock()
        output = {"mutations": [
            {"op": "stub_item", "id": "child1", "content": "c", "summary": "s",
             "tags": {"_source": "link"}, "queue_background_tasks": True},
        ]}
        _apply_mutations(kp, "coll", output)
        kp._stub_via_flow.assert_called_once_with(
            id="child1",
            content="c",
            summary="s",
            tags={"_source": "link"},
            created_at=None,
            queue_background_tasks=True,
        )

    def test_set_tags(self):
        kp = MagicMock()
        output = {"mutations": [{"op": "set_tags", "target": "d1", "tags": {"topic": "AI"}}]}
        _apply_mutations(kp, "coll", output)
        kp._document_store.update_tags.assert_called_once_with("coll", "d1", {"topic": "AI"})
        kp._store.update_tags.assert_called_once()

    def test_delete_prefix(self):
        kp = MagicMock()
        output = {"mutations": [{"op": "delete_prefix", "prefix": "d1@p"}]}
        _apply_mutations(kp, "coll", output)
        kp._store.delete_parts.assert_called_once()
        kp._document_store.delete_parts.assert_called_once()

    def test_resolve_ref(self):
        output = {"summary": "resolved value"}
        assert _resolve_ref("$output.summary", output) == "resolved value"
        assert _resolve_ref("literal", output) == "literal"
        assert _resolve_ref("$output.missing", output) == "$output.missing"

    def test_no_mutations_is_noop(self):
        kp = MagicMock()
        _apply_mutations(kp, "coll", {})
        _apply_mutations(kp, "coll", {"mutations": None})
        assert kp.method_calls == []

    def test_set_summary_reembeds_when_requested_with_hashes(self):
        kp = MagicMock()
        output = {"mutations": [{
            "op": "set_summary", "target": "d1",
            "summary": "short",
            "embed": True,
            "content_hash": "abc123", "content_hash_full": "abc123full",
        }]}
        _apply_mutations(kp, "coll", output)
        kp._apply_summary_mutation.assert_called_once_with(
            "coll",
            "d1",
            "short",
            intent="derived_summary_replace",
            embed=True,
            content_hash="abc123",
            content_hash_full="abc123full",
        )

    def test_set_summary_passes_explicit_intent(self):
        kp = MagicMock()
        output = {"mutations": [{
            "op": "set_summary",
            "target": "d1",
            "summary": "hello",
            "intent": "derived_description_append",
        }]}
        _apply_mutations(kp, "coll", output)
        kp._apply_summary_mutation.assert_called_once_with(
            "coll",
            "d1",
            "hello",
            intent="derived_description_append",
            embed=False,
            content_hash="",
            content_hash_full="",
        )
