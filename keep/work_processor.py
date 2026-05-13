"""Work processor — claims and executes queued work items.

Replaces FlowEngine's process_requested_work_batch with direct
task_workflows dispatch.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from .compute_context import attribution, counter_scope
from .protocol import WorkQueueProtocol
from .tracing import get_tracer

if TYPE_CHECKING:
    from .api import Keeper

logger = logging.getLogger(__name__)


def _queue_wait_ms(created_at: str | None) -> float | None:
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - dt).total_seconds() * 1000.0, 0.0)
    except Exception:
        return None


def process_work_batch(
    keeper: "Keeper",
    queue: WorkQueueProtocol,
    *,
    limit: int = 10,
    worker_id: Optional[str] = None,
    lease_seconds: int = 120,
    shutdown_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Claim and execute a batch of work items.

    Returns a summary dict compatible with the old FlowEngine interface.
    """
    worker = worker_id or f"local-daemon:{os.getpid()}"
    items = queue.claim(worker, limit=limit, lease_seconds=lease_seconds)

    stats: dict[str, int] = {"claimed": len(items), "processed": 0, "failed": 0, "dead_lettered": 0}
    errors: list[dict[str, str]] = []

    for item in items:
        if shutdown_check and shutdown_check():
            logger.info("Shutdown requested, stopping work batch")
            break

        # Skip if superseded by a newer item
        if (
            item.supersede_key
            and item.created_at
            and queue.has_superseding(item.work_id, item.supersede_key, item.created_at)
        ):
            queue.complete(item.work_id, {"status": "superseded"})
            stats["processed"] += 1
            continue

        target = item.input.get("item_id") or item.input.get("id") or "?"
        try:
            wait_ms = _queue_wait_ms(item.created_at)
            logger.info("Processing %s %s", item.kind, target)
            with get_tracer("work_queue").start_as_current_span(
                "work_item.execute",
                attributes={
                    "work_id": item.work_id,
                    "kind": item.kind,
                    "item_id": str(target),
                    "queue.priority": item.priority,
                },
            ) as span, attribution(
                work_id=item.work_id,
                item_id=str(target),
                kind=item.kind,
            ), counter_scope(label=f"work:{item.kind}") as counters:
                if wait_ms is not None:
                    span.set_attribute("queue.wait_ms", round(wait_ms, 3))
                outcome = _execute_work_item(
                    keeper,
                    item.kind,
                    item.input,
                    shutdown_check=shutdown_check,
                )
                # Merge compute counters into the outcome so they're recorded
                # in the work-queue result_json for later attribution queries.
                # Done inside the scope so an exception during _execute_work_item
                # leaves `outcome` unbound and falls through to the except below.
                if outcome is None:
                    outcome = {}
                if not counters.is_empty():
                    outcome = dict(outcome)
                    outcome["compute"] = counters.to_dict()
            status = outcome.get("status", "applied")
            details = outcome.get("details")
            if item.kind == "flow" and status == "stopped" and outcome.get("cursor"):
                resumed_input = dict(item.input)
                resumed_input["cursor"] = outcome["cursor"]
                resumed_input["state"] = outcome.get("state") or resumed_input.get("state", "")
                queue.enqueue(
                    "flow",
                    resumed_input,
                    supersede_key=item.supersede_key,
                    priority=item.priority,
                )
                queue.complete(
                    item.work_id,
                    {"status": "rescheduled", "cursor": outcome["cursor"]},
                )
                logger.info("Paused flow %s for resume", target)
                stats["processed"] += 1
                continue
            flow_retry_reason = _flow_retry_reason(details)
            if item.kind == "flow" and flow_retry_reason:
                # State-doc evaluation reports user/action failures as a
                # FlowResult instead of throwing. Convert only transient
                # provider outages back into queue failure semantics; permanent
                # validation/action errors should complete with their details.
                raise RuntimeError(flow_retry_reason)
            queue.complete(item.work_id, outcome)
            if status == "skipped":
                logger.info("Skipped %s %s: %s", item.kind, target, details or "")
            else:
                logger.info("Done %s %s", item.kind, target)
            stats["processed"] += 1
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.warning("Failed %s %s: %s", item.kind, target, msg)
            queue.fail(item.work_id, worker, msg)
            stats["failed"] += 1
            errors.append({"work_id": item.work_id, "error": msg})

    stats["errors"] = errors  # type: ignore[assignment]

    # Log queue depth after each batch with actual work
    if stats["processed"] or stats["failed"]:
        remaining = queue.count()
        if remaining > 0:
            by_kind = queue.count_by_kind()  # pre-sorted by priority
            parts = [f"{v} {k}" for k, v in by_kind.items()]
            logger.info("Queue: %d remaining (%s)", remaining, ", ".join(parts))

    return stats


def _flow_retry_reason(details: Any) -> str:
    """Extract a retryable provider-outage reason from a daemon flow outcome."""
    if not isinstance(details, dict):
        return ""
    data = details.get("data")
    if not isinstance(data, dict):
        return ""
    direct = str(data.get("reason") or data.get("error") or "").strip()
    if direct and _is_retryable_provider_error(direct):
        return direct
    for rule_id, value in data.items():
        if not isinstance(value, dict):
            continue
        error = str(value.get("error") or "").strip()
        if error and _is_retryable_provider_error(error):
            return f"{rule_id}: {error}"
    return ""


def _is_retryable_provider_error(message: str) -> bool:
    """Return whether an action error looks like a transient provider outage."""
    text = str(message or "").lower()
    if not text:
        return False
    provider_marker = (
        "provider" in text
        or "ollama" in text
        or "openai" in text
        or "voyage" in text
        or "gemini" in text
        or "mistral" in text
    )
    transient_marker = any(
        marker in text for marker in (
            "cannot reach",
            "connection refused",
            "connection reset",
            "connection timed out",
            "connecterror",
            "read timed out",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "rate limit",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        )
    )
    return provider_marker and transient_marker


def _execute_work_item(
    keeper: "Keeper",
    kind: str,
    input_data: dict[str, Any],
    *,
    shutdown_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute a single work item.

    Handles two kinds:
    - ``"flow"``: resume a state-doc flow from a cursor (daemon context).
    - anything else: single action via ``keeper._run_local_task_workflow``.

    Returns the outcome dict (keys: status, details).
    """
    # Special-case: git ingest is directory-level, not item-scoped
    if kind == "ingest_git":
        from .git_ingest import ingest_git_history
        directory = input_data.get("directory")
        if not directory:
            raise ValueError("ingest_git requires 'directory' in input")
        result = ingest_git_history(keeper, Path(directory))
        return {"status": "applied", "details": result}

    # Flow resume: re-run the flow in daemon context (foreground=False)
    if kind == "flow":
        return _execute_flow_item(keeper, input_data, shutdown_check=shutdown_check)

    task_type = input_data.get("task_type") or kind
    item_id = str(input_data.get("item_id") or input_data.get("id") or "").strip()
    if not item_id:
        raise ValueError(f"work item missing item_id (kind={kind})")

    collection = str(
        input_data.get("collection")
        or input_data.get("doc_coll")
        or keeper._resolve_doc_collection()
    )
    content = str(input_data.get("content") or "")
    metadata = input_data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    outcome = keeper._run_local_task_workflow(
        task_type=task_type,
        item_id=item_id,
        collection=collection,
        content=content,
        metadata=metadata,
    )
    return outcome or {"status": "applied"}


def _execute_flow_item(
    keeper: "Keeper",
    input_data: dict[str, Any],
    *,
    shutdown_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run a state-doc flow in daemon context.

    Supports both fresh starts (no cursor) and cursor-based resume.
    The flow runs with ``foreground=False`` so async actions execute
    inline.  If the flow produces mutations (via put, tag, etc.), they
    happen through the normal action context.
    """
    from .flow_env import LocalFlowEnvironment
    from .state_doc_runtime import (
        decode_cursor,
        make_action_runner,
        make_state_doc_loader,
        run_flow,
    )

    state_name = str(input_data.get("state") or "").strip()
    params = input_data.get("params") or {}
    budget = int(input_data.get("budget", 10))

    # Decode cursor if resuming a stopped flow
    cursor = None
    cursor_token = input_data.get("cursor")
    if cursor_token:
        cursor = decode_cursor(str(cursor_token))
        if cursor is None:
            raise ValueError("flow work item has invalid cursor")
        if not state_name:
            state_name = cursor.state

    if not state_name:
        raise ValueError("flow work item missing state name")

    # Daemon flow work can run during startup before deferred maintenance has
    # repaired bundled state docs.  Match the public flow path and make the
    # store-backed loader see the current .state/* docs before evaluation.
    keeper.ensure_sysdocs()

    # Extract item context for item-scoped actions (summarize, analyze, etc.)
    item_id = str(input_data.get("item_id") or params.get("item_id") or "").strip() or None
    text = input_data.get("content")

    env = LocalFlowEnvironment(keeper)
    loader = make_state_doc_loader(env)
    base_runner = make_action_runner(
        env, writable=True, item_id=item_id, item_text=text,
    )

    # Wrap the runner to apply mutations (set_summary, put_item, etc.)
    # after each action — the flow runtime only captures output as
    # bindings; mutations need explicit application.
    collection = keeper._resolve_doc_collection()

    def _mutation_runner(action_name: str, action_params: dict[str, Any]) -> dict[str, Any]:
        output = base_runner(action_name, action_params)
        if isinstance(output, dict) and output.get("mutations"):
            from .task_workflows import _apply_mutations
            _apply_mutations(keeper, collection, output)
        return output

    result = run_flow(
        state_name,
        params,
        budget=budget,
        load_state_doc=loader,
        run_action=_mutation_runner,
        cursor=cursor,
        foreground=False,  # daemon context — execute async actions inline
        should_stop=shutdown_check,
    )

    return {
        "status": result.status,
        "cursor": result.cursor,
        "state": state_name,
        "details": {
            "ticks": result.ticks,
            "history": result.history,
            "data": result.data,
        },
    }
