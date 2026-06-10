"""Tests for the flow-host primary interface.

Units 1 and 2 establish ``run_flow`` as the stable execution boundary,
with existing named state docs used where they already cover behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from keep.api import Keeper
from keep.config import StoreConfig
from keep.const import (
    STATE_COMPAT_FIND,
    STATE_COMPAT_GET_ITEM,
    STATE_DELETE,
    STATE_LIST,
    STATE_MOVE,
    STATE_PUT,
    STATE_TAG,
)
from keep.protocol import FlowHostProtocol
from keep.remote import RemoteKeeper
from keep.state_doc_runtime import FlowResult


@pytest.fixture
def kp(mock_providers, tmp_path: Path):
    keeper = Keeper(store_path=tmp_path)
    keeper._get_embedding_provider()
    return keeper


def test_keeper_implements_flow_host_protocol(kp):
    assert isinstance(kp, FlowHostProtocol)


def _roundtrip_flow_result(result: FlowResult) -> FlowResult:
    payload = json.loads(json.dumps({
        "status": result.status,
        "bindings": result.bindings,
        "data": result.data,
        "ticks": result.ticks,
        "history": result.history,
        "cursor": result.cursor,
    }))
    return FlowResult(
        status=payload.get("status", "error"),
        bindings=payload.get("bindings", {}),
        data=payload.get("data"),
        ticks=payload.get("ticks", 0),
        history=payload.get("history", []),
        cursor=payload.get("cursor"),
    )


def _wrap_run_flow_through_wire(
    host: Any,
    backend: Any,
    *,
    overrides: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    overrides = overrides or {}

    def wrapped(state, **kwargs):
        calls.append((state, kwargs))
        if state in overrides:
            return _roundtrip_flow_result(overrides[state](state, **kwargs))
        return _roundtrip_flow_result(backend(state, **kwargs))

    host.run_flow = wrapped  # type: ignore[method-assign]
    return calls


def _assert_last_call(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    state: str,
    params: dict[str, Any],
    writable: bool,
    has_state_doc_yaml: bool,
) -> None:
    actual_state, kwargs = calls[-1]
    assert actual_state == state
    assert kwargs["params"] == params
    assert kwargs["writable"] is writable
    if has_state_doc_yaml:
        assert isinstance(kwargs["state_doc_yaml"], str)
        assert kwargs["state_doc_yaml"]
    else:
        assert kwargs["state_doc_yaml"] is None


def _assert_public_flow_roundtrip(client, calls: list[tuple[str, dict[str, Any]]]) -> None:
    item = client.put("Round-trip put", id="rt-1", tags={"topic": "flows"}, force=True)
    assert item.id == "rt-1"
    assert item.tags.get("topic") == "flows"
    assert item.changed is True
    _assert_last_call(
        calls,
        state=STATE_PUT,
        params={
            "content": "Round-trip put",
            "uri": None,
            "id": "rt-1",
            "summary": None,
            "tags": {"topic": "flows"},
            "created_at": None,
            "force": True,
        },
        writable=True,
        has_state_doc_yaml=False,
    )

    fetched = client.get("rt-1")
    assert fetched is not None
    assert fetched.id == "rt-1"
    _assert_last_call(
        calls,
        state=STATE_COMPAT_GET_ITEM,
        params={"id": "rt-1"},
        writable=False,
        has_state_doc_yaml=True,
    )

    results = client.find(query="Round-trip", limit=5, include_hidden=True)
    assert any(result.id == "rt-1" for result in results)
    _assert_last_call(
        calls,
        state=STATE_COMPAT_FIND,
        params={
            "query": "Round-trip",
            "tags": None,
            "similar_to": None,
            "stored_only": False,
            "limit": 5,
            "since": None,
            "until": None,
            "include_self": False,
            "include_hidden": True,
            "scope": None,
            "deep": False,
            "deep_follow_depth": 10,
        },
        writable=False,
        has_state_doc_yaml=True,
    )

    tagged = client.tag("rt-1", {"status": "open"})
    assert tagged is not None
    assert tagged.tags.get("status") == "open"
    assert [state for state, _ in calls[-2:]] == [STATE_TAG, STATE_COMPAT_GET_ITEM]

    listed = client.find(tags={"status": "open"}, limit=5)
    assert any(result.id == "rt-1" for result in listed)
    _assert_last_call(
        calls,
        state=STATE_LIST,
        params={
            "prefix": None,
            "tags": {"status": "open"},
            "tag_keys": None,
            "since": None,
            "until": None,
            "order_by": "updated",
            "include_hidden": False,
            "limit": 5,
        },
        writable=False,
        has_state_doc_yaml=False,
    )

    scoped_now = client.set_now("Scoped context", scope="alice", tags={"project": "keep"})
    assert scoped_now.id == "now:alice"
    assert scoped_now.tags.get("user") == "alice"
    assert scoped_now.tags.get("project") == "keep"
    _assert_last_call(
        calls,
        state=STATE_PUT,
        params={
            "content": "Scoped context",
            "uri": None,
            "id": "now:alice",
            "summary": None,
            "tags": {"project": "keep", "user": "alice"},
            "created_at": None,
            "force": False,
        },
        writable=True,
        has_state_doc_yaml=False,
    )

    now_item = client.get_now(scope="alice")
    assert now_item.id == "now:alice"
    _assert_last_call(
        calls,
        state=STATE_COMPAT_GET_ITEM,
        params={"id": "now:alice"},
        writable=False,
        has_state_doc_yaml=True,
    )

    moved = client.move("rt-moved", source_id="now:alice", tags={"kind": "scratch"}, only_current=True)
    assert moved.id == "rt-moved"
    _assert_last_call(
        calls,
        state=STATE_MOVE,
        params={
            "name": "rt-moved",
            "source": "now:alice",
            "tags": {"kind": "scratch"},
            "only_current": True,
        },
        writable=True,
        has_state_doc_yaml=False,
    )

    deleted = client.delete("rt-1", delete_versions=False)
    assert deleted is True
    _assert_last_call(
        calls,
        state=STATE_DELETE,
        params={"id": "rt-1", "delete_versions": False},
        writable=True,
        has_state_doc_yaml=False,
    )


def test_keeper_public_memory_methods_delegate_via_run_flow(kp):
    kp.ensure_sysdocs()
    calls: list[str] = []
    original = kp.run_flow

    def tracking(state, **kwargs):
        calls.append(state)
        return original(state, **kwargs)

    kp.run_flow = tracking  # type: ignore[method-assign]

    item = kp.put("Flow-host put", id="fh-1", tags={"topic": "flows"})
    assert item.id == "fh-1"
    assert calls == [STATE_PUT]

    calls.clear()
    fetched = kp.get("fh-1")
    assert fetched is not None
    assert fetched.id == "fh-1"
    assert calls == ["compat-get-item"]

    calls.clear()
    results = kp.find(query="Flow-host")
    assert any(item.id == "fh-1" for item in results)
    assert calls == ["compat-find"]

    calls.clear()
    tagged = kp.tag("fh-1", {"status": "open"})
    assert tagged is not None
    assert tagged.tags.get("status") == "open"
    assert calls == [STATE_TAG, "compat-get-item"]

    calls.clear()
    now_item = kp.get_now()
    assert now_item.id == "now"
    assert calls == ["compat-get-item"]

    calls.clear()
    deleted = kp.delete("fh-1")
    assert deleted is True
    assert calls == [STATE_DELETE]


def test_keeper_public_memory_methods_roundtrip_serialized_flow_result(kp):
    kp.ensure_sysdocs()
    original = kp.run_flow
    calls = _wrap_run_flow_through_wire(
        kp,
        original,
        overrides={
            STATE_MOVE: lambda _state, **_kwargs: FlowResult(
                status="done",
                bindings={"moved": {"id": "rt-moved", "summary": "Moved item", "tags": {"kind": "scratch"}}},
                ticks=1,
            ),
        },
    )
    _assert_public_flow_roundtrip(kp, calls)


def test_named_writable_flows_apply_effects(kp):
    result = kp.run_flow(STATE_PUT, params={"content": "via flow", "id": "flow-note"})
    assert result.status == "done"
    assert kp.get("flow-note") is not None

    result = kp.run_flow(STATE_TAG, params={"id": "flow-note", "tags": {"status": "open"}})
    assert result.status == "done"
    tagged = kp.get("flow-note")
    assert tagged is not None
    assert tagged.tags.get("status") == "open"

    result = kp.run_flow(STATE_DELETE, params={"id": "flow-note"})
    assert result.status == "done"
    assert kp.get("flow-note") is None


def test_remote_keeper_implements_flow_host_protocol(tmp_path: Path):
    client = RemoteKeeper(
        api_url="http://127.0.0.1:1",
        api_key="",
        config=StoreConfig(path=tmp_path),
    )
    try:
        assert isinstance(client, FlowHostProtocol)
    finally:
        client.close()


def test_remote_keeper_public_memory_methods_delegate_via_run_flow(tmp_path: Path):
    client = RemoteKeeper(
        api_url="http://127.0.0.1:1",
        api_key="",
        config=StoreConfig(path=tmp_path),
    )
    calls: list[str] = []

    def fake_run_flow(state, **kwargs):
        calls.append(state)
        if state == STATE_COMPAT_GET_ITEM:
            item_id = kwargs.get("params", {}).get("id", "fh-remote")
            if item_id == "missing":
                return FlowResult(status="done", data={"item": None}, ticks=1)
            return FlowResult(
                status="done",
                data={"item": {"id": item_id, "summary": "remote item", "tags": {"kind": "test"}}},
                ticks=1,
            )
        if state == STATE_PUT:
            params = kwargs.get("params", {})
            return FlowResult(
                status="done",
                bindings={"stored": {"id": params.get("id", "fh-remote"), "summary": params.get("content", ""), "tags": params.get("tags", {}) or {}}},
                ticks=1,
            )
        if state == STATE_COMPAT_FIND:
            return FlowResult(
                status="done",
                data={
                    "items": [
                        {"id": "fh-remote", "summary": "remote item", "tags": {"kind": "test"}, "score": 0.9},
                    ],
                    "deep_groups": {},
                },
                ticks=1,
            )
        if state == STATE_TAG:
            return FlowResult(status="done", bindings={"tagged": {"count": 1, "ids": ["fh-remote"]}}, ticks=1)
        if state == STATE_DELETE:
            return FlowResult(status="done", data={"deleted": True}, ticks=1)
        raise AssertionError(f"unexpected flow state: {state}")

    client.run_flow = fake_run_flow  # type: ignore[method-assign]
    try:
        item = client.put(content="remote text", id="fh-remote", tags={"topic": "flows"})
        assert item.id == "fh-remote"
        assert calls == [STATE_PUT]

        calls.clear()
        fetched = client.get("fh-remote")
        assert fetched is not None
        assert fetched.id == "fh-remote"
        assert calls == ["compat-get-item"]

        calls.clear()
        assert client.get("missing") is None
        assert calls == ["compat-get-item"]

        calls.clear()
        results = client.find(query="remote")
        assert [item.id for item in results] == ["fh-remote"]
        assert calls == ["compat-find"]

        calls.clear()
        tagged = client.tag("fh-remote", {"status": "open"})
        assert tagged is not None
        assert tagged.tags == {"kind": "test"}
        assert calls == [STATE_TAG, "compat-get-item"]

        calls.clear()
        now_item = client.get_now()
        assert now_item.id == "now"
        assert calls == ["compat-get-item"]

        calls.clear()
        assert client.delete("fh-remote") is True
        assert calls == [STATE_DELETE]
    finally:
        client.close()


def test_remote_keeper_public_memory_methods_roundtrip_serialized_flow_result(
    mock_providers,
    tmp_path: Path,
):
    backend = Keeper(store_path=tmp_path / "backend")
    backend._get_embedding_provider()
    client = RemoteKeeper(
        api_url="http://127.0.0.1:1",
        api_key="",
        config=StoreConfig(path=tmp_path / "remote"),
    )
    try:
        calls = _wrap_run_flow_through_wire(
            client,
            backend.run_flow,
            overrides={
                STATE_MOVE: lambda _state, **_kwargs: FlowResult(
                    status="done",
                    bindings={"moved": {"id": "rt-moved", "summary": "Moved item", "tags": {"kind": "scratch"}}},
                    ticks=1,
                ),
            },
        )
        _assert_public_flow_roundtrip(client, calls)
    finally:
        client.close()
        backend.close()
