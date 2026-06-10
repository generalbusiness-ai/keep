"""Integration tests for RemoteKeeper against an in-process fake daemon server.

These tests construct a *real* RemoteKeeper (no MagicMock on _client) and fire
live HTTP requests at a ThreadingHTTPServer bound to an ephemeral port.  The
fake server speaks the daemon's /v1/flow contract as derived from
keep/daemon_server.py and keep/flow_client.py.

Design notes
------------
* The server records every request so tests can assert on what was sent.
* The fake /v1/flow handler branches on the ``state`` field and returns a
  FlowResult-shaped body (status/bindings/data/ticks/history/cursor) whose
  bindings/data keys satisfy the flow_client.py readers exactly.
* The server is bound to 127.0.0.1:0 (ephemeral) and runs in a daemon thread;
  teardown calls server.shutdown() + thread.join() so no socket or thread leaks.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from keep.config import StoreConfig
from keep.remote import RemoteKeeper

# ---------------------------------------------------------------------------
# Fake server
# ---------------------------------------------------------------------------

# The /v1/ready capabilities dict mirrors keep/daemon_server.py _daemon_status()
_FAKE_CAPABILITIES: dict[str, Any] = {
    "api_version": 1,
    "export_snapshot": True,
    "export_stream_ndjson": True,
    "export_bundle": True,
    "export_changes": True,
    "remote_incremental_markdown_sync": True,
}

# Canned items used by the fake server responses
_CANNED_ITEM_PUT = {
    "id": "test-put-item",
    "summary": "A stored item",
    "tags": {"category": "test"},
}

_CANNED_ITEM_GET = {
    "id": "test-get-item",
    "summary": "A fetched item",
    "tags": {"fetched": "yes"},
}

_CANNED_ITEM_FIND_1 = {
    "id": "find-result-1",
    "summary": "First find result",
    "tags": {"k": "v1"},
}

_CANNED_ITEM_FIND_2 = {
    "id": "find-result-2",
    "summary": "Second find result",
    "tags": {"k": "v2"},
}

_CANNED_ITEM_TAG = {
    "id": "test-tag-item",
    "summary": "A tagged item",
    "tags": {"priority": "high"},
}


def _flow_response_for_state(state: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return a FlowResult-shaped response body for the given flow state.

    The shape is derived from keep/daemon_server.py _handle_flow() and the
    flow_client.py reader for each state.  Specifically:

    * put (STATE_PUT="put"):
        flow_client.put_item reads bindings["stored"] for an item dict with "id".
    * compat-get-item (STATE_COMPAT_GET_ITEM):
        flow_client.get_item reads data["item"] for an item dict (or None if not found).
    * list (STATE_LIST="list"):
        flow_client.find_items (no-query path) reads bindings["results"]["results"]
        for a list of item dicts.
    * compat-find (STATE_COMPAT_FIND):
        flow_client.find_items (query path) reads data["items"] for a list of item dicts.
    * delete (STATE_DELETE="delete"):
        flow_client.delete_item reads bindings["result"]["deleted"] for a bool.
    * tag (STATE_TAG="tag"):
        flow_client.tag_item reads bindings["tagged"] (dict or error); then calls
        get_item internally, so the fake must also handle the follow-up get.
    """
    # Base: all responses are "done" with no ticks/history/cursor unless overridden
    base: dict[str, Any] = {
        "status": "done",
        "bindings": {},
        "data": None,
        "ticks": 1,
        "history": [state],
        "cursor": None,
    }

    if state == "put":
        # put_item reads: bindings["stored"] with an id
        base["bindings"] = {"stored": dict(_CANNED_ITEM_PUT)}
        return base

    if state == "compat-get-item":
        # get_item reads: data["item"] (None means not found)
        item_id = (params or {}).get("id", "")
        if item_id == "not-found-id":
            base["data"] = {"item": None}
        else:
            base["data"] = {"item": dict(_CANNED_ITEM_GET)}
        return base

    if state == "list":
        # find_items (no-query) reads: bindings["results"]["results"] as list
        base["bindings"] = {
            "results": {
                "results": [dict(_CANNED_ITEM_FIND_1), dict(_CANNED_ITEM_FIND_2)],
            }
        }
        return base

    if state == "compat-find":
        # find_items (with query) reads: data["items"] as list
        base["data"] = {
            "items": [dict(_CANNED_ITEM_FIND_1), dict(_CANNED_ITEM_FIND_2)],
            "deep_groups": {},
        }
        return base

    if state == "delete":
        # delete_item reads: bindings["result"]["deleted"] as bool
        base["bindings"] = {"result": {"deleted": True}}
        return base

    if state == "tag":
        # tag_item reads bindings["tagged"] (no error = success), then calls
        # get_item with the same id as a follow-up flow request.
        base["bindings"] = {"tagged": {"id": (params or {}).get("id", "tagged-id")}}
        return base

    if state == "fake-error":
        # Used by the 500-propagation test; the handler returns 500 for this state.
        # We never reach here — the handler short-circuits before calling this function.
        pass

    # Unknown state — return done with empty payload so the server doesn't hang
    return base


class _FakeHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that records requests and returns canned responses.

    The handler routes by (method, path):
      GET  /v1/ready   → daemon status with capabilities
      POST /v1/flow    → FlowResult-shaped body keyed on request["state"]
      GET  /v1/export  → ndjson stream of two item rows
      *    *           → 404

    For 500-triggering tests, the server checks if the ``X-Fake-Status`` request
    header is set and overrides the response status accordingly.
    """

    # Silence the default BaseHTTPRequestHandler access-log noise.
    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
        pass

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _send_json(self, status: int, body: Any) -> None:
        encoded = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _record(self, method: str, body: dict[str, Any]) -> None:
        """Append request metadata to the server's request log."""
        qs = urlparse(self.path).query
        params_qs = parse_qs(qs, keep_blank_values=False)
        self.server.requests.append({  # type: ignore[attr-defined]
            "method": method,
            "path": self.path.split("?", 1)[0],
            "body": body,
            "query": params_qs,
        })

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        # Honour the override header so error-propagation tests can inject a 500.
        override_status = self.headers.get("X-Fake-Status")
        self._record("GET", {})

        if override_status:
            body = {"error": "injected error", "request_id": "req-fake-500"}
            self._send_json(int(override_status), body)
            return

        if path == "/v1/ready":
            self._send_json(200, {
                "status": "ok",
                "pid": 0,
                "version": "0.0.0-fake",
                "store": "/fake/store",
                "embedding": "fake",
                "summarization": "fake",
                "needs_setup": False,
                "warnings": [],
                "capabilities": _FAKE_CAPABILITIES,
                "network": {"mode": "local", "bind_host": "127.0.0.1", "advertised_url": ""},
            })
            return

        if path == "/v1/export":
            # Emit two ndjson rows so export_iter can be exercised.
            rows = [
                {"id": "export-item-1", "summary": "First export item", "tags": {}},
                {"id": "export-item-2", "summary": "Second export item", "tags": {}},
            ]
            encoded_lines = (json.dumps(r) + "\n" for r in rows)
            body = "".join(encoded_lines).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        body = self._read_body()
        self._record("POST", body)

        if path == "/v1/flow":
            state = body.get("state", "")
            params = body.get("params") or {}
            # Any flow request with state "fake-error" returns a 500 so that
            # error-propagation tests can exercise _post → _raise_for_status.
            if state == "fake-error":
                err_body = {"error": "injected flow error", "request_id": "req-flow-500"}
                self._send_json(500, err_body)
                return
            resp = _flow_response_for_state(state, params)
            self._send_json(200, resp)
            return

        self._send_json(404, {"error": "not found"})


class FakeDaemonServer:
    """Lifecycle wrapper around a ThreadingHTTPServer bound to an ephemeral port.

    Usage::

        server = FakeDaemonServer()
        server.start()
        # ... make requests to server.base_url ...
        server.stop()   # always call in finally/teardown

    Or use the ``fake_server`` pytest fixture below.
    """

    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.requests: list[dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        assert self._server is not None, "server not started"
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def start(self) -> "FakeDaemonServer":
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
        # Share the request log so _FakeHandler can append to it.
        srv.requests = self.requests  # type: ignore[attr-defined]
        self._server = srv
        self._thread = threading.Thread(target=srv.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            assert not self._thread.is_alive(), "server thread did not stop within 5 s"
        self._server = None
        self._thread = None

    def last_flow_body(self) -> dict[str, Any]:
        """Return the body of the most-recent POST /v1/flow request."""
        for req in reversed(self.requests):
            if req["method"] == "POST" and req["path"] == "/v1/flow":
                return req["body"]
        return {}

    def flow_requests(self) -> list[dict[str, Any]]:
        """Return all recorded POST /v1/flow requests in order."""
        return [r for r in self.requests if r["method"] == "POST" and r["path"] == "/v1/flow"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_server():
    """Start a fake daemon server for one test; shut it down on teardown."""
    srv = FakeDaemonServer().start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture()
def keeper(fake_server, tmp_path):
    """Real RemoteKeeper pointed at the fake server; closed on teardown."""
    config = StoreConfig(path=tmp_path, config_dir=tmp_path)
    rk = RemoteKeeper(fake_server.base_url, "kn_test_fake", config)
    try:
        yield rk
    finally:
        rk.close()


# ---------------------------------------------------------------------------
# Step 1: server smoke test
# ---------------------------------------------------------------------------


def test_fake_server_starts_and_stops(tmp_path):
    """The fake server must start, respond, and stop without hanging."""
    srv = FakeDaemonServer().start()
    try:
        resp = httpx.get(f"{srv.base_url}/v1/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
    finally:
        srv.stop()
    # Verify the thread joined — stop() already asserts this internally.
    assert srv._thread is None


# ---------------------------------------------------------------------------
# Step 2: CRUD round-trips through the real httpx client
# ---------------------------------------------------------------------------


def test_put_round_trip(keeper, fake_server):
    """keeper.put() sends POST /v1/flow with state='put' and returns an Item."""
    item = keeper.put(
        content="hello world",
        id="test-put-item",
        tags={"category": "test"},
    )

    # The returned Item must match the canned response
    assert item.id == "test-put-item"
    assert item.summary == "A stored item"
    assert item.tags.get("category") == "test"

    # Assert what was sent to the server
    body = fake_server.last_flow_body()
    assert body["state"] == "put"
    params = body.get("params") or {}
    assert params.get("content") == "hello world"
    assert params.get("id") == "test-put-item"


def test_get_round_trip(keeper, fake_server):
    """keeper.get(id) sends POST /v1/flow with state='compat-get-item' and returns Item."""
    item = keeper.get("test-get-item")

    assert item is not None
    assert item.id == "test-get-item"
    assert item.summary == "A fetched item"

    body = fake_server.last_flow_body()
    assert body["state"] == "compat-get-item"
    params = body.get("params") or {}
    assert params.get("id") == "test-get-item"


def test_get_not_found_returns_none(keeper, fake_server):
    """keeper.get() returns None when the flow data['item'] is None."""
    result = keeper.get("not-found-id")
    assert result is None

    body = fake_server.last_flow_body()
    assert body["state"] == "compat-get-item"
    params = body.get("params") or {}
    assert params.get("id") == "not-found-id"


def test_find_with_query_round_trip(keeper, fake_server):
    """keeper.find(query=...) uses compat-find state and returns list[Item]."""
    items = keeper.find(query="hello", limit=5)

    assert len(items) == 2
    assert items[0].id == "find-result-1"
    assert items[1].id == "find-result-2"

    body = fake_server.last_flow_body()
    assert body["state"] == "compat-find"
    params = body.get("params") or {}
    assert params.get("query") == "hello"
    assert params.get("limit") == 5


def test_find_without_query_uses_list_state(keeper, fake_server):
    """keeper.find() with no query/similar_to uses the 'list' state."""
    items = keeper.find(limit=10)

    assert len(items) == 2
    assert items[0].id == "find-result-1"

    body = fake_server.last_flow_body()
    assert body["state"] == "list"


def test_delete_round_trip(keeper, fake_server):
    """keeper.delete(id) sends POST /v1/flow with state='delete' and returns True."""
    result = keeper.delete("some-item")

    assert result is True

    body = fake_server.last_flow_body()
    assert body["state"] == "delete"
    params = body.get("params") or {}
    assert params.get("id") == "some-item"


def test_tag_round_trip(keeper, fake_server):
    """keeper.tag(id, tags) sends POST /v1/flow with state='tag'.

    tag_item in flow_client.py calls run_flow("tag") and then immediately calls
    get_item to return the updated item — so two flow requests are sent.
    """
    # Override the per-state get response to use the tag item id so the
    # follow-up get returns something sensible (the fake server uses the
    # params["id"] to look up the canned item).
    # We just need the final returned Item to be non-None; the tag follow-up
    # get will use "test-tag-item" which resolves to _CANNED_ITEM_GET by
    # default (the fake returns _CANNED_ITEM_GET for any non-"not-found-id").
    item = keeper.tag("test-tag-item", tags={"priority": "high"})

    assert item is not None

    # Confirm the tag flow was sent first
    flows = fake_server.flow_requests()
    assert len(flows) >= 2
    tag_flow = flows[-2]
    assert tag_flow["body"]["state"] == "tag"
    params = tag_flow["body"].get("params") or {}
    assert params.get("id") == "test-tag-item"
    assert params.get("tags") == {"priority": "high"}

    # Confirm the follow-up get was sent
    get_flow = flows[-1]
    assert get_flow["body"]["state"] == "compat-get-item"


def test_server_info_and_capabilities(keeper, fake_server):
    """server_info() and capabilities() read GET /v1/ready and expose capabilities."""
    info = keeper.server_info()
    assert info["status"] == "ok"
    assert isinstance(info["capabilities"], dict)

    caps = keeper.capabilities()
    assert caps["api_version"] == 1
    assert caps["export_snapshot"] is True
    assert caps["export_bundle"] is True

    assert keeper.supports_capability("export_snapshot") is True
    assert keeper.supports_capability("nonexistent_capability") is False

    # /v1/ready must have been called exactly once (second call uses cache)
    ready_requests = [r for r in fake_server.requests if r["path"] == "/v1/ready"]
    assert len(ready_requests) == 1


def test_direct_calls_are_logged(fake_server, tmp_path):
    """Non-CRUD remote calls (server_info, export_bundle) also hit the audit log.

    Regression guard: all remote HTTP traffic routes through RemoteKeeper._request,
    so direct calls — not just _get/_post — produce a keep-client.log line with
    request correlation. Previously server_info()/export_* bypassed the logger.
    """
    config = StoreConfig(path=tmp_path, config_dir=tmp_path)
    keeper = RemoteKeeper(fake_server.base_url, "kn_test_fake", config)
    try:
        keeper.server_info()          # GET /v1/ready — was bypassing the logger
        keeper.export_bundle("note-1")  # GET /v1/export/bundles/... — likewise
    finally:
        keeper.close()

    body = (tmp_path / "keep-client.log").read_text(encoding="utf-8")
    assert "remote: GET /v1/ready" in body
    assert "remote: GET /v1/export/bundles/note-1" in body


def test_export_iter_streams_ndjson(keeper, fake_server):
    """export_iter() reads GET /v1/export as ndjson and yields item dicts."""
    rows = list(keeper.export_iter())

    assert len(rows) == 2
    assert rows[0]["id"] == "export-item-1"
    assert rows[1]["id"] == "export-item-2"

    export_requests = [r for r in fake_server.requests if r["path"] == "/v1/export"]
    assert len(export_requests) == 1


# ---------------------------------------------------------------------------
# Step 3: error propagation and client-log audit trail
# ---------------------------------------------------------------------------


def test_500_with_request_id_propagates_through_real_client(fake_server, tmp_path):
    """A 500 response from POST /v1/flow raises HTTPStatusError containing request_id.

    The fake server returns 500 for state="fake-error".  This exercises the real
    _post → _raise_for_status → _daemon_error_detail path end-to-end through the
    live httpx client (not the mock-based path in test_remote.py).
    """
    config = StoreConfig(path=tmp_path, config_dir=tmp_path)
    keeper = RemoteKeeper(fake_server.base_url, "kn_test_fake", config)
    try:
        # Call _post directly so we can pass the sentinel state without going
        # through run_flow's FlowResult wrapping (which would swallow the error).
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            keeper._post("/v1/flow", {"state": "fake-error"})

        msg = str(excinfo.value)
        assert "injected flow error" in msg
        assert "request_id=req-flow-500" in msg
    finally:
        keeper.close()


def test_client_log_contains_audit_trail(fake_server, tmp_path):
    """After real HTTP round-trips, keep-client.log contains the audit records.

    Exercises the _log_call() path: after a PUT (POST /v1/flow) and a direct
    _get call through the live httpx client, the rotating log file must exist
    and contain the expected prefixes.

    All remote HTTP calls route through RemoteKeeper._request, so every call —
    CRUD helpers and direct ones like server_info() — emits a log line.  Here we
    exercise _get directly alongside _post (via put) and assert both appear.
    """
    config = StoreConfig(path=tmp_path, config_dir=tmp_path)
    keeper = RemoteKeeper(fake_server.base_url, "kn_test_fake", config)
    try:
        # _post via put → logged as "remote: POST /v1/flow ..."
        keeper.put(content="audit test", id="audit-item")
        # _get called directly → logged as "remote: GET /v1/ready ..."
        # (/v1/ready returns valid JSON so _get can parse the response)
        keeper._get("/v1/ready")
    finally:
        keeper.close()

    log_path = tmp_path / "keep-client.log"
    assert log_path.exists(), "keep-client.log was not created"
    body = log_path.read_text(encoding="utf-8")

    # Both calls must be logged with the correct prefix and host.
    assert "remote: POST /v1/flow" in body
    assert "remote: GET /v1/ready" in body
    assert f"host={fake_server.base_url}" in body


# ---------------------------------------------------------------------------
# Step 4: resource-leak guard (no leaked thread or socket after stop)
# ---------------------------------------------------------------------------


def test_server_thread_is_joined_after_stop():
    """FakeDaemonServer.stop() must fully join the server thread."""
    srv = FakeDaemonServer().start()
    thread = srv._thread
    assert thread is not None
    assert thread.is_alive()

    srv.stop()

    # stop() asserts thread.is_alive() == False internally; double-check here.
    assert not thread.is_alive()
    assert srv._thread is None
    assert srv._server is None
