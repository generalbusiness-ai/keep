"""Tests for the daemon HTTP query server.

Uses mock_providers to avoid loading real ML models.
Tests both raw HTTP and RemoteKeeper round-trip.
"""

import json
import os
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from keep.api import Keeper
from keep.daemon_server import DaemonServer, PutRequest, _daemon_json_default


@pytest.fixture
def daemon(mock_providers, tmp_path):
    """Start a DaemonServer on an OS-assigned port."""
    kp = Keeper(store_path=tmp_path)
    server = DaemonServer(kp, port=0)
    port = server.start()
    yield server, kp, port
    server.stop()
    kp.close()


@pytest.fixture
def http(daemon):
    """httpx.Client with base_url and auth token pre-configured."""
    server, _, port = daemon
    client = httpx.Client(
        base_url=f"http://127.0.0.1:{port}",
        headers={"Authorization": f"Bearer {server.auth_token}"},
        timeout=5,
    )
    yield client
    client.close()


# --- Health ---

def test_ready(http):
    r = http.get("/v1/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "pid" in body
    assert "version" in body
    assert "store" in body
    assert "needs_setup" in body
    assert "warnings" in body
    assert body["capabilities"]["export_snapshot"] is True
    assert body["capabilities"]["export_bundle"] is True
    assert body["capabilities"]["export_changes"] is True
    assert body["network"]["mode"] == "local"
    assert "item_count" not in body


def test_health(http):
    r = http.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "pid" in body
    assert "version" in body
    assert "store" in body
    assert "embedding" in body
    assert "needs_setup" in body
    assert "warnings" in body
    assert isinstance(body["warnings"], list)
    assert body["capabilities"]["remote_incremental_markdown_sync"] is True
    assert body["network"]["bind_host"] == "127.0.0.1"


def test_request_threads_release_document_store_connections(daemon, http):
    """Bursting daemon requests must not leak one SQLite handle per thread."""
    _, kp, _ = daemon
    baseline = kp._document_store.release_thread_connection_calls

    for i in range(20):
        r = http.post("/v1/notes", json={"content": f"burst note {i}", "id": f"burst-{i}"})
        assert r.status_code == 200

    assert kp._document_store.release_thread_connection_calls == baseline + 20


def test_export_endpoint(http):
    http.post("/v1/notes", json={"content": "user note", "id": "export-1"})
    http.post("/v1/notes", json={"content": "system note", "id": ".system-export"})

    r = http.get("/v1/export")
    assert r.status_code == 200
    data = r.json()
    assert data["format"] == "keep-export"
    ids = {doc["id"] for doc in data["documents"]}
    assert "export-1" in ids
    assert ".system-export" in ids

    r = http.get("/v1/export", params={"include_system": "false"})
    assert r.status_code == 200
    data = r.json()
    ids = {doc["id"] for doc in data["documents"]}
    assert "export-1" in ids
    assert ".system-export" not in ids


def test_export_endpoint_stream_ndjson(http):
    http.post("/v1/notes", json={"content": "user note", "id": "export-stream-1"})
    http.post("/v1/notes", json={"content": "system note", "id": ".system-export-stream"})

    with http.stream(
        "GET",
        "/v1/export",
        params={"include_system": "false", "stream": "ndjson"},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].split(";", 1)[0] == "application/x-ndjson"
        rows = [
            json.loads(line)
            for line in r.iter_lines()
            if line
        ]

    assert rows[0]["format"] == "keep-export"
    ids = {row["id"] for row in rows[1:]}
    assert "export-stream-1" in ids
    assert ".system-export-stream" not in ids


def test_export_bundle_endpoint(http):
    http.post("/v1/notes", json={"content": "person note", "id": "alice"})
    http.post("/v1/notes", json={
        "content": "Conversation note. " * 40,
        "id": "conv-1",
        "tags": {"speaker": "alice"},
    })
    with patch("keep.analyzers.SlidingWindowAnalyzer.analyze") as mock_analyze:
        mock_analyze.return_value = [
            {"summary": "Opening exchange", "tags": {"speaker": "alice"}},
            {"summary": "Follow-up exchange"},
        ]
        http.post(
            "/v1/analyze",
            json={"id": "conv-1", "foreground": True, "force": True},
        )

    r = http.get("/v1/export/bundles/conv-1")
    assert r.status_code == 200
    bundle = r.json()
    assert bundle["document"]["id"] == "conv-1"
    assert "parts" in bundle["document"]
    assert "speaker" in bundle["edge_tag_keys"]
    assert bundle["current_inverse"] == []

    r = http.get("/v1/export/bundles/alice")
    assert r.status_code == 200
    bundle = r.json()
    assert any(edge[0] == "said" for edge in bundle["current_inverse"])

    r = http.get("/v1/export/bundles/conv-1", params={"include_parts": "false"})
    assert r.status_code == 200
    bundle = r.json()
    assert "parts" not in bundle["document"]

    r = http.get("/v1/export/bundles/nonexistent")
    assert r.status_code == 404


def test_export_changes_endpoint(http):
    http.post("/v1/notes", json={"content": "first body", "id": "changes-1"})
    http.post("/v1/notes", json={"content": "updated body", "id": "changes-1"})

    r = http.get("/v1/export/changes", params={"cursor": "0", "limit": "500"})
    assert r.status_code == 200
    data = r.json()
    assert data["format"] == "keep-export-changes"
    assert data["version"] == 1
    assert data["compacted"] is False
    event = next(row for row in data["events"] if row["entity_id"] == "changes-1")
    assert event["affected_note_ids"] == ["changes-1"]
    cursor = data["cursor"]

    r = http.get("/v1/export/changes", params={"cursor": cursor, "limit": "500"})
    assert r.status_code == 200
    data = r.json()
    assert data["events"] == []

    r = http.get("/v1/export/changes", params={"cursor": "bad"})
    assert r.status_code == 400


def test_export_changes_endpoint_caps_limit(http, daemon):
    server, kp, port = daemon
    with patch.object(kp, "export_changes", wraps=kp.export_changes) as wrapped:
        r = httpx.get(
            f"http://127.0.0.1:{port}/v1/export/changes?cursor=0&limit=999999999",
            headers={"Authorization": f"Bearer {server.auth_token}"},
            timeout=5,
        )
    assert r.status_code == 200
    assert wrapped.call_args.kwargs["limit"] == 10_000


def test_export_changes_endpoint_includes_dependent_targets_for_doc_update(http):
    http.post("/v1/notes", json={"content": "Joanna note", "id": "Joanna"})
    http.post("/v1/notes", json={
        "content": "Session body",
        "id": "session-feed-1",
        "tags": {"speaker": "Joanna"},
    })
    http.post("/v1/notes", json={
        "content": "Session renamed",
        "id": "session-feed-1",
        "summary": "Session renamed",
        "tags": {"speaker": "Joanna"},
    })

    r = http.get("/v1/export/changes", params={"cursor": "0", "limit": "500"})
    assert r.status_code == 200
    data = r.json()
    event = next(
        row for row in data["events"]
        if row["entity_id"] == "session-feed-1" and row["mutation"] == "doc_update"
    )
    assert "session-feed-1" in event["affected_note_ids"]
    assert "Joanna" in event["affected_note_ids"]


def test_ready_avoids_expensive_count(daemon):
    server, kp, port = daemon
    original = kp.count

    def fail_count():
        raise RuntimeError("count should not run on readiness probe")

    kp.count = fail_count  # type: ignore[method-assign]
    try:
        r = httpx.get(
            f"http://127.0.0.1:{port}/v1/ready",
            headers={"Authorization": f"Bearer {server.auth_token}"},
            timeout=5,
        )
        assert r.status_code == 200
        assert "item_count" not in r.json()
    finally:
        kp.count = original  # type: ignore[method-assign]


def test_health_tolerates_count_failure(daemon):
    server, kp, port = daemon
    original = kp.count

    def fail_count():
        raise RuntimeError("count failed")

    kp.count = fail_count  # type: ignore[method-assign]
    try:
        r = httpx.get(
            f"http://127.0.0.1:{port}/v1/health",
            headers={"Authorization": f"Bearer {server.auth_token}"},
            timeout=5,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["item_count"] is None
        assert "item count unavailable" in body["warnings"]
    finally:
        kp.count = original  # type: ignore[method-assign]


def test_401_without_token(daemon):
    _, _, port = daemon
    r = httpx.get(f"http://127.0.0.1:{port}/v1/ready", timeout=5)
    assert r.status_code == 401


def test_404_unknown_path(http):
    r = http.get("/v1/nonexistent")
    assert r.status_code == 404


def test_local_daemon_rejects_non_loopback_host_header(daemon):
    server, _, port = daemon
    r = httpx.get(
        f"http://127.0.0.1:{port}/v1/ready",
        headers={
            "Authorization": f"Bearer {server.auth_token}",
            "Host": "keep.example.test",
        },
        timeout=5,
    )
    assert r.status_code == 403


def test_remote_mode_accepts_advertised_host_header(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path)
    server = DaemonServer(
        kp,
        port=0,
        bind_host="0.0.0.0",
        advertised_url="https://keep.example.test",
        trusted_proxy=True,
    )
    port = server.start()
    try:
        r = httpx.get(
            f"http://127.0.0.1:{port}/v1/ready",
            headers={
                "Authorization": f"Bearer {server.auth_token}",
                "Host": "keep.example.test",
            },
            timeout=5,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["network"]["mode"] == "remote"
        assert body["network"]["advertised_url"] == "https://keep.example.test"

        r = httpx.get(
            f"http://127.0.0.1:{port}/v1/ready",
            headers={
                "Authorization": f"Bearer {server.auth_token}",
                "Host": "evil.example.test",
            },
            timeout=5,
        )
        assert r.status_code == 403
    finally:
        server.stop()
        kp.close()


def test_remote_mode_requires_trusted_proxy(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path)
    try:
        server = DaemonServer(
            kp,
            port=0,
            bind_host="192.0.2.10",
        )
        with pytest.raises(ValueError, match="trusted proxy mode"):
            server.start()
    finally:
        kp.close()


def test_wildcard_bind_requires_advertised_url(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path)
    try:
        server = DaemonServer(
            kp,
            port=0,
            bind_host="0.0.0.0",
            trusted_proxy=True,
        )
        with pytest.raises(ValueError, match="advertised-url"):
            server.start()
    finally:
        kp.close()


def test_flow_blank_budget_uses_default(http):
    r = http.post("/v1/flow", json={
        "state": "prompt",
        "params": {"list": True},
        "budget": "",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert "prompts" in body["data"]


# --- Put / Get / Delete ---

def test_put_and_get(http):
    r = http.post("/v1/notes", json={
        "content": "test note", "id": "test-1", "tags": {"topic": "cache"},
    })
    assert r.status_code == 200
    assert r.json()["id"] == "test-1"

    r = http.get("/v1/notes/test-1")
    assert r.status_code == 200
    assert r.json()["id"] == "test-1"

    r = http.get("/v1/notes/nonexistent")
    assert r.status_code == 404


def test_put_surfaces_value_error_message(daemon, http):
    from unittest.mock import patch

    message = (
        "Failed to create embedding provider 'ollama': Cannot reach Ollama at "
        "http://localhost:11434. Is Ollama running? Start it with: ollama serve"
    )
    with patch("keep.daemon_server.flow_put_item", side_effect=ValueError(message)):
        r = http.post("/v1/notes", json={"content": "test note", "id": "test-1"})

    assert r.status_code == 400
    assert r.json()["error"] == message


def test_put_keeps_internal_server_error_for_unexpected_exceptions(daemon, http):
    from unittest.mock import patch

    with patch("keep.daemon_server.flow_put_item", side_effect=RuntimeError("boom")):
        r = http.post("/v1/notes", json={"content": "test note", "id": "test-1"})

    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "internal server error"
    assert isinstance(body["request_id"], str)
    assert body["request_id"]


def test_put_rejects_non_object_json_body(http):
    r = http.post("/v1/notes", content=b'["not", "an", "object"]')

    assert r.status_code == 400
    assert r.json()["error"] == "request body must be a JSON object"


def test_put_rejects_invalid_field_types(http):
    r = http.post("/v1/notes", json={
        "content": "test note",
        "id": "bad-types",
        "tags": ["not", "a", "map"],
    })

    assert r.status_code == 400
    assert "invalid request body" in r.json()["error"]


def test_find_rejects_invalid_limit_type(http):
    r = http.post("/v1/search", json={"query": "alpha", "limit": "five"})

    assert r.status_code == 400
    assert "invalid request body" in r.json()["error"]


def test_delete(http):
    http.post("/v1/notes", json={"content": "to delete", "id": "del-1"})
    r = http.delete("/v1/notes/del-1")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    r = http.get("/v1/notes/del-1")
    assert r.status_code == 404


# --- Directory watch via PUT /v1/notes ---

def test_put_directory_watch_registers_without_put(daemon, http, tmp_path):
    """PUT with watch_kind=directory skips flow_put_item and registers a watch.

    Regression: previously the daemon called flow_put_item on the directory
    URI first, which failed with "Not a file" and left the watch unregistered.
    """
    from keep.watches import list_watches

    server, kp, port = daemon
    d = tmp_path / "community"
    d.mkdir()
    (d / "a.md").write_text("hi")

    r = http.post("/v1/notes", json={
        "uri": f"file://{d}",
        "watch": True,
        "watch_kind": "directory",
        "recurse": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("watch", {}).get("source") == str(d)

    entries = list_watches(kp)
    assert len(entries) == 1
    assert entries[0].source == str(d)
    assert entries[0].kind == "directory"
    assert entries[0].recurse is True


def test_put_directory_watch_does_not_create_document(daemon, http, tmp_path):
    """Directory watch registration must not create a document for the dir itself."""
    from keep.watches import list_watches

    server, kp, port = daemon
    d = tmp_path / "vault"
    d.mkdir()

    http.post("/v1/notes", json={
        "uri": f"file://{d}",
        "watch": True,
        "watch_kind": "directory",
    })

    # No document with the directory URI should exist.
    assert kp.get(f"file://{d}") is None
    assert len(list_watches(kp)) == 1


def test_put_directory_enqueue_git_queues_work(daemon, http, tmp_path):
    """Directory control requests can queue initial git-history ingest."""
    server, kp, port = daemon
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("hello\n")

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True, env=env)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, env=env)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo, check=True, capture_output=True, env=env)

    r = http.post("/v1/notes", json={
        "uri": f"file://{repo}",
        "watch_kind": "directory",
        "enqueue_git": True,
        "recurse": True,
    })
    assert r.status_code == 200, r.text
    assert r.json().get("git", {}).get("queued") == 1
    assert kp._get_work_queue().count_by_kind().get("ingest_git") == 1


def test_put_directory_unwatch(daemon, http, tmp_path):
    """Unwatch via PUT /v1/notes with watch_kind=directory removes the watch."""
    from keep.watches import add_watch, list_watches

    server, kp, port = daemon
    d = tmp_path / "vault"
    d.mkdir()
    add_watch(kp, str(d), "directory")
    assert len(list_watches(kp)) == 1

    r = http.post("/v1/notes", json={
        "uri": f"file://{d}",
        "unwatch": True,
        "watch_kind": "directory",
    })
    assert r.status_code == 200
    assert r.json().get("unwatch") is True
    assert list_watches(kp) == []


def test_markdown_sync_registration_exports_and_persists(daemon, http, tmp_path):
    from keep.markdown_mirrors import list_markdown_mirrors

    server, kp, port = daemon
    kp.put("mirror body", id="mirror-doc")

    root = tmp_path / "vault"
    root.mkdir()
    (root / ".obsidian").mkdir()

    r = http.post("/v1/admin/markdown-export", json={
        "root": str(root),
        "sync": True,
        "interval": "PT45S",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sync"]["root"] == str(root.resolve())
    assert body["sync"]["interval"] == "PT45S"
    assert (root / "mirror-doc.md").is_file()
    assert (root / ".keep-sync" / "map.tsv").is_file()

    entries = list_markdown_mirrors(kp)
    assert len(entries) == 1
    assert entries[0].root == str(root.resolve())
    assert entries[0].last_run
    assert entries[0].pending_since == ""
    assert kp._document_store.sync_outbox_depth() == 0


def test_markdown_sync_list_returns_registered_mirrors(daemon, http, tmp_path):
    from keep.markdown_mirrors import add_markdown_mirror

    server, kp, port = daemon
    root = tmp_path / "vault"
    root.mkdir()
    add_markdown_mirror(
        kp,
        root,
        include_parts=True,
        include_versions=True,
    )

    r = http.post("/v1/admin/markdown-export", json={"list": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "mirrors" in body
    assert len(body["mirrors"]) == 1
    entry = body["mirrors"][0]
    assert entry["root"] == str(root.resolve())
    assert entry["include_parts"] is True
    assert entry["include_versions"] is True
    assert entry["enabled"] is True


def test_markdown_sync_validate_only_checks_root_without_persisting(daemon, http, tmp_path):
    from keep.markdown_mirrors import list_markdown_mirrors

    server, kp, port = daemon
    root = tmp_path / "vault"
    root.mkdir()

    r = http.post("/v1/admin/markdown-export", json={
        "root": str(root),
        "sync": True,
        "validate_only": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["validated"] is True
    assert body["root"] == str(root.resolve())
    assert list_markdown_mirrors(kp) == []


def test_markdown_sync_register_only_marks_baseline_complete(daemon, http, tmp_path):
    from keep.markdown_mirrors import list_markdown_mirrors

    server, kp, port = daemon
    kp.put("mirror body", id="mirror-doc")
    assert kp._document_store.sync_outbox_depth() > 0
    root = tmp_path / "vault"
    root.mkdir()

    r = http.post("/v1/admin/markdown-export", json={
        "root": str(root),
        "sync": True,
        "register_only": True,
        "baseline_complete": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sync"]["root"] == str(root.resolve())
    assert not (root / ".keep-sync" / "map.tsv").exists()

    entries = list_markdown_mirrors(kp)
    assert len(entries) == 1
    assert entries[0].root == str(root.resolve())
    assert entries[0].last_run
    assert kp._document_store.sync_outbox_depth() == 0


def test_markdown_sync_stop_removes_registration(daemon, http, tmp_path):
    from keep.markdown_mirrors import add_markdown_mirror, list_markdown_mirrors

    server, kp, port = daemon
    root = tmp_path / "vault"
    root.mkdir()
    add_markdown_mirror(kp, root)
    assert len(list_markdown_mirrors(kp)) == 1

    r = http.post("/v1/admin/markdown-export", json={
        "root": str(root),
        "sync": True,
        "stop": True,
    })
    assert r.status_code == 200
    assert r.json()["stopped"] is True
    assert list_markdown_mirrors(kp) == []


# --- Tag ---

def test_tag(http):
    http.post("/v1/notes", json={"content": "tag test", "id": "tag-1"})
    r = http.patch("/v1/notes/tag-1/tags", json={"set": {"color": "blue"}})
    assert r.status_code == 200
    assert r.json()["tags"].get("color") == "blue"


# --- Find ---

def test_find(http):
    http.post("/v1/notes", json={"content": "alpha beta", "id": "s-1"})
    r = http.post("/v1/search", json={"query": "alpha", "limit": 5})
    assert r.status_code == 200
    assert "notes" in r.json()


def test_http_compat_routes_delegate_to_run_flow(daemon):
    server, kp, port = daemon
    kp.ensure_sysdocs()
    client = httpx.Client(
        base_url=f"http://127.0.0.1:{port}",
        headers={"Authorization": f"Bearer {server.auth_token}"},
        timeout=5,
    )
    calls: list[tuple[str, bool]] = []
    original = kp.run_flow

    def tracking(state, **kwargs):
        calls.append((state, kwargs.get("state_doc_yaml") is not None))
        return original(state, **kwargs)

    kp.run_flow = tracking  # type: ignore[method-assign]
    try:
        r = client.post("/v1/notes", json={"content": "route test", "id": "rt-compat"})
        assert r.status_code == 200
        assert calls == [("put", False)]

        calls.clear()
        r = client.get("/v1/notes/rt-compat")
        assert r.status_code == 200
        assert calls == [("compat-get-item", True)]

        calls.clear()
        r = client.post("/v1/search", json={"query": "route"})
        assert r.status_code == 200
        assert calls == [("compat-find", True)]

        calls.clear()
        r = client.patch("/v1/notes/rt-compat/tags", json={"set": {"color": "blue"}})
        assert r.status_code == 200
        assert calls == [("tag", False), ("compat-get-item", True)]

        calls.clear()
        r = client.delete("/v1/notes/rt-compat")
        assert r.status_code == 200
        assert calls == [("delete", False)]
    finally:
        client.close()


def test_request_models_ignore_unknown_fields():
    """Daemon request models accept but do not retain unknown payload keys."""
    req = PutRequest.model_validate({
        "content": "test note",
        "id": "extra-field-test",
        "unexpected": "ignored",
    })

    assert req.content == "test note"
    assert req.model_extra is None
    assert not hasattr(req, "unexpected")


def test_daemon_json_default_encodes_only_known_types():
    """Daemon JSON fallback is explicit instead of converting every object."""
    assert _daemon_json_default(datetime(2026, 4, 27, tzinfo=timezone.utc)) == (
        "2026-04-27T00:00:00+00:00"
    )
    assert _daemon_json_default(Path("/tmp/keep")) == "/tmp/keep"
    generated = uuid.uuid4()
    assert _daemon_json_default(generated) == str(generated)

    with pytest.raises(TypeError):
        _daemon_json_default(object())


# --- Flow ---

def test_flow(http):
    http.post("/v1/notes", json={"content": "flow test", "id": "f-1"})
    r = http.post("/v1/flow", json={
        "state": "get",
        "params": {"item_id": "f-1", "similar_limit": 1, "meta_limit": 1,
                   "parts_limit": 0, "edges_limit": 0, "versions_limit": 0},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert "bindings" in body


# --- Port fallback ---

def test_port_fallback(mock_providers, tmp_path):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    occupied_port = sock.getsockname()[1]
    sock.listen(1)
    try:
        kp = Keeper(store_path=tmp_path)
        server = DaemonServer(kp, port=occupied_port)
        actual_port = server.start()
        assert actual_port != occupied_port
        r = httpx.get(
            f"http://127.0.0.1:{actual_port}/v1/ready",
            headers={"Authorization": f"Bearer {server.auth_token}"},
            timeout=5,
        )
        assert r.status_code == 200
        server.stop()
        kp.close()
    finally:
        sock.close()


# --- RemoteKeeper round-trip ---

def test_remote_keeper_round_trip(daemon):
    server, kp, port = daemon
    from keep.remote import RemoteKeeper

    client = RemoteKeeper(
        api_url=f"http://127.0.0.1:{port}",
        api_key=server.auth_token, config=kp.config)

    item = client.put(content="round trip test", id="rt-1", tags={"status": "test"})
    assert item.id == "rt-1"

    item = client.get("rt-1")
    assert item is not None
    assert item.id == "rt-1"

    results = client.find(query="round trip")
    assert isinstance(results, list)

    tagged = client.tag("rt-1", {"color": "red"})
    assert tagged is not None

    assert client.exists("rt-1")
    assert not client.exists("nonexistent")
    assert client.delete("rt-1") is True

    client.close()


def test_remote_keeper_capabilities_round_trip(daemon):
    server, kp, port = daemon
    from keep.remote import RemoteKeeper

    client = RemoteKeeper(
        api_url=f"http://127.0.0.1:{port}",
        api_key=server.auth_token, config=kp.config)

    info = client.server_info()
    assert info["network"]["mode"] == "local"
    assert client.supports_capability("export_bundle") is True
    assert client.supports_capability("export_changes") is True

    client.close()


def test_remote_keeper_export_round_trip(daemon):
    server, kp, port = daemon
    from keep.remote import RemoteKeeper

    client = RemoteKeeper(
        api_url=f"http://127.0.0.1:{port}",
        api_key=server.auth_token, config=kp.config)

    client.put(content="export body", id="export-remote-1", tags={"topic": "transport"})
    client.put(content="system body", id=".system-remote-export")

    data = client.export_data(include_system=False)
    assert data["format"] == "keep-export"
    ids = {doc["id"] for doc in data["documents"]}
    assert "export-remote-1" in ids
    assert ".system-remote-export" not in ids

    chunks = list(client.export_iter(include_system=False))
    assert chunks[0]["format"] == "keep-export"
    ids = {doc["id"] for doc in chunks[1:]}
    assert "export-remote-1" in ids
    assert ".system-remote-export" not in ids

    client.close()


def test_remote_keeper_export_bundle_round_trip(daemon):
    server, kp, port = daemon
    from keep.remote import RemoteKeeper

    client = RemoteKeeper(
        api_url=f"http://127.0.0.1:{port}",
        api_key=server.auth_token, config=kp.config)

    client.put(content="person body", id="bob")
    client.put(
        content="Conversation body. " * 40,
        id="conv-bundle-1",
        tags={"speaker": "bob"},
    )
    with patch("keep.analyzers.SlidingWindowAnalyzer.analyze") as mock_analyze:
        mock_analyze.return_value = [
            {"summary": "Bundle opening", "tags": {"speaker": "bob"}},
            {"summary": "Bundle follow-up"},
        ]
        client._client.post(
            "/v1/analyze",
            json={"id": "conv-bundle-1", "foreground": True, "force": True},
        ).raise_for_status()

    bundle = client.export_bundle("conv-bundle-1")
    assert bundle is not None
    assert "parts" in bundle["document"]
    assert "speaker" in bundle["edge_tag_keys"]
    assert bundle["current_inverse"] == []

    bundle = client.export_bundle("conv-bundle-1", include_parts=False)
    assert bundle is not None
    assert bundle["document"]["id"] == "conv-bundle-1"
    assert "parts" not in bundle["document"]
    bundle = client.export_bundle("bob")
    assert bundle is not None
    assert any(edge[0] == "said" for edge in bundle["current_inverse"])

    assert client.export_bundle("nonexistent") is None

    client.close()


def test_remote_keeper_export_changes_round_trip(daemon):
    server, kp, port = daemon
    from keep.remote import RemoteKeeper

    client = RemoteKeeper(
        api_url=f"http://127.0.0.1:{port}",
        api_key=server.auth_token, config=kp.config)

    client.put(content="change feed body", id="remote-changes-1")
    feed = client.export_changes(cursor="0", limit=500)
    assert feed["format"] == "keep-export-changes"
    event = next(row for row in feed["events"] if row["entity_id"] == "remote-changes-1")
    assert event["affected_note_ids"] == ["remote-changes-1"]

    feed = client.export_changes(cursor=feed["cursor"], limit=500)
    assert feed["events"] == []

    client.close()


def test_context_endpoint(http):
    """The /context endpoint returns full ItemContext in one call."""
    http.post("/v1/notes", json={"content": "context endpoint test", "id": "ce-1"})
    r = http.get("/v1/notes/ce-1/context", params={"similar_limit": 2, "edges_limit": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["item"]["id"] == "ce-1"
    assert "similar" in body
    assert "meta" in body
    assert "parts" in body
    assert "prev" in body

    r = http.get("/v1/notes/nonexistent/context")
    assert r.status_code == 404


def test_remote_keeper_get_context_via_flow(daemon):
    """get_context() uses get + flow endpoint."""
    server, kp, port = daemon
    from keep.remote import RemoteKeeper

    client = RemoteKeeper(
        api_url=f"http://127.0.0.1:{port}",
        api_key=server.auth_token, config=kp.config)

    client.put(content="context test note", id="ctx-1")
    ctx = client.get_context("ctx-1", edges_limit=2, parts_limit=5)
    assert ctx is not None
    assert ctx.item.id == "ctx-1"
    assert isinstance(ctx.similar, list)
    assert isinstance(ctx.meta, dict)
    assert isinstance(ctx.parts, list)

    ctx = client.get_context("nonexistent")
    assert ctx is None

    client.close()


# --- Prompt via flow ---

def test_prompt_via_flow(daemon, http):
    _, kp, _ = daemon
    kp.put(
        content="# Test\nA test.\n\n## Prompt\nHello {get}",
        id=".prompt/agent/test-render",
        tags={"state": "get"},
    )
    r = http.post("/v1/flow", json={
        "state": "prompt", "params": {"name": "test-render"},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "done"
    assert "text" in body.get("data", {})
    assert len(body["data"]["text"]) > 0


def test_prompt_not_found_via_flow(http):
    r = http.post("/v1/flow", json={
        "state": "prompt", "params": {"name": "nonexistent-prompt"},
    })
    assert r.status_code == 200
    assert r.json()["status"] == "error"


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def test_stop_releases_socket(mock_providers, tmp_path):
    """stop() calls server_close() so the port is immediately reusable."""
    kp = Keeper(store_path=tmp_path)
    server = DaemonServer(kp, port=0)
    port = server.start()

    server.stop()

    # The port should be free for immediate rebind
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as e:
        pytest.fail(f"Port {port} still bound after stop(): {e}")
    finally:
        sock.close()
        kp.close()
