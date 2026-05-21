"""End-to-end smoke tests against the live keep backend (keepnotes.ai).

Opt-in: requires KEEPNOTES_API_KEY in the environment. Run with::

    .venv/bin/python -m pytest tests/test_smoke_remote.py -m smoke -v

These tests exercise the same three entry points that production users hit:

1. ``RemoteKeeper`` directly — the CLI's transport when ``[remote]`` is set.
2. The MCP ``_RemoteBackend`` — used by ``keep mcp`` in remote mode.
3. ``setup_wizard._verify_remote`` — the credential check run during setup.

Every smoke note is tagged with a unique ``keep_smoke_run`` value so cleanup
is safe even if a test aborts midway. The fixture deletes every note carrying
the run-id tag at the end of the session.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _require_api_key() -> str:
    """Skip smoke tests when no API key is available."""
    key = os.environ.get("KEEPNOTES_API_KEY")
    if not key:
        pytest.skip("KEEPNOTES_API_KEY not set; remote smoke tests skipped")
    return key


@pytest.fixture(scope="module")
def smoke_run_id() -> str:
    """One id shared by every note this module creates."""
    return f"keep-smoke-{uuid.uuid4().hex[:12]}"


@pytest.fixture(scope="module")
def remote_credentials() -> dict:
    """Resolve api_url / api_key / project from env (skip if missing)."""
    api_key = _require_api_key()
    return {
        "api_url": os.environ.get("KEEPNOTES_API_URL", "https://api.keepnotes.ai"),
        "api_key": api_key,
        "project": os.environ.get("KEEPNOTES_PROJECT"),
    }


@pytest.fixture
def remote_keeper(remote_credentials, smoke_run_id, tmp_path):
    """Yield a RemoteKeeper bound to the live backend; cleanup at teardown."""
    from keep.config import StoreConfig
    from keep.remote import RemoteKeeper

    cfg = StoreConfig(path=tmp_path, config_dir=tmp_path)
    keeper = RemoteKeeper(
        remote_credentials["api_url"],
        remote_credentials["api_key"],
        cfg,
        project=remote_credentials["project"],
    )
    created_ids: list[str] = []
    try:
        # Expose a way for tests to record ids without leaking on failure.
        keeper._smoke_created = created_ids  # type: ignore[attr-defined]
        yield keeper
    finally:
        _cleanup_by_tag(keeper, smoke_run_id)
        keeper.close()


def _cleanup_by_tag(keeper, smoke_run_id: str) -> None:
    """Delete every note carrying the smoke-run tag, best-effort."""
    try:
        items = keeper.find(tags={"keep_smoke_run": smoke_run_id}, limit=200)
    except Exception:
        return
    for item in items:
        try:
            keeper.delete(item.id)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. RemoteKeeper round-trip — the CLI transport
# ---------------------------------------------------------------------------


class TestRemoteKeeperRoundTrip:
    """RemoteKeeper put/get/find/delete against the live service."""

    def test_put_get_find_delete_roundtrip(self, remote_keeper, smoke_run_id):
        note_id = f"{smoke_run_id}-roundtrip"
        body = f"Smoke roundtrip {note_id}. Created by tests/test_smoke_remote.py."
        tags = {"keep_smoke_run": smoke_run_id, "phase": "put"}

        stored = remote_keeper.put(content=body, id=note_id, tags=tags)
        remote_keeper._smoke_created.append(note_id)
        assert stored.id == note_id

        fetched = remote_keeper.get(note_id)
        assert fetched is not None
        assert fetched.id == note_id
        # The remote may strip leading/trailing whitespace; substring match is enough.
        assert smoke_run_id in (fetched.summary or "") or \
            fetched.tags.get("keep_smoke_run") == smoke_run_id

        listed = remote_keeper.find(
            tags={"keep_smoke_run": smoke_run_id, "phase": "put"}, limit=5,
        )
        assert any(item.id == note_id for item in listed), \
            f"Expected {note_id} in tag-filtered find results"

        deleted = remote_keeper.delete(note_id)
        assert deleted is True
        assert remote_keeper.get(note_id) is None

    def test_server_info_reports_version(self, remote_keeper):
        """``/v1/ready`` returns capability/version info."""
        info = remote_keeper.server_info()
        assert isinstance(info, dict)
        # Don't pin the exact key — surface whatever the server advertises.
        assert info, "Expected non-empty server_info response"


# ---------------------------------------------------------------------------
# 2. MCP _RemoteBackend round-trip — the MCP transport
# ---------------------------------------------------------------------------


class TestMCPRemoteBackendRoundTrip:
    """The MCP backend's HTTP path should round-trip against the live service."""

    def test_mcp_remote_backend_put_get_delete(
        self, remote_credentials, smoke_run_id, tmp_path,
    ):
        from keep.config import RemoteConfig
        from keep.mcp import _RemoteBackend

        backend = _RemoteBackend(
            RemoteConfig(**remote_credentials),
            log_dir=tmp_path,
        )
        note_id = f"{smoke_run_id}-mcp"
        tags = {"keep_smoke_run": smoke_run_id, "phase": "mcp"}

        try:
            # put
            status, body = backend.post("/v1/flow", {
                "state": "put",
                "params": {
                    "id": note_id,
                    "content": f"MCP smoke {note_id}.",
                    "tags": tags,
                },
            })
            assert status == 200, body
            assert body.get("status") in ("done", "ok"), body

            # get (via flow — what MCP tools use)
            status, body = backend.post("/v1/flow", {
                "state": "get",
                "params": {"item_id": note_id},
            })
            assert status == 200, body
            data = body.get("data") or {}
            item = data.get("item") or data
            assert (item.get("id") if isinstance(item, dict) else None) == note_id, body

            # delete
            status, body = backend.post("/v1/flow", {
                "state": "delete",
                "params": {"id": note_id},
            })
            assert status == 200, body
        finally:
            # Fallback cleanup if any step left the note behind.
            try:
                from keep.config import StoreConfig
                from keep.remote import RemoteKeeper
                kp = RemoteKeeper(
                    remote_credentials["api_url"],
                    remote_credentials["api_key"],
                    StoreConfig(path=tmp_path, config_dir=tmp_path),
                    project=remote_credentials["project"],
                )
                try:
                    _cleanup_by_tag(kp, smoke_run_id)
                finally:
                    kp.close()
            except Exception:
                pass
            backend.close()

    def test_mcp_remote_backend_writes_client_log(
        self, remote_credentials, smoke_run_id, tmp_path,
    ):
        """A real round-trip leaves at least one entry in keep-client.log."""
        from keep.config import RemoteConfig
        from keep.mcp import _RemoteBackend

        backend = _RemoteBackend(
            RemoteConfig(**remote_credentials),
            log_dir=tmp_path,
        )
        try:
            # ready is cheap and authenticated.
            status, _ = backend.get("/v1/ready")
            assert status in (200, 404)  # ok if not implemented identically
        finally:
            backend.close()

        log_path = tmp_path / "keep-client.log"
        assert log_path.exists()
        body = log_path.read_text(encoding="utf-8")
        assert "mcp.remote: GET /v1/ready" in body


# ---------------------------------------------------------------------------
# 3. Setup-wizard verification against the live service
# ---------------------------------------------------------------------------


class TestWizardRemoteVerification:
    """``_verify_remote`` should succeed against real credentials and fail loudly otherwise."""

    def test_verify_remote_succeeds_with_valid_credentials(self, remote_credentials):
        from keep.config import RemoteConfig
        from keep.setup_wizard import _verify_remote

        ok, message = _verify_remote(RemoteConfig(**remote_credentials))
        assert ok, f"Expected verification to pass; got {message!r}"
        assert "OK" in message or "round trip" in message

    def test_verify_remote_rejects_bad_key(self, remote_credentials):
        from keep.config import RemoteConfig
        from keep.setup_wizard import _verify_remote

        bad = dict(remote_credentials)
        bad["api_key"] = "kn_obviously_invalid_smoke_key"
        ok, message = _verify_remote(RemoteConfig(**bad))
        assert not ok
        assert "failed" in message.lower() or "401" in message or "403" in message
