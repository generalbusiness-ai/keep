from unittest.mock import MagicMock, patch

import httpx
import pytest

from keep.config import StoreConfig
from keep.remote import RemoteKeeper
from tests.conftest import _write_test_store_config


def test_remote_http_error_includes_daemon_request_id(tmp_path):
    keeper = RemoteKeeper("http://localhost:9999", "", StoreConfig(path=tmp_path))
    request = httpx.Request("GET", "http://localhost:9999/v1/notes/missing")
    response = httpx.Response(
        500,
        json={"error": "internal server error", "request_id": "req-remote"},
        request=request,
    )

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        keeper._raise_for_status(response)

    message = str(excinfo.value)
    assert "internal server error" in message
    assert "request_id=req-remote" in message


def test_remote_streaming_http_error_reads_request_id(tmp_path):
    keeper = RemoteKeeper("http://localhost:9999", "", StoreConfig(path=tmp_path))
    request = httpx.Request("GET", "http://localhost:9999/v1/export")
    response = httpx.Response(
        500,
        stream=httpx.ByteStream(b'{"error":"export failed","request_id":"req-stream"}'),
        request=request,
    )

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        keeper._raise_for_status(response)

    message = str(excinfo.value)
    assert "export failed" in message
    assert "request_id=req-stream" in message


def test_remote_attaches_client_log_to_config_dir(tmp_path):
    """RemoteKeeper writes per-call records to {config_dir}/keep-client.log."""
    config = StoreConfig(path=tmp_path, config_dir=tmp_path)
    keeper = RemoteKeeper("http://localhost:9999", "kn_test", config)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "abc", "summary": "", "tags": {}}
    keeper._client = MagicMock()
    keeper._client.get.return_value = mock_response
    keeper._client.post.return_value = mock_response

    try:
        keeper._get("/v1/notes/abc")
        keeper._post("/v1/notes", {"id": "abc"})
    finally:
        keeper.close()

    log_path = tmp_path / "keep-client.log"
    assert log_path.exists()
    body = log_path.read_text(encoding="utf-8")
    assert "remote: GET /v1/notes/abc" in body
    assert "remote: POST /v1/notes" in body
    assert "host=http://localhost:9999" in body


def test_remote_close_removes_client_log_handler(tmp_path):
    """close() detaches the rotating handler so repeated open/close stays clean."""
    import logging

    config = StoreConfig(path=tmp_path, config_dir=tmp_path)
    keeper = RemoteKeeper("http://localhost:9999", "kn_test", config)
    handler = keeper._client_log_handler
    assert handler is not None
    assert handler in logging.getLogger("keep").handlers

    keeper.close()

    assert handler not in logging.getLogger("keep").handlers
    assert keeper._client_log_handler is None


def test_get_keeper_toml_remote_does_not_construct_local_keeper(tmp_path, monkeypatch):
    """TOML [remote] is authoritative before any local startup work can queue."""
    store = tmp_path / "store"
    store.mkdir()
    _write_test_store_config(store)
    with (store / "keep.toml").open("a", encoding="utf-8") as fh:
        fh.write(
            "\n[remote]\n"
            "api_url = \"https://api.example.test\"\n"
            "api_key = \"kn_test\"\n"
            "project = \"first-user\"\n"
        )

    monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
    monkeypatch.delenv("KEEP_CONFIG", raising=False)

    class FakeRemoteKeeper:
        def __init__(self, api_url, api_key, config, *, project=None):
            self.api_url = api_url
            self.api_key = api_key
            self.config = config
            self.project = project

        def close(self):
            pass

    with (
        patch(
            "keep.api.Keeper",
            side_effect=AssertionError("local Keeper must not be constructed"),
        ),
        patch("keep.remote.RemoteKeeper", FakeRemoteKeeper),
    ):
        from keep.console_support import _get_keeper

        keeper = _get_keeper(store)

    assert isinstance(keeper, FakeRemoteKeeper)
    assert keeper.api_url == "https://api.example.test"
    assert keeper.api_key == "kn_test"
    assert keeper.project == "first-user"


def test_get_keeper_env_key_overlays_store_toml_remote(tmp_path, monkeypatch):
    """Env credentials must not make _get_keeper ignore --store TOML fields."""
    store = tmp_path / "store"
    store.mkdir()
    _write_test_store_config(store)
    with (store / "keep.toml").open("a", encoding="utf-8") as fh:
        fh.write(
            "\n[remote]\n"
            "api_url = \"https://config.example.test\"\n"
            "api_key = \"kn_file\"\n"
            "project = \"from-file\"\n"
        )

    monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("KEEPNOTES_API_URL", raising=False)
    monkeypatch.delenv("KEEPNOTES_PROJECT", raising=False)
    monkeypatch.delenv("KEEP_CONFIG", raising=False)
    monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_env_only")

    class FakeRemoteKeeper:
        def __init__(self, api_url, api_key, config, *, project=None):
            self.api_url = api_url
            self.api_key = api_key
            self.config = config
            self.project = project

        def close(self):
            pass

    with (
        patch(
            "keep.api.Keeper",
            side_effect=AssertionError("local Keeper must not be constructed"),
        ),
        patch("keep.remote.RemoteKeeper", FakeRemoteKeeper),
    ):
        from keep.console_support import _get_keeper

        keeper = _get_keeper(store)

    assert isinstance(keeper, FakeRemoteKeeper)
    assert keeper.api_url == "https://config.example.test"
    assert keeper.api_key == "kn_env_only"
    assert keeper.project == "from-file"
    assert keeper.config.config_dir == store
