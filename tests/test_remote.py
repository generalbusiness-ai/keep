from pathlib import Path
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


def test_validate_remote_api_url_returns_normalized_string():
    """validate_remote_api_url must return the URL (regression: silent None)."""
    from keep.remote import validate_remote_api_url

    assert validate_remote_api_url("https://api.example.test/") == "https://api.example.test"
    assert validate_remote_api_url("http://127.0.0.1:8080") == "http://127.0.0.1:8080"


class TestResolveRemoteConfig:
    """Single source of truth for env-over-TOML remote resolution."""

    def test_returns_none_when_keep_local_only(self, monkeypatch):
        from keep.remote import resolve_remote_config

        monkeypatch.setenv("KEEP_LOCAL_ONLY", "1")
        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_test")
        assert resolve_remote_config(None) is None

    def test_returns_none_when_no_credentials_anywhere(self, monkeypatch):
        from keep.remote import resolve_remote_config

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        for var in ("KEEPNOTES_API_URL", "KEEPNOTES_API_KEY", "KEEPNOTES_PROJECT"):
            monkeypatch.delenv(var, raising=False)
        assert resolve_remote_config(None) is None

    def test_env_only_returns_remote_with_default_api_url(self, monkeypatch):
        from keep.remote import resolve_remote_config

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_URL", raising=False)
        monkeypatch.delenv("KEEPNOTES_PROJECT", raising=False)
        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_env")
        remote = resolve_remote_config(None)
        assert remote is not None
        assert remote.api_url == "https://api.keepnotes.ai"
        assert remote.api_key == "kn_env"
        assert remote.project is None

    def test_env_only_returns_remote_config_not_url_string(self, monkeypatch):
        """resolve_remote_config must return a RemoteConfig object, not a bare URL string.

        Guard against a regression where a stray ``return url`` statement (with
        ``url`` undefined in this scope) was present after the real return.  The
        real return is ``return RemoteConfig(...)``, so the result must always be
        a RemoteConfig instance, never a str.
        """
        from keep.config import RemoteConfig
        from keep.remote import resolve_remote_config

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_URL", raising=False)
        monkeypatch.delenv("KEEPNOTES_PROJECT", raising=False)
        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_guard")
        result = resolve_remote_config(None)
        assert isinstance(result, RemoteConfig), (
            f"resolve_remote_config must return a RemoteConfig, got {type(result)!r}"
        )
        assert result.api_url == "https://api.keepnotes.ai"
        assert result.api_key == "kn_guard"

    def test_env_overlays_toml_per_field(self, monkeypatch):
        """KEEPNOTES_API_KEY alone keeps TOML api_url/project."""
        from keep.config import RemoteConfig, StoreConfig
        from keep.remote import resolve_remote_config

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_URL", raising=False)
        monkeypatch.delenv("KEEPNOTES_PROJECT", raising=False)
        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_env")
        config = StoreConfig(
            path=Path("/tmp"),
            remote_persist=RemoteConfig(
                api_url="https://config.example.test",
                api_key="kn_file",
                project="from-file",
            ),
        )
        remote = resolve_remote_config(config)
        assert remote.api_url == "https://config.example.test"
        assert remote.api_key == "kn_env"
        assert remote.project == "from-file"

    def test_toml_only_returns_persisted_values(self, monkeypatch):
        from keep.config import RemoteConfig, StoreConfig
        from keep.remote import resolve_remote_config

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        for var in ("KEEPNOTES_API_URL", "KEEPNOTES_API_KEY", "KEEPNOTES_PROJECT"):
            monkeypatch.delenv(var, raising=False)
        config = StoreConfig(
            path=Path("/tmp"),
            remote_persist=RemoteConfig(
                api_url="https://config.example.test",
                api_key="kn_file",
                project="from-file",
            ),
        )
        remote = resolve_remote_config(config)
        assert remote.api_url == "https://config.example.test"
        assert remote.api_key == "kn_file"
        assert remote.project == "from-file"

    def test_prefers_remote_persist_over_remote(self, monkeypatch):
        """The on-disk view (remote_persist) wins over an already-overlaid remote."""
        from keep.config import RemoteConfig, StoreConfig
        from keep.remote import resolve_remote_config

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        for var in ("KEEPNOTES_API_URL", "KEEPNOTES_API_KEY", "KEEPNOTES_PROJECT"):
            monkeypatch.delenv(var, raising=False)
        config = StoreConfig(
            path=Path("/tmp"),
            remote=RemoteConfig(
                api_url="https://overlaid.example.test",
                api_key="kn_overlaid",
                project=None,
            ),
            remote_persist=RemoteConfig(
                api_url="https://persisted.example.test",
                api_key="kn_persisted",
                project="persisted-proj",
            ),
        )
        remote = resolve_remote_config(config)
        # remote_persist values take precedence over already-overlaid config.remote.
        assert remote.api_url == "https://persisted.example.test"
        assert remote.api_key == "kn_persisted"
        assert remote.project == "persisted-proj"


def test_all_call_sites_use_shared_resolver(monkeypatch):
    """Every site that decides "go remote?" must go through resolve_remote_config.

    The helper is imported by name at module load (cli_app) or via local
    import (mcp, setup_wizard, console_support), so we patch wherever each
    site looks it up and verify the call passes through.
    """
    from keep import cli_app, mcp, setup_wizard
    from keep.config import StoreConfig

    monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
    monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_test")

    sentinel_calls: list[str] = []

    def fake_resolver(config):
        sentinel_calls.append("called")
        return None  # Force the no-remote path in every caller.

    # Patch both the canonical location and the names imported at module
    # load. If a future call site is added that re-implements the overlay
    # rule inline, the per-site assertion below will not increment and the
    # test fails.
    monkeypatch.setattr("keep.remote.resolve_remote_config", fake_resolver)
    monkeypatch.setattr("keep.cli_app.resolve_remote_config", fake_resolver)

    # MCP: imported locally inside _load_remote_config.
    mcp._backend = None
    try:
        mcp._load_remote_config()
    finally:
        mcp._backend = None
    assert len(sentinel_calls) >= 1

    # CLI: imported at module load.
    cli_app._invalidate_cli_remote_cache()
    before = len(sentinel_calls)
    cli_app._compute_cli_remote()
    assert len(sentinel_calls) > before

    # Wizard: imported locally inside _detect_remote_config.
    before = len(sentinel_calls)
    setup_wizard._detect_remote_config(StoreConfig(path=Path("/tmp")))
    assert len(sentinel_calls) > before


def test_get_keeper_env_only_marks_config_as_env_sourced(tmp_path, monkeypatch):
    """No TOML [remote] + env credentials ⇒ remote_persist=None, remote_from_env=True.

    Aligns _get_keeper's resolution with load_config's contract so that a
    later save_config() through this Keeper never writes env-sourced
    credentials back to disk.
    """
    store = tmp_path / "store"
    store.mkdir()
    _write_test_store_config(store)  # no [remote] section appended

    monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("KEEP_CONFIG", raising=False)
    monkeypatch.delenv("KEEPNOTES_API_URL", raising=False)
    monkeypatch.delenv("KEEPNOTES_PROJECT", raising=False)
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

    assert keeper.api_key == "kn_env_only"
    # Critical: the resolved config must declare this as env-sourced so a
    # subsequent save_config() preserves an empty on-disk [remote].
    assert keeper.config.remote_persist is None
    assert keeper.config.remote_from_env is True


def test_get_keeper_toml_remote_preserves_persisted_credentials(tmp_path, monkeypatch):
    """TOML [remote] + env overlay ⇒ remote_persist=TOML, remote_from_env=False.

    Even with env vars overlaying api_key, the persisted snapshot must point
    at the file's values so save_config() keeps the [remote] section intact.
    """
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
    monkeypatch.delenv("KEEP_CONFIG", raising=False)
    monkeypatch.delenv("KEEPNOTES_API_URL", raising=False)
    monkeypatch.delenv("KEEPNOTES_PROJECT", raising=False)
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

    # Runtime credentials use the env overlay.
    assert keeper.api_key == "kn_env_only"
    # Persisted snapshot matches the TOML file, not the env value.
    assert keeper.config.remote_persist is not None
    assert keeper.config.remote_persist.api_key == "kn_file"
    assert keeper.config.remote_persist.api_url == "https://config.example.test"
    assert keeper.config.remote_persist.project == "from-file"
    assert keeper.config.remote_from_env is False
