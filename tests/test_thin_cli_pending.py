"""Tests for command-app pending command lifecycle behavior."""

import re
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from keep import cli_app
from keep.const import DAEMON_PORT_FILE, DAEMON_TOKEN_FILE
from keep.markdown_mirrors import MarkdownMirrorEntry
from tests.conftest import _write_test_store_config


def test_pending_stop_cleans_stale_discovery_files_without_pid(tmp_path, capsys):
    store = tmp_path / "store"
    store.mkdir()
    (store / DAEMON_PORT_FILE).write_text("5337")
    (store / DAEMON_TOKEN_FILE).write_text("token")

    with patch("keep.daemon_client.resolve_store_path", return_value=store):
        cli_app.pending(stop=True)

    captured = capsys.readouterr()
    assert "No daemon running." in captured.out
    assert not (store / DAEMON_PORT_FILE).exists()
    assert not (store / DAEMON_TOKEN_FILE).exists()


def test_pending_mentions_active_markdown_mirrors(capsys):
    kp = MagicMock()
    kp.pending_count.return_value = 0
    kp.pending_work_count.return_value = 0
    kp._pending_queue.stats.return_value = {
        "failed": 0, "processing": 0, "pending": 0, "delegated": 0,
    }
    kp._is_processor_running.return_value = True
    kp._store_path = MagicMock()

    with patch("keep.markdown_mirrors.list_markdown_mirrors", return_value=[
        MarkdownMirrorEntry(root="/tmp/vault", enabled=True),
    ]), \
         patch("keep.watches.has_active_watches", return_value=False), \
         patch("keep.console_support._tail_ops_log"), \
         patch("keep.console_support.typer.echo") as echo:
        from keep.console_support import print_pending_interactive

        print_pending_interactive(kp)

    messages = [call.args[0] for call in echo.call_args_list]
    assert "Markdown mirrors active: 1" in messages


def test_root_help_shows_daemon_and_hides_pending_alias():
    runner = CliRunner()

    result = runner.invoke(cli_app.app, ["--help"])

    assert result.exit_code == 0
    assert re.search(r"^\s+daemon\s", result.stdout, re.MULTILINE)
    assert not re.search(r"^\s+pending\s", result.stdout, re.MULTILINE)


def test_daemon_command_runs_foreground_daemon_with_transport_options(tmp_path):
    runner = CliRunner()
    daemon_run: dict[str, object] = {}

    class DummyKeeper:
        def __init__(self, *args, **kwargs):
            pass

    with (
        patch("keep.api.Keeper", DummyKeeper),
        patch(
            "keep.console_support.run_pending_daemon",
            side_effect=lambda *args, **kwargs: daemon_run.update(kwargs),
        ),
    ):
        result = runner.invoke(
            cli_app.app,
            [
                "--store", str(tmp_path),
                "daemon",
                "--bind", "0.0.0.0",
                "--advertised-url", "https://keep.example.test",
                "--trusted-proxy",
            ],
        )

    assert result.exit_code == 0, result.stdout
    assert daemon_run["bind_host"] == "0.0.0.0"
    assert daemon_run["advertised_url"] == "https://keep.example.test"
    assert daemon_run["trusted_proxy"] is True


def test_hidden_pending_alias_still_runs_interactive_mode(tmp_path):
    runner = CliRunner()
    kp = MagicMock()

    with (
        patch("keep.daemon_client.resolve_store_path", return_value=tmp_path),
        patch("keep.api.Keeper", return_value=kp),
        patch("keep.console_support.print_pending_interactive") as interactive,
    ):
        result = runner.invoke(cli_app.app, ["pending"])

    assert result.exit_code == 0, result.stdout
    interactive.assert_called_once_with(kp)
    kp.close.assert_called_once()


def _write_remote_store_config(store):
    _write_test_store_config(store)
    with (store / "keep.toml").open("a", encoding="utf-8") as fh:
        fh.write(
            "\n[remote]\n"
            "api_url = \"https://api.example.test\"\n"
            "api_key = \"kn_test\"\n"
            "project = \"first-user\"\n"
        )


def test_pending_does_not_open_local_store_when_remote_configured(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    _write_remote_store_config(store)

    monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
    monkeypatch.setattr(cli_app, "_global_store", None)
    runner = CliRunner()

    with (
        patch(
            "keep.api.Keeper",
            side_effect=AssertionError("local Keeper must not open in remote mode"),
        ),
        patch(
            "keep.console_support.run_pending_daemon",
            side_effect=AssertionError("local daemon must not run in remote mode"),
        ),
    ):
        pending_result = runner.invoke(
            cli_app.app,
            ["--store", str(store), "pending"],
            catch_exceptions=False,
        )

    assert pending_result.exit_code == 0, pending_result.output
    assert "Remote backend configured" in pending_result.stderr


def test_daemon_runs_local_services_when_remote_configured(tmp_path, monkeypatch):
    store = tmp_path / "store"
    store.mkdir()
    _write_remote_store_config(store)

    monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
    monkeypatch.setattr(cli_app, "_global_store", None)
    kp = MagicMock()
    runner = CliRunner()

    with (
        patch("keep.api.Keeper", return_value=kp) as keeper_cls,
        patch("keep.console_support.run_pending_daemon") as run_daemon,
    ):
        daemon_result = runner.invoke(
            cli_app.app,
            ["--store", str(store), "daemon"],
            catch_exceptions=False,
        )

    assert daemon_result.exit_code == 0, daemon_result.output
    keeper_cls.assert_called_once_with(store_path=store.resolve())
    run_daemon.assert_called_once_with(
        kp,
        bind_host=None,
        advertised_url=None,
        trusted_proxy=False,
    )


def test_reset_system_docs_targets_local_daemon_even_when_remote_configured(
    tmp_path, monkeypatch,
):
    """`keep config --reset-system-docs` is local maintenance; never route remote."""
    store = tmp_path / "store"
    store.mkdir()
    _write_remote_store_config(store)

    monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
    monkeypatch.setattr(cli_app, "_global_store", None)
    cli_app._invalidate_cli_remote_cache()

    daemon_calls: list[tuple[str, int, str, dict | None]] = []

    def fake_daemon_request(method, port, path, body=None):
        daemon_calls.append((method, port, path, body))
        return 200, {"reset": 5}

    with (
        patch("keep.cli_app._get_local_daemon_port", return_value=1234),
        patch(
            "keep.cli_app._remote_request",
            side_effect=AssertionError("reset-system-docs must not route remote"),
        ),
        patch("keep.cli_app._daemon_request", side_effect=fake_daemon_request),
    ):
        result = CliRunner().invoke(
            cli_app.app,
            ["--store", str(store), "config", "--reset-system-docs"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert daemon_calls == [
        ("POST", 1234, "/v1/admin/reset-system-docs", {})
    ]
    assert "Reset 5 system documents" in result.output


def test_load_cli_remote_is_cached_across_calls(tmp_path, monkeypatch):
    """Successive _load_cli_remote() calls reuse the cached load_config result."""
    store = tmp_path / "store"
    store.mkdir()
    _write_remote_store_config(store)

    monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
    monkeypatch.setattr(cli_app, "_global_store", store.resolve())
    cli_app._invalidate_cli_remote_cache()

    import keep.cli_app as _ca

    calls = {"n": 0}
    real_load = _ca.load_config

    def counting_load(path):
        calls["n"] += 1
        return real_load(path)

    monkeypatch.setattr(_ca, "load_config", counting_load)

    a = _ca._load_cli_remote()
    b = _ca._load_cli_remote()
    c = _ca._load_cli_remote()

    assert a is not None
    assert a is b is c                  # same cached tuple object
    assert calls["n"] == 1               # only one TOML read across three calls

    # Changing an env var that affects resolution must invalidate the cache.
    monkeypatch.setenv("KEEPNOTES_API_URL", "https://override.example.test")
    d = _ca._load_cli_remote()
    assert d is not None
    assert d[0].api_url == "https://override.example.test"
    assert calls["n"] == 2

    cli_app._invalidate_cli_remote_cache()
