"""Focused coverage for diagnostic logging on best-effort fallbacks."""

import logging
from types import SimpleNamespace
from unittest.mock import patch

from keep.api import Keeper
from keep.console_support import _render_context_from_flow_bindings
from keep.daemon_client import check_health


def test_embed_task_reindex_save_config_failure_is_logged(caplog):
    host = SimpleNamespace(
        _config=SimpleNamespace(embed_task_reindex_done=False),
        _config_uses_embed_task=lambda: False,
    )

    caplog.set_level(logging.WARNING, logger="keep.api")
    with patch("keep.api.save_config", side_effect=OSError("disk full")):
        Keeper._enqueue_embed_task_reindex(host)

    assert host._config.embed_task_reindex_done is True
    assert "Failed to persist embed_task_reindex_done" in caplog.text
    assert "disk full" in caplog.text


def test_context_binding_fallback_logs_context_failure(caplog):
    class BrokenContextHost:
        def get_context(self, item_id):
            raise RuntimeError(f"context failed for {item_id}")

    bindings = {"item": {"id": "note-1", "summary": "fallback summary", "tags": {}}}
    caplog.set_level(logging.INFO, logger="keep.console_support")

    output = _render_context_from_flow_bindings(bindings, BrokenContextHost())

    assert "fallback summary" in output
    assert "Failed to render full context for note-1" in caplog.text
    assert "context failed for note-1" in caplog.text


def test_daemon_health_check_failure_is_debug_logged(caplog):
    caplog.set_level(logging.DEBUG, logger="keep.daemon_client")
    with patch("keep.daemon_client.http.client.HTTPConnection", side_effect=OSError("no daemon")):
        assert check_health(43210) is False

    assert "Daemon health check failed for port 43210" in caplog.text
    assert "no daemon" in caplog.text


def test_notify_writes_to_stderr_for_local_keeper(capsys):
    """A CLI Keeper still gets its operator notices on stderr."""
    host = SimpleNamespace(_is_local=True)

    Keeper._notify(host, "Search is unavailable until reindex completes.")

    captured = capsys.readouterr()
    assert "Search is unavailable until reindex completes." in captured.err
    assert captured.out == ""


def test_notify_is_suppressed_for_hosted_keeper(capsys):
    """A hosted Keeper has no terminal, so notices must not reach stderr.

    Anything a hosted process writes to stderr is drained by the platform's
    log agent, which cannot parse it and files every such line as an error.
    """
    host = SimpleNamespace(_is_local=False)

    Keeper._notify(host, "Search is unavailable until reindex completes.")

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_embed_task_reindex_logs_instead_of_printing_when_hosted(caplog, capsys):
    """The migration notice reaches the log, not stderr, on a hosted Keeper.

    Regression guard: this notice used to be a bare print() to stderr, so a
    routine one-time migration surfaced as an ERROR in hosted log search.
    """
    host = SimpleNamespace(
        _is_local=False,
        _config=SimpleNamespace(embed_task_reindex_done=False),
        _config_uses_embed_task=lambda: True,
        enqueue_reindex=lambda: {"enqueued": 12, "versions": 3, "parts": 7},
    )
    # Bind the real implementation so the _is_local gate is what's tested.
    host._notify = lambda message, **kwargs: Keeper._notify(host, message, **kwargs)

    caplog.set_level(logging.INFO, logger="keep.api")
    with patch("keep.api.save_config"):
        Keeper._enqueue_embed_task_reindex(host)

    assert "Queued embedding task-type migration: 12 items, 3 versions, 7 parts" in caplog.text
    assert capsys.readouterr().err == ""
