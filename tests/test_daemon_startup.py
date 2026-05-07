"""Tests for daemon startup sequencing and deferred maintenance."""

import sys
from runpy import run_path
from unittest.mock import MagicMock, patch

from keep.api import Keeper


def test_keeper_deferred_startup_skips_scans_until_started(mock_providers, tmp_path):
    calls: list[str] = []

    def fake_labeled_ref(self, doc_coll):
        calls.append("labeled-ref")
        return {"documents": 0, "versions": 0, "parts": 0}

    def fake_part_reindex(self):
        calls.append("part-reindex")

    def fake_marker(self, chroma_coll, doc_coll, *, _doc_store=None):
        calls.append("marker")

    def fake_check(self, *, _doc_store=None):
        calls.append("reconcile-check")
        return False

    with (
        patch.object(Keeper, "_run_labeled_ref_format_migration", fake_labeled_ref),
        patch.object(Keeper, "_enqueue_migrated_part_reindex", fake_part_reindex),
        patch.object(Keeper, "_run_tag_marker_startup_check", fake_marker),
        patch.object(Keeper, "_check_store_consistency", fake_check),
    ):
        kp = Keeper(store_path=tmp_path, defer_startup_maintenance=True)
        try:
            assert calls == []

            kp._run_deferred_startup_maintenance()

            assert calls == [
                "labeled-ref", "part-reindex", "marker", "reconcile-check",
            ]
        finally:
            kp.close()


def test_start_deferred_startup_maintenance_starts_once(mock_providers, tmp_path):
    with patch.object(Keeper, "_run_deferred_startup_maintenance", return_value=None) as runner:
        kp = Keeper(store_path=tmp_path, defer_startup_maintenance=True)
        try:
            assert kp.start_deferred_startup_maintenance() is True
            assert kp.start_deferred_startup_maintenance() is False
            assert kp._startup_maintenance_thread is not None
            kp._startup_maintenance_thread.join(timeout=2)
            runner.assert_called_once()
        finally:
            kp.close()


def test_daemon_entrypoint_uses_deferred_startup_maintenance(tmp_path):
    captured: dict[str, object] = {}
    daemon_run: dict[str, object] = {}

    class DummyKeeper:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    with (
        patch("keep.api.Keeper", DummyKeeper),
        patch("keep.console_support.run_pending_daemon", side_effect=lambda *args, **kwargs: daemon_run.update(kwargs)),
        patch.object(sys, "argv", ["python", "--store", str(tmp_path)]),
    ):
        from keep import daemon

        daemon.main()

    kwargs = captured["kwargs"]
    assert kwargs["store_path"] == str(tmp_path)
    assert kwargs["defer_startup_maintenance"] is True
    assert daemon_run["bind_host"] is None
    assert daemon_run["advertised_url"] is None
    assert daemon_run["trusted_proxy"] is False


def test_daemon_entrypoint_exits_before_keeper_when_processor_locked(tmp_path):
    constructed = False

    class DummyKeeper:
        def __init__(self, *args, **kwargs):
            nonlocal constructed
            constructed = True

    from keep.model_lock import ModelLock

    lock = ModelLock(tmp_path / ".processor.lock")
    assert lock.acquire(blocking=False) is True
    try:
        with (
            patch("keep.api.Keeper", DummyKeeper),
            patch("keep.console_support.run_pending_daemon") as run_pending,
            patch.object(sys, "argv", ["python", "--store", str(tmp_path)]),
        ):
            from keep import daemon

            daemon.main()

        assert constructed is False
        run_pending.assert_not_called()
    finally:
        lock.release()


def test_daemon_script_entrypoint_supports_direct_python_execution(tmp_path):
    captured: dict[str, object] = {}
    daemon_run: dict[str, object] = {}

    class DummyKeeper:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

    with (
        patch("keep.api.Keeper", DummyKeeper),
        patch("keep.console_support.run_pending_daemon", side_effect=lambda *args, **kwargs: daemon_run.update(kwargs)),
        patch.object(sys, "argv", ["python", "--store", str(tmp_path)]),
    ):
        run_path("keep/daemon.py", run_name="__main__")

    kwargs = captured["kwargs"]
    assert kwargs["store_path"] == str(tmp_path)
    assert kwargs["defer_startup_maintenance"] is True
    assert daemon_run["bind_host"] is None
    assert daemon_run["advertised_url"] is None
    assert daemon_run["trusted_proxy"] is False


def test_daemon_entrypoint_passes_remote_transport_options(tmp_path):
    daemon_run: dict[str, object] = {}

    class DummyKeeper:
        def __init__(self, *args, **kwargs):
            pass

    with (
        patch("keep.api.Keeper", DummyKeeper),
        patch("keep.console_support.run_pending_daemon", side_effect=lambda *args, **kwargs: daemon_run.update(kwargs)),
        patch.object(
            sys,
            "argv",
            [
                "python",
                "--store", str(tmp_path),
                "--bind", "0.0.0.0",
                "--advertised-url", "https://keep.example.test",
                "--trusted-proxy",
            ],
        ),
    ):
        from keep import daemon

        daemon.main()

    assert daemon_run["bind_host"] == "0.0.0.0"
    assert daemon_run["advertised_url"] == "https://keep.example.test"
    assert daemon_run["trusted_proxy"] is True


def test_daemon_startup_logs_markdown_mirror_count(mock_providers, tmp_path):
    kp = Keeper(store_path=tmp_path)
    logger = MagicMock()
    try:
        with patch("keep.markdown_mirrors.list_markdown_mirrors", return_value=[]):
            from keep.console_support import _log_daemon_startup_state

            _log_daemon_startup_state(kp, logger)

        logger.info.assert_any_call("Markdown mirrors: %d configured", 0)
    finally:
        kp.close()


def test_log_daemon_batch_result_skips_idle_tick():
    from keep.console_support import _log_daemon_batch_result

    logger = MagicMock()
    _log_daemon_batch_result(
        logger=logger,
        result={"processed": 0, "failed": 0},
        delegated=0,
        flow_result={"processed": 0, "failed": 0, "dead_lettered": 0},
    )

    logger.info.assert_not_called()


def test_log_daemon_batch_result_logs_activity():
    from keep.console_support import _log_daemon_batch_result

    logger = MagicMock()
    _log_daemon_batch_result(
        logger=logger,
        result={"processed": 1, "failed": 0},
        delegated=0,
        flow_result={"processed": 0, "failed": 0, "dead_lettered": 0},
    )

    logger.info.assert_called_once_with(
        "%s: processed=%d failed=%d delegated=%d flow_processed=%d flow_failed=%d",
        "Daemon batch",
        1,
        0,
        0,
        0,
        0,
    )


def test_daemon_only_failed_detects_provider_outage_tick():
    from keep.console_support import _daemon_only_failed

    assert _daemon_only_failed(
        result={"processed": 0, "failed": 0},
        delegated=0,
        flow_result={"processed": 0, "failed": 1, "dead_lettered": 0},
    )


def test_daemon_only_failed_ignores_mixed_progress():
    from keep.console_support import _daemon_only_failed

    assert not _daemon_only_failed(
        result={"processed": 1, "failed": 0},
        delegated=0,
        flow_result={"processed": 0, "failed": 1, "dead_lettered": 0},
    )


def test_daemon_failure_backoff_caps_under_thirty_seconds():
    from keep.console_support import _daemon_failure_backoff_seconds

    assert _daemon_failure_backoff_seconds(1) == 0.5
    assert _daemon_failure_backoff_seconds(2) == 1.0
    assert _daemon_failure_backoff_seconds(5) == 8.0
    assert _daemon_failure_backoff_seconds(20) == 8.0
