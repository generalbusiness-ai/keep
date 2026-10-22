"""Regression coverage for daemon cleanup in the test isolation fixtures."""

import os
from pathlib import Path
import shutil

import pytest

from tests import conftest


@pytest.mark.parametrize("scan", ["_find_processor_pid_files", "_cleanup_daemons_under"])
def test_daemon_cleanup_survives_disappearing_directory(tmp_path, monkeypatch, scan):
    """A vanished Git directory must not prevent cleanup of a surviving store."""
    disappearing = tmp_path / "git-objects"
    disappearing.mkdir()
    store = tmp_path / "surviving-store"
    store.mkdir()
    pid_file = store / "processor.pid"
    pid_file.write_text("12345")
    real_scandir = os.scandir
    removed = []
    killed = []

    def scandir(path):
        # Simulate Git packing objects after the parent directory was listed,
        # immediately before the traversal opens the now-stale child path.
        if not isinstance(path, int) and Path(path) == disappearing and not removed:
            shutil.rmtree(disappearing)
            removed.append(disappearing)
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(conftest, "_terminate_pid", killed.append)
    monkeypatch.setattr(conftest, "_find_pytest_daemon_pids", lambda root: set())
    result = getattr(conftest, scan)(tmp_path)

    assert removed == [disappearing]
    if scan == "_find_processor_pid_files":
        assert result == [pid_file]
        assert pid_file.exists()
    else:
        assert killed == [12345]
        assert not pid_file.exists()
