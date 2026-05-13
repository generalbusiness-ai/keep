"""Tests for the watches module (daemon-driven source monitoring)."""

import os
import subprocess
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from keep.watches import (
    WatchEntry,
    parse_duration,
    load_watches,
    load_watches_lightweight,
    save_watches,
    add_watch,
    remove_watch,
    list_watches,
    has_active_watches,
    check_file,
    check_directory,
    check_url,
    format_watches,
    poll_watches,
    next_check_delay,
    _compute_walk_hash,
    _resolve_git_watch_state,
)


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "t@t.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "t@t.com",
        },
    )


# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------

class TestParseDuration:
    """Tests for duration string parsing."""

    def test_seconds(self):
        assert parse_duration("PT30S") == timedelta(seconds=30)

    def test_minutes(self):
        assert parse_duration("PT5M") == timedelta(minutes=5)

    def test_hours(self):
        assert parse_duration("PT1H") == timedelta(hours=1)

    def test_days(self):
        assert parse_duration("P7D") == timedelta(days=7)

    def test_combined(self):
        assert parse_duration("P1DT12H") == timedelta(days=1, hours=12)

    def test_case_insensitive(self):
        assert parse_duration("pt30s") == timedelta(seconds=30)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_duration("not-a-duration")

    def test_zero(self):
        with pytest.raises(ValueError):
            parse_duration("PT0S")


# ---------------------------------------------------------------------------
# WatchEntry.is_due
# ---------------------------------------------------------------------------

class TestWatchEntryIsDue:
    """Tests for watch entry due-check."""

    def test_never_checked(self):
        entry = WatchEntry(source="x", kind="file")
        assert entry.is_due()

    def test_not_yet_due(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        entry = WatchEntry(
            source="x", kind="file",
            last_checked=now.isoformat(),
            interval="PT30S",
        )
        assert not entry.is_due(now)

    def test_past_due(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        past = (now - timedelta(minutes=1)).isoformat()
        entry = WatchEntry(
            source="x", kind="file",
            last_checked=past,
            interval="PT30S",
        )
        assert entry.is_due(now)


# ---------------------------------------------------------------------------
# CRUD (requires Keeper with document store)
# ---------------------------------------------------------------------------

class TestWatchCRUD:
    """Tests for watch CRUD operations."""

    @pytest.fixture
    def kp(self, mock_providers, tmp_path):
        from keep.api import Keeper
        return Keeper(store_path=tmp_path)

    def test_empty_store(self, kp):
        assert list_watches(kp) == []

    def test_add_and_list(self, kp):
        entry = add_watch(kp, "file:///tmp/test.txt", "file")
        watches = list_watches(kp)
        assert len(watches) == 1
        assert watches[0].source == "file:///tmp/test.txt"
        assert watches[0].kind == "file"

    def test_add_duplicate_returns_existing(self, kp):
        add_watch(kp, "file:///tmp/test.txt", "file")
        entry = add_watch(kp, "file:///tmp/test.txt", "file")
        assert entry.source == "file:///tmp/test.txt"
        assert len(list_watches(kp)) == 1  # no duplicate created

    def test_update_interval(self, kp):
        add_watch(kp, "file:///tmp/test.txt", "file")
        entry = add_watch(kp, "file:///tmp/test.txt", "file", interval="PT1M")
        assert entry.interval == "PT1M"
        watches = list_watches(kp)
        assert len(watches) == 1
        assert watches[0].interval == "PT1M"

    def test_add_max_limit(self, kp):
        for i in range(3):
            add_watch(kp, f"file:///tmp/f{i}.txt", "file", max_watches=3)
        with pytest.raises(ValueError, match="Watch limit"):
            add_watch(kp, "file:///tmp/extra.txt", "file", max_watches=3)

    def test_remove(self, kp):
        add_watch(kp, "file:///tmp/test.txt", "file")
        assert remove_watch(kp, "file:///tmp/test.txt") is True
        assert list_watches(kp) == []

    def test_remove_nonexistent(self, kp):
        assert remove_watch(kp, "file:///tmp/nope.txt") is False

    def test_has_active_watches(self, kp, tmp_path):
        assert not has_active_watches(kp)
        f = tmp_path / "test.txt"
        f.write_text("hello")
        add_watch(kp, f"file://{f}", "file")
        assert has_active_watches(kp)

    def test_add_with_tags(self, kp):
        add_watch(kp, "file:///tmp/test.txt", "file", tags={"project": "docs"})
        watches = list_watches(kp)
        assert watches[0].tags == {"project": "docs"}

    def test_add_directory_with_recurse_and_exclude(self, kp, tmp_path):
        d = tmp_path / "mydir"
        d.mkdir()
        add_watch(kp, str(d), "directory", recurse=True, exclude=["*.log"])
        watches = list_watches(kp)
        assert watches[0].recurse is True
        assert watches[0].exclude == ["*.log"]

    def test_override_interval(self, kp):
        add_watch(kp, "https://example.com", "url", interval="PT5M")
        watches = list_watches(kp)
        assert len(watches) == 1
        assert watches[0].interval == "PT5M"

    def test_mixed_intervals(self, kp):
        add_watch(kp, "file:///tmp/a.txt", "file")  # default PT30S
        add_watch(kp, "https://example.com", "url", interval="PT5M")
        watches = list_watches(kp)
        assert len(watches) == 2
        intervals = {w.source: w.interval for w in watches}
        assert intervals["file:///tmp/a.txt"] == "PT30S"
        assert intervals["https://example.com"] == "PT5M"


class TestLightweightReader:
    """Watches read directly from documents.db.

    `load_watches_lightweight` reads watches directly from documents.db
    so `keep pending --list` doesn't need to spin up a full Keeper.
    These tests use a real on-disk DocumentStore — `mock_providers` would
    bypass the sqlite file the lightweight reader needs to query.
    """

    def test_no_db_returns_empty(self, tmp_path):
        # Store directory exists but documents.db has never been created.
        assert load_watches_lightweight(tmp_path) == []

    def test_reads_persisted_watches(self, tmp_path):
        from keep.const import DOCUMENTS_DB
        from keep.document_store import DocumentStore

        ds = DocumentStore(tmp_path / DOCUMENTS_DB)
        try:
            # save_watches only needs `keeper._document_store`; the lightweight
            # reader hits the same sqlite file on disk.
            class _DSHolder:
                _document_store = ds
            save_watches(
                _DSHolder(),
                [
                    WatchEntry(source="file:///tmp/a.txt", kind="file"),
                    WatchEntry(source="https://example.com", kind="url", interval="PT5M"),
                ],
            )
        finally:
            ds.close()

        entries = load_watches_lightweight(tmp_path)
        by_source = {e.source: e for e in entries}
        assert set(by_source.keys()) == {"file:///tmp/a.txt", "https://example.com"}
        assert by_source["https://example.com"].interval == "PT5M"
        assert by_source["file:///tmp/a.txt"].interval == "PT30S"


class TestFormatWatches:
    """Display lines for the `Watches:` section.

    `format_watches` renders one human-readable line per entry, used in
    the `Watches:` section of `keep pending --list`.
    """

    def test_renders_kind_source_and_interval(self):
        entry = WatchEntry(
            source="/Users/me/play/keep",
            kind="directory",
            interval="PT5M",
        )
        lines = format_watches([entry])
        assert len(lines) == 1
        assert "directory" in lines[0]
        assert "/Users/me/play/keep" in lines[0]
        assert "every 5m" in lines[0]
        # No last_checked → show "never" / "due now"
        assert "never" in lines[0]
        assert "due now" in lines[0]

    def test_renders_last_and_next(self):
        from datetime import datetime, timezone
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Last check 1 minute ago, every 5 minutes → next in 4m.
        entry = WatchEntry(
            source="file:///tmp/x.txt",
            kind="file",
            interval="PT5M",
            last_checked="2026-01-01T11:59:00+00:00",
        )
        line = format_watches([entry], now=now)[0]
        assert "last:" in line and "1m ago" in line
        assert "next:" in line and "in 4m" in line

class TestCheckFile:
    """Tests for file change detection."""

    def test_unchanged(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        st = f.stat()
        entry = WatchEntry(
            source=f"file://{f}",
            kind="file",
            mtime_ns=str(st.st_mtime_ns),
            file_size=str(st.st_size),
        )
        assert check_file(entry) is False

    def test_changed(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        st = f.stat()
        entry = WatchEntry(
            source=f"file://{f}",
            kind="file",
            mtime_ns=str(st.st_mtime_ns),
            file_size=str(st.st_size),
        )
        f.write_text("hello world")
        assert check_file(entry) is True
        # Fingerprint updated
        assert entry.mtime_ns != str(st.st_mtime_ns)

    def test_stale(self, tmp_path):
        entry = WatchEntry(
            source=f"file://{tmp_path / 'gone.txt'}",
            kind="file",
            mtime_ns="12345",
        )
        assert check_file(entry) is False
        assert entry.stale is True


# ---------------------------------------------------------------------------
# Change detection: directories
# ---------------------------------------------------------------------------

class TestCheckDirectory:
    """Tests for directory change detection."""

    def test_unchanged(self, tmp_path):
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "b.txt").write_text("B")
        walk_hash = _compute_walk_hash(tmp_path, recurse=False, exclude=None)
        entry = WatchEntry(
            source=str(tmp_path),
            kind="directory",
            walk_hash=walk_hash,
        )
        assert check_directory(entry) is False

    def test_changed_new_file(self, tmp_path):
        (tmp_path / "a.txt").write_text("A")
        walk_hash = _compute_walk_hash(tmp_path, recurse=False, exclude=None)
        entry = WatchEntry(
            source=str(tmp_path),
            kind="directory",
            walk_hash=walk_hash,
        )
        (tmp_path / "b.txt").write_text("B")
        assert check_directory(entry) is True

    def test_stale_dir_gone(self, tmp_path):
        d = tmp_path / "subdir"
        d.mkdir()
        entry = WatchEntry(source=str(d), kind="directory", walk_hash="old")
        d.rmdir()
        assert check_directory(entry) is False
        assert entry.stale is True

    def test_changed_git_head_without_worktree_change(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _run_git(repo, "init", "-b", "main")
        _run_git(repo, "config", "user.name", "Test")
        _run_git(repo, "config", "user.email", "t@t.com")
        (repo / "note.txt").write_text("stable\n")
        _run_git(repo, "add", ".")
        _run_git(repo, "commit", "-m", "Initial")

        walk_hash = _compute_walk_hash(repo, recurse=True, exclude=None)
        git_repo_root, git_dir, git_head = _resolve_git_watch_state(repo)
        entry = WatchEntry(
            source=str(repo),
            kind="directory",
            walk_hash=walk_hash,
            recurse=True,
            git_repo_root=git_repo_root,
            git_dir=git_dir,
            git_head=git_head,
        )

        _run_git(repo, "commit", "--allow-empty", "-m", "Empty commit")

        assert check_directory(entry) is True
        assert entry.git_head != git_head

    def test_resolves_gitdir_file_and_packed_refs(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        git_meta = tmp_path / "repo-meta"
        (git_meta / "refs").mkdir(parents=True)
        (repo / ".git").write_text(f"gitdir: {git_meta}\n")
        (git_meta / "HEAD").write_text("ref: refs/heads/main\n")
        (git_meta / "packed-refs").write_text(
            "# pack-refs with: peeled fully-peeled\n"
            "0123456789abcdef0123456789abcdef01234567 refs/heads/main\n"
        )

        git_repo_root, git_dir, git_head = _resolve_git_watch_state(repo)

        assert git_repo_root == str(repo.resolve())
        assert git_dir == str(git_meta.resolve())
        assert git_head == "0123456789abcdef0123456789abcdef01234567"


# ---------------------------------------------------------------------------
# Change detection: URLs
# ---------------------------------------------------------------------------

class TestCheckURL:
    """Tests for URL change detection."""

    def test_304_not_modified(self):
        entry = WatchEntry(
            source="https://example.com/doc",
            kind="url",
            etag='"abc123"',
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 304
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        with patch("keep.providers.http.http_session", return_value=mock_session):
            assert check_url(entry) is False

    def test_200_changed(self):
        entry = WatchEntry(
            source="https://example.com/doc",
            kind="url",
            etag='"abc123"',
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"ETag": '"def456"', "Last-Modified": "Mon, 17 Mar 2026"}
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        with patch("keep.providers.http.http_session", return_value=mock_session):
            assert check_url(entry) is True
            assert entry.etag == '"def456"'

    def test_404_stale(self):
        entry = WatchEntry(source="https://example.com/gone", kind="url")
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        with patch("keep.providers.http.http_session", return_value=mock_session):
            assert check_url(entry) is False
            assert entry.stale is True


# ---------------------------------------------------------------------------
# poll_watches integration
# ---------------------------------------------------------------------------

class TestPollWatches:
    """Tests for watch polling."""

    @pytest.fixture
    def kp(self, mock_providers, tmp_path):
        from keep.api import Keeper
        return Keeper(store_path=tmp_path)

    def test_poll_empty(self, kp):
        result = poll_watches(kp)
        assert result == {"checked": 0, "changed": 0, "stale": 0, "errors": 0}

    def test_poll_file_changed(self, kp, tmp_path):
        f = tmp_path / "watched.txt"
        f.write_text("original")
        add_watch(kp, f"file://{f}", "file")

        # Modify the file
        f.write_text("updated content")

        # Force the entry to be due (clear last_checked)
        entries = load_watches(kp)
        entries[0].last_checked = ""
        save_watches(kp, entries)

        result = poll_watches(kp)
        assert result["checked"] == 1
        assert result["changed"] == 1

    def test_poll_file_unchanged(self, kp, tmp_path):
        f = tmp_path / "watched.txt"
        f.write_text("stable")
        add_watch(kp, f"file://{f}", "file")

        # Force the entry to be due
        entries = load_watches(kp)
        entries[0].last_checked = ""
        save_watches(kp, entries)

        result = poll_watches(kp)
        assert result["checked"] == 1
        assert result["changed"] == 0

    def test_poll_stale_file(self, kp, tmp_path):
        f = tmp_path / "ephemeral.txt"
        f.write_text("here now")
        add_watch(kp, f"file://{f}", "file")
        f.unlink()

        entries = load_watches(kp)
        entries[0].last_checked = ""
        save_watches(kp, entries)

        result = poll_watches(kp)
        assert result["stale"] >= 1


# ---------------------------------------------------------------------------
# next_check_delay
# ---------------------------------------------------------------------------

class TestNextCheckDelay:
    """Tests for next-check delay calculation."""

    def test_empty(self):
        assert next_check_delay([]) == 30.0

    def test_never_checked(self):
        entry = WatchEntry(source="x", kind="file")
        assert next_check_delay([entry]) == 0.0

    def test_future(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        entry = WatchEntry(
            source="x", kind="file",
            last_checked=now.isoformat(),
            interval="PT30S",
        )
        delay = next_check_delay([entry])
        assert 0 < delay <= 30.0


# ---------------------------------------------------------------------------
# CLI integration: _handle_watch
# ---------------------------------------------------------------------------

class TestHandleWatch:
    """Tests for the CLI watch/unwatch wiring."""

    @pytest.fixture
    def kp(self, mock_providers, tmp_path):
        from keep.api import Keeper
        return Keeper(store_path=tmp_path)

    def test_watch_file(self, kp, tmp_path):
        from keep.console_support import _handle_watch
        f = tmp_path / "test.txt"
        f.write_text("hello")
        _handle_watch(kp, True, False, f"file://{f}", "file", {})
        watches = list_watches(kp)
        assert len(watches) == 1
        assert watches[0].kind == "file"

    def test_unwatch_file(self, kp, tmp_path):
        from keep.console_support import _handle_watch
        f = tmp_path / "test.txt"
        f.write_text("hello")
        _handle_watch(kp, True, False, f"file://{f}", "file", {})
        assert len(list_watches(kp)) == 1
        _handle_watch(kp, False, True, f"file://{f}", "file", {})
        assert len(list_watches(kp)) == 0

    def test_watch_with_interval(self, kp, tmp_path):
        from keep.console_support import _handle_watch
        f = tmp_path / "test.txt"
        f.write_text("hello")
        _handle_watch(kp, True, False, f"file://{f}", "file", {}, interval="PT5M")
        watches = list_watches(kp)
        assert watches[0].interval == "PT5M"

    def test_watch_directory_with_options(self, kp, tmp_path):
        from keep.console_support import _handle_watch
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.txt").write_text("A")
        _handle_watch(kp, True, False, str(d), "directory", {},
                      recurse=True, exclude=["*.log"])
        watches = list_watches(kp)
        assert watches[0].recurse is True
        assert watches[0].exclude == ["*.log"]

    def test_watch_noop_when_neither(self, kp):
        from keep.console_support import _handle_watch
        _handle_watch(kp, False, False, "file:///x", "file", {})
        assert list_watches(kp) == []

    def test_unwatch_nonexistent(self, kp):
        from keep.console_support import _handle_watch
        # Should not raise, just prints "Not watching"
        _handle_watch(kp, False, True, "file:///nope", "file", {})
        assert list_watches(kp) == []


# ---------------------------------------------------------------------------
# URL content hash fallback
# ---------------------------------------------------------------------------

class TestURLContentHashFallback:
    """Tests for URL content hash fallback."""

    def test_no_cache_headers_uses_content_hash(self):
        entry = WatchEntry(source="https://example.com/doc", kind="url")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}  # No ETag, no Last-Modified
        mock_resp.content = b"hello world"
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        with patch("keep.providers.http.http_session", return_value=mock_session):
            assert check_url(entry) is True
            first_hash = entry.walk_hash
            assert first_hash  # content hash was set

    def test_no_cache_headers_unchanged(self):
        import hashlib
        content = b"stable content"
        content_hash = hashlib.sha256(content).hexdigest()[:16]
        entry = WatchEntry(
            source="https://example.com/doc", kind="url",
            walk_hash=content_hash,
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.content = content
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        with patch("keep.providers.http.http_session", return_value=mock_session):
            assert check_url(entry) is False  # same content, no change
