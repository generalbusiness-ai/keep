"""Tests for the per-item analyze throttle.

A successful analyze records `_analyzed_at`; subsequent runs within
`KEEP_ANALYZE_MIN_INTERVAL_S` (default 300s) are skipped unless
`force=True`. The aim is to coalesce rapid edits (e.g. watched files
saved several times a minute) into a single decomposition run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from keep.actions.analyze import (
    DEFAULT_MIN_ANALYZE_INTERVAL_S,
    _min_analyze_interval_s,
    _params_force,
    _throttle_skip_reason,
)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


class TestThrottleSkipReason:
    def test_no_prior_analyze_does_not_throttle(self):
        assert _throttle_skip_reason({}) is None

    def test_recent_analyze_throttles(self):
        recent = datetime.now(timezone.utc) - timedelta(seconds=30)
        reason = _throttle_skip_reason({"_analyzed_at": _iso(recent)})
        assert reason is not None
        assert "throttled" in reason
        assert "30s ago" in reason

    def test_older_than_interval_does_not_throttle(self):
        long_ago = datetime.now(timezone.utc) - timedelta(
            seconds=DEFAULT_MIN_ANALYZE_INTERVAL_S + 60
        )
        assert _throttle_skip_reason({"_analyzed_at": _iso(long_ago)}) is None

    def test_unparseable_timestamp_does_not_throttle(self):
        # Garbage timestamps should fail open — running an extra analyze
        # is better than getting stuck never analyzing again.
        assert _throttle_skip_reason({"_analyzed_at": "not-a-date"}) is None

    def test_future_timestamp_does_not_throttle(self):
        # Clock skew between machines could yield a future timestamp;
        # don't pretend the throttle applies in that case.
        future = datetime.now(timezone.utc) + timedelta(minutes=5)
        assert _throttle_skip_reason({"_analyzed_at": _iso(future)}) is None

    def test_zero_interval_disables_throttle(self, monkeypatch):
        recent = datetime.now(timezone.utc) - timedelta(seconds=5)
        monkeypatch.setenv("KEEP_ANALYZE_MIN_INTERVAL_S", "0")
        assert _throttle_skip_reason({"_analyzed_at": _iso(recent)}) is None

    def test_env_overrides_default_interval(self, monkeypatch):
        # Bump the window so a timestamp from "an hour ago" still throttles.
        monkeypatch.setenv("KEEP_ANALYZE_MIN_INTERVAL_S", "7200")
        hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        assert _throttle_skip_reason({"_analyzed_at": _iso(hour_ago)}) is not None

    def test_bad_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("KEEP_ANALYZE_MIN_INTERVAL_S", "not-a-number")
        assert _min_analyze_interval_s() == DEFAULT_MIN_ANALYZE_INTERVAL_S


class TestForceParams:
    @pytest.mark.parametrize("raw,expected", [
        (True, True),
        (False, False),
        ("true", True),
        ("True", True),
        ("YES", True),
        ("1", True),
        ("0", False),
        ("no", False),
        (None, False),
    ])
    def test_force_param_truthiness(self, raw, expected):
        assert _params_force({"force": raw}) is expected

    def test_missing_force_is_false(self):
        assert _params_force({}) is False
