"""Regression tests for CLI logging configuration."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import sys
import warnings

import httpx

from keep.logging_config import configure_quiet_mode, enable_debug_mode


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _preserved_logging_state(*logger_names: str):
    root = logging.getLogger()
    saved_root = {
        "level": root.level,
        "handlers": list(root.handlers),
        "filters": list(root.filters),
        "disabled": root.disabled,
    }
    saved_loggers = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        saved_loggers[name] = {
            "level": logger.level,
            "handlers": list(logger.handlers),
            "filters": list(logger.filters),
            "propagate": logger.propagate,
            "disabled": logger.disabled,
        }
    tracked_env = (
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "TRANSFORMERS_VERBOSITY",
        "TOKENIZERS_PARALLELISM",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_DISABLE_SYMLINKS_WARNING",
    )
    saved_env = {name: os.environ.get(name) for name in tracked_env}
    saved_warnings = list(warnings.filters)

    try:
        yield
    finally:
        root.setLevel(saved_root["level"])
        root.handlers[:] = saved_root["handlers"]
        root.filters[:] = saved_root["filters"]
        root.disabled = saved_root["disabled"]
        for name, state in saved_loggers.items():
            logger = logging.getLogger(name)
            logger.setLevel(state["level"])
            logger.handlers[:] = state["handlers"]
            logger.filters[:] = state["filters"]
            logger.propagate = state["propagate"]
            logger.disabled = state["disabled"]
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        warnings.filters[:] = saved_warnings


def _emit_httpx_request_log() -> None:
    """Exercise the real httpx request-summary logger."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, request=request)
    )
    with httpx.Client(transport=transport) as client:
        client.get("https://api.keepnotes.ai/v1/export")


def test_quiet_mode_suppresses_httpx_request_info_logs() -> None:
    with _preserved_logging_state("httpx", "httpcore"):
        handler = _CollectingHandler()
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(logging.INFO)

        configure_quiet_mode(quiet=True)
        _emit_httpx_request_log()

    assert handler.records == []


def test_debug_mode_demotes_httpx_request_info_logs_to_debug() -> None:
    with _preserved_logging_state("httpx", "httpcore"):
        handler = _CollectingHandler()
        quiet_stderr_handler = logging.StreamHandler(sys.stderr)
        quiet_stderr_handler.setLevel(logging.CRITICAL + 1)
        root = logging.getLogger()
        root.handlers[:] = [handler, quiet_stderr_handler]

        enable_debug_mode()
        _emit_httpx_request_log()

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.name == "httpx"
    assert record.levelno == logging.DEBUG
    assert record.levelname == "DEBUG"
    assert record.getMessage().startswith("HTTP Request: GET https://api.keepnotes.ai")
