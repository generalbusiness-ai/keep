"""Logging configuration for keep.

Suppress verbose library output by default for better UX.
"""

import os
import sys
import warnings
import logging

from .const import (
    CLIENT_LOG_FILE,
    OPS_LOG_BACKUP_COUNT,
    OPS_LOG_FILE,
    OPS_LOG_MAX_BYTES,
)

# Set environment variables BEFORE any imports to suppress warnings early
if not os.environ.get("KEEP_VERBOSE"):
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class _HttpxRequestInfoDemoter(logging.Filter):
    """Treat httpx request summaries as DEBUG diagnostics in verbose mode."""

    _PREFIX = "HTTP Request:"

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            record.name == "httpx"
            and record.levelno == logging.INFO
            and str(record.msg).startswith(self._PREFIX)
        ):
            record.levelno = logging.DEBUG
            record.levelname = logging.getLevelName(logging.DEBUG)
        return True


_HTTPX_REQUEST_INFO_DEMOTER = _HttpxRequestInfoDemoter()


def _install_httpx_request_info_demoter() -> None:
    """Install the httpx request-summary demoter once per process."""
    httpx_logger = logging.getLogger("httpx")
    if not any(isinstance(f, _HttpxRequestInfoDemoter) for f in httpx_logger.filters):
        httpx_logger.addFilter(_HTTPX_REQUEST_INFO_DEMOTER)


def configure_quiet_mode(quiet: bool = True):
    """Configure logging to suppress verbose library output.

    This silences:
    - HuggingFace transformers progress bars
    - MLX model loading messages
    - Library warnings (deprecation, etc.)

    Args:
        quiet: If True, suppress verbose output. If False, show everything.
    """
    if quiet:
        _install_httpx_request_info_demoter()

        # Suppress HuggingFace progress bars and warnings
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Suppress Python warnings (including deprecation warnings)
        warnings.filterwarnings("ignore")

        # Configure Python logging to be less verbose
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("mlx").setLevel(logging.ERROR)
        logging.getLogger("chromadb").setLevel(logging.ERROR)
        # httpx emits one INFO record per completed request.  During remote
        # exports that can interleave with progress-bar rendering, so keep
        # third-party HTTP chatter out of normal CLI output.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


def enable_debug_mode():
    """Enable debug-level logging to stderr."""
    _install_httpx_request_info_demoter()

    # Re-enable warnings
    warnings.filterwarnings("default")

    # Restore library verbosity
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    os.environ.pop("TRANSFORMERS_VERBOSITY", None)

    # Configure root logger for debug output
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Add stderr handler if not already present
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stderr
               for h in root_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        root_logger.addHandler(handler)

    # Set library loggers to DEBUG
    for name in (
        "keep",
        "transformers",
        "sentence_transformers",
        "mlx",
        "chromadb",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(name).setLevel(logging.DEBUG)


def _attach_rotating_handler(log_path):
    """Attach a rotating INFO file handler at log_path to the keep logger."""
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(path),
        maxBytes=OPS_LOG_MAX_BYTES,
        backupCount=OPS_LOG_BACKUP_COUNT,
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    keep_logger = logging.getLogger("keep")
    keep_logger.addHandler(handler)
    # Ensure keep logger allows INFO through even in quiet mode
    if keep_logger.level == logging.NOTSET or keep_logger.level > logging.INFO:
        keep_logger.setLevel(logging.INFO)

    return handler


def configure_ops_log(store_path):
    """Configure a persistent operations log for the keep daemon/local Keeper.

    Writes to {store_path}/keep-ops.log using a rotating file handler
    (1MB max, 3 backups). Always active regardless of --verbose.
    Returns the handler so it can be removed on close().
    """
    from pathlib import Path
    return _attach_rotating_handler(Path(store_path) / OPS_LOG_FILE)


def configure_client_log(log_dir):
    """Configure a client-side operations log for RemoteKeeper/MCP-remote.

    Writes to {log_dir}/keep-client.log. Used when calls bypass the local
    daemon (e.g. when [remote] is configured) so the CLI/MCP process still
    produces an on-disk audit trail. Returns the handler so callers can
    detach it on shutdown.
    """
    from pathlib import Path
    return _attach_rotating_handler(Path(log_dir) / CLIENT_LOG_FILE)


class ClientCallLogger:
    """Shared remote-call logging lifecycle for RemoteKeeper and _RemoteBackend.

    Owns three concerns so neither client needs to duplicate them:
    - handler attach (rotating file at {log_dir}/keep-client.log)
    - a single standardised INFO line per HTTP call, including optional request_id
    - handler detach on close

    The emitted line format is::

        remote: <METHOD> <path> status=<n> wall=<ms>ms host=<url>[ request_id=<id>]

    The ``request_id=`` field is appended only when a non-empty value is
    supplied, so existing log assertions that don't expect it remain valid.

    This class must NOT import keep.remote or keep.mcp — it lives in
    logging_config to avoid circular imports.
    """

    def __init__(self, log_dir, prefix: str, host: str) -> None:
        """Attach the rotating handler and record the log-line prefix and host.

        Args:
            log_dir: Directory in which to create keep-client.log, or None to
                     skip handler attachment (tests, sandboxed environments).
            prefix:  Token that precedes the METHOD in each log line, e.g.
                     ``"remote"`` or ``"mcp.remote"``.
            host:    The API base URL emitted as the ``host=`` field.
        """
        self._prefix = prefix
        self._host = host
        self._handler = None
        if log_dir is not None:
            try:
                self._handler = configure_client_log(log_dir)
            except OSError as e:
                logging.getLogger("keep").debug(
                    "Could not attach client log at %s: %s", log_dir, e
                )

    @property
    def handler(self):
        """The attached RotatingFileHandler, or None if not attached."""
        return self._handler

    def log_call(
        self,
        method: str,
        path: str,
        status: int,
        wall_ms: int,
        request_id: str = "",
    ) -> None:
        """Emit one INFO line for a completed remote HTTP call.

        Args:
            method:     HTTP verb (GET, POST, PATCH, DELETE, …).
            path:       Request path.
            status:     HTTP status code (0 for transport errors).
            wall_ms:    Elapsed wall-clock milliseconds.
            request_id: Optional daemon request correlation id.  When non-empty
                        it is appended as ``request_id=<value>``.
        """
        if request_id:
            logging.getLogger("keep").info(
                "%s: %s %s status=%d wall=%dms host=%s request_id=%s",
                self._prefix, method, path, status, wall_ms, self._host, request_id,
            )
        else:
            logging.getLogger("keep").info(
                "%s: %s %s status=%d wall=%dms host=%s",
                self._prefix, method, path, status, wall_ms, self._host,
            )

    def close(self) -> None:
        """Detach and close the rotating handler."""
        if self._handler is not None:
            try:
                logging.getLogger("keep").removeHandler(self._handler)
                self._handler.close()
            except Exception:
                pass
            self._handler = None
