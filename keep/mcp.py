"""MCP stdio adapter for keep's shared reflective-memory surface.

When a remote service is configured, MCP calls use its REST API. Otherwise
they use the local daemon. The protocol contract itself lives in
``keep.mcp_surface`` so hosted and local deployments cannot drift.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from functools import partial
from typing import Any, Optional
from urllib.parse import quote

import anyio
from mcp.server.mcpserver.utilities.logging import configure_logging

from . import __version__
from .config import RemoteConfig
from .daemon_client import get_port, http_request_with_discovery_retry
from .help import get_help_topic
from .logging_config import ClientCallLogger
from .mcp_surface import FlowResponse, KeepMCPService, create_keep_mcp_surface

logger = logging.getLogger(__name__)


class _MCPBackend:
    """Routing abstraction for the REST service behind the MCP adapter."""

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        raise NotImplementedError

    def get(self, path: str) -> tuple[int, dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def warm_up(self) -> None:
        """Fail fast before stdio starts when a local daemon cannot start."""


class _DaemonBackend(_MCPBackend):
    """Default backend: call the local daemon on its discovered port."""

    def __init__(self) -> None:
        self._port: Optional[int] = None

    def _ensure(self) -> int:
        if self._port is None:
            self._port = get_port(os.environ.get("KEEP_STORE_PATH"))
        return self._port

    def warm_up(self) -> None:
        self._ensure()

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        status, result = http_request_with_discovery_retry(
            "POST",
            self._ensure(),
            path,
            body,
            store_override=os.environ.get("KEEP_STORE_PATH"),
        )
        if status == 401:
            # A restarted daemon may have moved to a new authenticated port.
            self._port = None
            status, result = http_request_with_discovery_retry(
                "POST",
                self._ensure(),
                path,
                body,
                store_override=os.environ.get("KEEP_STORE_PATH"),
            )
        return status, result

    def get(self, path: str) -> tuple[int, dict]:
        status, result = http_request_with_discovery_retry(
            "GET",
            self._ensure(),
            path,
            store_override=os.environ.get("KEEP_STORE_PATH"),
        )
        if status == 401:
            self._port = None
            status, result = http_request_with_discovery_retry(
                "GET",
                self._ensure(),
                path,
                store_override=os.environ.get("KEEP_STORE_PATH"),
            )
        return status, result


class _RemoteBackend(_MCPBackend):
    """Remote backend: call the hosted REST service over authenticated HTTPS."""

    def __init__(self, remote: RemoteConfig, log_dir=None) -> None:
        import httpx

        from .remote import validate_project_slug, validate_remote_api_url
        from .types import user_agent

        self._api_url = validate_remote_api_url(remote.api_url)
        project = validate_project_slug(remote.project)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {remote.api_key}",
            "User-Agent": user_agent(),
        }
        if project:
            headers["X-Project"] = project
        self._client = httpx.Client(base_url=self._api_url, headers=headers, timeout=30.0)
        self._call_logger = ClientCallLogger(log_dir, "mcp.remote", self._api_url)

    def _do(self, method: str, path: str, body: Optional[dict] = None) -> tuple[int, dict]:
        import httpx

        start = time.monotonic()
        try:
            if body is None:
                response = self._client.request(method, path)
            else:
                response = self._client.request(method, path, json=body)
        except httpx.HTTPError as exc:
            wall_ms = int((time.monotonic() - start) * 1000)
            self._call_logger.log_call(method, path, 0, wall_ms)
            return 0, {"error": str(exc)}

        wall_ms = int((time.monotonic() - start) * 1000)
        try:
            data = response.json()
            if not isinstance(data, dict):
                data = {"value": data}
        except ValueError:
            data = {"error": response.text}
        self._call_logger.log_call(
            method,
            path,
            response.status_code,
            wall_ms,
            request_id=data.get("request_id", ""),
        )
        return response.status_code, data

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        return self._do("POST", path, body)

    def get(self, path: str) -> tuple[int, dict]:
        return self._do("GET", path)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
        self._call_logger.close()


def _resolve_config_dir():
    """Locate configuration using the same order as ordinary CLI commands."""
    from .paths import get_config_dir

    if os.environ.get("KEEP_CONFIG"):
        return get_config_dir()
    store_override = os.environ.get("KEEP_STORE_PATH")
    if store_override:
        from pathlib import Path

        return Path(store_override).resolve()
    return get_config_dir()


def _load_remote_config() -> tuple[Optional[RemoteConfig], Optional[Any]]:
    """Return the configured remote backend and client-log directory."""
    from .remote import resolve_remote_config

    config = None
    config_dir = _resolve_config_dir()
    try:
        from .config import load_config

        config = load_config(config_dir)
    except (FileNotFoundError, ValueError):
        pass

    remote = resolve_remote_config(config)
    if remote is None:
        return None, None
    log_dir = (config.config_dir if config is not None else None) or config_dir
    return remote, log_dir


_backend: Optional[_MCPBackend] = None


def _get_backend() -> _MCPBackend:
    """Return the cached daemon or hosted REST backend."""
    global _backend
    if _backend is not None:
        return _backend
    remote, log_dir = _load_remote_config()
    if remote:
        _backend = _RemoteBackend(remote, log_dir=log_dir)
    else:
        _backend = _DaemonBackend()
    return _backend


def _post(path: str, body: dict) -> tuple[int, dict]:
    return _get_backend().post(path, body)


def _get(path: str) -> tuple[int, dict]:
    return _get_backend().get(path)


class _LocalMCPService(KeepMCPService):
    """Adapt REST operations to the shared semantic MCP service interface.

    Existing REST clients are synchronous. Worker threads prevent those calls
    from blocking MCP dispatch, and ``abandon_on_cancel`` lets a cancelled MCP
    request stop waiting while the daemon remains the owner of durable work.
    """

    async def execute_flow(
        self,
        state: str,
        params: dict[str, Any] | None,
        *,
        budget: int | None = None,
        cursor: str | None = None,
        state_doc_yaml: str | None = None,
        token_budget: int | None = None,
    ) -> FlowResponse:
        body: dict[str, Any] = {
            "state": state,
            "params": params,
            "budget": budget,
            "cursor": cursor,
            "state_doc_yaml": state_doc_yaml,
        }
        if token_budget is not None and token_budget > 0:
            body["token_budget"] = token_budget
        status, payload = await anyio.to_thread.run_sync(
            _post,
            "/v1/flow",
            body,
            abandon_on_cancel=True,
        )
        return FlowResponse(status_code=status, payload=payload)

    async def read_note(self, note_id: str) -> dict[str, Any] | None:
        status, payload = await anyio.to_thread.run_sync(
            _get,
            f"/v1/notes/{quote(note_id, safe='')}",
            abandon_on_cancel=True,
        )
        if status == 404:
            return None
        if status != 200:
            raise ValueError(str(payload.get("error", "keep service unavailable")))
        return payload

    async def get_help(self, topic: str) -> str:
        return await anyio.to_thread.run_sync(
            partial(get_help_topic, topic, link_style="mcp"),
            abandon_on_cancel=True,
        )


# MCPServer configures root logging on construction. The CLI imports this
# module while registering commands, so restore the caller's logging state and
# configure the server process explicitly in ``main``.
_root_logger = logging.getLogger()
_saved_handlers = list(_root_logger.handlers)
_saved_level = _root_logger.level

_surface = create_keep_mcp_surface(_LocalMCPService(), version=__version__)
mcp = _surface.server
keep_flow = _surface.keep_flow
keep_prompt = _surface.keep_prompt
keep_help = _surface.keep_help

_root_logger.handlers[:] = _saved_handlers
_root_logger.setLevel(_saved_level)
del _root_logger, _saved_handlers, _saved_level


def main() -> None:
    """Run the dual-era MCP server over stdio."""
    # AnyIO's stdin reader shields blocking readline from cancellation. Exit
    # directly on Ctrl+C so a stopped host cannot deadlock during shutdown.
    signal.signal(signal.SIGINT, lambda *_: os._exit(130))
    configure_logging(mcp.settings.log_level)
    _get_backend().warm_up()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
