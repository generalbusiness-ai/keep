"""Remote Keeper — HTTP client for a flow-hosted keep backend.

Used by the CLI to talk to the local daemon, and by the hosted
keep.generalbusiness.ai service. The stable interface is ``run_flow()``; higher-
level helpers delegate through the shared flow client layer.
"""

import json
import logging
import os
import re
import time
from typing import Any, Optional
from urllib.parse import quote

import httpx

from .config import StoreConfig
from .logging_config import configure_client_log
from .flow_client import (
    delete_item as flow_delete_item,
    find_items as flow_find_items,
    get_item as flow_get_item,
    get_now_item as flow_get_now_item,
    move_item as flow_move_item,
    put_item as flow_put_item,
    set_now_item as flow_set_now_item,
    tag_item as flow_tag_item,
)
from .types import (
    Item, ItemContext, SimilarRef, MetaRef, EdgeRef, VersionRef, PartRef,
    TagMap, local_date,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0

_SLUG_RE = re.compile(r'^[a-z][a-z0-9-]{0,61}[a-z0-9]$')


def validate_project_slug(project: Optional[str]) -> Optional[str]:
    """Validate a project slug. Returns the slug unchanged or None.

    Shared by RemoteKeeper and the MCP _RemoteBackend so both reject the
    same set of malformed slugs before any HTTP request goes out.
    """
    if not project:
        return None
    if not _SLUG_RE.match(project):
        raise ValueError(
            f"Invalid project slug '{project}'. "
            "Must start with a letter, 2-63 chars, lowercase letters/numbers/hyphens."
        )
    return project


def validate_remote_api_url(api_url: str) -> str:
    """Ensure a remote API URL is safe to send bearer tokens to.

    Only HTTPS, or http loopback (localhost/127.0.0.1/::1) for local dev, is
    accepted. Trailing slashes are stripped. Raises ValueError otherwise.
    """
    url = (api_url or "").rstrip("/")
    if not url.startswith("https://"):
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        if host not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                f"Remote API URL must use HTTPS (got {url}). "
                "Use HTTPS to protect API credentials, or use localhost for local development."
            )
    return url


def resolve_remote_config(config: Optional[StoreConfig]) -> Optional["RemoteConfig"]:
    """Compute the effective RemoteConfig given a (possibly loaded) StoreConfig.

    Single source of truth for the env-over-TOML overlay rule used by every
    caller that needs to decide "do we route this process through the hosted
    backend, and with what credentials?". Resolution order, per field:

        api_url    KEEPNOTES_API_URL    > config.remote_persist.api_url    > https://api.keepnotes.ai
        api_key    KEEPNOTES_API_KEY    > config.remote_persist.api_key    (required)
        project    KEEPNOTES_PROJECT    > config.remote_persist.project

    The function returns None when ``KEEP_LOCAL_ONLY`` is set in the
    environment or when no ``api_key`` is available from either source.

    Preferring ``config.remote_persist`` (the on-disk view) over the
    already-overlaid ``config.remote`` means callers who pre-loaded a
    StoreConfig still get the same field-by-field merge — and a caller that
    constructs a StoreConfig by hand (wizard, console_support fallback)
    still resolves correctly as long as it sets ``remote_persist``.
    """
    # Local import to avoid a circular dependency: keep.config imports from
    # keep.remote at module import time would create a cycle since remote.py
    # already imports StoreConfig at module scope.
    from .config import RemoteConfig

    if os.environ.get("KEEP_LOCAL_ONLY"):
        return None

    toml_remote: Optional[RemoteConfig] = None
    if config is not None:
        toml_remote = config.remote_persist or config.remote

    env_url = os.environ.get("KEEPNOTES_API_URL")
    env_key = os.environ.get("KEEPNOTES_API_KEY")
    env_project = os.environ.get("KEEPNOTES_PROJECT")

    api_url = (
        env_url
        or (toml_remote.api_url if toml_remote else None)
        or "https://api.keepnotes.ai"
    )
    api_key = env_key or (toml_remote.api_key if toml_remote else None)
    project = env_project or (toml_remote.project if toml_remote else None)

    if not api_key:
        return None
    return RemoteConfig(api_url=api_url, api_key=api_key, project=project or None)
    return url


class RemoteKeeper:
    """Flow-host client backed by the keep HTTP API."""

    def __init__(self, api_url: str, api_key: str, config: StoreConfig, *, project: Optional[str] = None):
        self.api_url = validate_remote_api_url(api_url)
        self.api_key = api_key
        self._config = config
        self.config = config  # alias for CLI compatibility

        self.project = validate_project_slug(
            project
            or (config.remote.project if config.remote else None)
            or os.environ.get("KEEPNOTES_PROJECT")
            or None
        )

        from .types import user_agent
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": user_agent(),
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if self.project:
            headers["X-Project"] = self.project

        self._client = httpx.Client(
            base_url=self.api_url, headers=headers, timeout=DEFAULT_TIMEOUT)
        self._server_info_cache: dict[str, Any] | None = None

        # Attach a client-side ops log so CLI/MCP traffic over the remote
        # backend leaves an on-disk audit trail. Use the config_dir when set
        # (config_dir is where keep.toml lives and is always writable), else
        # fall back to the store path. Failures are non-fatal — clients that
        # disable filesystem access (tests, MCPB sandboxing) still work.
        self._client_log_handler = None
        log_dir = (
            config.config_dir if config and config.config_dir
            else (config.path if config and config.path else None)
        )
        if log_dir is not None:
            try:
                self._client_log_handler = configure_client_log(log_dir)
            except OSError as e:
                logger.debug("Could not attach client log at %s: %s", log_dir, e)

    # -- HTTP helpers --

    @staticmethod
    def _q(id: str) -> str:
        return quote(id, safe="")

    def _log_call(self, method: str, path: str, status: int, wall_ms: int) -> None:
        """Emit a single-line INFO record per remote HTTP call."""
        logger.info(
            "remote: %s %s status=%d wall=%dms host=%s",
            method, path, status, wall_ms, self.api_url,
        )

    def _get(self, path: str, **params: Any) -> dict:
        filtered = {k: v for k, v in params.items() if v is not None}
        start = time.monotonic()
        resp = self._client.get(path, params=filtered)
        self._log_call("GET", path, resp.status_code, int((time.monotonic() - start) * 1000))
        self._raise_for_status(resp)
        return resp.json()

    def _post(self, path: str, json: dict) -> dict:
        filtered = {k: v for k, v in json.items() if v is not None}
        start = time.monotonic()
        resp = self._client.post(path, json=filtered)
        self._log_call("POST", path, resp.status_code, int((time.monotonic() - start) * 1000))
        self._raise_for_status(resp)
        return resp.json()

    def _patch(self, path: str, json: dict) -> dict:
        start = time.monotonic()
        resp = self._client.patch(path, json=json)
        self._log_call("PATCH", path, resp.status_code, int((time.monotonic() - start) * 1000))
        self._raise_for_status(resp)
        return resp.json()

    def _delete(self, path: str) -> dict:
        start = time.monotonic()
        resp = self._client.delete(path)
        self._log_call("DELETE", path, resp.status_code, int((time.monotonic() - start) * 1000))
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _daemon_error_detail(resp: httpx.Response) -> str:
        """Extract daemon error text and request_id without exposing bodies."""
        try:
            if not resp.is_closed and not resp.is_stream_consumed:
                resp.read()
            data = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, httpx.ResponseNotRead):
            return ""
        if not isinstance(data, dict):
            return ""
        error = data.get("error")
        request_id = data.get("request_id")
        parts: list[str] = []
        if error:
            parts.append(str(error))
        if request_id:
            parts.append(f"request_id={request_id}")
        return " ".join(parts)

    @classmethod
    def _raise_for_status(cls, resp: httpx.Response) -> None:
        """Raise HTTP errors with daemon request_id preserved in the message."""
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = cls._daemon_error_detail(resp)
            if not detail:
                raise
            raise httpx.HTTPStatusError(
                f"{exc} ({detail})",
                request=exc.request,
                response=exc.response,
            ) from exc

    @staticmethod
    def _to_item(data: dict) -> Item:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        item_id = data.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"Missing 'id' in response: {data!r:.200}")
        tags = data.get("tags", {})
        if not isinstance(tags, dict):
            tags = {}
        tags = {str(k): str(v) for k, v in tags.items()}
        if data.get("created_at"):
            tags.setdefault("_created", str(data["created_at"]))
        if data.get("updated_at"):
            tags.setdefault("_updated", str(data["updated_at"]))
        summary = data.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary)
        score = data.get("score")
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = None
        return Item(id=item_id, summary=summary, tags=tags, score=score)

    @staticmethod
    def _to_items(data: dict) -> list[Item]:
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        items = data.get("notes", data.get("items", []))
        if not isinstance(items, list):
            raise ValueError(f"Expected list, got {type(items).__name__}")
        return [RemoteKeeper._to_item(i) for i in items]

    def get(self, id: str) -> Optional[Item]:
        return flow_get_item(self, id)

    def export_iter(self, *, include_system: bool = True):
        with self._client.stream(
            "GET",
            "/v1/export",
            params={
                "include_system": str(include_system).lower(),
                "stream": "ndjson",
            },
        ) as resp:
            self._raise_for_status(resp)
            content_type = (resp.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
            if content_type == "application/x-ndjson":
                for line in resp.iter_lines():
                    if not line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        yield row
                return

            data = resp.json()
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict export payload, got {type(data).__name__}")
            header = {
                "format": data.get("format"),
                "version": data.get("version"),
                "exported_at": data.get("exported_at"),
                "store_info": data.get("store_info", {}),
            }
            yield header
            for doc in data.get("documents", []):
                if isinstance(doc, dict):
                    yield doc

    def export_data(self, *, include_system: bool = True) -> dict:
        it = self.export_iter(include_system=include_system)
        header = next(it)
        if not isinstance(header, dict):
            raise ValueError(f"Expected dict export header, got {type(header).__name__}")
        header["documents"] = list(it)
        return header

    def export_bundle(
        self,
        id: str,
        *,
        include_system: bool = True,
        include_parts: bool = True,
        include_versions: bool = True,
    ) -> dict | None:
        resp = self._client.get(
            f"/v1/export/bundles/{self._q(id)}",
            params={
                "include_system": str(include_system).lower(),
                "include_parts": str(include_parts).lower(),
                "include_versions": str(include_versions).lower(),
            },
        )
        if resp.status_code == 404:
            return None
        self._raise_for_status(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict export bundle payload, got {type(data).__name__}")
        return data

    def export_changes(
        self,
        *,
        cursor: str | None = None,
        limit: int = 1000,
    ) -> dict:
        resp = self._client.get(
            "/v1/export/changes",
            params={
                "cursor": cursor or "0",
                "limit": str(limit),
            },
        )
        self._raise_for_status(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict export changes payload, got {type(data).__name__}")
        return data

    def put(
        self,
        content: Optional[str] = None,
        *,
        uri: Optional[str] = None,
        id: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[TagMap] = None,
        created_at: Optional[str] = None,
        force: bool = False,
    ) -> Item:
        return flow_put_item(
            self,
            content,
            uri=uri,
            id=id,
            summary=summary,
            tags=tags,
            created_at=created_at,
            force=force,
        )

    def delete(self, id: str, *, delete_versions: bool = True) -> bool:
        return flow_delete_item(self, id, delete_versions=delete_versions)

    def find(
        self,
        query: Optional[str] = None,
        *,
        tags: Optional[TagMap] = None,
        similar_to: Optional[str] = None,
        limit: int = 10,
        since: Optional[str] = None,
        until: Optional[str] = None,
        include_self: bool = False,
        include_hidden: bool = False,
        deep: bool = False,
        scope: Optional[str] = None,
    ) -> list[Item]:
        return flow_find_items(
            self,
            query,
            tags=tags,
            similar_to=similar_to,
            limit=limit,
            since=since,
            until=until,
            include_self=include_self,
            include_hidden=include_hidden,
            deep=deep,
            scope=scope,
        )

    def tag(self, id: str, tags: Optional[TagMap] = None) -> Optional[Item]:
        return flow_tag_item(self, id, tags)

    def run_flow(
        self,
        state: str,
        *,
        params: Optional[dict[str, Any]] = None,
        budget: Optional[int] = None,
        cursor_token: Optional[str] = None,
        state_doc_yaml: Optional[str] = None,
        writable: bool = True,
    ) -> Any:
        from .state_doc_runtime import FlowResult
        resp = self._post("/v1/flow", json={
            "state": state, "params": params, "budget": budget,
            "cursor_token": cursor_token, "state_doc_yaml": state_doc_yaml,
            "writable": writable,
        })
        return FlowResult(
            status=resp.get("status", "error"),
            bindings=resp.get("bindings", {}),
            data=resp.get("data"),
            ticks=resp.get("ticks", 0),
            history=resp.get("history", []),
            cursor=resp.get("cursor"),
        )

    def run_flow_command(
        self,
        state: str,
        *,
        params: Optional[dict[str, Any]] = None,
        budget: Optional[int] = None,
        cursor_token: Optional[str] = None,
        state_doc_yaml: Optional[str] = None,
        writable: bool = True,
    ) -> Any:
        return self.run_flow(
            state,
            params=params,
            budget=budget,
            cursor_token=cursor_token,
            state_doc_yaml=state_doc_yaml,
            writable=writable,
        )

    # ---- Flow-based convenience methods ----

    def get_context(
        self,
        id: str,
        *,
        version: int | None = None,
        similar_limit: int = 3,
        meta_limit: int = 3,
        parts_limit: int = 10,
        edges_limit: int = 5,
        versions_limit: int = 3,
        include_similar: bool = True,
        include_meta: bool = True,
        include_parts: bool = True,
        include_versions: bool = True,
    ) -> ItemContext | None:
        """Assemble display context via the /context endpoint."""
        params: dict[str, str | int] = {
            "similar_limit": similar_limit if include_similar else 0,
            "meta_limit": meta_limit if include_meta else 0,
            "parts_limit": parts_limit if include_parts else 0,
            "edges_limit": edges_limit,
            "versions_limit": versions_limit if include_versions else 0,
            "include_similar": str(include_similar).lower(),
            "include_meta": str(include_meta).lower(),
            "include_parts": str(include_parts).lower(),
            "include_versions": str(include_versions).lower(),
        }
        if version is not None:
            params["version"] = version
        resp = self._client.get(f"/v1/notes/{self._q(id)}/context", params=params)
        if resp.status_code == 404:
            return None
        self._raise_for_status(resp)
        return ItemContext.from_dict(resp.json())

    def get_now(self, *, scope: Optional[str] = None) -> Item:
        return flow_get_now_item(self, scope=scope)

    def set_now(self, content: str, *, scope: Optional[str] = None, tags: Optional[TagMap] = None) -> Item:
        return flow_set_now_item(self, content, scope=scope, tags=tags)

    def move(
        self,
        name: str,
        *,
        source_id: str = "now",
        tags: Optional[TagMap] = None,
        only_current: bool = False,
    ) -> Item:
        return flow_move_item(
            self,
            name,
            source_id=source_id,
            tags=tags,
            only_current=only_current,
        )

    def exists(self, id: str) -> bool:
        return self.get(id) is not None

    def count(self) -> int:
        try:
            resp = self._client.get("/v1/health")
            if resp.status_code == 200:
                return resp.json().get("item_count", 0)
        except Exception:
            pass
        return 0

    def server_info(self, *, refresh: bool = False) -> dict[str, Any]:
        if self._server_info_cache is not None and not refresh:
            return dict(self._server_info_cache)
        resp = self._client.get("/v1/ready")
        self._raise_for_status(resp)
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict server info payload, got {type(data).__name__}")
        self._server_info_cache = dict(data)
        return dict(self._server_info_cache)

    def capabilities(self, *, refresh: bool = False) -> dict[str, Any]:
        info = self.server_info(refresh=refresh)
        caps = info.get("capabilities")
        return dict(caps) if isinstance(caps, dict) else {}

    def supports_capability(self, name: str, *, refresh: bool = False) -> bool:
        return bool(self.capabilities(refresh=refresh).get(name))

    def close(self) -> None:
        self._client.close()
        if self._client_log_handler is not None:
            try:
                logging.getLogger("keep").removeHandler(self._client_log_handler)
                self._client_log_handler.close()
            except Exception:
                pass
            self._client_log_handler = None
