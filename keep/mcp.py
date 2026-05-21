"""MCP stdio server for keep — reflective memory for AI agents.

Thin HTTP wrapper over a keep backend. When [remote] is configured (via env
or keep.toml), MCP tool calls go directly to the hosted keep service over
HTTPS. Otherwise they route to the local daemon. Either way the MCP layer
holds no Keeper, no models, and no database of its own.

Three tools: keep_flow (all operations), keep_help (documentation),
keep_prompt (practice prompts).

Usage:
    keep mcp                        # stdio server (via CLI)
    claude --mcp-server keep="keep mcp"   # Claude Code integration
"""

import json
import logging
import os
import signal
import sys
import time
from typing import Annotated, Any, Optional
from urllib.parse import quote, unquote

from .const import STATE_PROMPT
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import (
    CallToolResult,
    ErrorData,
    GetPromptResult,
    INTERNAL_ERROR,
    Prompt as MCPPrompt,
    PromptArgument as MCPPromptArgument,
    PromptMessage,
    TextContent,
    ToolAnnotations,
)
from pydantic import BaseModel, ConfigDict, Field

from ._context_resolution import _SUPPORTED_MCP_PROMPT_ARGS
from .config import RemoteConfig
from .daemon_client import get_port, http_request_with_discovery_retry
from .help import get_help_topic
from .logging_config import configure_client_log

logger = logging.getLogger(__name__)

_MCP_PROMPT_ARG_DESCRIPTIONS: dict[str, str] = {
    "text": "Optional text or query used for prompt context.",
    "id": 'Optional note ID for context (default: "now").',
    "since": "Optional lower time bound for contextual search.",
    "token_budget": "Optional token budget for prompt-context rendering.",
}
assert set(_MCP_PROMPT_ARG_DESCRIPTIONS) == set(_SUPPORTED_MCP_PROMPT_ARGS), (
    f"MCP prompt arg descriptions {set(_MCP_PROMPT_ARG_DESCRIPTIONS)} != "
    f"supported args {set(_SUPPORTED_MCP_PROMPT_ARGS)}"
)


class _MCPBackend:
    """Routing abstraction: where do MCP tool calls go?"""

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        raise NotImplementedError

    def get(self, path: str) -> tuple[int, dict]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # Used by the daemon-mode entry point in main() for eager fail-fast.
    # Remote-mode backends override to a no-op.
    def warm_up(self) -> None:
        pass


class _DaemonBackend(_MCPBackend):
    """Default backend — talk to the local keep daemon (auto-discovered port)."""

    def __init__(self) -> None:
        self._port: Optional[int] = None

    def _ensure(self) -> int:
        if self._port is None:
            self._port = get_port(os.environ.get("KEEP_STORE_PATH"))
        return self._port

    def warm_up(self) -> None:
        # Surface daemon startup issues immediately rather than at first tool call.
        self._ensure()

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        status, result = http_request_with_discovery_retry(
            "POST", self._ensure(), path, body,
            store_override=os.environ.get("KEEP_STORE_PATH"),
        )
        if status == 401:
            # Daemon may have restarted on a new port. Re-resolve.
            self._port = None
            status, result = http_request_with_discovery_retry(
                "POST", self._ensure(), path, body,
                store_override=os.environ.get("KEEP_STORE_PATH"),
            )
        return status, result

    def get(self, path: str) -> tuple[int, dict]:
        status, result = http_request_with_discovery_retry(
            "GET", self._ensure(), path,
            store_override=os.environ.get("KEEP_STORE_PATH"),
        )
        if status == 401:
            self._port = None
            status, result = http_request_with_discovery_retry(
                "GET", self._ensure(), path,
                store_override=os.environ.get("KEEP_STORE_PATH"),
            )
        return status, result


class _RemoteBackend(_MCPBackend):
    """Remote backend — talk to the hosted keep service over HTTPS."""

    def __init__(self, remote: RemoteConfig, log_dir=None) -> None:
        import httpx
        from .remote import validate_project_slug, validate_remote_api_url
        from .types import user_agent

        # Same URL/project guards as RemoteKeeper — never send bearer tokens
        # over plain HTTP, and reject malformed project slugs upfront.
        self._api_url = validate_remote_api_url(remote.api_url)
        project = validate_project_slug(remote.project)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {remote.api_key}",
            "User-Agent": user_agent(),
        }
        if project:
            headers["X-Project"] = project
        self._client = httpx.Client(
            base_url=self._api_url, headers=headers, timeout=30.0,
        )
        # Attach the client-side ops log so MCP tool traffic produces an
        # on-disk audit trail in remote mode. Failures are non-fatal.
        self._log_handler = None
        if log_dir is not None:
            try:
                self._log_handler = configure_client_log(log_dir)
            except OSError as e:
                logger.debug("Could not attach client log at %s: %s", log_dir, e)

    def _log_call(self, method: str, path: str, status: int, wall_ms: int) -> None:
        logger.info(
            "mcp.remote: %s %s status=%d wall=%dms host=%s",
            method, path, status, wall_ms, self._api_url,
        )

    def _do(self, method: str, path: str, body: Optional[dict] = None) -> tuple[int, dict]:
        import httpx
        start = time.monotonic()
        try:
            if body is None:
                resp = self._client.request(method, path)
            else:
                resp = self._client.request(method, path, json=body)
        except httpx.HTTPError as e:
            wall_ms = int((time.monotonic() - start) * 1000)
            self._log_call(method, path, 0, wall_ms)
            return 0, {"error": str(e)}
        wall_ms = int((time.monotonic() - start) * 1000)
        self._log_call(method, path, resp.status_code, wall_ms)
        try:
            data = resp.json()
            if not isinstance(data, dict):
                data = {"value": data}
        except ValueError:
            data = {"error": resp.text}
        return resp.status_code, data

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        return self._do("POST", path, body)

    def get(self, path: str) -> tuple[int, dict]:
        return self._do("GET", path)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
        if self._log_handler is not None:
            try:
                logging.getLogger("keep").removeHandler(self._log_handler)
                self._log_handler.close()
            except Exception:
                pass
            self._log_handler = None


def _load_remote_config() -> tuple[Optional[RemoteConfig], Optional[Any]]:
    """Detect remote-backend config from env vars layered over keep.toml.

    Returns (remote_config_or_None, config_dir_for_logging). The config_dir
    is the directory containing keep.toml (used for the client log location).

    Resolution order, per field (api_url, api_key, project):
        env var  >  keep.toml [remote]  >  bundled default (api_url only)

    Honors KEEP_LOCAL_ONLY so tests and constrained environments stay on the
    daemon backend.
    """
    if os.environ.get("KEEP_LOCAL_ONLY"):
        return None, None

    # Always try to load TOML first so config-owned api_url/project still
    # apply when only the api_key lives in the environment.
    toml_remote: Optional[RemoteConfig] = None
    toml_config_dir = None
    try:
        from .config import load_config
        config_dir = _resolve_config_dir()
        config = load_config(config_dir)
        if config.remote_persist:
            toml_remote = config.remote_persist
        toml_config_dir = config.config_dir or config_dir
    except (FileNotFoundError, ValueError):
        toml_config_dir = _resolve_config_dir()

    env_url = os.environ.get("KEEPNOTES_API_URL")
    env_key = os.environ.get("KEEPNOTES_API_KEY")
    env_project = os.environ.get("KEEPNOTES_PROJECT")

    # Overlay env on TOML, field by field.
    api_url = env_url or (toml_remote.api_url if toml_remote else None) \
        or "https://api.keepnotes.ai"
    api_key = env_key or (toml_remote.api_key if toml_remote else None)
    project = env_project or (toml_remote.project if toml_remote else None)

    if not api_key:
        return None, None
    return RemoteConfig(api_url=api_url, api_key=api_key, project=project or None), \
        toml_config_dir


def _resolve_config_dir():
    """Locate the config directory keep would use for this process.

    Mirrors console_support._get_keeper's resolution order so a user invoking
    ``keep --store /path mcp`` (which sets KEEP_STORE_PATH but not KEEP_CONFIG)
    still picks up that store's keep.toml.
    """
    from .paths import get_config_dir
    if os.environ.get("KEEP_CONFIG"):
        return get_config_dir()
    store_override = os.environ.get("KEEP_STORE_PATH")
    if store_override:
        from pathlib import Path
        return Path(store_override).resolve()
    return get_config_dir()


_backend: Optional[_MCPBackend] = None


def _get_backend() -> _MCPBackend:
    """Return the cached backend (daemon or remote, decided at first call)."""
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


def _list_agent_prompt_metadata(*, suppress_errors: bool = False) -> list[dict[str, Any]]:
    """Return agent prompt metadata from the daemon's prompt flow."""
    try:
        status, resp = _post("/v1/flow", {"state": STATE_PROMPT, "params": {"list": True}})
    except Exception as exc:
        if suppress_errors:
            logger.warning("MCP prompt discovery unavailable: %s", exc)
            return []
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"keep daemon unavailable: {exc}", data=None)
        ) from exc
    if status != 200:
        error = str(resp.get("error", "unknown"))
        if suppress_errors:
            logger.warning("MCP prompt discovery failed: %s", error)
            return []
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"keep daemon unavailable: {error}", data=None)
        )
    prompts = (resp.get("data") or {}).get("prompts", [])
    if not isinstance(prompts, list):
        return []
    return [prompt for prompt in prompts if isinstance(prompt, dict)]


def _normalize_optional_arg(value: Any) -> Any:
    """Treat blank-string optional MCP arguments as absent."""
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _describe_keep_prompt_tool() -> str:
    """Build a tool description that lists currently available prompts."""
    prompts = _list_agent_prompt_metadata(suppress_errors=True)
    if not prompts:
        return (
            "Render an agent prompt with context injected from memory. "
            "Call with no name to list available prompts."
        )

    prompt_names = [str(prompt.get("name", "")).strip() for prompt in prompts]
    prompt_names = [name for name in prompt_names if name]
    names_text = ", ".join(prompt_names) if prompt_names else "none"
    return (
        "Render an agent prompt with context injected from memory. "
        f"Available prompts: {names_text}. "
        "Call with no name to return the full list."
    )


class KeepFastMCP(FastMCP):
    """FastMCP server with prompt exposure sourced dynamically from keep."""

    async def list_tools(self):
        tools = await super().list_tools()
        updated = []
        for tool in tools:
            if tool.name == "keep_prompt":
                updated.append(tool.model_copy(update={"description": _describe_keep_prompt_tool()}))
            else:
                updated.append(tool)
        return updated

    async def list_prompts(self) -> list[MCPPrompt]:
        prompts = _list_agent_prompt_metadata(suppress_errors=True)
        result: list[MCPPrompt] = []
        for prompt in prompts:
            args = prompt.get("mcp_arguments") or []
            if not isinstance(args, list) or not args:
                continue
            result.append(
                MCPPrompt(
                    name=str(prompt.get("name", "")),
                    description=str(prompt.get("summary", "") or ""),
                    arguments=[
                        MCPPromptArgument(
                            name=arg,
                            description=_MCP_PROMPT_ARG_DESCRIPTIONS.get(arg),
                            required=False,
                        )
                        for arg in args
                        if isinstance(arg, str)
                    ],
                )
            )
        return result

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> GetPromptResult:
        flow_params: dict[str, Any] = {"name": name}
        for arg in _SUPPORTED_MCP_PROMPT_ARGS:
            value = _normalize_optional_arg((arguments or {}).get(arg))
            if value is not None:
                # MCP prompt arguments are always strings; coerce numerics early.
                if arg == "token_budget":
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        continue
                flow_params[arg] = value

        status, resp = _post("/v1/flow", {"state": STATE_PROMPT, "params": flow_params})
        if status != 200:
            raise ValueError(resp.get("error", "unknown"))

        if resp.get("status") == "error":
            error = (resp.get("data") or {}).get("error", f"prompt not found: {name}")
            raise ValueError(str(error))

        text = str((resp.get("data") or {}).get("text", ""))
        return GetPromptResult(
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=text),
                )
            ],
        )


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = KeepFastMCP(
    "keep",
    instructions=(
        "Reflective memory with semantic search. "
        "Store facts, preferences, decisions, and documents. "
        "Search by meaning. Persist context across sessions."
    ),
)


def _read_note_resource(note_id: str) -> dict[str, Any]:
    """Read a note as the canonical note JSON payload."""
    status, resp = _get(f"/v1/notes/{quote(note_id, safe='')}")
    if status == 404:
        raise ValueError(f"note not found: {note_id}")
    if status != 200:
        raise ValueError(str(resp.get("error", "unknown")))
    return resp


@mcp.resource(
    "keep://now",
    name="now",
    title="Current Note",
    description="Current working note as JSON.",
    mime_type="application/json",
)
def keep_now_resource() -> dict[str, Any]:
    return _read_note_resource("now")


@mcp.resource(
    "keep://{id}",
    name="note",
    title="Keep Note",
    description=(
        "Read a keep note as JSON. Examples: keep://now, "
        "keep://meeting-notes, "
        "keep://file%3A%2F%2F%2FUsers%2Fhugh%2Fnotes.md, "
        "keep://https%3A%2F%2Fexample.com%2Fdoc"
    ),
    mime_type="application/json",
)
def keep_note_resource(id: str) -> dict[str, Any]:
    return _read_note_resource(unquote(id))


# ---------------------------------------------------------------------------
# Tool annotations
# ---------------------------------------------------------------------------

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False)
_IDEMPOTENT = ToolAnnotations(idempotentHint=True, destructiveHint=False)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class FlowParams(BaseModel):
    """Common flow parameters exposed explicitly in the MCP schema.

    Extra keys remain allowed so custom state docs and less-common built-ins
    still work without schema churn.
    """

    model_config = ConfigDict(extra="allow")

    id: Annotated[Optional[str], Field(
        description="Generic target note ID. Used by operations like put, tag, delete, and move.",
    )] = None
    item_id: Annotated[Optional[str], Field(
        description='Note ID for read flows like get. Use "now" for current working context.',
    )] = None
    name: Annotated[Optional[str], Field(
        description="Target name/ID for move-like flows.",
    )] = None
    source_id: Annotated[Optional[str], Field(
        description='Source note ID for move-like flows. Defaults to "now" when omitted.',
    )] = None
    content: Annotated[Optional[str], Field(
        description="Inline text content to store or update.",
    )] = None
    uri: Annotated[Optional[str], Field(
        description="URI to ingest, such as file://, https://, or http://.",
    )] = None
    summary: Annotated[Optional[str], Field(
        description="Optional summary override for put-like flows.",
    )] = None
    tags: Annotated[Optional[dict[str, str | list[str]]], Field(
        description="Tag filter or tag updates, depending on the flow.",
    )] = None
    query: Annotated[Optional[str], Field(
        description="Natural-language search query.",
    )] = None
    similar_to: Annotated[Optional[str], Field(
        description="Find items similar to this note ID.",
    )] = None
    prefix: Annotated[Optional[str], Field(
        description='ID prefix or glob, for example ".tag/*".',
    )] = None
    scope: Annotated[Optional[str], Field(
        description='ID glob to constrain search results, for example "file:///path/to/dir*".',
    )] = None
    since: Annotated[Optional[str], Field(
        description="Only include notes updated since this date/duration.",
    )] = None
    until: Annotated[Optional[str], Field(
        description="Only include notes updated before this date/duration.",
    )] = None
    limit: Annotated[Optional[int], Field(
        description="Maximum number of results to return.",
    )] = None
    token_budget: Annotated[Optional[int], Field(
        description="Token budget for rendered text output.",
    )] = None
    deep: Annotated[Optional[bool], Field(
        description="Follow tags/edges to discover related notes.",
    )] = None
    include_hidden: Annotated[Optional[bool], Field(
        description="Include system notes in results.",
    )] = None
    include_meta: Annotated[Optional[bool], Field(
        description="Include meta sections during context assembly.",
    )] = None
    include_parts: Annotated[Optional[bool], Field(
        description="Include structural parts during context assembly.",
    )] = None
    include_similar: Annotated[Optional[bool], Field(
        description="Include similar notes during context assembly.",
    )] = None
    include_versions: Annotated[Optional[bool], Field(
        description="Include version navigation during context assembly.",
    )] = None
    only_current: Annotated[Optional[bool], Field(
        description="Move or operate on only the current version when supported.",
    )] = None
    analyze: Annotated[Optional[bool], Field(
        description="Analyze the note into structural parts when supported.",
    )] = None
    bias: Annotated[Optional[dict[str, float]], Field(
        description='Per-item score weighting, for example {"now": 0}.',
    )] = None


class PromptSummary(BaseModel):
    """Structured summary for one exposed agent prompt."""

    name: str
    summary: str = ""


class KeepPromptStructured(BaseModel):
    """Structured output for the keep_prompt tool."""

    mode: str
    prompts: list[PromptSummary] | None = None
    name: str | None = None
    text: str | None = None
    error: str | None = None


@mcp.tool(
    description=(
        "Execute a keep operation via state-doc flow. "
        "Examples:\n"
        '  Search: state="query-resolve", params={"query": "auth patterns"}\n'
        '  Get context: state="get", params={"item_id": "now"}\n'
        '  Store text: state="put", params={"content": "decision: use JWT", "tags": {"project": "auth"}}\n'
        '  Store with ID: state="put", params={"id": "meeting-notes", "content": "..."}\n'
        '  Store file: state="put", params={"uri": "file:///path/to/doc.md"}\n'
        '  Store URL: state="put", params={"uri": "https://example.com/article"}\n'
        '  List items: state="list", params={"prefix": ".tag/", "include_hidden": true}\n'
        '  Resume stopped search: state="query-resolve", cursor="<cursor from previous call>"\n'
        "When status is 'stopped', pass the returned cursor to continue. "
        "Set token_budget for rendered text output instead of raw JSON. "
        'List available flows: keep_help(topic="flow_state_docs").'
    ),
    annotations=_IDEMPOTENT,
)
async def keep_flow(
    state: Annotated[str, Field(
        description="State doc name (e.g. 'query-resolve', 'get', 'put', 'tag', 'delete', 'move', 'stats').",
    )],
    params: Annotated[Optional[FlowParams], Field(
        description=(
            "Flow parameters as a JSON object. Do not pass YAML or a plain string. "
            'Examples: {"item_id": "now"}, {"query": "auth patterns"}, '
            '{"content": "decision: use JWT", "tags": {"project": "auth"}}.'
        ),
        examples=[
            {"item_id": "now"},
            {"query": "auth patterns", "tags": {"project": "myapp"}},
            {"content": "decision: use JWT", "tags": {"project": "auth"}},
        ],
    )] = None,
    budget: Annotated[Optional[int], Field(
        description="Max ticks for this invocation (default: from config).",
    )] = None,
    cursor: Annotated[Optional[str], Field(
        description="Cursor from a previous stopped flow to resume.",
    )] = None,
    state_doc_yaml: Annotated[Optional[str], Field(
        description="Inline YAML state doc (instead of loading from store).",
    )] = None,
    token_budget: Annotated[Optional[int], Field(
        description="Token budget for rendering results (default: raw JSON).",
    )] = None,
) -> str:
    """Run a state-doc flow."""
    if isinstance(params, BaseModel):
        params_body = params.model_dump(exclude_none=True)
    else:
        params_body = params
    body: dict = {
        "state": state,
        "params": params_body,
        "budget": budget,
        "cursor": cursor,
        "state_doc_yaml": state_doc_yaml,
    }
    if token_budget and token_budget > 0:
        body["token_budget"] = token_budget
    status, resp = _post("/v1/flow", body)
    if status != 200:
        return f"Error: {resp.get('error', 'unknown')}"

    # Server-side rendered output (when token_budget was provided)
    if resp.get("rendered"):
        return resp["rendered"]

    # Build selective JSON (same shape as before — no bindings/history)
    output: dict[str, Any] = {
        "status": resp.get("status"),
        "ticks": resp.get("ticks"),
    }
    data = resp.get("data")
    if data:
        output["data"] = data
    cursor_val = resp.get("cursor")
    if cursor_val:
        output["cursor"] = cursor_val
    tried = resp.get("tried_queries")
    if tried:
        output["tried_queries"] = tried
    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Render an agent prompt with context injected from memory. "
        "Returns actionable instructions for reflection, session start, etc. "
        "Call with no name to list available prompts."
    ),
    annotations=_READ_ONLY,
)
async def keep_prompt(
    name: Annotated[Optional[str], Field(
        description='Prompt name (e.g. "reflect", "session-start"). Omit to list available prompts.',
    )] = None,
    text: Annotated[Optional[str], Field(
        description="Optional search query for additional context injection.",
    )] = None,
    id: Annotated[Optional[str], Field(
        description='Item ID for context (default: "now").',
    )] = None,
    tags: Annotated[Optional[dict[str, str | list[str]]], Field(
        description="Filter search context by tags.",
    )] = None,
    since: Annotated[Optional[str], Field(
        description="Only include items updated since this value (ISO duration or date).",
    )] = None,
    until: Annotated[Optional[str], Field(
        description="Only include items updated before this value (ISO duration or date).",
    )] = None,
    deep: Annotated[bool, Field(
        description="Follow tags from results to discover related items.",
    )] = False,
    scope: Annotated[Optional[str], Field(
        description="ID glob to constrain search results (e.g. 'file:///path/to/dir*').",
    )] = None,
    token_budget: Annotated[Optional[int], Field(
        description="Token budget for search results context (template default if not set).",
    )] = None,
) -> Annotated[CallToolResult, KeepPromptStructured]:
    """Render an agent prompt with injected context."""
    flow_params: dict[str, Any] = {}
    if not _normalize_optional_arg(name):
        flow_params["list"] = True
    else:
        flow_params["name"] = name
        for key, val in [
            ("text", text), ("id", id), ("since", since),
            ("until", until), ("scope", scope),
        ]:
            normalized = _normalize_optional_arg(val)
            if normalized is not None:
                flow_params[key] = normalized
        if tags:
            flow_params["tags"] = tags
        if deep:
            flow_params["deep"] = deep
        if token_budget:
            flow_params["token_budget"] = token_budget

    status, resp = _post("/v1/flow", {"state": STATE_PROMPT, "params": flow_params})
    if status != 200:
        error = f"Error: {resp.get('error', 'unknown')}"
        return CallToolResult(
            content=[TextContent(type="text", text=error)],
            structuredContent={"mode": "error", "error": error},
            isError=True,
        )

    flow_data = resp.get("data", {})
    flow_status = resp.get("status")

    # List mode
    if not name:
        prompts = flow_data.get("prompts", [])
        if not prompts:
            return CallToolResult(
                content=[TextContent(type="text", text="No agent prompts available.")],
                structuredContent={"mode": "list", "prompts": []},
            )
        prompt_rows = [
            {
                "name": str(p.get("name", "")),
                "summary": str(p.get("summary", "") or ""),
            }
            for p in prompts
            if isinstance(p, dict)
        ]
        lines = [f"Available prompts ({len(prompt_rows)}):"]
        lines.extend(
            f"- {row['name']}: {row['summary']}".rstrip()
            for row in prompt_rows
        )
        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(lines))],
            structuredContent={"mode": "list", "prompts": prompt_rows},
        )

    # Error
    if flow_status == "error":
        error = str(flow_data.get("error", f"prompt not found: {name}"))
        return CallToolResult(
            content=[TextContent(type="text", text=error)],
            structuredContent={"mode": "render", "name": name, "error": error},
            isError=True,
        )

    # Render mode — daemon already expanded the prompt
    text_out = str(flow_data.get("text", ""))
    return CallToolResult(
        content=[TextContent(type="text", text=text_out)],
        structuredContent={"mode": "render", "name": name, "text": text_out},
    )


# ---------------------------------------------------------------------------
# Help / documentation
# ---------------------------------------------------------------------------

@mcp.tool(
    description=(
        "Comprehensive keep documentation with examples for all commands, "
        "flows, tagging, prompts, and architecture. "
        "Call with topic=\"index\" to see all available guides."
    ),
    annotations=_READ_ONLY,
)
async def keep_help(
    topic: Annotated[str, Field(
        description='Documentation topic, e.g. "index", "quickstart", "keep-put", "tagging". '
                    'Use "index" to see all available topics.',
    )] = "index",
) -> str:
    return get_help_topic(topic, link_style="mcp")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the MCP stdio server."""
    # anyio's stdin reader uses abandon_on_cancel=False, which shields the
    # blocking readline from task cancellation.  The first Ctrl+C only cancels
    # the task (which can't take effect), so install our own handler.
    # Use os._exit to avoid SystemExit during interpreter shutdown, which
    # can deadlock on the stdin buffer lock held by the reader thread.
    signal.signal(signal.SIGINT, lambda *_: os._exit(130))

    # Eagerly warm up the backend so setup issues surface immediately:
    # daemon mode resolves the port; remote mode is a no-op (the first tool
    # call covers it). Failures in remote mode would emerge on first use.
    _get_backend().warm_up()

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
