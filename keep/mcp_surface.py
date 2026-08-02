"""Shared, transport-neutral MCP surface for local and hosted keep servers.

The protocol contract belongs in the public package so every deployment
advertises the same tools, prompts, resources, schemas, annotations, and error
semantics.  Transport, authentication, quotas, and storage remain adapter
concerns supplied through :class:`KeepMCPService`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Annotated, Any, Awaitable, Callable, Optional, Protocol
from urllib.parse import unquote

from mcp import MCPError
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    Prompt as MCPPrompt,
    PromptArgument as MCPPromptArgument,
    PromptMessage,
    TextContent,
    ToolAnnotations,
)
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from ._context_resolution import _SUPPORTED_MCP_PROMPT_ARGS
from .const import STATE_PROMPT

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


@dataclass(frozen=True)
class FlowResponse:
    """Transport-independent response from keep's state-doc flow service."""

    status_code: int
    payload: dict[str, Any]


class KeepMCPService(Protocol):
    """Semantic operations required by the shared MCP contract.

    Implementations may call the local daemon, a hosted Keeper, or another
    service.  They are responsible for deployment-specific authorization and
    quotas before returning a flow response.
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
    ) -> FlowResponse: ...

    async def read_note(self, note_id: str) -> dict[str, Any] | None: ...

    async def get_help(self, topic: str) -> str: ...


class FlowParams(BaseModel):
    """Common flow parameters exposed explicitly in the MCP schema.

    Extra keys remain allowed so custom state docs and less-common built-ins
    work without schema churn.  ``source_id`` remains an accepted input alias
    for older callers, but the generated schema advertises canonical
    ``source`` because that is what the move state consumes.
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
    source: Annotated[Optional[str], Field(
        description='Source note ID for move-like flows. Defaults to "now" when omitted.',
        validation_alias=AliasChoices("source", "source_id"),
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
        description="Find notes similar to this note ID.",
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
        description="Maximum number of results when supported.",
    )] = None
    top_k: Annotated[Optional[int], Field(
        description="Maximum number of ranked results or statistics entries.",
    )] = None
    token_budget: Annotated[Optional[int], Field(
        description="Token budget for rendered text output within a flow.",
    )] = None
    deep: Annotated[Optional[bool], Field(
        description="Follow tags and graph edges to discover related notes.",
    )] = None
    include_hidden: Annotated[Optional[bool], Field(
        description="Include hidden/system notes when supported.",
    )] = None
    include_edges: Annotated[Optional[bool], Field(
        description="Include graph edges during context assembly.",
    )] = None
    include_meta: Annotated[Optional[bool], Field(
        description="Include metadata sections during context assembly.",
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
        description='Per-note score weighting, for example {"now": 0}.',
    )] = None


class PromptSummary(BaseModel):
    """Structured summary for one exposed agent prompt."""

    name: str
    summary: str = ""


class KeepPromptStructured(BaseModel):
    """Structured output for the ``keep_prompt`` tool."""

    mode: str
    prompts: list[PromptSummary] | None = None
    name: str | None = None
    text: str | None = None
    error: str | None = None


class KeepFlowStructured(BaseModel):
    """Stable envelope around the deliberately flexible flow result data."""

    status: str | None = None
    ticks: int | None = None
    data: Any = None
    cursor: str | None = None
    tried_queries: list[Any] | None = None
    rendered: str | None = None


_READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
)
_MAY_MUTATE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
)


class MCPHTTPHeaderNormalizer:
    """Apply RFC 9110 optional-whitespace parsing at the MCP ASGI edge.

    MCP SDK 2.0.0 validates modern routing headers before dispatch but compares
    their raw ASGI byte values. HTTP field parsing excludes leading/trailing
    spaces and tabs, so normalize only MCP routing/custom-header values while
    preserving header order and duplicates for the SDK's ambiguity checks.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        normalized_scope = dict(scope)
        normalized_headers: list[tuple[bytes, bytes]] = []
        for name, value in scope.get("headers", []):
            lower_name = name.lower()
            if lower_name in {
                b"mcp-protocol-version",
                b"mcp-method",
                b"mcp-name",
            } or lower_name.startswith(b"mcp-param-"):
                value = value.strip(b" \t")
            normalized_headers.append((name, value))
        normalized_scope["headers"] = normalized_headers
        await self.app(normalized_scope, receive, send)


def _normalize_optional_arg(value: Any) -> Any:
    """Treat blank-string optional MCP arguments as absent."""
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


class KeepMCPServer(MCPServer):
    """MCPServer whose dynamic catalogs are supplied by a keep service."""

    def __init__(self, service: KeepMCPService, **kwargs: Any) -> None:
        self.keep_service = service
        super().__init__(**kwargs)
        # MCP SDK 2.0.0 installs ``subscriptions/listen`` unconditionally and
        # therefore advertises listChanged/resource-subscribe capabilities.
        # Keep's catalogs live behind an external daemon/service, so this
        # process cannot truthfully publish every mutation. The SDK has no
        # public opt-out yet; remove its optional handler until keep has a
        # daemon-owned cross-process event stream to back that promise.
        self._lowlevel_server._request_handlers.pop("subscriptions/listen", None)

    def streamable_http_app(self, **kwargs: Any) -> ASGIApp:
        """Return the SDK transport behind keep's HTTP parsing boundary."""
        return MCPHTTPHeaderNormalizer(super().streamable_http_app(**kwargs))

    async def _prompt_metadata(self, *, suppress_errors: bool = False) -> list[dict[str, Any]]:
        try:
            response = await self.keep_service.execute_flow(STATE_PROMPT, {"list": True})
        except Exception as exc:
            if suppress_errors:
                logger.warning("MCP prompt discovery unavailable: %s", exc)
                return []
            raise MCPError(INTERNAL_ERROR, f"keep service unavailable: {exc}") from exc

        if response.status_code != 200:
            error = str(response.payload.get("error", "unknown"))
            if suppress_errors:
                logger.warning("MCP prompt discovery failed: %s", error)
                return []
            raise MCPError(INTERNAL_ERROR, f"keep service unavailable: {error}")

        prompts = (response.payload.get("data") or {}).get("prompts", [])
        if not isinstance(prompts, list):
            return []
        return [prompt for prompt in prompts if isinstance(prompt, dict)]

    async def list_tools(self):
        """Keep the prompt tool description synchronized with the store catalog."""
        tools = await super().list_tools()
        prompts = await self._prompt_metadata(suppress_errors=True)
        prompt_names = [str(prompt.get("name", "")).strip() for prompt in prompts]
        prompt_names = [name for name in prompt_names if name]
        if prompt_names:
            description = (
                "Render an agent prompt with context injected from memory. "
                f"Available prompts: {', '.join(prompt_names)}. "
                "Call with no name to return the full list."
            )
        else:
            description = (
                "Render an agent prompt with context injected from memory. "
                "Call with no name to list available prompts."
            )
        return [
            tool.model_copy(update={"description": description})
            if tool.name == "keep_prompt"
            else tool
            for tool in tools
        ]

    async def list_prompts(self) -> list[MCPPrompt]:
        """Expose selected store prompts as native MCP prompts."""
        prompts = await self._prompt_metadata(suppress_errors=True)
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
        context: Context | None = None,
    ) -> GetPromptResult:
        """Render a native MCP prompt through the same state-doc flow as the tool."""
        del context  # The keep flow is request-scoped and needs no client back-channel.
        flow_params: dict[str, Any] = {"name": name}
        for arg in _SUPPORTED_MCP_PROMPT_ARGS:
            value = _normalize_optional_arg((arguments or {}).get(arg))
            if value is not None:
                if arg == "token_budget":
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        continue
                flow_params[arg] = value

        response = await self.keep_service.execute_flow(STATE_PROMPT, flow_params)
        if response.status_code != 200:
            raise MCPError(
                INTERNAL_ERROR,
                str(response.payload.get("error", "keep service unavailable")),
            )
        if response.payload.get("status") == "error":
            error = (response.payload.get("data") or {}).get("error", f"prompt not found: {name}")
            raise MCPError(INVALID_PARAMS, str(error), {"name": name})

        text = str((response.payload.get("data") or {}).get("text", ""))
        return GetPromptResult(
            messages=[PromptMessage(role="user", content=TextContent(type="text", text=text))],
        )


@dataclass(frozen=True)
class KeepMCPSurface:
    """Server plus named handlers retained for direct compatibility tests."""

    server: KeepMCPServer
    keep_flow: Callable[..., Awaitable[CallToolResult]]
    keep_prompt: Callable[..., Awaitable[CallToolResult]]
    keep_help: Callable[..., Awaitable[str]]


def create_keep_mcp_surface(service: KeepMCPService, *, version: str) -> KeepMCPSurface:
    """Create the complete keep MCP contract around a deployment adapter."""
    server = KeepMCPServer(
        service,
        name="keep",
        version=version,
        instructions=(
            "Reflective memory with semantic search. "
            "Store facts, preferences, decisions, and notes. "
            "Search by meaning. Persist context across sessions."
        ),
    )

    @server.resource(
        "keep://now",
        name="now",
        title="Current Note",
        description="Current working note as JSON.",
        mime_type="application/json",
    )
    async def keep_now_resource() -> dict[str, Any]:
        note = await service.read_note("now")
        if note is None:
            raise ResourceNotFoundError("note not found: now")
        return note

    @server.resource(
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
    async def keep_note_resource(id: str) -> dict[str, Any]:
        note_id = unquote(id)
        note = await service.read_note(note_id)
        if note is None:
            raise ResourceNotFoundError(f"note not found: {note_id}")
        return note

    @server.tool(
        description=(
            "Execute a keep operation via state-doc flow. "
            "Examples:\n"
            '  Search: state="query-resolve", params={"query": "auth patterns"}\n'
            '  Get context: state="get", params={"item_id": "now"}\n'
            '  Store text: state="put", params={"content": "decision: use JWT", "tags": {"project": "auth"}}\n'
            '  Store with ID: state="put", params={"id": "meeting-notes", "content": "..."}\n'
            '  Store file: state="put", params={"uri": "file:///path/to/doc.md"}\n'
            '  Store URL: state="put", params={"uri": "https://example.com/article"}\n'
            '  List notes: state="list", params={"prefix": ".tag/", "include_hidden": true}\n'
            '  Resume stopped search: state="query-resolve", cursor="<cursor from previous call>"\n'
            "When status is 'stopped', pass the returned cursor to continue. "
            "Set token_budget for rendered text output instead of raw JSON. "
            'List available flows: keep_help(topic="flow_state_docs").'
        ),
        annotations=_MAY_MUTATE,
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
    ) -> Annotated[CallToolResult, KeepFlowStructured]:
        params_body = params.model_dump(exclude_none=True) if isinstance(params, BaseModel) else params
        response = await service.execute_flow(
            state,
            params_body,
            budget=budget,
            cursor=cursor,
            state_doc_yaml=state_doc_yaml,
            token_budget=token_budget,
        )
        if response.status_code != 200:
            raise ValueError(str(response.payload.get("error", "keep service unavailable")))
        if response.payload.get("status") == "error":
            error = (response.payload.get("data") or {}).get("error", "flow failed")
            raise ValueError(str(error))

        if response.payload.get("rendered"):
            rendered = str(response.payload["rendered"])
            structured = {
                "status": response.payload.get("status"),
                "ticks": response.payload.get("ticks"),
                "rendered": rendered,
            }
            # Rendering is an additional representation, not a replacement
            # for control-plane state. In particular, stopped flows must keep
            # their cursor so a token-budget client can resume them.
            for key in ("data", "cursor", "tried_queries"):
                value = response.payload.get(key)
                if value:
                    structured[key] = value
            return CallToolResult(
                content=[TextContent(type="text", text=rendered)],
                structured_content=structured,
            )

        structured: dict[str, Any] = {
            "status": response.payload.get("status"),
            "ticks": response.payload.get("ticks"),
        }
        for key in ("data", "cursor", "tried_queries"):
            value = response.payload.get(key)
            if value:
                structured[key] = value
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(structured, indent=2))],
            structured_content=structured,
        )

    @server.tool(
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
            description='Note ID for context (default: "now").',
        )] = None,
        tags: Annotated[Optional[dict[str, str | list[str]]], Field(
            description="Filter search context by tags.",
        )] = None,
        since: Annotated[Optional[str], Field(
            description="Only include notes updated since this value (ISO duration or date).",
        )] = None,
        until: Annotated[Optional[str], Field(
            description="Only include notes updated before this value (ISO duration or date).",
        )] = None,
        deep: Annotated[bool, Field(
            description="Follow tags from results to discover related notes.",
        )] = False,
        scope: Annotated[Optional[str], Field(
            description="ID glob to constrain search results (e.g. 'file:///path/to/dir*').",
        )] = None,
        token_budget: Annotated[Optional[int], Field(
            description="Token budget for search results context (template default if not set).",
        )] = None,
    ) -> Annotated[CallToolResult, KeepPromptStructured]:
        flow_params: dict[str, Any] = {}
        if not _normalize_optional_arg(name):
            flow_params["list"] = True
        else:
            flow_params["name"] = name
            for key, value in (
                ("text", text),
                ("id", id),
                ("since", since),
                ("until", until),
                ("scope", scope),
            ):
                normalized = _normalize_optional_arg(value)
                if normalized is not None:
                    flow_params[key] = normalized
            if tags:
                flow_params["tags"] = tags
            if deep:
                flow_params["deep"] = deep
            if token_budget:
                flow_params["token_budget"] = token_budget

        response = await service.execute_flow(STATE_PROMPT, flow_params)
        if response.status_code != 200:
            error = str(response.payload.get("error", "keep service unavailable"))
            return CallToolResult(
                content=[TextContent(type="text", text=error)],
                structured_content={"mode": "error", "error": error},
                is_error=True,
            )

        flow_data = response.payload.get("data", {})
        if not name:
            prompts = flow_data.get("prompts", [])
            prompt_rows = [
                {"name": str(row.get("name", "")), "summary": str(row.get("summary", "") or "")}
                for row in prompts
                if isinstance(row, dict)
            ]
            if prompt_rows:
                lines = [f"Available prompts ({len(prompt_rows)}):"]
                lines.extend(f"- {row['name']}: {row['summary']}".rstrip() for row in prompt_rows)
                content = "\n".join(lines)
            else:
                content = "No agent prompts available."
            return CallToolResult(
                content=[TextContent(type="text", text=content)],
                structured_content={"mode": "list", "prompts": prompt_rows},
            )

        if response.payload.get("status") == "error":
            error = str(flow_data.get("error", f"prompt not found: {name}"))
            return CallToolResult(
                content=[TextContent(type="text", text=error)],
                structured_content={"mode": "render", "name": name, "error": error},
                is_error=True,
            )

        rendered = str(flow_data.get("text", ""))
        return CallToolResult(
            content=[TextContent(type="text", text=rendered)],
            structured_content={"mode": "render", "name": name, "text": rendered},
        )

    @server.tool(
        description=(
            "Comprehensive keep documentation with examples for all commands, "
            "flows, tagging, prompts, and architecture. "
            'Call with topic="index" to see all available guides.'
        ),
        annotations=_READ_ONLY,
    )
    async def keep_help(
        topic: Annotated[str, Field(
            description='Documentation topic, e.g. "index", "quickstart", "keep-put", "tagging". '
            'Use "index" to see all available topics.',
        )] = "index",
    ) -> str:
        return await service.get_help(topic)

    return KeepMCPSurface(
        server=server,
        keep_flow=keep_flow,
        keep_prompt=keep_prompt,
        keep_help=keep_help,
    )


__all__ = [
    "FlowParams",
    "FlowResponse",
    "KeepFlowStructured",
    "KeepMCPServer",
    "KeepMCPService",
    "KeepMCPSurface",
    "KeepPromptStructured",
    "MCPHTTPHeaderNormalizer",
    "PromptSummary",
    "create_keep_mcp_surface",
]
