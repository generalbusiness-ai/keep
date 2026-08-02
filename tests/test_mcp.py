"""Tests for the MCP stdio server tool functions.

Tests the tool layer in isolation by mocking HTTP calls to the daemon —
verifies parameter mapping, return formatting, and edge cases for the
three tools: keep_flow, keep_prompt, keep_help.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from keep.const import STATE_DELETE, STATE_PUT, STATE_QUERY_RESOLVE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_daemon():
    """Force MCP onto the daemon backend with a stubbed HTTP transport.

    Pre-resolves the daemon backend's port so the helper short-circuits port
    discovery; patches http_request_with_discovery_retry to drive responses.
    Resets the cached backend on teardown to avoid cross-test bleed.
    """
    from keep import mcp as mcp_mod

    backend = mcp_mod._DaemonBackend()
    backend._port = 9999
    mcp_mod._backend = backend
    try:
        with patch("keep.mcp.http_request_with_discovery_retry") as mock_http:
            yield mock_http
    finally:
        mcp_mod._backend = None


async def _keep_flow_schema(server):
    for tool in await server.list_tools():
        if tool.name == "keep_flow":
            return tool.input_schema
    raise AssertionError("keep_flow schema not found")


# ---------------------------------------------------------------------------
# keep_flow
# ---------------------------------------------------------------------------

class TestKeepFlow:
    """Tests for MCP keep-flow endpoint."""

    def test_flow_schema_exposes_common_param_keys(self):
        from keep.mcp import mcp

        schema = asyncio.run(_keep_flow_schema(mcp))
        params_ref = schema["properties"]["params"]["anyOf"][0]["$ref"]
        params_schema = schema
        for part in params_ref.removeprefix("#/").split("/"):
            params_schema = params_schema[part]

        assert "properties" in params_schema
        assert "item_id" in params_schema["properties"]
        assert "query" in params_schema["properties"]
        assert "content" in params_schema["properties"]
        assert "source" in params_schema["properties"]
        assert "source_id" not in params_schema["properties"]
        assert "deep" in params_schema["properties"]
        assert "token_budget" in params_schema["properties"]
        assert params_schema["additionalProperties"] is True
        assert schema["properties"]["params"]["examples"][0] == {"item_id": "now"}

    @pytest.mark.asyncio
    async def test_flow_returns_json(self, mock_daemon):
        from keep.mcp import keep_flow
        mock_daemon.return_value = (200, {
            "status": "done", "ticks": 1,
            "data": {"id": "test-123"},
            "bindings": {}, "history": [], "cursor": None, "tried_queries": [],
        })
        result = await keep_flow(state=STATE_PUT, params={"content": "hello"})
        parsed = result.structured_content
        assert parsed["status"] == "done"
        assert parsed["data"]["id"] == "test-123"

    @pytest.mark.asyncio
    async def test_flow_with_cursor(self, mock_daemon):
        from keep.mcp import keep_flow
        mock_daemon.return_value = (200, {
            "status": "stopped", "ticks": 3,
            "data": {"reason": "budget"}, "cursor": "abc123",
            "tried_queries": ["test query"],
            "bindings": {}, "history": [],
        })
        result = await keep_flow(
            state=STATE_QUERY_RESOLVE, params={"query": "test"}, budget=3,
        )
        parsed = result.structured_content
        assert parsed["status"] == "stopped"
        assert parsed["cursor"] == "abc123"
        assert parsed["tried_queries"] == ["test query"]
        assert "bindings" not in parsed
        assert "history" not in parsed

    @pytest.mark.asyncio
    async def test_flow_error(self, mock_daemon):
        from keep.mcp import keep_flow
        mock_daemon.return_value = (500, {"error": "bad params"})
        with pytest.raises(ValueError, match="bad params"):
            await keep_flow(state=STATE_PUT)

    @pytest.mark.asyncio
    async def test_flow_no_data_in_output(self, mock_daemon):
        from keep.mcp import keep_flow
        mock_daemon.return_value = (200, {
            "status": "done", "ticks": 1,
            "data": None, "bindings": {}, "history": [],
            "cursor": None, "tried_queries": [],
        })
        result = await keep_flow(state=STATE_DELETE, params={"id": "x"})
        parsed = result.structured_content
        assert "data" not in parsed

    @pytest.mark.asyncio
    async def test_flow_with_token_budget(self, mock_daemon):
        from keep.mcp import keep_flow
        mock_daemon.return_value = (200, {
            "status": "done", "ticks": 1,
            "data": {}, "bindings": {}, "history": [],
            "cursor": None, "tried_queries": [],
            "rendered": "Rendered output text",
        })
        result = await keep_flow(state=STATE_QUERY_RESOLVE, token_budget=4000)
        assert result.content[0].text == "Rendered output text"
        assert result.structured_content["rendered"] == "Rendered output text"
        # Verify token_budget was sent in the request
        call_body = mock_daemon.call_args[0][3]  # body arg
        assert call_body.get("token_budget") == 4000

    @pytest.mark.asyncio
    async def test_rendered_stopped_flow_preserves_resume_cursor(self, mock_daemon):
        """Token-budget rendering must not discard resumability metadata."""
        from keep.mcp import keep_flow

        mock_daemon.return_value = (200, {
            "status": "stopped",
            "ticks": 3,
            "data": {"reason": "budget"},
            "cursor": "resume-123",
            "tried_queries": ["needle"],
            "rendered": "Rendered output text",
        })

        result = await keep_flow(state=STATE_QUERY_RESOLVE, token_budget=4000)

        assert result.structured_content["cursor"] == "resume-123"
        assert result.structured_content["data"] == {"reason": "budget"}
        assert result.structured_content["tried_queries"] == ["needle"]

    @pytest.mark.asyncio
    async def test_flow_uses_shared_discovery_retry_helper(self):
        from keep import mcp as mcp_mod
        from keep.mcp import keep_flow

        backend = mcp_mod._DaemonBackend()
        backend._port = 9999
        mcp_mod._backend = backend
        try:
            with patch("keep.mcp.http_request_with_discovery_retry") as mock_http:
                mock_http.return_value = (200, {
                    "status": "done", "ticks": 1,
                    "data": {"ok": True},
                    "bindings": {}, "history": [], "cursor": None, "tried_queries": [],
                })

                result = await keep_flow(state=STATE_PUT, params={"content": "hello"})

            parsed = result.structured_content
            assert parsed["status"] == "done"
            mock_http.assert_called_once()
            assert mock_http.call_args.args[:3] == ("POST", 9999, "/v1/flow")
        finally:
            mcp_mod._backend = None


class TestMCPProtocolContract:
    """Serialized modern and legacy behavior through the SDK client."""

    @pytest.mark.asyncio
    async def test_modern_discovery_results_and_annotations(self, mock_daemon):
        from mcp import Client

        from keep.mcp import mcp

        mock_daemon.return_value = (200, {
            "status": "done", "ticks": 1, "data": {"prompts": []},
        })

        async with Client(mcp, mode="2026-07-28", raise_exceptions=True) as client:
            assert client.protocol_version == "2026-07-28"
            result = await client.list_tools()

        assert result.result_type == "complete"
        assert result.ttl_ms == 0
        assert result.cache_scope == "private"
        flow_tool = next(tool for tool in result.tools if tool.name == "keep_flow")
        assert flow_tool.annotations.read_only_hint is False
        assert flow_tool.annotations.destructive_hint is True
        assert flow_tool.annotations.idempotent_hint is False
        assert mcp._lowlevel_server.get_request_handler("subscriptions/listen") is None

    @pytest.mark.asyncio
    async def test_legacy_client_remains_supported(self, mock_daemon):
        from mcp import Client

        from keep.mcp import mcp

        mock_daemon.return_value = (200, {
            "status": "done", "ticks": 1, "data": {"prompts": []},
        })

        async with Client(mcp, mode="legacy") as client:
            assert client.protocol_version == "2025-11-25"
            result = await client.list_tools()

        assert [tool.name for tool in result.tools] == [
            "keep_flow", "keep_prompt", "keep_help",
        ]

    @pytest.mark.asyncio
    async def test_flow_failure_is_a_tool_error(self, mock_daemon):
        from mcp import Client

        from keep.mcp import mcp

        mock_daemon.return_value = (500, {"error": "bad params"})

        async with Client(mcp, mode="2026-07-28", raise_exceptions=True) as client:
            result = await client.call_tool("keep_flow", {"state": "put"})

        assert result.is_error is True
        assert "bad params" in result.content[0].text

    @pytest.mark.asyncio
    async def test_legacy_source_id_is_forwarded_as_canonical_source(self, mock_daemon):
        from mcp import Client

        from keep.mcp import mcp

        mock_daemon.return_value = (200, {
            "status": "done", "ticks": 1, "data": {"id": "archive"},
        })

        async with Client(mcp, mode="2026-07-28", raise_exceptions=True) as client:
            result = await client.call_tool(
                "keep_flow",
                {"state": "move", "params": {"name": "archive", "source_id": "source-note"}},
            )

        assert result.is_error is False
        request_body = next(
            call.args[3]
            for call in mock_daemon.call_args_list
            if call.args[3].get("state") == "move"
        )
        assert request_body["params"]["source"] == "source-note"
        assert "source_id" not in request_body["params"]

    @pytest.mark.asyncio
    async def test_http_routing_headers_trim_optional_whitespace(self):
        from keep.mcp_surface import MCPHTTPHeaderNormalizer

        captured = {}

        async def downstream(scope, receive, send):
            del receive
            captured["headers"] = scope["headers"]
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        sent = []

        async def send(message):
            sent.append(message)

        app = MCPHTTPHeaderNormalizer(downstream)
        await app(
            {
                "type": "http",
                "method": "POST",
                "path": "/mcp",
                "headers": [
                    (b"mcp-method", b" tools/call\t"),
                    (b"mcp-name", b"  keep_help  "),
                    (b"x-unrelated", b"  preserve  "),
                ],
            },
            receive,
            send,
        )

        assert captured["headers"] == [
            (b"mcp-method", b"tools/call"),
            (b"mcp-name", b"keep_help"),
            (b"x-unrelated", b"  preserve  "),
        ]


# ---------------------------------------------------------------------------
# keep_prompt
# ---------------------------------------------------------------------------

class TestKeepPrompt:
    """Tests for MCP keep-prompt endpoint."""

    @pytest.mark.asyncio
    async def test_list_prompts(self, mock_daemon):
        from keep.mcp import keep_prompt
        mock_daemon.return_value = (200, {
            "status": "done", "ticks": 1,
            "data": {
                "prompts": [
                    {"name": "reflect", "summary": "The reflection practice"},
                    {"name": "session-start", "summary": "Session startup"},
                ],
            },
            "bindings": {}, "history": [], "cursor": None, "tried_queries": [],
        })
        result = await keep_prompt()
        assert result.structured_content == {
            "mode": "list",
            "prompts": [
                {"name": "reflect", "summary": "The reflection practice"},
                {"name": "session-start", "summary": "Session startup"},
            ],
        }
        assert "reflect" in result.content[0].text
        assert "session-start" in result.content[0].text

    @pytest.mark.asyncio
    async def test_render_prompt(self, mock_daemon):
        from keep.mcp import keep_prompt
        mock_daemon.return_value = (200, {
            "status": "done", "ticks": 1,
            "data": {"text": "Reflect on your recent work..."},
            "bindings": {}, "history": [], "cursor": None, "tried_queries": [],
        })
        result = await keep_prompt(name="reflect")
        assert result.structured_content == {
            "mode": "render",
            "name": "reflect",
            "text": "Reflect on your recent work...",
        }
        assert "Reflect on" in result.content[0].text

    @pytest.mark.asyncio
    async def test_prompt_not_found(self, mock_daemon):
        from keep.mcp import keep_prompt
        mock_daemon.return_value = (200, {
            "status": "error", "ticks": 1,
            "data": {"error": "prompt not found: nonexistent"},
            "bindings": {}, "history": [], "cursor": None, "tried_queries": [],
        })
        result = await keep_prompt(name="nonexistent")
        assert result.is_error is True
        assert result.structured_content == {
            "mode": "render",
            "name": "nonexistent",
            "error": "prompt not found: nonexistent",
        }
        assert "not found" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_prompt_http_error_returns_structured_error(self, mock_daemon):
        from keep.mcp import keep_prompt

        mock_daemon.return_value = (500, {"error": "bad upstream"})

        result = await keep_prompt(name="reflect")

        assert result.is_error is True
        assert result.structured_content == {
            "mode": "error",
            "error": "bad upstream",
        }
        assert result.content[0].text == "bad upstream"


class TestMCPPromptExposure:
    """Tests for MCP-native prompt exposure."""

    @pytest.mark.asyncio
    async def test_list_prompts_filters_to_mcp_exposed_prompts(self, mock_daemon):
        from keep.mcp import mcp

        mock_daemon.return_value = (200, {
            "status": "done",
            "ticks": 1,
            "data": {
                "prompts": [
                    {
                        "name": "reflect",
                        "summary": "The reflection practice",
                        "mcp_arguments": ["text", "id", "since", "token_budget"],
                    },
                    {
                        "name": "session-start",
                        "summary": "Session startup",
                    },
                ],
            },
            "bindings": {},
            "history": [],
            "cursor": None,
            "tried_queries": [],
        })

        prompts = await mcp.list_prompts()

        assert len(prompts) == 1
        assert prompts[0].name == "reflect"
        assert [arg.name for arg in (prompts[0].arguments or [])] == [
            "text",
            "id",
            "since",
            "token_budget",
        ]
        assert all(arg.required is False for arg in (prompts[0].arguments or []))

    @pytest.mark.asyncio
    async def test_list_prompts_returns_empty_when_daemon_is_unavailable(self, mock_daemon):
        from keep.mcp import mcp

        mock_daemon.side_effect = ConnectionRefusedError(61, "refused")

        prompts = await mcp.list_prompts()

        assert prompts == []

    @pytest.mark.asyncio
    async def test_get_prompt_renders_via_existing_prompt_flow(self, mock_daemon):
        from keep.mcp import mcp

        mock_daemon.return_value = (200, {
            "status": "done",
            "ticks": 1,
            "data": {"text": "Rendered reflect prompt"},
            "bindings": {},
            "history": [],
            "cursor": None,
            "tried_queries": [],
        })

        result = await mcp.get_prompt(
            "reflect",
            {"text": "auth", "id": "now", "since": "P7D", "ignored": "x"},
        )

        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content.text == "Rendered reflect prompt"
        render_body = mock_daemon.call_args.args[3]
        assert render_body["params"] == {
            "name": "reflect",
            "text": "auth",
            "id": "now",
            "since": "P7D",
        }

    @pytest.mark.asyncio
    async def test_get_prompt_ignores_empty_string_optional_args(self, mock_daemon):
        from keep.mcp import mcp

        mock_daemon.return_value = (200, {
            "status": "done",
            "ticks": 1,
            "data": {"text": "Rendered reflect prompt"},
            "bindings": {},
            "history": [],
            "cursor": None,
            "tried_queries": [],
        })

        await mcp.get_prompt(
            "reflect",
            {"text": "", "id": "", "since": "  ", "token_budget": ""},
        )

        render_body = mock_daemon.call_args.args[3]
        assert render_body["params"] == {"name": "reflect"}

    @pytest.mark.asyncio
    async def test_get_prompt_raises_on_unknown_prompt(self, mock_daemon):
        from keep.mcp import mcp

        mock_daemon.return_value = (200, {
            "status": "error",
            "ticks": 1,
            "data": {"error": "prompt not found: nonexistent"},
            "bindings": {},
            "history": [],
            "cursor": None,
            "tried_queries": [],
        })

        from mcp import MCPError

        with pytest.raises(MCPError, match="prompt not found: nonexistent"):
            await mcp.get_prompt("nonexistent")


# ---------------------------------------------------------------------------
# MCP resources
# ---------------------------------------------------------------------------

class TestMCPResources:
    """Tests for MCP resource and template exposure."""

    @pytest.mark.asyncio
    async def test_list_resources_includes_now(self, mock_daemon):
        from keep.mcp import mcp

        resources = await mcp.list_resources()

        assert any(str(resource.uri) == "keep://now" for resource in resources)

    @pytest.mark.asyncio
    async def test_list_resource_templates_includes_note_template(self, mock_daemon):
        from keep.mcp import mcp

        templates = await mcp.list_resource_templates()

        assert any(template.uri_template == "keep://{id}" for template in templates)

    @pytest.mark.asyncio
    async def test_read_now_resource_returns_note_json(self, mock_daemon):
        from keep.mcp import mcp

        mock_daemon.return_value = (200, {"id": "now", "summary": "Current note", "tags": {}})

        contents = await mcp.read_resource("keep://now")

        assert len(contents) == 1
        assert contents[0].mime_type == "application/json"
        data = json.loads(contents[0].content)
        assert data["id"] == "now"
        assert mock_daemon.call_args.args[0] == "GET"
        assert mock_daemon.call_args.args[2] == "/v1/notes/now"

    @pytest.mark.asyncio
    async def test_read_template_resource_decodes_note_id(self, mock_daemon):
        from keep.mcp import mcp

        mock_daemon.return_value = (
            200,
            {"id": "file:///tmp/note.md", "summary": "File note", "tags": {}},
        )

        contents = await mcp.read_resource("keep://file%3A%2F%2F%2Ftmp%2Fnote.md")

        assert len(contents) == 1
        data = json.loads(contents[0].content)
        assert data["id"] == "file:///tmp/note.md"
        assert mock_daemon.call_args.args[2] == "/v1/notes/file%3A%2F%2F%2Ftmp%2Fnote.md"

    @pytest.mark.asyncio
    async def test_missing_resource_uses_protocol_error(self, mock_daemon):
        from mcp.server.mcpserver.exceptions import ResourceNotFoundError

        from keep.mcp import mcp

        mock_daemon.return_value = (404, {"error": "not found"})

        with pytest.raises(ResourceNotFoundError, match="note not found: missing"):
            await mcp.read_resource("keep://missing")


class TestMCPToolDescriptions:
    """Tests for dynamic MCP tool descriptions."""

    @pytest.mark.asyncio
    async def test_keep_prompt_tool_description_lists_available_prompts(self, mock_daemon):
        from keep.mcp import mcp

        mock_daemon.return_value = (200, {
            "status": "done",
            "ticks": 1,
            "data": {
                "prompts": [
                    {"name": "reflect", "summary": "Reflect"},
                    {"name": "conversation", "summary": "Conversation"},
                    {"name": "query", "summary": "Query"},
                ],
            },
            "bindings": {},
            "history": [],
            "cursor": None,
            "tried_queries": [],
        })

        tools = await mcp.list_tools()
        keep_prompt_tool = next(tool for tool in tools if tool.name == "keep_prompt")

        assert keep_prompt_tool.description is not None
        assert "Available prompts:" in keep_prompt_tool.description
        assert "reflect" in keep_prompt_tool.description
        assert "conversation" in keep_prompt_tool.description
        assert "query" in keep_prompt_tool.description


# ---------------------------------------------------------------------------
# keep_help
# ---------------------------------------------------------------------------

class TestKeepHelp:
    """Tests for MCP keep-help endpoint."""

    @pytest.mark.asyncio
    async def test_help_index(self):
        from keep.mcp import keep_help
        result = await keep_help(topic="index")
        assert "quickstart" in result.lower() or "guide" in result.lower()

    @pytest.mark.asyncio
    async def test_help_specific_topic(self):
        from keep.mcp import keep_help
        result = await keep_help(topic="flow-actions")
        assert "find" in result.lower()


# ---------------------------------------------------------------------------
# Backend selection: daemon vs remote
# ---------------------------------------------------------------------------

class TestMCPBackendSelection:
    """The MCP backend picks daemon-vs-remote from env/config, not flags."""

    def setup_method(self):
        from keep import mcp as mcp_mod
        mcp_mod._backend = None

    def teardown_method(self):
        from keep import mcp as mcp_mod
        if mcp_mod._backend is not None:
            try:
                mcp_mod._backend.close()
            except Exception:
                pass
        mcp_mod._backend = None

    def test_no_remote_yields_daemon_backend(self, monkeypatch):
        from keep import mcp as mcp_mod

        for var in ("KEEPNOTES_API_KEY", "KEEPNOTES_API_URL", "KEEPNOTES_PROJECT"):
            monkeypatch.delenv(var, raising=False)
        # KEEP_LOCAL_ONLY is the strongest signal — already set by conftest.
        backend = mcp_mod._get_backend()
        assert isinstance(backend, mcp_mod._DaemonBackend)

    def test_keepnotes_env_yields_remote_backend(self, monkeypatch, tmp_path):
        from keep import mcp as mcp_mod

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_test")
        monkeypatch.setenv("KEEPNOTES_API_URL", "https://api.example.test")
        monkeypatch.setattr("keep.paths.get_config_dir", lambda: tmp_path)

        backend = mcp_mod._get_backend()
        assert isinstance(backend, mcp_mod._RemoteBackend)
        assert backend._api_url == "https://api.example.test"

    def test_keep_local_only_overrides_remote_env(self, monkeypatch):
        """KEEP_LOCAL_ONLY should win even if KEEPNOTES_API_KEY is set."""
        from keep import mcp as mcp_mod

        monkeypatch.setenv("KEEP_LOCAL_ONLY", "1")
        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_test")
        backend = mcp_mod._get_backend()
        assert isinstance(backend, mcp_mod._DaemonBackend)

    def test_toml_remote_section_yields_remote_backend(self, monkeypatch, tmp_path):
        """[remote] in keep.toml is picked up when no env override is present."""
        from keep import mcp as mcp_mod

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        for var in ("KEEPNOTES_API_KEY", "KEEPNOTES_API_URL", "KEEPNOTES_PROJECT"):
            monkeypatch.delenv(var, raising=False)

        (tmp_path / "keep.toml").write_text(
            """
[store]
version = 2

[remote]
api_url = "https://api.example.test"
api_key = "kn_test"
project = "demo"
""".strip() + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("keep.paths.get_config_dir", lambda: tmp_path)

        backend = mcp_mod._get_backend()
        assert isinstance(backend, mcp_mod._RemoteBackend)
        assert backend._api_url == "https://api.example.test"

    def test_env_key_overlays_on_toml_api_url(self, monkeypatch, tmp_path):
        """KEEPNOTES_API_KEY alone must not erase api_url/project from TOML."""
        from keep import mcp as mcp_mod

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_URL", raising=False)
        monkeypatch.delenv("KEEPNOTES_PROJECT", raising=False)
        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_env_only")

        (tmp_path / "keep.toml").write_text(
            """
[store]
version = 2

[remote]
api_url = "https://config.example.test"
api_key = "kn_file"
project = "from-file"
""".strip() + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("KEEP_CONFIG", str(tmp_path))

        remote, _ = mcp_mod._load_remote_config()
        assert remote is not None
        # api_url and project come from TOML; only the api_key was overridden.
        assert remote.api_url == "https://config.example.test"
        assert remote.api_key == "kn_env_only"
        assert remote.project == "from-file"

    def test_keep_store_path_resolves_config_dir(self, monkeypatch, tmp_path):
        """`keep --store /path mcp` (sets KEEP_STORE_PATH) must find /path/keep.toml."""
        from keep import mcp as mcp_mod

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEP_CONFIG", raising=False)
        for var in ("KEEPNOTES_API_KEY", "KEEPNOTES_API_URL", "KEEPNOTES_PROJECT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("KEEP_STORE_PATH", str(tmp_path))

        (tmp_path / "keep.toml").write_text(
            """
[store]
version = 2

[remote]
api_url = "https://store-path.example.test"
api_key = "kn_store_path"
""".strip() + "\n",
            encoding="utf-8",
        )

        backend = mcp_mod._get_backend()
        assert isinstance(backend, mcp_mod._RemoteBackend)
        assert backend._api_url == "https://store-path.example.test"


class TestMCPRemoteBackendRouting:
    """Remote backend's HTTP behavior: routing, logging, error handling."""

    def setup_method(self):
        from keep import mcp as mcp_mod
        mcp_mod._backend = None

    def teardown_method(self):
        from keep import mcp as mcp_mod
        if mcp_mod._backend is not None:
            try:
                mcp_mod._backend.close()
            except Exception:
                pass
        mcp_mod._backend = None

    @pytest.mark.asyncio
    async def test_keep_flow_uses_remote_backend_when_configured(
        self, monkeypatch, tmp_path,
    ):
        from keep import mcp as mcp_mod
        from keep.config import RemoteConfig

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        # Force remote backend with a mocked httpx client.
        remote = RemoteConfig(
            api_url="https://api.example.test", api_key="kn_test", project=None,
        )
        backend = mcp_mod._RemoteBackend(remote, log_dir=tmp_path)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "done", "ticks": 1, "data": {"id": "abc"},
            "bindings": {}, "history": [], "cursor": None, "tried_queries": [],
        }
        mock_client.request.return_value = mock_response
        backend._client = mock_client
        mcp_mod._backend = backend

        result = await mcp_mod.keep_flow(state="put", params={"content": "hi"})

        parsed = result.structured_content
        assert parsed["status"] == "done"
        assert mock_client.request.called
        args, _ = mock_client.request.call_args
        assert args[0] == "POST"
        assert args[1] == "/v1/flow"

    def test_remote_backend_logs_each_call_to_client_log(
        self, monkeypatch, tmp_path,
    ):
        from keep import mcp as mcp_mod
        from keep.config import RemoteConfig

        remote = RemoteConfig(
            api_url="https://api.example.test", api_key="kn_test", project=None,
        )
        backend = mcp_mod._RemoteBackend(remote, log_dir=tmp_path)
        try:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"ok": True}
            mock_client.request.return_value = mock_response
            backend._client = mock_client

            backend.post("/v1/flow", {"state": "list"})
            backend.get("/v1/notes/foo")
        finally:
            backend.close()

        client_log = tmp_path / "keep-client.log"
        assert client_log.exists()
        body = client_log.read_text(encoding="utf-8")
        assert "POST /v1/flow" in body
        assert "GET /v1/notes/foo" in body
        assert "status=200" in body
        assert "host=https://api.example.test" in body

    def test_remote_backend_get_returns_status_and_dict(self, monkeypatch, tmp_path):
        """Remote backend matches the (status, dict) contract of the daemon backend."""
        from keep import mcp as mcp_mod
        from keep.config import RemoteConfig

        remote = RemoteConfig(
            api_url="https://api.example.test", api_key="kn_test", project=None,
        )
        backend = mcp_mod._RemoteBackend(remote, log_dir=tmp_path)
        try:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.json.side_effect = ValueError("not json")
            mock_response.text = "not found"
            mock_client.request.return_value = mock_response
            backend._client = mock_client

            status, body = backend.get("/v1/notes/missing")
            assert status == 404
            assert body == {"error": "not found"}
        finally:
            backend.close()

    def test_remote_backend_rejects_non_https_url(self, tmp_path):
        """Bearer tokens must not be sent over plain HTTP (except loopback)."""
        from keep import mcp as mcp_mod
        from keep.config import RemoteConfig

        bad = RemoteConfig(
            api_url="http://api.public.example/", api_key="kn_test", project=None,
        )
        with pytest.raises(ValueError, match="HTTPS"):
            mcp_mod._RemoteBackend(bad, log_dir=tmp_path)

    def test_remote_backend_allows_http_loopback(self, tmp_path):
        """http://localhost is allowed for local-dev / smoke setups."""
        from keep import mcp as mcp_mod
        from keep.config import RemoteConfig

        ok = RemoteConfig(
            api_url="http://127.0.0.1:8080", api_key="kn_test", project=None,
        )
        backend = mcp_mod._RemoteBackend(ok, log_dir=tmp_path)
        try:
            assert backend._api_url == "http://127.0.0.1:8080"
        finally:
            backend.close()

    def test_remote_backend_rejects_bad_project_slug(self, tmp_path):
        """Malformed project slugs are caught before any HTTP request."""
        from keep import mcp as mcp_mod
        from keep.config import RemoteConfig

        bad = RemoteConfig(
            api_url="https://api.example.test", api_key="kn_test",
            project="Bad_Slug!",
        )
        with pytest.raises(ValueError, match="Invalid project slug"):
            mcp_mod._RemoteBackend(bad, log_dir=tmp_path)

    def test_remote_backend_http_error_returns_zero_status(self, monkeypatch, tmp_path):
        """Connection failures surface as status=0 instead of raising."""
        import httpx
        from keep import mcp as mcp_mod
        from keep.config import RemoteConfig

        remote = RemoteConfig(
            api_url="https://api.example.test", api_key="kn_test", project=None,
        )
        backend = mcp_mod._RemoteBackend(remote, log_dir=tmp_path)
        try:
            mock_client = MagicMock()
            mock_client.request.side_effect = httpx.ConnectError("refused")
            backend._client = mock_client

            status, body = backend.post("/v1/flow", {"state": "x"})
            assert status == 0
            assert "refused" in body["error"]
        finally:
            backend.close()
