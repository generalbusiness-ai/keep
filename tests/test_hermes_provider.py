"""Tests for keep.hermes.provider — the Hermes MemoryProvider integration.

Uses mock_providers fixture to avoid loading real ML models or databases.
Tests cover the full MemoryProvider protocol surface.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from keep.hermes.provider import KeepMemoryProvider
from keep.hermes.const import EMBEDDING_EMPTY_HINTS


class TestLifecycle:
    """Provider creation, initialization, and shutdown."""

    def test_is_available(self):
        p = KeepMemoryProvider()
        # keep is importable in the test environment
        assert p.is_available() is True

    def test_name(self):
        p = KeepMemoryProvider()
        assert p.name == "keep"

    def test_initialize_creates_keeper(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize(
            "test-session",
            hermes_home=str(tmp_path),
            platform="cli",
            agent_identity="test",
        )
        assert p._keeper is not None
        p.shutdown()
        assert p._keeper is None

    def test_initialize_skips_cron(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize(
            "test-session",
            hermes_home=str(tmp_path),
            platform="cron",
            agent_context="cron",
        )
        assert p._keeper is None

    def test_shutdown_idempotent(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        p.shutdown()
        p.shutdown()  # second call should not raise

    def test_initialize_keeps_working_when_daemon_start_fails(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        with patch("keep.daemon_client.get_port", side_effect=SystemExit(1)):
            p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        assert p._setup_required is False
        assert p._keeper is not None
        p.shutdown()


@pytest.fixture(autouse=True)
def _stub_keep_daemon_port():
    """Avoid real daemon startup in Hermes provider tests by default."""
    with patch("keep.daemon_client.get_port", return_value=9999):
        yield


class TestSystemPrompt:
    """system_prompt_block() rendering."""

    def test_returns_empty_when_not_initialized(self):
        p = KeepMemoryProvider()
        assert p.system_prompt_block() == ""

    def test_returns_system_doc_when_initialized(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        block = p.system_prompt_block()
        # The system doc (.prompt/agent/system) is a sysdoc migrated on init
        assert "Reflective Memory" in block
        assert "keep_flow" in block
        p.shutdown()

    def test_setup_required_message(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p._setup_required = True
        block = p.system_prompt_block()
        assert "not set up yet" in block
        assert "hermes memory setup" in block

    def test_setup_required_uses_profile_name(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p._setup_required = True
        p._setup_cmd = "coder memory setup"
        block = p.system_prompt_block()
        assert "coder memory setup" in block


class TestPrefetch:
    """prefetch() and queue_prefetch()."""

    def test_prefetch_empty_when_not_initialized(self):
        p = KeepMemoryProvider()
        assert p.prefetch("test query") == ""

    def test_prefetch_returns_context(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        # Put something to find
        p._keeper.put("Authentication uses JWT tokens", tags={"topic": "auth"})
        result = p.prefetch("auth tokens")
        # May be empty if mock embeddings don't produce meaningful similarity,
        # but should not raise
        assert isinstance(result, str)
        p.shutdown()

    def test_queue_prefetch_does_not_raise(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        p.queue_prefetch("auth tokens")
        # Wait for background thread
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)
        p.shutdown()


class TestSyncTurn:
    """sync_turn() conversation recording."""

    def test_sync_turn_writes_item(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli",
                     agent_identity="test")
        p.sync_turn("Hello agent", "Hello! How can I help?")
        # Wait for background thread
        if p._sync_thread:
            p._sync_thread.join(timeout=5.0)
        # Verify the item was written
        item = p._keeper.get(p._session_item_id)
        assert item is not None
        assert item.summary  # non-empty summary (content or auto-generated)
        p.shutdown()

    def test_sync_turn_overlays_turn_user_identity_tags(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="discord",
                     agent_identity="test")
        p.sync_turn(
            "ping",
            "pong",
            user_id="42",
            user_name="Alice",
        )
        if p._sync_thread:
            p._sync_thread.join(timeout=5.0)
        item = p._keeper.get(p._session_item_id)
        assert item is not None
        assert item.tags["user_id"] == "[[contact:discord:42|Alice]]"
        assert item.tags["user_name"] == "Alice"
        p.shutdown()

    def test_sync_turn_uses_cached_turn_identity_when_sync_args_absent(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="discord",
                     agent_identity="test")
        p.on_turn_start(
            1,
            "ping",
            user_id="42",
            user_name="Alice",
        )
        p.sync_turn("ping", "pong")
        if p._sync_thread:
            p._sync_thread.join(timeout=5.0)
        item = p._keeper.get(p._session_item_id)
        assert item is not None
        assert item.tags["user_id"] == "[[contact:discord:42|Alice]]"
        assert item.tags["user_name"] == "Alice"
        p.shutdown()

    def test_sync_turn_includes_cached_session_title(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="discord",
                     agent_identity="test")
        p.on_turn_start(
            1,
            "ping",
            session_title="Research Thread",
        )
        p.sync_turn("ping", "pong")
        if p._sync_thread:
            p._sync_thread.join(timeout=5.0)
        item = p._keeper.get(p._session_item_id)
        assert item is not None
        assert item.tags["title"] == "Research Thread"
        p.shutdown()

    def test_sync_turn_skips_when_not_initialized(self):
        p = KeepMemoryProvider()
        # Should not raise
        p.sync_turn("Hello", "Hi")


class TestToolSchemas:
    """get_tool_schemas() and handle_tool_call()."""

    def test_tools_available_before_initialize(self):
        """Tool schemas are static — available even before initialize()."""
        p = KeepMemoryProvider()
        schemas = p.get_tool_schemas()
        assert len(schemas) == 3

    def test_three_tools_when_initialized(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        schemas = p.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "keep_flow" in names
        assert "keep_help" in names
        assert "keep_prompt" in names
        p.shutdown()

    def test_tool_error_when_not_initialized(self):
        p = KeepMemoryProvider()
        result = json.loads(p.handle_tool_call("keep_flow", {}))
        assert "error" in result

    def test_tool_error_setup_required(self):
        p = KeepMemoryProvider()
        p._setup_required = True
        result = json.loads(p.handle_tool_call("keep_flow", {}))
        assert "error" in result
        assert "not configured" in result["error"]
        assert "hint" in result

    def test_setup_hints_mention_openrouter(self):
        assert any("OPENROUTER_API_KEY" in hint for hint in EMBEDDING_EMPTY_HINTS)


class TestKeepFlow:
    """keep_flow tool — state-doc flow execution."""

    def test_flow_requires_state(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        result = json.loads(p.handle_tool_call("keep_flow", {}))
        assert "error" in result
        assert "state" in result["error"].lower()
        p.shutdown()

    def test_flow_put(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        result = json.loads(p.handle_tool_call("keep_flow", {
            "state": "put",
            "params": {"content": "test fact", "tags": {"topic": "test"}},
        }))
        assert "result" in result  # rendered text
        assert "done" in result["result"]
        p.shutdown()

    def test_flow_get(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        # Put then get
        p._keeper.put("important fact", id="test-item")
        result = json.loads(p.handle_tool_call("keep_flow", {
            "state": "get",
            "params": {"id": "test-item"},
        }))
        assert "result" in result  # rendered text
        assert "important fact" in result["result"]
        assert "intentions:" not in result["result"]
        p.shutdown()

    def test_flow_get_now_renders_full_body(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        result = json.loads(p.handle_tool_call("keep_flow", {
            "state": "get",
            "params": {"id": "now"},
        }))
        assert "result" in result
        assert "---\nid: now" in result["result"]
        assert "# Keep" in result["result"]
        assert "Read the practice guide" in result["result"]
        assert "intentions:" not in result["result"]
        p.shutdown()

    def test_flow_list_versions_exposes_version_history(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        # The mock document store used in this fixture does not persist
        # archived versions automatically, so stub the history explicitly.
        p._keeper._document_store.list_versions = lambda collection, id, limit=10: [
            SimpleNamespace(summary="initial schedule", created_at="2026-04-05T02:03:28")
        ]
        result = json.loads(p.handle_tool_call("keep_flow", {
            "state": "list_versions",
            "params": {"id": "schedule-note", "limit": 5},
        }))
        assert "result" in result
        assert "versions:" in result["result"]
        assert "@V{1}" in result["result"]
        assert "initial schedule" in result["result"]
        p.shutdown()

    def test_flow_inline_state_doc_yaml_uses_inline_doc(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        yaml_doc = """\
match: sequence
rules:
  - id: wanted
    do: get
    with:
      id: ".library"
post:
  - return: done
"""
        result = json.loads(p.handle_tool_call("keep_flow", {
            "state": "get",
            "state_doc_yaml": yaml_doc,
        }))
        assert "result" in result
        assert ".library" in result["result"]
        assert "reflective memory" in result["result"].lower()
        p.shutdown()

    def test_flow_state_doc_yaml_action_shorthand_rescues_to_params(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        yaml_doc = """\
action: get
id: .library
"""
        result = json.loads(p.handle_tool_call("keep_flow", {
            "state": "get",
            "state_doc_yaml": yaml_doc,
        }))
        assert "result" in result
        assert ".library" in result["result"]
        assert "# Library" in result["result"]
        p.shutdown()

    def test_flow_state_doc_yaml_do_with_shorthand_rescues_to_params(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        yaml_doc = """\
do: get
with:
  id: .library
"""
        result = json.loads(p.handle_tool_call("keep_flow", {
            "state": "get",
            "state_doc_yaml": yaml_doc,
        }))
        assert "result" in result
        assert ".library" in result["result"]
        assert "# Library" in result["result"]
        p.shutdown()

    def test_flow_unknown_state(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        result = json.loads(p.handle_tool_call("keep_flow", {
            "state": "nonexistent-state-doc",
        }))
        assert "result" in result
        assert "error" in result["result"]
        p.shutdown()


class TestKeepHelp:
    """keep_help tool."""

    def test_help_index(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        result = json.loads(p.handle_tool_call("keep_help", {}))
        assert "result" in result
        assert "Guides" in result["result"] or "guide" in result["result"].lower()
        p.shutdown()

    def test_help_topic(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        result = json.loads(p.handle_tool_call("keep_help", {
            "topic": "agent-guide",
        }))
        assert "result" in result
        assert len(result["result"]) > 100
        p.shutdown()

    def test_help_unknown_topic(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        result = json.loads(p.handle_tool_call("keep_help", {
            "topic": "does-not-exist",
        }))
        # get_help_topic returns guidance text listing available topics
        assert "result" in result
        assert "Available topics" in result["result"]
        p.shutdown()


class TestKeepPrompt:
    """keep_prompt tool."""

    def test_prompt_list(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        result = json.loads(p.handle_tool_call("keep_prompt", {}))
        assert "prompts" in result
        assert isinstance(result["prompts"], list)
        p.shutdown()

    def test_prompt_render(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        prompts = json.loads(p.handle_tool_call("keep_prompt", {}))
        if prompts.get("prompts"):
            name = prompts["prompts"][0]
            result = json.loads(p.handle_tool_call("keep_prompt", {
                "name": name,
            }))
            # Should have result or error (prompt may need context)
            assert "result" in result or "error" in result
        p.shutdown()

    def test_prompt_not_found(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        result = json.loads(p.handle_tool_call("keep_prompt", {
            "name": "nonexistent-prompt",
        }))
        assert result.get("error") is not None or result.get("result") is not None
        p.shutdown()


class TestObservationHooks:
    """on_memory_write, on_pre_compress, on_delegation."""

    def test_on_turn_start_updates_session_title_and_turn_identity(self):
        p = KeepMemoryProvider()
        p._session_tags = {"source": "hermes", "platform": "telegram", "agent_identity": "coder"}
        p._session_id = "old-session"
        p._session_item_id = "coder:telegram:old-session"

        p.on_turn_start(
            3,
            "hello",
            session_id="new-session",
            session_title="Planning",
            user_id="42",
            user_name="Alice",
        )

        assert p._turn_count == 3
        assert p._turn_user_id == "42"
        assert p._turn_user_name == "Alice"
        assert p._session_id == "new-session"
        assert p._session_item_id == "coder:telegram:new-session"
        assert p._session_tags["title"] == "Planning"

    def test_on_turn_start_clears_title_when_explicitly_empty(self):
        p = KeepMemoryProvider()
        p._session_tags = {"source": "hermes", "title": "Old Title"}

        p.on_turn_start(1, "hello", session_title="")

        assert "title" not in p._session_tags

    def test_on_memory_write(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        p.on_memory_write("add", "memory", "User prefers dark mode")
        # Background thread — wait for it
        import time
        time.sleep(0.5)
        p.shutdown()

    def test_on_memory_write_skips_remove(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        # "remove" action should be ignored
        p.on_memory_write("remove", "memory", "something")
        p.shutdown()

    def test_on_pre_compress(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = p.on_pre_compress(messages)
        assert result == ""  # Returns empty string (no contribution to compressor)
        p.shutdown()

    def test_on_pre_compress_empty(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        assert p.on_pre_compress([]) == ""
        p.shutdown()

    def test_on_delegation(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        p.on_delegation("search for auth docs", "Found 3 relevant files",
                        child_session_id="child-123")
        import time
        time.sleep(0.5)
        p.shutdown()

    def test_on_session_end(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
        p.on_session_end([{"role": "user", "content": "bye"}])
        p.shutdown()


class TestSessionItemId:
    """_build_session_item_id — channel scoping."""

    def test_cli_stable_key(self):
        p = KeepMemoryProvider()
        item_id = p._build_session_item_id(
            "random-uuid-123", platform="cli", agent_identity="coder"
        )
        assert item_id == "coder:cli"
        # Same key regardless of session_id
        item_id2 = p._build_session_item_id(
            "different-uuid-456", platform="cli", agent_identity="coder"
        )
        assert item_id == item_id2

    def test_gateway_includes_platform_and_session(self):
        p = KeepMemoryProvider()
        item_id = p._build_session_item_id(
            "agent:main:telegram:dm:12345", platform="telegram", agent_identity="default"
        )
        assert item_id == "default:telegram:agent:main:telegram:dm:12345"

    def test_gateway_discord_thread(self):
        p = KeepMemoryProvider()
        item_id = p._build_session_item_id(
            "20260404_174301_4de0b03a", platform="discord", agent_identity="default"
        )
        assert item_id == "default:discord:20260404_174301_4de0b03a"


class TestSessionTags:
    """_build_session_tags — metadata tagging."""

    def test_minimal_tags(self):
        p = KeepMemoryProvider()
        tags = p._build_session_tags("s1")
        assert tags == {"source": "hermes"}

    def test_platform_tag(self):
        p = KeepMemoryProvider()
        tags = p._build_session_tags("s1", platform="telegram")
        assert tags["platform"] == "telegram"

    def test_identity_tag(self):
        p = KeepMemoryProvider()
        tags = p._build_session_tags("s1", agent_identity="coder")
        assert tags["agent_identity"] == "coder"

    def test_session_tags_omit_user_identity(self):
        """user_id/user_name vary per message — not baked into session tags."""
        p = KeepMemoryProvider()
        tags = p._build_session_tags("s1", platform="telegram", user_id="42", user_name="Alice")
        assert "user_id" not in tags
        assert "user_name" not in tags


class TestConfigSchema:
    """get_config_schema() and save_config()."""

    def test_config_schema_returns_plain_string_choices(self, mock_providers):
        """Choices must be plain strings for Hermes memory_setup compat."""
        p = KeepMemoryProvider()
        embed_choices = [
            {"name": "Ollama (nomic-embed-text)", "hint": "-- local", "value": ("ollama", {"model": "nomic-embed-text"}), "default": True},
            {"name": "OpenAI (text-embedding-3-small)", "hint": "-- API", "value": ("openai", {"model": "text-embedding-3-small"})},
        ]
        summ_choices = [
            {"name": "Ollama (llama3)", "hint": "-- local", "value": ("ollama", {"model": "llama3"}), "default": True},
        ]
        with patch.object(KeepMemoryProvider, "_setup_choices", return_value=(embed_choices, summ_choices)):
            schema = p.get_config_schema()

        assert len(schema) == 2
        embed_field = schema[0]
        assert embed_field["key"] == "embedding_choice"
        assert embed_field["choices"] == ["Ollama (nomic-embed-text)", "OpenAI (text-embedding-3-small)"]
        assert embed_field["default"] == "Ollama (nomic-embed-text)"
        assert all(isinstance(c, str) for c in embed_field["choices"])

        summ_field = schema[1]
        assert summ_field["choices"] == ["Ollama (llama3)"]

    def test_config_schema_empty_when_no_embeddings(self, capsys):
        """When no embedding providers are available, returns empty schema with guidance."""
        p = KeepMemoryProvider()
        with patch.object(KeepMemoryProvider, "_setup_choices", return_value=([], [])):
            schema = p.get_config_schema()
        assert schema == []
        out = capsys.readouterr().out
        assert "No embedding provider available" in out

    def test_config_schema_empty_without_keep(self):
        """When keep is not importable, returns empty list."""
        p = KeepMemoryProvider()
        with patch.object(KeepMemoryProvider, "_setup_choices", return_value=([], [])):
            schema = p.get_config_schema()
        assert isinstance(schema, list)

    def test_save_config_maps_name_back_to_structured_value(self, mock_providers, tmp_path):
        """save_config receives plain string names and maps them back to provider configs."""
        p = KeepMemoryProvider()
        embed_choices = [
            {"name": "Ollama (nomic-embed-text)", "hint": "-- local", "value": ("ollama", {"model": "nomic-embed-text"}), "default": True},
        ]
        summ_choices = [
            {"name": "Ollama (llama3)", "hint": "-- local", "value": ("ollama", {"model": "llama3"})},
        ]

        saved_config = {}

        def fake_save_config(config):
            saved_config["embedding"] = (config.embedding.name, dict(config.embedding.params))
            if config.summarization:
                saved_config["summarization"] = (config.summarization.name, dict(config.summarization.params))

        with (
            patch.object(KeepMemoryProvider, "_setup_choices", return_value=(embed_choices, summ_choices)),
            patch("keep.config.save_config", fake_save_config),
            patch("keep.config.create_default_config") as mock_create,
        ):
            from keep.config import StoreConfig
            mock_cfg = StoreConfig(path=tmp_path)
            mock_create.return_value = mock_cfg

            p.save_config(
                {"embedding_choice": "Ollama (nomic-embed-text)", "summarization_choice": "Ollama (llama3)"},
                str(tmp_path),
            )

        assert saved_config["embedding"] == ("ollama", {"model": "nomic-embed-text"})
        assert saved_config["summarization"] == ("ollama", {"model": "llama3"})


class TestTokenBudgets:
    """_configure_token_budgets — hermes config mapping."""

    def test_default_budgets(self):
        p = KeepMemoryProvider()
        p._configure_token_budgets()
        assert p._system_prompt_token_budget > 0
        assert p._prefetch_token_budget > 0

    def test_custom_budgets(self):
        p = KeepMemoryProvider()
        p._configure_token_budgets(memory_char_limit=4400, user_char_limit=2750)
        # Larger limits → larger budgets
        assert p._system_prompt_token_budget > 1300

    def test_zero_limits(self):
        p = KeepMemoryProvider()
        p._configure_token_budgets(memory_char_limit=0, user_char_limit=0)
        assert p._system_prompt_token_budget == 1300  # fallback


class TestSetupCommand:
    """Profile-aware setup command."""

    def test_default_command(self):
        p = KeepMemoryProvider()
        assert p._setup_cmd == "hermes memory setup"

    def test_profile_command(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli",
                     agent_identity="coder")
        assert p._setup_cmd == "coder memory setup"

    def test_default_identity_keeps_hermes(self, mock_providers, tmp_path):
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli",
                     agent_identity="default")
        assert p._setup_cmd == "hermes memory setup"


class TestReadWriteEnvVar:
    """_read_env_var / _write_env_var round-trip and edge cases."""

    def test_read_missing_file(self, tmp_path):
        from keep.hermes.provider import _read_env_var
        assert _read_env_var(tmp_path / ".env", "FOO") is None

    def test_read_missing_key(self, tmp_path):
        from keep.hermes.provider import _read_env_var
        env = tmp_path / ".env"
        env.write_text("BAR=123\n")
        assert _read_env_var(env, "FOO") is None

    def test_write_then_read(self, tmp_path):
        from keep.hermes.provider import _read_env_var, _write_env_var
        env = tmp_path / ".env"
        _write_env_var(env, "KEEP_STORE_PATH", "/custom/path")
        assert _read_env_var(env, "KEEP_STORE_PATH") == "/custom/path"

    def test_write_updates_existing(self, tmp_path):
        from keep.hermes.provider import _read_env_var, _write_env_var
        env = tmp_path / ".env"
        env.write_text("KEEP_STORE_PATH=/old/path\nOTHER=1\n")
        _write_env_var(env, "KEEP_STORE_PATH", "/new/path")
        assert _read_env_var(env, "KEEP_STORE_PATH") == "/new/path"
        # Other vars preserved
        assert _read_env_var(env, "OTHER") == "1"


class TestSetupStorePathPreservation:
    """hermes memory setup should honour an existing KEEP_STORE_PATH."""

    def test_setup_preserves_env_store_path(self, mock_providers, tmp_path):
        """If KEEP_STORE_PATH is already in .env, save_config should use it."""
        from keep.hermes.provider import _read_env_var, _write_env_var
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        env_path = hermes_home / ".env"
        custom_path = str(tmp_path / "my-custom-keep")
        _write_env_var(env_path, "KEEP_STORE_PATH", custom_path)

        p = KeepMemoryProvider()
        # Stub the provider resolution chain that save_config uses
        with patch.object(p, "_current_keep_providers", return_value=(None, None)), \
             patch.object(p, "_setup_choices", return_value=(
                 [{"name": "Ollama (nomic-embed-text)", "value": "ollama|nomic-embed-text"}],
                 [{"name": "Skip", "value": None}],
             )), \
             patch.object(p, "_provider_config_from_choice", side_effect=[
                 SimpleNamespace(name="ollama", params={"model": "nomic-embed-text"}),
                 None,
             ]), \
             patch("keep.config.create_default_config") as mock_create, \
             patch("keep.config.save_config"), \
             patch.dict(os.environ, {}, clear=False):
            # Remove KEEP_STORE_PATH from env so only .env file is the source
            os.environ.pop("KEEP_STORE_PATH", None)
            p.save_config(
                {"embedding_choice": "Ollama (nomic-embed-text)",
                 "summarization_choice": "Skip"},
                hermes_home=str(hermes_home),
            )

        # create_default_config should have received the custom path, not default
        call_path = mock_create.call_args[0][0]
        assert str(call_path) == str(Path(custom_path).resolve())
        # .env should still have the custom path
        assert _read_env_var(env_path, "KEEP_STORE_PATH") == str(Path(custom_path).resolve())

    def test_setup_prefers_profile_env_over_process_env(self, mock_providers, tmp_path):
        """Profile .env must win over an inherited process KEEP_STORE_PATH."""
        from keep.hermes.provider import _read_env_var, _write_env_var

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        env_path = hermes_home / ".env"
        profile_path = str(tmp_path / "profile-store")
        inherited_path = str(tmp_path / "inherited-store")
        _write_env_var(env_path, "KEEP_STORE_PATH", profile_path)

        p = KeepMemoryProvider()
        with patch.object(p, "_current_keep_providers", return_value=(None, None)), \
             patch.object(p, "_setup_choices", return_value=(
                 [{"name": "Ollama (nomic-embed-text)", "value": "ollama|nomic-embed-text"}],
                 [{"name": "Skip", "value": None}],
             )), \
             patch.object(p, "_provider_config_from_choice", side_effect=[
                 SimpleNamespace(name="ollama", params={"model": "nomic-embed-text"}),
                 None,
             ]), \
             patch("keep.config.create_default_config") as mock_create, \
             patch("keep.config.save_config"), \
             patch.dict(os.environ, {"KEEP_STORE_PATH": inherited_path}, clear=False):
            p.save_config(
                {"embedding_choice": "Ollama (nomic-embed-text)",
                 "summarization_choice": "Skip"},
                hermes_home=str(hermes_home),
            )

        call_path = mock_create.call_args[0][0]
        assert str(call_path) == str(Path(profile_path).resolve())
        assert _read_env_var(env_path, "KEEP_STORE_PATH") == str(Path(profile_path).resolve())

    def test_setup_defaults_when_no_existing_path(self, mock_providers, tmp_path):
        """Without existing KEEP_STORE_PATH, save_config uses hermes_home/keep."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()

        p = KeepMemoryProvider()
        with patch.object(p, "_current_keep_providers", return_value=(None, None)), \
             patch.object(p, "_setup_choices", return_value=(
                 [{"name": "Ollama (nomic-embed-text)", "value": "ollama|nomic-embed-text"}],
                 [{"name": "Skip", "value": None}],
             )), \
             patch.object(p, "_provider_config_from_choice", side_effect=[
                 SimpleNamespace(name="ollama", params={"model": "nomic-embed-text"}),
                 None,
             ]), \
             patch("keep.config.create_default_config") as mock_create, \
             patch("keep.config.save_config"), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KEEP_STORE_PATH", None)
            p.save_config(
                {"embedding_choice": "Ollama (nomic-embed-text)",
                 "summarization_choice": "Skip"},
                hermes_home=str(hermes_home),
            )

        call_path = mock_create.call_args[0][0]
        assert str(call_path) == str((hermes_home / "keep").resolve())

    def test_initialize_prefers_profile_env_over_process_env(self, tmp_path):
        """initialize() must use the profile store when both sources exist."""
        from keep.hermes.provider import _write_env_var

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        profile_path = (tmp_path / "profile-store").resolve()
        inherited_path = (tmp_path / "inherited-store").resolve()
        _write_env_var(hermes_home / ".env", "KEEP_STORE_PATH", str(profile_path))

        p = KeepMemoryProvider()
        with patch.dict(os.environ, {"KEEP_STORE_PATH": str(inherited_path)}, clear=False):
            p.initialize("s1", hermes_home=str(hermes_home), platform="cron")

        assert p._store_path == str(profile_path)


class TestConversationTagging:
    """Verify type=conversation is set on automatic message captures only."""

    def test_sync_turn_sets_type_conversation(self, mock_providers, tmp_path):
        """Automatic message capture via sync_turn → type=conversation."""
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli",
                     agent_identity="test")
        p.sync_turn("Hello", "Hi there!")
        if p._sync_thread:
            p._sync_thread.join(timeout=5.0)
        item = p._keeper.get(p._session_item_id)
        assert item is not None
        assert item.tags.get("type") == "conversation"
        p.shutdown()

    def test_on_delegation_does_not_set_type_conversation(self, mock_providers, tmp_path):
        """Delegation is operational, not conversation capture."""
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli",
                     agent_identity="test")
        p.on_delegation("do X", "done X", child_session_id="child-1")
        # Wait for background thread
        import time
        time.sleep(1.0)
        # Find the delegation item (it has kind=delegation)
        results = p._keeper._find_direct(
            query="do X", tags={"kind": "delegation"},
        )
        if results:
            assert results[0].tags.get("type") != "conversation"
        p.shutdown()

    def test_on_pre_compress_does_not_set_type_conversation(self, mock_providers, tmp_path):
        """Compression snapshots are operational, not conversation."""
        p = KeepMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path), platform="cli",
                     agent_identity="test")
        p.on_pre_compress([{"role": "user", "content": "msg1"}, {"role": "assistant", "content": "msg2"}])
        results = p._keeper._find_direct(
            query="msg1", tags={"kind": "compression-snapshot"},
        )
        if results:
            assert results[0].tags.get("type") != "conversation"
        p.shutdown()


class TestHookConversationTags:
    """Verify IDE hook commands include type=conversation."""

    def test_claude_code_hook_includes_type_conversation(self):
        """Claude Code UserPromptSubmit hook command tags with type=conversation."""
        from keep.integrations import CLAUDE_CODE_HOOKS
        commands = [
            h["command"]
            for entry in CLAUDE_CODE_HOOKS.get("UserPromptSubmit", [])
            for h in entry.get("hooks", [])
        ]
        assert any("type=conversation" in cmd for cmd in commands)

    def test_installed_claude_hooks_pass_through_keep_stderr(self):
        """Installed hooks must not hide the real keep error or blame Ollama."""
        from keep.integrations import CLAUDE_CODE_HOOKS

        commands = [
            h["command"]
            for entries in CLAUDE_CODE_HOOKS.values()
            for entry in entries
            for h in entry.get("hooks", [])
        ]

        assert commands
        assert all("ollama serve" not in cmd for cmd in commands)
        assert all("embedding provider unavailable" not in cmd for cmd in commands)
        assert all("2>/dev/null" not in cmd for cmd in commands)
        assert all("2>&1" not in cmd for cmd in commands)
        assert any("see keep error above" in cmd for cmd in commands)

    def test_plugin_claude_hooks_pass_through_keep_stderr(self):
        """The distributable plugin hook JSON follows the same fallback rule."""
        hook_file = Path(__file__).parent.parent / "claude-code-plugin" / "hooks" / "hooks.json"
        data = json.loads(hook_file.read_text())
        commands = [
            h["command"]
            for entries in data["hooks"].values()
            for entry in entries
            for h in entry.get("hooks", [])
        ]

        assert commands
        assert all("ollama serve" not in cmd for cmd in commands)
        assert all("embedding provider unavailable" not in cmd for cmd in commands)
        assert all("2>/dev/null" not in cmd for cmd in commands)
        assert all("2>&1" not in cmd for cmd in commands)
        assert any("see keep error above" in cmd for cmd in commands)

    def test_kiro_hook_includes_type_conversation(self):
        """Kiro promptSubmit hook command tags with type=conversation."""
        from pathlib import Path
        hook_file = Path(__file__).parent.parent / "keep" / "data" / "kiro-hooks" / "keep-prompt.kiro.hook"
        content = hook_file.read_text()
        assert "type=conversation" in content
        assert "2>/dev/null" not in content
        assert "see keep error above" in content

    def test_kiro_spawn_hook_uses_valid_get_command(self):
        """Kiro agent-spawn hook must call a real CLI option set."""
        hook_file = Path(__file__).parent.parent / "keep" / "data" / "kiro-hooks" / "keep-spawn.kiro.hook"
        content = hook_file.read_text()
        assert "keep get now -n 10" in content
        assert "keep now -n" not in content
        assert "2>/dev/null" not in content
