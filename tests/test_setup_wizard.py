"""Tests for the first-run setup wizard."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from keep.setup_wizard import (
    _run_interactive_setup,
    needs_wizard,
    detect_embedding_choices,
    detect_summarization_choices,
    detect_tool_choices,
    run_wizard,
)


@pytest.fixture(autouse=True)
def _clear_local_only(monkeypatch):
    """Wizard choice tests should not inherit suite-wide local-only mode."""
    monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)


class TestNeedsWizard:
    """Tests for wizard-needed detection."""
    def test_needs_wizard_no_config(self, tmp_path):
        assert needs_wizard(tmp_path) is True

    def test_needs_wizard_with_config(self, tmp_path):
        (tmp_path / "keep.toml").write_text("[store]\nversion = 3\n")
        assert needs_wizard(tmp_path) is False


class TestDetectToolChoices:
    """Tests for tool choice detection."""
    def test_detects_tools(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".kiro").mkdir()

        choices = detect_tool_choices()
        found = {c["key"]: c["found"] for c in choices}
        assert found["claude_code"] is True
        assert found["kiro"] is True
        assert found["codex"] is False
        assert found["openclaw"] is False

    def test_no_tools_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        choices = detect_tool_choices()
        assert all(not c["found"] for c in choices)


class TestDetectEmbeddingChoices:
    """Tests for embedding choice detection."""

    def test_broken_local_stack_surfaces_original_import_error(self, monkeypatch):
        """Installed-but-broken local packages are not reported as missing."""
        monkeypatch.setattr("keep.setup_wizard._detect_ollama", lambda: None)
        monkeypatch.setattr("keep.setup_wizard.platform.system", lambda: "Darwin")
        monkeypatch.setattr("keep.setup_wizard.platform.machine", lambda: "arm64")

        def probe(module_name: str):
            if module_name == "sentence_transformers":
                return "broken", "tokenizers<=0.23.0 is required, found 0.23.1"
            return "available", None

        monkeypatch.setattr("keep.setup_wizard.probe_optional_dependency", probe)

        choices = detect_embedding_choices()
        local_choices = [
            choice for choice in choices
            if choice["name"].startswith(("MLX", "sentence-transformers"))
        ]

        assert len(local_choices) == 2
        assert all(choice["available"] is False for choice in local_choices)
        assert all(
            "installed but unusable" in choice["hint"]
            and "tokenizers<=0.23.0" in choice["hint"]
            for choice in local_choices
        )

    def test_missing_local_stack_keeps_install_guidance(self, monkeypatch):
        """A genuinely absent package retains the local-extra instruction."""
        monkeypatch.setattr("keep.setup_wizard._detect_ollama", lambda: None)
        monkeypatch.setattr("keep.setup_wizard.platform.system", lambda: "Linux")
        monkeypatch.setattr(
            "keep.setup_wizard.probe_optional_dependency",
            lambda _module_name: ("missing", None),
        )

        choices = detect_embedding_choices()
        sentence_transformers = next(
            choice for choice in choices
            if choice["name"].startswith("sentence-transformers")
        )

        assert sentence_transformers["available"] is False
        assert sentence_transformers["hint"] == (
            "requires: uv tool install 'keep-skill[local]'"
        )

    def test_mlx_hint_reports_missing_and_broken_dependencies(self, monkeypatch):
        """A mixed failure must not hide either required repair action."""
        monkeypatch.setattr("keep.setup_wizard._detect_ollama", lambda: None)
        monkeypatch.setattr("keep.setup_wizard.platform.system", lambda: "Darwin")
        monkeypatch.setattr("keep.setup_wizard.platform.machine", lambda: "arm64")

        def probe(module_name: str):
            if module_name == "mlx.core":
                return "missing", None
            return "broken", "tokenizers is incompatible"

        monkeypatch.setattr("keep.setup_wizard.probe_optional_dependency", probe)

        choices = detect_embedding_choices()
        mlx = next(choice for choice in choices if choice["name"].startswith("MLX"))

        assert "requires: uv tool install 'keep-skill[local]'" in mlx["hint"]
        assert (
            "sentence-transformers installed but unusable: "
            "tokenizers is incompatible"
        ) in mlx["hint"]

    def test_ollama_available(self, monkeypatch):
        monkeypatch.setattr(
            "keep.setup_wizard._detect_ollama",
            lambda: {"base_url": "http://localhost:11434", "models": ["nomic-embed-text"]},
        )
        # Suppress API key detection
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KEEP_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        choices = detect_embedding_choices()
        ollama_choices = [c for c in choices if "Ollama" in c["name"]]
        assert len(ollama_choices) == 1
        assert ollama_choices[0]["available"] is True
        assert ollama_choices[0]["default"] is True

    def test_empty_ollama_server_uses_default_embedding(self, monkeypatch):
        """A fresh Ollama install is available even before any models are pulled."""
        monkeypatch.setattr(
            "keep.setup_wizard._detect_ollama",
            lambda: {"base_url": "http://localhost:11434", "models": []},
        )
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KEEP_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        choices = detect_embedding_choices()

        ollama_choices = [c for c in choices if "Ollama" in c["name"]]
        assert len(ollama_choices) == 1
        assert ollama_choices[0]["available"] is True
        assert ollama_choices[0]["value"] == ("ollama", {"model": "nomic-embed-text"})

    def test_no_ollama_api_key_default(self, monkeypatch):
        monkeypatch.setattr("keep.setup_wizard._detect_ollama", lambda: None)
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KEEP_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        choices = detect_embedding_choices()
        voyage = [c for c in choices if "Voyage" in c["name"]]
        assert len(voyage) == 1
        assert voyage[0]["available"] is True
        assert voyage[0]["default"] is True

    def test_unavailable_shows_requirement(self, monkeypatch):
        monkeypatch.setattr("keep.setup_wizard._detect_ollama", lambda: None)
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KEEP_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        choices = detect_embedding_choices()
        openai = [c for c in choices if "OpenAI" in c["name"]]
        assert len(openai) == 1
        assert openai[0]["available"] is False
        assert "requires" in openai[0]["hint"]

    def test_openrouter_shown_only_when_key_present(self, monkeypatch):
        monkeypatch.setattr("keep.setup_wizard._detect_ollama", lambda: None)
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KEEP_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        choices = detect_embedding_choices()
        assert not any("OpenRouter" in c["name"] for c in choices)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        choices = detect_embedding_choices()
        openrouter = [c for c in choices if "OpenRouter" in c["name"]]
        assert len(openrouter) == 1
        assert openrouter[0]["available"] is True
        assert openrouter[0]["value"] == ("openrouter", {"model": "openai/text-embedding-3-small"})


class TestDetectSummarizationChoices:
    """Tests for summarization choice detection."""

    def test_broken_mlx_stack_surfaces_original_import_error(self, monkeypatch):
        """A transitive MLX failure is reported as broken, not uninstalled."""
        monkeypatch.setattr("keep.setup_wizard._detect_ollama", lambda: None)
        monkeypatch.setattr("keep.setup_wizard.platform.system", lambda: "Darwin")
        monkeypatch.setattr("keep.setup_wizard.platform.machine", lambda: "arm64")
        monkeypatch.setattr(
            "keep.setup_wizard.probe_optional_dependency",
            lambda module_name: (
                "broken",
                "tokenizers<=0.23.0 is required, found 0.23.1",
            ) if module_name == "mlx_lm" else ("missing", None),
        )

        choices = detect_summarization_choices()
        mlx = next(choice for choice in choices if choice["name"].startswith("MLX"))

        assert mlx["available"] is False
        assert mlx["hint"] == (
            "installed but unusable: "
            "tokenizers<=0.23.0 is required, found 0.23.1"
        )

    def test_missing_mlx_stack_keeps_install_guidance(self, monkeypatch):
        """A genuinely absent mlx-lm package retains the local-extra hint."""
        monkeypatch.setattr("keep.setup_wizard._detect_ollama", lambda: None)
        monkeypatch.setattr("keep.setup_wizard.platform.system", lambda: "Darwin")
        monkeypatch.setattr("keep.setup_wizard.platform.machine", lambda: "arm64")
        monkeypatch.setattr(
            "keep.setup_wizard.probe_optional_dependency",
            lambda _module_name: ("missing", None),
        )

        choices = detect_summarization_choices()
        mlx = next(choice for choice in choices if choice["name"].startswith("MLX"))

        assert mlx["available"] is False
        assert mlx["hint"] == "requires: uv tool install 'keep-skill[local]'"

    def test_always_has_truncate_fallback(self, monkeypatch):
        monkeypatch.setattr("keep.setup_wizard._detect_ollama", lambda: None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KEEP_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        choices = detect_summarization_choices()
        truncate = [c for c in choices if "truncate" in c["name"]]
        assert len(truncate) == 1
        assert truncate[0]["available"] is True

    def test_empty_ollama_server_uses_default_summarization(self, monkeypatch):
        """A fresh Ollama install can pull the default chat model on first use."""
        monkeypatch.setattr(
            "keep.setup_wizard._detect_ollama",
            lambda: {"base_url": "http://localhost:11434", "models": []},
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KEEP_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        choices = detect_summarization_choices()

        ollama_choices = [c for c in choices if "Ollama" in c["name"]]
        assert len(ollama_choices) == 1
        assert ollama_choices[0]["available"] is True
        assert ollama_choices[0]["value"] == ("ollama", {"model": "llama3.2"})

    def test_openrouter_summarization_shown_only_when_key_present(self, monkeypatch):
        monkeypatch.setattr("keep.setup_wizard._detect_ollama", lambda: None)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("KEEP_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        choices = detect_summarization_choices()
        assert not any("OpenRouter" in c["name"] for c in choices)

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        choices = detect_summarization_choices()
        openrouter = [c for c in choices if "OpenRouter" in c["name"]]
        assert len(openrouter) == 1
        assert openrouter[0]["available"] is True
        assert openrouter[0]["value"] == ("openrouter", {"model": "openai/gpt-4.1-mini"})


class TestRunWizardNonInteractive:
    """Tests for non-interactive wizard fallback."""
    def test_non_interactive_fallback(self, tmp_path, monkeypatch, mock_providers):
        """Non-interactive mode creates config without installing integrations."""
        monkeypatch.setattr("keep.setup_wizard._is_interactive", lambda: False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        with patch("keep.integrations.check_and_install", side_effect=AssertionError("should not be called")):
            config = run_wizard(tmp_path)
        assert config is not None
        assert (tmp_path / "keep.toml").exists()

    def test_non_interactive_remote_runs_content_verification(
        self, tmp_path, monkeypatch,
    ):
        """Smoke/non-TTY setup must verify hosted credentials end-to-end."""
        monkeypatch.setattr("keep.setup_wizard._is_interactive", lambda: False)
        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_test")
        monkeypatch.setenv("KEEPNOTES_API_URL", "https://api.example.test")
        monkeypatch.setenv("KEEPNOTES_PROJECT", "alpha")

        with (
            patch(
                "keep.setup_wizard._verify_remote",
                return_value=(True, "content round trip OK"),
            ) as verify,
            patch("keep.setup_wizard.detect_tool_choices") as detect_tools,
            patch("keep.setup_wizard.stop_daemon"),
        ):
            config = run_wizard(tmp_path)

        verify.assert_called_once()
        remote = verify.call_args.args[0]
        assert remote.api_url == "https://api.example.test"
        assert remote.api_key == "kn_test"
        assert remote.project == "alpha"
        detect_tools.assert_not_called()
        assert config.remote is None  # env credentials are not persisted
        assert "[remote]" not in (tmp_path / "keep.toml").read_text(encoding="utf-8")

    def test_non_interactive_remote_failure_does_not_write(
        self, tmp_path, monkeypatch,
    ):
        """Broken remote credentials abort setup before keep.toml is written."""
        monkeypatch.setattr("keep.setup_wizard._is_interactive", lambda: False)
        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_test")

        with (
            patch(
                "keep.setup_wizard._verify_remote",
                return_value=(False, "content round trip failed: request_id=rid"),
            ),
            patch("keep.setup_wizard.save_config") as save_config,
        ):
            with pytest.raises(SystemExit) as exc:
                run_wizard(tmp_path)

        assert exc.value.code == 1
        save_config.assert_not_called()
        assert not (tmp_path / "keep.toml").exists()


class TestRemoteShortCircuit:
    """Wizard short-circuits embedding/summarization prompts when remote is set."""

    @pytest.fixture
    def _clean_env(self, monkeypatch):
        for var in ("KEEPNOTES_API_URL", "KEEPNOTES_API_KEY", "KEEPNOTES_PROJECT"):
            monkeypatch.delenv(var, raising=False)

    def test_verify_remote_requires_content_round_trip(self, monkeypatch):
        """Remote verification writes, reads, lists, and deletes a probe note."""
        from keep.config import RemoteConfig
        from keep.setup_wizard import _verify_remote

        class FakeRemoteKeeper:
            instances = []

            def __init__(self, api_url, api_key, config, *, project=None):
                self.api_url = api_url
                self.api_key = api_key
                self.config = config
                self.project = project
                self.calls = []
                self.item = None
                FakeRemoteKeeper.instances.append(self)

            def server_info(self):
                raise AssertionError("readiness-only checks are insufficient")

            def put(self, *, content, id, tags):
                self.calls.append(("put", id, content, tags))
                self.item = SimpleNamespace(id=id, summary=content, tags=tags)
                return self.item

            def get(self, id):
                self.calls.append(("get", id))
                return self.item if self.item and self.item.id == id else None

            def find(self, *, tags, limit):
                self.calls.append(("find", tags, limit))
                return [self.item] if self.item and self.item.tags == tags else []

            def delete(self, id):
                self.calls.append(("delete", id))
                return True

            def close(self):
                self.calls.append(("close",))

        monkeypatch.setattr("keep.remote.RemoteKeeper", FakeRemoteKeeper)

        ok, message = _verify_remote(
            RemoteConfig(
                api_url="https://api.example.test",
                api_key="kn_test",
                project="alpha",
            )
        )

        assert ok is True
        assert message == "content round trip OK"

        instance = FakeRemoteKeeper.instances[0]
        assert instance.api_url == "https://api.example.test"
        assert instance.api_key == "kn_test"
        assert instance.project == "alpha"
        assert [call[0] for call in instance.calls] == [
            "put", "get", "find", "delete", "close",
        ]
        put_call = instance.calls[0]
        assert put_call[1].startswith("keep-setup-probe-")
        assert put_call[1] in put_call[2]
        assert put_call[3] == {"keep_setup_probe": put_call[1]}

    def test_verify_remote_reports_content_probe_failure(self, monkeypatch):
        """Write failures fail setup verification before config is persisted."""
        from keep.config import RemoteConfig
        from keep.setup_wizard import _verify_remote

        class FailingRemoteKeeper:
            instances = []

            def __init__(self, *args, **kwargs):
                self.calls = []
                FailingRemoteKeeper.instances.append(self)

            def put(self, **kwargs):
                self.calls.append(("put", kwargs))
                raise RuntimeError("write rejected")

            def delete(self, id):
                self.calls.append(("delete", id))

            def close(self):
                self.calls.append(("close",))

        monkeypatch.setattr("keep.remote.RemoteKeeper", FailingRemoteKeeper)

        ok, message = _verify_remote(
            RemoteConfig(api_url="https://api.example.test", api_key="kn_test")
        )

        assert ok is False
        assert "content round trip failed" in message
        assert "write rejected" in message
        assert [call[0] for call in FailingRemoteKeeper.instances[0].calls] == [
            "put", "close",
        ]

    def test_env_credentials_skip_provider_prompts(self, tmp_path, monkeypatch, _clean_env):
        """KEEPNOTES_API_KEY in env triggers the remote setup path."""
        from keep.setup_wizard import _run_interactive_setup

        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_test")
        monkeypatch.setenv("KEEPNOTES_API_URL", "https://api.example.test")

        config_dir = tmp_path / "config"
        config_dir.mkdir()

        with (
            patch("keep.setup_wizard.detect_tool_choices", return_value=[]),
            patch(
                "keep.setup_wizard._verify_remote", return_value=(True, "reachable"),
            ),
            patch("keep.setup_wizard._run_provider_selection") as mock_select,
            patch("keep.setup_wizard.stop_daemon"),
        ):
            config = _run_interactive_setup(
                config_dir=config_dir,
                store_path=None,
                actual_store=config_dir,
                existing=None,
            )

        mock_select.assert_not_called()  # provider prompts skipped
        assert config.embedding is None  # remote handles embedding
        # Env-sourced credentials are NOT written to TOML (kept in env)
        saved = (config_dir / "keep.toml").read_text(encoding="utf-8")
        assert "[remote]" not in saved

    def test_existing_toml_credentials_skip_provider_prompts(
        self, tmp_path, monkeypatch, _clean_env,
    ):
        """An existing [remote] in keep.toml triggers the remote setup path."""
        from keep.config import RemoteConfig, StoreConfig
        from keep.setup_wizard import _run_interactive_setup

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        existing = StoreConfig(
            path=config_dir,
            config_dir=config_dir,
            embedding=None,
            remote=RemoteConfig(
                api_url="https://api.example.test",
                api_key="kn_test",
                project="alpha",
            ),
        )

        with (
            patch("keep.setup_wizard.detect_tool_choices", return_value=[]),
            patch(
                "keep.setup_wizard._verify_remote", return_value=(True, "reachable"),
            ),
            patch("keep.setup_wizard._run_provider_selection") as mock_select,
            patch("keep.setup_wizard.stop_daemon"),
        ):
            config = _run_interactive_setup(
                config_dir=config_dir,
                store_path=None,
                actual_store=config_dir,
                existing=existing,
            )

        mock_select.assert_not_called()
        assert config.embedding is None
        assert config.remote is not None
        assert config.remote.api_key == "kn_test"
        saved = (config_dir / "keep.toml").read_text(encoding="utf-8")
        assert "[remote]" in saved

    def test_env_key_overlays_toml_api_url_in_wizard(
        self, tmp_path, monkeypatch, _clean_env,
    ):
        """KEEPNOTES_API_KEY alone must merge with TOML-owned api_url/project."""
        from keep.config import RemoteConfig, StoreConfig
        from keep.setup_wizard import _detect_remote_config

        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_env_only")

        existing = StoreConfig(
            path=tmp_path,
            config_dir=tmp_path,
            embedding=None,
            remote_persist=RemoteConfig(
                api_url="https://config.example.test",
                api_key="kn_file",
                project="from-file",
            ),
        )

        detected = _detect_remote_config(existing)
        assert detected is not None
        assert detected.api_url == "https://config.example.test"
        assert detected.api_key == "kn_env_only"
        assert detected.project == "from-file"

    def test_remote_verification_failure_exits_without_writing(
        self, tmp_path, monkeypatch, _clean_env,
    ):
        """Verification failure aborts setup so a broken config isn't persisted."""
        from keep.setup_wizard import _run_interactive_setup

        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_test")
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        with (
            patch("keep.setup_wizard.detect_tool_choices", return_value=[]),
            patch(
                "keep.setup_wizard._verify_remote",
                return_value=(False, "unreachable: timeout"),
            ),
            patch("keep.setup_wizard.stop_daemon") as mock_stop,
            patch("keep.setup_wizard.save_config") as mock_save,
        ):
            with pytest.raises(SystemExit) as exc:
                _run_interactive_setup(
                    config_dir=config_dir,
                    store_path=None,
                    actual_store=config_dir,
                    existing=None,
                )

        assert exc.value.code == 1
        mock_save.assert_not_called()
        mock_stop.assert_not_called()
        assert not (config_dir / "keep.toml").exists()


class TestInteractiveSetup:
    """Interactive setup behavior around daemon restart."""

    def test_explicit_store_restarts_daemon_for_store_path(self, tmp_path):
        """Explicit store configs must restart the daemon for the real store."""
        config_dir = tmp_path / "config"
        actual_store = tmp_path / "store"
        config_dir.mkdir()
        actual_store.mkdir()

        with (
            patch("keep.setup_wizard.detect_tool_choices", return_value=[]),
            patch(
                "keep.setup_wizard.detect_embedding_choices",
                return_value=[{"name": "Embed", "available": True, "value": ("ollama", {"model": "m"})}],
            ),
            patch(
                "keep.setup_wizard.detect_summarization_choices",
                return_value=[{"name": "Skip", "available": True, "value": None}],
            ),
            patch(
                "keep.setup_wizard._run_provider_selection",
                side_effect=[("ollama", {"model": "m"}), None],
            ),
            patch("keep.setup_wizard.detect_default_providers", return_value={}),
            patch("keep.setup_wizard.save_config"),
            patch("keep.setup_wizard.stop_daemon") as mock_stop,
        ):
            config = _run_interactive_setup(
                config_dir=config_dir,
                store_path=actual_store,
                actual_store=actual_store,
                existing=None,
            )

        assert config.path == actual_store
        mock_stop.assert_called_once_with(actual_store, force=True)
