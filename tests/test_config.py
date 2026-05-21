"""Configuration and provider-absence regression tests."""

from pathlib import Path
from unittest.mock import patch

import pytest


class TestOllamaDetection:
    """Tests for Ollama availability and model selection."""

    def test_empty_model_list_uses_pullable_defaults(self) -> None:
        """A fresh Ollama server should select models that can be pulled later."""
        from keep.config import _ollama_pick_models

        embed_model, chat_model = _ollama_pick_models([])

        assert embed_model == "nomic-embed-text"
        assert chat_model == "llama3.2"


class TestEmbeddingProviderAbsent:
    """Tests for behavior when no embedding provider is configured."""

    def test_get_embedding_provider_raises_with_message(self, tmp_path) -> None:
        """_get_embedding_provider raises RuntimeError with install instructions."""
        from keep.api import Keeper
        from keep.config import StoreConfig

        config = StoreConfig(path=tmp_path, embedding=None)

        with patch("keep.api.load_or_create_config", return_value=config), \
             patch("keep.store.ChromaStore"), \
             patch("keep.document_store.DocumentStore"), \
             patch("keep.pending_summaries.PendingSummaryQueue"):
            kp = Keeper(store_path=tmp_path)
            with pytest.raises(RuntimeError, match="No embedding provider configured"):
                kp._get_embedding_provider()

    def test_error_message_includes_install_options(self, tmp_path) -> None:
        """Error message mentions pip install and API key options."""
        from keep.api import Keeper
        from keep.config import StoreConfig

        config = StoreConfig(path=tmp_path, embedding=None)

        with patch("keep.api.load_or_create_config", return_value=config), \
             patch("keep.store.ChromaStore"), \
             patch("keep.document_store.DocumentStore"), \
             patch("keep.pending_summaries.PendingSummaryQueue"):
            kp = Keeper(store_path=tmp_path)
            try:
                kp._get_embedding_provider()
            except RuntimeError as e:
                msg = str(e)
                assert "keep-skill[local]" in msg
                assert "VOYAGE_API_KEY" in msg

    def test_store_config_accepts_none_embedding(self) -> None:
        """StoreConfig can be created with embedding=None."""
        from keep.config import StoreConfig
        config = StoreConfig(path=Path("/tmp/test"), embedding=None)
        assert config.embedding is None
        assert config.summarization.name == "truncate"  # default still works

    def test_save_config_handles_none_embedding(self, tmp_path) -> None:
        """save_config doesn't crash when embedding is None."""
        from keep.config import StoreConfig, save_config

        config = StoreConfig(path=tmp_path, config_dir=tmp_path, embedding=None)
        # Should not raise
        save_config(config)

        # Verify config file exists and doesn't have embedding section
        config_file = tmp_path / "keep.toml"
        assert config_file.exists()

    def test_load_config_reads_unified_remote_section(self, tmp_path, monkeypatch) -> None:
        """[remote] populates config.remote (the single remote-backend field)."""
        from keep.config import load_config

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_URL", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        monkeypatch.delenv("KEEPNOTES_PROJECT", raising=False)

        (tmp_path / "keep.toml").write_text(
            """
[store]
version = 2

[remote]
api_url = "https://api.example.test"
api_key = "kn_test_123"
project = "demo"
""".strip() + "\n",
            encoding="utf-8",
        )

        config = load_config(tmp_path)

        assert config.remote is not None
        assert config.remote.api_url == "https://api.example.test"
        assert config.remote.api_key == "kn_test_123"
        assert config.remote.project == "demo"

    def test_load_config_rejects_legacy_remote_store_section(self, tmp_path) -> None:
        """[remote_store] is no longer accepted — load_config raises."""
        from keep.config import load_config

        (tmp_path / "keep.toml").write_text(
            """
[store]
version = 2

[remote_store]
api_url = "https://api.example.test"
api_key = "kn_test_123"
""".strip() + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=r"\[remote_store\].*rename to \[remote\]"):
            load_config(tmp_path)

    def test_load_config_rejects_legacy_remote_task_section(self, tmp_path) -> None:
        """[remote_task] is no longer accepted — load_config raises."""
        from keep.config import load_config

        (tmp_path / "keep.toml").write_text(
            """
[store]
version = 2

[remote_task]
api_url = "https://api.example.test"
api_key = "kn_test_123"
""".strip() + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=r"\[remote_task\].*rename to \[remote\]"):
            load_config(tmp_path)

    def test_save_config_writes_unified_remote_section(self, tmp_path, monkeypatch) -> None:
        """save_config writes a single [remote] section."""
        from keep.config import RemoteConfig, StoreConfig, save_config

        monkeypatch.delenv("KEEPNOTES_API_URL", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)

        config = StoreConfig(
            path=tmp_path,
            config_dir=tmp_path,
            embedding=None,
            remote=RemoteConfig(
                api_url="https://store.example.test",
                api_key="kn_store",
                project="alpha",
            ),
        )

        save_config(config)

        saved = (tmp_path / "keep.toml").read_text(encoding="utf-8")
        assert "[remote]" in saved
        assert "[remote_store]" not in saved
        assert "[remote_task]" not in saved
        assert 'api_url = "https://store.example.test"' in saved
        assert 'api_key = "kn_store"' in saved
        assert 'project = "alpha"' in saved

    def test_save_config_preserves_persisted_remote_under_env_override(
        self, tmp_path, monkeypatch,
    ) -> None:
        """Ambient smoke/debug env must not erase or overwrite [remote]."""
        from keep.config import load_config, save_config

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.setenv("KEEPNOTES_API_URL", "https://env.example.test")
        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_env")
        monkeypatch.setenv("KEEPNOTES_PROJECT", "env-project")

        (tmp_path / "keep.toml").write_text(
            """
[store]
version = 2

[remote]
api_url = "https://file.example.test"
api_key = "kn_file"
project = "file-project"
""".strip() + "\n",
            encoding="utf-8",
        )

        config = load_config(tmp_path)
        assert config.remote is not None
        assert config.remote.api_url == "https://env.example.test"
        assert config.remote.api_key == "kn_env"
        assert config.remote.project == "env-project"

        save_config(config)

        saved = (tmp_path / "keep.toml").read_text(encoding="utf-8")
        assert "[remote]" in saved
        assert 'api_url = "https://file.example.test"' in saved
        assert 'api_key = "kn_file"' in saved
        assert 'project = "file-project"' in saved
        assert "kn_env" not in saved

    def test_save_config_chmods_persisted_remote_under_local_only(
        self, tmp_path, monkeypatch,
    ) -> None:
        """remote_persist secrets still require 0600 when remote is disabled."""
        from keep.config import load_config, save_config

        monkeypatch.setenv("KEEP_LOCAL_ONLY", "1")
        monkeypatch.delenv("KEEPNOTES_API_URL", raising=False)
        monkeypatch.delenv("KEEPNOTES_API_KEY", raising=False)
        config_path = tmp_path / "keep.toml"
        config_path.write_text(
            """
[store]
version = 2

[remote]
api_url = "https://file.example.test"
api_key = "kn_file"
project = "file-project"
""".strip() + "\n",
            encoding="utf-8",
        )
        config_path.chmod(0o644)

        config = load_config(tmp_path)
        assert config.remote is None
        assert config.remote_persist is not None

        save_config(config)

        assert config_path.stat().st_mode & 0o777 == 0o600
        saved = config_path.read_text(encoding="utf-8")
        assert "[remote]" in saved
        assert 'api_key = "kn_file"' in saved

    def test_save_config_omits_env_only_remote(self, tmp_path, monkeypatch) -> None:
        """Env-only remote credentials should not be written to keep.toml."""
        from keep.config import load_config, save_config

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)
        monkeypatch.setenv("KEEPNOTES_API_URL", "https://env.example.test")
        monkeypatch.setenv("KEEPNOTES_API_KEY", "kn_env")

        (tmp_path / "keep.toml").write_text(
            "[store]\nversion = 2\n",
            encoding="utf-8",
        )

        config = load_config(tmp_path)
        assert config.remote is not None
        assert config.remote_from_env is True

        save_config(config)

        saved = (tmp_path / "keep.toml").read_text(encoding="utf-8")
        assert "[remote]" not in saved
        assert "kn_env" not in saved
