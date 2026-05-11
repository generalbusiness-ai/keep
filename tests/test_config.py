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

    def test_load_config_treats_legacy_remote_as_remote_store(self, tmp_path, monkeypatch) -> None:
        """Legacy [remote] config still routes the authoritative store."""
        from keep.config import load_config

        monkeypatch.delenv("KEEP_LOCAL_ONLY", raising=False)

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

        assert config.remote is None
        assert config.remote_store is not None
        assert config.remote_store.api_url == "https://api.example.test"
        assert config.remote_store.api_key == "kn_test_123"
        assert config.remote_store.project == "demo"

    def test_save_config_writes_remote_store_and_remote_task_sections(self, tmp_path) -> None:
        """save_config persists authoritative-store and task delegation separately."""
        from keep.config import RemoteConfig, StoreConfig, save_config

        config = StoreConfig(
            path=tmp_path,
            config_dir=tmp_path,
            embedding=None,
            remote_store=RemoteConfig(
                api_url="https://store.example.test",
                api_key="kn_store",
                project="alpha",
            ),
            remote=RemoteConfig(
                api_url="https://tasks.example.test",
                api_key="kn_tasks",
                project="beta",
            ),
        )

        save_config(config)

        saved = (tmp_path / "keep.toml").read_text(encoding="utf-8")
        assert "[remote_store]" in saved
        assert 'api_url = "https://store.example.test"' in saved
        assert 'project = "alpha"' in saved
        assert "[remote_task]" in saved
        assert 'api_url = "https://tasks.example.test"' in saved
        assert 'project = "beta"' in saved
