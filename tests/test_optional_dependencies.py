"""Regression coverage for optional dependency import diagnostics."""

import builtins
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from keep.optional_dependencies import (
    is_missing_optional_dependency,
    probe_optional_dependency,
)
from keep.providers.base import get_registry
from keep.providers.embeddings import SentenceTransformerEmbedding
from keep.providers.mlx import MLXEmbedding


def test_missing_dependency_matches_only_requested_top_level_package() -> None:
    """Nested missing modules must be treated as broken installed stacks."""
    missing_package = ModuleNotFoundError(
        "No module named 'sentence_transformers'",
        name="sentence_transformers",
    )
    missing_transitive = ModuleNotFoundError(
        "No module named 'tokenizers'",
        name="tokenizers",
    )

    assert is_missing_optional_dependency(
        missing_package, "sentence_transformers",
    ) is True
    assert is_missing_optional_dependency(
        missing_transitive, "sentence_transformers",
    ) is False


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (
            ModuleNotFoundError(
                "No module named 'sentence_transformers'",
                name="sentence_transformers",
            ),
            "missing",
            None,
        ),
        (
            ImportError("tokenizers<=0.23.0 is required, found 0.23.1"),
            "broken",
            "tokenizers<=0.23.0 is required, found 0.23.1",
        ),
    ],
)
def test_probe_classifies_missing_and_broken_imports(
    monkeypatch,
    error: ImportError,
    expected_status: str,
    expected_detail: str | None,
) -> None:
    """Setup probes retain the real error for installed-but-broken packages."""
    def fail_import(_module_name: str) -> None:
        raise error

    monkeypatch.setattr(
        "keep.optional_dependencies.importlib.import_module", fail_import,
    )

    assert probe_optional_dependency("sentence_transformers") == (
        expected_status,
        expected_detail,
    )


def _raise_for_import(target: str, error: ImportError):
    """Return an import hook that fails only for the requested module."""
    real_import = builtins.__import__

    def importing(name, globals=None, locals=None, fromlist=(), level=0):
        if name == target:
            raise error
        return real_import(name, globals, locals, fromlist, level)

    return importing


def test_sentence_transformer_provider_guides_genuinely_missing_package() -> None:
    """A missing top-level package still receives the concise install hint."""
    error = ModuleNotFoundError(
        "No module named 'sentence_transformers'",
        name="sentence_transformers",
    )

    with patch("builtins.__import__", side_effect=_raise_for_import(
        "sentence_transformers", error,
    )):
        with pytest.raises(RuntimeError, match="requires 'sentence-transformers'"):
            SentenceTransformerEmbedding(model="all-MiniLM-L6-v2")


def test_sentence_transformer_provider_surfaces_broken_stack_error() -> None:
    """The provider factory must not replace a transitive version conflict."""
    detail = "tokenizers<=0.23.0 is required, found tokenizers==0.23.1"

    with patch("builtins.__import__", side_effect=_raise_for_import(
        "sentence_transformers", ImportError(detail),
    )):
        with pytest.raises(RuntimeError) as caught:
            get_registry().create_embedding(
                "sentence-transformers",
                {"model": "all-MiniLM-L6-v2"},
            )

    message = str(caught.value)
    assert detail in message
    assert "requires 'sentence-transformers' library" not in message


def test_mlx_provider_surfaces_sentence_transformer_stack_error() -> None:
    """MLX must also preserve failures from an installed local model stack."""
    fake_mlx = ModuleType("mlx")
    fake_mlx_core = ModuleType("mlx.core")
    fake_mlx.core = fake_mlx_core
    detail = "tokenizers version is incompatible"

    with (
        patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_mlx_core}),
        patch("builtins.__import__", side_effect=_raise_for_import(
            "sentence_transformers", ImportError(detail),
        )),
    ):
        with pytest.raises(ImportError, match=detail):
            MLXEmbedding(model="all-MiniLM-L6-v2")
