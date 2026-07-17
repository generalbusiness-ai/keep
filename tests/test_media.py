"""Tests for media description: state doc rules, dispatch, and task workflow."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from keep.api import Keeper
from keep.config import ProviderConfig
from keep.providers.base import MediaDescriber, get_registry


# -----------------------------------------------------------------------------
# Protocol Tests
# -----------------------------------------------------------------------------

class TestMediaDescriberProtocol:
    """Verify MediaDescriber protocol compliance."""

    def test_protocol_is_runtime_checkable(self):
        class FakeDescriber:
            def describe(self, path: str, content_type: str) -> str | None:
                return "a test description"

        assert isinstance(FakeDescriber(), MediaDescriber)

    def test_non_conforming_class_fails_check(self):
        class NotADescriber:
            def summarize(self, content: str) -> str:
                return content

        assert not isinstance(NotADescriber(), MediaDescriber)


# -----------------------------------------------------------------------------
# Registry Tests
# -----------------------------------------------------------------------------

class TestMediaRegistry:
    """Tests for media describer registry."""

    def test_register_and_create(self):
        registry = get_registry()

        class TestDescriber:
            def __init__(self, greeting="hello"):
                self.greeting = greeting
            def describe(self, path: str, content_type: str) -> str | None:
                return f"{self.greeting}: {path}"

        registry.register_media("test-media", TestDescriber)
        try:
            describer = registry.create_media("test-media", {"greeting": "hi"})
            assert describer.describe("/img.jpg", "image/jpeg") == "hi: /img.jpg"
        finally:
            del registry._media_providers["test-media"]

    def test_create_unknown_raises(self):
        registry = get_registry()
        with pytest.raises(ValueError, match="Unknown media provider"):
            registry.create_media("nonexistent-provider")


# -----------------------------------------------------------------------------
# Config Tests
# -----------------------------------------------------------------------------

class TestMediaConfig:
    """Tests for media configuration."""

    def test_config_media_defaults_to_none(self, tmp_path):
        from keep.config import StoreConfig
        config = StoreConfig(path=tmp_path)
        assert config.media is None

    def test_config_roundtrip_with_media(self, tmp_path):
        from keep.config import StoreConfig, ProviderConfig, save_config, load_config

        config = StoreConfig(
            path=tmp_path,
            config_dir=tmp_path,
            media=ProviderConfig("mlx", {
                "vision_model": "test-vision",
                "whisper_model": "test-whisper",
            }),
        )
        save_config(config)
        loaded = load_config(tmp_path)

        assert loaded.media is not None
        assert loaded.media.name == "mlx"
        assert loaded.media.params["vision_model"] == "test-vision"

    def test_config_roundtrip_without_media(self, tmp_path):
        from keep.config import StoreConfig, save_config, load_config

        config = StoreConfig(path=tmp_path, config_dir=tmp_path, media=None)
        save_config(config)
        # Media re-detection must find nothing regardless of which optional
        # packages (mlx-vlm, mlx-whisper) happen to be installed locally.
        with (
            patch("keep.config._detect_ollama", return_value=None),
            patch("keep.config._local_dependency_available", return_value=False),
        ):
            loaded = load_config(tmp_path)

        assert loaded.media is None


# -----------------------------------------------------------------------------
# LockedMediaDescriber Tests
# -----------------------------------------------------------------------------

class TestLockedMediaDescriber:
    """Tests for locked media describer delegation."""

    def test_locked_describer_delegates(self, tmp_path):
        from keep.model_lock import LockedMediaDescriber

        inner = MagicMock()
        inner.describe.return_value = "a cat"
        locked = LockedMediaDescriber(inner, tmp_path / ".media.lock")

        result = locked.describe("/test.jpg", "image/jpeg")
        assert result == "a cat"
        inner.describe.assert_called_once_with("/test.jpg", "image/jpeg")

    def test_locked_describer_release(self, tmp_path):
        from keep.model_lock import LockedMediaDescriber

        inner = MagicMock()
        locked = LockedMediaDescriber(inner, tmp_path / ".media.lock")
        locked.release()
        assert locked._provider is None


# -----------------------------------------------------------------------------
# After-write state doc rule evaluation
# -----------------------------------------------------------------------------

class TestAfterWriteStateDoc:
    """Verify after-write state doc rules drive background tasks.

    These tests evaluate the state doc directly -- no Keeper, no mocks.
    If a rule is wrong here, the entire post-write pipeline is wrong.
    """

    @pytest.fixture
    def after_write_doc(self, mock_providers, tmp_path):
        kp = Keeper(store_path=tmp_path)
        kp.ensure_sysdocs()
        doc = kp._load_state_doc("after-write")
        assert doc is not None
        return doc

    def _eval(self, doc, system=None, **item_overrides):
        from keep.state_doc import evaluate_state_doc
        # Unified item context shape
        item = {
            "id": "%test-item",
            "summary": "",
            "content_length": 50,
            "content_type": "",
            "uri": "",
            "created": "",
            "updated": "",
            "accessed": "",
            "tags": {},
        }
        item.update(item_overrides)
        sys = {"has_media_provider": True}
        if system is not None:
            sys.update(system)
        ctx = {"item": item, "params": {"max_summary_length": 2000, "item_id": item["id"]}, "system": sys}
        result = evaluate_state_doc(doc, ctx, run_action=None)
        return [a["action"] for a in result.actions]

    def test_inline_text_fires_analyze_and_tag(self, after_write_doc):
        """Inline text above min length → analyze + auto_tag + resolve_duplicates."""
        actions = self._eval(after_write_doc, content_length=1000)
        assert "analyze" in actions
        assert "auto_tag" in actions
        assert "resolve_duplicates" in actions

    def test_short_content_skips_analyze(self, after_write_doc):
        """Short inline notes skip analyze until they cross the size threshold."""
        actions = self._eval(after_write_doc, content_length=50)
        assert "analyze" not in actions
        assert "auto_tag" in actions  # tagging still runs

    def test_link_stub_skips_analyze(self, after_write_doc):
        """Extracted-link stubs should not spend analyzer work on placeholder content."""
        actions = self._eval(
            after_write_doc,
            content_length=1000,
            tags={"_source": "link"},
        )
        assert "analyze" not in actions

    def test_auto_vivify_stub_skips_analyze(self, after_write_doc):
        """Auto-vivified stubs also skip analyze."""
        actions = self._eval(
            after_write_doc,
            content_length=1000,
            tags={"_source": "auto-vivify"},
        )
        assert "analyze" not in actions

    def test_long_content_fires_summarize(self, after_write_doc):
        """Content exceeding max_summary_length fires summarize."""
        actions = self._eval(after_write_doc, content_length=5000)
        assert "summarize" in actions

    def test_system_note_skips_analyze_and_tag(self, after_write_doc):
        """System notes (dot-prefix IDs) skip analyze, auto_tag, and resolve_duplicates."""
        actions = self._eval(after_write_doc, id=".meta/test")
        assert "analyze" not in actions
        assert "auto_tag" not in actions
        assert "resolve_duplicates" not in actions

    def test_image_uri_fires_describe(self, after_write_doc):
        """URI-backed image content fires describe."""
        actions = self._eval(after_write_doc,
                             uri="file:///tmp/photo.jpg",
                             content_type="image/jpeg")
        assert "describe" in actions

    def test_text_uri_skips_describe(self, after_write_doc):
        """URI-backed text content does NOT fire describe."""
        actions = self._eval(after_write_doc,
                             uri="file:///tmp/doc.txt",
                             content_type="text/plain")
        assert "describe" not in actions

    def test_remote_binary_uri_skips_describe(self, after_write_doc):
        """Remote binary URIs skip describe because no local media path survives."""
        actions = self._eval(
            after_write_doc,
            uri="https://example.com/policy.pdf",
            content_type="application/pdf",
        )
        assert "describe" not in actions

    def test_ocr_pages_fires_ocr(self, after_write_doc):
        """Items with _ocr_pages tag and URI fire OCR."""
        actions = self._eval(after_write_doc,
                             uri="file:///scan.pdf",
                             tags={"_ocr_pages": "[1,2]"})
        assert "ocr" in actions

    def test_no_content_skips_tag(self, after_write_doc):
        """Empty inline content skips both auto_tag and analyze."""
        actions = self._eval(after_write_doc, content_length=0)
        assert "auto_tag" not in actions
        assert "analyze" not in actions

    def test_markdown_fires_extract_links(self, after_write_doc):
        """Markdown content fires extract_links."""
        actions = self._eval(after_write_doc, content_type="text/markdown")
        assert "extract_links" in actions

    def test_plain_text_fires_extract_links(self, after_write_doc):
        """Plain text content fires extract_links for bare URLs and emails."""
        actions = self._eval(after_write_doc, content_type="text/plain")
        assert "extract_links" in actions

    def test_pdf_extract_links_receives_doc_links(self, after_write_doc):
        """PDF after-write passes provider-extracted doc_links into extract_links."""
        from keep.state_doc import evaluate_state_doc

        ctx = {
            "item": {
                "id": "%pdf-test",
                "summary": "",
                "content_length": 50,
                "content_type": "application/pdf",
                "uri": "https://example.com/policy.pdf",
                "created": "",
                "updated": "",
                "accessed": "",
                "tags": {},
            },
            "params": {
                "item_id": "%pdf-test",
                "max_summary_length": 2000,
                "metadata": {
                    "doc_links": ["https://travel.example.com"],
                },
            },
            "system": {"has_media_provider": True},
        }
        result = evaluate_state_doc(after_write_doc, ctx, run_action=None)
        linked = next(action for action in result.actions if action["action"] == "extract_links")
        assert linked["params"]["doc_links"] == ["https://travel.example.com"]


# -----------------------------------------------------------------------------
# Integration: put() → state doc → work queue
# -----------------------------------------------------------------------------

def _make_mock_doc(uri, content, content_type, tags=None, metadata=None):
    """Create a mock Document for testing."""
    mock_doc = MagicMock()
    mock_doc.uri = uri
    mock_doc.content = content
    mock_doc.content_type = content_type
    mock_doc.metadata = metadata
    mock_doc.tags = tags
    return mock_doc


def _keeper_bootstrap_sysdocs(kp):
    """Ensure store-backed system docs exist for flow-backed APIs."""
    kp.ensure_sysdocs()


def _claimed_flow_items(kp, limit=20):
    """Claim work queue items and return them."""
    claimed = kp._work_queue.claim("test", limit=limit)
    return claimed


def _flow_item_context(kp):
    """Claim the single after-write flow item and return its item context."""
    items = _claimed_flow_items(kp)
    flow_items = [i for i in items if i.kind == "flow"]
    if not flow_items:
        return None
    params = flow_items[0].input.get("params", {})
    return params.get("item", {})


class TestAfterWriteDispatch:
    """Verify _dispatch_after_write_flow enqueues correct flow work items.

    The after-write state doc evaluates these context fields at daemon
    execution time to decide which actions fire.
    """

    def test_image_put_enqueues_flow_with_media_context(self, mock_providers, tmp_path):
        """Image URI with media config → flow item with media context."""
        from keep.api import Keeper

        mock_doc = _make_mock_doc(
            "file:///test.jpg",
            "Dimensions: 1920x1080\nCamera: Canon EOS R5",
            "image/jpeg",
            tags={"dimensions": "1920x1080"},
        )
        mock_providers["document"].fetch = lambda uri: mock_doc

        kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)
        kp._config.media = ProviderConfig("ollama", {"model": "test"})

        kp.put(uri="file:///test.jpg")

        ctx = _flow_item_context(kp)
        assert ctx is not None, "Should enqueue a flow item"
        assert ctx["content_type"] == "image/jpeg"
        assert ctx["uri"].startswith("file://")
        kp.close()

    def test_image_put_without_media_config(self, mock_providers, tmp_path, monkeypatch):
        """Image URI without media config → flow item without media provider."""
        from keep.api import Keeper

        mock_doc = _make_mock_doc(
            "file:///test.jpg", "Dimensions: 100x100", "image/jpeg",
        )
        mock_providers["document"].fetch = lambda uri: mock_doc

        monkeypatch.delenv("KEEP_CONFIG", raising=False)
        # Media re-detection must find nothing regardless of which optional
        # packages (mlx-vlm, mlx-whisper) happen to be installed locally.
        with (
            patch("keep.config._detect_ollama", return_value=None),
            patch("keep.config._local_dependency_available", return_value=False),
        ):
            kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)
        assert kp._config.media is None

        kp.put(uri="file:///test.jpg")

        items = _claimed_flow_items(kp)
        flow_items = [i for i in items if i.kind == "flow"]
        assert len(flow_items) == 1
        params = flow_items[0].input.get("params", {})
        assert params.get("system", {}).get("has_media_provider") is False
        kp.close()

    def test_audio_put_enqueues_flow(self, mock_providers, tmp_path):
        """Audio URI with media config → flow item with audio context."""
        from keep.api import Keeper

        mock_doc = _make_mock_doc(
            "file:///test.mp3", "Title: Song\nArtist: Band", "audio/mpeg",
            tags={"title": "Song"},
        )
        mock_providers["document"].fetch = lambda uri: mock_doc

        kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)
        kp._config.media = ProviderConfig("ollama", {"model": "test"})

        kp.put(uri="file:///test.mp3")

        ctx = _flow_item_context(kp)
        assert ctx is not None
        assert ctx["content_type"] == "audio/mpeg"
        assert ctx["uri"].startswith("file://")
        kp.close()

    def test_remote_pdf_put_skips_describe_context(self, mock_providers, tmp_path):
        """Remote PDFs should not advertise describe-ready media context."""
        from keep.api import Keeper

        mock_doc = _make_mock_doc(
            "https://example.com/policy.pdf",
            "Corporate travel policy",
            "application/pdf",
        )
        mock_providers["document"].fetch = lambda uri: mock_doc

        kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)
        kp._config.media = ProviderConfig("ollama", {"model": "test"})

        kp.put(uri="https://example.com/policy.pdf")

        ctx = _flow_item_context(kp)
        assert ctx is not None
        assert ctx["uri"] == "https://example.com/policy.pdf"
        assert ctx["content_type"] == "application/pdf"
        kp.close()

    def test_text_uri_context(self, mock_providers, tmp_path):
        """Text/markdown URI → flow item with text content type."""
        from keep.api import Keeper

        mock_doc = _make_mock_doc(
            "file:///test.md", "# Hello World", "text/markdown",
        )
        mock_providers["document"].fetch = lambda uri: mock_doc

        kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)
        kp._config.media = ProviderConfig("ollama", {"model": "test"})

        kp.put(uri="file:///test.md")

        ctx = _flow_item_context(kp)
        assert ctx is not None
        assert ctx["content_type"] == "text/markdown"
        kp.close()

    def test_inline_text_context(self, mock_providers, tmp_path):
        """Inline text → flow item with content context, no URI."""
        from keep.api import Keeper

        kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)

        kp.put("A note about architecture. " * 30, id="note1")  # >500 chars

        ctx = _flow_item_context(kp)
        assert ctx is not None
        assert ctx["content_length"] > 0
        assert ctx["uri"] == ""
        assert ctx["content_type"] == "text/markdown"
        kp.close()

    def test_system_note_enqueues_nothing(self, mock_providers, tmp_path):
        """System note (dot-prefix) → no work enqueued at all."""
        from keep.api import Keeper

        kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)

        kp.put("System data", id=".sys/test")

        if kp._work_queue is not None:
            items = _claimed_flow_items(kp)
            assert len(items) == 0, "System notes should not enqueue any work"
        kp.close()


# -----------------------------------------------------------------------------
# Describe task workflow
# -----------------------------------------------------------------------------

class TestDescribeTaskWorkflow:
    """Test the describe task workflow (background execution path)."""

    def test_describe_enriches_summary(self, mock_providers, tmp_path):
        """Describe appends description to existing summary."""
        from keep.api import Keeper
        from keep.task_workflows import TaskRequest, run_local_task

        kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)

        # Create item with initial summary
        kp.put("Dimensions: 100x100", id="img1")

        # Set up a mock media describer
        mock_describer = MagicMock()
        mock_describer.describe.return_value = "A photo of a sunset"
        kp._media_describer = mock_describer
        kp._config.media = ProviderConfig("ollama", {"model": "test"})

        # Create a file for the describer to find
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image data")

        req = TaskRequest(
            task_type="describe",
            id="img1",
            collection=kp._resolve_doc_collection(),
            content="",
            metadata={"uri": str(test_file), "content_type": "image/jpeg"},
        )
        # Patch path validation (tmp_path is outside home)
        with patch("keep.actions.describe.validate_path_within_home"):
            result = run_local_task(kp, req)

        assert result.status == "applied"
        # Verify the summary was enriched
        item = kp.get("img1")
        assert "Description:" in item.summary
        assert "A photo of a sunset" in item.summary
        kp.close()

    def test_describe_skips_when_no_provider(self, mock_providers, tmp_path):
        """Describe skips gracefully without a media provider."""
        from keep.api import Keeper
        from keep.task_workflows import TaskRequest, run_local_task

        with patch("keep.config._detect_ollama", return_value=None):
            kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)
        kp.put("Dimensions: 100x100", id="img1")

        req = TaskRequest(
            task_type="describe",
            id="img1",
            collection=kp._resolve_doc_collection(),
            content="",
            metadata={"uri": "/test.jpg", "content_type": "image/jpeg"},
        )
        result = run_local_task(kp, req)
        assert result.status == "skipped"
        kp.close()

    def test_describe_skips_on_empty_description(self, mock_providers, tmp_path):
        """Describe skips when describer returns empty string."""
        from keep.api import Keeper
        from keep.task_workflows import TaskRequest, run_local_task

        kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)
        kp.put("Original summary", id="img1")

        mock_describer = MagicMock()
        mock_describer.describe.return_value = "   "
        kp._media_describer = mock_describer
        kp._config.media = ProviderConfig("ollama", {"model": "test"})

        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake")

        req = TaskRequest(
            task_type="describe",
            id="img1",
            collection=kp._resolve_doc_collection(),
            content="",
            metadata={"uri": str(test_file), "content_type": "image/jpeg"},
        )
        with patch("keep.actions.describe.validate_path_within_home"):
            result = run_local_task(kp, req)
        assert result.status == "skipped"

        # Original summary unchanged
        item = kp.get("img1")
        assert "Description:" not in item.summary
        kp.close()

    def test_describe_skips_non_local_uri(self, mock_providers, tmp_path):
        """Describe rejects remote URIs instead of resolving them as local paths."""
        from keep.api import Keeper
        from keep.task_workflows import TaskRequest, run_local_task

        kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)
        kp.put("Original summary", id="img1")

        mock_describer = MagicMock()
        kp._media_describer = mock_describer
        kp._config.media = ProviderConfig("ollama", {"model": "test"})

        req = TaskRequest(
            task_type="describe",
            id="img1",
            collection=kp._resolve_doc_collection(),
            content="",
            metadata={"uri": "https://example.com/test.jpg", "content_type": "image/jpeg"},
        )
        result = run_local_task(kp, req)

        assert result.status == "skipped"
        assert result.details["reason"] == "non_local_uri"
        mock_describer.describe.assert_not_called()
        kp.close()

    def test_describe_does_not_overwrite_markdown_authored_body(
        self, mock_providers, tmp_path,
    ):
        """Describe leaves markdown-authored note bodies unchanged."""
        from keep.api import Keeper
        from keep.task_workflows import TaskRequest, run_local_task

        kp = Keeper(store_path=tmp_path)
        _keeper_bootstrap_sysdocs(kp)
        original = "Original authored body"
        kp._put_direct(
            content=original,
            id="img1",
            queue_background_tasks=False,
            _body_authority="markdown",
        )

        mock_describer = MagicMock()
        mock_describer.describe.return_value = "A photo of a sunset"
        kp._media_describer = mock_describer
        kp._config.media = ProviderConfig("ollama", {"model": "test"})

        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image data")

        req = TaskRequest(
            task_type="describe",
            id="img1",
            collection=kp._resolve_doc_collection(),
            content="",
            metadata={"uri": str(test_file), "content_type": "image/jpeg"},
        )
        with patch("keep.actions.describe.validate_path_within_home"):
            result = run_local_task(kp, req)

        assert result.status == "applied"
        item = kp.get("img1")
        assert item.summary == original
        assert item.tags["_body_authority"] == "markdown"
        kp.close()


# -----------------------------------------------------------------------------
# MLX Describer Unit Tests (no real models)
# -----------------------------------------------------------------------------

class TestMLXMediaDescriber:
    """Tests for MLX media describer."""

    def test_returns_none_for_text(self):
        from keep.providers.mlx import MLXMediaDescriber

        describer = object.__new__(MLXMediaDescriber)
        describer._vision = None
        describer._whisper = None
        describer._vision_checked = True
        describer._whisper_checked = True

        assert describer.describe("/test.txt", "text/plain") is None

    def test_image_delegates_to_vision(self):
        from keep.providers.mlx import MLXMediaDescriber

        describer = object.__new__(MLXMediaDescriber)
        mock_vision = MagicMock()
        mock_vision.describe.return_value = "A cat"
        describer._vision = mock_vision
        describer._vision_checked = True
        describer._whisper = None
        describer._whisper_checked = True

        result = describer.describe("/cat.jpg", "image/jpeg")
        assert result == "A cat"
        mock_vision.describe.assert_called_once_with("/cat.jpg", "image/jpeg")

    def test_audio_delegates_to_whisper(self):
        from keep.providers.mlx import MLXMediaDescriber

        describer = object.__new__(MLXMediaDescriber)
        describer._vision = None
        describer._vision_checked = True
        mock_whisper = MagicMock()
        mock_whisper.describe.return_value = "Hello world"
        describer._whisper = mock_whisper
        describer._whisper_checked = True

        result = describer.describe("/speech.mp3", "audio/mpeg")
        assert result == "Hello world"
        mock_whisper.describe.assert_called_once_with("/speech.mp3", "audio/mpeg")
