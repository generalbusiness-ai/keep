"""Tests for filesystem-backed document ingestion."""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from keep.providers.documents import FileDocumentProvider


def test_file_provider_warns_when_following_symlink(tmp_path, monkeypatch, caplog):
    """Symlink traversal stays allowed, but it must be visible in logs."""
    target = tmp_path / "target.txt"
    target.write_text("hello through link", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    monkeypatch.setattr(
        "keep.providers.documents.validate_path_within_home",
        lambda path: path,
    )
    caplog.set_level(logging.WARNING, logger="keep.providers.documents")

    doc = FileDocumentProvider().fetch(f"file://{link}")

    assert "hello through link" in doc.content
    assert any("File ingest followed symlink" in rec.message for rec in caplog.records)

# ---------------------------------------------------------------------------


class TestIsPrivateUrl:
    """Tests for HttpDocumentProvider._is_private_url SSRF protection."""

    @pytest.fixture
    def provider(self):
        from keep.providers.documents import HttpDocumentProvider
        return HttpDocumentProvider()

    def test_loopback_ipv4(self, provider) -> None:
        assert provider._is_private_url("http://127.0.0.1/secret") is True

    def test_loopback_ipv6(self, provider) -> None:
        assert provider._is_private_url("http://[::1]/secret") is True

    def test_private_10_range(self, provider) -> None:
        assert provider._is_private_url("http://10.0.0.1/internal") is True

    def test_private_172_range(self, provider) -> None:
        assert provider._is_private_url("http://172.16.0.1/internal") is True

    def test_private_192_range(self, provider) -> None:
        assert provider._is_private_url("http://192.168.1.1/internal") is True

    def test_link_local(self, provider) -> None:
        assert provider._is_private_url("http://169.254.169.254/metadata") is True

    def test_cloud_metadata_endpoint(self, provider) -> None:
        assert provider._is_private_url("http://metadata.google.internal/v1") is True

    def test_no_hostname(self, provider) -> None:
        """URLs without a hostname are blocked."""
        assert provider._is_private_url("http:///path") is True

    def test_public_ip(self, provider) -> None:
        assert provider._is_private_url("http://8.8.8.8/dns") is False

    def test_public_domain(self, provider) -> None:
        """Real public domains are allowed."""
        assert provider._is_private_url("https://example.com/page") is False

    def test_localhost_name(self, provider) -> None:
        """Localhost resolves to 127.0.0.1, should be blocked."""
        assert provider._is_private_url("http://localhost/secret") is True

    def test_unspecified_address(self, provider) -> None:
        """0.0.0.0 (unspecified) should be blocked."""
        assert provider._is_private_url("http://0.0.0.0/path") is True

    def test_multicast_address(self, provider) -> None:
        """Multicast addresses should be blocked."""
        assert provider._is_private_url("http://224.0.0.1/path") is True

    def test_fetch_blocks_private(self, provider) -> None:
        """fetch() raises IOError for private URLs."""
        with pytest.raises(IOError, match="private"):
            provider.fetch("http://127.0.0.1/secret")

    def test_fetch_blocks_redirect_to_private(self, provider) -> None:
        """fetch() blocks redirects to private addresses."""
        mock_resp = MagicMock()
        mock_resp.is_redirect = True
        mock_resp.headers = {"Location": "http://127.0.0.1/internal"}
        mock_resp.close = MagicMock()

        mock_session = MagicMock()
        mock_session.build_request.return_value = object()
        mock_session.send.return_value = mock_resp

        with patch("keep.providers.http.http_session", return_value=mock_session):
            with pytest.raises(IOError, match="private"):
                provider.fetch("https://example.com/redirect")


# ---------------------------------------------------------------------------
# Embedding provider absent scenarios
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------


class TestFileBirthtime:
    """File creation time should be used as created_at for file:// URIs."""

    @pytest.fixture
    def home_tmp(self):
        """Temp dir under home so FileDocumentProvider's safety check passes."""
        d = Path.home() / ".keep-test-birthtime"
        d.mkdir(exist_ok=True)
        yield d
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    def test_file_provider_includes_birthtime(self, home_tmp):
        """FileDocumentProvider.fetch() includes birthtime in metadata."""
        from keep.providers.documents import FileDocumentProvider

        f = home_tmp / "note.md"
        f.write_text("hello")
        provider = FileDocumentProvider()
        doc = provider.fetch(str(f))
        # macOS always has st_birthtime; skip on platforms that don't
        if hasattr(os.stat_result, "st_birthtime"):
            assert "birthtime" in doc.metadata
            assert isinstance(doc.metadata["birthtime"], float)
        else:
            # On platforms without birthtime, key should be absent
            assert "birthtime" not in doc.metadata

    def test_put_file_uses_birthtime_as_created(self, mock_providers, home_tmp):
        """put() with a file:// URI sets _created from file birthtime."""
        from keep.api import Keeper
        from keep.providers.documents import FileDocumentProvider

        f = home_tmp / "old-note.md"
        f.write_text("historical content")

        kp = Keeper(store_path=home_tmp / "store")
        kp._document_provider = FileDocumentProvider()

        before = datetime.now(timezone.utc)
        item = kp.put(uri=f"file://{f}")

        if hasattr(os.stat_result, "st_birthtime"):
            created_str = item.tags["_created"]
            created = datetime.fromisoformat(created_str)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            # File was just created, so birthtime should be before 'now'
            # and close to it (within a few seconds)
            assert created < before or (created - before).total_seconds() < 2
        # Without birthtime, _created falls back to current time (existing behavior)

    def test_put_file_explicit_created_at_wins(self, mock_providers, home_tmp):
        """Explicit created_at overrides file birthtime."""
        from keep.api import Keeper
        from keep.providers.documents import FileDocumentProvider

        f = home_tmp / "note.md"
        f.write_text("content")

        kp = Keeper(store_path=home_tmp / "store")
        kp._document_provider = FileDocumentProvider()

        explicit = "2020-01-01T00:00:00+00:00"
        item = kp.put(uri=f"file://{f}", created_at=explicit)
        assert item.tags["_created"].startswith("2020-01-01")


# ---------------------------------------------------------------------------
# Part immutability
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Audio metadata (tinytag)
# ---------------------------------------------------------------------------


class TestAudioMetadata:
    """Audio ingest reads real files through tinytag, not a mocked extractor.

    These exercise the dependency itself so a tinytag upgrade that changes
    attribute names or parsing behaviour fails here rather than in the field.
    """

    @pytest.fixture
    def allow_tmp_path(self, monkeypatch):
        """tmp_path lives outside $HOME; bypass the home-directory guard."""
        monkeypatch.setattr(
            "keep.providers.documents.validate_path_within_home",
            lambda path: path,
        )

    def test_wav_metadata_is_extracted(self, tmp_path, allow_tmp_path):
        """A real WAV yields audio properties as structured text."""
        import struct
        import wave

        wav = tmp_path / "tone.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(struct.pack("<" + "h" * 8000, *([0] * 8000)))  # 1 s silence

        doc = FileDocumentProvider().fetch(f"file://{wav}")

        assert doc.content_type == "audio/wav"
        assert "Duration: 0:01" in doc.content
        assert "Sample Rate: 8000 Hz" in doc.content

    def test_corrupt_audio_degrades_to_placeholder(self, tmp_path, allow_tmp_path):
        """Unparseable audio returns the filename placeholder and never hangs."""
        bad = tmp_path / "garbage.wav"
        bad.write_bytes(b"RIFF\x10\x00\x00\x00WAVEjunkjunkjunk")

        doc = FileDocumentProvider().fetch(f"file://{bad}")

        assert doc.content_type == "audio/wav"
        assert doc.content == "Audio file: garbage.wav"
