"""Regression tests for provider HTTP calls using httpx."""

import httpx
import pytest

from keep.providers.embeddings import MistralEmbedding, OllamaEmbedding, VoyageEmbedding
from keep.providers.llm import MistralContentExtractor, MistralSummarization, OllamaSummarization
from keep.providers.ollama_utils import ollama_ensure_model
from keep.types import user_agent


def test_shared_http_session_uses_httpx_and_user_agent():
    from keep.providers import http as provider_http

    provider_http.close_http_session()
    session = provider_http.http_session()
    try:
        assert isinstance(session, httpx.Client)
        assert session.headers["User-Agent"] == user_agent()
    finally:
        provider_http.close_http_session()


def test_ollama_ensure_model_uses_shared_httpx_session(monkeypatch):
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "nomic-embed-text:latest"}]}

    class FakeSession:
        def __init__(self):
            self.calls = []

        def get(self, url, *, timeout):
            self.calls.append((url, timeout))
            return FakeResponse()

    session = FakeSession()
    monkeypatch.setattr("keep.providers.ollama_utils.ollama_session", lambda: session)

    ollama_ensure_model("http://localhost:11434", "nomic-embed-text")

    assert session.calls == [("http://localhost:11434/api/tags", 5)]


def test_ollama_embedding_uses_httpx_success_property(monkeypatch):
    class FakeResponse:
        is_success = True
        status_code = 200
        text = ""

        def json(self):
            return {"embedding": [0.1, 0.2, 0.3]}

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("keep.providers.ollama_utils.ollama_ensure_model", lambda *args, **kwargs: None)
    monkeypatch.setattr("keep.providers.ollama_utils.ollama_session", lambda: FakeSession())

    provider = OllamaEmbedding(model="nomic-embed-text", base_url="http://localhost:11434")

    assert provider.embed("hello") == [0.1, 0.2, 0.3]


def test_ollama_summarization_uses_httpx_success_property(monkeypatch):
    class FakeResponse:
        is_success = True
        status_code = 200
        text = ""

        def json(self):
            return {"message": {"content": "short summary"}}

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("keep.providers.ollama_utils.ollama_ensure_model", lambda *args, **kwargs: None)
    monkeypatch.setattr("keep.providers.ollama_utils.ollama_session", lambda: FakeSession())

    provider = OllamaSummarization(model="llama3.2", base_url="http://localhost:11434")

    assert provider.generate("system", "user") == "short summary"


def test_voyage_request_errors_are_reported(monkeypatch):
    class FailingSession:
        def post(self, *args, **kwargs):
            request = httpx.Request("POST", "https://api.voyageai.com/v1/embeddings")
            raise httpx.ConnectError("network unreachable", request=request)

    monkeypatch.setattr("keep.providers.http.http_session", lambda: FailingSession())
    provider = VoyageEmbedding(model="voyage-3-lite", api_key="test-key")

    with pytest.raises(RuntimeError, match="Cannot reach Voyage AI API"):
        provider.embed("hello")


# ---------------------------------------------------------------------------
# Mistral providers: REST-only, no mistralai SDK
# ---------------------------------------------------------------------------


class _RecordingResponse:
    """Minimal stand-in for httpx.Response used by Mistral tests."""

    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.headers: dict = {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}",
                request=httpx.Request("POST", "https://api.mistral.ai/"),
                response=httpx.Response(self.status_code),
            )


class _RecordingSession:
    """Fake http_session() that records POSTs and returns a canned response."""

    def __init__(self, payload: dict, status_code: int = 200):
        self._response = _RecordingResponse(payload, status_code)
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append((url, {"headers": headers, "json": json, "timeout": timeout}))
        return self._response


def test_mistral_embedding_posts_rest_payload_and_parses_data(monkeypatch):
    session = _RecordingSession({
        "data": [
            {"index": 1, "embedding": [0.4, 0.5]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ],
    })
    monkeypatch.setattr("keep.providers.http.http_session", lambda: session)

    provider = MistralEmbedding(model="mistral-embed", api_key="test-key")
    out = provider.embed_batch(["a", "b"])

    # Order must follow input order (sorted by `index`), not response order.
    assert out == [[0.1, 0.2], [0.4, 0.5]]
    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url == "https://api.mistral.ai/v1/embeddings"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"] == {"model": "mistral-embed", "input": ["a", "b"]}


def test_mistral_embedding_reports_auth_error(monkeypatch):
    session = _RecordingSession({}, status_code=401)
    monkeypatch.setattr("keep.providers.http.http_session", lambda: session)

    provider = MistralEmbedding(model="mistral-embed", api_key="bad-key")
    with pytest.raises(RuntimeError, match="Mistral API authentication failed"):
        provider.embed("hello")


def test_mistral_summarization_extracts_choice_content(monkeypatch):
    session = _RecordingSession({
        "choices": [{"message": {"content": "short summary"}}],
    })
    monkeypatch.setattr("keep.providers.http.http_session", lambda: session)

    provider = MistralSummarization(model="mistral-small-latest", api_key="test-key")
    assert provider.generate("system", "user") == "short summary"

    url, kwargs = session.calls[0]
    assert url == "https://api.mistral.ai/v1/chat/completions"
    body = kwargs["json"]
    assert body["model"] == "mistral-small-latest"
    assert body["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert "max_tokens" in body


def test_mistral_summarization_returns_none_when_no_choices(monkeypatch):
    session = _RecordingSession({"choices": []})
    monkeypatch.setattr("keep.providers.http.http_session", lambda: session)
    provider = MistralSummarization(model="mistral-small-latest", api_key="test-key")
    assert provider.generate("s", "u") is None


def test_mistral_ocr_image_uses_image_url_chunk(monkeypatch, tmp_path):
    img = tmp_path / "scan.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    session = _RecordingSession({
        "pages": [
            {"index": 0, "markdown": "# Heading\n\nbody text"},
            {"index": 1, "markdown": "page two"},
        ],
    })
    monkeypatch.setattr("keep.providers.http.http_session", lambda: session)

    provider = MistralContentExtractor(model="mistral-ocr-latest", api_key="test-key")
    text = provider.extract(str(img), "image/png")

    assert text == "# Heading\n\nbody text\n\npage two"
    url, kwargs = session.calls[0]
    assert url == "https://api.mistral.ai/v1/ocr"
    doc = kwargs["json"]["document"]
    assert doc["type"] == "image_url"
    assert doc["image_url"].startswith("data:image/png;base64,")


def test_mistral_ocr_pdf_uses_document_url_chunk(monkeypatch, tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    session = _RecordingSession({"pages": [{"index": 0, "markdown": "content"}]})
    monkeypatch.setattr("keep.providers.http.http_session", lambda: session)

    provider = MistralContentExtractor(model="mistral-ocr-latest", api_key="test-key")
    # Short text returns None (matches existing 10-char guard)
    assert provider.extract(str(pdf), "application/pdf") is None

    url, kwargs = session.calls[0]
    doc = kwargs["json"]["document"]
    assert doc["type"] == "document_url"
    assert doc["document_url"].startswith("data:application/pdf;base64,")


def test_mistral_ocr_skips_unsupported_content_type(monkeypatch, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("plain text")

    called = {"count": 0}

    def fake_session():
        called["count"] += 1
        raise AssertionError("OCR should not be invoked for unsupported types")

    monkeypatch.setattr("keep.providers.http.http_session", fake_session)
    provider = MistralContentExtractor(model="mistral-ocr-latest", api_key="test-key")
    assert provider.extract(str(f), "text/plain") is None
    assert called["count"] == 0


def test_mistral_embedding_network_error_is_actionable(monkeypatch):
    class FailingSession:
        def post(self, *args, **kwargs):
            request = httpx.Request("POST", "https://api.mistral.ai/v1/embeddings")
            raise httpx.ConnectError("network unreachable", request=request)

    monkeypatch.setattr("keep.providers.http.http_session", lambda: FailingSession())
    provider = MistralEmbedding(model="mistral-embed", api_key="test-key")
    with pytest.raises(RuntimeError, match="Cannot reach Mistral API"):
        provider.embed("hello")
