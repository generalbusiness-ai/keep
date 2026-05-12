"""Summarization providers using LLMs."""

import base64
import json
import os

from .base import (
    get_registry,
    build_summarization_prompt,
    require_provider_param,
    strip_summary_preamble,
    SUMMARIZATION_SYSTEM_PROMPT,
)
from .openai_client import create_openai_client


# -----------------------------------------------------------------------------
# Summarization Providers
# -----------------------------------------------------------------------------

class AnthropicSummarization:
    """Summarization provider using Anthropic's Claude API.

    Authentication (checked in priority order):
    1. api_key parameter (if provided)
    2. ANTHROPIC_API_KEY (recommended: API key from console.anthropic.com)
    3. CLAUDE_CODE_OAUTH_TOKEN (OAuth token from 'claude setup-token')

    Note: OAuth tokens (sk-ant-oat01-...) are primarily for Claude Code CLI.
    For production use, prefer API keys (sk-ant-api03-...) from console.anthropic.com.

    Default model is claude-haiku-4.5 ($1.00/$5.00 per MTok).
    Configure via keep.toml [summarization] section for other models:
    - claude-haiku-4-5-20251001: Default, best quality/cost for summaries
    - claude-3-5-haiku-20241022: Previous generation
    - claude-3-haiku-20240307: Legacy, cheapest ($0.25/$1.25 per MTok)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 150,
    ):
        from anthropic import Anthropic  # noqa: PLC0415

        model = require_provider_param(model, provider="AnthropicSummarization")
        self.model_name = model
        self.max_tokens = max_tokens

        # Try multiple auth sources in priority order:
        # 1. Explicit api_key parameter
        # 2. ANTHROPIC_API_KEY (API key from console.anthropic.com: sk-ant-api03-...)
        # 3. CLAUDE_CODE_OAUTH_TOKEN (OAuth token from 'claude setup-token': sk-ant-oat01-...)
        key = (
            api_key or
            os.environ.get("ANTHROPIC_API_KEY") or
            os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        )
        if not key:
            raise ValueError(
                "Anthropic authentication required. Set one of:\n"
                "  ANTHROPIC_API_KEY (API key from console.anthropic.com)\n"
                "  CLAUDE_CODE_OAUTH_TOKEN (OAuth token from 'claude setup-token')"
            )

        self._client = Anthropic(api_key=key)
    
    def summarize(
        self,
        content: str,
        *,
        max_length: int = 500,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate summary using Anthropic Claude."""
        truncated = content[:50000] if len(content) > 50000 else content
        prompt = build_summarization_prompt(truncated, context)
        result = self.generate(system_prompt or SUMMARIZATION_SYSTEM_PROMPT, prompt)
        if not result:
            return truncated[:max_length]
        return strip_summary_preamble(result)

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
    ) -> str | None:
        """Send a raw prompt to Anthropic and return generated text."""
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if response.content:
            return response.content[0].text
        return None


class OpenAISummarization:
    """Summarization provider using OpenAI's chat API or any compatible endpoint.

    Works with OpenAI, llama-server, vLLM, LM Studio, LocalAI, or any service
    that implements the ``/v1/chat/completions`` endpoint.

    Default model is gpt-4.1-mini ($0.40/$1.60 per MTok).
    Good alternatives: gpt-4.1, gpt-5-mini.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 200,
    ):
        model = require_provider_param(model, provider="OpenAISummarization")
        self.model_name = model
        self.max_tokens = max_tokens
        self._client = create_openai_client(api_key=api_key, base_url=base_url)

        # GPT-5+ and reasoning models use a different API surface:
        # - max_completion_tokens instead of max_tokens
        # - temperature must be omitted (only default=1 supported)
        # Only applies to actual OpenAI models, not local compatible servers.
        self._new_api = (
            not base_url and self.model_name.startswith(("gpt-5", "o3", "o4"))
        )

    def _completion_kwargs(self, max_tokens: int) -> dict:
        """Return model-appropriate kwargs for token limit and temperature."""
        if self._new_api:
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens, "temperature": 0.3}

    def summarize(
        self,
        content: str,
        *,
        max_length: int = 500,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a summary using OpenAI."""
        truncated = content[:50000] if len(content) > 50000 else content
        prompt = build_summarization_prompt(truncated, context)
        result = self.generate(system_prompt or SUMMARIZATION_SYSTEM_PROMPT, prompt)
        if not result:
            return truncated[:max_length]
        return strip_summary_preamble(result)

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
    ) -> str | None:
        """Send a raw prompt to OpenAI and return generated text."""
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self._completion_kwargs(max_tokens),
        )
        if response.choices:
            return response.choices[0].message.content
        return None


class OllamaSummarization:
    """Summarization provider using Ollama's local API.

    Respects OLLAMA_HOST env var (default: http://localhost:11434).

    ``context_length`` sets ``num_ctx`` in the Ollama API call.  Many
    small models default to 4096 which silently truncates longer prompts.
    Configurable via ``keep.toml``::

        [summarization]
        context_length = 16384
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        context_length: int = 16384,
    ):
        from .ollama_utils import ollama_base_url, ollama_ensure_model  # noqa: PLC0415

        model = require_provider_param(model, provider="OllamaSummarization")
        self.model_name = model
        self.context_length = context_length
        self.base_url = ollama_base_url(base_url)
        ollama_ensure_model(self.base_url, self.model_name)

    def summarize(
        self,
        content: str,
        *,
        max_length: int = 500,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a summary using Ollama."""
        # Small local models degrade with long inputs; truncate aggressively.
        limit = 15000
        truncated = content[:limit] if len(content) > limit else content
        prompt = build_summarization_prompt(truncated, context)
        result = self.generate(system_prompt or SUMMARIZATION_SYSTEM_PROMPT, prompt)
        if not result:
            return truncated[:max_length]
        return strip_summary_preamble(result)

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
    ) -> str | None:
        """Send a raw prompt to Ollama and return generated text."""
        from .ollama_utils import ollama_session  # noqa: PLC0415
        response = ollama_session().post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"num_ctx": self.context_length},
                "keep_alive": "30m",
            },
            timeout=(10, 120),  # (connect, read)
        )
        if not response.is_success:
            detail = response.text[:200] if response.text else ""
            raise RuntimeError(
                f"Ollama generate failed (model={self.model_name}): "
                f"HTTP {response.status_code} from {self.base_url}. {detail}"
            )
        return response.json()["message"]["content"].strip()


class GeminiSummarization:
    """Summarization provider using Google's Gemini API.

    Authentication (checked in priority order):
    1. api_key parameter (if provided, uses Google AI Studio)
    2. GOOGLE_CLOUD_PROJECT env var (uses Vertex AI with ADC)
    3. GEMINI_API_KEY or GOOGLE_API_KEY (uses Google AI Studio)

    Default model is gemini-2.5-flash (cost-effective: ~$0.075/$0.30 per MTok).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 150,
    ):
        model = require_provider_param(model, provider="GeminiSummarization")
        self.model_name = model
        from .gemini_client import create_gemini_client  # noqa: PLC0415

        self.max_tokens = max_tokens
        self._client = create_gemini_client(api_key)

    def summarize(
        self,
        content: str,
        *,
        max_length: int = 500,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Generate summary using Google Gemini."""
        truncated = content[:50000] if len(content) > 50000 else content
        prompt = build_summarization_prompt(truncated, context)
        result = self.generate(system_prompt or SUMMARIZATION_SYSTEM_PROMPT, prompt)
        if not result:
            return truncated[:max_length]
        return strip_summary_preamble(result)

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
    ) -> str | None:
        """Send a raw prompt to Gemini and return generated text."""
        full_prompt = f"{system}\n\n{user}"
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
        )
        return response.text


class PassthroughSummarization:
    """Summarization provider that returns the first N characters.

    Useful for testing or when LLM summarization is not needed.
    """

    def __init__(self, max_chars: int = 500):
        self.max_chars = max_chars

    def summarize(
        self,
        content: str,
        *,
        max_length: int = 500,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Return truncated content as summary (ignores context/system_prompt)."""
        limit = min(self.max_chars, max_length)
        if len(content) <= limit:
            return content
        return content[:limit].rsplit(" ", 1)[0] + "..."

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
    ) -> str | None:
        """Passthrough has no LLM — return None."""
        return None


# -----------------------------------------------------------------------------
# Media Description Providers
# -----------------------------------------------------------------------------

class OllamaMediaDescriber:
    """Media description using Ollama's vision models.

    Supports image description via multimodal models (llava, moondream, etc.).
    Audio transcription is not supported via Ollama.

    Respects OLLAMA_HOST env var (default: http://localhost:11434).
    """

    IMAGE_PROMPT = (
        "Describe this image in detail. Include the subject, setting, "
        "colors, composition, and any text visible in the image. "
        "Be specific and factual."
    )

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        context_length: int = 16384,
    ):
        from .ollama_utils import ollama_base_url, ollama_ensure_model  # noqa: PLC0415

        model = require_provider_param(model, provider="OllamaMediaDescriber")
        self.model_name = model
        self.context_length = context_length
        self.base_url = ollama_base_url(base_url)
        ollama_ensure_model(self.base_url, self.model_name)

    def describe(self, path: str, content_type: str) -> str | None:
        """Describe an image using Ollama vision model."""
        if not content_type.startswith("image/"):
            return None

        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        from .ollama_utils import ollama_session  # noqa: PLC0415
        response = ollama_session().post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": self.IMAGE_PROMPT,
                        "images": [image_data],
                    },
                ],
                "stream": False,
                "options": {"num_ctx": self.context_length},
                "keep_alive": "30m",
            },
            timeout=120,
        )
        if not response.is_success:
            detail = response.text[:200] if response.text else ""
            raise RuntimeError(
                f"Ollama vision failed (model={self.model_name}): "
                f"HTTP {response.status_code} from {self.base_url}. {detail}"
            )

        text = response.json()["message"]["content"].strip()
        return text if text else None


class OllamaContentExtractor:
    """OCR content extraction using Ollama.

    Uses GLM-OCR via Ollama's /api/generate endpoint (recommended for
    vision tasks with this model). Extracts actual text from document images.

    Respects OLLAMA_HOST env var (default: http://localhost:11434).
    """

    OCR_PROMPT = "Extract all text from this image exactly as written."

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        context_length: int = 16384,
    ):
        from .ollama_utils import ollama_base_url, ollama_ensure_model  # noqa: PLC0415

        model = require_provider_param(model, provider="OllamaContentExtractor")
        self.model_name = model
        self.context_length = context_length
        self.base_url = ollama_base_url(base_url)
        ollama_ensure_model(self.base_url, self.model_name)

    def extract(self, path: str, content_type: str) -> str | None:
        """Extract text from an image using Ollama OCR model."""
        if not content_type.startswith("image/"):
            return None

        with open(path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        from .ollama_utils import ollama_session  # noqa: PLC0415
        response = ollama_session().post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": self.OCR_PROMPT,
                "images": [image_data],
                "stream": False,
                "options": {"num_ctx": self.context_length},
                "keep_alive": "30m",
            },
            timeout=120,
        )
        if not response.is_success:
            detail = response.text[:200] if response.text else ""
            raise RuntimeError(
                f"Ollama OCR failed (model={self.model_name}): "
                f"HTTP {response.status_code} from {self.base_url}. {detail}"
            )

        text = response.json().get("response", "").strip()
        return text if len(text) > 10 else None


def _mistral_post(path: str, api_key: str, payload: dict, timeout: int) -> dict:
    """POST to a Mistral API endpoint and return the parsed JSON body.

    Uses the shared httpx session — no mistralai SDK dependency. Auth
    failures and network errors are reported with actionable messages
    that name the env var to set.
    """
    import httpx  # noqa: PLC0415
    from .http import http_session  # noqa: PLC0415

    url = f"https://api.mistral.ai{path}"
    try:
        response = http_session().post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Cannot reach Mistral API: {exc}") from exc

    if response.status_code in (401, 403):
        raise RuntimeError(
            f"Mistral API authentication failed ({response.status_code}). "
            "Check your MISTRAL_API_KEY environment variable."
        )
    response.raise_for_status()
    return response.json()


class MistralSummarization:
    """Summarization provider using Mistral AI's chat API.

    Uses direct HTTP calls — no mistralai SDK needed.

    Requires: MISTRAL_API_KEY environment variable.

    Default model is mistral-small-latest (cost-effective).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 200,
    ):
        model = require_provider_param(model, provider="MistralSummarization")
        self.model_name = model
        self.max_tokens = max_tokens

        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable.\n"
                "Get your API key at: https://console.mistral.ai/"
            )

    def summarize(
        self,
        content: str,
        *,
        max_length: int = 500,
        context: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        truncated = content[:50000] if len(content) > 50000 else content
        prompt = build_summarization_prompt(truncated, context)
        result = self.generate(system_prompt or SUMMARIZATION_SYSTEM_PROMPT, prompt)
        if not result:
            return truncated[:max_length]
        return strip_summary_preamble(result)

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
    ) -> str | None:
        data = _mistral_post(
            "/v1/chat/completions",
            self._api_key,
            payload={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        choices = data.get("choices") or []
        if not choices:
            return None
        return choices[0].get("message", {}).get("content")


class MistralContentExtractor:
    """Content extraction (OCR) using Mistral's OCR API.

    Supports images (PNG, JPEG, etc.) and PDFs.
    Uses mistral-ocr-latest model which returns structured markdown.

    Uses direct HTTP calls — no mistralai SDK needed.

    Requires: MISTRAL_API_KEY environment variable.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        model = require_provider_param(model, provider="MistralContentExtractor")
        self.model_name = model

        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable.\n"
                "Get your API key at: https://console.mistral.ai/"
            )

    def extract(self, path: str, content_type: str) -> str | None:
        if not (content_type.startswith("image/") or content_type == "application/pdf"):
            return None

        with open(path, "rb") as f:
            data_b64 = base64.b64encode(f.read()).decode("utf-8")

        data_url = f"data:{content_type};base64,{data_b64}"

        # OCR chunk shape: images use {"type": "image_url", "image_url": ...};
        # PDFs and other documents use {"type": "document_url", "document_url": ...}.
        if content_type.startswith("image/"):
            document = {"type": "image_url", "image_url": data_url}
        else:
            document = {"type": "document_url", "document_url": data_url}

        result = _mistral_post(
            "/v1/ocr",
            self._api_key,
            payload={"model": self.model_name, "document": document},
            timeout=300,
        )

        # Combine markdown from all pages
        pages = []
        for page in result.get("pages") or []:
            md = (page.get("markdown") or "").strip()
            if md:
                pages.append(md)

        text = "\n\n".join(pages)
        return text if len(text) > 10 else None


# Register providers
_registry = get_registry()
_registry.register_summarization("anthropic", AnthropicSummarization)
_registry.register_summarization("openai", OpenAISummarization)
_registry.register_summarization("ollama", OllamaSummarization)
_registry.register_summarization("gemini", GeminiSummarization)
_registry.register_summarization("mistral", MistralSummarization)
_registry.register_summarization("passthrough", PassthroughSummarization)
_registry.register_media("ollama", OllamaMediaDescriber)
_registry.register_content_extractor("ollama", OllamaContentExtractor)
_registry.register_content_extractor("mistral", MistralContentExtractor)
