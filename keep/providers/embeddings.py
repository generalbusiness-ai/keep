"""Embedding providers for generating vector representations of text."""

import logging
import os
import sys
import time

import httpx

from .base import EmbedTask, get_registry, require_provider_param
from .openai_client import create_openai_client


class SentenceTransformerEmbedding:
    """Embedding provider using sentence-transformers library.

    Runs locally, no API key required. Good default for getting started.

    Requires: pip install sentence-transformers
    """

    def __init__(self, model: str | None = None, trust_remote_code: bool = False):
        """Initialize.

        Args:
        model: Model name from sentence-transformers hub
        trust_remote_code: Allow models with custom code (e.g. nomic-embed-text-v1.5).
            Disabled by default for security — only enable for models you trust.
        """
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "SentenceTransformerEmbedding requires 'sentence-transformers' library. "
                "Install with: pip install sentence-transformers"
            )

        model = require_provider_param(model, provider="SentenceTransformerEmbedding")
        self.model_name = model

        # Check if model is already cached locally to avoid network calls
        # Expand short model names (e.g. "all-MiniLM-L6-v2" -> "sentence-transformers/all-MiniLM-L6-v2")
        local_only = False
        try:
            from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415
            repo_id = model if "/" in model else f"sentence-transformers/{model}"
            cached = try_to_load_from_cache(repo_id, "config.json")
            local_only = cached is not None
        except ImportError:
            pass

        if not local_only:
            logging.getLogger(__name__).info("Downloading embedding model '%s' (first use)...", model)
            print(f"Downloading embedding model '{model}' (first use)...", file=sys.stderr)

        self._model = SentenceTransformer(
            model, local_files_only=local_only,
            trust_remote_code=trust_remote_code,
        )
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension from the model."""
        return self._model.get_sentence_embedding_dimension()
    
    def _prompt_name(self, task: EmbedTask) -> str | None:
        """Map task to prompt_name for models that support it (e.g. nomic)."""
        prompts = getattr(self._model, "prompts", None) or {}
        if not prompts:
            return None
        name = "search_query" if task == EmbedTask.QUERY else "search_document"
        return name if name in prompts else None

    def embed(self, text: str, *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[float]:
        """Generate embedding for a single text."""
        kwargs: dict = {"convert_to_numpy": True}
        prompt_name = self._prompt_name(task)
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        embedding = self._model.encode(text, **kwargs)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        kwargs: dict = {"convert_to_numpy": True}
        prompt_name = self._prompt_name(task)
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name
        embeddings = self._model.encode(texts, **kwargs)
        return embeddings.tolist()


class OpenAIEmbedding:
    """Embedding provider using OpenAI's API or any OpenAI-compatible endpoint.

    Works with OpenAI, llama-server, vLLM, LM Studio, LocalAI, or any service
    that implements the ``/v1/embeddings`` endpoint.

    Requires: pip install openai
    """

    # Model dimensions (as of 2024)
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize.

        Args:
        model: Embedding model name.
        api_key: API key (defaults to environment variable).  Not required
            when ``base_url`` points at a local server.
        base_url: Override the API base URL (e.g.
            ``http://localhost:8801/v1`` for llama-server).  When set, the
            provider connects to this endpoint instead of api.openai.com.
        """
        # Include base_url in model_name so embedding identity and cache keys
        # distinguish between different servers using the same model string.
        model = require_provider_param(model, provider="OpenAIEmbedding")
        self.model_name = f"{model}@{base_url}" if base_url else model
        self._api_model = model  # raw model name for API calls
        self._dimension = self.MODEL_DIMENSIONS.get(model)
        self._client = create_openai_client(api_key=api_key, base_url=base_url)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension for the model (detected lazily if unknown)."""
        if self._dimension is None:
            # Unknown model: detect from first embedding
            test_embedding = self.embed("dimension test")
            self._dimension = len(test_embedding)
        return self._dimension

    def embed(self, text: str, *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[float]:
        """Generate embedding for a single text."""
        response = self._client.embeddings.create(
            model=self._api_model,
            input=text,
        )
        embedding = response.data[0].embedding
        # Cache dimension if not yet known
        if self._dimension is None:
            self._dimension = len(embedding)
        return embedding

    def embed_batch(self, texts: list[str], *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        response = self._client.embeddings.create(
            model=self._api_model,
            input=texts,
        )
        # Sort by index to ensure order matches input
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]


class GeminiEmbedding:
    """Embedding provider using Google's Gemini API.

    Authentication (checked in priority order):
    1. api_key parameter (if provided, uses Google AI Studio)
    2. GOOGLE_CLOUD_PROJECT env var (uses Vertex AI with ADC)
    3. GEMINI_API_KEY or GOOGLE_API_KEY (uses Google AI Studio)
    """

    # Default output dimensions per model (full native dimension).
    # These are used only when no output_dimensionality is requested.
    MODEL_DIMENSIONS = {
        "text-embedding-004": 768,
        "embedding-001": 768,
        "gemini-embedding-001": 3072,
    }

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        output_dimensionality: int | None = None,
    ):
        """Initialize.

        Args:
        model: Gemini embedding model name
        api_key: API key (defaults to environment variable)
        output_dimensionality: Optional reduced dimension (e.g. 768 for
            gemini-embedding-001 which defaults to 3072). When set, the
            API returns truncated vectors via Matryoshka representation.
        """
        from google.genai import types as genai_types  # noqa: PLC0415
        from .gemini_client import create_gemini_client  # noqa: PLC0415

        model = require_provider_param(model, provider="GeminiEmbedding")
        self.model_name = model
        self._client = create_gemini_client(api_key)
        self._genai_types = genai_types

        # Build embed config if dimensionality is requested
        self._embed_config: genai_types.EmbedContentConfig | None = None
        if output_dimensionality is not None:
            self._embed_config = self._genai_types.EmbedContentConfig(
                output_dimensionality=output_dimensionality,
            )
            self._dimension: int | None = output_dimensionality
        else:
            self._dimension = self.MODEL_DIMENSIONS.get(model)

    @property
    def dimension(self) -> int:
        """Get embedding dimension for the model (detected lazily if unknown)."""
        if self._dimension is None:
            # Unknown model: detect from first embedding
            test_embedding = self.embed("dimension test")
            self._dimension = len(test_embedding)
        return self._dimension

    _TASK_TYPES = {
        EmbedTask.DOCUMENT: "RETRIEVAL_DOCUMENT",
        EmbedTask.QUERY: "RETRIEVAL_QUERY",
    }

    def _embed_config_for(self, task: EmbedTask):
        """Build EmbedContentConfig with task_type merged in."""
        task_type = self._TASK_TYPES.get(task)
        if self._embed_config is not None:
            return self._genai_types.EmbedContentConfig(
                output_dimensionality=self._embed_config.output_dimensionality,
                task_type=task_type,
            )
        return self._genai_types.EmbedContentConfig(task_type=task_type)

    def embed(self, text: str, *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[float]:
        """Generate embedding for a single text."""
        kwargs: dict = dict(
            model=self.model_name, contents=text,
            config=self._embed_config_for(task),
        )
        result = self._client.models.embed_content(**kwargs)
        embedding = list(result.embeddings[0].values)
        # Cache dimension if not yet known
        if self._dimension is None:
            self._dimension = len(embedding)
        return embedding

    def embed_batch(self, texts: list[str], *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        kwargs: dict = dict(
            model=self.model_name, contents=texts,
            config=self._embed_config_for(task),
        )
        result = self._client.models.embed_content(**kwargs)
        return [list(e.values) for e in result.embeddings]


class OllamaEmbedding:
    """Embedding provider using Ollama's local API.

    Requires: Ollama running locally.
    Respects OLLAMA_HOST env var (default: http://localhost:11434).
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize.

        Args:
        model: Ollama model name
        base_url: Ollama API base URL (default: OLLAMA_HOST or http://localhost:11434).
        """
        from .ollama_utils import ollama_base_url, ollama_ensure_model  # noqa: PLC0415

        model = require_provider_param(model, provider="OllamaEmbedding")
        self.model_name = model
        self.base_url = ollama_base_url(base_url)
        self._dimension: int | None = None
        ollama_ensure_model(self.base_url, self.model_name)

    @property
    def dimension(self) -> int:
        """Get embedding dimension (determined on first embed call)."""
        if self._dimension is None:
            # Generate a test embedding to determine dimension
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension

    @staticmethod
    def _truncate_at_word(text: str, max_chars: int) -> str:
        """Truncate text to max_chars, breaking at a word boundary."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > max_chars // 2:
            return truncated[:last_space]
        return truncated

    # Prefix-based task instructions for nomic-embed-text models.
    # Other Ollama models don't use prefixes and get the text as-is.
    _NOMIC_PREFIXES = {
        EmbedTask.DOCUMENT: "search_document: ",
        EmbedTask.QUERY: "search_query: ",
    }

    def _uses_task_prefix(self) -> bool:
        """Whether this model uses nomic-style task prefixes."""
        return "nomic-embed" in self.model_name

    def embed(self, text: str, *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[float]:
        """Generate embedding, auto-truncating if the model rejects the input.

        Tries the full text first. If the model returns a context-length
        error (instant 500), trims ~10% and retries. Only the final
        successful call does real compute; rejections are near-free.
        """
        from .ollama_utils import ollama_session  # noqa: PLC0415

        prompt = text
        if self._uses_task_prefix():
            prompt = self._NOMIC_PREFIXES[task] + text

        attempt = prompt
        for _ in range(30):  # 0.9^30 ≈ 4% — covers even extreme cases
            response = ollama_session().post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": attempt, "keep_alive": "30m"},
                timeout=(10, 120),  # (connect, read) — model loading can be slow
            )
            if response.is_success:
                embedding = response.json()["embedding"]
                if self._dimension is None:
                    self._dimension = len(embedding)
                return embedding

            # Context-length error: trim 10% and retry (rejection is instant)
            if response.status_code == 500 and "context length" in response.text:
                new_len = int(len(attempt) * 0.9)
                if new_len < 50:
                    break
                attempt = self._truncate_at_word(prompt, new_len)
                continue

            # Non-retryable error — break immediately
            break

        detail = response.text[:200] if response.text else ""
        raise RuntimeError(
            f"Ollama embedding failed (model={self.model_name}): "
            f"HTTP {response.status_code} from {self.base_url}. {detail}"
        )

    def embed_batch(self, texts: list[str], *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[list[float]]:
        """Generate embeddings for multiple texts (sequential for Ollama)."""
        return [self.embed(text, task=task) for text in texts]


class VoyageEmbedding:
    """Embedding provider using Voyage AI's REST API.

    Voyage AI is Anthropic's recommended embedding partner.
    Works well in Claude Desktop and other Anthropic-integrated environments.

    Uses direct HTTP calls - no voyageai SDK needed (avoids heavy dependencies).
    Includes automatic retry with exponential backoff for rate limits.

    Requires: VOYAGE_API_KEY environment variable.
    """

    # Model dimensions (as of 2025)
    # All current models default to 1024 dims
    MODEL_DIMENSIONS = {
        "voyage-3-large": 1024,
        "voyage-3.5": 1024,
        "voyage-3.5-lite": 1024,
        "voyage-code-3": 1024,
    }

    API_URL = "https://api.voyageai.com/v1/embeddings"

    # Retry settings
    MAX_RETRIES = 5
    INITIAL_BACKOFF = 1.0  # seconds
    MAX_BACKOFF = 60.0  # seconds

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize.

        Args:
        model: Voyage embedding model name
        api_key: API key (defaults to environment variable).
        """
        model = require_provider_param(model, provider="VoyageEmbedding")
        self.model_name = model
        # Use lookup table if available, otherwise detect lazily from first embedding
        self._dimension = self.MODEL_DIMENSIONS.get(model)

        # Resolve API key
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Voyage API key required. Set VOYAGE_API_KEY environment variable.\n"
                "Get your API key at: https://dash.voyageai.com/"
            )

    @property
    def dimension(self) -> int:
        """Get embedding dimension for the model (detected lazily if unknown)."""
        if self._dimension is None:
            # Unknown model: detect from first embedding
            test_embedding = self.embed("dimension test")
            self._dimension = len(test_embedding)
        return self._dimension

    def _request_with_retry(self, payload: dict, timeout: int) -> dict:
        """Make API request with exponential backoff retry for rate limits."""
        from .http import http_session  # noqa: PLC0415

        backoff = self.INITIAL_BACKOFF
        last_exception = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = http_session().post(
                    self.API_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                )

                # Handle rate limiting (429)
                if response.status_code == 429:
                    # Check for Retry-After header
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            wait_time = backoff
                    else:
                        wait_time = backoff

                    wait_time = min(wait_time, self.MAX_BACKOFF)
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, self.MAX_BACKOFF)
                    continue

                # Auth errors — fail immediately, don't retry
                if response.status_code in (401, 403):
                    raise RuntimeError(
                        f"Voyage AI API authentication failed ({response.status_code}).\n"
                        "Check your VOYAGE_API_KEY environment variable."
                    )

                response.raise_for_status()
                return response.json()

            except httpx.HTTPError as e:
                last_exception = e
                error_msg = str(e).lower()

                # Network errors - provide helpful message
                if "connection" in error_msg or "network" in error_msg or "resolve" in error_msg:
                    raise RuntimeError(
                        f"Cannot reach Voyage AI API: {e}\n\n"
                        "If running in a sandboxed environment (e.g., Claude Desktop):\n"
                        "Add api.voyageai.com to your network allowlist."
                    ) from e

                # Other request errors - retry with backoff
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF)
                    continue
                raise

        # Exhausted retries
        raise RuntimeError(
            f"Voyage AI API rate limit exceeded after {self.MAX_RETRIES} retries. "
            "Please wait and try again."
        ) from last_exception

    _INPUT_TYPES = {
        EmbedTask.DOCUMENT: "document",
        EmbedTask.QUERY: "query",
    }

    def embed(self, text: str, *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[float]:
        """Generate embedding for a single text."""
        data = self._request_with_retry(
            payload={
                "input": [text],
                "model": self.model_name,
                "input_type": self._INPUT_TYPES[task],
            },
            timeout=60,
        )

        embedding = data["data"][0]["embedding"]

        # Cache dimension if not yet known
        if self._dimension is None:
            self._dimension = len(embedding)
        return embedding

    def embed_batch(self, texts: list[str], *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        data = self._request_with_retry(
            payload={
                "input": texts,
                "model": self.model_name,
                "input_type": self._INPUT_TYPES[task],
            },
            timeout=120,
        )

        # Sort by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [d["embedding"] for d in sorted_data]


class MistralEmbedding:
    """Embedding provider using Mistral AI's REST API.

    Uses direct HTTP calls — no mistralai SDK needed.

    Requires: MISTRAL_API_KEY environment variable.
    """

    MODEL_DIMENSIONS = {
        "mistral-embed": 1024,
    }

    API_URL = "https://api.mistral.ai/v1/embeddings"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        model = require_provider_param(model, provider="MistralEmbedding")
        self.model_name = model
        self._dimension = self.MODEL_DIMENSIONS.get(model)

        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable.\n"
                "Get your API key at: https://console.mistral.ai/"
            )

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            test_embedding = self.embed("dimension test")
            self._dimension = len(test_embedding)
        return self._dimension

    def _post_embeddings(self, inputs: list[str], timeout: int) -> dict:
        from .http import http_session  # noqa: PLC0415

        try:
            response = http_session().post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model_name, "input": inputs},
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

    def embed(self, text: str, *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[float]:
        data = self._post_embeddings([text], timeout=60)
        embedding = data["data"][0]["embedding"]
        if self._dimension is None:
            self._dimension = len(embedding)
        return embedding

    def embed_batch(self, texts: list[str], *, task: EmbedTask = EmbedTask.DOCUMENT) -> list[list[float]]:
        if not texts:
            return []
        data = self._post_embeddings(texts, timeout=120)
        # Sort by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in sorted_data]


# Register providers
_registry = get_registry()
_registry.register_embedding("sentence-transformers", SentenceTransformerEmbedding)
_registry.register_embedding("openai", OpenAIEmbedding)
_registry.register_embedding("gemini", GeminiEmbedding)
_registry.register_embedding("ollama", OllamaEmbedding)
_registry.register_embedding("voyage", VoyageEmbedding)
_registry.register_embedding("mistral", MistralEmbedding)
