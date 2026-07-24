"""Provider lifecycle management mixin.

Lazy initialization with double-checked locking for all ML/AI providers
(embedding, summarization, media, content extraction, analysis).
Includes release methods for GPU memory management.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from .provider_identity import provider_model_name
from .providers import get_registry
from .providers.embedding_cache import CachingEmbeddingProvider

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .providers.base import (
        EmbeddingProvider,
        MediaDescriber,
        SummarizationProvider,
    )


class ProviderLifecycleMixin:
    """Lazy-init and release for all ML/AI providers.

    Requires the composing class to provide:
    - _config: StoreConfig
    - _provider_init_lock: threading.RLock
    - _embedding_provider, _summarization_provider, _media_describer,
      _content_extractor, _analyzer: provider slots (initially None)
    - _is_local: bool
    - _store_path: Path
    - _store: VectorStoreProtocol
    - _document_store: DocumentStoreProtocol
    - _validate_embedding_identity(provider): method
    - _notify(message, *, flush=False): method — operator notice, local-only
    """

    def _get_embedding_provider(self) -> "EmbeddingProvider":
        """Get embedding provider, creating it lazily on first use.

        Thread-safe: uses a lock to prevent concurrent model loading
        when the reconcile thread and main thread both need embeddings.

        This allows read-only operations to work without loading
        the embedding model upfront.
        """
        if self._embedding_provider is not None:
            return self._embedding_provider

        with self._provider_init_lock:
            # Double-check after acquiring lock (another thread may have created it)
            if self._embedding_provider is not None:
                return self._embedding_provider

            if self._config.embedding is None:
                raise RuntimeError(
                    "No embedding provider configured.\n"
                    "\n"
                    "To use keep, configure a provider:\n"
                    "  API-based:  export VOYAGE_API_KEY=...  (or OPENAI_API_KEY, GEMINI_API_KEY)\n"
                    "  Local:      pip install 'keep-skill[local]'\n"
                    "\n"
                    "Read-only operations (get, list, find) work without embeddings.\n"
                    "Find uses full-text search when no embedding provider is configured."
                )
            registry = get_registry()
            base_provider = registry.create_embedding(
                self._config.embedding.name,
                self._config.embedding.params,
            )
            # Wrap local GPU providers with lifecycle lock
            # Local-only: model locks and embedding cache use filesystem
            if self._is_local:
                if self._config.embedding.name == "mlx":
                    from .model_lock import LockedEmbeddingProvider
                    base_provider = LockedEmbeddingProvider(
                        base_provider,
                        self._store_path / ".embedding.lock",
                    )
                cache_path = self._store_path / "embedding_cache.db"
                self._embedding_provider = CachingEmbeddingProvider(
                    base_provider,
                    cache_path=cache_path,
                    provider_name=self._config.embedding.name,
                )
            else:
                self._embedding_provider = base_provider
            # Validate or record embedding identity
            self._validate_embedding_identity(self._embedding_provider)
            # Update store's embedding dimension if it wasn't known at init
            if self._store.embedding_dimension is None:
                self._store.reset_embedding_dimension(self._embedding_provider.dimension)
        return self._embedding_provider

    def _try_dedup_embedding(
        self,
        doc_coll: str,
        chroma_coll: str,
        content_hash: Optional[str],
        exclude_id: str,
        content: str = "",
    ) -> Optional[list[float]]:
        """Look up an existing embedding from a donor doc with the same content hash.

        Returns the embedding if found and dimension-validated, None otherwise.
        Passes the full SHA256 for collision-safe verification.
        """
        from .processors import _content_hash_full

        if not content_hash:
            return None
        full_hash = _content_hash_full(content) if content else ""
        donor = self._document_store.find_by_content_hash(
            doc_coll, content_hash,
            content_hash_full=full_hash,
            exclude_id=exclude_id,
        )
        if donor is None:
            return None
        donor_embedding = self._store.get_embedding(chroma_coll, donor.id)
        if donor_embedding is None:
            return None
        if len(donor_embedding) != self._get_embedding_provider().dimension:
            return None
        logger.debug("Dedup: reusing embedding from %s for %s", donor.id, exclude_id)
        return donor_embedding

    def _get_summarization_provider(self) -> "SummarizationProvider":
        """Get summarization provider, creating it lazily on first use.

        Thread-safe: uses _provider_init_lock with double-checked locking.
        """
        if self._summarization_provider is not None:
            return self._summarization_provider

        with self._provider_init_lock:
            if self._summarization_provider is not None:
                return self._summarization_provider
            registry = get_registry()
            provider = registry.create_summarization(
                self._config.summarization.name,
                self._config.summarization.params,
            )
            if self._is_local and self._config.summarization.name == "mlx":
                from .model_lock import LockedSummarizationProvider
                provider = LockedSummarizationProvider(
                    provider,
                    self._store_path / ".summarization.lock",
                )
            self._summarization_provider = provider
        return self._summarization_provider

    def _release_summarization_provider(self) -> None:
        """Release summarization model to free GPU/unified memory.

        Always clears the provider reference so the lazy getter will
        reconstruct it on next use. For GPU-resident providers (MLX),
        also calls release() to free model weights immediately.

        Safe to call at any time.
        """
        with self._provider_init_lock:
            provider = self._summarization_provider
            self._summarization_provider = None

        if provider is not None:
            if hasattr(provider, 'release'):
                provider.release()

    def _release_embedding_provider(self) -> None:
        """Release embedding model to free GPU/unified memory.

        Always clears the provider reference so the lazy getter will
        reconstruct it on next use. For GPU-resident providers (MLX),
        also calls release() to free model weights immediately.

        Also closes the embedding cache when releasing.
        Safe to call at any time.
        """
        with self._provider_init_lock:
            provider = self._embedding_provider
            self._embedding_provider = None

        if provider is not None:
            # Release the locked inner provider (frees model weights)
            inner = getattr(provider, '_provider', None)
            if hasattr(inner, 'release'):
                inner.release()
            # Close the embedding cache
            if hasattr(provider, 'close'):
                provider.close()
            elif hasattr(provider, '_cache'):
                cache = provider._cache
                if hasattr(cache, 'close'):
                    cache.close()

    def _get_media_describer(self) -> "Optional[MediaDescriber]":
        """Get media describer, creating it lazily on first use.

        Thread-safe: uses _provider_init_lock with double-checked locking.
        Returns None if no media provider is configured or creation fails.
        """
        if self._media_describer is not None:
            return self._media_describer
        if self._config.media is None:
            return None

        with self._provider_init_lock:
            if self._media_describer is not None:
                return self._media_describer
            registry = get_registry()
            try:
                provider = registry.create_media(
                    self._config.media.name,
                    self._config.media.params,
                )
            except (ValueError, RuntimeError) as e:
                logger.warning("Media describer unavailable: %s", e)
                return None
            if self._is_local and self._config.media.name == "mlx":
                from .model_lock import LockedMediaDescriber
                provider = LockedMediaDescriber(
                    provider,
                    self._store_path / ".media.lock",
                )
            self._media_describer = provider
        return self._media_describer

    def _get_content_extractor(self):
        """Get content extractor, creating it lazily on first use.

        Thread-safe: uses _provider_init_lock with double-checked locking.
        Used by the background OCR processor. Returns None if no content
        extractor is configured or creation fails.
        """
        if self._content_extractor is not None:
            return self._content_extractor
        if self._config.content_extractor is None:
            return None

        with self._provider_init_lock:
            if self._content_extractor is not None:
                return self._content_extractor
            registry = get_registry()
            try:
                provider = registry.create_content_extractor(
                    self._config.content_extractor.name,
                    self._config.content_extractor.params,
                )
            except (ValueError, RuntimeError) as e:
                logger.warning("Content extractor unavailable: %s", e)
                return None
            if self._is_local and self._config.content_extractor.name == "mlx":
                from .model_lock import LockedContentExtractor
                provider = LockedContentExtractor(
                    provider,
                    self._store_path / ".extractor.lock",
                )
            self._content_extractor = provider
        return self._content_extractor

    def _release_content_extractor(self) -> None:
        """Release content extractor to free GPU/unified memory."""
        with self._provider_init_lock:
            provider = self._content_extractor
            self._content_extractor = None

        if provider is not None:
            if hasattr(provider, 'release'):
                provider.release()

    def _get_analyzer(self):
        """Get analyzer provider, creating it lazily on first use.

        Thread-safe: uses _provider_init_lock with double-checked locking.
        """
        if self._analyzer is not None:
            return self._analyzer

        with self._provider_init_lock:
            if self._analyzer is not None:
                return self._analyzer
            if self._config.analyzer:
                from .providers import get_registry

                registry = get_registry()
                self._analyzer = registry.create_analyzer(
                    self._config.analyzer.name,
                    self._config.analyzer.params,
                )
            else:
                # Default: sliding-window analyzer with the summarization provider,
                # budget auto-selected based on the model's effective context quality.
                from .analyzers import SlidingWindowAnalyzer, get_budget_for_model
                provider = self._get_summarization_provider()
                model = provider_model_name(provider, "")
                provider_name = self._config.summarization.name if self._config.summarization else ""
                budget = get_budget_for_model(model, provider_name)
                self._analyzer = SlidingWindowAnalyzer(
                    provider=provider,
                    context_budget=budget,
                )
        return self._analyzer
