"""Tests for deferred embedding in cloud mode.

When _is_local is False, Keeper.put() should skip embedding computation
and enqueue an "embed" task for the background worker instead.
"""

from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from keep.api import Keeper
from keep.pending_summaries import PendingSummary


# ---------------------------------------------------------------------------
# Helpers — tracking pending queue
# ---------------------------------------------------------------------------

class TrackingPendingQueue:
    """Pending queue that records enqueued tasks for assertions."""

    def __init__(self):
        self._items: list[dict] = []

    def enqueue(
        self, id: str, collection: str, content: str,
        *, task_type: str = "summarize", metadata: Optional[dict] = None,
    ) -> None:
        # Replace existing item with same (id, collection, task_type)
        # — matches real PendingSummaryQueue INSERT OR REPLACE behavior
        self._items = [
            i for i in self._items
            if not (i["id"] == id and i["collection"] == collection
                    and i["task_type"] == task_type)
        ]
        self._items.append({
            "id": id,
            "collection": collection,
            "content": content,
            "task_type": task_type,
            "metadata": metadata or {},
        })

    def dequeue(self, limit: int = 10) -> list[PendingSummary]:
        """Return enqueued items as PendingSummary for process_pending."""
        results = []
        for item in self._items[:limit]:
            results.append(PendingSummary(
                id=item["id"],
                collection=item["collection"],
                content=item["content"],
                queued_at="2026-01-01T00:00:00Z",
                task_type=item["task_type"],
                metadata=item["metadata"],
            ))
        return results

    def complete(self, id: str, collection: str, task_type: str = "summarize") -> None:
        self._items = [
            i for i in self._items
            if not (i["id"] == id and i["collection"] == collection
                    and i["task_type"] == task_type)
        ]

    def count(self) -> int:
        return len(self._items)

    def stats(self) -> dict:
        return {"pending": self.count()}

    def clear(self) -> int:
        n = len(self._items)
        self._items.clear()
        return n

    def fail(
        self, id: str, collection: str, task_type: str = "summarize",
        error: str | None = None,
    ) -> None:
        pass  # no-op in test mock

    def abandon(
        self, id: str, collection: str, task_type: str = "summarize",
        error: str | None = None,
    ) -> None:
        self._items = [
            i for i in self._items
            if not (i["id"] == id and i["collection"] == collection
                    and i["task_type"] == task_type)
        ]

    def get_status(self, id: str) -> dict | None:
        return None

    def list_failed(self) -> list[dict]:
        return []

    def retry_failed(self) -> int:
        return 0

    def close(self) -> None:
        self._items.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeferredEmbedding:
    """Cloud mode defers embedding to background worker."""

    def _make_keeper(self, mock_providers, tmp_path) -> tuple[Keeper, TrackingPendingQueue]:
        """Create a Keeper with injected stores (cloud mode: _is_local=False)."""
        from tests.conftest import MockChromaStore, MockDocumentStore

        doc_store = MockDocumentStore(tmp_path / "docs.db")
        vector_store = MockChromaStore(tmp_path)
        queue = TrackingPendingQueue()

        kp = Keeper(
            store_path=tmp_path,
            doc_store=doc_store,
            vector_store=vector_store,
            pending_queue=queue,
        )
        # Injected stores → _is_local=False (cloud path)
        assert not kp._is_local

        # Bootstrap store-backed state docs once, then clear any setup work.
        kp.ensure_sysdocs()

        # Wire up the mock embedding provider and reset call counters
        # (Keeper init may trigger reconciliation that calls embed)
        embed_prov = mock_providers["embedding"]
        kp._embedding_provider = embed_prov
        kp._embedding_provider_loaded = True
        embed_prov.embed_calls = 0
        embed_prov.batch_calls = 0

        # Clear any items enqueued during init
        queue.clear()

        return kp, queue

    def test_put_defers_embedding(self, mock_providers, tmp_path):
        """put() should NOT call embed() in cloud mode; should enqueue 'embed' task."""
        kp, queue = self._make_keeper(mock_providers, tmp_path)
        embed = mock_providers["embedding"]

        kp.put("hello world", id="test-note", tags={"topic": "test"})

        # Embedding was NOT called synchronously
        assert embed.embed_calls == 0

        # An embed task was enqueued
        assert queue.count() >= 1
        embed_tasks = [i for i in queue._items if i["task_type"] == "embed"]
        assert len(embed_tasks) == 1
        assert embed_tasks[0]["id"] == "test-note"
        assert embed_tasks[0]["content"] == "hello world"

    def test_put_writes_doc_store_immediately(self, mock_providers, tmp_path):
        """Doc store should have the record even before embedding runs."""
        kp, queue = self._make_keeper(mock_providers, tmp_path)

        result = kp.put("some content", id="doc1", tags={"type": "note"})

        assert result is not None
        assert result.id == "doc1"
        # Doc is findable by ID
        doc = kp._document_store.get("default", "doc1")
        assert doc is not None
        assert doc.summary == "some content"

    def test_put_no_vector_store_entry_before_processing(self, mock_providers, tmp_path):
        """Vector store should NOT have the entry until background processes it."""
        kp, queue = self._make_keeper(mock_providers, tmp_path)

        kp.put("some content", id="doc1")

        # Nothing in vector store yet
        vec = kp._store.get("default", "doc1")
        assert vec is None

    def test_process_pending_embed_writes_vector_store(self, mock_providers, tmp_path):
        """process_pending should compute embedding and write to vector store."""
        kp, queue = self._make_keeper(mock_providers, tmp_path)
        embed = mock_providers["embedding"]

        # Put defers embedding
        kp.put("content for embedding", id="doc1")
        assert embed.embed_calls == 0

        # Process the queue
        result = kp.process_pending(limit=10)

        # Embedding was computed
        assert embed.embed_calls == 1

        # Vector store now has the entry
        chroma_coll = kp._resolve_chroma_collection()
        vec = kp._store.get(chroma_coll, "doc1")
        assert vec is not None

    def test_deferred_embed_content_change_archives_old(self, mock_providers, tmp_path):
        """When content changes, the old embedding should be archived as a version."""
        kp, queue = self._make_keeper(mock_providers, tmp_path)
        embed = mock_providers["embedding"]

        # First put — creates initial doc + deferred embed
        kp.put("version one content", id="doc1")
        kp.process_pending(limit=10)
        assert embed.embed_calls == 1

        # Verify vector store has the embedding
        chroma_coll = kp._resolve_chroma_collection()
        old_embedding = kp._store.get_embedding(chroma_coll, "doc1")
        assert old_embedding is not None

        # Second put — different content
        kp.put("version two content, completely different", id="doc1")

        # Should have enqueued with content_changed=True
        embed_tasks = [i for i in queue._items if i["task_type"] == "embed"]
        assert len(embed_tasks) == 1
        assert embed_tasks[0]["metadata"].get("content_changed") is True

        # Process
        kp.process_pending(limit=10)
        assert embed.embed_calls == 2

        # New embedding is in vector store
        new_embedding = kp._store.get_embedding(chroma_coll, "doc1")
        assert new_embedding is not None

    def test_deferred_embed_idempotent_content(self, mock_providers, tmp_path):
        """Same content re-put should NOT set content_changed flag."""
        kp, queue = self._make_keeper(mock_providers, tmp_path)

        kp.put("same content", id="doc1")
        kp.process_pending(limit=10)

        # Put same content again
        kp.put("same content", id="doc1")
        embed_tasks = [i for i in queue._items if i["task_type"] == "embed"]
        for task in embed_tasks:
            assert task["metadata"].get("content_changed") is not True

    def test_deleted_doc_skipped_by_embed_processor(self, mock_providers, tmp_path):
        """If doc is deleted before embed runs, the task should be a no-op."""
        kp, queue = self._make_keeper(mock_providers, tmp_path)
        embed = mock_providers["embedding"]

        kp.put("content to delete", id="doc1")

        # Delete before processing
        kp.delete("doc1")

        # Process — should not crash, should not embed
        kp.process_pending(limit=10)
        assert embed.embed_calls == 0

    def test_multiple_puts_last_content_wins(self, mock_providers, tmp_path):
        """Multiple puts should result in last content being embedded."""
        kp, queue = self._make_keeper(mock_providers, tmp_path)
        embed = mock_providers["embedding"]

        kp.put("first version", id="doc1")
        kp.put("second version", id="doc1")

        # Process all pending
        kp.process_pending(limit=10)

        # Vector store should have entry
        chroma_coll = kp._resolve_chroma_collection()
        vec = kp._store.get(chroma_coll, "doc1")
        assert vec is not None


class TestLocalModeUnchanged:
    """Local mode should continue to embed synchronously."""

    def test_local_put_embeds_immediately(self, mock_providers, tmp_path):
        """In local mode (_is_local=True), put() should embed synchronously."""
        kp = Keeper(store_path=tmp_path)
        embed = mock_providers["embedding"]

        assert kp._is_local  # Factory-created stores → local mode

        # Trigger system doc migration with a throwaway put, then reset counter
        kp.put("warmup", id="_warmup")
        kp.delete("_warmup")
        embed.embed_calls = 0

        kp.put("hello world", id="doc1")

        # Embedding was called synchronously (exactly 1 call for the content)
        assert embed.embed_calls == 1

        # Vector store has the entry immediately
        chroma_coll = kp._resolve_chroma_collection()
        vec = kp._store.get(chroma_coll, "doc1")
        assert vec is not None

    def test_local_put_records_note_when_embedding_unavailable(self, mock_providers, tmp_path):
        """Local put should preserve the note and queue reindex if embedding is offline."""
        kp = Keeper(store_path=tmp_path)

        # Warm system docs before simulating provider outage.
        kp.put("warmup", id="_warmup")
        kp.delete("_warmup")

        with patch.object(kp, "_get_embedding_provider", side_effect=RuntimeError("offline")):
            item = kp.put("hello world", id="doc-offline")

        assert item.id == "doc-offline"
        doc = kp._document_store.get("default", "doc-offline")
        assert doc is not None
        assert doc.summary == "hello world"

        chroma_coll = kp._resolve_chroma_collection()
        assert kp._store.get(chroma_coll, "doc-offline") is None

        queued = [
            item for item in kp._pending_queue._queue
            if item["id"] == "doc-offline"
        ]
        assert len(queued) == 1
        assert queued[0]["task_type"] == "reindex"

    def test_local_put_removes_stale_vector_when_reindex_deferred(self, mock_providers, tmp_path):
        """Updating an indexed note offline should not leave the old vector row live."""
        kp = Keeper(store_path=tmp_path)

        kp.put("old summary", id="doc-stale")
        chroma_coll = kp._resolve_chroma_collection()
        kp._store.upsert(
            collection=chroma_coll,
            id="doc-stale",
            embedding=[0.1] * mock_providers["embedding"].dimension,
            summary="old summary",
            tags={},
        )
        assert kp._store.get(chroma_coll, "doc-stale") is not None

        with patch.object(kp, "_get_embedding_provider", side_effect=RuntimeError("offline")):
            item = kp.put("new summary", id="doc-stale")

        assert item.summary == "new summary"
        doc = kp._document_store.get("default", "doc-stale")
        assert doc is not None
        assert doc.summary == "new summary"
        assert kp._store.get(chroma_coll, "doc-stale") is None

    def test_summary_mutation_removes_stale_vector_when_reindex_deferred(self, mock_providers, tmp_path):
        """Derived summary updates should not keep stale embeddings when offline."""
        kp = Keeper(store_path=tmp_path)

        kp.put("old summary", id="doc-summary-stale")
        chroma_coll = kp._resolve_chroma_collection()
        kp._store.upsert(
            collection=chroma_coll,
            id="doc-summary-stale",
            embedding=[0.2] * mock_providers["embedding"].dimension,
            summary="old summary",
            tags={},
        )
        assert kp._store.get(chroma_coll, "doc-summary-stale") is not None

        with patch.object(kp, "_get_embedding_provider", side_effect=RuntimeError("offline")):
            kp._apply_summary_mutation(
                "default",
                "doc-summary-stale",
                "new summary",
                intent="derived_summary_replace",
                embed=True,
            )

        doc = kp._document_store.get("default", "doc-summary-stale")
        assert doc is not None
        assert doc.summary == "new summary"
        assert kp._store.get(chroma_coll, "doc-summary-stale") is None


class TestEmbeddingDedup:
    """Content-hash dedup: reuse embeddings from docs with identical content."""

    def test_dedup_reuses_embedding_for_same_content(self, mock_providers, tmp_path):
        """Second put with same content but different ID should skip embed()."""
        kp = Keeper(store_path=tmp_path)
        embed = mock_providers["embedding"]

        # Warmup (system doc migration)
        kp.put("warmup", id="_warmup")
        kp.delete("_warmup")
        embed.embed_calls = 0

        # First put — must embed
        kp.put("identical content", id="file-a")
        assert embed.embed_calls == 1

        # Second put — same content, different ID → should reuse
        kp.put("identical content", id="file-b")
        assert embed.embed_calls == 1  # No additional embed call

        # Both have embeddings in vector store
        chroma_coll = kp._resolve_chroma_collection()
        assert kp._store.get_embedding(chroma_coll, "file-a") is not None
        assert kp._store.get_embedding(chroma_coll, "file-b") is not None

    def test_dedup_skips_when_donor_has_no_embedding(self, mock_providers, tmp_path):
        """If donor exists in doc store but not vector store, fall through to embed."""
        from tests.conftest import MockChromaStore, MockDocumentStore

        doc_store = MockDocumentStore(tmp_path / "docs.db")
        vector_store = MockChromaStore(tmp_path)
        queue = TrackingPendingQueue()

        kp = Keeper(
            store_path=tmp_path,
            doc_store=doc_store,
            vector_store=vector_store,
            pending_queue=queue,
        )
        kp.ensure_sysdocs()
        embed = mock_providers["embedding"]
        kp._embedding_provider = embed
        kp._embedding_provider_loaded = True
        embed.embed_calls = 0
        queue.clear()

        # First put defers embedding (cloud mode) — donor in doc store but no vector entry
        kp.put("shared content", id="file-a")
        assert embed.embed_calls == 0

        # Second put also defers — donor has no embedding yet
        kp.put("shared content", id="file-b")
        assert embed.embed_calls == 0

        # Process first embed task — no donor embedding available, must compute
        kp.process_pending(limit=1)
        assert embed.embed_calls == 1

        # Process second embed task — now donor has embedding, should dedup
        kp.process_pending(limit=1)
        assert embed.embed_calls == 1  # No additional call

    def test_dedup_dimension_mismatch_falls_through(self, mock_providers, tmp_path):
        """If donor embedding has wrong dimension, fall through to embed()."""
        kp = Keeper(store_path=tmp_path)
        embed = mock_providers["embedding"]

        kp.put("warmup", id="_warmup")
        kp.delete("_warmup")
        embed.embed_calls = 0

        # First put embeds normally
        kp.put("some content", id="file-a")
        assert embed.embed_calls == 1

        # Tamper: change the stored embedding dimension
        chroma_coll = kp._resolve_chroma_collection()
        old_emb = kp._store.get_embedding(chroma_coll, "file-a")
        # Store a wrong-dimension embedding
        kp._store.upsert(
            collection=chroma_coll,
            id="file-a",
            embedding=old_emb[:2],  # truncate to 2 dims
            summary="some content",
            tags={},
        )

        # Second put — donor has wrong dimension, must embed fresh
        kp.put("some content", id="file-b")
        assert embed.embed_calls == 2  # Had to call embed()

    def test_dedup_does_not_affect_different_content(self, mock_providers, tmp_path):
        """Different content should always embed independently."""
        kp = Keeper(store_path=tmp_path)
        embed = mock_providers["embedding"]

        kp.put("warmup", id="_warmup")
        kp.delete("_warmup")
        embed.embed_calls = 0

        kp.put("content A", id="file-a")
        kp.put("content B", id="file-b")
        assert embed.embed_calls == 2


    def test_cloud_dedup_skips_embed_enqueue(self, mock_providers, tmp_path):
        """Cloud mode: if donor has embedding, copy it and skip enqueue."""
        from tests.conftest import MockChromaStore, MockDocumentStore

        doc_store = MockDocumentStore(tmp_path / "docs.db")
        vector_store = MockChromaStore(tmp_path)
        queue = TrackingPendingQueue()

        kp = Keeper(
            store_path=tmp_path,
            doc_store=doc_store,
            vector_store=vector_store,
            pending_queue=queue,
        )
        kp.ensure_sysdocs()
        embed = mock_providers["embedding"]
        kp._embedding_provider = embed
        kp._embedding_provider_loaded = True
        embed.embed_calls = 0
        queue.clear()

        # First put + process: donor gets an embedding
        kp.put("shared content", id="file-a")
        kp.process_pending(limit=10)
        assert embed.embed_calls == 1

        # Re-inject mock provider (process_pending releases it)
        kp._embedding_provider = embed
        embed.embed_calls = 0
        queue.clear()

        # Second put: same content, different ID — should dedup and skip enqueue
        kp.put("shared content", id="file-b")
        embed_tasks = [i for i in queue._items if i["task_type"] == "embed"]
        assert len(embed_tasks) == 0  # No embed enqueued
        assert embed.embed_calls == 0  # No embed call needed

        # Vector store has the entry immediately (copied from donor)
        # Use "default" — the collection used before embedding identity was resolved
        assert vector_store.get_embedding("default", "file-b") is not None

    def test_full_hash_mismatch_rejects_dedup(self, mock_providers, tmp_path):
        """Short hash collision: if full hashes differ, dedup should not match."""
        kp = Keeper(store_path=tmp_path)
        embed = mock_providers["embedding"]

        kp.put("warmup", id="_warmup")
        kp.delete("_warmup")
        embed.embed_calls = 0

        # First put
        kp.put("content alpha", id="file-a")
        assert embed.embed_calls == 1

        # Tamper: set the same short hash but different full hash on the donor
        doc_coll = "default"
        donor = kp._document_store.get(doc_coll, "file-a")
        # Overwrite with a fake full hash that won't match
        import hashlib
        kp._document_store.upsert(
            doc_coll, "file-a", donor.summary, donor.tags,
            content_hash=donor.content_hash,
            content_hash_full="fake_full_hash_that_does_not_match",
        )

        # Second put with same short hash but different actual content
        # (simulate a collision — same _content_hash but different _content_hash_full)
        from keep.processors import _content_hash
        # Create content that happens to have the same short hash... hard to do.
        # Instead, directly manipulate: set file-b's content_hash = file-a's content_hash
        # by using the same content (same short hash), but the donor's full hash is wrong.
        kp.put("content alpha", id="file-b")
        # The donor's full hash was tampered to "fake_full_hash_that_does_not_match",
        # but file-b's actual full hash is SHA256("content alpha").
        # find_by_content_hash should reject the donor → must embed fresh.
        assert embed.embed_calls == 2

    def test_content_unchanged_missing_embedding_tries_dedup(self, mock_providers, tmp_path):
        """When re-putting same content but embedding is missing, try dedup before embed."""
        kp = Keeper(store_path=tmp_path)
        embed = mock_providers["embedding"]

        kp.put("warmup", id="_warmup")
        kp.delete("_warmup")
        embed.embed_calls = 0

        # Create two docs with same content
        kp.put("shared content", id="file-a")
        kp.put("shared content", id="file-b")
        assert embed.embed_calls == 1  # file-b deduped from file-a

        # Delete file-b's embedding from vector store (simulating a gap)
        chroma_coll = kp._resolve_chroma_collection()
        kp._store.delete(chroma_coll, "file-b")

        embed.embed_calls = 0

        # Re-put file-b with same content → content_unchanged=True, but no embedding.
        # Should dedup from file-a instead of calling embed().
        kp.put("shared content", id="file-b")
        assert embed.embed_calls == 0  # Deduped from file-a, no embed needed


class TestNullPendingQueueSignature:
    """NullPendingQueue should accept task_type and metadata kwargs."""

    def test_enqueue_accepts_kwargs(self):
        from keep.backend import NullPendingQueue
        q = NullPendingQueue()
        # Should not raise
        q.enqueue("id", "coll", "content", task_type="embed", metadata={"key": "val"})
        assert q.count() == 0  # still no-ops
