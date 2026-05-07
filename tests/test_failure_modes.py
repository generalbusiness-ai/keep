"""Tests for production failure modes.

Covers the scenarios that actually broke in production:
- Embedding dimension mismatches (384d errors)
- Dual-write partial failures (doc store OK, vector store fails)
- process_pending retry/abandon flow (pending stalls)
- Provider timeout propagation
- Concurrent writers on same doc
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from keep.api import Keeper
from keep.document_store import PartInfo
from tests.conftest import MockChromaStore

MAX_SUMMARY_ATTEMPTS = 5  # default; tests use this as the expected value
from keep.pending_summaries import PendingSummaryQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FailingChromaStore:
    """ChromaDB mock that raises on upsert — simulates partial dual-write."""

    def __init__(self, real_store):
        self._real = real_store
        self.fail_upsert = False
        self.upsert_calls = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def upsert(self, *args, **kwargs):
        self.upsert_calls += 1
        if self.fail_upsert:
            raise RuntimeError("ChromaDB simulated failure")
        return self._real.upsert(*args, **kwargs)


class TimeoutEmbeddingProvider:
    """Embedding provider that times out on Nth call."""

    dimension = 384
    model_name = "timeout-model"

    def __init__(self, fail_until: int = 0):
        self.embed_calls = 0
        self.fail_until = fail_until

    def embed(self, text: str, **kwargs) -> list[float]:
        self.embed_calls += 1
        if self.embed_calls <= self.fail_until:
            raise TimeoutError(f"Connection timed out (call {self.embed_calls})")
        return [0.1] * self.dimension

    def embed_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Embedding dimension validation
# ---------------------------------------------------------------------------

class TestEmbeddingDimensionValidation:
    """Tests for _validate_embedding_identity — the 384d production failure."""

    def test_first_use_records_identity(self, mock_providers, tmp_path):
        """First embed should record identity to config."""
        kp = Keeper(store_path=tmp_path)
        embed = mock_providers["embedding"]
        assert embed.dimension == 384

        kp.put("hello", id="doc1")

        assert kp._config.embedding_identity is not None
        assert kp._config.embedding_identity.dimension == 384

        kp.close()

    def test_first_use_sets_store_dimension(self, mock_providers, tmp_path):
        """First embed should set the vector store dimension."""
        kp = Keeper(store_path=tmp_path)
        kp.put("hello", id="doc1")

        assert kp._store.embedding_dimension == 384
        kp.close()

    def test_dimension_change_drops_collection_and_updates(self, mock_providers, tmp_path):
        """Changing embedding dimension should drop collection and update config."""
        kp = Keeper(store_path=tmp_path)

        # First put establishes 384d identity
        kp.put("hello", id="doc1")
        assert kp._config.embedding_identity.dimension == 384

        # Simulate a new provider with different dimension
        new_embed = mock_providers["embedding"]
        new_embed.dimension = 768

        # Directly call validation — this is what _get_embedding_provider calls
        kp._validate_embedding_identity(new_embed)

        # Identity updated to 768d
        assert kp._config.embedding_identity.dimension == 768
        # Store dimension updated
        assert kp._store.embedding_dimension == 768
        kp.close()

    def test_same_dimension_different_model_reindexes(self, mock_providers, tmp_path):
        """Same dimension but different model should update identity and reindex."""
        kp = Keeper(store_path=tmp_path)
        kp.put("hello", id="doc1")

        # Create a provider with different model name (same dimension)
        new_embed = mock_providers["embedding"]
        new_embed.model_name = "different-model"

        kp._validate_embedding_identity(new_embed)

        # Identity should reflect new model
        assert kp._config.embedding_identity.model == "different-model"
        kp.close()

    def test_openrouter_compatible_model_switch_skips_reindex(self, mock_providers, tmp_path):
        """OpenAI and OpenRouter identities for the same model reuse the index."""
        kp = Keeper(store_path=tmp_path)
        kp.put("hello", id="doc1")

        kp._config.embedding.name = "openrouter"
        openrouter_embed = MagicMock()
        openrouter_embed.dimension = 384
        openrouter_embed.model_name = "openai/mock-model"
        kp.enqueue_reindex = MagicMock(return_value={"enqueued": 0, "versions": 0})

        kp._validate_embedding_identity(openrouter_embed)

        assert kp._config.embedding_identity.provider == "openrouter"
        assert kp._config.embedding_identity.model == "openai/mock-model"
        assert kp._config.embedding_identity.key == "openai_mock_model"
        kp.enqueue_reindex.assert_not_called()
        kp.close()

    def test_identity_uses_legacy_provider_model_attr(self, tmp_path):
        """Identity resolution falls back to `.model` when `.model_name` is absent."""
        kp = Keeper(store_path=tmp_path)

        legacy_embed = MagicMock()
        legacy_embed.dimension = 384
        legacy_embed.model = "legacy-model"

        kp._validate_embedding_identity(legacy_embed)

        assert kp._config.embedding_identity is not None
        assert kp._config.embedding_identity.model == "legacy-model"
        kp.close()


# ---------------------------------------------------------------------------
# Dual-write partial failures
# ---------------------------------------------------------------------------

class TestDualWriteRecovery:
    """Tests for partial failures during dual-write to doc store + vector store."""

    def test_doc_store_ok_vector_store_fails(self, mock_providers, tmp_path):
        """If ChromaDB fails after doc store write, doc should still exist."""
        kp = Keeper(store_path=tmp_path)
        embed = mock_providers["embedding"]

        # Warmup
        kp.put("warmup", id="_w")
        kp.delete("_w")
        embed.embed_calls = 0

        # Wrap the vector store to fail on next upsert
        real_store = kp._store
        failing_store = FailingChromaStore(real_store)
        kp._store = failing_store
        failing_store.fail_upsert = True

        # put() should raise via the flow wrapper's binding-error surface.
        with pytest.raises(ValueError, match="ChromaDB simulated"):
            kp.put("important content", id="doc1")

        # Doc store should have the record (written first)
        doc = kp._document_store.get("default", "doc1")
        assert doc is not None
        assert doc.summary == "important content"

        # Vector store should NOT have it
        chroma_coll = kp._resolve_chroma_collection()
        assert real_store.get(chroma_coll, "doc1") is None

        kp.close()

    def test_reconcile_detects_drift(self, mock_providers, tmp_path):
        """reconcile() should detect doc store / vector store drift."""
        kp = Keeper(store_path=tmp_path)

        # Create a doc normally
        kp.put("content", id="doc1")

        # Delete from vector store only (simulate partial failure)
        chroma_coll = kp._resolve_chroma_collection()
        kp._store.delete(chroma_coll, "doc1")
        assert kp._store.get(chroma_coll, "doc1") is None

        # Reconcile (detect only) should report missing
        stats = kp.reconcile(fix=False)
        assert stats["missing_from_index"] > 0
        assert "doc1" in stats.get("missing_ids", [])
        kp.close()

    def test_reconcile_fixes_drift(self, mock_providers, tmp_path):
        """reconcile(fix=True) should re-embed docs missing from vector store."""
        kp = Keeper(store_path=tmp_path)
        embed = mock_providers["embedding"]

        # Create a doc normally
        kp.put("content", id="doc1")

        # Delete from vector store only (simulate partial failure)
        chroma_coll = kp._resolve_chroma_collection()
        kp._store.delete(chroma_coll, "doc1")

        # Reconcile with fix — should re-embed
        embed.embed_calls = 0
        stats = kp.reconcile(fix=True)

        assert stats["fixed"] > 0
        # doc1 should be back in vector store
        assert kp._store.get(chroma_coll, "doc1") is not None
        kp.close()

    def test_system_note_put_skips_vector_index(self, mock_providers, tmp_path):
        """Dot-prefixed system notes should not create searchable vector rows."""
        kp = Keeper(store_path=tmp_path)
        chroma_coll = kp._resolve_chroma_collection()

        item = kp.put("cursor payload", id=".cursor/test")

        assert item.id == ".cursor/test"
        assert kp._document_store.get("default", ".cursor/test") is not None
        assert kp._store.get(chroma_coll, ".cursor/test") is None
        kp.close()

    def test_reconcile_ignores_system_notes_missing_from_index(self, mock_providers, tmp_path):
        """System notes absent from Chroma should not be treated as drift."""
        kp = Keeper(store_path=tmp_path)
        kp.put("cursor payload", id=".cursor/test")

        stats = kp.reconcile(fix=False)

        assert ".cursor/test" not in stats.get("missing_ids", [])
        assert stats["missing_from_index"] == 0
        kp.close()

    def test_reconcile_removes_stale_system_note_rows(self, mock_providers, tmp_path):
        """Reconcile should delete old system-note rows that still exist in Chroma."""
        kp = Keeper(store_path=tmp_path)
        chroma_coll = kp._resolve_chroma_collection()

        kp.put("cursor payload", id=".cursor/test")
        kp._store.upsert(
            collection=chroma_coll,
            id=".cursor/test",
            embedding=[0.1] * 384,
            summary="cursor payload",
            tags={"_source": "inline"},
        )

        stats = kp.reconcile(fix=True)

        assert ".cursor/test" in stats.get("orphaned_ids", [])
        assert stats["removed"] > 0
        assert kp._store.get(chroma_coll, ".cursor/test") is None
        kp.close()


# ---------------------------------------------------------------------------
# process_pending retry and abandon flow
# ---------------------------------------------------------------------------

class TestProcessPendingRetry:
    """Tests for the full retry → backoff → abandon → dead letter flow."""

    def _make_keeper(self, mock_providers, tmp_path):
        """Create a Keeper with real pending queue for retry testing."""
        kp = Keeper(store_path=tmp_path)
        kp._needs_sysdoc_migration = False
        # Replace mock pending queue with real SQLite-backed one
        kp._pending_queue = PendingSummaryQueue(tmp_path / "pending.db")
        embed = mock_providers["embedding"]
        kp._embedding_provider = embed
        embed.embed_calls = 0
        return kp

    def test_failed_item_retried_on_next_process(self, mock_providers, tmp_path):
        """Failed items should be retried (after backoff)."""
        kp = self._make_keeper(mock_providers, tmp_path)

        # Enqueue an embed task
        kp._pending_queue.enqueue("doc1", "default", "content", task_type="embed")
        assert kp._pending_queue.count() == 1

        # Inject an embedding provider that fails on first call
        embed = mock_providers["embedding"]
        original_embed = embed.embed
        call_count = [0]
        def failing_embed(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                raise RuntimeError("transient embedding failure")
            return original_embed(*args, **kwargs)
        embed.embed = failing_embed

        # Put a doc so embed actually runs
        kp._document_store.upsert("default", "doc1", "raw content", {})
        result = kp.process_pending(limit=10)
        assert result["failed"] == 1

        # Item still in queue (with backoff)
        assert kp._pending_queue.count() == 1

        # Clear backoff for test speed
        kp._pending_queue._conn.execute(
            "UPDATE pending_summaries SET retry_after = NULL"
        )
        kp._pending_queue._conn.commit()

        # Second process — provider succeeds now
        embed.embed = original_embed

        result = kp.process_pending(limit=10)
        assert result["processed"] == 1
        assert kp._pending_queue.count() == 0

        kp.close()

    def test_item_abandoned_after_max_attempts(self, mock_providers, tmp_path):
        """Items should be dead-lettered after MAX_SUMMARY_ATTEMPTS."""
        kp = self._make_keeper(mock_providers, tmp_path)

        kp._pending_queue.enqueue("doc1", "default", "content", task_type="embed")

        # Simulate MAX_SUMMARY_ATTEMPTS worth of failed dequeues
        for _ in range(MAX_SUMMARY_ATTEMPTS):
            kp._pending_queue._conn.execute(
                "UPDATE pending_summaries SET retry_after = NULL"
            )
            kp._pending_queue._conn.commit()
            kp._pending_queue.dequeue(limit=1)
            kp._pending_queue.fail("doc1", "default", task_type="embed",
                                   error="simulated failure")

        # Clear backoff for the final dequeue
        kp._pending_queue._conn.execute(
            "UPDATE pending_summaries SET retry_after = NULL"
        )
        kp._pending_queue._conn.commit()

        # Process — should abandon (attempts >= MAX)
        kp._document_store.upsert("default", "doc1", "content", {})
        result = kp.process_pending(limit=10)
        assert result["abandoned"] == 1
        assert result["processed"] == 0

        # Item is in dead letter, not pending
        assert kp._pending_queue.count() == 0
        failed = kp._pending_queue.list_failed()
        assert len(failed) == 1
        assert failed[0]["id"] == "doc1"

        kp.close()

    def test_deleted_doc_completed_not_failed(self, mock_providers, tmp_path):
        """Deleted docs should be completed (not retried forever)."""
        kp = self._make_keeper(mock_providers, tmp_path)

        # Enqueue embed for a doc, then delete the doc
        kp._pending_queue.enqueue("doc1", "default", "content", task_type="embed")
        # Don't put doc in doc store — simulates deletion

        result = kp.process_pending(limit=10)
        # Should complete (not fail) — deleted doc is a no-op
        assert result["processed"] == 1
        assert result["failed"] == 0
        assert kp._pending_queue.count() == 0

        kp.close()

    def test_backoff_prevents_immediate_retry(self, mock_providers, tmp_path):
        """Failed items should not be dequeued during backoff period."""
        kp = self._make_keeper(mock_providers, tmp_path)

        kp._pending_queue.enqueue("doc1", "default", "content", task_type="embed")
        kp._document_store.upsert("default", "doc1", "content", {})

        # Make embed fail
        kp._embedding_provider = TimeoutEmbeddingProvider(fail_until=99)


        # First process — fails, sets backoff
        result = kp.process_pending(limit=10)
        assert result["failed"] == 1

        # Immediate second process — item should be skipped (backoff active)
        result = kp.process_pending(limit=10)
        assert result["processed"] == 0
        assert result["failed"] == 0

        kp.close()

    def test_reindex_provider_creation_failure_uses_queue_backoff(self, mock_providers, tmp_path):
        """Reindex batches should not crash the daemon when embeddings are offline."""
        kp = self._make_keeper(mock_providers, tmp_path)
        kp._pending_queue.enqueue("doc1", "default", "summary", task_type="reindex")
        kp._document_store.upsert("default", "doc1", "raw content", {})

        with patch.object(kp, "_get_embedding_provider", side_effect=RuntimeError("offline")):
            result = kp.process_pending(limit=10)

        assert result["failed"] == 1
        assert result["processed"] == 0
        row = kp._pending_queue._conn.execute(
            "SELECT status, retry_after, last_error FROM pending_summaries WHERE id = 'doc1'"
        ).fetchone()
        assert row[0] == "pending"
        assert row[1] is not None
        assert "offline" in row[2]

        kp.close()

    def test_reindex_provider_creation_failure_preserves_deduped_writes(self, mock_providers, tmp_path):
        """Provider outages should not fail entries that already have reusable embeddings."""
        kp = self._make_keeper(mock_providers, tmp_path)
        kp._pending_queue.enqueue("deduped", "default", "deduped summary", task_type="reindex")
        kp._pending_queue.enqueue("needs-provider", "default", "unique summary", task_type="reindex")
        kp._document_store.upsert("default", "deduped", "deduped summary", {})
        kp._document_store.upsert("default", "needs-provider", "unique summary", {})

        embedding = [0.2] * mock_providers["embedding"].dimension
        with (
            patch.object(kp, "_try_dedup_embedding", side_effect=[embedding, None]),
            patch.object(kp, "_get_embedding_provider", side_effect=RuntimeError("offline")),
        ):
            result = kp.process_pending(limit=10)

        assert result["processed"] == 1
        assert result["failed"] == 1
        chroma_coll = kp._resolve_chroma_collection()
        assert kp._store.get(chroma_coll, "deduped") is not None
        row = kp._pending_queue._conn.execute(
            "SELECT status, retry_after FROM pending_summaries WHERE id = 'needs-provider'"
        ).fetchone()
        assert row[0] == "pending"
        assert row[1] is not None

        kp.close()


# ---------------------------------------------------------------------------
# Provider timeout propagation
# ---------------------------------------------------------------------------

class TestProviderTimeouts:
    """Tests for timeout/error propagation from providers to pending queue."""

    def test_embed_timeout_marks_item_failed(self, mock_providers, tmp_path):
        """TimeoutError from embed() should be caught and item marked failed."""
        kp = Keeper(store_path=tmp_path)
        kp._needs_sysdoc_migration = False
        kp._pending_queue = PendingSummaryQueue(tmp_path / "pending.db")

        kp._pending_queue.enqueue("doc1", "default", "content", task_type="embed")
        kp._document_store.upsert("default", "doc1", "content", {})

        # Inject timeout provider
        kp._embedding_provider = TimeoutEmbeddingProvider(fail_until=1)


        result = kp.process_pending(limit=10)
        assert result["failed"] == 1
        assert "timed out" in result["errors"][0].lower() or "TimeoutError" in result["errors"][0]

        # Item still pending (will be retried after backoff)
        assert kp._pending_queue.count() == 1
        kp.close()

    def test_provider_error_doesnt_crash_batch(self, mock_providers, tmp_path):
        """One item failing shouldn't prevent others from processing."""
        kp = Keeper(store_path=tmp_path)
        kp._needs_sysdoc_migration = False
        kp._pending_queue = PendingSummaryQueue(tmp_path / "pending.db")

        # Enqueue two embed tasks
        kp._pending_queue.enqueue("doc1", "default", "content1", task_type="embed")
        kp._pending_queue.enqueue("doc2", "default", "content2", task_type="embed")
        kp._document_store.upsert("default", "doc1", "content1", {})
        kp._document_store.upsert("default", "doc2", "content2", {})

        # Fails on first call, succeeds on second
        kp._embedding_provider = TimeoutEmbeddingProvider(fail_until=1)


        result = kp.process_pending(limit=10)
        # One fails, one succeeds
        assert result["failed"] + result["processed"] == 2
        assert result["failed"] >= 1
        kp.close()


class TestReindexOptimizations:
    """Focused tests for reindex correctness and batch behavior."""

    def _make_keeper(self, mock_providers, tmp_path):
        kp = Keeper(store_path=tmp_path)
        kp._needs_sysdoc_migration = False
        kp._pending_queue = PendingSummaryQueue(tmp_path / "pending.db")
        kp._embedding_provider = mock_providers["embedding"]
        return kp

    def test_reindex_uses_fresh_main_doc_summary(self, mock_providers, tmp_path):
        kp = self._make_keeper(mock_providers, tmp_path)
        try:
            kp._document_store.upsert("default", "doc1", "fresh summary", {"topic": "new"})
            kp._pending_queue.enqueue(
                "doc1", "default", "stale summary",
                task_type="reindex",
                metadata={"tags": {"topic": "old"}},
            )

            embed = kp._embedding_provider
            original_embed = embed.embed
            seen_texts: list[str] = []

            def tracking_embed(text: str, **kwargs):
                seen_texts.append(text)
                return original_embed(text, **kwargs)

            embed.embed = tracking_embed

            result = kp.process_pending(limit=10)

            chroma_coll = kp._resolve_chroma_collection()
            stored = kp._store.get(chroma_coll, "doc1")
            assert result["processed"] == 1
            assert stored is not None
            assert stored.summary == "fresh summary"
            assert seen_texts == ["fresh summary"]
        finally:
            kp.close()

    def test_single_reindex_path_reuses_donor_embedding(self, mock_providers, tmp_path):
        kp = self._make_keeper(mock_providers, tmp_path)
        try:
            chroma_coll = kp._resolve_chroma_collection()
            donor_embedding = [0.9] * 384
            kp._document_store.upsert(
                "default",
                "donor",
                "shared summary",
                {},
                content_hash="shared-hash",
            )
            kp._store.upsert(
                chroma_coll,
                "donor",
                donor_embedding,
                "shared summary",
                {},
            )
            kp._document_store.upsert(
                "default",
                "doc1",
                "shared summary",
                {},
                content_hash="shared-hash",
            )

            def fail_embed(text: str, **kwargs):
                raise AssertionError("single reindex path should reuse donor embedding")

            kp._embedding_provider.embed = fail_embed

            kp._process_pending_reindex(
                SimpleNamespace(
                    id="doc1",
                    collection="default",
                    content="stale summary",
                    metadata={},
                ),
            )

            stored = kp._store.get(chroma_coll, "doc1")
            assert stored is not None
            assert kp._store.get_embedding(chroma_coll, "doc1") == donor_embedding
        finally:
            kp.close()

    def test_reindex_missing_version_deletes_stale_index_entry(self, mock_providers, tmp_path):
        kp = self._make_keeper(mock_providers, tmp_path)
        try:
            chroma_coll = kp._resolve_chroma_collection()
            kp._store.upsert_version(
                chroma_coll,
                "doc1",
                2,
                [0.1] * 384,
                "old version",
                {"topic": "history"},
            )
            kp._pending_queue.enqueue(
                "doc1@v2", "default", "stale version",
                task_type="reindex",
                metadata={"version": 2, "base_id": "doc1", "tags": {"topic": "stale"}},
            )

            result = kp.process_pending(limit=10)

            assert result["processed"] == 1
            assert kp._store.get(chroma_coll, "doc1@v2") is None
        finally:
            kp.close()

    def test_reindex_missing_part_deletes_stale_index_entry(self, mock_providers, tmp_path):
        kp = self._make_keeper(mock_providers, tmp_path)
        try:
            chroma_coll = kp._resolve_chroma_collection()
            kp._store.upsert_part(
                chroma_coll,
                "doc1",
                3,
                [0.1] * 384,
                "old part",
                {"topic": "history"},
            )
            kp._pending_queue.enqueue(
                "doc1@p3", "default", "stale part",
                task_type="reindex",
                metadata={"part_num": 3, "base_id": "doc1", "tags": {"topic": "stale"}},
            )

            result = kp.process_pending(limit=10)

            assert result["processed"] == 1
            assert kp._store.get(chroma_coll, "doc1@p3") is None
        finally:
            kp.close()

    def test_reindex_batch_sleeps_and_releases_once(self, mock_providers, tmp_path):
        kp = self._make_keeper(mock_providers, tmp_path)
        try:
            kp._document_store.upsert("default", "doc1", "summary one", {})
            kp._document_store.upsert("default", "doc2", "summary two", {})
            kp._pending_queue.enqueue("doc1", "default", "stale one", task_type="reindex")
            kp._pending_queue.enqueue("doc2", "default", "stale two", task_type="reindex")

            with (
                patch.object(kp, "_release_embedding_provider") as release_provider,
                patch("keep._background_processing.time.sleep") as sleep_mock,
            ):
                result = kp.process_pending(limit=10)

            assert result["processed"] == 2
            release_provider.assert_called_once()
            sleep_mock.assert_called_once_with(0.1)
        finally:
            kp.close()

    def test_enqueue_reindex_bulk_fetches_versions_and_parts(self, mock_providers, tmp_path):
        kp = self._make_keeper(mock_providers, tmp_path)
        try:
            kp._document_store.upsert("default", "doc1", "summary", {})
            kp._document_store._versions[("default", "doc1")] = [{
                "version": 2,
                "summary": "archived summary",
                "tags": {"topic": "history"},
                "content_hash": None,
                "created_at": "2026-01-01T00:00:00",
            }]
            kp._document_store.upsert_single_part(
                "default",
                "doc1",
                PartInfo(
                    part_num=3,
                    summary="part summary",
                    tags={"topic": "part"},
                    created_at="2026-01-02T00:00:00",
                ),
            )

            original_list_versions_many = kp._document_store.list_versions_many
            original_list_parts_many = kp._document_store.list_parts_many
            bulk_version_calls: list[list[str]] = []
            bulk_part_calls: list[list[str]] = []

            def fail_list_versions(collection: str, id: str, limit: int = 10):
                raise AssertionError("enqueue_reindex should use list_versions_many")

            def fail_list_parts(collection: str, id: str):
                raise AssertionError("enqueue_reindex should use list_parts_many")

            def tracking_list_versions_many(collection: str, ids: list[str]):
                bulk_version_calls.append(list(ids))
                return original_list_versions_many(collection, ids)

            def tracking_list_parts_many(collection: str, ids: list[str]):
                bulk_part_calls.append(list(ids))
                return original_list_parts_many(collection, ids)

            kp._document_store.list_versions = fail_list_versions
            kp._document_store.list_parts = fail_list_parts
            kp._document_store.list_versions_many = tracking_list_versions_many
            kp._document_store.list_parts_many = tracking_list_parts_many

            kp.enqueue_reindex()

            assert bulk_version_calls == [["doc1"]]
            assert bulk_part_calls == [["doc1"]]
            queued_ids = [item.id for item in kp._pending_queue.dequeue(limit=10)]
            assert queued_ids == ["doc1", "doc1@v2", "doc1@p3"]
        finally:
            kp.close()

    def test_enqueue_reindex_skips_system_docs(self, mock_providers, tmp_path):
        kp = self._make_keeper(mock_providers, tmp_path)
        try:
            kp._document_store.upsert("default", "doc1", "summary", {})
            kp._document_store.upsert("default", ".cursor/x", "cursor", {})

            kp.enqueue_reindex()

            queued_ids = [item.id for item in kp._pending_queue.dequeue(limit=10)]
            assert "doc1" in queued_ids
            assert ".cursor/x" not in queued_ids
        finally:
            kp.close()

    def test_edge_tag_auto_vivify_queues_empty_reindex_content(self, mock_providers, tmp_path):
        kp = Keeper(store_path=tmp_path)
        try:
            doc_coll = kp._resolve_doc_collection()
            kp._document_store.upsert(doc_coll, ".tag/ref", "ref tag", {"_inverse": "backref"})
            kp._process_edge_tags("source", {"ref": "target"}, {}, doc_coll)

            target_items = [item for item in kp._pending_queue._queue if item["id"] == "target"]
            assert target_items
            assert target_items[-1]["content"] == ""
        finally:
            kp.close()

    def test_backfill_auto_vivify_queues_empty_reindex_content(self, mock_providers, tmp_path):
        kp = Keeper(store_path=tmp_path)
        try:
            doc_coll = kp._resolve_doc_collection()
            kp._document_store.upsert(doc_coll, ".tag/ref", "ref tag", {"_inverse": "backref"})
            kp._document_store.upsert(doc_coll, "source", "source summary", {"ref": "target"})

            item = SimpleNamespace(
                id="_backfill:ref",
                collection=doc_coll,
                metadata={"predicate": "ref", "inverse": "backref"},
            )

            kp._process_pending_backfill_edges(item)

            target_items = [entry for entry in kp._pending_queue._queue if entry["id"] == "target"]
            assert target_items
            assert target_items[-1]["content"] == ""
        finally:
            kp.close()

    def test_reindex_fast_path_uses_embed_batch_once_for_unique_summaries(self, mock_providers, tmp_path):
        kp = self._make_keeper(mock_providers, tmp_path)
        try:
            kp._document_store.upsert("default", "doc1", "same summary", {})
            kp._document_store.upsert("default", "doc2", "same summary", {})
            kp._pending_queue.enqueue("doc1", "default", "stale one", task_type="reindex")
            kp._pending_queue.enqueue("doc2", "default", "stale two", task_type="reindex")

            seen_batches: list[list[str]] = []

            def fake_embed_batch(texts: list[str], **kwargs):
                seen_batches.append(list(texts))
                return [[0.1] * 384 for _ in texts]

            kp._embedding_provider.embed_batch = fake_embed_batch

            result = kp.process_pending(limit=10)

            assert result["processed"] == 2
            assert seen_batches == [["same summary"]]
        finally:
            kp.close()

    def test_reindex_batch_preserves_version_and_part_metadata(self, mock_providers, tmp_path):
        kp = self._make_keeper(mock_providers, tmp_path)
        try:
            chroma_coll = kp._resolve_chroma_collection()
            kp._document_store._versions[("default", "doc1")] = [{
                "version": 2,
                "summary": "archived summary",
                "tags": {"topic": "history"},
                "content_hash": None,
                "created_at": "2026-01-01T00:00:00",
            }]
            kp._document_store.upsert_single_part(
                "default",
                "doc1",
                PartInfo(
                    part_num=3,
                    summary="part summary",
                    tags={"topic": "part"},
                    created_at="2026-01-02T00:00:00",
                ),
            )
            kp._pending_queue.enqueue(
                "doc1@v2", "default", "stale version",
                task_type="reindex",
                metadata={"version": 2, "base_id": "doc1"},
            )
            kp._pending_queue.enqueue(
                "doc1@p3", "default", "stale part",
                task_type="reindex",
                metadata={"part_num": 3, "base_id": "doc1"},
            )

            result = kp.process_pending(limit=10)

            assert result["processed"] == 2
            version_entry = kp._store.get(chroma_coll, "doc1@v2")
            part_entry = kp._store.get(chroma_coll, "doc1@p3")
            assert version_entry is not None
            assert part_entry is not None
            assert version_entry.tags["_base_id"] == "doc1"
            assert version_entry.tags["_version"] == "2"
            assert part_entry.tags["_base_id"] == "doc1"
            assert part_entry.tags["_part_num"] == "3"
        finally:
            kp.close()

    def test_reindex_fast_path_batches_mixed_doc_version_part_in_one_write(self, mock_providers, tmp_path):
        kp = self._make_keeper(mock_providers, tmp_path)
        try:
            kp._document_store.upsert("default", "doc1", "doc summary", {})
            kp._document_store._versions[("default", "doc1")] = [{
                "version": 2,
                "summary": "archived summary",
                "tags": {"topic": "history"},
                "content_hash": None,
                "created_at": "2026-01-01T00:00:00",
            }]
            kp._document_store.upsert_single_part(
                "default",
                "doc1",
                PartInfo(
                    part_num=3,
                    summary="part summary",
                    tags={"topic": "part"},
                    created_at="2026-01-02T00:00:00",
                ),
            )
            kp._pending_queue.enqueue("doc1", "default", "stale doc", task_type="reindex")
            kp._pending_queue.enqueue(
                "doc1@v2", "default", "stale version",
                task_type="reindex",
                metadata={"version": 2, "base_id": "doc1"},
            )
            kp._pending_queue.enqueue(
                "doc1@p3", "default", "stale part",
                task_type="reindex",
                metadata={"part_num": 3, "base_id": "doc1"},
            )

            original_upsert_batch = kp._store.upsert_batch
            call_count = 0

            def tracking_upsert_batch(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return original_upsert_batch(*args, **kwargs)

            kp._store.upsert_batch = tracking_upsert_batch

            result = kp.process_pending(limit=10)

            assert result["processed"] == 3
            assert call_count == 1
        finally:
            kp.close()


# ---------------------------------------------------------------------------
# Concurrent writers
# ---------------------------------------------------------------------------

class TestConcurrentWriters:
    """Tests for concurrent put() calls on same document."""

    def test_concurrent_puts_both_succeed(self, mock_providers, tmp_path):
        """Two threads writing to same ID should both complete without error."""
        kp = Keeper(store_path=tmp_path)
        embed = mock_providers["embedding"]

        # Warmup system docs
        kp.put("warmup", id="_w")
        kp.delete("_w")

        errors = []
        results = [None, None]

        def writer(idx, content):
            try:
                results[idx] = kp.put(content, id="shared-doc")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer, args=(0, "version A content"))
        t2 = threading.Thread(target=writer, args=(1, "version B content"))

        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # No crashes
        assert not errors, f"Concurrent writes caused errors: {errors}"

        # Doc exists (one version wins)
        doc = kp.get("shared-doc")
        assert doc is not None
        assert doc.summary in ("version A content", "version B content")

        kp.close()

    def test_concurrent_put_and_delete(self, mock_providers, tmp_path):
        """Concurrent put + delete shouldn't crash."""
        kp = Keeper(store_path=tmp_path)

        # Warmup
        kp.put("warmup", id="_w")
        kp.delete("_w")

        errors = []

        def putter():
            try:
                for i in range(5):
                    kp.put(f"content {i}", id="contested-doc")
            except Exception as e:
                errors.append(e)

        def deleter():
            try:
                for _ in range(5):
                    kp.delete("contested-doc")
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=putter)
        t2 = threading.Thread(target=deleter)

        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Should not crash regardless of outcome
        assert not errors, f"Concurrent put/delete caused errors: {errors}"
        kp.close()

    def test_concurrent_process_pending(self, mock_providers, tmp_path):
        """Two process_pending calls shouldn't double-process items."""
        kp = Keeper(store_path=tmp_path)
        kp._needs_sysdoc_migration = False
        kp._pending_queue = PendingSummaryQueue(tmp_path / "pending.db")
        embed = mock_providers["embedding"]
        kp._embedding_provider = embed
        embed.embed_calls = 0

        # Enqueue 10 embed tasks
        for i in range(10):
            kp._pending_queue.enqueue(
                f"doc{i}", "default", f"content {i}", task_type="embed"
            )
            kp._document_store.upsert("default", f"doc{i}", f"content {i}", {})

        results = [None, None]

        def processor(idx):
            results[idx] = kp.process_pending(limit=10)

        t1 = threading.Thread(target=processor, args=(0,))
        t2 = threading.Thread(target=processor, args=(1,))

        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        # Total processed should be 10 (no double-processing)
        total = (results[0] or {}).get("processed", 0) + (results[1] or {}).get("processed", 0)
        assert total == 10, f"Expected 10 total, got {total}: {results}"

        kp.close()
