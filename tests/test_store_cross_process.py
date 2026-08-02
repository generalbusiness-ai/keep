"""Tests for ChromaDB cross-process safety (write lock + epoch sentinel).

These tests verify that ChromaStore correctly serializes writes via file
locks and detects stale in-memory indexes via the epoch sentinel file.
"""

import tempfile
import time
from pathlib import Path

import pytest

# Skip all tests if chromadb not installed
chromadb = pytest.importorskip("chromadb")

from keep.store import ChromaStore, StoreResult


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def store_path():
    """Provide a temporary directory for store tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def store(store_path):
    """Create a ChromaStore instance for testing."""
    instance = ChromaStore(store_path, embedding_dimension=4)
    yield instance
    instance.close()


@pytest.fixture
def sample_embedding():
    """A simple 4-dimensional embedding for testing."""
    return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def alt_embedding():
    """A different 4-dimensional embedding."""
    return [0.5, 0.6, 0.7, 0.8]


# -----------------------------------------------------------------------------
# Epoch sentinel basics
# -----------------------------------------------------------------------------

class TestEpochSentinel:
    """Tests for the epoch sentinel file mechanics."""

    def test_epoch_starts_at_zero(self, store_path):
        """New store has no epoch file and _last_epoch == 0."""
        with ChromaStore(store_path, embedding_dimension=4) as store:
            assert store._last_epoch == 0.0
            assert not (store_path / ".chroma.epoch").exists()

    def test_write_creates_epoch_file(self, store, store_path, sample_embedding):
        """First write creates the .chroma.epoch sentinel file."""
        store.upsert("test", "doc:1", sample_embedding, "Test", {})
        assert (store_path / ".chroma.epoch").exists()
        assert store._last_epoch > 0

    def test_write_bumps_epoch(self, store, sample_embedding):
        """Each write updates the epoch."""
        store.upsert("test", "doc:1", sample_embedding, "First", {})
        epoch1 = store._last_epoch

        # Ensure mtime differs (some filesystems have coarse resolution)
        time.sleep(0.05)
        store.upsert("test", "doc:2", sample_embedding, "Second", {})
        epoch2 = store._last_epoch

        assert epoch2 >= epoch1

    def test_delete_bumps_epoch(self, store, sample_embedding):
        """Delete operations bump the epoch."""
        store.upsert("test", "doc:1", sample_embedding, "Test", {})
        epoch_after_write = store._last_epoch

        time.sleep(0.05)
        store.delete("test", "doc:1")
        assert store._last_epoch >= epoch_after_write

    def test_all_write_methods_bump_epoch(self, store, store_path, sample_embedding):
        """Every write method creates/updates the epoch file."""
        epoch_path = store_path / ".chroma.epoch"

        # upsert
        store.upsert("test", "doc:1", sample_embedding, "Test", {})
        assert epoch_path.exists()
        e1 = store._last_epoch

        # upsert_version
        time.sleep(0.05)
        store.upsert_version("test", "doc:1", 1, sample_embedding, "V1", {})
        assert store._last_epoch >= e1

        # upsert_part
        time.sleep(0.05)
        store.upsert_part("test", "doc:1", 1, sample_embedding, "P1", {})
        e2 = store._last_epoch

        # update_summary
        time.sleep(0.05)
        store.update_summary("test", "doc:1", "Updated summary")
        assert store._last_epoch >= e2

        # update_tags
        time.sleep(0.05)
        store.update_tags("test", "doc:1", {"tag": "value"})
        e3 = store._last_epoch

        # delete_parts
        time.sleep(0.05)
        store.delete_parts("test", "doc:1")
        assert store._last_epoch >= e3

        # upsert_batch
        time.sleep(0.05)
        store.upsert_batch("test", ["doc:2"], [sample_embedding], ["Batch"], [{"_t": "1"}])
        e4 = store._last_epoch

        # delete_entries
        time.sleep(0.05)
        store.delete_entries("test", ["doc:2"])
        assert store._last_epoch >= e4

        # delete_collection
        time.sleep(0.05)
        store.delete_collection("test")
        assert store._last_epoch >= e4


# -----------------------------------------------------------------------------
# Freshness / reload
# -----------------------------------------------------------------------------

class TestFreshnessReload:
    """Tests for the freshness check and client reload."""

    def test_read_reloads_on_epoch_change(self, store, store_path, sample_embedding):
        """Externally touching the sentinel triggers a client reload on read."""
        store.upsert("test", "doc:1", sample_embedding, "Test", {})
        old_client = store._client

        # Simulate another process bumping the epoch
        time.sleep(0.05)
        (store_path / ".chroma.epoch").touch()

        # Read should detect the change and reload
        store.get("test", "doc:1")
        assert store._client is not old_client

    def test_no_reload_when_epoch_unchanged(self, store, sample_embedding):
        """Client stays the same when no external writes happened."""
        store.upsert("test", "doc:1", sample_embedding, "Test", {})
        old_client = store._client

        # Multiple reads without external epoch changes
        store.get("test", "doc:1")
        store.exists("test", "doc:1")

        assert store._client is old_client

    def test_own_write_does_not_trigger_reload(self, store, sample_embedding):
        """Our own writes update _last_epoch, so no unnecessary reload."""
        store.upsert("test", "doc:1", sample_embedding, "First", {})
        old_client = store._client

        store.upsert("test", "doc:2", sample_embedding, "Second", {})
        # Should still be the same client (we updated _last_epoch in sync)
        assert store._client is old_client

    def test_freshness_check_on_closed_store(self, store, sample_embedding):
        """Freshness check is a no-op on a closed store."""
        store.upsert("test", "doc:1", sample_embedding, "Test", {})
        store.close()
        # Should not raise
        store._check_freshness()


# -----------------------------------------------------------------------------
# Two-instance visibility (simulates cross-process)
# -----------------------------------------------------------------------------

class TestCrossInstanceVisibility:
    """Tests using two ChromaStore instances on the same directory."""

    def test_write_visible_to_other_instance(self, store_path, sample_embedding):
        """Store A writes, store B sees the data."""
        a = ChromaStore(store_path, embedding_dimension=4)
        b = ChromaStore(store_path, embedding_dimension=4)

        a.upsert("test", "doc:1", sample_embedding, "From A", {"source": "a"})

        # B should see it after freshness check
        result = b.get("test", "doc:1")
        assert result is not None
        assert result.summary == "From A"

        a.close()
        b.close()

    def test_delete_visible_to_other_instance(self, store_path, sample_embedding):
        """Store A writes then deletes, store B sees the deletion."""
        a = ChromaStore(store_path, embedding_dimension=4)
        a.upsert("test", "doc:1", sample_embedding, "To delete", {})

        b = ChromaStore(store_path, embedding_dimension=4)
        # B can see the item
        assert b.get("test", "doc:1") is not None

        a.delete("test", "doc:1")

        # B should see the deletion
        assert b.get("test", "doc:1") is None

        a.close()
        b.close()

    def test_similarity_search_after_cross_instance_write(
        self, store_path, sample_embedding, alt_embedding,
    ):
        """Similarity search finds items written by another instance."""
        a = ChromaStore(store_path, embedding_dimension=4)
        b = ChromaStore(store_path, embedding_dimension=4)

        a.upsert("test", "doc:1", sample_embedding, "Searchable", {"topic": "test"})

        # B should find it via embedding search
        results = b.query_embedding("test", sample_embedding, limit=5)
        assert len(results) >= 1
        assert results[0].id == "doc:1"

        a.close()
        b.close()

    def test_interleaved_writes_from_both_instances(
        self, store_path, sample_embedding, alt_embedding,
    ):
        """Interleaved writes from two instances are all visible."""
        a = ChromaStore(store_path, embedding_dimension=4)
        b = ChromaStore(store_path, embedding_dimension=4)

        a.upsert("test", "doc:1", sample_embedding, "From A", {})
        b.upsert("test", "doc:2", alt_embedding, "From B", {})
        a.upsert("test", "doc:3", sample_embedding, "Also from A", {})

        # Both instances should see all three
        assert a.count("test") == 3
        assert b.count("test") == 3

        a.close()
        b.close()

    def test_closing_unrelated_store_does_not_evict_live_system(
        self, store_path, sample_embedding,
    ):
        """A stale store destructor must not clear another path's System."""
        first = ChromaStore(store_path / "first", embedding_dimension=4)
        second = ChromaStore(store_path / "second", embedding_dimension=4)
        second.upsert("test", "doc:1", sample_embedding, "Still live", {})

        first.close()

        assert second.get("test", "doc:1") is not None
        second.upsert("test", "doc:2", sample_embedding, "Still writable", {})
        assert second.count("test") == 2
        second.close()

    def test_closing_shared_store_keeps_other_lease_live(
        self, store_path, sample_embedding,
    ):
        """Closing one same-path store must not stop its peer's System."""
        first = ChromaStore(store_path, embedding_dimension=4)
        second = ChromaStore(store_path, embedding_dimension=4)
        shared_system = first._client_system
        assert second._client_system is shared_system

        first.close()

        second.upsert("test", "doc:1", sample_embedding, "Still writable", {})
        assert second.get("test", "doc:1") is not None
        assert ChromaStore._system_leases[id(shared_system)][1] == 1
        second.close()
        assert id(shared_system) not in ChromaStore._system_leases


# -----------------------------------------------------------------------------
# Lock file
# -----------------------------------------------------------------------------

class TestWriteLock:
    """Tests for the write lock file."""

    def test_lock_file_created_on_write(self, store, store_path, sample_embedding):
        """The .chroma.lock file is created on first write."""
        store.upsert("test", "doc:1", sample_embedding, "Test", {})
        assert (store_path / ".chroma.lock").exists()

    def test_close_with_lock(self, store, sample_embedding):
        """Write then close — no error, lock file persists (expected)."""
        store.upsert("test", "doc:1", sample_embedding, "Test", {})
        store.close()  # Should not raise
