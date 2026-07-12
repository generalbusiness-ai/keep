"""Tests for find(tags=...) pre-filter parameter.

Uses mock providers — no ML models or network.
"""

import pytest

from keep.api import Keeper
from keep.types import Item


@pytest.fixture
def kp(mock_providers, tmp_path):
    """Create a Keeper with tagged seed data."""
    kp = Keeper(store_path=tmp_path)
    kp._get_embedding_provider()

    # Seed items with different tags
    kp.put("Alice likes cats and dogs", id="alice:pets", tags={"user": "alice", "topic": "pets"})
    kp.put("Alice works on project X", id="alice:work", tags={"user": "alice", "topic": "work"})
    kp.put("Bob likes birds", id="bob:pets", tags={"user": "bob", "topic": "pets"})
    kp.put("Bob works on project Y", id="bob:work", tags={"user": "bob", "topic": "work"})

    return kp


class TestFindTagsFilter:
    """Test find() with tags pre-filter."""

    def test_find_without_tags_returns_all(self, kp):
        """find() without tags returns results from all users."""
        results = kp.find("pets")
        ids = {r.id for r in results}
        assert "alice:pets" in ids
        assert "bob:pets" in ids

    def test_find_with_single_tag_filters(self, kp):
        """find() with tags={user: alice} only returns alice's items."""
        results = kp.find("pets", tags={"user": "alice"})
        ids = {r.id for r in results}
        assert "alice:pets" in ids
        assert "bob:pets" not in ids

    def test_find_with_multiple_tags(self, kp):
        """find() with multiple tags filters by all of them."""
        results = kp.find("work", tags={"user": "alice", "topic": "work"})
        ids = {r.id for r in results}
        assert "alice:work" in ids
        assert "bob:work" not in ids
        assert "alice:pets" not in ids

    def test_find_tags_key_case_insensitive_value_case_sensitive(self, kp):
        """Keys are case-insensitive; values remain case-sensitive."""
        results = kp.find("pets", tags={"User": "Alice"})
        ids = {r.id for r in results}
        assert "alice:pets" not in ids
        exact = kp.find("pets", tags={"User": "alice"})
        assert "alice:pets" in {r.id for r in exact}

    def test_find_tags_no_match(self, kp):
        """find() with non-matching tags returns empty."""
        results = kp.find("pets", tags={"user": "charlie"})
        assert len(results) == 0

    def test_put_same_key_replaces_value(self, kp):
        """Repeated writes with same key replace (not accumulate)."""
        kp.put("alice with two topics", id="alice:pets", tags={"topic": "animals"})
        item = kp.get("alice:pets")
        assert item is not None
        assert item.tags["topic"] == "animals"

    def test_put_multivalue_explicit(self, kp):
        """Explicit multi-value in a single put is preserved."""
        kp.put("alice topics", id="alice:pets", tags={"topic": ["animals", "pets"]})
        item = kp.get("alice:pets")
        assert item is not None
        assert set(item.tags["topic"]) == {"animals", "pets"}

    def test_find_matches_replaced_value(self, kp):
        """Tag filters match the replaced value, not the old one."""
        kp.put("alice with two topics", id="alice:pets", tags={"topic": "animals"})
        by_animals = {r.id for r in kp.find("alice", tags={"topic": "animals"})}
        assert "alice:pets" in by_animals

    def test_find_hybrid_with_tags(self, kp):
        """find() hybrid search also respects tags filter."""
        results = kp.find("cats", tags={"user": "alice"})
        ids = {r.id for r in results}
        assert "alice:pets" in ids
        assert "bob:pets" not in ids

    def test_find_tags_none_is_noop(self, kp):
        """find() with tags=None is same as no tags."""
        results_none = kp.find("pets", tags=None)
        results_default = kp.find("pets")
        assert {r.id for r in results_none} == {r.id for r in results_default}

    def test_ids_matching_tags_paginates_for_multi_tag_conjunction(
        self, mock_providers, tmp_path,
    ):
        """Multi-tag conjunctions must scan past the first LIMIT rows.

        Regression for the reviewer's finding: the old implementation
        asked SQLite for the first ``limit`` rows matching the first
        tag pair, then post-filtered in Python. If the true multi-tag
        match sat outside that recency window, it disappeared.
        """
        kp = Keeper(store_path=tmp_path)

        # The rare match — older than the decoys below.
        kp.put(
            "rare match content",
            id="rare-match",
            tags={"project": "journal", "topic": "rare"},
        )

        # Decoys: more than the _FIND_PART_JOIN_LIMIT cap, all matching
        # only the first tag pair, with newer updated_at timestamps.
        decoy_count = kp._FIND_PART_JOIN_LIMIT + 50
        for i in range(decoy_count):
            kp.put(
                f"decoy {i}",
                id=f"decoy:{i}",
                tags={"project": "journal"},
            )

        doc_coll = kp._resolve_doc_collection()
        from keep.types import casefold_tags
        cf = casefold_tags({"project": "journal", "topic": "rare"})
        matching = kp._ids_matching_tags(
            doc_coll, cf, limit=kp._FIND_PART_JOIN_LIMIT,
        )

        # The rare match must surface despite being older than the
        # first LIMIT rows of the broad first-tag query.
        assert "rare-match" in matching

    def test_find_rejects_invalid_tag_key(self, kp):
        """find() rejects tag keys that fail the shared key validator."""
        with pytest.raises(ValueError, match="invalid characters"):
            kp.find("pets", tags={"bad!key": "alice"})

    def test_gather_context_flattens_multivalue_related_tags(self, kp):
        """Gathered summarize context must handle list-valued related tags."""
        kp.find = lambda similar_to, limit=20: [  # type: ignore[method-assign]
            Item(id="alice:pets", summary="seed", tags={"topic": "pets"}, score=1.0),
            Item(
                id="related:1",
                summary="related",
                tags={"topic": ["animals", "pets"], "project": "alpha"},
                score=0.9,
            ),
        ]

        context = kp._gather_context("alice:pets", {"topic": "pets"})

        assert context == "Related topics: alpha, animals, pets"


class TestFindSinceFilter:
    """Regression tests for find() with since= filter.

    find() with since= must not drop items whose ChromaDB metadata lacks
    _updated_date (version refs, legacy items).
    """

    def test_find_since_returns_recent_items(self, kp):
        """find(since='P1D') includes items stored today."""
        results = kp.find("cats and dogs", since="P1D")
        assert any(r.id == "alice:pets" for r in results)

    def test_find_since_includes_versioned_items(self, mock_providers, tmp_path):
        """find(since='P1D') includes items without _updated_date.

        Covers the case where only version embeddings are in ChromaDB
        (whose metadata lacks _updated_date).
        """
        kp = Keeper(store_path=tmp_path)
        kp._get_embedding_provider()
        kp.put("Alpha content about animals", id="test:ver")
        kp.put("Beta content about animals", id="test:ver")

        results = kp.find("animals", since="P1D")
        assert any(r.id == "test:ver" for r in results)

    def test_find_similar_to_with_since(self, mock_providers, tmp_path):
        """find(similar_to=..., since='P1D') returns recently-updated matches."""
        kp = Keeper(store_path=tmp_path)
        kp._get_embedding_provider()
        kp.put("Reference doc about weather", id="test:ref")
        kp.put("Similar doc about climate", id="test:sim")

        results = kp.find(similar_to="test:ref", since="P1D")
        assert any(r.id == "test:sim" for r in results)

    def test_find_similar_to_uses_canonical_anchor_when_index_missing(
        self, mock_providers, tmp_path,
    ):
        """Fresh content-addressed notes must not depend on immediate index visibility."""
        kp = Keeper(store_path=tmp_path)
        kp._get_embedding_provider()

        anchor = kp.put("Shared text for similar-to anchor")
        kp.put("Shared text for similar-to anchor", id="test:match")

        assert anchor.id.startswith("%")
        chroma_coll = kp._resolve_chroma_collection()
        kp._store.delete(chroma_coll, anchor.id)

        results = kp.find(similar_to=anchor.id, limit=5)
        assert any(r.id == "test:match" for r in results)

    def test_find_until_excludes_recent(self, mock_providers, tmp_path):
        """find(until=<yesterday>) excludes items stored today."""
        kp = Keeper(store_path=tmp_path)
        kp._get_embedding_provider()
        kp.put("Today doc about planets", id="test:today")

        # until=yesterday should exclude items created today
        from datetime import datetime, timedelta, timezone
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        results = kp.find("planets", until=yesterday)
        assert not any(r.id == "test:today" for r in results)

    def test_find_until_includes_older(self, mock_providers, tmp_path):
        """find(until=<tomorrow>) includes items stored today."""
        kp = Keeper(store_path=tmp_path)
        kp._get_embedding_provider()
        kp.put("Today doc about galaxies", id="test:gal")

        from datetime import datetime, timedelta, timezone
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        results = kp.find("galaxies", until=tomorrow)
        assert any(r.id == "test:gal" for r in results)

    def test_find_since_and_until_combined(self, mock_providers, tmp_path):
        """find(since=..., until=...) applies both bounds."""
        kp = Keeper(store_path=tmp_path)
        kp._get_embedding_provider()
        kp.put("Bounded doc about oceans", id="test:ocean")

        from datetime import datetime, timedelta, timezone
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        # Window that includes today
        results = kp.find("oceans", since=yesterday, until=tomorrow)
        assert any(r.id == "test:ocean" for r in results)
        # Window that excludes today (past)
        last_week = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        results = kp.find("oceans", since=last_week, until=two_days_ago)
        assert not any(r.id == "test:ocean" for r in results)

    def test_find_mixed_items_with_and_without_updated_date(self, mock_providers, tmp_path):
        """find(since=...) handles mixed _updated_date presence.

        Covers a mix of items where some have _updated_date in ChromaDB
        and some don't (e.g. head vs version).
        """
        kp = Keeper(store_path=tmp_path)
        kp._get_embedding_provider()
        # head item — ChromaDB will have _updated_date
        kp.put("Fresh doc about rivers", id="test:fresh")
        # versioned item — version entry in ChromaDB lacks _updated_date
        kp.put("First draft about rivers and streams", id="test:versioned")
        kp.put("Second draft about rivers and streams", id="test:versioned")

        results = kp.find("rivers", since="P1D")
        ids = {r.id for r in results}
        assert "test:fresh" in ids
        assert "test:versioned" in ids


class TestFindEnrichmentBatching:
    """Result enrichment must hydrate canonical tags with a batched read."""

    def test_find_enrichment_uses_get_many_not_per_item_get(self, kp, monkeypatch):
        """Regression: enrichment issued one SQLite get() per result."""
        ds = kp._document_store
        calls = {"get": 0, "get_many": 0}
        orig_get, orig_get_many = ds.get, ds.get_many
        # MockDocumentStore.get_many fans out to get() internally; count
        # only get() calls made directly (outside get_many).
        depth = {"get_many": 0}

        def counting_get(collection, id):
            if depth["get_many"] == 0:
                calls["get"] += 1
            return orig_get(collection, id)

        def counting_get_many(collection, ids):
            calls["get_many"] += 1
            depth["get_many"] += 1
            try:
                return orig_get_many(collection, ids)
            finally:
                depth["get_many"] -= 1

        monkeypatch.setattr(ds, "get", counting_get)
        monkeypatch.setattr(ds, "get_many", counting_get_many)

        results = kp.find("likes", limit=4)

        assert len(results) >= 2
        # Enrichment still hydrates the canonical tags from the doc store
        assert all("user" in item.tags for item in results)
        # ... via a batched read, not one lookup per result item
        assert calls["get_many"] >= 1
        assert calls["get"] == 0, (
            f"find() issued {calls['get']} per-item doc-store reads"
        )
