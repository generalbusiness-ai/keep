"""Search augmentation mixin for Keeper.

Extracts deep-follow, recency decay, and RRF fusion logic from the
main Keeper class into a composable mixin.  These methods are called
by ``find()`` and ``_rank_by_relevance()`` in the Keeper class.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from .types import (
    EvidenceUnit,
    ContextWindow,
    Item,
    iter_tag_pairs,
    parse_utc_timestamp,
)
from .utils import _record_to_item

if TYPE_CHECKING:
    from .protocol import DocumentStoreProtocol, VectorStoreProtocol

logger = logging.getLogger(__name__)


class SearchAugmentationMixin:
    """Mixin providing deep-follow, recency decay, and RRF fusion.

    Expects the host class to provide:
    - ``_decay_half_life_days: float``
    - ``_store: VectorStoreProtocol``
    - ``_document_store: DocumentStoreProtocol``
    - ``_build_tag_where(tags) -> dict | None``
    - ``_run_read_flow(name, params, *, query_embedding) -> result``
    """

    # ------------------------------------------------------------------
    # Recency decay
    # ------------------------------------------------------------------

    def _apply_recency_decay(self, items: list[Item]) -> list[Item]:
        """Apply ACT-R style recency decay to search results.

        Multiplies each item's similarity score by a decay factor based on
        time since last update. Uses exponential decay with configurable half-life.

        Formula: effective_score = similarity * 0.5^(days_elapsed / half_life)
        """
        if self._decay_half_life_days <= 0:
            return items  # Decay disabled

        now = datetime.now(timezone.utc)
        decayed_items = []

        for item in items:
            # Get last update time from tags
            updated_str = item.tags.get("_updated")
            if updated_str and item.score is not None:
                try:
                    updated = parse_utc_timestamp(updated_str)
                    days_elapsed = (now - updated).total_seconds() / 86400

                    # Exponential decay: 0.5^(days/half_life)
                    decay_factor = 0.5 ** (days_elapsed / self._decay_half_life_days)
                    decayed_score = item.score * decay_factor

                    # Create new Item with decayed score
                    decayed_items.append(Item(
                        id=item.id,
                        summary=item.summary,
                        tags=item.tags,
                        score=decayed_score
                    ))
                except (ValueError, TypeError):
                    # If timestamp parsing fails, keep original
                    decayed_items.append(item)
            else:
                decayed_items.append(item)

        # Re-sort by decayed score (highest first)
        decayed_items.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)

        return decayed_items

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        semantic_items: list[Item],
        fts_items: list[Item],
        k: int = 60,
        fts_weight: float = 2.0,
    ) -> list[Item]:
        """Fuse two ranked lists using weighted Reciprocal Rank Fusion.

        score(d) = 1/(k + rank_sem) + fts_weight/(k + rank_fts)

        FTS gets a higher weight because keyword matches signal entity-level
        relevance that semantic similarity can miss (e.g., "Max" the dog vs
        "How old is Luna?" which is semantically similar but wrong entity).

        Scores are normalized to [0, 1] where 1.0 = rank 1 in both lists.
        """
        scores: dict[str, float] = {}
        items_by_id: dict[str, Item] = {}

        for rank, item in enumerate(semantic_items, start=1):
            scores[item.id] = scores.get(item.id, 0) + 1 / (k + rank)
            items_by_id[item.id] = item  # prefer semantic item (has tags)

        for rank, item in enumerate(fts_items, start=1):
            scores[item.id] = scores.get(item.id, 0) + fts_weight / (k + rank)
            if item.id not in items_by_id:
                items_by_id[item.id] = item

        # Theoretical max: rank 1 in both lists
        max_score = (1.0 + fts_weight) / (k + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        for item_id, rrf_score in ranked:
            source = items_by_id[item_id]
            result.append(Item(
                id=source.id,
                summary=source.summary,
                tags=source.tags,
                score=round(rrf_score / max_score, 4),
            ))
        return result

    # ------------------------------------------------------------------
    # Deep tag follow (Tier 2 fallback for items without edges)
    # ------------------------------------------------------------------

    def _deep_tag_follow(self, primary_items, chroma_coll, doc_coll, *,
                         embedding=None, top_k=10,
                         per_tag_fetch=1000, max_per_group=5):
        """Follow tags from primary results to discover bridge documents.

        Per-tag queries use ``query_embedding`` when an embedding is provided,
        giving each candidate a semantic similarity score that serves as a
        tiebreaker within the same tag-overlap tier.  Falls back to
        ``query_metadata`` (no semantic ranking) when no embedding is given.

        Versions and parts are collapsed to their parent document during
        collection so a single popular document doesn't consume all slots.

        Args:
            primary_items: Items from the primary search results.
            chroma_coll: Chroma collection name.
            doc_coll: Document store collection name.
            embedding: Query embedding for semantic tiebreaking (optional).
            top_k: Number of top primary items to collect tags from.
            per_tag_fetch: Max raw items to fetch per tag query.
            max_per_group: Max deep items to show per primary.

        Returns:
            dict mapping primary item ID to list of deep-discovered Items,
            sorted by (tag-overlap, semantic similarity) within each group.
        """
        # 1. Collect non-system tag pairs, tracking which primary has each
        tag_to_sources: dict[tuple[str, str], set[str]] = {}
        for item in primary_items[:top_k]:
            for k, v in iter_tag_pairs(item.tags, include_system=False):
                tag_to_sources.setdefault((k, v), set()).add(item.id)
        if not tag_to_sources:
            return {}
        # Drop tag pairs shared by ALL top primaries (non-distinctive)
        n_primaries = len(primary_items[:top_k])
        if n_primaries > 1:
            tag_to_sources = {
                tp: sids for tp, sids in tag_to_sources.items()
                if len(sids) < n_primaries
            }
        if not tag_to_sources:
            return {}

        # 1b. Compute IDF weights for tag-overlap scoring
        total_docs = max(self._document_store.count(doc_coll), 1)
        pair_counts = self._document_store.tag_pair_counts(doc_coll)
        idf: dict[tuple[str, str], float] = {}
        for (k, v), df in pair_counts.items():
            idf[(k.casefold(), v)] = math.log(total_docs / df)

        # 2. Metadata-only queries — find items sharing each tag,
        #    collapsing versions/parts to parent IDs immediately.
        #    Only exclude items from the top_k (not the full over-fetched pool)
        #    so genuine bridge items aren't accidentally excluded.
        top_ids = {item.id for item in primary_items[:top_k]}
        top_parents = set()
        for pid in top_ids:
            if "@" in pid:
                top_parents.add(pid.split("@")[0])
            else:
                top_parents.add(pid)

        candidates: dict[str, Item] = {}        # parent_id -> Item
        candidate_tags: dict[str, set] = {}     # parent_id -> matched (k, v)
        candidate_sources: dict[str, set] = {}  # parent_id -> primary IDs
        candidate_sem: dict[str, float] = {}    # parent_id -> best semantic score

        for (k, v), source_ids in tag_to_sources.items():
            where = self._build_tag_where({k: v})
            if where is None:
                continue
            if embedding is not None:
                results = self._store.query_embedding(
                    chroma_coll, embedding, limit=per_tag_fetch, where=where,
                )
            else:
                results = self._store.query_metadata(
                    chroma_coll, where=where, limit=per_tag_fetch,
                )
            seen_parents_this_tag: set[str] = set()
            for r in results:
                # Collapse to parent ID
                raw_id = r.id
                is_child = "@" in raw_id
                parent_id = raw_id.split("@")[0] if is_child else raw_id
                # Skip primaries (both raw and uplifted)
                if raw_id in top_ids or parent_id in top_parents:
                    continue
                # Deduplicate: only count each parent once per tag query
                if parent_id in seen_parents_this_tag:
                    continue
                seen_parents_this_tag.add(parent_id)
                # Track best semantic score for this parent
                item_from_r = r.to_item()
                if item_from_r.score is not None:
                    prev = candidate_sem.get(parent_id, 0)
                    candidate_sem[parent_id] = max(prev, item_from_r.score)
                # Register parent as candidate with head-doc tags
                if parent_id not in candidates:
                    head_doc = self._document_store.get(doc_coll, parent_id)
                    if head_doc:
                        head_item = _record_to_item(head_doc)
                        candidates[parent_id] = Item(
                            id=parent_id, summary=head_item.summary,
                            tags=head_item.tags, score=0,
                        )
                    else:
                        candidates[parent_id] = Item(
                            id=parent_id, summary=r.summary, tags=r.tags, score=0,
                        )
                candidate_tags.setdefault(parent_id, set()).add((k, v))
                candidate_sources.setdefault(parent_id, set()).update(source_ids)

        # 3. Assign each candidate to ONE primary (most shared HEAD tags)
        #    Score uses the candidate's head-doc tags (not version tags)
        primary_order = {item.id: i for i, item in enumerate(primary_items[:top_k])}
        # Pre-compute tag set per primary, limited to followed tag pairs
        followed_keys = set(tag_to_sources.keys())
        primary_tag_sets: dict[str, set] = {}
        for item in primary_items[:top_k]:
            primary_tag_sets[item.id] = {
                (k, v) for k, v in iter_tag_pairs(item.tags, include_system=False)
                if (k, v) in followed_keys
            }

        groups: dict[str, list[Item]] = {}
        for cid, item in candidates.items():
            # Use the candidate's HEAD doc tags for scoring (casefolded
            # to match ChromaDB's casefolded primary_tag_sets)
            head_tags = {
                (k.casefold(), v) for k, v in iter_tag_pairs(item.tags, include_system=False)
            }
            sources = candidate_sources.get(cid, set())

            def _idf_overlap(tag_set):
                return sum(idf.get(tp, 0) for tp in head_tags & tag_set)

            # Pick primary with highest IDF-weighted tag overlap
            best_source = min(
                sources,
                key=lambda sid: (
                    -_idf_overlap(primary_tag_sets.get(sid, set())),
                    primary_order.get(sid, 999),
                ),
            )
            # Composite score: IDF-weighted tag overlap + semantic tiebreaker
            overlap = _idf_overlap(primary_tag_sets.get(best_source, set()))
            sem = candidate_sem.get(cid, 0)
            scored = Item(id=cid, summary=item.summary,
                          tags=item.tags, score=overlap + sem)
            groups.setdefault(best_source, []).append(scored)

        # 4. Sort each group by composite score desc, cap size
        for source_id in groups:
            groups[source_id].sort(key=lambda x: x.score or 0, reverse=True)
            groups[source_id] = groups[source_id][:max_per_group]

        return groups

    # ------------------------------------------------------------------
    # Deep edge follow
    # ------------------------------------------------------------------

    def _deep_edge_follow(
        self,
        primary_items: list[Item],
        chroma_coll: str,
        doc_coll: str,
        *,
        query: str,
        embedding: list[float],
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> dict[str, list[Item]]:
        """Follow inverse edges from primary results to discover related items.

        For each primary, traverses its inverse edges (e.g. speaker->said)
        to collect candidate source IDs.  Then runs a scoped hybrid search
        (FTS pre-filter + embedding post-filter + RRF fusion) over only
        those candidates to surface relevant evidence.

        Returns all candidates per group — the renderer caps output via
        token budget.

        Args:
            primary_items: Items from the primary search results.
            chroma_coll: Chroma collection name.
            doc_coll: Document store collection name.
            query: Original search query text (used for FTS pre-filter).
            embedding: Query embedding vector.
            top_k: Number of top primary items to follow edges from.
            exclude_ids: IDs to exclude from deep results (e.g. items
                the user will already see as primaries).

        Returns:
            dict mapping primary item ID to list of deep-discovered Items,
            sorted by RRF score within each group.
        """
        query_stopwords: frozenset[str] = frozenset()
        get_stopwords = getattr(self._document_store, "get_stopwords", None)
        if callable(get_stopwords):
            try:
                query_stopwords = get_stopwords()
            except Exception:
                query_stopwords = frozenset()

        def _tokenize(text: str) -> set[str]:
            return {
                tok for tok in re.findall(r"[a-z0-9]+", (text or "").lower())
                if len(tok) > 2 and tok not in query_stopwords
            }

        def _extract_focus(id_value: str) -> tuple[str, Optional[str], Optional[str]]:
            """Return (parent_id, focus_part, focus_version) from a doc/part/version ID."""
            if "@p" in id_value:
                parent, suffix = id_value.rsplit("@p", 1)
                if suffix.isdigit():
                    return parent, suffix, None
            if "@v" in id_value:
                parent, suffix = id_value.rsplit("@v", 1)
                if suffix.isdigit():
                    return parent, None, suffix
            return id_value, None, None

        query_terms = _tokenize(query)

        def _query_overlap(text: str) -> float:
            if not query_terms:
                return 0.0
            # Normalize to [0, 1] so lexical signal doesn't swamp semantic score.
            return len(query_terms & _tokenize(text)) / max(len(query_terms), 1)

        # 1. Traverse edges for each primary, collect candidate IDs.
        #    Two traversal paths:
        #    a) Inverse edges: primary is a target -> collect sources
        #       (e.g. entity "Melanie" <- said <- session docs)
        #    b) Forward + inverse (two-hop): primary is a source ->
        #       follow forward to targets -> collect THEIR inverse sources
        #       (e.g. session -> speaker -> entity -> said -> other sessions)
        #
        #    Caps prevent runaway fan-out on high-degree entities:
        #    - max_forward: forward edges per primary (hop-1 fan-out)
        #    - max_candidates: total candidate pool across all primaries
        _MAX_FORWARD = 20       # forward edges to traverse per primary
        _MAX_CANDIDATES = 500   # total candidate IDs before FTS/embedding

        primary_to_sources: dict[str, set[str]] = {}
        all_source_ids: set[str] = set()
        get_inverse_version_edges = getattr(
            self._document_store, "get_inverse_version_edges", None,
        )
        has_archived_versions = False
        if callable(get_inverse_version_edges):
            count_versions = getattr(self._document_store, "count_versions", None)
            if callable(count_versions):
                has_archived_versions = count_versions(doc_coll) > 0
            else:
                # Compatibility path for test doubles or alternate stores.
                has_archived_versions = True
        # IDs to exclude from deep results: caller-provided exclusion set
        # (items the user will see as primaries) or fall back to all items
        top_ids = set(exclude_ids) if exclude_ids is not None else set()

        for item in primary_items[:top_k]:
            parent_id = item.id.split("@")[0] if "@" in item.id else item.id
            if exclude_ids is None:
                top_ids.add(parent_id)
            sources: set[str] = set()

            # Path a: inverse edges (primary is target)
            inv_edges = self._document_store.get_inverse_edges(
                doc_coll, parent_id)
            for _inv, src_id, _created in inv_edges:
                sources.add(src_id)

            # Version-note parity: include sources whose archived versions
            # had edge tags pointing at this primary target.
            if has_archived_versions:
                ver_inv_edges = get_inverse_version_edges(
                    doc_coll, parent_id, limit=_MAX_CANDIDATES,
                )
                for _inv, src_id, _created in ver_inv_edges:
                    sources.add(src_id)

            # Path b: two-hop via forward edges (primary is source)
            fwd_edges = self._document_store.get_forward_edges(
                doc_coll, parent_id)
            for _pred, target_id, _created in fwd_edges[:_MAX_FORWARD]:
                hop2 = self._document_store.get_inverse_edges(
                    doc_coll, target_id)
                for _inv2, src_id2, _created2 in hop2:
                    sources.add(src_id2)

            if sources:
                primary_to_sources[parent_id] = sources
                all_source_ids.update(sources)
                if len(all_source_ids) >= _MAX_CANDIDATES:
                    break

        if not all_source_ids:
            return {}

        source_list = list(all_source_ids)

        # 2. FTS pre-filter: narrow candidates using cheap text matching
        fts_fetch = max(len(source_list), 100)
        fts_rows = self._document_store.query_fts_scoped(
            doc_coll, query, source_list, limit=fts_fetch,
        )
        fts_items = [Item(id=r[0], summary=r[1]) for r in fts_rows]

        # 3. Embedding search — post-filter to ALL edge source IDs.
        #    Unlike FTS (which is cheap enough to scope), embedding
        #    search queries the full collection and post-filters.
        sem_fetch = max(len(all_source_ids) * 3, 200)
        sem_results = self._store.query_embedding(
            chroma_coll, embedding, limit=sem_fetch,
        )
        sem_items = []
        for r in sem_results:
            base = r.id.split("@")[0] if "@" in r.id else r.id
            if base in all_source_ids:
                sem_items.append(r.to_item())

        sem_items = self._apply_recency_decay(sem_items)

        logger.debug("Deep edge follow: sources=%d fts=%d sem=%d top=%s",
                     len(all_source_ids), len(fts_items), len(sem_items),
                     top_ids)

        # 4. RRF fuse scoped FTS + embedding results
        if fts_items and sem_items:
            fused = self._rrf_fuse(sem_items, fts_items)
        elif fts_items:
            fused = fts_items
        elif sem_items:
            fused = sem_items
        else:
            return {}

        # 5. Candidate generation: map fused hits into evidence units.
        #    Keep this broad/high-recall; reranking happens in stage 6.
        candidate_units: list[tuple[str, EvidenceUnit, Item]] = []

        for item in fused:
            parent_id, focus_part, focus_version = _extract_focus(item.id)
            has_focus = focus_part is not None or focus_version is not None

            # Find which primary has an edge to this source.
            # Prefer non-excluded primaries (injected entities) so that
            # edge sources get grouped under the entity rather than a
            # sibling session that happens to share the same edges.
            best_primary = None
            for pid, sources in primary_to_sources.items():
                if parent_id in sources:
                    if pid not in top_ids:
                        best_primary = pid
                        break  # entity match — best possible
                    elif best_primary is None:
                        best_primary = pid  # fallback to excluded primary

            # Skip items that are already visible as primaries, UNLESS
            # they belong to an entity group (entity not in top_ids).
            if parent_id in top_ids and (best_primary is None or best_primary in top_ids):
                continue

            if best_primary is None:
                continue

            # Enrich with head-doc summary/tags from SQLite
            head_doc = self._document_store.get(doc_coll, parent_id)
            if head_doc:
                head_item = _record_to_item(head_doc)
                tags = dict(head_item.tags)
                # Preserve matched evidence text for renderer/LLM context.
                if item.summary and (has_focus or item.summary != head_item.summary):
                    tags["_focus_summary"] = item.summary
                if focus_part:
                    tags["_focus_part"] = focus_part
                if focus_version:
                    tags["_focus_version"] = focus_version
                tags["_anchor_id"] = item.id
                if focus_part:
                    tags["_anchor_type"] = "part"
                elif focus_version:
                    tags["_anchor_type"] = "version"
                else:
                    tags["_anchor_type"] = "document"
                scored = Item(
                    id=item.id, summary=head_item.summary,
                    tags=tags, score=item.score,
                )
            else:
                tags = dict(item.tags)
                if item.summary and has_focus:
                    tags["_focus_summary"] = item.summary
                if focus_part:
                    tags["_focus_part"] = focus_part
                if focus_version:
                    tags["_focus_version"] = focus_version
                tags["_anchor_id"] = item.id
                if focus_part:
                    tags["_anchor_type"] = "part"
                elif focus_version:
                    tags["_anchor_type"] = "version"
                else:
                    tags["_anchor_type"] = "document"
                scored = Item(
                    id=item.id, summary=item.summary,
                    tags=tags, score=item.score,
                )

            evidence_text = scored.tags.get("_focus_summary", scored.summary)
            lane = "derived" if focus_part is not None else "authoritative"
            scored.tags["_lane"] = lane

            unit = EvidenceUnit(
                unit_id=item.id,
                source_id=parent_id,
                version=int(focus_version) if focus_version else None,
                part_num=int(focus_part) if focus_part else None,
                lane=lane,
                text=evidence_text,
                parent_summary=scored.summary,
                created=scored.tags.get("_created"),
                score_sem=item.score or 0.0,
                score_lex=float(_query_overlap(evidence_text)),
                score_focus=1.0 if has_focus else 0.0,
                score_coherence=0.2 if scored.tags.get("_focus_summary") else 0.0,
                provenance={
                    "anchor_id": item.id,
                    "anchor_type": scored.tags.get("_anchor_type", "document"),
                },
            )

            candidate_units.append((best_primary, unit, scored))

        if not candidate_units:
            return {}

        # 6. Query-conditioned rerank and parent-level dedup.
        #    Weights are generic (dataset-agnostic): semantic + lexical
        #    relevance + focus quality + coherence signal.
        _W_SEM = 1.0
        _W_LEX = 0.6
        _W_FOCUS = 0.4
        _W_COH = 0.2
        _MAX_ANCHORS_PER_SOURCE = 2
        by_source: dict[str, list[tuple[float, str, ContextWindow, Item]]] = {}

        for primary_id, unit, scored in candidate_units:
            total = (
                _W_SEM * unit.score_sem
                + _W_LEX * unit.score_lex
                + _W_FOCUS * unit.score_focus
                + _W_COH * unit.score_coherence
            )
            reranked = EvidenceUnit(
                unit_id=unit.unit_id,
                source_id=unit.source_id,
                version=unit.version,
                part_num=unit.part_num,
                lane=unit.lane,
                text=unit.text,
                parent_summary=unit.parent_summary,
                created=unit.created,
                score_sem=unit.score_sem,
                score_lex=unit.score_lex,
                score_focus=unit.score_focus,
                score_coherence=unit.score_coherence,
                score_total=total,
                provenance=unit.provenance,
            )
            window = ContextWindow(
                anchor=reranked,
                members=[reranked],  # first slice: anchor-only windows
                score_total=total,
                tokens_est=max(len(reranked.text) // 4, 1),
            )
            by_source.setdefault(reranked.source_id, []).append(
                (window.score_total, primary_id, window, scored)
            )

        groups: dict[str, list[Item]] = {}
        for entries in by_source.values():
            entries.sort(key=lambda t: t[0], reverse=True)
            selected: list[tuple[float, str, ContextWindow, Item]] = []
            seen_focus: set[tuple[Optional[int], Optional[int], str]] = set()
            for score_total, primary_id, window, scored in entries:
                focus_key = (
                    window.anchor.version,
                    window.anchor.part_num,
                    (scored.tags.get("_focus_summary") or "").strip().lower(),
                )
                if focus_key in seen_focus:
                    continue
                seen_focus.add(focus_key)
                selected.append((score_total, primary_id, window, scored))
                if len(selected) >= _MAX_ANCHORS_PER_SOURCE:
                    break
            for score_total, primary_id, _window, scored in selected:
                scored.tags["_window_score"] = f"{score_total:.4f}"
                out_score = scored.score if scored.score is not None else score_total
                scored_item = Item(
                    id=scored.id,
                    summary=scored.summary,
                    tags=scored.tags,
                    score=out_score,
                )
                groups.setdefault(primary_id, []).append(scored_item)

        for primary_id, group in groups.items():
            group.sort(
                key=lambda x: (
                    float(x.tags.get("_window_score", "0") or 0),
                    x.score or 0.0,
                ),
                reverse=True,
            )
            # Hide internal rerank helper tag from downstream output.
            for gi in group:
                if "_window_score" in gi.tags:
                    del gi.tags["_window_score"]
            groups[primary_id] = group

        return groups

    # ------------------------------------------------------------------
    # Deep follow via state-doc flow (fallback)
    # ------------------------------------------------------------------

    def _deep_follow_via_flow(
        self,
        *,
        query: str,
        limit: int = 10,
        deep_limit: int = 5,
        deep_follow_depth: int = 10,
        embedding: Any = None,
    ) -> dict[str, list[Item]]:
        """Run the find-deep state-doc flow for deep follow.

        Used as the tag-follow fallback when the store has no edges.
        The find-deep flow runs a search then traverses the results,
        exercising the CEL predicate ``search.count == 0``.

        Args:
            query: Search query string.
            limit: Max primary results to fetch.
            deep_limit: Max deep items per primary group.
            deep_follow_depth: How many primary items contribute tags to the
                Tier-2 tag-follow step.  Defaults to 10; clamped by
                ``traverse_related``.
            embedding: Query embedding for semantic tiebreaking.
        """
        if not query:
            return {}

        result = self._run_read_flow(
            "find-deep",
            {
                "query": query,
                "limit": limit,
                "deep_limit": deep_limit,
                "deep_follow_depth": deep_follow_depth,
            },
            query_embedding=embedding,
        )
        if result.status != "done":
            return {}

        related_binding = result.bindings.get("related", {})
        groups_raw = related_binding.get("groups") or {}

        # Map action output dicts back to Item objects
        deep_groups: dict[str, list[Item]] = {}
        for source_id, items_raw in groups_raw.items():
            if not isinstance(items_raw, list):
                continue
            group = []
            for r in items_raw:
                if not isinstance(r, dict):
                    continue
                group.append(Item(
                    id=r.get("id", ""),
                    summary=r.get("summary", ""),
                    tags=r.get("tags") or {},
                    score=r.get("score"),
                ))
            if group:
                deep_groups[source_id] = group
        return deep_groups
