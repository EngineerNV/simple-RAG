from __future__ import annotations

import pytest

from utils.semantic_cache import (
    SemanticCache,
    cached_retrieve_and_rerank,
    cosine_similarity,
)


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_defined() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_exact_text_hit_avoids_semantic_scan() -> None:
    cache = SemanticCache(max_size=5, similarity_threshold=0.93)
    cache.put("what is a chunk?", [1.0, 0.0, 0.0], k=3, value="chunk answer")

    result = cache.get("what is a chunk?", [1.0, 0.0, 0.0], k=3)

    assert result == "chunk answer"
    assert cache.stats.hits_exact == 1
    assert cache.stats.hits_semantic == 0
    assert cache.stats.misses == 0


def test_exact_hit_ignores_case_and_whitespace() -> None:
    cache = SemanticCache()
    cache.put("What is  RAG?", [1.0, 0.0], k=3, value="rag answer")

    assert cache.get("  what is rag? ", [1.0, 0.0], k=3) == "rag answer"
    assert cache.stats.hits_exact == 1


def test_semantic_hit_for_paraphrase_above_threshold() -> None:
    cache = SemanticCache(max_size=5, similarity_threshold=0.93)
    cache.put("when did the band debut?", [1.0, 0.02, 0.0], k=3, value="debut answer")

    # A different phrasing, near-identical embedding -> should still hit.
    result = cache.get("what year did the band first debut?", [0.999, 0.03, 0.0], k=3)

    assert result == "debut answer"
    assert cache.stats.hits_semantic == 1
    assert cache.stats.hits_exact == 0


def test_dissimilar_query_is_a_miss() -> None:
    cache = SemanticCache(max_size=5, similarity_threshold=0.93)
    cache.put("when did the band debut?", [1.0, 0.0, 0.0], k=3, value="debut answer")

    result = cache.get("what is Pikachu's typing?", [0.0, 1.0, 0.0], k=3)

    assert result is None
    assert cache.stats.misses == 1


def test_matching_query_different_k_is_a_miss() -> None:
    cache = SemanticCache(max_size=5, similarity_threshold=0.93)
    cache.put("what is a chunk?", [1.0, 0.0], k=3, value="k3 answer")

    result = cache.get("what is a chunk?", [1.0, 0.0], k=5)

    assert result is None
    assert cache.stats.misses == 1


def test_lru_eviction_drops_least_recently_used() -> None:
    cache = SemanticCache(max_size=2, similarity_threshold=0.93)
    cache.put("query one", [1.0, 0.0, 0.0], k=3, value="one")
    cache.put("query two", [0.0, 1.0, 0.0], k=3, value="two")

    # Touch "query one" so "query two" becomes the least-recently-used entry.
    assert cache.get("query one", [1.0, 0.0, 0.0], k=3) == "one"

    cache.put("query three", [0.0, 0.0, 1.0], k=3, value="three")

    assert len(cache) == 2
    assert cache.stats.evictions == 1
    assert cache.get("query two", [0.0, 1.0, 0.0], k=3) is None
    assert cache.get("query one", [1.0, 0.0, 0.0], k=3) == "one"
    assert cache.get("query three", [0.0, 0.0, 1.0], k=3) == "three"


def test_clear_resets_entries_and_stats() -> None:
    cache = SemanticCache()
    cache.put("query", [1.0, 0.0], k=3, value="value")
    cache.get("query", [1.0, 0.0], k=3)

    cache.clear()

    assert len(cache) == 0
    assert cache.stats.total_lookups == 0


def test_invalid_max_size_rejected() -> None:
    with pytest.raises(ValueError):
        SemanticCache(max_size=0)


def test_invalid_similarity_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        SemanticCache(similarity_threshold=1.5)


def test_exact_text_hit_without_embedding_argument() -> None:
    cache = SemanticCache(max_size=5, similarity_threshold=0.93)
    cache.put("what is a chunk?", [1.0, 0.0, 0.0], k=3, value="chunk answer")

    # Passing embedding=None should still hit exact match without error.
    result = cache.get("what is a chunk?", embedding=None, k=3)

    assert result == "chunk answer"
    assert cache.stats.hits_exact == 1
    assert cache.stats.hits_semantic == 0
    assert cache.stats.misses == 0


class _Counter:
    def __init__(self) -> None:
        self.calls = 0


def test_cached_retrieve_and_rerank_skips_work_on_hit() -> None:
    cache = SemanticCache(max_size=5, similarity_threshold=0.93)
    embed_calls = _Counter()
    retrieve_calls = _Counter()
    rerank_calls = _Counter()
    received_queries = []

    embeddings = {
        "when did the band debut?": [1.0, 0.0, 0.0],
        "what year did the band first debut?": [0.999, 0.03, 0.0],
        "what is Pikachu's typing?": [0.0, 1.0, 0.0],
    }

    def fake_embed(text: str):
        embed_calls.calls += 1
        return embeddings[text]

    def fake_retrieve(store, query, k):
        retrieve_calls.calls += 1
        received_queries.append(query)
        return [("doc", 0.5)]

    def fake_rerank(results, query):
        rerank_calls.calls += 1
        return results

    cached_fn = cached_retrieve_and_rerank(
        cache=cache, embed_fn=fake_embed, retrieve_fn=fake_retrieve, rerank_fn=fake_rerank
    )

    first = cached_fn(object(), "when did the band debut?", 3)
    assert embed_calls.calls == 1
    assert retrieve_calls.calls == 1
    assert rerank_calls.calls == 1
    assert received_queries[0] == [1.0, 0.0, 0.0]  # Vector passed directly on miss

    # Exact repeat of the same question -> exact hit, ZERO embedding calls!
    repeat = cached_fn(object(), "when did the band debut?", 3)
    assert repeat == first
    assert embed_calls.calls == 1  # Not incremented!
    assert retrieve_calls.calls == 1
    assert cache.stats.hits_exact == 1

    # Paraphrase of the same question -> semantic hit, 1 embed call, no retrieve call.
    second = cached_fn(object(), "what year did the band first debut?", 3)
    assert second == first
    assert embed_calls.calls == 2
    assert retrieve_calls.calls == 1
    assert rerank_calls.calls == 1
    assert cache.stats.hits_semantic == 1

    # Unrelated question -> real cache miss, does the work again.
    third = cached_fn(object(), "what is Pikachu's typing?", 3)
    assert third == first  # fake_retrieve always returns the same stub
    assert embed_calls.calls == 3
    assert retrieve_calls.calls == 2
    assert rerank_calls.calls == 2
    assert received_queries[1] == [0.0, 1.0, 0.0]
