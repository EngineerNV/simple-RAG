"""semantic_cache.py — a small LRU cache that matches on meaning, not just text.

**The problem.** Every chat turn that uses RAG re-runs the full retrieval path:
embed the query, search Chroma's HNSW index, fetch/decode the matching chunks,
then rerank them with a cross-encoder. That's wasted work if the same question
(or a differently-worded one with the same meaning) was already answered —
likely here, since the demo corpus is only ~120 chunks, so a handful of chunks
get retrieved across many different questions.

**The fix.** Cache the (query -> retrieved+reranked results) mapping, keyed
two ways:
  1. An exact hash of the normalized query text — O(1), zero false-positive
     risk, catches literal repeats.
  2. A cosine-similarity scan over whatever's currently cached — catches
     paraphrases ("When did X debut?" vs. "What year did X first debut?")
     that embed to nearly the same vector but hash differently.

**Why brute-force cosine, not an ANN index (HNSW/IVF), for step 2?** This
cache is deliberately tiny — `max_size` defaults to 20 entries. A linear scan
over <=20 vectors is both faster *and* far simpler to read than building and
maintaining a second ANN index just for this. ANN structures start paying for
themselves at thousands-to-millions of vectors; this project's *actual*
vector store (Chroma, HNSW-backed, ~120 chunks) is already close to the line
where that tradeoff is debatable. A 20-slot cache is nowhere near it. See the
README's "Semantic result caching" section for the full comparison.

**Why cache the query, not the whole conversation?** The cache key here is
expected to be the *rewritten*, self-contained search query (see
`agent_orchestration_helper.rewrite_for_retrieval`), not the raw user
message. The rewriter already resolves conversational context (anaphora,
follow-ups) into a standalone string, so caching at that point is safe: two
different conversations that rewrite down to the same search intent *should*
share a cache entry. Caching the raw message instead would be wrong, since
identical phrasing can mean different things depending on prior turns.

**Persistence.** This cache is in-memory and lives only as long as the
`ChatEngine` instance holding it — the simplest thing that works at demo
scale. For a persistent or multi-process cache, swap the `OrderedDict` below
for `diskcache.Cache` or a Redis-backed store; the `get`/`put` interface
would stay the same.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

ValueT = TypeVar("ValueT")

DEFAULT_MAX_SIZE = 20
DEFAULT_SIMILARITY_THRESHOLD = 0.93


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors, in [-1, 1].

    Plain Python (no numpy): at cache-sized inputs (<=20 comparisons of a
    few hundred dimensions each) a numpy dependency would buy nothing.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalize_query(text: str) -> str:
    """Collapse case/whitespace differences so trivial variations still hash equal."""
    return " ".join(text.strip().lower().split())


def _exact_key(query_text: str, k: int) -> str:
    normalized = _normalize_query(query_text)
    return hashlib.sha256(f"{k}:{normalized}".encode("utf-8")).hexdigest()


@dataclass
class CacheEntry(Generic[ValueT]):
    """One cached (query, k) -> value mapping, plus what's needed to match it semantically."""

    query_text: str
    embedding: List[float]
    k: int
    value: ValueT


@dataclass
class CacheStats:
    """Running counters, mainly so the cache's behavior is observable while debugging/teaching."""

    hits_exact: int = 0
    hits_semantic: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total_lookups(self) -> int:
        return self.hits_exact + self.hits_semantic + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_lookups
        return 0.0 if total == 0 else (self.hits_exact + self.hits_semantic) / total


class SemanticCache(Generic[ValueT]):
    """LRU cache keyed by query text, matched by exact hash then cosine similarity.

    ``k`` (the number of contexts requested) is part of both the key and the
    match: a result cached for ``k=3`` is never served for a ``k=5`` request,
    even if the query text/embedding is identical.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if not (0.0 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be within [0.0, 1.0]")
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self._entries: "OrderedDict[str, CacheEntry[ValueT]]" = OrderedDict()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._entries)

    def get(
        self,
        query_text: str,
        embedding: Optional[Sequence[float]] = None,
        k: int = 3,
    ) -> Optional[ValueT]:
        """Return a cached value for this (query, k), or ``None`` on a miss.

        Checks exact hash first (requiring no embedding math). If embedding is
        provided, performs cosine similarity scan over cached entries on an exact miss.
        """

        exact_key = _exact_key(query_text, k)
        exact_entry = self._entries.get(exact_key)
        if exact_entry is not None:
            self._entries.move_to_end(exact_key)
            self.stats.hits_exact += 1
            logger.debug("Semantic cache HIT (exact) for query=%r k=%d", query_text, k)
            return exact_entry.value

        if embedding is None:
            return None

        best_key: Optional[str] = None
        best_score = 0.0
        for key, entry in self._entries.items():
            if entry.k != k:
                continue
            score = cosine_similarity(embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_key = key

        if best_key is not None and best_score >= self.similarity_threshold:
            self._entries.move_to_end(best_key)
            self.stats.hits_semantic += 1
            logger.debug(
                "Semantic cache HIT (semantic, score=%.4f) for query=%r k=%d", best_score, query_text, k
            )
            return self._entries[best_key].value

        self.stats.misses += 1
        logger.debug(
            "Semantic cache MISS for query=%r k=%d (best_score=%.4f)", query_text, k, best_score
        )
        return None

    def put(self, query_text: str, embedding: Sequence[float], k: int, value: ValueT) -> None:
        """Insert or refresh a (query, k) -> value mapping, evicting the LRU entry if full."""

        key = _exact_key(query_text, k)
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = CacheEntry(query_text=query_text, embedding=list(embedding), k=k, value=value)
        if len(self._entries) > self.max_size:
            self._entries.popitem(last=False)
            self.stats.evictions += 1

    def clear(self) -> None:
        self._entries.clear()
        self.stats = CacheStats()


def cached_retrieve_and_rerank(
    cache: SemanticCache,
    embed_fn: Callable[[str], Sequence[float]],
    retrieve_fn: Callable[..., ValueT],
    rerank_fn: Optional[Callable[..., ValueT]] = None,
) -> Callable[..., ValueT]:
    """Wrap retrieve(+rerank) with ``cache``, matching ``retrieve_contexts_fn(store, query, k)``.

    That's the exact call signature `agent_orchestration_helper.apply_rewrite_and_retrieve`
    already expects, so the returned function is a drop-in replacement for
    ``retrieve_contexts`` — no changes needed anywhere else in the pipeline.
    On an exact cache hit, no embedding calculation is performed.
    On a semantic cache miss, the computed embedding vector is reused for vector store
    retrieval to prevent redundant embedding calculations.
    """

    def _retrieve_with_cache(store: object, query: str, k: int) -> ValueT:
        # 1. Exact match hit? O(1) text hash lookup, zero embedding inference!
        cached_value = cache.get(query, embedding=None, k=k)
        if cached_value is not None:
            return cached_value

        # 2. Exact miss: generate query embedding vector for semantic scan
        embedding = embed_fn(query)
        cached_value = cache.get(query, embedding=embedding, k=k)
        if cached_value is not None:
            return cached_value

        # 3. Semantic cache miss: reuse the pre-computed embedding vector for vector store retrieval
        try:
            results = retrieve_fn(store, embedding, k)
        except (TypeError, ValueError, AttributeError):
            results = retrieve_fn(store, query, k)

        if rerank_fn is not None:
            try:
                results = rerank_fn(results, query)
            except Exception:
                # Reranker is best-effort elsewhere in this pipeline too; keep
                # that contract here rather than caching a partial failure mode.
                pass
        cache.put(query, embedding, k, results)
        return results

    return _retrieve_with_cache
