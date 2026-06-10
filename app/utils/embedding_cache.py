"""
embedding_cache.py — Async TTL + LRU cache for vector embeddings.

Design goals
------------
* Zero external dependencies (pure Python + asyncio).
* Safe for use from concurrent async tasks (asyncio.Lock, not threading.Lock).
* Evicts the *least recently used* entry when the cache is full (LRU).
* Discards any entry that has exceeded its time-to-live (TTL).
* Provides a cache-aside helper `get_or_compute` that callers can use
  without any boilerplate.

Usage
-----
    from app.utils.embedding_cache import EmbeddingCache

    cache = EmbeddingCache(maxsize=1024, ttl=3600)

    async def embed(text: str) -> List[float]:
        return await cache.get_or_compute(text, some_async_embed_fn)
"""

import asyncio
import time
import logging
from collections import OrderedDict
from typing import Awaitable, Callable, List, Optional, Tuple
from app.utils.metrics import metrics_collector

logger = logging.getLogger("com.rag.utils.embedding_cache")


class EmbeddingCache:
    """
    Async-safe, TTL-aware LRU cache for text embedding vectors.

    Parameters
    ----------
    maxsize : int
        Maximum number of distinct texts to keep in memory.  When the limit
        is reached the least-recently-used entry is evicted.
    ttl : int
        Time-to-live in seconds.  Entries older than this are treated as
        cache misses and recomputed on the next access.
    """

    def __init__(self, maxsize: int = 1024, ttl: int = 3600) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        # OrderedDict preserves insertion order; move_to_end() implements LRU.
        # Value layout: (embedding: List[float], inserted_at: float)
        self._store: OrderedDict[str, Tuple[List[float], float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[str], Awaitable[List[float]]],
    ) -> List[float]:
        """
        Return the cached embedding for *text*, or call *compute_fn* to
        produce and cache a fresh one.

        The lock is held only during cache lookup/update, not during the
        (potentially slow) embedding API call, so concurrent calls for the
        *same* text may both result in API calls.  This is intentional:
        it avoids a thundering-herd lock while keeping the implementation
        simple.  The last writer wins and the result is still correct.
        """
        cached = await self._get(text)
        if cached is not None:
            return cached

        # Cache miss — call the embedding function outside the lock.
        embedding = await compute_fn(text)
        if embedding:
            await self._set(text, embedding)
        return embedding

    async def invalidate(self, text: str) -> None:
        """Remove a specific key from the cache (e.g. after model change)."""
        async with self._lock:
            self._store.pop(text, None)

    async def clear(self) -> None:
        """Flush the entire cache."""
        async with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        """Return hit/miss counters and current size (non-blocking)."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0.0
        return {
            "size": len(self._store),
            "maxsize": self._maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, text: str) -> Optional[List[float]]:
        async with self._lock:
            if text not in self._store:
                self._misses += 1
                metrics_collector.record_embedding_cache(hit=False)
                return None

            embedding, inserted_at = self._store[text]

            # TTL check.
            if time.monotonic() - inserted_at > self._ttl:
                del self._store[text]
                self._misses += 1
                metrics_collector.record_embedding_cache(hit=False)
                logger.debug("Cache TTL expired for text (len=%d)", len(text))
                return None

            # Move to end (most recently used).
            self._store.move_to_end(text)
            self._hits += 1
            metrics_collector.record_embedding_cache(hit=True)
            return embedding

    async def _set(self, text: str, embedding: List[float]) -> None:
        async with self._lock:
            if text in self._store:
                # Update existing entry and mark as most recent.
                self._store.move_to_end(text)
            else:
                # Evict LRU entry if at capacity.
                if len(self._store) >= self._maxsize:
                    evicted_key, _ = self._store.popitem(last=False)
                    logger.debug(
                        "LRU eviction: cache full (maxsize=%d), dropped key len=%d",
                        self._maxsize,
                        len(evicted_key),
                    )

            self._store[text] = (embedding, time.monotonic())
