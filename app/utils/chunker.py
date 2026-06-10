"""
chunker.py — Advanced text chunking strategies for the RAG ingestion pipeline.

Three complementary algorithms are provided, all sharing the same public interface:

  TextChunker           — Recursive character splitter (default, no I/O dependencies).
  SemanticChunker       — Embedding-similarity based chunker (requires async embedding fn).
  RecursiveChunker      — Alias for TextChunker; kept for explicit naming at call sites.

The calling code in document_service.py only uses TextChunker.chunk(text) synchronously,
so SemanticChunker is exposed as a separate class and invoked with
`await SemanticChunker(embed_fn).chunk(text)` when the caller can await.
"""

import re
import math
import logging
from typing import List, Callable, Awaitable

logger = logging.getLogger("com.rag.utils.chunker")


# ---------------------------------------------------------------------------
# 1. Recursive Character Splitter
# ---------------------------------------------------------------------------

class TextChunker:
    """
    Hierarchical recursive character splitter.

    Splitting priority (highest to lowest):
        1. Double newlines  →  paragraph boundaries
        2. Single newlines  →  line boundaries
        3. Period + space   →  sentence boundaries
        4. Space            →  word boundaries

    At each level the algorithm tries to split on the current separator.
    If a resulting piece is still larger than MAX_TOKENS it recurses to the
    next separator.  Once the piece fits, it is appended to the current
    running chunk; when adding a new piece would overflow the chunk the
    current chunk is flushed and a fresh one begins — but the last
    CHUNK_OVERLAP *words* of the just-flushed chunk are prepended to the
    next one (sliding-window overlap).

    Token counting is word-based (split on whitespace) — fast and sufficient
    given that 1 word ≈ 1.3 GPT/Gemini tokens on average.
    """

    # Maximum words per chunk before it must be flushed.
    # Maximum words per chunk before it must be flushed.
    CHUNK_SIZE: int = 512

    # Words carried over from the previous chunk into the next (overlap window).
    CHUNK_OVERLAP: int = 64

    # Ordered separator hierarchy: try coarser splits first.
    SEPARATORS: List[str] = ["\n\n", "\n", ". ", " "]

    @property
    def MAX_TOKENS(self) -> int:
        return self.CHUNK_SIZE

    @MAX_TOKENS.setter
    def MAX_TOKENS(self, value: int) -> None:
        self.CHUNK_SIZE = value

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        # Normalise runs of blank lines to a single double newline so the
        # paragraph separator always works reliably.
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip()

        pieces = self._split_recursive(text, self.SEPARATORS)
        return self._merge_with_overlap(pieces)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _word_count(self, text: str) -> int:
        return len(text.split())

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split *text* using the first separator that produces
        multiple pieces.  Pieces still larger than CHUNK_SIZE are split
        again with the next separator in the hierarchy.
        """
        if not separators:
            # Leaf level — hard word split as last resort.
            words = text.split()
            result = []
            for i in range(0, len(words), self.CHUNK_SIZE):
                result.append(" ".join(words[i: i + self.CHUNK_SIZE]))
            return result

        sep, remaining_seps = separators[0], separators[1:]

        if self._word_count(text) <= self.CHUNK_SIZE:
            # Already fits — no splitting needed.
            return [text.strip()] if text.strip() else []

        raw_splits = text.split(sep)
        if len(raw_splits) == 1:
            # Separator not found in text — try the next one.
            return self._split_recursive(text, remaining_seps)

        result = []
        for piece in raw_splits:
            piece = piece.strip()
            if not piece:
                continue
            if self._word_count(piece) > self.CHUNK_SIZE:
                # Piece is still too large — recurse with next separator.
                result.extend(self._split_recursive(piece, remaining_seps))
            else:
                result.append(piece)

        return result

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """
        Merge small pieces into chunks up to CHUNK_SIZE words, then slide
        the window forward while carrying CHUNK_OVERLAP words into the next
        chunk.
        """
        if not pieces:
            return []

        chunks: List[str] = []
        current_words: List[str] = []

        for piece in pieces:
            piece_words = piece.split()
            # If adding this piece overflows the current chunk, flush first.
            if current_words and (len(current_words) + len(piece_words)) > self.CHUNK_SIZE:
                chunks.append(" ".join(current_words))
                # Carry overlap from the tail of the flushed chunk.
                overlap_words = current_words[-self.CHUNK_OVERLAP:]
                current_words = overlap_words + piece_words
            else:
                current_words.extend(piece_words)

        if current_words:
            chunks.append(" ".join(current_words))

        return [c.strip() for c in chunks if c.strip()]


# Explicit alias so callers can import either name.
RecursiveChunker = TextChunker


# ---------------------------------------------------------------------------
# 2. Semantic Chunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """
    Embedding-similarity semantic chunker.

    Algorithm
    ---------
    1. Split text into sentences using a simple regex.
    2. Embed each sentence by calling the provided async *embed_fn*.
    3. Compute the cosine similarity between each pair of adjacent sentences.
    4. Identify breakpoints where similarity drops below *similarity_threshold*
       (a topic shift).
    5. Merge sentences between breakpoints into chunks; if a merged chunk
       exceeds MAX_TOKENS words it is further split by the RecursiveChunker
       so every output chunk still respects size limits.
    6. Apply a sliding-window overlap of CHUNK_OVERLAP *words* between
       consecutive final chunks (same mechanic as TextChunker).

    Parameters
    ----------
    embed_fn : async callable (str) -> List[float]
        Any async function that returns a float vector for a given text.
        Typically ``gemini_service.generate_embedding``.
    similarity_threshold : float
        Cosine-similarity value below which a new chunk is started.
        Lower → more (finer) chunks.  Default 0.75.
    max_tokens : int
        Maximum words per output chunk.  Default 400.
    chunk_overlap : int
        Words carried over between consecutive chunks.  Default 80.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], Awaitable[List[float]]],
        similarity_threshold: float = 0.75,
        max_tokens: int = 400,
        chunk_overlap: int = 80,
    ):
        self._embed_fn = embed_fn
        self._threshold = similarity_threshold
        self._max_tokens = max_tokens
        self._overlap = chunk_overlap
        self._recursive = TextChunker()
        self._recursive.MAX_TOKENS = max_tokens
        self._recursive.CHUNK_OVERLAP = chunk_overlap

    async def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        # Short texts: skip embedding overhead and use recursive splitter.
        if len(sentences) <= 2:
            return self._recursive.chunk(text)

        # Embed all sentences (respect the rate limiter in gemini_service via
        # sequential calls — the semaphore inside generate_embedding handles it).
        embeddings: List[List[float]] = []
        for sent in sentences:
            emb = await self._embed_fn(sent)
            embeddings.append(emb)

        # Find semantic breakpoints.
        breakpoints: List[int] = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self._threshold:
                breakpoints.append(i + 1)  # start of new group

        # Group sentences between breakpoints.
        groups = self._group_sentences(sentences, breakpoints)

        # Convert groups → raw chunk texts, then apply size-aware merge + overlap.
        raw_chunks: List[str] = []
        for group in groups:
            merged = " ".join(group)
            if len(merged.split()) > self._max_tokens:
                # Too big even after semantic grouping — recurse.
                raw_chunks.extend(self._recursive.chunk(merged))
            else:
                raw_chunks.append(merged)

        return self._recursive._merge_with_overlap(
            [piece for chunk in raw_chunks for piece in [chunk]]
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Sentence-tokenise using a lightweight regex that handles common
        abbreviations, decimal numbers, and multi-sentence lines without
        requiring NLTK or spaCy.
        """
        # Normalise whitespace first.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text)

        # Split on sentence-ending punctuation followed by whitespace + capital.
        # Pattern: end-of-sentence punctuation (.!?) followed by optional
        # closing quotes/brackets, then whitespace, then uppercase or newline.
        pattern = r'(?<=[.!?])\s+(?=[A-Z\n"\'])'
        raw = re.split(pattern, text)

        sentences = []
        for s in raw:
            s = s.strip()
            if s:
                sentences.append(s)
        return sentences

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0.0 or mag_b == 0.0:
            return 0.0
        return dot / (mag_a * mag_b)

    @staticmethod
    def _group_sentences(
        sentences: List[str], breakpoints: List[int]
    ) -> List[List[str]]:
        groups: List[List[str]] = []
        prev = 0
        for bp in breakpoints:
            groups.append(sentences[prev:bp])
            prev = bp
        groups.append(sentences[prev:])
        return [g for g in groups if g]
