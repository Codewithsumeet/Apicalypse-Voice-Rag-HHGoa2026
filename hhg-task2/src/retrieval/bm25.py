"""
Local BM25 Sparse Retriever.

Wrapper around rank-bm25 to enable keyword/sparse search.
Tokenizes Hindi and English text, builds a BM25 index per namespace,
and returns document scores for fusion.
"""

import re
import time
import numpy as np
import structlog
from rank_bm25 import BM25Okapi

logger = structlog.get_logger(__name__)


def tokenize(text: str) -> list[str]:
    """
    Tokenize query/passage text.
    Handles Hindi and English unicode characters correctly by splitting on whitespace
    and stripping punctuation.
    """
    if not text:
        return []
    
    # Split by whitespace
    tokens = text.lower().split()
    
    # Strip common punctuation from tokens
    cleaned = []
    punctuation_chars = '.,?!;:"()[]{}<>’\'"_*#-'
    for token in tokens:
        token = token.strip(punctuation_chars)
        if token:
            cleaned.append(token)
            
    return cleaned


class BM25Searcher:
    """Namespace-aware BM25 sparse index builder and retriever."""

    def __init__(self, texts: list[str], metadatas: list[dict]):
        self.texts = texts
        self.metadatas = metadatas
        # Cache BM25Okapi index and mapping of (filtered_index -> original_index) per namespace
        # Schema: {namespace: (bm25_instance, mapping_list)}
        self._cache: dict[str, tuple[BM25Okapi, list[int]]] = {}

    def _get_index(self, namespace: str) -> tuple[BM25Okapi, list[int]]:
        """Get or lazily build the BM25 index and index mapping for a namespace."""
        if namespace in self._cache:
            return self._cache[namespace]

        start_time = time.perf_counter()

        # Filter texts by namespace
        filtered_indices = []
        filtered_texts = []
        for i, meta in enumerate(self.metadatas):
            if meta.get("namespace") == namespace:
                filtered_indices.append(i)
                filtered_texts.append(self.texts[i])

        if not filtered_texts:
            # Empty fallback
            bm25 = BM25Okapi([[]])
            self._cache[namespace] = (bm25, [])
            return bm25, []

        # Tokenize corpus
        tokenized_corpus = [tokenize(text) for text in filtered_texts]

        # Build index
        bm25 = BM25Okapi(tokenized_corpus)
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "bm25_index_built",
            namespace=namespace,
            documents=len(filtered_texts),
            duration_ms=round(duration_ms, 2),
        )

        self._cache[namespace] = (bm25, filtered_indices)
        return bm25, filtered_indices

    def query(self, query_str: str, top_k: int = 5, namespace: str = "default") -> list[tuple[int, float]]:
        """
        Query the BM25 index.

        Args:
            query_str: Raw text query.
            top_k: Number of results to return.
            namespace: Document namespace to search within.

        Returns:
            List of (original_corpus_index, bm25_score) tuples, sorted by score descending.
        """
        if not query_str or not query_str.strip():
            return []

        bm25, filtered_indices = self._get_index(namespace)
        if not filtered_indices:
            return []

        tokenized_query = tokenize(query_str)
        if not tokenized_query:
            return []

        # Get scores for all docs in the namespace
        scores = bm25.get_scores(tokenized_query)

        # Select only the best candidates; sorting every document made BM25 the
        # dominant local latency stage on the fixed demo index.
        candidate_count = min(top_k, len(scores))
        if candidate_count == 0:
            return []
        candidate_positions = np.argpartition(scores, -candidate_count)[-candidate_count:]
        candidate_positions = candidate_positions[np.argsort(scores[candidate_positions])[::-1]]
        return [
            (filtered_indices[int(position)], float(scores[int(position)]))
            for position in candidate_positions
            if scores[int(position)] > 0
        ]
