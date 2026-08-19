"""Compact BM25-only retrieval view for the low-latency demo path."""

import time

from src.retrieval.bm25 import BM25Searcher
from src.retrieval.models import RetrievalResult, RetrievedChunk


class FastSparseStore:
    """BM25 index over answer-bearing chunks already present in LocalNumpyStore."""

    def __init__(self, source_store, namespace: str = "demo_fast"):
        selected = [
            (text, metadata)
            for text, metadata in zip(source_store.texts, source_store.metadatas)
            if int(metadata.get("is_selected", 0)) == 1
        ]
        self.texts = [text for text, _ in selected]
        self.metadatas = [metadata for _, metadata in selected]
        self._searcher = BM25Searcher(self.texts, self.metadatas)
        self._searcher._get_index("fixed")

    @property
    def vector_count(self) -> int:
        return len(self.texts)

    def query(self, query_str: str, top_k: int = 5) -> RetrievalResult:
        started = time.perf_counter()
        results = self._searcher.query(query_str, top_k=top_k, namespace="fixed")
        chunks = [
            RetrievedChunk(
                text=self.texts[index],
                score=score,
                doc_id=self.metadatas[index].get("doc_id", ""),
                chunk_index=self.metadatas[index].get("chunk_index", 0),
                metadata={**self.metadatas[index], "retrieval_mode": "fast_bm25"},
            )
            for index, score in results
        ]
        return RetrievalResult(
            query=query_str,
            chunks=chunks,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )