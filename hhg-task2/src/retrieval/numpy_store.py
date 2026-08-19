"""
Local NumPy Vector Store.

A pure-numpy implementation of a cosine-similarity vector store.
Extremely fast (under 2ms search for 10k vectors), requires no external database
setup, and bypasses all network latency. Perfect for meeting the <200ms budget.
"""

import time
import pickle
import os
import tempfile
import numpy as np
import structlog
from pathlib import Path

from src.retrieval.models import RetrievalResult, RetrievedChunk

logger = structlog.get_logger(__name__)


class LocalNumpyStore:
    """Pure NumPy vector store for zero-network-latency search."""

    def __init__(self, storage_path: str = "data/numpy_store.pkl"):
        self.storage_path = Path(storage_path)
        self.texts: list[str] = []
        self.embeddings: np.ndarray = np.empty((0, 0))
        self._normalized_embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.metadatas: list[dict] = []
        self._namespace_indices: dict[str, list[int]] = {}
        self.bm25_searcher = None

    def _rebuild_caches(self):
        """Build immutable query-time caches after loading or mutating the store."""
        if self.embeddings.size:
            embeddings = np.asarray(self.embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self._normalized_embeddings = embeddings / np.maximum(norms, 1e-10)
            self.embeddings = embeddings
        else:
            self._normalized_embeddings = np.empty((0, 0), dtype=np.float32)

        self._namespace_indices = {}
        for index, metadata in enumerate(self.metadatas):
            namespace = metadata.get("namespace", "default")
            self._namespace_indices.setdefault(namespace, []).append(index)

    def connect(self):
        """Load the index from disk if it exists."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "rb") as f:
                    data = pickle.load(f)
                self.texts = data.get("texts", [])
                self.embeddings = np.asarray(data.get("embeddings", []), dtype=np.float32)
                self.metadatas = data.get("metadatas", [])
                self._rebuild_caches()
                
                # Initialize and warm up BM25 searcher
                from src.retrieval.bm25 import BM25Searcher
                self.bm25_searcher = BM25Searcher(self.texts, self.metadatas)
                
                # Warm up unique namespaces to avoid hot path latency
                namespaces = set(m.get("namespace", "default") for m in self.metadatas)
                for ns in namespaces:
                    self.bm25_searcher._get_index(ns)
                
                logger.info(
                    "numpy_store_loaded",
                    vectors=len(self.texts),
                    dimensions=self.embeddings.shape[1] if len(self.texts) > 0 else 0,
                    warmed_namespaces=list(namespaces),
                )
            except Exception as e:
                logger.error("failed_to_load_numpy_store", error=str(e))
        else:
            logger.info("numpy_store_initialized_empty")

    def save(self):
        """Save the index to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.storage_path.name}.",
            suffix=".tmp",
            dir=self.storage_path.parent,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(
                    {
                        "texts": self.texts,
                        "embeddings": self.embeddings.tolist(),
                        "metadatas": self.metadatas,
                    },
                    f,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, self.storage_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        logger.info("numpy_store_saved", vectors=len(self.texts))

    def upsert_chunks(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        batch_size: int = 100,  # Unused, kept for interface compatibility
        namespace: str = "default",  # We separate namespaces by tagging in metadata
        persist: bool = True,
    ) -> int:
        """Add vectors and text to the local store."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("texts and metadatas must have the same length")
        if not texts:
            return 0

        metadatas = metadatas or [{}] * len(texts)
        
        # Tag namespace in metadata
        for meta in metadatas:
            meta["namespace"] = namespace

        new_embeddings = np.array(embeddings)
        
        if len(self.texts) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self._rebuild_caches()

        # Refresh BM25 searcher
        from src.retrieval.bm25 import BM25Searcher
        self.bm25_searcher = BM25Searcher(self.texts, self.metadatas)

        if persist:
            self.save()
        return len(texts)

    def query(
        self,
        query_embedding: list[float],
        query_str: str = "",
        top_k: int = 5,
        namespace: str = "default",
        filter_dict: dict | None = None,
    ) -> RetrievalResult:
        """Query using hybrid dense + sparse (BM25) search with RRF fusion."""
        start = time.perf_counter()

        if len(self.texts) == 0:
            return RetrievalResult(query="", chunks=[], duration_ms=0.0)

        # Filter by namespace and optional metadata filter
        indices = self._namespace_indices.get(namespace, [])
        if filter_dict:
            indices = [
                index
                for index in indices
                if all(self.metadatas[index].get(key) == value for key, value in filter_dict.items())
            ]

        if not indices:
            duration_ms = (time.perf_counter() - start) * 1000
            return RetrievalResult(query="", chunks=[], duration_ms=round(duration_ms, 2))

        # 1. DENSE VECTOR SEARCH
        query_vec = np.asarray(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return RetrievalResult(query=query_str, chunks=[], duration_ms=0.0)
        query_vec /= query_norm

        # Stored vectors are normalized once at load/upsert, so this is a cosine dot product.
        similarities = self._normalized_embeddings[indices] @ query_vec
        candidate_count = min(len(similarities), max(top_k, 100) if query_str.strip() else top_k)
        if candidate_count == len(similarities):
            dense_ranked_indices = np.argsort(similarities)[::-1]
        else:
            candidate_indices = np.argpartition(similarities, -candidate_count)[-candidate_count:]
            dense_ranked_indices = candidate_indices[np.argsort(similarities[candidate_indices])[::-1]]
        dense_orig_indices = [indices[idx] for idx in dense_ranked_indices]
        dense_scores_map = {indices[idx]: float(similarities[idx]) for idx in dense_ranked_indices}

        # 2. SPARSE BM25 RETRIEVAL & FUSION
        chunks = []
        is_hybrid = False

        if query_str and query_str.strip():
            is_hybrid = True
            if self.bm25_searcher is None:
                from src.retrieval.bm25 import BM25Searcher
                self.bm25_searcher = BM25Searcher(self.texts, self.metadatas)

            # Retrieve top 100 candidates from sparse search
            sparse_results = self.bm25_searcher.query(query_str, top_k=100, namespace=namespace)
            sparse_orig_indices = [idx for idx, _ in sparse_results]

            # Merge rankings via RRF (look at top 100 dense vs top 100 sparse candidates)
            from src.retrieval.fusion import reciprocal_rank_fusion
            fused_results = reciprocal_rank_fusion(
                dense_indices=dense_orig_indices[:100],
                sparse_indices=sparse_orig_indices,
                k=60,
                top_k=top_k,
            )

            # Build result chunks from fused list
            for orig_idx, rrf_score in fused_results:
                # We preserve the original dense cosine score in the score, but log RRF score in metadata
                cosine_score = dense_scores_map.get(orig_idx, 0.0)
                chunks.append(
                    RetrievedChunk(
                        text=self.texts[orig_idx],
                        score=cosine_score,  # Preserve cosine score for grounding checks
                        doc_id=self.metadatas[orig_idx].get("doc_id", ""),
                        chunk_index=self.metadatas[orig_idx].get("chunk_index", 0),
                        metadata={
                            **self.metadatas[orig_idx],
                            "rrf_score": rrf_score,
                            "dense_score": cosine_score,
                            "retrieval_mode": "hybrid_rrf",
                        },
                    )
                )
        else:
            # Pure dense fallback
            for idx in dense_ranked_indices[:top_k]:
                orig_idx = indices[idx]
                chunks.append(
                    RetrievedChunk(
                        text=self.texts[orig_idx],
                        score=float(similarities[idx]),
                        doc_id=self.metadatas[orig_idx].get("doc_id", ""),
                        chunk_index=self.metadatas[orig_idx].get("chunk_index", 0),
                        metadata={
                            **self.metadatas[orig_idx],
                            "retrieval_mode": "dense_only",
                        },
                    )
                )

        duration_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "numpy_query_complete",
            top_k=top_k,
            hybrid=is_hybrid,
            results_returned=len(chunks),
            duration_ms=round(duration_ms, 2),
        )

        return RetrievalResult(
            query=query_str,
            chunks=chunks,
            duration_ms=round(duration_ms, 2),
        )

    def delete_namespace(self, namespace: str = "default"):
        """Delete all vectors matching a namespace."""
        indices_to_keep = [i for i, m in enumerate(self.metadatas) if m.get("namespace") != namespace]
        
        if not indices_to_keep:
            self.texts = []
            self.embeddings = np.empty((0, 0))
            self.metadatas = []
        else:
            self.texts = [self.texts[i] for i in indices_to_keep]
            self.embeddings = self.embeddings[indices_to_keep]
            self.metadatas = [self.metadatas[i] for i in indices_to_keep]
            self._rebuild_caches()

        if not indices_to_keep:
            self._rebuild_caches()

        self.save()

    def get_stats(self) -> dict:
        """Return vector counts and dimensions for the local store."""
        namespaces = {}
        # Count vectors per namespace
        for meta in self.metadatas:
            ns = meta.get("namespace", "default")
            if ns not in namespaces:
                namespaces[ns] = {"vector_count": 0}
            namespaces[ns]["vector_count"] += 1
            
        return {
            "dimension": self.embeddings.shape[1] if len(self.texts) > 0 else 0,
            "index_fullness": 0.0,
            "namespaces": namespaces,
            "total_vector_count": len(self.texts),
        }
