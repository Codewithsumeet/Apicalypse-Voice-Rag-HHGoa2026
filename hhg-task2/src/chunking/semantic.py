"""
Strategy B: Semantic Chunking.

Splits text at semantic boundaries detected by embedding similarity drops.
Uses a sliding window of sentences — when cosine similarity between consecutive
sentence embeddings drops below a threshold, a chunk boundary is placed.
"""

import numpy as np
import structlog

from src.chunking.base import BaseChunker, Chunk

logger = structlog.get_logger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticChunker(BaseChunker):
    """Semantic boundary-based text chunker."""

    def __init__(
        self,
        embedding_model=None,
        similarity_threshold: float = 0.85,
        min_chunk_size: int = 128,
        max_chunk_size: int = 1024,
    ):
        """
        Args:
            embedding_model: A SentenceTransformer model instance (shared, loaded once).
            similarity_threshold: Cosine similarity threshold — split when similarity drops below.
            min_chunk_size: Minimum characters per chunk (prevents tiny fragments).
            max_chunk_size: Maximum characters per chunk (forces split even without boundary).
        """
        self._model = embedding_model
        self._threshold = similarity_threshold
        self._min_size = min_chunk_size
        self._max_size = max_chunk_size

    def set_embedding_model(self, model):
        """Set the embedding model (allows lazy initialization)."""
        self._model = model

    @property
    def strategy_name(self) -> str:
        return f"semantic_threshold_{self._threshold}"

    def chunk(self, text: str, doc_id: str = "", metadata: dict | None = None) -> list[Chunk]:
        """Split text at semantic boundaries using embedding similarity drops."""
        if not text or not text.strip():
            return []
        if self._model is None:
            raise RuntimeError("Embedding model not set. Call set_embedding_model() first.")

        metadata = metadata or {}

        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [
                Chunk(
                    text=text.strip(),
                    chunk_index=0,
                    source_doc_id=doc_id,
                    metadata={**metadata, "strategy": self.strategy_name},
                )
            ]

        # Encode all sentences in batch (efficient)
        embeddings = self._model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

        # Find semantic boundaries
        chunks = []
        current_sentences = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            current_text = ". ".join(current_sentences)

            # Place boundary if similarity drops AND chunk is large enough
            if sim < self._threshold and len(current_text) >= self._min_size:
                chunks.append(current_text)
                current_sentences = [sentences[i]]
            elif len(current_text) >= self._max_size:
                # Force split even if similarity is high (prevent mega-chunks)
                chunks.append(current_text)
                current_sentences = [sentences[i]]
            else:
                current_sentences.append(sentences[i])

        # Don't forget the last chunk
        if current_sentences:
            remaining = ". ".join(current_sentences)
            if remaining.strip():
                chunks.append(remaining)

        # Build Chunk objects
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            result.append(
                Chunk(
                    text=chunk_text,
                    chunk_index=i,
                    source_doc_id=doc_id,
                    metadata={
                        **metadata,
                        "strategy": self.strategy_name,
                        "similarity_threshold": self._threshold,
                    },
                )
            )

        logger.debug(
            "semantic_chunking_complete",
            doc_id=doc_id,
            num_sentences=len(sentences),
            num_chunks=len(result),
        )

        return result

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences (simple but effective)."""
        import re

        # Split on period, question mark, exclamation — but not abbreviations
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
