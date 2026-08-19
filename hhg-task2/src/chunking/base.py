"""
Abstract base class for chunking strategies.

All chunking implementations must inherit from BaseChunker and implement `chunk()`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A single text chunk with metadata."""

    text: str
    chunk_index: int
    source_doc_id: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def char_length(self) -> int:
        return len(self.text)


class BaseChunker(ABC):
    """Abstract interface for text chunking strategies."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable name of this chunking strategy."""
        ...

    @abstractmethod
    def chunk(self, text: str, doc_id: str = "", metadata: dict | None = None) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: The text to chunk.
            doc_id: Source document identifier.
            metadata: Additional metadata to attach to each chunk.

        Returns:
            List of Chunk objects.
        """
        ...

    def chunk_batch(self, texts: list[dict]) -> list[Chunk]:
        """
        Chunk a batch of texts.

        Args:
            texts: List of dicts with keys 'text', 'doc_id', and optional 'metadata'.

        Returns:
            Flat list of all chunks from all texts.
        """
        all_chunks = []
        for item in texts:
            chunks = self.chunk(
                text=item["text"],
                doc_id=item.get("doc_id", ""),
                metadata=item.get("metadata"),
            )
            all_chunks.extend(chunks)
        return all_chunks
