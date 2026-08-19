"""
Strategy A: Fixed-Size Chunking with Overlap.

The baseline chunking strategy. Splits text into fixed-size chunks with configurable
overlap. Uses recursive splitting to try to break at natural boundaries (paragraphs,
sentences, words) before falling back to character-level splits.
"""

import structlog

from src.chunking.base import BaseChunker, Chunk

logger = structlog.get_logger(__name__)


class FixedSizeChunker(BaseChunker):
    """Fixed-size text chunker with overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 102):
        """
        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between consecutive chunks.
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = ["\n\n", "\n", ". ", ", ", " ", ""]

    @property
    def strategy_name(self) -> str:
        return f"fixed_size_{self._chunk_size}_overlap_{self._chunk_overlap}"

    def chunk(self, text: str, doc_id: str = "", metadata: dict | None = None) -> list[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        raw_chunks = self._recursive_split(text)
        chunks = []

        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_index=i,
                    source_doc_id=doc_id,
                    metadata={
                        **metadata,
                        "strategy": self.strategy_name,
                        "chunk_size": self._chunk_size,
                        "chunk_overlap": self._chunk_overlap,
                    },
                )
            )

        logger.debug(
            "fixed_size_chunking_complete",
            doc_id=doc_id,
            num_chunks=len(chunks),
            avg_chunk_len=sum(c.char_length for c in chunks) / max(len(chunks), 1),
        )

        return chunks

    def _recursive_split(self, text: str, separator_start: int = 0) -> list[str]:
        """Recursively split text using separator hierarchy."""
        if len(text) <= self._chunk_size:
            return [text]

        # Try each separator from most to least preferred
        for separator_index, sep in enumerate(self._separators[separator_start:], separator_start):
            if sep and sep in text:
                parts = text.split(sep)
                return self._merge_parts(parts, sep, separator_index + 1)

        # Fallback: hard split at chunk_size
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = max(start + 1, end - self._chunk_overlap)
        return chunks

    def _merge_parts(self, parts: list[str], separator: str, separator_start: int) -> list[str]:
        """Merge split parts back together respecting chunk_size, with overlap."""
        chunks = []
        current_chunk = ""

        for part in parts:
            candidate = f"{current_chunk}{separator}{part}" if current_chunk else part

            if len(candidate) <= self._chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Create overlap from end of previous chunk
                    overlap_text = current_chunk[-self._chunk_overlap:] if self._chunk_overlap > 0 else ""
                    current_chunk = f"{overlap_text}{separator}{part}" if overlap_text else part
                else:
                    # Single part is larger than chunk_size — recurse
                    sub_chunks = self._recursive_split(part, separator_start)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
