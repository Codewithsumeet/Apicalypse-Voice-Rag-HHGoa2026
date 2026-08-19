"""
Strategy C: Metadata-Aware / Query-Passage Chunking.

Leverages MSMARCO-XI's query-passage structure. Each chunk carries the original
query as metadata, enabling hybrid retrieval (dense vector + keyword/BM25 boost).
"""

import structlog

from src.chunking.base import BaseChunker, Chunk
from src.chunking.fixed_size import FixedSizeChunker

logger = structlog.get_logger(__name__)


class MetadataAwareChunker(BaseChunker):
    """MSMARCO-specific chunker that preserves query-passage relationships."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        """
        Args:
            chunk_size: Maximum characters per sub-chunk within a passage.
            chunk_overlap: Overlap between sub-chunks.
        """
        self._inner_chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @property
    def strategy_name(self) -> str:
        return "metadata_aware_query_passage"

    def chunk(self, text: str, doc_id: str = "", metadata: dict | None = None) -> list[Chunk]:
        """
        Chunk a passage while preserving its associated query in metadata.

        The `metadata` dict should contain a 'query' key from the MSMARCO record.
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        query = metadata.get("query", "")

        # Use inner fixed-size chunker for the actual text splitting
        inner_chunks = self._inner_chunker.chunk(text=text, doc_id=doc_id)

        # Enrich each chunk with query metadata for hybrid retrieval
        result = []
        for i, inner_chunk in enumerate(inner_chunks):
            result.append(
                Chunk(
                    text=inner_chunk.text,
                    chunk_index=i,
                    source_doc_id=doc_id,
                    metadata={
                        **metadata,
                        "strategy": self.strategy_name,
                        "original_query": query,
                        "has_query_metadata": bool(query),
                        "parent_chunk_count": len(inner_chunks),
                    },
                )
            )

        logger.debug(
            "metadata_aware_chunking_complete",
            doc_id=doc_id,
            num_chunks=len(result),
            has_query=bool(query),
        )

        return result

    def chunk_from_record(self, record: dict) -> list[Chunk]:
        """
        Convenience method to chunk directly from an MSMARCO-XI dataset record.

        Args:
            record: Dict with keys like 'query', 'passage', 'docid', etc.

        Returns:
            List of Chunk objects with query metadata attached.
        """
        passage = record.get("passage", record.get("text", ""))
        query = record.get("query", "")
        doc_id = str(record.get("docid", record.get("id", "")))

        return self.chunk(
            text=passage,
            doc_id=doc_id,
            metadata={"query": query},
        )
