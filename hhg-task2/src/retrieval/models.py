"""
Pydantic models for the retrieval module.
"""

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store."""

    text: str = Field(..., description="The chunk text content")
    score: float = Field(..., description="Similarity score (0-1)")
    doc_id: str = Field(default="", description="Source document ID")
    chunk_index: int = Field(default=0, description="Position of this chunk in the source document")
    metadata: dict = Field(default_factory=dict, description="Additional metadata from the chunk")


class RetrievalResult(BaseModel):
    """Structured output from the retrieval stage."""

    query: str = Field(..., description="The query that was searched")
    chunks: list[RetrievedChunk] = Field(default_factory=list, description="Retrieved chunks ranked by relevance")
    duration_ms: float = Field(default=0.0, description="Retrieval latency in milliseconds")
    total_candidates: int = Field(default=0, description="Total vectors searched")

    @property
    def top_chunk(self) -> RetrievedChunk | None:
        """Return the highest-scoring chunk, or None if empty."""
        return self.chunks[0] if self.chunks else None

    @property
    def context_text(self) -> str:
        """Concatenate all retrieved chunks into a single context string."""
        return "\n\n---\n\n".join(chunk.text for chunk in self.chunks)
