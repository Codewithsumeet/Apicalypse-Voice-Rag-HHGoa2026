"""
Pydantic models for the pipeline harness.
"""

from pydantic import BaseModel, Field
from src.guardrails.models import RefusalReason


class LatencyBreakdown(BaseModel):
    """Per-stage latency breakdown for observability."""

    stt_ms: float = Field(default=0.0, description="Speech-to-text latency")
    embedding_ms: float = Field(default=0.0, description="Query embedding latency")
    retrieval_ms: float = Field(default=0.0, description="Vector DB retrieval latency")
    guardrail_pre_ms: float = Field(default=0.0, description="Pre-generation guardrail latency")
    generation_ms: float = Field(default=0.0, description="LLM generation latency")
    guardrail_post_ms: float = Field(default=0.0, description="Post-generation grounding check latency")
    total_ms: float = Field(default=0.0, description="RAG pipeline total latency (excluding STT/network)")
    e2e_ms: float = Field(default=0.0, description="Total end-to-end latency including STT")


class PipelineResult(BaseModel):
    """Complete structured output from the RAG pipeline."""

    # Core output
    answer: str = Field(default="", description="The final answer text")
    transcript: str = Field(default="", description="STT transcript (if voice input)")
    query: str = Field(default="", description="The query used for retrieval")

    # Status
    success: bool = Field(default=True, description="Whether the pipeline completed successfully")
    refused: bool = Field(default=False, description="Whether the query was refused by guardrails")
    refusal_reason: RefusalReason = Field(default=RefusalReason.PASS, description="Reason for refusal")
    refusal_message: str = Field(default="", description="Human-readable refusal message")

    # Metadata
    model_used: str = Field(default="", description="LLM model used for generation")
    is_fallback: bool = Field(default=False, description="Whether a fallback model was used")
    trace_id: str = Field(default="", description="Request trace ID for debugging")

    # Latency
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown, description="Per-stage latency breakdown")

    # Retrieved context (for debugging/display)
    retrieved_chunks: list[dict] = Field(default_factory=list, description="Retrieved context chunks")
