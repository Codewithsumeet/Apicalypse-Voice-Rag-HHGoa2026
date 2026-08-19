"""
Pipeline state definition.

Defines the RAGState TypedDict used by LangGraph to track data flowing
through the pipeline, and the PipelineStage enum for state machine transitions.
"""

from enum import Enum
from typing import TypedDict


class PipelineStage(str, Enum):
    """Pipeline execution stages for the state machine."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    RETRIEVING = "RETRIEVING"
    GUARDRAIL_CHECK = "GUARDRAIL_CHECK"
    GENERATING = "GENERATING"
    GROUNDING_CHECK = "GROUNDING_CHECK"
    COMPLETE = "COMPLETE"
    REFUSED = "REFUSED"
    ERROR = "ERROR"


class RAGState(TypedDict, total=False):
    """State object flowing through the LangGraph pipeline."""

    # Input
    audio_bytes: bytes
    query_text: str  # For text-only mode (bypasses STT)

    # STT output
    transcript: str
    stt_confidence: float
    stt_language: str
    stt_duration_ms: float

    # Retrieval output
    retrieved_chunks: list[dict]
    retrieval_context: str
    retrieval_duration_ms: float

    # Guardrail outputs
    pre_guardrail_passed: bool
    pre_guardrail_reason: str
    pre_guardrail_message: str
    post_guardrail_passed: bool
    post_guardrail_reason: str
    post_guardrail_message: str

    # Generation output
    answer: str
    generation_model: str
    generation_duration_ms: float

    # Pipeline metadata
    stage: str
    total_duration_ms: float
    error: str
    trace_id: str

    # Latency breakdown
    latency_breakdown: dict
