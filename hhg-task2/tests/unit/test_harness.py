"""Unit tests for the harness/pipeline module."""

import pytest
from src.harness.models import PipelineResult, LatencyBreakdown
from src.harness.state import PipelineStage, RAGState
from src.guardrails.models import RefusalReason
from src.generation.extractive import extractive_answer


class TestPipelineResult:
    """Tests for PipelineResult model."""

    def test_successful_result(self):
        result = PipelineResult(
            answer="Machine learning is...",
            query="What is ML?",
            success=True,
            model_used="llama-3.1-70b",
        )
        assert result.success is True
        assert result.refused is False
        assert result.answer == "Machine learning is..."

    def test_refused_result(self):
        result = PipelineResult(
            query="What's the weather?",
            refused=True,
            refusal_reason=RefusalReason.OFF_TOPIC,
            refusal_message="Off topic",
        )
        assert result.refused is True
        assert result.refusal_reason == RefusalReason.OFF_TOPIC

    def test_latency_breakdown(self):
        latency = LatencyBreakdown(
            stt_ms=60.0,
            embedding_ms=5.0,
            retrieval_ms=25.0,
            generation_ms=50.0,
            total_ms=145.0,
        )
        assert latency.total_ms == 145.0


class TestPipelineStage:
    """Tests for pipeline stage enum."""

    def test_stages_exist(self):
        assert PipelineStage.IDLE == "IDLE"
        assert PipelineStage.TRANSCRIBING == "TRANSCRIBING"
        assert PipelineStage.COMPLETE == "COMPLETE"
        assert PipelineStage.ERROR == "ERROR"


class TestExtractiveAnswer:
    """Tests for the no-LLM grounded answer path."""

    def test_returns_source_sentences(self):
        chunks = [
            type("Chunk", (), {"text": "A corporation is a legal entity. It can conduct business."})(),
        ]
        answer = extractive_answer("what is a corporation?", chunks)
        assert answer == chunks[0].text

    def test_empty_chunks_refuse(self):
        assert extractive_answer("question", []) == ""
