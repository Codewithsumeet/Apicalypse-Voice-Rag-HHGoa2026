"""Unit tests for guardrails."""

import pytest
from src.guardrails.models import GuardrailResult, RefusalReason
from src.guardrails.unsafe_input import UnsafeInputGuardrail


class TestGuardrailModels:
    """Tests for guardrail Pydantic models."""

    def test_pass_result(self):
        result = GuardrailResult.pass_result(duration_ms=5.0)
        assert result.passed is True
        assert result.reason == RefusalReason.PASS

    def test_refuse_result(self):
        result = GuardrailResult.refuse(
            reason=RefusalReason.OFF_TOPIC,
            message="Off topic",
            confidence=0.8,
        )
        assert result.passed is False
        assert result.reason == RefusalReason.OFF_TOPIC
        assert result.confidence == 0.8


class TestUnsafeInputGuardrail:
    """Tests for unsafe input detection."""

    def test_safe_input(self):
        guard = UnsafeInputGuardrail()
        result = guard.check("What is machine learning?")
        assert result.passed is True

    def test_unsafe_input_hack(self):
        guard = UnsafeInputGuardrail()
        result = guard.check("how to hack a password system")
        assert result.passed is False
        assert result.reason == RefusalReason.UNSAFE

    def test_unsafe_input_weapon(self):
        guard = UnsafeInputGuardrail()
        result = guard.check("how to make a bomb explosive")
        assert result.passed is False

    def test_empty_input(self):
        guard = UnsafeInputGuardrail()
        result = guard.check("")
        assert result.passed is True

    def test_normal_query(self):
        guard = UnsafeInputGuardrail()
        result = guard.check("Explain the concept of neural networks")
        assert result.passed is True


class TestCoverageGuardrail:
    """Tests for the pre-generation context coverage guardrail."""

    def test_sufficient_overlap(self):
        from src.guardrails.coverage import CoverageGuardrail
        guard = CoverageGuardrail(threshold=0.15)
        # Query words: "what", "is", "corporation"
        # Context words contain: "corporation", "what", "is"
        result = guard.check("What is a corporation?", "A corporation is a legal entity that is separate from its owners.")
        assert result.passed is True

    def test_insufficient_overlap(self):
        from src.guardrails.coverage import CoverageGuardrail
        guard = CoverageGuardrail(threshold=0.20)
        # Query words: "rachel", "carson", "endure"
        # Context words: "apple", "banana", "orange"
        result = guard.check("Why did Rachel Carson write an obligation to endure?", "Apple banana orange are healthy fruits to eat daily.")
        assert result.passed is False
        assert result.reason == RefusalReason.UNGROUNDED

    def test_empty_inputs(self):
        from src.guardrails.coverage import CoverageGuardrail
        guard = CoverageGuardrail()
        assert guard.check("", "Some context").passed is False
        assert guard.check("Some query", "").passed is False

