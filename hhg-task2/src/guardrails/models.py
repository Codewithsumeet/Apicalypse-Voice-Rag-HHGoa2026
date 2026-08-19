"""
Pydantic models for the guardrails module.
"""

from enum import Enum
from pydantic import BaseModel, Field


class RefusalReason(str, Enum):
    """Structured refusal reason codes."""

    PASS = "PASS"
    OFF_TOPIC = "OFF_TOPIC"
    UNSAFE = "UNSAFE"
    UNGROUNDED = "UNGROUNDED"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class GuardrailResult(BaseModel):
    """Structured output from guardrail checks."""

    passed: bool = Field(..., description="Whether the input/output passed all guardrails")
    reason: RefusalReason = Field(default=RefusalReason.PASS, description="Refusal reason if failed")
    message: str = Field(default="", description="Human-readable explanation for the refusal")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in the guardrail decision")
    duration_ms: float = Field(default=0.0, description="Guardrail check latency")

    @staticmethod
    def pass_result(duration_ms: float = 0.0) -> "GuardrailResult":
        """Factory for a passing result."""
        return GuardrailResult(passed=True, reason=RefusalReason.PASS, duration_ms=duration_ms)

    @staticmethod
    def refuse(reason: RefusalReason, message: str, confidence: float = 1.0, duration_ms: float = 0.0) -> "GuardrailResult":
        """Factory for a refusal result."""
        return GuardrailResult(
            passed=False,
            reason=reason,
            message=message,
            confidence=confidence,
            duration_ms=duration_ms,
        )
