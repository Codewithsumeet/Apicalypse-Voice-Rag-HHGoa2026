"""
Unsafe Input Detection Guardrail.

Detects potentially harmful, toxic, or inappropriate queries using:
1. A keyword/pattern blocklist (fast, zero-latency)
2. Optional LLM-based toxicity check (if extra safety needed)
"""

import re
import time
import structlog

from src.guardrails.models import GuardrailResult, RefusalReason

logger = structlog.get_logger(__name__)

# Blocklist patterns — add more as needed
UNSAFE_PATTERNS = [
    r"\b(hack|exploit|crack|breach)\b.*\b(password|system|server|bank|account)\b",
    r"\b(how\s+to|ways?\s+to)\b.*\b(kill|murder|harm|attack|bomb|weapon)\b",
    r"\b(make|build|create)\b.*\b(bomb|explosive|weapon|drug|meth)\b",
    r"\b(steal|fraud|scam|phish)\b",
    r"\b(child|minor)\b.*\b(abuse|exploit|porn)\b",
    r"\b(suicide|self[- ]harm)\b.*\b(how|method|way)\b",
]

# Compiled patterns for performance
_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


class UnsafeInputGuardrail:
    """Detects unsafe/inappropriate input using keyword patterns."""

    def check(self, text: str) -> GuardrailResult:
        """
        Check if input text contains unsafe/inappropriate content.

        Args:
            text: The query or transcript text to check.

        Returns:
            GuardrailResult — passed=True if safe, False if unsafe.
        """
        start = time.perf_counter()

        if not text or not text.strip():
            duration_ms = (time.perf_counter() - start) * 1000
            return GuardrailResult.pass_result(duration_ms=round(duration_ms, 2))

        # Check against compiled patterns
        for pattern in _compiled_patterns:
            if pattern.search(text):
                duration_ms = (time.perf_counter() - start) * 1000
                logger.warning("unsafe_input_detected", pattern=pattern.pattern)
                return GuardrailResult.refuse(
                    reason=RefusalReason.UNSAFE,
                    message="I'm unable to process this request as it appears to contain inappropriate or potentially harmful content.",
                    confidence=0.9,
                    duration_ms=round(duration_ms, 2),
                )

        duration_ms = (time.perf_counter() - start) * 1000
        return GuardrailResult.pass_result(duration_ms=round(duration_ms, 2))
