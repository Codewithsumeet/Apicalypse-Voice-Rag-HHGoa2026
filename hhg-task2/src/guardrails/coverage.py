"""
Coverage Guardrail.

A pre-generation guardrail that checks whether the retrieved context chunks
contain sufficient keyword/token overlap with the user's query.
Prevents sending unanswerable queries to the LLM.
"""

import time
import structlog
from src.guardrails.models import GuardrailResult, RefusalReason
from src.retrieval.bm25 import tokenize

logger = structlog.get_logger(__name__)


class CoverageGuardrail:
    """Pre-generation check ensuring context has sufficient overlap with query keywords."""

    def __init__(self, threshold: float = 0.15):
        self._threshold = threshold

    def check(self, query: str, context: str) -> GuardrailResult:
        """
        Check if the retrieved context covers the query terms.

        Args:
            query: User's search query string.
            context: Retrieved context text concatenated.

        Returns:
            GuardrailResult — passed=True if overlap is above threshold.
        """
        start = time.perf_counter()

        if not query or not context:
            duration_ms = (time.perf_counter() - start) * 1000
            return GuardrailResult.refuse(
                reason=RefusalReason.UNGROUNDED,
                message="I couldn't find sufficient information in our database to answer your question.",
                duration_ms=round(duration_ms, 2),
            )

        query_tokens = tokenize(query)
        context_tokens = set(tokenize(context))

        if not query_tokens:
            duration_ms = (time.perf_counter() - start) * 1000
            return GuardrailResult.pass_result(duration_ms=round(duration_ms, 2))

        # Check how many unique query tokens are present in context
        unique_query_tokens = set(query_tokens)
        matches = [t for t in unique_query_tokens if t in context_tokens]

        overlap_ratio = len(matches) / len(unique_query_tokens)
        duration_ms = (time.perf_counter() - start) * 1000

        if overlap_ratio < self._threshold:
            logger.info(
                "insufficient_context_coverage",
                overlap_ratio=round(overlap_ratio, 4),
                threshold=self._threshold,
                query=query[:100],
            )
            return GuardrailResult.refuse(
                reason=RefusalReason.UNGROUNDED,
                message="The retrieved context does not contain enough information about your request.",
                confidence=float(max(0.0, min(1.0, 1.0 - overlap_ratio))),
                duration_ms=round(duration_ms, 2),
            )

        logger.debug(
            "context_coverage_passed",
            overlap_ratio=round(overlap_ratio, 4),
            duration_ms=round(duration_ms, 2),
        )
        return GuardrailResult.pass_result(duration_ms=round(duration_ms, 2))
