"""
Grounding / Hallucination Check Guardrail.

Verifies that the generated answer is actually grounded in the retrieved context.
Uses embedding similarity between answer and context as a lightweight check
(faster than NLI model, ~5ms vs ~30ms).
"""

import time
import numpy as np
import structlog

from src.guardrails.models import GuardrailResult, RefusalReason
from src.config import settings

logger = structlog.get_logger(__name__)


class GroundingGuardrail:
    """Checks if generated answer is grounded in retrieved context."""

    def __init__(self, embedding_service=None, threshold: float | None = None):
        """
        Args:
            embedding_service: EmbeddingService for encoding answer and context.
            threshold: Similarity threshold (below = ungrounded).
        """
        self._embedding_service = embedding_service
        self._threshold = threshold or settings.grounding_threshold

    def check(self, answer: str, context: str) -> GuardrailResult:
        """
        Check if the answer is grounded in the context.

        Uses cosine similarity between answer embedding and context embedding.
        If similarity is below threshold, the answer is likely hallucinated.

        Args:
            answer: The generated answer text.
            context: The retrieved context that was provided to the LLM.

        Returns:
            GuardrailResult — passed=True if grounded, False if ungrounded.
        """
        start = time.perf_counter()

        if not answer or not context:
            duration_ms = (time.perf_counter() - start) * 1000
            return GuardrailResult.refuse(
                reason=RefusalReason.UNGROUNDED,
                message="I couldn't find sufficient information to provide a reliable answer.",
                duration_ms=round(duration_ms, 2),
            )

        if self._embedding_service is None:
            # If no embedding service, skip the check (fail-open)
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning("grounding_check_skipped_no_embedding_service")
            return GuardrailResult.pass_result(duration_ms=round(duration_ms, 2))

        # Encode answer and context
        answer_embedding = np.array(self._embedding_service.encode_query(answer))
        context_embedding = np.array(self._embedding_service.encode_query(context[:2000]))  # Truncate for speed

        # Cosine similarity
        similarity = float(
            np.dot(answer_embedding, context_embedding)
            / (np.linalg.norm(answer_embedding) * np.linalg.norm(context_embedding) + 1e-10)
        )

        duration_ms = (time.perf_counter() - start) * 1000

        if similarity < self._threshold:
            logger.warning(
                "ungrounded_answer_detected",
                similarity=round(similarity, 4),
                threshold=self._threshold,
            )
            return GuardrailResult.refuse(
                reason=RefusalReason.UNGROUNDED,
                message="I couldn't verify my answer against the available data. Let me know if you'd like to try rephrasing your question.",
                confidence=max(0.0, min(1.0, 1.0 - similarity)),
                duration_ms=round(duration_ms, 2),
            )

        logger.debug(
            "grounding_check_passed",
            similarity=round(similarity, 4),
            duration_ms=round(duration_ms, 2),
        )

        return GuardrailResult.pass_result(duration_ms=round(duration_ms, 2))
