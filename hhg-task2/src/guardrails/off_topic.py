"""
Off-Topic Detection Guardrail.

Checks whether the user's query is relevant to the dataset by comparing the
query embedding against a precomputed dataset centroid (average embedding).
If cosine similarity is below threshold, the query is considered off-topic.
"""

import time
import numpy as np
import structlog

from src.guardrails.models import GuardrailResult, RefusalReason
from src.config import settings

logger = structlog.get_logger(__name__)


class OffTopicGuardrail:
    """Detects off-topic queries by comparing to dataset embedding centroid."""

    def __init__(self, embedding_service=None, threshold: float | None = None):
        """
        Args:
            embedding_service: EmbeddingService instance for encoding queries.
            threshold: Cosine similarity threshold (below = off-topic).
        """
        self._embedding_service = embedding_service
        self._threshold = threshold or settings.off_topic_threshold
        self._centroid: np.ndarray | None = None

    def set_centroid(self, centroid: np.ndarray | list[float]):
        """Set the dataset centroid embedding (precomputed during ingestion)."""
        if isinstance(centroid, list):
            centroid = np.array(centroid)
        self._centroid = centroid

    def compute_centroid(self, sample_embeddings: list[list[float]]):
        """Compute centroid from a sample of dataset embeddings."""
        self._centroid = np.mean(np.array(sample_embeddings), axis=0)
        logger.info("off_topic_centroid_computed", num_samples=len(sample_embeddings))

    def check(self, query_embedding: list[float]) -> GuardrailResult:
        """
        Check if a query is off-topic.

        Args:
            query_embedding: The query's embedding vector.

        Returns:
            GuardrailResult — passed=True if on-topic, False if off-topic.
        """
        start = time.perf_counter()

        if self._centroid is None:
            logger.warning("off_topic_guardrail_no_centroid")
            duration_ms = (time.perf_counter() - start) * 1000
            return GuardrailResult.pass_result(duration_ms=round(duration_ms, 2))

        query_vec = np.array(query_embedding)
        similarity = float(
            np.dot(query_vec, self._centroid)
            / (np.linalg.norm(query_vec) * np.linalg.norm(self._centroid) + 1e-10)
        )

        duration_ms = (time.perf_counter() - start) * 1000

        if similarity < self._threshold:
            logger.info(
                "off_topic_detected",
                similarity=round(similarity, 4),
                threshold=self._threshold,
            )
            return GuardrailResult.refuse(
                reason=RefusalReason.OFF_TOPIC,
                message="Your question doesn't appear to be related to the topics covered in our knowledge base. Please ask a question related to the available data.",
                confidence=max(0.0, min(1.0, 1.0 - similarity)),
                duration_ms=round(duration_ms, 2),
            )

        return GuardrailResult.pass_result(duration_ms=round(duration_ms, 2))
