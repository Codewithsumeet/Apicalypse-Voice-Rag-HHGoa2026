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

ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

HINDI_STOP_WORDS = {
    "है", "हैं", "हो", "था", "थे", "थी", "का", "के", "की", "को", "में", "से", "पर", "ने",
    "और", "या", "यह", "वह", "ये", "वे", "एक", "तो", "भी", "ही", "कि", "जो", "कर", "किया",
    "गया", "गए", "गई", "दिया", "दिए", "हुए", "हुआ", "हुई", "द्वारा", "लिए", "अपने", "अपनी"
}


class CoverageGuardrail:
    """Pre-generation check ensuring context has sufficient overlap with query keywords."""

    def __init__(self, threshold: float = 0.15, semantic_threshold: float = 0.40):
        self._threshold = threshold
        self._semantic_threshold = semantic_threshold

    def check(self, query: str, context: str, semantic_score: float | None = None) -> GuardrailResult:
        """
        Check if the retrieved context covers the query terms or is semantically grounded.

        Args:
            query: User's search query string.
            context: Retrieved context text concatenated.
            semantic_score: Optional dense cosine similarity score between query and top retrieved chunk.

        Returns:
            GuardrailResult — passed=True if overlap or semantic grounding is above threshold.
        """
        start = time.perf_counter()

        if not query or not context:
            duration_ms = (time.perf_counter() - start) * 1000
            return GuardrailResult.refuse(
                reason=RefusalReason.UNGROUNDED,
                message="I couldn't find sufficient information in our database to answer your question.",
                duration_ms=round(duration_ms, 2),
            )

        # If strong semantic similarity exists across languages, coverage is validated semantically
        if semantic_score is not None and semantic_score >= self._semantic_threshold:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                "context_coverage_passed_semantic",
                semantic_score=round(semantic_score, 4),
                threshold=self._semantic_threshold,
                duration_ms=round(duration_ms, 2),
            )
            return GuardrailResult.pass_result(duration_ms=round(duration_ms, 2))

        query_tokens = tokenize(query)
        context_tokens = set(tokenize(context))

        if not query_tokens:
            duration_ms = (time.perf_counter() - start) * 1000
            return GuardrailResult.pass_result(duration_ms=round(duration_ms, 2))

        # Filter stop words to prevent single stop-word matches from passing
        content_tokens = [
            t for t in set(query_tokens)
            if t not in ENGLISH_STOP_WORDS and t not in HINDI_STOP_WORDS
        ]

        # If query consists solely of stop words and lacks semantic score, refuse
        if not content_tokens:
            duration_ms = (time.perf_counter() - start) * 1000
            return GuardrailResult.refuse(
                reason=RefusalReason.UNGROUNDED,
                message="The retrieved context does not contain enough information about your request.",
                confidence=1.0,
                duration_ms=round(duration_ms, 2),
            )

        matches = [t for t in content_tokens if t in context_tokens]
        overlap_ratio = len(matches) / len(content_tokens)
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
            "context_coverage_passed_lexical",
            overlap_ratio=round(overlap_ratio, 4),
            duration_ms=round(duration_ms, 2),
        )
        return GuardrailResult.pass_result(duration_ms=round(duration_ms, 2))
