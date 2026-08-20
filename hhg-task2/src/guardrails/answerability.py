"""
Answerability Guardrail for Apicalypse Voice RAG.
Verifies that the retrieved passage contains specific evidence to answer the user query,
distinguishing topic similarity from question answerability.
"""

import structlog
from src.guardrails.models import GuardrailResult, RefusalReason
from src.utils.language import compute_answerability, detect_language

logger = structlog.get_logger(__name__)


class AnswerabilityGuardrail:
    """
    Evaluates whether the selected evidence contains answer-bearing information
    for the specific question asked, rejecting vague or tangential matches.
    """

    def __init__(self, min_answerability: float = 0.40):
        self._min_answerability = min_answerability

    def check(
        self,
        query: str,
        evidence_text: str,
        language: str | None = None,
        query_language: str | None = None,
    ) -> GuardrailResult:
        return self.validate(query, evidence_text, language=language or query_language)

    def validate(
        self,
        query: str,
        evidence_text: str,
        language: str | None = None,
        query_language: str | None = None,
    ) -> GuardrailResult:
        """
        Validate question answerability against retrieved evidence.
        """
        lang = language or query_language or detect_language(query)
        score = compute_answerability(query, evidence_text, lang)

        if score >= self._min_answerability:
            logger.debug(
                "answerability_passed",
                query=query,
                score=score,
                min_threshold=self._min_answerability,
            )
            return GuardrailResult.pass_result()

        logger.info(
            "insufficient_answerability_detected",
            query=query,
            score=score,
            min_threshold=self._min_answerability,
        )
        return GuardrailResult.refuse(
            reason=RefusalReason.UNGROUNDED,
            message="No sufficiently answerable evidence was found in the knowledge base for this specific question.",
        )
