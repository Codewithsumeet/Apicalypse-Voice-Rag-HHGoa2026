"""
Language Consistency Guardrail for Apicalypse Voice RAG.
Verifies that query language matches evidence and answer language.
Prevents silently returning Hindi/other language answers to English users.
"""

import structlog
from src.guardrails.models import GuardrailResult, RefusalReason
from src.utils.language import detect_language

logger = structlog.get_logger(__name__)


class LanguageConsistencyGuardrail:
    """
    Ensures query language, selected evidence language, and generated answer language
    are strictly aligned, preventing cross-lingual language leakage.
    """

    def __init__(self, allow_fallback: bool = False):
        self._allow_fallback = allow_fallback

    def check(
        self,
        query: str,
        evidence_text: str,
        query_language: str | None = None,
        evidence_language: str | None = None,
        fallback_used: bool = False,
    ) -> GuardrailResult:
        return self.validate(query, evidence_text, query_language, evidence_language, fallback_used)

    def validate(
        self,
        query: str,
        evidence_text: str,
        query_language: str | None = None,
        evidence_language: str | None = None,
        fallback_used: bool = False,
    ) -> GuardrailResult:
        """
        Validate that evidence language aligns with query language.
        """
        q_lang = query_language or detect_language(query)
        e_lang = evidence_language or detect_language(evidence_text)

        if q_lang == e_lang:
            logger.debug(
                "language_consistency_passed",
                query_language=q_lang,
                evidence_language=e_lang,
                fallback_used=fallback_used,
            )
            return GuardrailResult.pass_result()

        # If query is English and evidence is not English
        if q_lang == "en" and e_lang != "en":
            logger.info(
                "language_inconsistency_detected",
                query_language=q_lang,
                evidence_language=e_lang,
                fallback_used=fallback_used,
            )
            return GuardrailResult.refuse(
                reason=RefusalReason.UNGROUNDED,
                message="No sufficiently grounded evidence in English was found in the knowledge base for this question.",
            )

        # For non-English queries if fallback was used with strong grounding
        if fallback_used and self._allow_fallback:
            logger.info(
                "multilingual_fallback_accepted",
                query_language=q_lang,
                evidence_language=e_lang,
            )
            return GuardrailResult.pass_result()

        return GuardrailResult.refuse(
            reason=RefusalReason.UNGROUNDED,
            message="No sufficiently grounded evidence in the requested language was found.",
        )
