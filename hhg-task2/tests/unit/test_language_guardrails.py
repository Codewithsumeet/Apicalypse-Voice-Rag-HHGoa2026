"""
Unit tests for Language Detection, Language Consistency Guardrail, and Answerability Guardrail.
"""

import pytest
from src.utils.language import detect_language, compute_answerability, QueryObject, normalize_language
from src.guardrails.language_consistency import LanguageConsistencyGuardrail
from src.guardrails.answerability import AnswerabilityGuardrail
from src.guardrails.models import RefusalReason
from src.generation.extractive import extractive_answer
from src.retrieval.models import RetrievedChunk


class TestLanguageDetection:
    def test_provider_language_codes_are_normalized(self):
        assert normalize_language("eng") == "en"
        assert normalize_language("hin") == "hi"
        assert normalize_language("guj") == "gu"

    def test_english_detection(self):
        assert detect_language("What is machine learning?") == "en"
        assert detect_language("How do neural networks learn?") == "en"
        assert detect_language("What is the capital of France?") == "en"

    def test_hindi_detection(self):
        assert detect_language("मशीन लर्निंग क्या है?") == "hi"
        assert detect_language("तंत्रिका नेटवर्क कैसे सीखते हैं?") == "hi"

    def test_gujarati_detection(self):
        assert detect_language("મશીન લર્નિંગ શું છે?") == "gu"
        assert detect_language("ગોવા ક્યાં આવેલું છે?") == "gu"

    def test_mixed_script_detection(self):
        assert detect_language("Goa ક્યાં છે?") == "gu"
        assert detect_language("Machine Learning શું છે?") == "gu"


class TestLanguageConsistencyGuardrail:
    def test_matching_languages_pass(self):
        guard = LanguageConsistencyGuardrail()
        # English query with English evidence
        res_en = guard.validate("What is AI?", "Artificial intelligence is a branch of computer science.")
        assert res_en.passed is True

        # Hindi query with Hindi evidence
        res_hi = guard.validate("मशीन लर्निंग क्या है?", "मशीन लर्निंग कृत्रिम बुद्धिमत्ता की एक शाखा है।")
        assert res_hi.passed is True

    def test_english_query_hindi_evidence_refused(self):
        guard = LanguageConsistencyGuardrail()
        res = guard.validate("What is machine learning?", "मशीन लर्निंग कृत्रिम बुद्धिमत्ता की एक शाखा है।")
        assert res.passed is False
        assert res.reason == RefusalReason.UNGROUNDED


class TestAnswerabilityGuardrail:
    def test_answerable_passage_passes(self):
        guard = AnswerabilityGuardrail(min_answerability=0.40)
        query = "What is machine learning?"
        passage = "Machine learning is a subset of artificial intelligence that allows algorithms to learn from data."
        res = guard.validate(query, passage)
        assert res.passed is True

    def test_unanswerable_passage_refused(self):
        guard = AnswerabilityGuardrail(min_answerability=0.40)
        query = "What is integration by parts in calculus?"
        passage = "Corporate integration combines multiple business departments into a single structure."
        res = guard.validate(query, passage)
        assert res.passed is False
        assert res.reason == RefusalReason.UNGROUNDED

    def test_interchangeable_parts_are_not_integration_by_parts(self):
        guard = AnswerabilityGuardrail(min_answerability=0.40)
        res = guard.validate(
            "What is integration by parts?",
            "Interchangeable parts are identical components used in assembly.",
        )
        assert res.passed is False

    def test_historical_versailles_text_is_not_current_france_capital(self):
        guard = AnswerabilityGuardrail(min_answerability=0.40)
        res = guard.validate(
            "What is the capital of France?",
            "Versailles was made the capital of France during the reign of Louis XIV.",
        )
        assert res.passed is False


class TestExtractiveAnswer:
    def test_skips_question_fragment_and_selects_source_sentence(self):
        chunks = [
            RetrievedChunk(
                text="What is deep learning? Machine learning algorithms learn from data.",
                score=0.8,
                doc_id="doc-1",
            )
        ]
        answer = extractive_answer("What is machine learning?", chunks)
        assert answer == "Machine learning algorithms learn from data."
