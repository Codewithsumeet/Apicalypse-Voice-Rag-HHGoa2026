"""Fast grounded answer extraction from retrieved source text."""

import re

from src.retrieval.bm25 import tokenize


def extractive_answer(query: str, chunks: list) -> str:
    """Return up to two source sentences with the strongest query overlap."""
    if not chunks:
        return ""

    query_terms = set(tokenize(query))
    ranked = sorted(
        chunks,
        key=lambda chunk: len(query_terms.intersection(tokenize(chunk.text))),
        reverse=True,
    )
    source_text = ranked[0].text.strip()
    if not source_text:
        return ""

    sentences = [part.strip() for part in re.split(r"(?<=[.!?।])\s+", source_text) if part.strip()]
    return " ".join(sentences[:2])[:600] or source_text[:600]