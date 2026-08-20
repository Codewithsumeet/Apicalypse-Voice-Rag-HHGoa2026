"""Fast grounded answer extraction from retrieved source text."""

import re

from src.retrieval.bm25 import tokenize


def extractive_answer(query: str, chunks: list) -> str:
    """Return up to two relevant source sentences without inventing text."""
    if not chunks:
        return ""

    query_terms = set(tokenize(query))
    candidates = []
    selected_chunk = chunks[0]
    for sentence_rank, sentence in enumerate(
        part.strip() for part in re.split(r"(?<=[.!?।])\s+", selected_chunk.text.strip()) if part.strip()
    ):
        if sentence.endswith(("?", "؟")):
            continue
        overlap = len(query_terms.intersection(tokenize(sentence)))
        candidates.append((overlap, -sentence_rank, sentence))

    if not candidates:
        return ""

    candidates.sort(reverse=True)
    return " ".join(item[2] for item in candidates[:2])[:600]