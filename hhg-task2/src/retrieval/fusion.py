"""
Reciprocal Rank Fusion (RRF).

Combines ranks from dense (vector) retrieval and sparse (keyword/BM25) retrieval
to produce a robust hybrid rank score.
"""


def reciprocal_rank_fusion(
    dense_indices: list[int],
    sparse_indices: list[int],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """
    Compute RRF score for documents across dense and sparse ranked lists.

    Formula: RRF_Score(doc) = Sum(1 / (k + rank_i(doc)))

    Args:
        dense_indices: Ordered list of document indices from dense vector search.
        sparse_indices: Ordered list of document indices from sparse keyword search.
        k: Constant parameter (typically 60) to regulate the impact of top ranks.
        top_k: Number of fused results to return.

    Returns:
        List of (corpus_index, rrf_score) sorted descending by RRF score.
    """
    rrf_scores: dict[int, float] = {}

    # Dense ranks
    for rank, idx in enumerate(dense_indices, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

    # Sparse ranks
    for rank, idx in enumerate(sparse_indices, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

    # Sort descending by score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results[:top_k]
