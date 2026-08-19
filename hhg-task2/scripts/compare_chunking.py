"""
Compare chunking strategies side by side.

Runs the same set of queries against all chunking strategies and compares
retrieval quality and latency. Generates docs/CHUNKING_COMPARISON.md.

Usage:
    python scripts/compare_chunking.py --data data/msmarco_xi_train.parquet
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def compare_strategies(data_path: str, num_test_queries: int = 20):
    """Compare all chunking strategies on retrieval quality and speed."""
    import pandas as pd
    from dotenv import load_dotenv
    load_dotenv()

    from src.chunking.factory import list_strategies, get_chunker
    from src.embeddings.multilingual import EmbeddingService
    from src.retrieval.numpy_store import LocalNumpyStore

    print("[INFO] Initializing...")
    embedding_service = EmbeddingService()
    embedding_service.load_model()

    store = LocalNumpyStore()
    store.connect()

    df = pd.read_parquet(data_path)
    strategies = list_strategies()

    results = {}

    for strat in strategies:
        print(f"\n[INFO] Testing strategy: {strat}")
        namespace = strat

        # Check if namespace has vectors
        stats = store.get_stats()
        ns_stats = stats.get("namespaces", {}).get(namespace, {})
        vector_count = ns_stats.get("vector_count", 0)

        if vector_count == 0:
            print(f"   [WARNING] Namespace '{namespace}' is empty. Run ingest.py first.")
            continue

        print(f"   Vectors in namespace: {vector_count}")

        # Run test queries
        latencies = []
        scores = []

        # Sample queries from dataset
        query_col = None
        for field in ["query", "question"]:
            if field in df.columns:
                query_col = field
                break

        if query_col:
            test_queries = df[query_col].dropna().head(num_test_queries).tolist()
        else:
            test_queries = ["What is machine learning?", "How does NLP work?"] * (num_test_queries // 2)

        for query in test_queries:
            query_emb = embedding_service.encode_query(str(query))
            retrieval_result = store.query(query_emb, query_str=str(query), namespace=namespace)

            latencies.append(retrieval_result.duration_ms)
            if retrieval_result.chunks:
                scores.append(retrieval_result.chunks[0].score)

        results[strat] = {
            "vector_count": vector_count,
            "avg_latency_ms": round(np.mean(latencies), 1) if latencies else 0,
            "p50_latency_ms": round(np.percentile(latencies, 50), 1) if latencies else 0,
            "p100_latency_ms": round(max(latencies), 1) if latencies else 0,
            "avg_top_score": round(np.mean(scores), 4) if scores else 0,
            "min_top_score": round(min(scores), 4) if scores else 0,
        }

    # Generate report
    report_lines = [
        "# Chunking Strategy Comparison\n",
        f"**Test Queries:** {num_test_queries}\n",
        "\n## Results\n",
        "| Strategy | Vectors | Avg Latency | P50 Latency | P100 Latency | Avg Top Score | Min Top Score |",
        "|---|---|---|---|---|---|---|",
    ]

    for strat, stats in results.items():
        report_lines.append(
            f"| {strat} | {stats['vector_count']:,} | {stats['avg_latency_ms']}ms | "
            f"{stats['p50_latency_ms']}ms | {stats['p100_latency_ms']}ms | "
            f"{stats['avg_top_score']} | {stats['min_top_score']} |"
        )

    # Recommendation
    if results:
        best = max(results.items(), key=lambda x: x[1]["avg_top_score"])
        report_lines.extend([
            f"\n## Recommendation\n",
            f"**Winner:** `{best[0]}` — highest average relevance score ({best[1]['avg_top_score']})\n",
        ])

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    report_path = docs_dir / "CHUNKING_COMPARISON.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\n[REPORT] Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare chunking strategies")
    parser.add_argument("--data", default="data/msmarco_xi_train.parquet")
    parser.add_argument("--queries", type=int, default=20)
    args = parser.parse_args()

    compare_strategies(data_path=args.data, num_test_queries=args.queries)
