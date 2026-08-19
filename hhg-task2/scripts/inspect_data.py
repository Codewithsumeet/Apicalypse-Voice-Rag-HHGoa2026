"""
Inspect the MSMARCO-XI dataset — schema analysis and statistics.

Generates docs/DATASET_ANALYSIS.md with:
- Row count, column names and types
- Passage length distribution (mean, median, P90, max)
- Language samples
- Recommended chunk size ranges

Usage:
    python scripts/inspect_data.py [--data data/msmarco_xi_train.parquet]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def inspect_dataset(data_path: str):
    """Analyze dataset schema and write report."""
    import pandas as pd
    import numpy as np

    print(f"[INFO] Inspecting dataset: {data_path}")

    df = pd.read_parquet(data_path)

    print(f"[SUCCESS] Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")

    # Identify text columns
    text_cols = [col for col in df.columns if df[col].dtype == "object"]

    # Passage length analysis
    report_lines = [
        "# Dataset Analysis — ai4bharat/MSMARCO-XI\n",
        f"**Total Records:** {len(df):,}\n",
        f"**Columns:** {', '.join(df.columns)}\n",
        "\n## Column Types\n",
        "| Column | Type | Non-Null | Unique |",
        "|---|---|---|---|",
    ]

    for col in df.columns:
        non_null = df[col].notna().sum()
        try:
            unique = df[col].nunique()
        except Exception:
            unique = "N/A"
        report_lines.append(f"| {col} | {df[col].dtype} | {non_null:,} | {unique} |")

    report_lines.append("\n## Text Length Statistics\n")

    for col in text_cols:
        if df[col].notna().any():
            lengths = df[col].dropna().str.len()
            if len(lengths) > 0:
                report_lines.extend([
                    f"### `{col}`\n",
                    f"| Metric | Characters |",
                    f"|---|---|",
                    f"| Mean | {lengths.mean():.0f} |",
                    f"| Median (P50) | {lengths.median():.0f} |",
                    f"| P75 | {np.percentile(lengths, 75):.0f} |",
                    f"| P90 | {np.percentile(lengths, 90):.0f} |",
                    f"| P95 | {np.percentile(lengths, 95):.0f} |",
                    f"| Max | {lengths.max():.0f} |",
                    f"| Min | {lengths.min():.0f} |",
                    "",
                ])

    # Sample records
    report_lines.append("\n## Sample Records (First 5)\n")
    for i, row in df.head(5).iterrows():
        report_lines.append(f"### Record {i + 1}\n")
        for col in df.columns:
            val = str(row[col])[:200]
            report_lines.append(f"- **{col}:** {val}")
        report_lines.append("")

    # Chunk size recommendations
    report_lines.extend([
        "\n## Recommended Chunk Sizes\n",
        "Based on passage length distribution:\n",
    ])

    for col in text_cols:
        if df[col].notna().any():
            lengths = df[col].dropna().str.len()
            median = lengths.median()
            p90 = np.percentile(lengths, 90)
            report_lines.extend([
                f"**For `{col}`:**",
                f"- If most passages are short (median={median:.0f} chars): chunk_size=256-512",
                f"- If passages vary widely (P90={p90:.0f} chars): chunk_size=512-768 with 20% overlap",
                f"- Overlap recommendation: {max(50, int(median * 0.2))} chars",
                "",
            ])

    # Write report
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    report_path = docs_dir / "DATASET_ANALYSIS.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"[REPORT] Report written to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect MSMARCO-XI dataset")
    parser.add_argument("--data", default="data/msmarco_xi_train.parquet", help="Path to parquet file")
    args = parser.parse_args()

    inspect_dataset(data_path=args.data)
