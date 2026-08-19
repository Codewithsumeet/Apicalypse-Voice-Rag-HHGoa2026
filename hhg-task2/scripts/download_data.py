"""
Download the ai4bharat/MSMARCO-XI dataset from HuggingFace.

Usage:
    python scripts/download_data.py [--split train] [--output data/]
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DEMO_MAX_RECORDS = 10_000


def download_dataset(split: str = "train", output_dir: str = "data", limit: int = 10000):
    """Download hinval.parquet (461MB) directly, slice to limit, and clean up."""
    if limit < 1 or limit > DEMO_MAX_RECORDS:
        raise ValueError(
            f"Demo dataset limit must be between 1 and {DEMO_MAX_RECORDS:,} records; received {limit:,}."
        )
    import httpx
    import pandas as pd
    import os

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    temp_path = output_path / "temp.parquet"
    final_path = output_path / f"msmarco_xi_{split}.parquet"

    # We use validation split as the source since it's 461MB vs 3.7GB train split
    url = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/hinval.parquet"

    print(f"[INFO] Downloading validation split for slicing (limit={limit})...")
    print(f"   Source: {url}")
    print(f"   Temp File: {temp_path}")

    try:
        # Stream download with progress indicator
        with httpx.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            
            with open(temp_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):  # 1MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"   Downloaded: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({pct:.1f}%)", end="\r")
            print("\n   [SUCCESS] Download complete.")

        # Load and slice
        print(f"   Loading parquet file into pandas...")
        df = pd.read_parquet(temp_path)
        print(f"   Total records in source: {len(df)}")

        # Slice and save
        sliced_df = df.head(limit) if limit > 0 else df
        sliced_df.to_parquet(final_path)
        print(f"[SUCCESS] Sliced to {len(sliced_df)} records and saved to {final_path}")

        # Clean up temp file
        if temp_path.exists():
            os.remove(temp_path)
            print("   [CLEANUP] Temporary file removed.")

        # Save sample for inspection
        sample_path = output_path / f"msmarco_xi_{split}_sample.jsonl"
        sliced_df.head(100).to_json(sample_path, orient="records", lines=True)
        print(f"[SAVED] Sample (100 rows) saved to {sample_path}")

    except Exception as e:
        print(f"[ERROR] Download/slice failed: {e}")
        if temp_path.exists():
            os.remove(temp_path)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MSMARCO-XI dataset")
    parser.add_argument("--split", default="train", help="Dataset split to download")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEMO_MAX_RECORDS,
        help=f"Number of records to download (maximum {DEMO_MAX_RECORDS:,})",
    )
    args = parser.parse_args()

    download_dataset(split=args.split, output_dir=args.output, limit=args.limit)


