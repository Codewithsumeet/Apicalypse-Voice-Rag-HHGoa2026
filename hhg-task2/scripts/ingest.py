"""
Ingest dataset into the local NumPy store — chunk, embed, and upsert.

Usage:
    python scripts/ingest.py --strategy fixed
    python scripts/ingest.py --strategy semantic
    python scripts/ingest.py --strategy metadata_aware
    python scripts/ingest.py --strategy all
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DEMO_MAX_RECORDS = 10_000
CHECKPOINT_DIR = Path("data/checkpoints")


def _fingerprint(path: str, max_records: int) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    digest.update(str(file_path.resolve()).encode())
    digest.update(str(file_path.stat().st_size).encode())
    digest.update(str(file_path.stat().st_mtime_ns).encode())
    digest.update(str(max_records).encode())
    return digest.hexdigest()


def _write_checkpoint(path: Path, checkpoint: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _load_checkpoint(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot load checkpoint {path}: {error}") from error


def ingest(
    data_path: str,
    strategy: str,
    max_records: int | None = None,
    batch_size: int = 100,
    resume: bool = False,
    fresh: bool = False,
    smoke_test: bool = False,
):
    """Chunk, embed, and upsert dataset records into LocalNumpyStore."""
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv()

    from src.config import settings
    from src.chunking.factory import get_chunker, list_strategies
    from src.embeddings.multilingual import EmbeddingService
    from src.retrieval.numpy_store import LocalNumpyStore

    if smoke_test:
        max_records = 20
        if not resume:
            fresh = True
    if resume and fresh:
        raise ValueError("Use only one of --resume or --fresh.")
    if not resume and not fresh:
        raise ValueError("Choose --fresh for a new run or --resume to continue a checkpoint.")
    if max_records is not None and (max_records < 1 or max_records > DEMO_MAX_RECORDS):
        raise ValueError(
            f"Demo dataset limit must be between 1 and {DEMO_MAX_RECORDS:,} records; received {max_records:,}."
        )

    ingestion_start = time.perf_counter()
    print(f"[INFO] Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    if max_records is not None:
        df = df.head(max_records)
        print(f"   Using first {max_records} records")

    print(f"   Total records: {len(df)}")

    # Initialize services
    print("[INFO] Loading embedding model...")
    embedding_service = EmbeddingService()
    embedding_service.load_model()

    print("[INFO] Connecting to vector store...")
    store = LocalNumpyStore()
    store.connect()

    # Determine strategies to run
    strategies = list_strategies() if strategy == "all" else [strategy]

    ingestion_stats = []

    for strat_name in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strat_name}")
        print(f"{'='*60}")

        namespace = strat_name
        chunker = get_chunker(strat_name)
        checkpoint_path = CHECKPOINT_DIR / f"{namespace}.json"
        fingerprint = _fingerprint(data_path, len(df))
        if resume:
            checkpoint = _load_checkpoint(checkpoint_path)
            expected = {
                "dataset_fingerprint": fingerprint,
                "strategy": namespace,
                "max_records": len(df),
                "batch_size": batch_size,
                "embedding_model": settings.embedding_model,
            }
            mismatches = [key for key, value in expected.items() if checkpoint.get(key) != value]
            if mismatches:
                raise RuntimeError(f"Checkpoint is incompatible; mismatched fields: {', '.join(mismatches)}")
        else:
            store.delete_namespace(namespace)
            checkpoint = {
                "dataset_fingerprint": fingerprint,
                "strategy": namespace,
                "max_records": len(df),
                "batch_size": batch_size,
                "embedding_model": settings.embedding_model,
                "completed_document_ids": [],
                "failed_document_ids": [],
                "processed_count": 0,
                "chunk_count": 0,
                "timestamp": time.time(),
                "status": "running",
            }
            _write_checkpoint(checkpoint_path, checkpoint)
        completed = set(checkpoint.get("completed_document_ids", []))
        failed = set(checkpoint.get("failed_document_ids", []))

        # If semantic chunker, set the embedding model
        if hasattr(chunker, "set_embedding_model"):
            chunker.set_embedding_model(embedding_service.model)

        # Process bounded batches so the full corpus is never held as Python objects.
        batch_texts = []
        batch_metadatas = []
        source_documents = 0
        skipped_records = 0
        chunk_count = 0
        vector_count = 0
        start = time.perf_counter()

        batch_document_ids = []

        def mark_interrupted():
            checkpoint["status"] = "interrupted"
            checkpoint["timestamp"] = time.time()
            _write_checkpoint(checkpoint_path, checkpoint)
            print(f"[INTERRUPTED] Checkpoint saved: {checkpoint_path}")

        def flush_batch():
            nonlocal batch_texts, batch_metadatas, batch_document_ids, vector_count
            if not batch_texts:
                return
            try:
                embeddings = embedding_service.encode_batch(batch_texts, batch_size=64)
                vector_count += store.upsert_chunks(
                    texts=batch_texts,
                    embeddings=embeddings,
                    metadatas=batch_metadatas,
                    namespace=namespace,
                    batch_size=batch_size,
                    persist=False,
                )
                store.save()
            except KeyboardInterrupt:
                mark_interrupted()
                raise SystemExit(130)
            completed.update(batch_document_ids)
            checkpoint["completed_document_ids"] = sorted(completed)
            checkpoint["failed_document_ids"] = sorted(failed)
            checkpoint["processed_count"] = len(completed) + len(failed)
            checkpoint["chunk_count"] = checkpoint.get("chunk_count", 0) + len(batch_texts)
            checkpoint["timestamp"] = time.time()
            _write_checkpoint(checkpoint_path, checkpoint)
            elapsed = time.perf_counter() - start
            remaining = max(len(df) - checkpoint["processed_count"], 0)
            rate = checkpoint["processed_count"] / max(elapsed, 0.001)
            eta = remaining / max(rate, 0.001)
            print(f"{namespace} | docs {checkpoint['processed_count']}/{len(df)} | chunks {checkpoint['chunk_count']} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s")
            batch_texts = []
            batch_metadatas = []
            batch_document_ids = []

        for idx, row in df.iterrows():
            source_documents += 1
            document_id = str(row.get("query_id", idx))
            if document_id in completed:
                continue
            query = str(row.get("query", ""))
            query_id = document_id
            passages_dict = row.get("passages")
            
            if not isinstance(passages_dict, dict) and not hasattr(passages_dict, "items"):
                skipped_records += 1
                failed.add(document_id)
                continue
                
            translated_passages = passages_dict.get("Translated_passages", [])
            is_selected_list = passages_dict.get("is_selected", [])
            
            for p_idx, text in enumerate(translated_passages):
                text = str(text).strip()
                if not text:
                    continue
                
                is_selected = int(is_selected_list[p_idx]) if p_idx < len(is_selected_list) else 0
                doc_id = f"{query_id}_{p_idx}"
                
                try:
                    chunks = chunker.chunk(
                        text=text,
                        doc_id=doc_id,
                        metadata={
                            "query": query,
                            "query_id": query_id,
                            "passage_index": p_idx,
                            "is_selected": is_selected
                        }
                    )
                except KeyboardInterrupt:
                    mark_interrupted()
                    return
                except Exception as error:
                    failed.add(document_id)
                    print(f"[WARN] document {document_id} failed: {error}")
                    break
                
                for chunk in chunks:
                    batch_texts.append(chunk.text)
                    batch_metadatas.append({
                        "doc_id": chunk.source_doc_id,
                        "chunk_index": chunk.chunk_index,
                        "strategy": strat_name,
                        **chunk.metadata,
                    })
                    chunk_count += 1
                if document_id not in failed and document_id not in batch_document_ids:
                    batch_document_ids.append(document_id)
                if len(batch_document_ids) >= batch_size:
                    flush_batch()

        flush_batch()

        chunk_time = time.perf_counter() - start
        store.save()
        checkpoint["status"] = "complete"
        checkpoint["timestamp"] = time.time()
        _write_checkpoint(checkpoint_path, checkpoint)
        duration = time.perf_counter() - start
        namespace_stats = store.get_stats().get("namespaces", {}).get(namespace, {})
        print(f"   [SUCCESS] Processed {source_documents} records -> {chunk_count} chunks ({chunk_time:.1f}s)")
        print(f"   [SUCCESS] Persisted {vector_count} vectors to namespace '{namespace}'")
        ingestion_stats.append({
            "strategy": namespace,
            "source_documents": source_documents,
            "chunks": chunk_count,
            "vectors": vector_count,
            "skipped_records": skipped_records,
            "dimension": store.get_stats().get("dimension", 0),
            "namespace_total": namespace_stats.get("vector_count", 0),
            "duration_s": round(duration, 2),
        })

    print(f"\n[INFO] Ingestion complete in {time.perf_counter() - ingestion_start:.1f}s")
    for stats in ingestion_stats:
        print(f"[STATS] {stats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest dataset into LocalNumpyStore")
    parser.add_argument("--data", default="data/msmarco_xi_train.parquet", help="Path to parquet file")
    parser.add_argument("--strategy", default="fixed", choices=["fixed", "semantic", "metadata_aware", "all"])
    parser.add_argument(
        "--max-records",
        type=int,
        default=DEMO_MAX_RECORDS,
        help=f"Bounded demo record count (maximum {DEMO_MAX_RECORDS:,})",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Upsert batch size")
    parser.add_argument("--resume", action="store_true", help="Resume a compatible checkpoint")
    parser.add_argument("--fresh", action="store_true", help="Start a fresh namespace run")
    parser.add_argument("--smoke-test", action="store_true", help="Process exactly 20 documents")
    args = parser.parse_args()

    ingest(
        data_path=args.data,
        strategy=args.strategy,
        max_records=args.max_records,
        batch_size=args.batch_size,
        resume=args.resume,
        fresh=args.fresh,
        smoke_test=args.smoke_test,
    )
