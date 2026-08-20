# Latency Measurement & Forensic Analysis

> **Purpose:** Detailed latency accounting, stage timing methodology, and benchmark results for APIcalypse Voice RAG.

---

## 1. Executive Summary

- **Measured RAG Latency (Local CPU):** **`21.62 – 24.80 ms` (P50)** / **`26.67 ms` (P95)** / **`31.33 ms` (P100 Max)**
- **Budget Ceiling:** `< 200 ms` (Target `< 100 ms`)
- **Execution Mode:** Extractive fast-path with single-pass PyTorch CPU embedding and in-memory NumPy cosine similarity matrix.

---

## 2. Why RAG Latency Excludes External Network / STT

```
[Total User Turnaround: ~1.2 – 1.8 s]
   │
   ├── 1. Browser WebRTC Audio Capture: ~200 – 400 ms
   ├── 2. Network Transit to ElevenLabs API: ~150 – 300 ms
   ├── 3. ElevenLabs Speech-to-Text Transcription: ~800 – 1300 ms
   │
   └── 4. RAG PIPELINE (APIcalypse Core Engine): ~22 – 25 ms ◄── MEASURED RAG BUDGET
```

The competition RAG budget specifically evaluates the retrieval, guardrails, and answer generation harness. Speech-to-text latency is dominated by third-party external API transit.

---

## 3. Controlled Benchmark Results (N = 100 Runs)

| Stage | P50 (ms) | P70 (ms) | P95 (ms) | P100 (ms) | Operational Details |
| :--- | :---:| :---:| :---:| :---:| :--- |
| **Query Embedding** | **10.59** | 11.40 | 13.67 | 16.50 | `paraphrase-multilingual-MiniLM-L12-v2` via PyTorch CPU thread scheduling (`torch.set_num_threads`) |
| **Vector Retrieval + Reranking** | **10.73** | 11.33 | 12.57 | 13.80 | In-memory NumPy cosine matrix over 15,679 chunks + 2-stage language-aware candidate reranking |
| **Pre-Retrieval Guardrails** | **0.06** | 0.07 | 0.08 | 0.11 | Compiled multi-pattern regex (`UnsafeInputGuardrail`) + dataset centroid cosine check |
| **Post-Retrieval Guardrails** | **0.28** | 0.32 | 0.35 | 0.43 | `AnswerabilityGuardrail` + `LanguageConsistencyGuardrail` + exact-source substring `GroundingGuardrail` |
| **Answer Generation** | **0.00** | 0.00 | 0.00 | 0.00 | Extractive fast-path (sentence selection directly from retrieved evidence with zero LLM lag) |
| **TOTAL RAG PIPELINE** | **`22.76`** | **`23.84`** | **`26.67`** | **`31.33`** | **Sub-35 ms across English, Hindi, and Gujarati queries** |

---

## 4. Engineering Latency Optimizations

1. **PyTorch CPU Thread Allocation:** Configured `torch.set_num_threads(min(12, max(2, num_cores)))` and added graph warmup at startup to eliminate CPU thread scheduling stalls.
2. **Exact-Source Grounding Fast-Path:** If the extracted answer is an attested substring of the retrieved context, `GroundingGuardrail` verifies containment in `0.001 ms` instead of computing two 2,000-character CPU embeddings (~400 ms).
3. **Stage Timer Isolation:** Separated `pre_guard_ms` from `embedding_ms` in `pipeline.py` to prevent double-counting inference time inside guardrail metrics.
4. **Deterministic Memoization:** Added `@functools.lru_cache` to `classify_question`, `normalize_language`, and `detect_language`.

---

## 5. Reproduction

To reproduce these measurements locally:

```bash
# Run the live API benchmark on localhost:8000
python scratch/test_live_api.py

# Run the standalone pipeline stabilization benchmark
python scratch/test_stabilization.py
```
