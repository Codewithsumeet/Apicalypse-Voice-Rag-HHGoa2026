# Latency Benchmark Report

## Current Checkout Status (2026-08-19)

The current bounded demo corpus contains 10,000 source records and 99,985 passages. The preserved fixed namespace contains 108,350 vectors with 384-dimensional embeddings in a roughly 470 MB `data/numpy_store.pkl` file. No full MSMARCO-XI source ingestion was performed.

The requested 30-query baseline was run with the project virtual environment and the active `LocalNumpyStore`.

| Metric | Result |
|---|---:|
| Queries run | 30 |
| Successful RAG queries | 0 |
| Refused queries | 30 |
| Failed queries | 0 |
| Startup/model load | 4,411.83 ms |
| Warm embedding refusals | approximately 10-19 ms |
| Local vectors/namespaces | 108,350 / fixed |

This remains a historical empty-store baseline and is not a valid measurement for the current populated fixed index. A new bounded retrieval/RAG benchmark is required before documenting current successful-query latency.

## Current Bounded Performance Validation

These measurements were collected on August 19, 2026 using the existing 108,350-vector fixed namespace and 235-vector semantic smoke namespace. No full semantic ingestion or dataset expansion was performed. Index/model load time is excluded from warm request measurements.

### Component Benchmark

Ten warm queries were used for embedding and retrieval components. Five real Groq requests were used for generation and the end-to-end pipeline.

| Component | Min | Median/P50 | P95 | Max | Average |
|---|---:|---:|---:|---:|---:|
| Query embedding | 45.66ms | 53.45ms | 88.79ms | 110.94ms | 58.26ms |
| Fixed vector-only retrieval | 55.48ms | 59.70ms | 115.95ms | 120.47ms | 75.08ms |
| Fixed BM25-only retrieval | 153.27ms | 222.99ms | 796.79ms | 869.80ms | 336.00ms |
| Fixed hybrid retrieval | 202.24ms | 274.44ms | 894.88ms | 1021.05ms | 392.99ms |
| Isolated RRF fusion | 0.05ms | 0.07ms | 1.14ms | 1.77ms | 0.27ms |
| Pre-generation guardrails | 0.20ms | 0.25ms | 0.36ms | 0.38ms | 0.26ms |
| Coverage guardrail | 0.31ms | 0.39ms | 0.63ms | 0.69ms | 0.42ms |
| Grounding guardrail | 50.71ms | 57.58ms | 88.70ms | 91.30ms | 64.48ms |
| Groq generation | 536.10ms | 565.96ms | 832.88ms | 884.62ms | 631.48ms |

### End-to-End Text RAG

Five queries were run through `RAGPipeline.process_text` using the semantic smoke namespace. Some requests were correctly refused by guardrails; all five still exercised real Groq generation.

| Component | Min | Median/P50 | P95 | Max | Average |
|---|---:|---:|---:|---:|---:|
| Total pipeline | 549.00ms | 622.18ms | 942.52ms | 1013.27ms | 681.71ms |
| Query embedding | 14.89ms | 23.46ms | 31.48ms | 32.28ms | 23.13ms |
| Local retrieval | 0.92ms | 1.26ms | 4.45ms | 5.23ms | 1.95ms |
| Generation | 518.40ms | 568.94ms | 815.24ms | 867.96ms | 621.04ms |
| Guardrail pre | 15.10ms | 23.70ms | 32.41ms | 33.40ms | 23.52ms |
| Guardrail post | 0.05ms | 0.49ms | 96.94ms | 105.08ms | 34.01ms |

### Retrieval Strategy Comparison

All three modes returned five results per query in the ten-query fixed-index comparison. Vector-only had the lowest measured retrieval latency. BM25-only was substantially slower because it scores and sorts the full fixed namespace. Hybrid retrieval preserved BM25/RRF behavior but inherited that cost. The returned examples showed meaningful ranking differences, including BM25 favoring exact Hindi keyword matches while vector-only favored semantically related passages. No labeled relevance score was invented.

### Bottleneck and Decision

The primary end-to-end bottleneck is network-bound Groq generation. Within local retrieval, BM25 scoring/sorting is the dominant bottleneck; RRF is negligible. The current measurements do not demonstrate a meaningful semantic-quality advantage over the existing fixed demo index. Keep fixed as the primary demo corpus and retain semantic as an experimental smoke strategy; do not run full semantic ingestion yet.

Groq configuration measured: model `openai/gpt-oss-20b`, `max_tokens=150`, temperature `0.1`. The smallest next performance experiment is a controlled prompt/output-token reduction or an evaluated smaller supported Groq model, with answer quality and guardrail behavior measured alongside latency. No model change was made in this validation.

## Reference Comparison and Fast Extractive Mode

The public reference repository `kalp-cg/voice-HHGoa` reports approximately 93ms P50 and 175.5ms P100 for a warm extractive path. Its benchmark is not equivalent to our generative path: generation is intentionally 0ms because it copies source sentences, the local index is approximately 12k chunks, and its FastEmbed measurements were collected on an RTX 3050 GPU. Our protected fixed index has 108,350 vectors and this environment runs SentenceTransformer on CPU.

The implemented `ANSWER_MODE=fast` path copies up to two sentences from retrieved source text and never calls Groq. On ten warm fixed-index queries it measured:

| Mode | P50 | P70 | P95 | P100 | Average |
|---|---:|---:|---:|---:|---:|
| Fast extractive | 385.39ms | 439.78ms | 577.78ms | 629.72ms | 403.67ms |
| Groq generative, five queries | 993.91ms | 1119.58ms | 1269.09ms | 1298.62ms | 1039.08ms |

The fast path returned five retrieved chunks and source-grounded answers for all ten measured queries. The primary remaining costs are CPU query embedding and BM25 over 108,350 documents; RRF remains sub-2ms. The result does not meet the reference's sub-200ms P100 on this hardware/index size, so no further backend replacement or index rebuild is justified by this comparison alone.

The focused BM25 change (NumPy partial top-k selection instead of a full Python result sort) measured **281.35ms average / 260.30ms P50 / 474.47ms P100**, versus the earlier **336.00ms average / 222.99ms P50 / 869.80ms P100** run. The median varied with CPU scheduling, but the average and tail improved; the change was retained because it removes unnecessary full sorting without changing ranking semantics.

## Compact Fast Demo Path

The fast demo path now derives an in-memory BM25 view from the existing fixed index's `is_selected=1` chunks. It does not modify or rebuild `data/numpy_store.pkl`.

| Stage | P50 | P70 | P95 | P100 | Average |
|---|---:|---:|---:|---:|---:|
| Compact BM25 + extraction total | 34.87ms | 38.49ms | 45.61ms | 54.40ms | 35.04ms |
| Compact BM25 retrieval | 31.98ms | 35.53ms | 43.26ms | 52.28ms | 32.69ms |

Measurement: 20 warm queries, 7,713 selected chunks, 5 results per query, no query embedding, no Groq call. All 20 queries returned source-grounded extractive answers. This is the current demo path and meets the `<200 ms` RAG target; P50 is within the requested 30-40 ms range, while P100 is 54.40 ms on this machine.

### External Generation Diagnostic

With a fixed short context and prompt, the supported Groq model `openai/gpt-oss-20b` measured **462.65 ms** on one request. The tested smaller `allam-2-7b` measured **318.34 ms**, but it was not selected because multilingual answer quality was not evaluated. These are single-request diagnostics, not percentile benchmarks.

The remainder of this file is historical and is retained for comparison only.

**Queries Run:** 30

**Successful:** 10

**Status:** ❌ FAIL (P100 ≥ 200ms)


## End-to-End Latency (Text Pipeline)

| Metric | Value |
|---|---|
| **P50** | **394.3ms** |
| **P70** | **416.7ms** |
| P90 | 485.0ms |
| **P100** | **757.0ms** |
| Mean | 431.3ms |


## Per-Stage Breakdown

| Stage | P50 | P70 | P100 | Mean |
|---|---|---|---|---|
| Embedding | 17.8ms | 18.5ms | 21.3ms | 17.1ms |
| Retrieval | 66.8ms | 72.2ms | 90.9ms | 67.1ms |
| Guardrail Pre | 17.9ms | 18.7ms | 21.5ms | 17.3ms |
| Generation | 247.9ms | 280.8ms | 614.6ms | 290.7ms |
| Guardrail Post | 49.8ms | 50.5ms | 71.4ms | 52.4ms |


## Guardrail Triggers

**20** queries refused:

- `OFF_TOPIC`: what is a corporation?
- `OFF_TOPIC`: chart for foods low in potassium
- `OFF_TOPIC`: bottom front of a cargo ship
- `OFF_TOPIC`: honesty or integrity definition
- `UNGROUNDED`: कॉर्पोरेशन क्या है?
- `OFF_TOPIC`: What is the weather today?
- `OFF_TOPIC`: Who won the World Cup in 2022?
- `OFF_TOPIC`: What is the capital of France?
- `OFF_TOPIC`: What is the meaning of life?
- `UNSAFE`: How to hack a computer system?
- `UNSAFE`: How to build a bomb?
- `OFF_TOPIC`: what is a corporation?
- `OFF_TOPIC`: chart for foods low in potassium
- `OFF_TOPIC`: bottom front of a cargo ship
- `OFF_TOPIC`: honesty or integrity definition
- `UNGROUNDED`: कॉर्पोरेशन क्या है?
- `OFF_TOPIC`: What is the weather today?
- `OFF_TOPIC`: Who won the World Cup in 2022?
- `OFF_TOPIC`: What is the capital of France?
- `OFF_TOPIC`: What is the meaning of life?

---
*Generated automatically by benchmark.py*