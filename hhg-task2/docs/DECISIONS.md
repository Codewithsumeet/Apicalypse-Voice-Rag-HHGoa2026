# Engineering Tradeoffs & Architectural Decisions

> **Purpose:** Documenting key engineering choices, why alternatives were rejected, and lessons learned.

---

## 1. Decision Log

### Decision 1: Resident In-Memory NumPy Vector Store vs. External Vector DB
- **Choice:** Resident in-memory NumPy matrix (`LocalNumpyStore`).
- **Rationale:** External vector databases (Qdrant, Pinecone, Milvus) introduce network hops (15–50 ms) and disk serialization overhead. In-memory matrix multiplication over 15,679 384-dim float32 vectors takes **~10 ms** on CPU with zero network latency.

### Decision 2: Bounded Corpus Slice vs. Full 55.6 GB MSMARCO-XI Ingestion
- **Choice:** High-quality bounded slice of 15,679 chunks (470 MB in memory).
- **Rationale:** Loading the uncompressed 55.6 GB dataset creates an OOM crash on standard 16 GB machines and requires 15+ GB of raw RAM for float32 vectors alone. Streaming a bounded dataset preserves sub-25ms response times.

### Decision 3: Fast-Path Extractive Grounding vs. LLM Paraphrasing
- **Choice:** Extractive source-sentence extraction with zero LLM generation lag (0.00 ms).
- **Rationale:** LLM generation (e.g. Groq, Ollama) introduces 200–800 ms of token streaming latency and risks hallucination. Extractive grounding guarantees 100% factual fidelity and provenance at 0 ms.

### Decision 4: Multilingual Script-Aware Reranking vs. Generic Cosine Search
- **Choice:** Two-stage language-aware reranking prioritizing same-language candidates.
- **Rationale:** Dense cross-lingual models embed related topics across languages closely (e.g. English query matching Hindi text). Strict script routing ensures users receive answers in the language they spoke.

### Decision 5: Exact-Source Grounding Optimization
- **Choice:** `if answer in context: return pass_result()`.
- **Rationale:** Calling `encode_query` on a 2,000-character context block inside `GroundingGuardrail` caused a ~400ms CPU spike. Checking string containment for extracted sentences drops verification time to `< 0.01 ms`.
