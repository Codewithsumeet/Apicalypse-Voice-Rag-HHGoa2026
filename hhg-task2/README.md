# HHG Voice RAG — Voice-Enabled RAG Pipeline

> **Hacker House Goa 2026 — Task #2**  
> Speak a question → Get a grounded, guardrailed answer in <200ms

#RAGInGoa

---

## 🏗️ Architecture

```
Mic Input → ElevenLabs STT → Query Embedding (MiniLM-L12-v2)
  → Local NumPy Vector Retrieval → Guardrail Checks
  → Groq Llama 3.1 70B → Structured Answer Output
```

### Pipeline Stages
| Stage | Tool | Latency Budget |
|---|---|---|
| Speech-to-Text | ElevenLabs Scribe | ~60ms |
| Query Embedding | MiniLM-L12-v2 (local) | ~5ms |
| Vector Retrieval | LocalNumpyStore (in-memory cosine + BM25/RRF) | measured locally |
| Guardrails | Embedding similarity + regex | ~5ms |
| Answer Generation | Groq `openai/gpt-oss-20b` | measured externally |
| **Total** | | **<200ms** |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API keys: ElevenLabs and Groq for voice/generation; HuggingFace only when downloading data (see `.env.example`)

### Setup

```bash
# 1. Clone and enter
git clone <repo-url>
cd hhg-voice-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Download and inspect the bounded demo dataset (maximum 10,000 records)
python scripts/download_data.py --limit 10000
python scripts/inspect_data.py

# 6. Ingest the bounded demo slice (never exceeds 10,000 records)
python scripts/ingest.py --strategy fixed --max-records 10000
python scripts/ingest.py --strategy semantic --max-records 10000
python scripts/ingest.py --strategy metadata_aware --max-records 10000

# 7. Start the server
uvicorn src.api.main:app --reload

# 8. Open browser
# Navigate to http://localhost:8000
```

---

## 📐 Chunking Strategies

We implement **3 genuinely different** chunking strategies:

### Strategy A: Fixed-Size with Overlap (Baseline)
- Splits at `512 characters` with `20% overlap`
- Recursive splitting at natural boundaries (paragraphs > sentences > words)

### Strategy B: Semantic Chunking
- Uses embedding cosine similarity to detect topic boundaries
- Splits when sentence similarity drops below `0.85` threshold

### Strategy C: Metadata-Aware (MSMARCO-Specific)
- Preserves MSMARCO query-passage relationships
- Attaches original query as metadata for hybrid retrieval

Run comparison: `python scripts/compare_chunking.py`

The demo scope is intentionally capped at 10,000 source records. The downloader
and ingester reject larger limits before starting work.

---

## 🛡️ Guardrails

| Guardrail | Method | Trigger |
|---|---|---|
| Off-topic Detection | Query vs dataset centroid similarity | Cosine sim < 0.35 |
| Unsafe Input | Compiled regex patterns | Harmful/toxic content match |
| Hallucination Check | Answer-context embedding similarity | Sim < 0.7 → refuse |
| Structured Refusal | Pydantic enum | `OFF_TOPIC`, `UNSAFE`, `UNGROUNDED`, `SYSTEM_ERROR` |

---

## ⚡ Latency Numbers

Run benchmark: `python scripts/benchmark.py --queries 50`

See full report: [docs/LATENCY_REPORT.md](docs/LATENCY_REPORT.md)

---

## 🐳 Docker

```bash
# Build and run
docker-compose up --build

# Or standalone
docker build -t hhg-voice-rag .
docker run -p 8000:8000 --env-file .env hhg-voice-rag
```

---

## 📁 Project Structure

```
hhg-voice-rag/
├── src/
│   ├── stt/          — ElevenLabs Speech-to-Text
│   ├── chunking/     — 3 chunking strategies + factory
│   ├── embeddings/   — MiniLM-L12-v2 singleton service
│   ├── retrieval/    — Local NumPy vector store + BM25/RRF
│   ├── generation/   — Groq (primary) + OpenAI (fallback)
│   ├── guardrails/   — Off-topic, unsafe, grounding checks
│   ├── harness/      — Pipeline orchestrator, retry logic
│   └── api/          — FastAPI server + frontend
├── scripts/          — CLI tools (download, ingest, benchmark)
├── tests/            — Unit + integration tests
└── docs/             — Auto-generated reports
```

---

## 📄 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/query/voice` | Voice query (audio upload) |
| `POST` | `/api/query/text` | Text query (JSON body) |
| `GET` | `/api/stats` | Vector DB statistics |
| `GET` | `/health` | Health check |

---

## 👥 Team

Built for Hacker House Goa 2026 Open Trials.

---

*#RAGInGoa*
