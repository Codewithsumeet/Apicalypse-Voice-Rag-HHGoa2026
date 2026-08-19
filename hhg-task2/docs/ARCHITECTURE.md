# Architecture — HHG Voice RAG Pipeline

## System Overview

Current active vector store: `LocalNumpyStore`, persisted at `data/numpy_store.pkl` and loaded once during application startup. Pinecone is not part of the runtime, configuration, or dependency set.

```
┌──────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                          │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────────┐       │
│  │ WebRTC   │───▶│ Audio     │───▶│ POST /api/query/voice│       │
│  │ getUserM │    │ Recording │    │ (multipart/form-data)│       │
│  └──────────┘    └───────────┘    └──────────────────────┘       │
│                                                                  │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐     │
│  │ Answer Card  │  │ Latency Bars   │  │ Context Chunks   │     │
│  └──────────────┘  └────────────────┘  └──────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI + Uvicorn (ASGI)                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────── RAG PIPELINE ────────────────────────┐   │
│  │                                                          │   │
│  │  1. TRANSCRIBE  ──▶  ElevenLabs Scribe API              │   │
│  │     │                (async, connection pool)            │   │
│  │     ▼                                                    │   │
│  │  2. GUARDRAIL (Pre)  ──▶  Unsafe Input Check (regex)    │   │
│  │     │                    Off-Topic Check (embedding)     │   │
│  │     ▼                                                    │   │
│  │  3. EMBED QUERY  ──▶  MiniLM-L12-v2 (local, singleton)  │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  4. RETRIEVE  ──▶  LocalNumpyStore (hybrid, top-k=5)   │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  5. GENERATE  ──▶  Groq openai/gpt-oss-20b (primary)   │   │
│  │     │              OpenAI GPT-4o-mini (fallback)         │   │
│  │     ▼                                                    │   │
│  │  6. GUARDRAIL (Post)  ──▶  Grounding Check (embedding)  │   │
│  │     │                                                    │   │
│  │     ▼                                                    │   │
│  │  7. FORMAT OUTPUT  ──▶  PipelineResult (Pydantic)       │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  CROSS-CUTTING:                                                  │
│  • Retry with exponential backoff                                │
│  • Structured JSON logging (structlog)                          │
│  • Per-stage latency instrumentation                            │
│  • Request trace IDs                                            │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      EXTERNAL SERVICES                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ ElevenLabs │  │ Local NumPy│  │   Groq /   │                │
│  │   Scribe   │  │   index    │  │   OpenAI   │                │
│  │    STT     │  │            │  │    LLM     │                │
│  └────────────┘  └────────────┘  └────────────┘                │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Audio Capture** → Browser WebRTC captures mic input
2. **STT** → Audio bytes sent to ElevenLabs, returns transcript
3. **Pre-Guardrails** → Unsafe content filter + off-topic detection
4. **Embedding** → Query encoded via local MiniLM-L12-v2 model
5. **Retrieval** → Top-5 similar chunks from the resident local NumPy index, fused with BM25 when query text is available
6. **Generation** → Grounded answer via Groq (fast) or OpenAI (fallback)
7. **Post-Guardrail** → Grounding check (answer vs context similarity)
8. **Response** → Structured JSON with answer, latency breakdown, context
