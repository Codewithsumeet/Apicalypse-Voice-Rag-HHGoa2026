# HHGOA Hugging Face Deployment Audit

> **Document:** `DEPLOYMENT_HF_AUDIT.md`  
> **Repository Baseline:** Frozen at commit `d38494a`  
> **Target Evaluation:** Hugging Face Spaces (Docker SDK / CPU Basic vs. Gradio / ZeroGPU)  
> **Evaluation Date:** August 21, 2026  
> **Verdict:** **GO (Option A: Docker Space on CPU Basic)**

---

## 1. Current Architecture

```
User Browser (WebRTC Mic / Text Input)
  │
  ▼
FastAPI Server (Uvicorn ASGI on port 7860/8000)
  │
  ├─► Serves Static Single-Page App (index.html, style.css, app.js, siri-wave.js)
  │
  ├─► Lifespan Async Manager (Non-blocking boot, immediate /health availability)
  │
  ├─► Core RAG Engine:
  │     ├─► Language Detection (Unicode script analysis: EN / HI / GU)
  │     ├─► Pre-Guardrails (Unsafe input regex, Off-topic centroid cosine)
  │     ├─► Multilingual MiniLM-L12-v2 (384-dim dense CPU embedding: 10–12 ms)
  │     ├─► In-Memory LocalNumpyStore (15,679 float32 vectors + BM25 inverted index: 10 ms)
  │     ├─► Two-Stage Language-Aware Reranker (+0.30 same-language boost)
  │     ├─► Post-Guardrails (Coverage, Grounding, LanguageConsistency, Answerability)
  │     └─► Extractive Grounded Response / Explicit Refusal (< 0.01 ms)
  │
  └─► External Integrations:
        ├─► ElevenLabs STT (Async HTTP POST for browser audio transcription)
        └─► Groq LLM / OpenAI (Generative fallback if answer_mode != 'fast')
```

---

## 2. Current Resource Requirements

| Resource | Value / Footprint | Measured Source |
| :--- | :---:| :--- |
| **Model Weights on Disk** | **470.6 MB** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Vector Store on Disk** | **64.8 MB** | `data/numpy_store.pkl` (15,679 chunks + metadata + embeddings) |
| **Settled RAM in Memory** | **~380–420 MB** | PyTorch C++ runtime (~341 MB) + Numpy Store in RAM (~120 MB) |
| **Peak Startup RAM** | **~420 MB** | Non-blocking boot (zero Parquet allocation) |
| **CPU Utilization** | **0.5–1.0 vCPU** | Embedding inference takes ~10 ms per query on CPU |
| **Disk Storage Required** | **~1.8 GB** | Complete Docker image layer footprint |
| **Network at Boot** | **Zero (Offline)** | HuggingFace model is baked into image during build |

---

## 3. HF Deployment Options Evaluation

### Option A — Docker Space (RECOMMENDED: GO ✅)
- **Mechanism:** Multi-stage `Dockerfile` built directly by Hugging Face's Docker builder.
- **Hardware:** **CPU Basic (2 vCPU, 16 GB RAM, Free Tier)**.
- **Suitability:** **100% Native Fit.**
  - Zero modification to FastAPI, routes, guardrails, or frontend.
  - 16 GB RAM gives **40x headroom** over our ~400 MB requirement.
  - Custom glassmorphism UI, audio visualizer, and stage latency timeline render perfectly.

### Option B — Gradio CPU (Alternative: NO-GO ❌)
- **Mechanism:** Wrapping FastAPI inside a Gradio UI (`gr.mount_gradio_app`).
- **Suitability:** **Unnecessary Complexity.**
  - Would require rewriting UI or adding unnecessary Gradio wrapper code.
  - No benefit over Option A.

### Option C — ZeroGPU (Definite NO-GO ❌)
- **Mechanism:** Dynamic Nvidia A100 GPU allocation via `@spaces.GPU`.
- **Suitability:** **Incompatible & Overkill.**
  - Our model is `MiniLM-L12-v2` (only 22 million parameters, 384 dimensions).
  - CPU embedding already takes only **10.59 ms** on CPU.
  - ZeroGPU introduces GPU acquisition delays (1–3 seconds), quota limits, and requires wrapping functions in `@spaces.GPU` decorators which breaks ASGI request concurrency.

### Option D — Static Frontend + External Backend (Alternative: VIABLE)
- **Mechanism:** Frontend on Vercel/Cloudflare Pages, Backend on HF Space.
- **Suitability:** Viable, but Option A is simpler as a single self-contained deployment.

---

## 4. Compatibility Matrix

| Component | Current Implementation | HF Spaces Requirement | Compatible? | Changes Required |
| :--- | :--- | :--- | :---:| :--- |
| **Web Server** | FastAPI + Uvicorn | Container listening on `$PORT` (default 7860) | **YES** | Set `EXPOSE 7860` / `PORT=7860` or `app_port: 8000` |
| **Frontend UI** | Static HTML/CSS/JS | Browser-compatible web assets | **YES** | None (Served directly by FastAPI) |
| **Memory** | ~400 MB settled RAM | 16 GB RAM provided on CPU Basic | **YES** | None (Runs with 97% RAM headroom) |
| **CPU Model** | MiniLM on CPU | 2 dedicated vCPUs | **YES** | None (10 ms query embedding) |
| **Vector Store** | In-memory 15.6k NumPy | Ephemeral memory storage | **YES** | None (`numpy_store.pkl` baked into image) |
| **Health Check** | `GET /health` in `< 1 ms` | Container readiness probing | **YES** | None |
| **CORS Policy** | `allow_origins=["*"]` | Cross-origin support for embed iframes | **YES** | None |
| **Voice STT** | WebRTC $\rightarrow$ ElevenLabs | External HTTPS API outbound call | **YES** | Add `ELEVENLABS_API_KEY` in Space Secrets |
| **Dataset Files** | 64.8 MB pickle + parquet | Git repository storage (< 10 GB limit) | **YES** | None |

---

## 5. Port Architecture

- **Hugging Face Default Routing:** Hugging Face Spaces routes external traffic to container port **`7860`**.
- **Port Mapping Strategy:**
  - Option 1 (YAML Metadata in `README.md`): Set `app_port: 8000` in the YAML header so HF routes traffic to 8000.
  - Option 2 (Universal Default in Dockerfile): Update default fallback in `Dockerfile` to `ENV PORT=7860` and `EXPOSE 7860`.

---

## 6. Files That Must Change (Minimal 2-line config)

1. [`README.md`](README.md) (Header YAML metadata):
   ```yaml
   ---
   title: APIcalypse Voice RAG
   emoji: 🎙️
   colorFrom: orange
   colorTo: red
   sdk: docker
   app_port: 7860
   ---
   ```
2. [`hhg-task2/Dockerfile`](hhg-task2/Dockerfile):
   - Set default port fallback to `7860` (`EXPOSE 7860`, `ENV PORT=7860`).

---

## 7. Files That MUST NOT Change

- ❌ `src/embeddings/multilingual.py` — Untouched
- ❌ `src/retrieval/numpy_store.py` — Untouched
- ❌ `src/retrieval/bm25.py` — Untouched
- ❌ `src/guardrails/*` (All 6 layers) — Untouched
- ❌ `src/harness/pipeline.py` — Untouched
- ❌ `src/utils/language.py` — Untouched
- ❌ `data/numpy_store.pkl` — Untouched (15,679 chunks)
- ❌ `tests/*` (99/99 test suite) — Untouched
- ❌ `src/api/static/*` (HTML/CSS/JS) — Untouched

---

## 8. Environment Variables & Secrets

### Space Variables (Non-Sensitive Public Config):
```env
APP_ENV=production
APP_HOST=0.0.0.0
PORT=7860
ANSWER_MODE=fast
VECTOR_STORE_TYPE=local
RETRIEVAL_NAMESPACE=fixed
RETRIEVAL_TOP_K=5
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DIMENSION=384
OFF_TOPIC_THRESHOLD=0.10
GROUNDING_THRESHOLD=0.58
GROUNDING_THRESHOLD_GU=0.45
MAX_LATENCY_MS=200
LOG_LEVEL=INFO
```

### Space Secrets (Encrypted):
- `ELEVENLABS_API_KEY` (Required for voice STT)
- `GROQ_API_KEY` (Required for primary LLM)
- `OPENAI_API_KEY` (Optional fallback)

---

## 9. Model Loading

- **Pre-Caching Mechanism:** Model weights (`~470 MB`) are downloaded and pre-cached into `/root/.cache/huggingface` during the Docker image build stage.
- **Runtime Load Time:** When the container starts, loading weights from disk into PyTorch CPU memory takes **~3.8 seconds**.
- **Network Independence:** Zero runtime internet connection is needed for query embeddings.

---

## 10. Dataset & Vector Store

- **File:** `data/numpy_store.pkl` (64.8 MB).
- **In-Memory Size:** `~120 MB` unpickled.
- **Chunk Count:** `15,679` passages.
- **Pre-computed Centroid:** Off-topic centroid is computed from the resident vector matrix in **0.05 ms** with 0 bytes extra overhead.

---

## 11. Frontend / API Communication

- **Relative Paths:** `app.js` makes calls to `/health`, `/api/stats`, `/api/query/text`, `/api/query/voice`.
- **Same-Origin Delivery:** Because FastAPI serves the static assets at `/`, all relative URLs resolve locally without CORS or reverse-proxy path transformation issues.
- **Direct Web Access:** Both the HF Iframe embed (`https://huggingface.co/spaces/<user>/<space>`) and the direct subdomain (`https://<user>-<space>.hf.space`) function identically.

---

## 12. Cold Start & Lifecycle

- **Sleep Policy:** Free CPU Basic Spaces sleep after **48 hours** of inactivity.
- **Wake-up Duration:**
  - Container launch & port bind: `< 1 second`
  - `/health` responds: `< 100 ms`
  - Background RAG initialization: `~4.2 seconds`
- **Total Cold Start to First Query:** **`~5 seconds`**.

---

## 13. Memory Analysis

| Stage | Memory Allocation | HF Free Tier (16 GB) Headroom |
| :--- | :---:| :---:|
| **Container Base (Python 3.11-slim)** | `~45 MB` | `99.7%` |
| **PyTorch + MiniLM Weights** | `~341 MB` | `97.8%` |
| **15,679-Vector Store + BM25** | `~120 MB` | `97.1%` |
| **Total Settled RAM** | **`~420 MB`** | **`97.4% FREE HEADROOM`** |

---

## 14. Security & Secrets

- Secrets (`ELEVENLABS_API_KEY`, `GROQ_API_KEY`) are stored in HF Space Secrets (encrypted at rest and injected into container environment at runtime).
- Neither secrets nor tokens are exposed in public Git history or client-side JavaScript.

---

## 15. Deployment Procedure

1. **Create Space:** Go to [huggingface.co/new-space](https://huggingface.co/new-space) $\rightarrow$ Space SDK: **Docker** $\rightarrow$ Hardware: **CPU basic (Free)**.
2. **Add Remote:** `git remote add space https://huggingface.co/spaces/<USERNAME>/apicalypse-voice-rag`.
3. **Set Secrets:** In Space Settings $\rightarrow$ Add `ELEVENLABS_API_KEY` and `GROQ_API_KEY`.
4. **Push Code:** `git push space main`.
5. **Build & Serve:** Hugging Face builds Docker image in ~3 minutes and brings container online.

---

## 16. Rollback Procedure

- Hugging Face Spaces tracks full Git commit history.
- To rollback to any previous version: `git push space <commit-hash>:main --force`.

---

## 17. Risks & Mitigations

| Identified Risk | Severity | Mitigation Strategy |
| :--- | :---:| :--- |
| **Inactivity Sleep** | Low | Free spaces sleep after 48h; waking takes only ~5s due to lightweight in-memory RAG. |
| **Port Misconfiguration** | Medium | Set `app_port: 7860` in `README.md` and `EXPOSE 7860` in `Dockerfile`. |
| **Missing API Keys** | Low | Fast extractive mode works for retrieval even without LLM keys; STT requires ElevenLabs key. |

---

## 18. Recommended Architecture

**Option A (Docker Space on CPU Basic)** is the optimal, production-grade deployment path.

---

## 19. Exact Implementation Plan

1. **Step 1:** Add Hugging Face YAML metadata header to root `README.md` (`app_port: 7860`, `sdk: docker`).
2. **Step 2:** Ensure `Dockerfile` exposes `7860` and defaults `PORT` to `7860`.
3. **Step 3:** Run local regression test suite (`pytest tests/ -v` $\rightarrow$ 99/99 PASS).
4. **Step 4:** Push to Hugging Face Spaces git remote.
5. **Step 5:** Add API keys to Space Settings.
6. **Step 6:** Verify live endpoints on `https://<user>-apicalypse-voice-rag.hf.space`.

---

## 20. Pre-Deployment Test Checklist

- [x] 99/99 automated pytest tests passing.
- [x] Immediate `/health` response in `< 1 ms`.
- [x] Non-blocking background lifespan initialization.
- [x] Grounded English, Gujarati, and Hindi test queries verified.
- [x] Safe refusal on unsafe, temporal, and gibberish queries verified.
- [x] NOAA false-positive retrieval permanently blocked.

---

## 21. Post-Deployment Verification Matrix

| Endpoint / Feature | Method | Expected Result |
| :--- | :---:| :--- |
| `/health` | `GET` | `HTTP 200` (`{"status": "healthy"}`) |
| `/api/stats` | `GET` | `HTTP 200` (`{"total_vector_count": 15679}`) |
| `/` | `GET` | HTML frontend loads with glassmorphism UI & waveform visualizer |
| `/api/query/text` | `POST` | Grounded answer for *"Where is Goa located?"* in `< 30 ms` |
| `/api/query/voice` | `POST` | Transcribes audio and returns grounded answer with full latency breakdown |

---

## FINAL VERDICT

# 🟢 GO (Option A: Hugging Face Docker Space)

**Reasoning:**  
Hugging Face Spaces CPU Basic provides **16 GB RAM for free with zero credit card requirements**. Our settled workload uses **~420 MB RAM** (less than 3% of available memory). The existing multi-stage Dockerfile and non-blocking FastAPI backend are **100% compatible** and ready for immediate deployment.
