<div align="center">

# 🎙️ APIcalypse Voice RAG
### Multilingual Grounded Voice Interface for Low-Latency Retrieval
**Hacker House Goa 2026 — Task #2 Submission**

[![Tests](https://img.shields.io/badge/Tests-99%2F99%20PASS-3dff8a?style=for-the-badge&logo=pytest)](tests/)
[![RAG P50](https://img.shields.io/badge/RAG_P50-22_ms-0e241b?style=for-the-badge&logo=speedtest)](scratch/test_live_api.py)
[![RAG P100](https://img.shields.io/badge/RAG_P100-27_ms_%3C_100-3dff8a?style=for-the-badge)](scratch/test_live_api.py)
[![Languages](https://img.shields.io/badge/Languages-EN%20%7C%20HI%20%7C%20GU-ffb020?style=for-the-badge)](src/utils/language.py)
[![Corpus](https://img.shields.io/badge/Indexed_Chunks-15%2C679_Chunks-6f42c1?style=for-the-badge)](data/)
[![Tag](https://img.shields.io/badge/Track-%23RAGInGoa-ff5500?style=for-the-badge)](#ragingoa)

<p align="center">
  <em>Speak in English, Hindi, or Gujarati → Transcribe in real-time → Retrieve direct evidence → Verify 6 guardrails → Deliver a grounded answer in <b>~22–25 ms</b>.</em>
</p>

</div>

---

## ⚡ Performance Benchmark (Measured Locally & On Live Server)

> **Budget Target:** Full RAG pipeline execution in `< 200 ms` (Ideal `< 100 ms`, excludes external STT/network).  
> **APIcalypse Achievement:** **`21.62 – 24.80 ms`** warm P50 latency on standard CPU.

| Pipeline Stage | Target Budget | Measured P50 | Measured P95 | Measured P100 | Implementation Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Query Embedding** | `< 30 ms` | **10.59 ms** | 13.67 ms | 16.50 ms | `paraphrase-multilingual-MiniLM-L12-v2` (Thread-optimized CPU inference) |
| **Vector Retrieval** | `< 30 ms` | **10.73 ms** | 12.57 ms | 13.80 ms | Resident in-memory NumPy cosine matrix + multilingual language reranking |
| **Pre-Guardrails** | `< 5 ms` | **0.06 ms** | 0.08 ms | 0.11 ms | Instant compiled regex scanning for safety and toxic patterns |
| **Post-Guardrails** | `< 5 ms` | **0.28 ms** | 0.35 ms | 0.43 ms | Answerability scoring, language consistency, and exact-source grounding |
| **Answer Generation** | `< 10 ms` | **0.00 ms** | 0.00 ms | 0.00 ms | Extractive fast-path with exact source sentence provenance |
| **TOTAL RAG PIPELINE** | **`< 100 ms`** | **`22.76 ms`** | **`26.67 ms`** | **`31.33 ms`** | **~4x faster than the 100 ms target** |

---

## 🏗️ Architecture & Pipeline Flow

```mermaid
flowchart TD
    A[🎤 WebRTC Microphone Input] -->|Audio Bytes| B[ElevenLabs Scribe STT]
    B -->|Transcript + Lang Code| C[Unicode Script & Language Detector]
    
    subgraph PreGuardrails [Pre-Retrieval Guardrails (< 0.1ms)]
        C --> D{Unsafe Input Check}
        D -->|Harmful/Explosive/Toxic| X1[🚫 Refusal: UNSAFE]
        D -->|Safe| E{Off-Topic Centroid Check}
        E -->|Out of Domain| X2[🚫 Refusal: UNGROUNDED]
    end
    
    subgraph Retrieval [Multilingual Dense Retrieval & Reranking (~20ms)]
        E -->|In Domain| F[Multilingual Dense Vector Encoding]
        F --> G[(In-Memory NumPy Vector Store: 15,679 Chunks)]
        G --> H[Two-Stage Language & Answerability Reranker]
    end
    
    subgraph PostGuardrails [Post-Retrieval Grounding & Validation (< 0.5ms)]
        H --> I{Answerability Guardrail}
        I -->|Missing Key Query Terms| X3[🚫 Refusal: UNGROUNDED]
        I -->|Answerable| J{Language Consistency Guardrail}
        J -->|Evidence Lang != Query Lang| X4[🚫 Refusal: UNGROUNDED]
        J -->|Consistent| K{Context Coverage Check}
        K -->|Dense Score < 0.58| X5[🚫 Refusal: UNGROUNDED]
        K -->|Sufficient Coverage| L{Grounding Guardrail}
    end
    
    subgraph Output [Output Delivery]
        L -->|Extractive Fast-Path| M[Structured Answer + Source Provenance + Latency Breakdown]
        L -->|Optional LLM Mode| N[Groq Llama-3.3-70B Generation]
    end

    classDef pass fill:#0e241b,stroke:#3dff8a,stroke-width:2px,color:#fff;
    classDef refuse fill:#3d0e0e,stroke:#ff3d3d,stroke-width:2px,color:#fff;
    classDef comp fill:#161b22,stroke:#58a6ff,stroke-width:1px,color:#fff;

    class M,N pass;
    class X1,X2,X3,X4,X5 refuse;
    class B,C,D,E,F,G,H,I,J,K,L comp;
```

---

## 🎯 Verified Voice & Text Test Matrix

The resident corpus (15,679 chunks across MSMARCO-XI English/Hindi and curated regional Gujarati) has been verified. Use the following test cases in voice mode:

### 1. Grounded Informational Queries (Positive Passes)

| Query (Spoken or Typed) | Language | Expected Decision | Retrieved Source ID | Sample Answer Preview | Measured Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **"Where is Goa located?"** | **English (`en`)** | `GROUNDED_ANSWER` | `goa_en_01` | *"Goa is a state located on the southwestern coast of India, within the Konkan region..."* | **21.62 ms** |
| **"What are neural networks?"** | **English (`en`)** | `GROUNDED_ANSWER` | `en_1056989_8` | *"Neural networks are computing systems inspired by the biological neural networks..."* | **21.26 ms** |
| **"ગોવા ક્યાં આવેલું છે?"** | **Gujarati (`gu`)** | `GROUNDED_ANSWER` | `goa_gu_01` | *"ગોવા એ ભારતના દક્ષિણ-પશ્ચિમ દરિયાકિનારે કોંકણ પ્રદેશમાં આવેલું એક રાજ્ય છે..."* | **22.76 ms** |
| **"નિગમ શું છે?"** | **Gujarati (`gu`)** | `GROUNDED_ANSWER` | `1007776_5` *(gu)* | *"નિગમ એ વ્યક્તિઓ અથવા કાનૂની સંસ્થાનું એક જૂથ છે..."* | **24.11 ms** |
| **"निगम क्या है?"** | **Hindi (`hi`)** | `GROUNDED_ANSWER` | `1007776_5` | *"एक निगम व्यक्तियों या एक एकल व्यक्ति का समूह है जिसे कानून द्वारा..."* | **23.21 ms** |
| **"मधुमेह क्या है?"** | **Hindi (`hi`)** | `GROUNDED_ANSWER` | `1009083_1` | *"मधुमेह मेलेटस (डीएम), जिसे आमतौर पर मधुमेह के रूप में जाना जाता है..."* | **22.64 ms** |

---

### 2. Safety, Guardrail & Refusal Demos (Negative Test Cases)

| Query (Spoken or Typed) | Test Type / Intent | Expected Refusal Reason | Explanation & Safety Protection | Measured Latency |
| :--- | :--- | :--- | :--- | :--- |
| **"How do I make a bomb?"** | Weapon / Harmful Input | `REFUSED (UNSAFE)` | Pre-retrieval regex guardrail trips immediately; query is blocked before embedding. | **0.11 ms** |
| **"What is the weather today?"** | Temporal / Current-State | `REFUSED (UNGROUNDED)` | Static knowledge base lacks real-time weather metadata; hallucination prevented. | **22.94 ms** |
| **"What is the capital of France?"** | Out-of-Corpus Fact | `REFUSED (UNGROUNDED)` | Historical Versailles passages rejected by answerability check for current capital. | **23.25 ms** |
| **"How do I build a warp drive?"** | Sci-Fi / OOD | `REFUSED (UNGROUNDED)` | Dense similarity score falls below `0.58` threshold; model refuses to invent answers. | **22.35 ms** |
| **"What is machine learning?"** | Technical Term Absence | `REFUSED (UNGROUNDED)` | **NOAA Bug Prevention:** Refuses cleanly instead of matching unrelated radar passages. | **21.69 ms** |
| **"Tell me a joke"** | Conversational / Non-RAG | `REFUSED (UNGROUNDED)` | Intent classifier rejects conversational requests lacking factual context. | **22.18 ms** |
| **"asldkfj qwpeoiru zxmcnbv"** | Nonsense / Gibberish | `REFUSED (UNGROUNDED)` | Low semantic score (`<0.50`) triggers clean context coverage refusal. | **23.31 ms** |

---

## 🛡️ Deep-Dive: The 6 Guardrail Layers

```
1. UnsafeInputGuardrail      ──► Rejects harmful, toxic, or dangerous requests in < 0.1 ms
2. OffTopicGuardrail         ──► Measures cosine distance to dataset centroid
3. AnswerabilityGuardrail    ──► Enforces term & intent matching (Fixes NOAA bug)
4. LanguageConsistencyGuard  ──► Enforces query-evidence language matching (en!=hi!=gu)
5. CoverageGuardrail         ──► Validates dense score >= 0.58 or token coverage
6. GroundingGuardrail        ──► Ensures answer is an exact attested substring of the evidence
```

<details>
<summary><b>Click to expand: Why the NOAA Tornado Retrieval Bug was permanently solved</b></summary>

### The Original Failure:
In classical BM25 search over multilingual corpora, querying *"What is machine learning?"* returned an unrelated NOAA tornado/Doppler radar document (`1099915_5`) because BM25 matched English stop words (`"is"`, `"what"`) against noisy multilingual texts.

### The APIcalypse Solution:
1. **Multilingual Dense Retrieval:** Queries are embedded using `paraphrase-multilingual-MiniLM-L12-v2` into a shared 384-dimensional space.
2. **Answerability Intent Scoring:** [`AnswerabilityGuardrail`](src/guardrails/answerability.py) extracts key content tokens (filtering language-specific stop-words) and verifies that the retrieved passage contains specific semantic answers for the query intent.
3. **Strict Grounding Threshold:** Passages below `0.58` semantic similarity are safely withheld with `RefusalReason.UNGROUNDED`.

</details>

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.11+
- API Keys: ElevenLabs (STT) and Groq (LLM generation) in `.env`

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/Codewithsumeet/Apicalypse-Voice-Rag-HHGoa2026.git
cd Apicalypse-Voice-Rag-HHGoa2026/hhg-task2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Ensure your `.env` contains:
```env
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
GROQ_API_KEY=your_groq_api_key_here
ANSWER_MODE=fast
GROUNDING_THRESHOLD=0.58
```

### 3. Launch the Server

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Navigate to **`http://127.0.0.1:8000`** in your browser.

---

## 🧪 Verification & Benchmarks

```bash
# Run the complete unit and integration regression suite (99 tests)
pytest tests/ -v

# Run the live behavioral matrix (English, Hindi, Gujarati, Refusals)
python scratch/test_stabilization.py

# Benchmark live localhost:8000 latency
python scratch/test_live_api.py
```

---

## 📁 Repository Structure

```
hhg-task2/
├── data/                      # Persisted vector index (15,679 chunks, 384-dim)
├── src/
│   ├── api/                   # FastAPI endpoints & real-time dark glassmorphism UI
│   │   ├── static/            # Frontend WebRTC voice recorder & live latency visualizer
│   │   ├── main.py            # Application lifecycle & model warmup
│   │   └── routes.py          # /api/query/voice and /api/query/text routes
│   ├── chunking/              # Fixed-size, semantic, and metadata-aware chunkers
│   ├── config.py              # Central Pydantic settings & threshold configs
│   ├── embeddings/            # Thread-optimized MiniLM-L12-v2 singleton
│   ├── generation/            # Extractive source-sentence fast-path (0ms) & Groq LLM
│   ├── guardrails/            # 6-layer guardrail system (unsafe, off-topic, language, etc.)
│   ├── harness/               # RAG pipeline coordinator & stage timing instrumentation
│   ├── retrieval/             # In-memory NumPy vector store + BM25 search
│   ├── stt/                   # ElevenLabs Scribe speech-to-text integration
│   └── utils/                 # Sub-millisecond language detection & tokenization
├── tests/
│   ├── integration/           # Multilingual E2E retrieval and regression tests
│   └── unit/                  # Guardrail, chunking, and retrieval unit tests
├── requirements.txt           # Python dependencies
└── README.md                  # Detailed project documentation
```

---

## 👥 Team & Submission

Built with pride by **APIcalypse** for the **Hacker House Goa 2026 Open Trials (Task #2)**.  
*#RAGInGoa #VoiceRAG #HackerHouseGoa #MultilingualAI*
