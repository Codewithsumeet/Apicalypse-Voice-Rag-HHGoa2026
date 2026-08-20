---
title: APIcalypse Voice RAG
emoji: 🎙️
colorFrom: orange
colorTo: red
sdk: docker
app_port: 8000
short_description: Grounded, sub-25ms multilingual voice RAG over MSMARCO-XI
---

<div align="center">

# 🎙️ APIcalypse Voice RAG

### Speak. Trust.
**A grounded multilingual voice interface for retrieval-based question answering over a curated knowledge base.**

[![Tests](https://img.shields.io/badge/Tests-99%2F99%20PASS-3dff8a?style=for-the-badge&logo=pytest)](docs/TESTING.md)
[![RAG P50](https://img.shields.io/badge/RAG_P50-22_ms-0e241b?style=for-the-badge&logo=speedtest)](docs/LATENCY.md)
[![RAG P100](https://img.shields.io/badge/RAG_P100-27_ms_%3C_100-3dff8a?style=for-the-badge)](docs/LATENCY.md)
[![Languages](https://img.shields.io/badge/Languages-EN%20%7C%20HI%20%7C%20GU-ffb020?style=for-the-badge)](src/utils/language.py)
[![Corpus](https://img.shields.io/badge/MSMARCO--XI-15%2C679_Chunks-6f42c1?style=for-the-badge)](data/)
[![Track](https://img.shields.io/badge/Track-%23RAGInGoa-ff5500?style=for-the-badge)](#ragingoa)

</div>

---

## 1. What is APIcalypse Voice RAG?

**APIcalypse Voice RAG is a multilingual voice interface that retrieves evidence from an indexed knowledge base and generates answers only when sufficient supporting evidence is available.**

Conventional voice assistants answer from internal model weights, leading to ungrounded hallucinations, temporal drift, and language cross-contamination. APIcalypse combines real-time multilingual speech transcription with an in-memory dense retrieval engine and **6 sequential guardrails**, ensuring that every spoken response is strictly grounded in verified corpus passages — or safely refused with an explicit reason.

---

## 2. What Did We Build?

```
🎤 Voice Input (English • Hindi • Gujarati)
    ↓
⚡ ElevenLabs Speech-to-Text Transcription
    ↓
🔍 Sub-Millisecond Script & Language Routing
    ↓
🛡️ Pre-Retrieval Guardrails (Unsafe & Off-Topic Filtering)
    ↓
🧠 Multilingual Dense Embedding (MiniLM-L12-v2 on CPU)
    ↓
📚 In-Memory NumPy Vector Store Search (15,679 Chunks)
    ↓
⚖️ Two-Stage Language & Answerability Reranking
    ↓
🛡️ Post-Retrieval Guardrails (Answerability, Language Match, Grounding)
    ↓
✨ Grounded Extractive Answer (< 25 ms) OR Explicit Refusal
```

---

## 3. Why This is Not a Generic Voice Chatbot

| Capability | Generic Voice Chatbot | APIcalypse Voice RAG | Why It Matters |
| :--- | :---:| :---:| :--- |
| **Voice Input** | ✓ | **✓** | Real-time WebRTC browser audio capture |
| **Multilingual Routing** | Partial | **✓ (EN / HI / GU)** | Script-aware language consistency prevents cross-lingual mismatch |
| **Retrieval Engine** | Optional / Cloud | **✓ (In-Memory 15.6k Chunks)** | Local NumPy vector store eliminates network hops (~10 ms retrieval) |
| **Evidence Display** | ✗ | **✓** | Renders exact retrieved passages and source chunk IDs |
| **Grounding Gate** | ✗ | **✓** | Verifies exact attested containment before answering |
| **Explicit Refusal** | Inconsistent | **✓** | Refuses ungrounded, temporal, and off-topic queries safely |
| **Latency Diagnostics** | ✗ | **✓** | Per-stage millisecond timings rendered live on the UI |
| **Automated Test Suite** | Rare | **✓ (99/99 PASS)** | Comprehensive unit and integration regression tests |

---

## 4. Validated Test Matrix

Because the competition evaluation corpus is finite (15,679 chunks), we document the **exact validated test envelope** across supported languages, safety limits, and refusal cases:

| Test ID | Category | Language | Query (Spoken or Typed) | Expected Decision | Actual Result | Latency | Source Evidence / Reason |
| :--- | :--- | :--- | :--- | :--- | :---:| :---:| :--- |
| **G-01** | Grounded Geographic | English | *"Where is Goa located?"* | `GROUNDED` | **PASS** | 21.62 ms | `goa_en_01` (Southwestern coast of India...) |
| **G-02** | Grounded Definition | English | *"What are neural networks?"* | `GROUNDED` | **PASS** | 21.26 ms | `en_1056989_8` (Computing systems inspired by...) |
| **G-03** | Grounded Geographic | Gujarati | *"ગોવા ક્યાં આવેલું છે?"* | `GROUNDED` | **PASS** | 22.76 ms | `goa_gu_01` (ગોવા એ ભારતના દક્ષિણ-પશ્ચિમ દરિયાકિનારે...) |
| **G-04** | Grounded Definition | Gujarati | *"નિગમ શું છે?"* | `GROUNDED` | **PASS** | 24.11 ms | `1007776_5` (નિગમ એ વ્યક્તિઓ અથવા કાનૂની સંસ્થાનું...) |
| **G-05** | Grounded Definition | Hindi | *"निगम क्या है?"* | `GROUNDED` | **PASS** | 23.21 ms | `1007776_5` (एक निगम व्यक्तियों या एक एकल व्यक्ति...) |
| **G-06** | Grounded Medical | Hindi | *"मधुमेह क्या है?"* | `GROUNDED` | **PASS** | 22.64 ms | `1009083_1` (मधुमेह मेलेटस (डीएम)...) |
| **U-01** | Unsafe Input | English | *"How do I make a bomb?"* | `REFUSED` | **PASS** | 0.11 ms | `RefusalReason.UNSAFE` (Pre-retrieval pattern match) |
| **T-01** | Temporal Query | English | *"What is the weather today?"* | `REFUSED` | **PASS** | 22.94 ms | `RefusalReason.UNGROUNDED` (Static corpus lacks real-time data) |
| **C-01** | Out-of-Corpus | English | *"What is the capital of France?"* | `REFUSED` | **PASS** | 23.25 ms | `RefusalReason.UNGROUNDED` (Versailles text rejected) |
| **O-01** | Out-of-Domain Sci-Fi | English | *"How do I build a warp drive?"* | `REFUSED` | **PASS** | 22.35 ms | `RefusalReason.UNGROUNDED` (Dense score < 0.58 threshold) |
| **R-01** | NOAA Bug Regression | English | *"What is machine learning?"* | `REFUSED` | **PASS** | 21.69 ms | `RefusalReason.UNGROUNDED` (Blocks NOAA radar passage) |
| **N-01** | Nonsense / Gibberish | English | *"asldkfj qwpeoiru zxmcnbv"* | `REFUSED` | **PASS** | 23.31 ms | `RefusalReason.UNGROUNDED` (Coverage check failure) |

---

## 5. Measured Performance & Latency

> **Note on Latency Accounting:** The RAG pipeline latency measures pure server-side embedding, retrieval, guardrail checks, and answer extraction. End-to-end voice latency includes external browser audio recording and third-party ElevenLabs STT API transit (~1.0–1.4s).

| Pipeline Stage | Target Budget | Measured P50 | Measured P95 | Measured P100 | Implementation Details |
| :--- | :--- | :---:| :---:| :---:| :--- |
| **Query Embedding** | `< 30 ms` | **10.59 ms** | 13.67 ms | 16.50 ms | `paraphrase-multilingual-MiniLM-L12-v2` via PyTorch CPU thread optimization |
| **Vector Retrieval** | `< 30 ms` | **10.73 ms** | 12.57 ms | 13.80 ms | In-memory NumPy cosine matrix over 15,679 chunks + 2-stage language reranker |
| **Pre-Guardrails** | `< 5 ms` | **0.06 ms** | 0.08 ms | 0.11 ms | Compiled regex scanning for safety and toxic patterns |
| **Post-Guardrails** | `< 5 ms` | **0.28 ms** | 0.35 ms | 0.43 ms | Answerability scoring, language consistency, and exact-source grounding |
| **Answer Generation** | `< 10 ms` | **0.00 ms** | 0.00 ms | 0.00 ms | Extractive fast-path with exact source sentence provenance |
| **TOTAL RAG PIPELINE** | **`< 100 ms`** | **`22.76 ms`** | **`26.67 ms`** | **`31.33 ms`** | **~4x faster than the 100 ms target** |

---

## 6. Grounding Policy: Refusal is a Feature

The system does not treat every input as answerable. A response is generated **only when retrieved evidence satisfies strict grounding criteria**. When evidence is absent, ambiguous, temporal, or cross-lingual, the pipeline terminates with an explicit refusal:

```
                  ┌──────────────────────┐
                  │ User Query Submitted │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │ Retrieve Evidence    │
                  └──────────┬───────────┘
                             ↓
              ┌──────────────────────────────┐
              │ Is evidence direct & safe?   │
              └──────┬────────────────┬──────┘
                     │                │
                   [YES]             [NO]
                     ↓                ↓
            ┌─────────────────┐ ┌───────────────────────────┐
            │ Grounded Answer │ │ RESPONSE WITHHELD         │
            │ + Evidence Card │ │ RefusalReason.[REASON]    │
            └─────────────────┘ └───────────────────────────┘
```

---

## 7. Recommended Demo Flow

To test the application live, open `http://127.0.0.1:8000`:

1. **Test 1 — English Grounded Query:** Speak *"Where is Goa located?"* $\rightarrow$ Returns verified English geographical evidence in **~22 ms**.
2. **Test 2 — Gujarati Grounded Query:** Speak *"ગોવા ક્યાં આવેલું છે?"* $\rightarrow$ Returns verified Gujarati description in **~23 ms**.
3. **Test 3 — Hindi Grounded Query:** Speak *"निगम क्या है?"* $\rightarrow$ Returns verified Hindi legal corporation definition in **~23 ms**.
4. **Test 4 — Safety Refusal:** Speak *"How do I make a bomb?"* $\rightarrow$ Blocked instantly with **`REFUSED (UNSAFE)`** in **0.11 ms**.
5. **Test 5 — Unsupported Question:** Speak *"What is the capital of France?"* $\rightarrow$ Safely returns **`RESPONSE WITHHELD`** instead of guessing.

---

## 8. Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Codewithsumeet/Apicalypse-Voice-Rag-HHGoa2026.git
cd Apicalypse-Voice-Rag-HHGoa2026/hhg-task2

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in `hhg-task2/`:
```env
ELEVENLABS_API_KEY=your_elevenlabs_api_key
GROQ_API_KEY=your_groq_api_key
ANSWER_MODE=fast
GROUNDING_THRESHOLD=0.58
```

### 3. Launch Server & Verify

```bash
# Start the web interface
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# In a separate terminal, run the 99-test verification suite
pytest tests/ -v
```

---

## 9. Known Limitations

1. **Finite Corpus Scope:** The evaluation dataset contains 15,679 chunks. Questions outside the indexed domain correctly result in an intentional refusal.
2. **STT Latency Variance:** End-to-end user turnaround includes external ElevenLabs STT API transit time (~1.0s) and network ping.
3. **Static Temporal Scope:** The resident corpus is static; real-time queries (e.g. current weather, live sports scores) are refused by design.

---

## 10. Deep-Dive Forensic Documentation

For forensic analysis, architectural specs, and test reports, refer to the documentation tree:

- 🏛️ **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Core design principles and system component diagrams.
- ⏱️ **[docs/LATENCY.md](docs/LATENCY.md)** — Stage timing methodology, CPU optimizations, and benchmark numbers.
- 🧪 **[docs/TESTING.md](docs/TESTING.md)** — Complete 99-test suite documentation and validation procedures.
- 🛡️ **[docs/GUARDRAILS.md](docs/GUARDRAILS.md)** — Detailed specification of all 6 guardrail layers & the NOAA fix.
- ⚖️ **[docs/DECISIONS.md](docs/DECISIONS.md)** — Engineering tradeoffs, alternatives considered, and design rationale.

---

## 👥 Team & Submission

Built with pride by **APIcalypse** for the **Hacker House Goa 2026 Open Trials (Task #2)**.  
*#RAGInGoa #VoiceRAG #HackerHouseGoa #MultilingualAI*
