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
### High-Performance Multilingual Grounded Voice Interface
**Hacker House Goa 2026 — Task #2 Submission**

[![Tests](https://img.shields.io/badge/Tests-99%2F99%20PASS-3dff8a?style=for-the-badge&logo=pytest)](hhg-task2/tests/)
[![RAG P50](https://img.shields.io/badge/RAG_P50-22_ms-0e241b?style=for-the-badge&logo=speedtest)](hhg-task2/scratch/test_live_api.py)
[![RAG P100](https://img.shields.io/badge/RAG_P100-27_ms_%3C_100-3dff8a?style=for-the-badge)](hhg-task2/scratch/test_live_api.py)
[![Languages](https://img.shields.io/badge/Languages-EN%20%7C%20HI%20%7C%20GU-ffb020?style=for-the-badge)](hhg-task2/src/utils/language.py)
[![Corpus](https://img.shields.io/badge/MSMARCO--XI-15%2C679_Chunks-6f42c1?style=for-the-badge)](hhg-task2/data/)
[![Track](https://img.shields.io/badge/Track-%23RAGInGoa-ff5500?style=for-the-badge)](#ragingoa)

<p align="center">
  <b>Speak a question in English, Hindi, or Gujarati.</b><br>
  Get an answer extracted <i>strictly from indexed passages</i> — with source provenance, six guardrails, and per-stage millisecond diagnostics.
</p>

</div>

---

## 🚀 Quick Links & Navigation

The full Task #2 implementation, web application, test suite, and benchmarks are located in [`hhg-task2/`](hhg-task2/).

- **Full Documentation & Architectural Deep-Dive:** [`hhg-task2/README.md`](hhg-task2/README.md)
- **Live Local Web UI:** `http://127.0.0.1:8000` (Launch via `python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000` from `hhg-task2/`)
- **Full Test Suite:** `pytest tests/ -v` (**99/99 PASS**)

---

## ⚡ Measured Latency Summary (STT Excluded)

| Stage | P50 (Median) | P70 | P95 | P100 (Max) | Target Budget |
| :--- | :---:| :---:| :---:| :---:| :--- |
| **Query Embedding** | **10.59 ms** | 11.40 ms | 13.67 ms | 16.50 ms | `< 30 ms` |
| **Vector Retrieval + Reranking** | **10.73 ms** | 11.33 ms | 12.57 ms | 13.80 ms | `< 30 ms` |
| **6-Layer Guardrails** | **0.34 ms** | 0.39 ms | 0.43 ms | 0.54 ms | `< 5 ms` |
| **Answer Generation** | **0.00 ms** | 0.00 ms | 0.00 ms | 0.00 ms | `< 10 ms` |
| **TOTAL RAG LATENCY** | **`22.76 ms`** | **`23.84 ms`** | **`26.67 ms`** | **`31.33 ms`** | **`< 100 ms`** |

---

*For full details on chunking strategies, the NOAA tornado fix, and the complete test matrix, visit [hhg-task2/README.md](hhg-task2/README.md).*
