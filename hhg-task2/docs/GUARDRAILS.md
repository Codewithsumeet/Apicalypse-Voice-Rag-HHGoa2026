# Guardrails, Safety & Grounding Policies

> **Purpose:** Detailed architectural specification of the 6-layer guardrail defense and explicit refusal policies.

---

## 1. The 6 Guardrail Layers

```
[User Query]
     │
     ├── 1. UnsafeInputGuardrail           (Regex pattern match)                 ──► Execution: < 0.10 ms
     ├── 2. OffTopicGuardrail              (MSMARCO Centroid Cosine Distance)    ──► Execution: < 0.10 ms
     ├── 3. LanguageConsistencyGuardrail   (Query Script vs Evidence Script)     ──► Execution: < 0.10 ms
     ├── 4. AnswerabilityGuardrail         (Intent & Key Term Match)             ──► Execution: < 0.25 ms
     ├── 5. CoverageGuardrail              (Dense Cosine >= 0.58 Threshold)      ──► Execution: < 0.10 ms
     └── 6. GroundingGuardrail             (Attested Substring / Semantic Check) ──► Execution: < 0.01 ms
```

---

## 2. Guardrail Specifications

| Guardrail | Layer | Trigger Condition | Outcome / Refusal Reason | Latency |
| :--- | :--- | :--- | :--- | :---:|
| **`UnsafeInputGuardrail`** | Pre-Retrieval | Match against dangerous patterns (weapons, explosives, toxic acts) | `RefusalReason.UNSAFE` | `0.10 ms` |
| **`OffTopicGuardrail`** | Pre-Retrieval | Cosine distance to corpus centroid `< 0.10` | `RefusalReason.UNGROUNDED` | `0.08 ms` |
| **`LanguageConsistencyGuardrail`** | Post-Retrieval | Query language does not match evidence language (e.g. English query matching Hindi passage) | `RefusalReason.UNGROUNDED` | `0.10 ms` |
| **`AnswerabilityGuardrail`** | Post-Retrieval | Retrieved text shares stop-words but lacks specific answer terms or mismatches intent (e.g. temporal query) | `RefusalReason.UNGROUNDED` | `0.20 ms` |
| **`CoverageGuardrail`** | Post-Retrieval | Dense semantic cosine score `< 0.58` and token overlap `< 0.15` | `RefusalReason.UNGROUNDED` | `0.10 ms` |
| **`GroundingGuardrail`** | Post-Generation | Answer is not an attested exact substring or semantic similarity to context `< 0.58` | `RefusalReason.UNGROUNDED` | `0.01 ms` |

---

## 3. Refusal as an Architectural Feature

The system treats refusal as an intentional safety mechanism rather than a pipeline error:
- **`RefusalReason.UNSAFE`**: Harmful, dangerous, or toxic input.
- **`RefusalReason.UNGROUNDED`**: Missing corpus evidence, temporal queries, out-of-domain requests, or cross-lingual mismatch.
- **`RefusalReason.OFF_TOPIC`**: Conversational or non-informational queries.
- **`RefusalReason.SYSTEM_ERROR`**: Hardware, network, or third-party STT connection faults.

---

## 4. The NOAA Tornado Regression Fix

In initial BM25 retrieval, querying *"What is machine learning?"* matched English stop-words (`"is"`, `"what"`) against noisy multilingual texts, returning an unrelated NOAA tornado passage (`1099915_5`).

**The Permanent Solution:**
1. **Multilingual Dense Space:** MiniLM-L12-v2 maps queries and passages semantically rather than relying on lexical stop-word counts.
2. **Answerability Intent Scoring:** [`AnswerabilityGuardrail`](../src/guardrails/answerability.py) filters stop-words and validates that key query nouns appear meaningfully in candidate passages.
3. **Language-Consistent Reranking:** Biases candidates by `+0.30` for query language matches, eliminating false-positive lexical noise.
