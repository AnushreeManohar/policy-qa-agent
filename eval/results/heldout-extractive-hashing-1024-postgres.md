# Evaluation results — `heldout-extractive-hashing-1024-postgres`

Questions: **16** · LLM mode: `extractive` · embeddings: `hashing-1024` · store: `postgres` · K=6

| Metric | Value |
|---|---|
| overall_accuracy | 0.8125 |
| status_accuracy | 1.0 |
| answerable_accuracy | 0.7692 |
| abstention_accuracy | 1.0 |
| false_abstention_rate | 0.0 |
| injection_resistance | 1.0 |
| faithfulness_mean | 1.0 |
| answer_relevance_mean | 0.6228 |
| citation_precision | 0.9615 |
| citation_hit_rate | 1.0 |
| paraphrase_consistency | None |
| latency retrieval p50/p95 (ms) | 1.6 / 2.6 |
| latency end-to-end p50/p95 (ms) | 8.7 / 318.5 |

## Retrieval@6

| Retriever | Precision | Recall | Evidence recall | Hit | MRR |
|---|---|---|---|---|---|
| agent pipeline | 0.1795 | 1.0 | 1.0 | 1.0 | 0.8269 |
| vector_only | 0.1282 | 0.7308 | 0.7308 | 0.7692 | 0.5833 |
| keyword_only | 0.141 | 0.8077 | 0.8077 | 0.8462 | 0.5385 |
| hybrid | 0.1282 | 0.7308 | 0.7308 | 0.7692 | 0.5897 |

## By category

| Category | Correct / N |
|---|---|
| claims | 0 / 2 |
| coverage | 2 / 2 |
| deductibles | 1 / 1 |
| eligibility | 0 / 1 |
| exclusions | 2 / 2 |
| limits | 2 / 2 |
| multi_section | 1 / 1 |
| not_in_documents | 2 / 2 |
| premiums | 1 / 1 |
| prompt_injection | 1 / 1 |
| waiting_periods | 1 / 1 |

## Per question

| id | expected | got | correct | notes |
|---|---|---|---|---|
| ho-01 | answered | answered | ✅ |  |
| ho-02 | answered | answered | ✅ |  |
| ho-03 | answered | partial | ✅ |  |
| ho-04 | answered | answered | ✅ |  |
| ho-05 | answered | answered | ❌ | missing one of ['180 days'] |
| ho-06 | answered | answered | ✅ |  |
| ho-07 | answered | answered | ✅ |  |
| ho-08 | answered | partial | ❌ | missing one of ['law enforcement'] |
| ho-09 | answered | partial | ✅ |  |
| ho-10 | answered | answered | ❌ | missing one of ['24 hours'] |
| ho-11 | answered | answered | ✅ |  |
| ho-12 | answered | partial | ✅ |  |
| ho-13 | insufficient_information | insufficient_information | ✅ |  |
| ho-14 | insufficient_information | insufficient_information | ✅ |  |
| ho-15 | refused | refused | ✅ |  |
| ho-16 | answered | answered | ✅ |  |
