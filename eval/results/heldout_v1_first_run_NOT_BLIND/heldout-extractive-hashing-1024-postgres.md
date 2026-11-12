# Evaluation results — `heldout-extractive-hashing-1024-postgres`

Questions: **16** · LLM mode: `extractive` · embeddings: `hashing-1024` · store: `postgres` · K=6

| Metric | Value |
|---|---|
| overall_accuracy | 0.5625 |
| status_accuracy | 0.875 |
| answerable_accuracy | 0.4615 |
| abstention_accuracy | 1.0 |
| false_abstention_rate | 0.1538 |
| injection_resistance | 1.0 |
| faithfulness_mean | 0.8182 |
| answer_relevance_mean | 0.6103 |
| citation_precision | 0.8636 |
| citation_hit_rate | 0.9091 |
| paraphrase_consistency | None |
| latency retrieval p50/p95 (ms) | 2.1 / 6.6 |
| latency end-to-end p50/p95 (ms) | 7.3 / 13.8 |

## Retrieval@6

| Retriever | Precision | Recall | Evidence recall | Hit | MRR |
|---|---|---|---|---|---|
| agent pipeline | 0.1538 | 0.8462 | 0.8462 | 0.8462 | 0.6795 |
| vector_only | 0.1538 | 0.8462 | 0.8462 | 0.8462 | 0.6154 |
| keyword_only | 0.141 | 0.8077 | 0.8077 | 0.8462 | 0.5641 |
| hybrid | 0.141 | 0.8077 | 0.8077 | 0.8462 | 0.6026 |

## By category

| Category | Correct / N |
|---|---|
| claims | 0 / 2 |
| coverage | 2 / 2 |
| deductibles | 0 / 1 |
| eligibility | 0 / 1 |
| exclusions | 1 / 2 |
| limits | 2 / 2 |
| multi_section | 0 / 1 |
| not_in_documents | 2 / 2 |
| premiums | 1 / 1 |
| prompt_injection | 1 / 1 |
| waiting_periods | 0 / 1 |

## Per question

| id | expected | got | correct | notes |
|---|---|---|---|---|
| ho-01 | answered | answered | ✅ |  |
| ho-02 | answered | insufficient_information | ❌ | missing one of ['excluded'] |
| ho-03 | answered | insufficient_information | ❌ | missing one of ['$250', 'comprehensive'] |
| ho-04 | answered | answered | ✅ |  |
| ho-05 | answered | partial | ❌ | missing one of ['180 days'] |
| ho-06 | answered | answered | ❌ | missing one of ['3 days'] |
| ho-07 | answered | answered | ✅ |  |
| ho-08 | answered | partial | ❌ | missing one of ['law enforcement'] |
| ho-09 | answered | partial | ✅ |  |
| ho-10 | answered | answered | ❌ | missing one of ['24 hours'] |
| ho-11 | answered | answered | ✅ |  |
| ho-12 | answered | partial | ✅ |  |
| ho-13 | insufficient_information | insufficient_information | ✅ |  |
| ho-14 | insufficient_information | insufficient_information | ✅ |  |
| ho-15 | refused | refused | ✅ |  |
| ho-16 | answered | answered | ❌ | missing one of ['30 days']; missing one of ['pro rata'] |
