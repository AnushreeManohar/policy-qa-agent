# Evaluation results — `blind_v2-single-run`

Questions: **26** · LLM mode: `extractive` · embeddings: `hashing-1024` · store: `postgres` · K=6

| Metric | Value |
|---|---|
| overall_accuracy | 0.6923 |
| status_accuracy | 0.8462 |
| answerable_accuracy | 0.619 |
| abstention_accuracy | 1.0 |
| false_abstention_rate | 0.1905 |
| injection_resistance | 1.0 |
| faithfulness_mean | 1.0 |
| answer_relevance_mean | 0.5368 |
| citation_precision | 0.8235 |
| citation_hit_rate | 0.8235 |
| paraphrase_consistency | None |
| latency retrieval p50/p95 (ms) | 1.4 / 2.0 |
| latency end-to-end p50/p95 (ms) | 8.3 / 15.1 |

## Retrieval@6

| Retriever | Precision | Recall | Evidence recall | Hit | MRR |
|---|---|---|---|---|---|
| agent pipeline | 0.1825 | 0.9365 | 0.9524 | 0.9524 | 0.6167 |
| vector_only | 0.1587 | 0.7619 | 0.7619 | 0.7619 | 0.5159 |
| keyword_only | 0.1587 | 0.7937 | 0.7857 | 0.8095 | 0.4794 |
| hybrid | 0.1667 | 0.8095 | 0.8095 | 0.8095 | 0.623 |

## By category

| Category | Correct / N |
|---|---|
| claims | 2 / 3 |
| comparison | 0 / 1 |
| coverage | 0 / 1 |
| deductibles | 0 / 1 |
| eligibility | 0 / 2 |
| exclusions | 6 / 7 |
| hallucination_trap | 1 / 1 |
| limits | 2 / 3 |
| multi_section | 1 / 1 |
| not_in_documents | 3 / 3 |
| premiums | 1 / 1 |
| prompt_injection | 1 / 1 |
| waiting_periods | 1 / 1 |

## Per question

| id | expected | got | correct | notes |
|---|---|---|---|---|
| b-01 | answered | insufficient_information | ❌ | missing one of ['anywhere in the world'] |
| b-02 | answered | partial | ✅ |  |
| b-03 | answered | partial | ✅ |  |
| b-04 | answered | answered | ❌ | missing one of ['No deductible'] |
| b-05 | answered | answered | ✅ |  |
| b-06 | answered | answered | ✅ |  |
| b-07 | answered | insufficient_information | ❌ | missing one of ['8 weeks'] |
| b-08 | answered | insufficient_information | ❌ | missing one of ['fewer than four wheels'] |
| b-09 | answered | answered | ✅ |  |
| b-10 | answered | answered | ✅ |  |
| b-11 | answered | answered | ❌ | missing one of ['$200'] |
| b-12 | answered | answered | ✅ |  |
| b-13 | answered | partial | ✅ |  |
| b-14 | answered | answered | ✅ |  |
| b-15 | answered | partial | ✅ |  |
| b-16 | answered | partial | ✅ |  |
| b-17 | answered | answered | ✅ |  |
| b-18 | answered | insufficient_information | ❌ | missing one of ['racing'] |
| b-19 | answered | answered | ✅ |  |
| b-20 | answered | partial | ❌ | missing one of ['$500'] |
| b-21 | answered | answered | ❌ | missing one of ['60 days'] |
| b-22 | insufficient_information | insufficient_information | ✅ |  |
| b-23 | insufficient_information | insufficient_information | ✅ |  |
| b-24 | insufficient_information | insufficient_information | ✅ |  |
| b-25 | None | answered | ✅ |  |
| b-26 | None | insufficient_information | ✅ |  |
