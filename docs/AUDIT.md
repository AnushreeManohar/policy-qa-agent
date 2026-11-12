# Audit report — Insurance Policy Q&A Agent

**Date:** 2026-11-12 · The audit ran on a local history (initial build, blind-set freeze, audit fixes); the
public repository was published as a single squashed commit, so those intermediate commits are not on GitHub.
The blind set can still be checked by its hash: `sha256sum eval/blind_v2.jsonl` → `58cd1fe7…`.

**Conflict of interest.** This audit was performed by the same assistant that built the project, so it is a
self-audit, not an independent review. Three controls limit the bias: (1) a new blind test set was written from
the policy text and committed *before* any fix (`eval/blind_v2.jsonl`, sha256 `58cd1fe7…`) and run exactly
once at the end; (2) citations were verified with a different PDF extractor (poppler `pdftotext`) than the
application uses (pypdf); (3) every adversarial case had its pass criterion written before it was first run.
A human reviewer reading this repository cold would still be the better test.

**Environment.** 2 vCPUs, 7 GB RAM, Python 3.13, PostgreSQL 16 + pgvector 0.6.0, Node 22, headless Chromium.
No LLM API key; Hugging Face downloads blocked; no Docker daemon. GitHub releases reachable (used for spaCy's
`en_core_web_md` word vectors).

---

## 1. Overall assessment

The application genuinely works end to end: PDFs are parsed, chunked by section with page provenance,
embedded, stored in PostgreSQL/pgvector, retrieved with hybrid search, run through a LangGraph state machine,
and returned with citations that were independently confirmed to be on the cited page (35/35). It is a real
RAG pipeline, not an LLM with documents pasted into a prompt; in its default mode there is no LLM at all.

Its weakness is **answer quality on unseen wording**: on the blind set it answered **61.9%** of answerable
questions correctly. It never invented coverage and abstained correctly on every unanswerable question
(100%), but it abstained wrongly on 19% of answerable ones and quoted a real-but-irrelevant clause with
confidence on 4 of 26 questions. The LLM mode that should fix this **has never been run against a real
model**. The audit also found that a published claim ("56.3% blind") was not true, and that the default
configuration allowed anyone to poison the knowledge base. Both are fixed.

## 2. Architecture

**Well designed**
- Clear layering: `ingestion/` → `store/` (Protocol with PostgreSQL and in-memory implementations) →
  `retrieval.py` → `agent/` (LangGraph) → `api/`. Configuration is centralised and 12-factor.
- Trust decisions (abstain, broaden, verify) are explicit graph edges with their own tests.
- Idempotent, transactional ingestion keyed by file hash; schema guards against embedder mismatch.
- Errors are typed (`StoreError`, `LLMError`, `PdfValidationError`) and mapped to clean HTTP responses.

**Needs improvement**
- Answer quality rests on a growing set of lexical heuristics (stop lists, filler lists, a thesaurus,
  stemming rules, IDF gating). Each is defensible; together they are brittle, and the audit had to fix
  five of them. This is the honest ceiling of LLM-free answering.
- `grounding.py` is shared by the agent and the evaluator. Convenient, but it means the evaluator's
  "answer relevance" metric uses the same term logic the agent optimises.
- Single process, CPU-bound; the in-process rate limiter blocks naive horizontal scaling.

## 3. RAG evaluation

**Pipeline traced stage by stage (verified by execution):** question → `guard_input` (injection heuristics)
→ `plan_query` (sub-question split, policy-type routing, synonym expansion) → `retrieve` (embedding +
pgvector cosine search + PostgreSQL full-text search, fused with RRF, near-duplicates and flagged chunks kept
out of the evidence slots) → `grade_evidence` (IDF-weighted coverage of topic terms) → `generate`
(extractive sentence selection, or LLM with `<source>` blocks) → `verify` (citations exist, unflagged, each
sentence supported) → answer + citations.

**Datasets.** Dev 52 (tuned on), held-out v1 16 (no longer blind), **blind v2 26 (run once)**, adversarial
57. Corpus: 3 fictional policies + 1 poisoned notice, ~28 chunks.

| Metric | Blind v2 | Dev | Method and limitation |
|---|---|---|---|
| Overall accuracy | **69.2%** | 96.2% | Correct status AND required facts present (substring), forbidden content absent. Substring checks can pass an answer that says "not excluded". |
| Answerable accuracy | **61.9%** | 95.3% | Same, on questions with an answer in the documents. |
| Abstention accuracy | **100%** (3/3) | 100% (5/5) | Tiny n; a single miss moves it 33 points. |
| False abstention | **19.0%** | 0% | Share of answerable questions the agent refused to answer. |
| Injection resistance | 100% (1/1) | 100% (5/5) | Plus 7 adversarial injection cases (A35–A41), all pass. |
| Precision@6 / Recall@6 | 0.18 / 0.94 | 0.21 / 1.00 | Chunk is relevant if it contains a gold evidence phrase. Precision is ~0.2 by construction (one relevant chunk, six returned). Recall@6 over a ~28-chunk corpus is an easy bar. |
| MRR / Hit@6 | 0.62 / 0.95 | 0.77 / 1.00 | MRR is the more informative retrieval number here. |
| Citation hit rate | 82.4% | 97.7% | A cited chunk contains gold evidence. |
| Citation page accuracy | 35/35 (adversarial run) | — | Each cited sentence found on the cited page by poppler `pdftotext`. Verifies traceability, not relevance. |
| Faithfulness | 1.0 | 1.0 | **Not meaningful in extractive mode** (verbatim quotes). Before the audit, the only deductions came from the app's own caveat note. |
| Answer relevance | 0.54 | 0.69 | Lexical overlap with the question; shares logic with the agent, so it is not independent. |
| Retrieval latency p50/p95 | 1.4 / 2.0 ms | 1.3 / 2.3 ms | Local, single user. |
| End-to-end p50/p95 | 8.3 / 15.1 ms | 8.7 / 12.3 ms | No LLM. With an LLM, model latency would dominate. |

**Questions asked by the audit brief**
- *Correct documents retrieved?* Yes for 95% of blind questions (hit@6). Policy-type routing works; an
  explicit user scope is now never broadened (it was before).
- *Irrelevant chunks?* Five of six returned chunks are irrelevant by construction; harmless for extractive
  answering because sentence selection filters them, but they would cost LLM tokens.
- *Duplicates?* Before the audit, a repeated clause filled all six slots (measured on a 119-page binder).
  Fixed: near-duplicate suppression.
- *Exclusions vs coverage, deductibles vs premiums?* Quoted text preserves the distinction ("is excluded";
  A09: asked whether $1,842 is the deductible, it quoted the premium clause). The agent does not
  *interpret*; the user reads the clause.
- *Conflicting sections?* A conflicting endorsement ($2,500 vs $1,000) was uploaded: the answer quoted both
  with separate citations (A31 pass). It does not say "these conflict".
- *Admits missing information?* Yes, 100% on all sets, and it names the missing terms.
- *Fabricated citations?* Not possible in extractive mode (ids are generated from retrieved chunks). In LLM
  mode `verify` rejects out-of-range ids (tested with a scripted model only).

## 4. LangGraph assessment

Eight nodes, three conditional routers, a typed state with an additive `trace` reducer. Every path was
executed and recorded:

| Scenario | Path |
|---|---|
| Answerable | guard → plan → retrieve → grade → generate → verify |
| Weak evidence, auto-detected type | … grade → **retrieve (broadened)** → grade → generate → verify |
| Unanswerable | … grade → retrieve → grade → **abstain** |
| Injection | guard → **refuse** |
| LLM uncited, then fixed* | … generate → verify → **generate** → verify |
| LLM hallucinates twice* | … generate → verify → generate → verify → **abstain** |
| LLM reports insufficient* | … generate → verify → abstain |
| LLM raises* | falls back to extractive answer, `degraded: true` (before the audit: 502) |
| Database down | `StoreError` → HTTP 503 |

\*with a scripted chat model, not a real LLM.

**Verdict:** LangGraph is used meaningfully: real conditional routing, a bounded retry loop, and explicit
termination (LangGraph's recursion limit is a second safety net). A sceptic could fairly say the same logic
fits in ~60 lines of plain Python; the graph's value is explicitness, per-node tracing and testability, not
capability. No checkpointing or persistence (stateless per request), which is appropriate here.

## 5. Testing results

| Suite | Result |
|---|---|
| Unit + API + agent + PostgreSQL integration | **109 passed**, 0 failed, 91% line coverage |
| Same, semantic layer disabled (`NLP_MODEL=none`) | 96 passed, 11 skipped (no DB in that run), 2 deselected (semantic-only) |
| Adversarial over HTTP (57 cases) | before fixes **43/57** · after fixes **49/57** |
| Browser end to end (headless Chromium) | 14/14 |
| Evaluation quality gate (dev) | passes (accuracy >= 0.9, abstention 1.0, injection 1.0) |
| Docker image build | **passed in CI** (GitHub Actions run #1, 2026-11-12); not run in the audit environment (no daemon) |
| GitHub Actions | **passed**: run #1 green on all four jobs (tests with PostgreSQL + evaluation gate, frontend build, browser end-to-end, Docker build) |
| Real LLM | **not executed** (no key, no model download) |

**Tests that passed without testing the right thing (found by the audit):**
`test_ingest_search_duplicate_and_cascade_delete` called PostgreSQL keyword search with pre-stemmed words
(`"earthquak"`), which happened to match, so the double-stemming bug (H1) went unnoticed. It now uses raw
words, and a parity test checks PostgreSQL against the in-memory store. One test written during this audit
contained `assert … or True`; it was removed before commit.

**Remaining adversarial failures (8):** A08 hitting a tree → collision deductible; A11 premium vs. pet age;
A12 kitten accident waiting period (wrong sentence chosen); A17 time limit to sue (wrong sentence); A18 fence
limit figure ($35,000 not quoted); A29/A30 comparisons across policies; A57 follow-up question. All need
semantic reasoning that extractive mode does not have. Note: the adversarial suite became partly a
development set during this audit (A41 in particular was used to design a fix).

## 6. Security assessment

| Severity | Finding | Status |
|---|---|---|
| Critical | Default config: unauthenticated uploads → an anonymous "amendment" made the agent state flood is "covered in full" (A42/A43); no way to delete without a key | **Fixed**: uploads need `API_KEY` unless `ALLOW_ANONYMOUS_UPLOADS=true` |
| High | Poisoned chunks consumed evidence slots and displaced real clauses (a way to *suppress* a clause) | **Fixed** |
| High | Injection without trigger words quoted verbatim (A41) | **Mitigated** (two heuristics); detection remains pattern-based and evadable |
| Medium | `API_KEY=` (empty) treated as a real key | **Fixed** |
| Low | Uploaded filenames are logged (could contain names) | Open |
| Low | `/metrics` and `/api/health` unauthenticated; health reveals embedder and LLM mode | Open (normal for internal deployments; protect at the gateway) |
| Info | Single shared API key; no per-user auth or tenant isolation | Documented limitation |

**Verified controls (executed):** SQL metacharacters inert in question, `policy_type` and full-text terms;
malformed JSON, wrong types, empty and over-long questions → clean 422; non-PDF, renamed executable,
encrypted PDF → 422; 1.2 MB → 400 MB decompression bomb rejected in 0.1 s (+49 MB RSS) by pypdf's limits;
rate limit returned 429 on request 31 at the default 30/min; CORS allow-list enforced; no secrets or question
text in logs; error bodies contain no internals (database host and password, provider key and message all
checked).

## 7. Performance assessment

| Measurement | Before audit | After audit (semantic layer on) |
|---|---|---|
| Throughput, 1 → 64 concurrent | 112 → 118 req/s | 82 → 65 req/s |
| p95 latency at 64 concurrent | 705 ms | 1,450 ms |
| Server memory | 180 MB | 455 MB |
| Errors under load | 0 | 0 (429s only when the test itself exceeded the rate limit) |
| Ingest 119-page PDF | 1.2 s, 425 chunks, 81 MB peak | unchanged path |
| Retrieval, 451-chunk corpus | p50 2.0 ms | — |

**Bottleneck:** one CPU-bound Python process (tokenisation, sentence scoring, spaCy tagging under a lock);
latency grows linearly with concurrency. **Biggest wins, in order:** (1) multiple workers with a shared
(Redis) rate limiter; (2) cache spaCy analyses per question; (3) with an LLM, stream responses — the model
would dominate latency. No scalability claim is made beyond these measurements.

## 8. Bugs discovered

| # | Severity | Issue | Evidence | Fix |
|---|---|---|---|---|
| C1 | Critical | "Blind" held-out score was not blind | "guns", "sued", "lawyer" in the thesaurus appear only in held-out questions | Claim withdrawn; blind v2 frozen and run once |
| C2 | Critical | Answer poisoning via anonymous upload | A42 → 201, A43 answer "covered in full" | Upload auth by default |
| H1 | High | PostgreSQL never matched "deductible", "eligible", "driving", "premises" | `to_tsvector('deductible') @@ to_tsquery('deductibl')` = false | Raw words to stores; parity test |
| H2 | High | False abstentions and wrong sentences on unseen wording | 7/21 basic adversarial failures | Numbers don't gate; missing verbs don't count; stemming; soft matching for selection |
| H3 | High | Subtle document injection echoed | A41 | Heuristics added |
| H4 | High | Faithfulness metric circular | all sub-1.0 scores came from the caveat note | Caveat excluded; metric labelled |
| H5 | High | Flagged chunks displaced real evidence | flood exclusion missing from top 6 on PostgreSQL | Flagged chunks outside the evidence budget |
| M1 | Medium | Explicit scope overridden | A52 answered from other policies | Never broaden a user scope |
| M2 | Medium | Duplicate clauses fill all slots | 6/6 copies on a 119-page binder | Near-duplicate suppression |
| M3 | Medium | LLM outage → 502 | graph path test | Extractive fallback, `degraded` flag in API and UI |
| M4 | Medium | Chat-like UI, stateless backend | A57 | Placeholder says questions are independent; documented |
| M5 | Medium | `API_KEY=` empty blocks uploads | default-config probe returned 401 | Empty secrets treated as unset |
| M6 | Medium | Short plurals unstemmed ("dogs" ≠ "dog") | caught by a new test | Plural rule for 3-letter stems |
| L1 | Low | `reset-db` crashed on memory store | AttributeError | Clean error, exit 2 |
| L2 | Low | Question limit hardcoded at 1000 | schema `max_length=1000` | Configurable limit enforced in route |
| R1 | — | **Regression introduced by the audit**: word vectors in the evidence gate dropped abstention to 50% | "motorcycle", "identity theft" answered | Reverted; regression tests added |
| R2 | — | **Regression introduced by the audit**: soft matching quoted the Canada clause for a Mexico question | existing test failed | Soft match only when the exact word is absent |
| R3 | — | **Regression introduced by the audit**: topics-only gate abstained on "hail damages my roof" | A07 | Present words always count |

## 9. Changes made

`backend/src/policyqa/`: `grounding.py` (query words, soft matching, topic-aware coverage), `semantics.py`
(new), `text.py` (stemming), `retrieval.py` (raw keyword words, duplicate and flagged-chunk handling),
`store/postgres.py` and `store/memory.py` (raw-word keyword search), `agent/graph.py` (user scope, LLM
fallback, flagged chunks, soft selection), `embeddings.py` (optional word-vector embedder), `security.py`
(two patterns), `config.py` (secure upload default, fallback, empty secrets), `api/app.py` and
`api/schemas.py` (upload auth, health fields, `degraded`, question limit), `cli.py` (reset-db).
`frontend/src/`: degraded notice, follow-up hint, up-front API key field. Tests: 109 total (84 before the audit), several rewritten where the audit changed intended behaviour.
`eval/`: `blind_v2.jsonl`, `adversarial_audit.py` (new), faithfulness fix, raw-word ablation.
`README.md`: corrected claims and numbers. `Dockerfile`, CI, `.env.example`, `constraints.txt`: semantic extra.

## 10. Remaining limitations

Not verified: real LLM answers, transformer embeddings, `docker compose up` end to end, behaviour with
real (non-synthetic) policy PDFs, filtered HNSW recall at scale. Known weak: answer quality on unseen wording
(62% blind), cross-policy comparison, follow-up questions, injection detection beyond known patterns,
single-key auth, single-process throughput.

## 11. Portfolio readiness scores

| Area | Score | Evidence |
|---|---|---|
| Architecture | **7/10** | Clean layering and typed boundaries; answer logic is a pile of interacting heuristics |
| Python engineering | **7/10** | Typed, linted, 91% coverage; but the audit found 15 real bugs, one in core retrieval |
| RAG implementation | **6/10** | Real hybrid retrieval with verified provenance; generation is extractive, no real embedder or LLM measured |
| LangGraph implementation | **6/10** | Genuine branching, retry and fallback, all executed; modest problem that plain Python could also solve |
| Retrieval quality | **6/10** | Blind recall@6 0.94, MRR 0.62 — on a 28-chunk corpus, which flatters any retriever |
| Answer quality | **4/10** | 62% on answerable blind questions; confident wrong-clause answers on 15% of blind questions |
| Citation accuracy | **8/10** | 35/35 on the cited page by an independent extractor; but 18% of blind citations point at a real, non-answering clause |
| Testing | **8/10** | Unit, integration, adversarial, blind, browser and load tests; one key test hid H1 until this audit |
| Security | **6/10** | Many verified controls; critical default fixed only now; heuristic injection defence; single shared key |
| Frontend | **7/10** | All features wired to working endpoints and verified in a browser; stateless "chat" |
| DevOps | **7/10** | CI green on first run in a clean environment (tests, PostgreSQL, eval gate, browser test, image build); Compose stack not yet run end to end; no deployment |
| Documentation | **7/10** | Thorough and now honest; it previously contained a false "blind" claim |

## 12. Verdict

**FUNCTIONAL BUT NEEDS IMPROVEMENT.**

The core application works and is honestly measured, and its safety behaviour (no invented coverage,
correct abstention, injection resistance, traceable citations) is solid. It is not yet a convincing *AI
engineering* portfolio piece, because the parts an AI-engineering interviewer cares most about — LLM
generation quality and semantic retrieval — have never been measured. Three steps would move it to
**Portfolio ready**:

1. Run `python eval/run_eval.py --judge --dataset blind_v2.jsonl` with `LLM_PROVIDER=anthropic` (or OpenAI)
   and `EMBEDDING_PROVIDER=openai`, and publish the comparison with the extractive baseline. (The blind set is
   now used; write a fresh one first, or report it clearly as a second run.)
2. ~~Push to GitHub and get CI green, including the Docker build.~~ Done: CI run #1 passed on 2026-11-12.
3. Replace or augment the fictional corpus with a few real public policy forms and re-run the evaluation.

---

## Appendix: interview questions this implementation invites

★ = needs real depth; an interviewer will probe past the first answer.

1. **★ Your blind accuracy is 62% but dev is 95%. Why, and what does that tell you?** Dev was used for
   tuning; lexical heuristics overfit the dev phrasing. The gap is the measured cost of no semantic
   understanding. Be ready to name concrete failures (tree → collision).
2. **★ You once reported a "blind" score that wasn't. How did that happen, and how do you prevent it?**
   Synonyms were added while the held-out set existed. Prevention: freeze and hash the test set before
   changes, run once, keep the set out of the working context.
3. **Why hybrid search and RRF instead of a weighted sum?** RRF uses ranks only, so no calibration
   between cosine similarity and `ts_rank_cd`; k=60 damps the dominance of rank 1.
4. **★ Explain the PostgreSQL stemming bug.** The app stemmed "deductible" to "deductibl"; `to_tsquery`
   stemmed that again and never matched the indexed "deduct". Lesson: one stemmer per index; parity tests.
5. **How does the agent decide to abstain?** IDF-weighted share of the question's terms found in the
   evidence; absent nouns and adjectives count against, absent verbs and numbers don't; threshold 0.5.
6. **★ Why did adding word vectors to the gate hurt?** Similar is not equivalent: "identity" ~ "theft",
   "Mexico" ~ "Canada". Fine for choosing a sentence, dangerous for deciding whether to answer at all.
7. **What does LangGraph give you here that plain Python wouldn't?** Explicit, typed state transitions,
   per-node traces, bounded retry and testable routing. Not new capability; be honest about that.
8. **★ How do you defend against prompt injection in retrieved documents?** Layers: flag at ingestion,
   exclude flagged text from evidence and from slots, `<source>` data framing, verification. Admit that
   patterns can be evaded and that the real boundary is who may add documents.
9. **How are citations made trustworthy?** Chunks carry page ranges from extraction; answers cite chunk ids
   assigned by the system; an independent extractor confirmed every cited sentence on its page.
10. **Why is your faithfulness metric not meaningful?** Extractive answers are verbatim, so lexical support
    is 1.0 by construction. You'd need an LLM judge or NLI on generated answers.
11. **★ What happens at 10,000 policies?** pgvector HNSW still fine; filtered HNSW may return too few rows
    with selective filters (pgvector 0.6); BM25-style ranking needs tuning; RRF candidate pool size matters;
    duplicate boilerplate across policies becomes the main relevance problem.
12. **Why is throughput only ~65 req/s?** CPU-bound single process plus spaCy under a lock; scale with
    workers and a shared rate limiter; with an LLM the bottleneck moves to the model.
13. **What would you change to support follow-up questions?** Send conversation history, rewrite the
    follow-up into a standalone question (LLM or rules), then retrieve; keep citations per turn.
14. **★ How would you evaluate the LLM mode properly?** Fixed blind set, LLM judge plus human spot-checks,
    faithfulness via NLI or claim extraction, cost and latency, and comparison against the extractive
    baseline on the same questions.
15. **How do you stop a confident wrong-clause answer?** Today: nothing beyond lexical scoring. Options: a
    cross-encoder re-ranker on sentence-question pairs, a minimum-relevance threshold for the selected
    sentence, or LLM verification that the quote answers the question.
