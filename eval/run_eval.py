"""Reproducible evaluation of retrieval and answers.

    python eval/run_eval.py                 # uses VECTOR_STORE/DATABASE_URL/LLM_PROVIDER from the env
    python eval/run_eval.py --store memory  # no database needed (CI)
    python eval/run_eval.py --judge         # additionally runs an LLM-as-judge (needs an LLM provider)

The corpus is rebuilt from the deterministic sample PDFs on every run (`--store postgres` resets the
tables), so results are reproducible. Writes eval/results/<label>.json and eval/results/<label>.md.

Metrics
  Retrieval  Precision@K, Recall@K (chunk level), evidence recall (share of gold phrases retrieved),
             Hit@K and MRR — for the full agent pipeline and for vector-only / keyword-only / hybrid ablations.
  Answers    status accuracy (answer vs abstain vs refuse), answer correctness (required facts present,
             forbidden content absent), faithfulness (lexical support of answer sentences by cited text),
             answer relevance (share of question terms addressed), citation precision and citation hit rate,
             injection resistance, paraphrase consistency.
  Latency    retrieval and end-to-end p50 / p95 (local machine; not a load test).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from policyqa import grounding  # noqa: E402
from policyqa.agent import PolicyQAAgent  # noqa: E402
from policyqa.config import Settings  # noqa: E402
from policyqa.embeddings import build_embedder  # noqa: E402
from policyqa.ingestion.pipeline import ingest_bytes  # noqa: E402
from policyqa.retrieval import HybridRetriever, reciprocal_rank_fusion  # noqa: E402
from policyqa.store import build_store  # noqa: E402

CORPUS = [ROOT / "data/policies/northwind_homeowners_ho3.pdf", ROOT / "data/policies/northwind_personal_auto.pdf",
          ROOT / "data/policies/northwind_pet_health.pdf", ROOT / "data/adversarial/coastal_flood_supplement.pdf"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def retrieval_metrics(ranked_ids: list[str], relevant: set[str], texts: dict[str, str], evidence: list[str],
                      k: int) -> dict:
    top = ranked_ids[:k]
    hits = [cid for cid in top if cid in relevant]
    rr = next((1 / i for i, cid in enumerate(top, start=1) if cid in relevant), 0.0)
    joined = " ".join(norm(texts[c]) for c in top)
    ev_found = sum(1 for e in evidence if norm(e) in joined)
    return {"precision": len(hits) / k, "recall": len(set(hits)) / len(relevant), "hit": float(bool(hits)),
            "mrr": rr, "evidence_recall": ev_found / len(evidence)}


def check_answer(answer: str, item: dict) -> tuple[bool, list[str]]:
    a = norm(answer)
    problems = []
    for group in item["must_include"]:
        if not any(norm(alt) in a for alt in group):
            problems.append(f"missing one of {group}")
    for bad in item["must_not_include"]:
        if norm(bad) in a:
            problems.append(f"contains forbidden '{bad}'")
    return not problems, problems


def llm_judge(chat_model, question: str, answer: str, evidence: str) -> dict | None:  # pragma: no cover
    from policyqa.agent.llm import message_text, parse_json_object

    prompt = ("Rate the ANSWER to the QUESTION using only the EVIDENCE. Respond with JSON "
              '{"faithfulness": 1-5, "relevance": 1-5}. Faithfulness: every claim supported by evidence. '
              "Relevance: answers what was asked.\n\n"
              f"QUESTION: {question}\n\nANSWER: {answer}\n\nEVIDENCE:\n{evidence[:6000]}")
    obj = parse_json_object(message_text(chat_model.invoke(prompt)))
    return obj if obj and {"faithfulness", "relevance"} <= obj.keys() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", choices=["postgres", "memory"], default=os.environ.get("VECTOR_STORE", "postgres"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--label", default=None)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--dataset", default="dataset.jsonl", help="dataset.jsonl (dev) or heldout.jsonl")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between questions (free-tier LLM APIs have per-minute rate limits)")
    ap.add_argument("--gate", action="append", default=[], metavar="METRIC=MIN",
                    help="fail (exit 1) if a summary metric is below MIN, e.g. --gate abstention_accuracy=1.0")
    args = ap.parse_args()

    settings = Settings(vector_store=args.store, retrieval_top_k=args.k)
    embedder = build_embedder(settings)
    store = build_store(settings, embedder.name)
    if args.store == "postgres":
        store.reset()  # type: ignore[attr-defined]
    store.init_schema()
    for pdf in CORPUS:
        ingest_bytes(pdf.read_bytes(), pdf.name, settings=settings, store=store, embedder=embedder)

    retriever = HybridRetriever(store, embedder, settings)
    agent = PolicyQAAgent(retriever, settings)
    judge_model = agent.llm if args.judge else None
    if args.judge and judge_model is None:
        print("--judge requires LLM_PROVIDER other than 'extractive'; skipping judge.", file=sys.stderr)

    # All chunk texts, to resolve gold evidence phrases to relevant chunk ids.
    chunk_text = {c.chunk_id: c.text for c in store.all_chunks()}

    dataset = [json.loads(line) for line in (ROOT / "eval" / args.dataset).read_text().splitlines() if line.strip()]
    integrity_errors = []
    for item in dataset:
        for ev in item["gold_evidence"]:
            if not any(norm(ev) in norm(t) for t in chunk_text.values()):
                integrity_errors.append(f"{item['id']}: gold phrase not found in corpus: {ev!r}")
    if integrity_errors:
        print("\n".join(integrity_errors), file=sys.stderr)
        return 2

    rows, ablation = [], {"vector_only": [], "keyword_only": [], "hybrid": []}
    for item in dataset:
        relevant = {cid for cid, t in chunk_text.items() if any(norm(e) in norm(t) for e in item["gold_evidence"])}
        t0 = time.perf_counter()
        if args.sleep and rows:
            time.sleep(args.sleep)
        res = agent.ask(item["question"])
        e2e_ms = (time.perf_counter() - t0) * 1000
        src_ids = [s["chunk_id"] for s in res.sources]
        row = {"id": item["id"], "category": item["category"], "question": item["question"],
               "expected_status": item["expected_status"], "status": res.status, "answer": res.answer,
               "cited_chunks": [c.chunk_id for c in res.citations], "retrieval_ms": res.retrieval_ms,
               "e2e_ms": round(e2e_ms, 2), "paraphrase_of": item["paraphrase_of"], "degraded": res.degraded}

        exp = item["expected_status"]
        row["status_ok"] = exp is None or res.status == exp or (exp == "answered" and res.status == "partial")
        content_ok, problems = check_answer(res.answer, item)
        row["content_ok"], row["problems"] = content_ok, problems
        row["correct"] = row["status_ok"] and content_ok

        if relevant:
            row["retrieval"] = retrieval_metrics(src_ids, relevant, chunk_text, item["gold_evidence"], args.k)
            q_words = grounding.raw_words(item["question"])
            qvec = embedder.embed([item["question"]])[0]
            vec = [h.chunk.chunk_id for h in store.vector_search(qvec, settings.retrieval_candidates)]
            kw = [h.chunk.chunk_id for h in store.keyword_search(q_words, settings.retrieval_candidates)]
            fused = sorted(reciprocal_rank_fusion([vec, kw], settings.rrf_k).items(), key=lambda kv: -kv[1])
            for name, ranking in (("vector_only", vec), ("keyword_only", kw), ("hybrid", [c for c, _ in fused])):
                ablation[name].append(retrieval_metrics(ranking, relevant, chunk_text, item["gold_evidence"], args.k))

        if res.status in ("answered", "partial") and res.citations:
            cited_text = [chunk_text[c.chunk_id] for c in res.citations]
            # Score only the answer body: the "Note: the documents don't mention X" caveat is a system message,
            # not a claim (audit H4: scoring it produced every sub-1.0 value in extractive mode).
            body = res.answer.split("\n\n")[0]
            row["faithfulness"] = round(grounding.answer_support(body, cited_text)[0], 4)
            row["answer_relevance"] = round(grounding.term_coverage(item["question"], [res.answer])[0], 4)
            if relevant:
                cited_rel = [c.chunk_id in relevant for c in res.citations]
                row["citation_precision"] = sum(cited_rel) / len(cited_rel)
                row["citation_hit"] = float(any(cited_rel))
            if judge_model is not None:  # pragma: no cover - needs an API key
                row["judge"] = llm_judge(judge_model, item["question"], res.answer, "\n\n".join(cited_text))
        rows.append(row)

    # ---------------------------------------------------------------- aggregate
    def agg_retrieval(items: list[dict]) -> dict:
        return {m: mean([x[m] for x in items]) for m in ("precision", "recall", "evidence_recall", "hit", "mrr")}

    by_cat: dict[str, dict] = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"n": 0, "correct": 0})
        c["n"] += 1
        c["correct"] += int(r["correct"])
    para = []
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        if r["paraphrase_of"]:
            o = by_id[r["paraphrase_of"]]
            a, b = set(r["cited_chunks"]), set(o["cited_chunks"])
            para.append({"pair": f"{o['id']} ~ {r['id']}", "both_correct": r["correct"] and o["correct"],
                         "same_status": r["status"] == o["status"],
                         "citation_jaccard": round(len(a & b) / len(a | b), 3) if a | b else 1.0})
    injection = [r for r in rows if r["category"] in ("prompt_injection", "document_injection")]
    unanswerable = [r for r in rows if r["expected_status"] == "insufficient_information"]
    answerable = [r for r in rows if r["expected_status"] == "answered"]
    retrieval_ms = [r["retrieval_ms"] for r in rows if r["retrieval_ms"] is not None]
    e2e = [r["e2e_ms"] for r in rows]

    summary = {
        "label": args.label or f"{Path(args.dataset).stem}-{agent.mode}-{embedder.name}-{args.store}",
        "dataset": args.dataset,
        "config": {"llm_mode": agent.mode, "embedding": embedder.name, "store": args.store, "k": args.k,
                   "chunk_size": settings.chunk_size_chars, "min_term_coverage": settings.min_term_coverage},
        "n_questions": len(rows),
        # LLM failures (e.g. free-tier rate limits) fall back to extractive answers. They are counted here so an
        # "LLM" score is never silently an extractive one. Re-run with --sleep if this is not 0.
        "degraded_answers": sum(1 for r in rows if r.get("degraded")),
        "overall_accuracy": mean([float(r["correct"]) for r in rows]),
        "status_accuracy": mean([float(r["status_ok"]) for r in rows]),
        "answerable_accuracy": mean([float(r["correct"]) for r in answerable]),
        "abstention_accuracy": mean([float(r["status"] == "insufficient_information") for r in unanswerable]),
        "false_abstention_rate": mean([float(r["status"] == "insufficient_information") for r in answerable]),
        "injection_resistance": mean([float(r["correct"]) for r in injection]),
        "faithfulness_mean": mean([r.get("faithfulness") for r in rows]),
        "faithfulness_note": ("NOT a quality signal in extractive mode: answers are verbatim quotes, so lexical "
                              "support is ~1.0 by construction. Meaningful only for LLM-generated answers."
                              if agent.mode == "extractive" else "lexical support of answer sentences by cited text"),
        "answer_relevance_mean": mean([r.get("answer_relevance") for r in rows]),
        "citation_precision": mean([r.get("citation_precision") for r in rows]),
        "citation_hit_rate": mean([r.get("citation_hit") for r in rows]),
        f"pipeline_retrieval@{args.k}": agg_retrieval([r["retrieval"] for r in rows if "retrieval" in r]),
        f"ablation_retrieval@{args.k}": {k: agg_retrieval(v) for k, v in ablation.items()},
        "paraphrase_consistency": {"pairs": para,
                                   "rate": mean([float(p["both_correct"] and p["same_status"]) for p in para])},
        "latency_ms": {"retrieval_p50": round(pct(retrieval_ms, .5), 1),
                       "retrieval_p95": round(pct(retrieval_ms, .95), 1),
                       "e2e_p50": round(pct(e2e, .5), 1), "e2e_p95": round(pct(e2e, .95), 1)},
        "by_category": {k: {**v, "accuracy": round(v["correct"] / v["n"], 3)} for k, v in sorted(by_cat.items())},
        "judge": ({"faithfulness": mean([r["judge"]["faithfulness"] for r in rows if r.get("judge")]),
                   "relevance": mean([r["judge"]["relevance"] for r in rows if r.get("judge")])}
                  if judge_model is not None else "not run"),
    }
    out_dir = ROOT / "eval/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = summary["label"]
    (out_dir / f"{label}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    (out_dir / f"{label}.md").write_text(render_markdown(summary, rows, args.k))
    print(json.dumps({k: v for k, v in summary.items() if k != "paraphrase_consistency"}, indent=2))
    print("\nFailures:")
    for r in rows:
        if not r["correct"]:
            print(f"  {r['id']:9} expected={r['expected_status']} got={r['status']} {r['problems']}")
    store.close()

    failed_gates = []
    for gate in args.gate:
        metric, _, minimum = gate.partition("=")
        value = summary.get(metric)
        if not isinstance(value, (int, float)) or value < float(minimum):
            failed_gates.append(f"{metric}={value} < {minimum}")
    if failed_gates:
        print("\nQUALITY GATE FAILED: " + "; ".join(failed_gates), file=sys.stderr)
        return 1
    if args.gate:
        print(f"\nquality gates passed: {', '.join(args.gate)}")
    return 0


def render_markdown(s: dict, rows: list[dict], k: int) -> str:
    lines = [f"# Evaluation results — `{s['label']}`", "",
             f"Questions: **{s['n_questions']}** · LLM mode: `{s['config']['llm_mode']}` · embeddings: "
             f"`{s['config']['embedding']}` · store: `{s['config']['store']}` · K={k}", "",
             *([f"**Warning: {s['degraded_answers']} answers fell back to extractive mode because the LLM call "
                "failed; this run is not a pure LLM measurement.**", ""] if s.get("degraded_answers") else []),
             "| Metric | Value |", "|---|---|"]
    for key in ("overall_accuracy", "status_accuracy", "answerable_accuracy", "abstention_accuracy",
                "false_abstention_rate", "injection_resistance", "faithfulness_mean", "answer_relevance_mean",
                "citation_precision", "citation_hit_rate"):
        lines.append(f"| {key} | {s[key]} |")
    lines.append(f"| paraphrase_consistency | {s['paraphrase_consistency']['rate']} |")
    lat = s["latency_ms"]
    lines += [f"| latency retrieval p50/p95 (ms) | {lat['retrieval_p50']} / {lat['retrieval_p95']} |",
              f"| latency end-to-end p50/p95 (ms) | {lat['e2e_p50']} / {lat['e2e_p95']} |", "",
              f"## Retrieval@{k}", "", "| Retriever | Precision | Recall | Evidence recall | Hit | MRR |",
              "|---|---|---|---|---|---|"]
    rows_r = [("agent pipeline", s[f"pipeline_retrieval@{k}"]), *s[f"ablation_retrieval@{k}"].items()]
    for name, m in rows_r:
        lines.append(f"| {name} | {m['precision']} | {m['recall']} | {m['evidence_recall']} | {m['hit']} "
                     f"| {m['mrr']} |")
    lines += ["", "## By category", "", "| Category | Correct / N |", "|---|---|"]
    for cat, v in s["by_category"].items():
        lines.append(f"| {cat} | {v['correct']} / {v['n']} |")
    lines += ["", "## Per question", "", "| id | expected | got | correct | notes |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['expected_status']} | {r['status']} | {'✅' if r['correct'] else '❌'} | "
                     f"{'; '.join(r['problems'])} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
