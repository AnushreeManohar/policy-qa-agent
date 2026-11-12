"""Side-by-side comparison of evaluation summaries.

    python eval/compare_results.py eval/results/A.json eval/results/B.json [...] > eval/results/comparison.md
"""

import json
import sys
from pathlib import Path

METRICS = [
    ("overall_accuracy", "Overall accuracy"), ("answerable_accuracy", "Correct on answerable"),
    ("abstention_accuracy", "Correct abstentions"), ("false_abstention_rate", "False abstentions"),
    ("injection_resistance", "Injection resistance"), ("citation_hit_rate", "Citation hit rate"),
    ("faithfulness_mean", "Faithfulness (lexical)"), ("degraded_answers", "LLM failures -> extractive"),
]


def main(paths: list[str]) -> None:
    runs = [json.loads(Path(p).read_text())["summary"] for p in paths]
    print("| Metric | " + " | ".join(f"`{r['label']}`" for r in runs) + " |")
    print("|---|" + "---|" * len(runs))
    for key, name in METRICS:
        print(f"| {name} | " + " | ".join(str(r.get(key, "-")) for r in runs) + " |")
    print("| Latency p50 / p95 (ms) | " + " | ".join(
        f"{r['latency_ms']['e2e_p50']} / {r['latency_ms']['e2e_p95']}" for r in runs) + " |")
    print("| Questions | " + " | ".join(str(r["n_questions"]) for r in runs) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
