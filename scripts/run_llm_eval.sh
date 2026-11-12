#!/usr/bin/env bash
# Evaluate the LLM mode with a FREE model and compare it with the extractive baseline.
#
#   Ollama (local, free):   ollama pull qwen2.5:7b && PRESET=ollama scripts/run_llm_eval.sh
#   Gemini (free tier):     GEMINI_API_KEY=... PRESET=gemini scripts/run_llm_eval.sh
#   Any OpenAI-compatible:  LLM_BASE_URL=... OPENAI_API_KEY=... LLM_MODEL=... PRESET=custom scripts/run_llm_eval.sh
#
# Needs no database (uses the in-memory store). Results land in eval/results/, comparison in
# eval/results/llm-comparison.md. Override the model with LLM_MODEL=...
set -euo pipefail
cd "$(dirname "$0")/.."

case "${PRESET:-ollama}" in
  ollama)
    export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:11434/v1}"
    export LLM_MODEL="${LLM_MODEL:-qwen2.5:7b}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama}"   # Ollama ignores the key, the client requires one
    SLEEP="${SLEEP:-0}"
    curl -sf "${LLM_BASE_URL%/v1}/api/tags" >/dev/null || { echo "Ollama is not running at $LLM_BASE_URL (run: ollama serve)"; exit 1; }
    ;;
  gemini)
    : "${GEMINI_API_KEY:?set GEMINI_API_KEY (free key from https://aistudio.google.com/apikey)}"
    export LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
    export LLM_MODEL="${LLM_MODEL:-gemini-2.5-flash}"   # check AI Studio for the current free Flash model name
    export OPENAI_API_KEY="$GEMINI_API_KEY"
    SLEEP="${SLEEP:-7}"   # free tiers allow only a few requests per minute; each question makes 2-3 calls
    ;;
  custom)
    : "${LLM_BASE_URL:?}" "${LLM_MODEL:?}" "${OPENAI_API_KEY:?}"
    SLEEP="${SLEEP:-0}"
    ;;
  *) echo "PRESET must be ollama, gemini or custom"; exit 2 ;;
esac

export VECTOR_STORE=memory EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-hashing}"
tag="$(printf '%s' "$LLM_MODEL" | tr -c 'A-Za-z0-9._-' '-')"
[ -f data/policies/northwind_homeowners_ho3.pdf ] || python scripts/generate_sample_policies.py

for ds in dataset blind_v2; do
  echo "== extractive baseline: $ds"
  LLM_PROVIDER=extractive python eval/run_eval.py --store memory --dataset "$ds.jsonl" --label "$ds-extractive-memory" >/dev/null
  echo "== LLM ($LLM_MODEL): $ds"
  LLM_PROVIDER=openai python eval/run_eval.py --store memory --dataset "$ds.jsonl" --label "$ds-llm-$tag" --sleep "$SLEEP" \
    | grep -E '"(overall_accuracy|abstention_accuracy|degraded_answers)"' || true
done

{
  echo "# LLM mode vs extractive baseline ($LLM_MODEL)"; echo
  echo "Note: blind_v2 was first run in extractive mode during the audit. This is the first run in LLM mode;"
  echo "nothing was tuned on it, but it is no longer a pristine blind set."; echo
  for ds in dataset blind_v2; do
    echo "## $ds"; echo
    python eval/compare_results.py "eval/results/$ds-extractive-memory.json" "eval/results/$ds-llm-$tag.json"; echo
  done
} > eval/results/llm-comparison.md
cat eval/results/llm-comparison.md
