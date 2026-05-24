#!/usr/bin/env bash
set -euo pipefail

cd "${RAG_FLOW_REPO_ROOT:-/root/RAG-Flow}"
set -a
source "${RAG_FLOW_ENV_FILE:-.local/rag-flow.env}"
set +a

WORK_ROOT="${RAG_FLOW_EXPERIMENT_WORK_ROOT:-thesis/experiments/v2-final}"
DB_ROOT="${RAG_FLOW_EXPERIMENT_DB_ROOT:-/root/autodl-tmp/rag-flow-v2-experiments/qdrant}"
PYTHON_BIN="${RAG_FLOW_PIPELINE_PYTHON_BIN:-/root/autodl-tmp/envs/rag-flow-pipeline/bin/python}"
mkdir -p "$WORK_ROOT/logs"

profiles=(
  "800 100 100"
  "1200 100 100"
  "1500 150 150"
  "2000 200 200"
  "2500 250 200"
  "3000 300 200"
  "4000 400 200"
  "5000 500 200"
)

for profile in "${profiles[@]}"; do
  read -r max_tokens overlap_tokens min_tokens <<<"$profile"
  run_id="chunk-coarse-m${max_tokens}-o${overlap_tokens}-n${min_tokens}"
  summary_path="$WORK_ROOT/runs/${run_id}.json"
  log_path="$WORK_ROOT/logs/${run_id}.log"
  if [[ -s "$summary_path" ]]; then
    echo "[skip] $run_id already has $summary_path"
    continue
  fi
  echo "[run] $run_id"
  started_at="$(date +%s)"
  "$PYTHON_BIN" scripts/experiments/v2_experiment_runner.py run-config \
    --work-root "$WORK_ROOT" \
    --source-root source-pdfs \
    --output-root output-pdfs \
    --db-root "$DB_ROOT" \
    --collection rag-flow-v2 \
    --run-id "$run_id" \
    --chunk-mode auto \
    --chunk-max-tokens "$max_tokens" \
    --chunk-overlap-tokens "$overlap_tokens" \
    --chunk-min-tokens "$min_tokens" \
    --text-batch-size 8 \
    --context-cap 16000 \
    --retrieval-k 150 \
    --final-top-k 80 \
    --rrf-k 10 \
    --min-score-ratio 1.0 \
    --max-tokens 4000 \
    --limit 50 \
    2>&1 | tee "$log_path"
  finished_at="$(date +%s)"
  echo "[time] $run_id elapsed_seconds=$((finished_at - started_at))" | tee -a "$log_path"
done

echo "[done] chunking coarse experiments"
