#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${RAG_FLOW_EXPERIMENT_WORK_ROOT:-thesis/experiments/v2-final}"
DB_ROOT="${RAG_FLOW_EXPERIMENT_DB_ROOT:-/root/autodl-tmp/rag-flow-v2-experiments/qdrant}"
PYTHON_BIN="${RAG_FLOW_EXPERIMENT_PYTHON:-/root/autodl-tmp/envs/rag-flow-pipeline/bin/python}"
RUN_PREFIX="${RAG_FLOW_EXPERIMENT_RUN_PREFIX:-chunk-profile}"
TEXT_BATCH_SIZE="${RAG_FLOW_EXPERIMENT_TEXT_BATCH_SIZE:-8}"
CONTEXT_CAP="${RAG_FLOW_EXPERIMENT_CONTEXT_CAP:-16000}"
RETRIEVAL_K="${RAG_FLOW_EXPERIMENT_RETRIEVAL_K:-150}"
FINAL_TOP_K="${RAG_FLOW_EXPERIMENT_FINAL_TOP_K:-80}"
RRF_K="${RAG_FLOW_EXPERIMENT_RRF_K:-10}"
MIN_SCORE_RATIO="${RAG_FLOW_EXPERIMENT_MIN_SCORE_RATIO:-1.0}"
MAX_TOKENS="${RAG_FLOW_EXPERIMENT_MAX_TOKENS:-4000}"
LIMIT="${RAG_FLOW_EXPERIMENT_LIMIT:-50}"

if [[ -z "${RAG_FLOW_EXPERIMENT_PROFILES:-}" ]]; then
  echo "Set RAG_FLOW_EXPERIMENT_PROFILES, for example: auto,1500,150,150 token,1500,150,150" >&2
  exit 2
fi

mkdir -p "${WORK_ROOT}/logs"

for profile in ${RAG_FLOW_EXPERIMENT_PROFILES}; do
  IFS=',' read -r mode max_tokens overlap_tokens min_tokens <<<"${profile}"
  run_id="${RUN_PREFIX}-${mode}-m${max_tokens}-o${overlap_tokens}-n${min_tokens}"
  log_path="${WORK_ROOT}/logs/${run_id}.log"
  echo "[run] ${run_id}" | tee "${log_path}"
  started="$(date +%s)"
  "$PYTHON_BIN" scripts/experiments/v2_experiment_runner.py run-config \
    --work-root "${WORK_ROOT}" \
    --source-root source-pdfs \
    --output-root output-pdfs \
    --db-root "${DB_ROOT}" \
    --collection rag-flow-v2 \
    --run-id "${run_id}" \
    --chunk-mode "${mode}" \
    --chunk-max-tokens "${max_tokens}" \
    --chunk-overlap-tokens "${overlap_tokens}" \
    --chunk-min-tokens "${min_tokens}" \
    --text-batch-size "${TEXT_BATCH_SIZE}" \
    --context-cap "${CONTEXT_CAP}" \
    --retrieval-k "${RETRIEVAL_K}" \
    --final-top-k "${FINAL_TOP_K}" \
    --rrf-k "${RRF_K}" \
    --min-score-ratio "${MIN_SCORE_RATIO}" \
    --max-tokens "${MAX_TOKENS}" \
    --limit "${LIMIT}" 2>&1 | tee -a "${log_path}"
  elapsed="$(( $(date +%s) - started ))"
  echo "[time] ${run_id} elapsed_seconds=${elapsed}" | tee -a "${log_path}"
done
