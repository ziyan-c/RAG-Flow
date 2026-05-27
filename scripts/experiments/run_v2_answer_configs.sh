#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${RAG_FLOW_EXPERIMENT_WORK_ROOT:-thesis-v2/experiments/v2-final}"
PYTHON_BIN="${RAG_FLOW_EXPERIMENT_PYTHON:-/root/autodl-tmp/envs/rag-flow-pipeline/bin/python}"
QUERY_SET="${RAG_FLOW_EXPERIMENT_QUERY_SET:-${WORK_ROOT}/qa/qa50.jsonl}"
COLLECTION="${RAG_FLOW_EXPERIMENT_COLLECTION:-rag-flow-v2}"
DB_PATH="${RAG_FLOW_EXPERIMENT_DB_PATH:-}"
RUN_PREFIX="${RAG_FLOW_EXPERIMENT_RUN_PREFIX:-answer}"
LIMIT="${RAG_FLOW_EXPERIMENT_LIMIT:-50}"

if [[ -z "${DB_PATH}" ]]; then
  echo "Set RAG_FLOW_EXPERIMENT_DB_PATH to an existing Qdrant local db path." >&2
  exit 2
fi

if [[ -z "${RAG_FLOW_EXPERIMENT_CONFIGS:-}" ]]; then
  cat >&2 <<'EOF'
Set RAG_FLOW_EXPERIMENT_CONFIGS as whitespace-separated configs:
context_cap,retrieval_k,final_top_k,rrf_k,min_score_ratio,route_mode,visual_bonus,visual_weight,max_tokens,images,thinking,label

Example:
10000,150,80,10,1.0,text,none,2.5,4000,0,0,text10k
EOF
  exit 2
fi

mkdir -p "${WORK_ROOT}/logs"

for config in ${RAG_FLOW_EXPERIMENT_CONFIGS}; do
  IFS=',' read -r context_cap retrieval_k final_top_k rrf_k min_score_ratio route_mode visual_bonus visual_weight max_tokens images thinking label <<<"${config}"
  run_id="${RUN_PREFIX}-${label}"
  log_path="${WORK_ROOT}/logs/${run_id}.log"
  echo "[run] ${run_id}" | tee "${log_path}"
  args=(
    scripts/experiments/v2_experiment_runner.py answer-config
    --work-root "${WORK_ROOT}"
    --db-path "${DB_PATH}"
    --collection "${COLLECTION}"
    --query-set "${QUERY_SET}"
    --run-id "${run_id}"
    --context-cap "${context_cap}"
    --retrieval-k "${retrieval_k}"
    --final-top-k "${final_top_k}"
    --rrf-k "${rrf_k}"
    --min-score-ratio "${min_score_ratio}"
    --route-mode "${route_mode}"
    --visual-bonus "${visual_bonus}"
    --visual-weight "${visual_weight}"
    --max-tokens "${max_tokens}"
    --limit "${LIMIT}"
  )
  if [[ "${images}" == "1" ]]; then
    args+=(--final-output-images)
  fi
  if [[ "${thinking}" == "1" ]]; then
    args+=(--enable-thinking)
  fi
  started="$(date +%s)"
  "$PYTHON_BIN" "${args[@]}" 2>&1 | tee -a "${log_path}"
  elapsed="$(( $(date +%s) - started ))"
  echo "[time] ${run_id} elapsed_seconds=${elapsed}" | tee -a "${log_path}"
done
