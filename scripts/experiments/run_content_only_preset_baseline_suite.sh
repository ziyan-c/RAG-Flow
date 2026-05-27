#!/usr/bin/env bash
set -euo pipefail

QUERY_SET="${RAG_FLOW_CONTENT_ONLY_QUERY_SET:-qa-goldset/source-pdfs-qa-200.codex-reviewed.json}"
WORK_ROOT="${RAG_FLOW_CONTENT_ONLY_WORK_ROOT:-thesis-v2/experiments/v2-final}"
DIRECT_OUTPUT_DIR="${RAG_FLOW_CODEX_DIRECT_OUTPUT_DIR:-thesis-v2/experiments/codex-direct-pdf}"
DIRECT_RUN_ID="${RAG_FLOW_CODEX_DIRECT_RUN_ID:-codex-direct-pdf-qa200}"
SUMMARY_OUTPUT="${RAG_FLOW_CONTENT_ONLY_SUMMARY:-thesis-v2/experiments/content-only-preset-vs-codex-direct-pdf-summary.csv}"
PASSES="${RAG_FLOW_CONTENT_ONLY_PASSES:-3}"
BATCH_SIZE="${RAG_FLOW_CONTENT_ONLY_BATCH_SIZE:-3}"
TIMEOUT="${RAG_FLOW_CONTENT_ONLY_TIMEOUT:-900}"
JOBS="${RAG_FLOW_CONTENT_ONLY_JOBS:-1}"
DIRECT_JOBS="${RAG_FLOW_CODEX_DIRECT_JOBS:-1}"
DIRECT_TIMEOUT="${RAG_FLOW_CODEX_DIRECT_TIMEOUT:-1800}"
SCORE_TAG="${RAG_FLOW_CONTENT_ONLY_SCORE_TAG:-content_only_scored}"
RUN_CODEX_DIRECT="${RAG_FLOW_RUN_CODEX_DIRECT:-1}"
CODEX_MODEL="${RAG_FLOW_CODEX_DIRECT_MODEL:-}"
CODEX_PROFILE="${RAG_FLOW_CODEX_DIRECT_PROFILE:-}"
CODEX_PROFILE_V2="${RAG_FLOW_CODEX_DIRECT_PROFILE_V2:-}"

RAG_RUN_DIRS=(
  "${WORK_ROOT}/answering-runs/final200-compact-ratio0p4"
  "${WORK_ROOT}/answering-runs/final200-default-text"
  "${WORK_ROOT}/answering-runs/final200-high-context-16k"
  "${WORK_ROOT}/answering-runs/final200-default-images-fixed"
  "${WORK_ROOT}/answering-runs/final200-visual-naive-w1"
)

for run_dir in "${RAG_RUN_DIRS[@]}"; do
  if [[ ! -f "${run_dir}/answering_metrics.csv" ]]; then
    echo "Missing existing RAG answer run: ${run_dir}/answering_metrics.csv" >&2
    exit 2
  fi
done

DIRECT_RUN_DIR="${DIRECT_OUTPUT_DIR}/${DIRECT_RUN_ID}"
if [[ "${RUN_CODEX_DIRECT}" == "1" ]]; then
  args=(
    python3 scripts/experiments/codex_direct_pdf_baseline.py
    --query-set "${QUERY_SET}"
    --output-dir "${DIRECT_OUTPUT_DIR}"
    --run-id "${DIRECT_RUN_ID}"
    --resume
    --jobs "${DIRECT_JOBS}"
    --timeout "${DIRECT_TIMEOUT}"
  )
  if [[ -n "${CODEX_MODEL}" ]]; then
    args+=(--model "${CODEX_MODEL}")
  fi
  if [[ -n "${CODEX_PROFILE}" ]]; then
    args+=(--profile "${CODEX_PROFILE}")
  fi
  if [[ -n "${CODEX_PROFILE_V2}" ]]; then
    args+=(--profile-v2 "${CODEX_PROFILE_V2}")
  fi
  "${args[@]}"
elif [[ ! -f "${DIRECT_RUN_DIR}/answering_metrics.csv" ]]; then
  echo "Codex direct-PDF run is missing and RAG_FLOW_RUN_CODEX_DIRECT=0: ${DIRECT_RUN_DIR}" >&2
  exit 2
fi

score_args=(
  python3 scripts/experiments/codex_score_answer_runs.py
  --query-set "${QUERY_SET}"
  --scoring-mode content-only
  --score-tag "${SCORE_TAG}"
  --passes "${PASSES}"
  --batch-size "${BATCH_SIZE}"
  --timeout "${TIMEOUT}"
  --jobs "${JOBS}"
  --summary-output "${SUMMARY_OUTPUT}"
)

for run_dir in "${RAG_RUN_DIRS[@]}"; do
  score_args+=(--run-dir "${run_dir}")
done
score_args+=(--run-dir "${DIRECT_RUN_DIR}")

"${score_args[@]}"

python3 - "${SUMMARY_OUTPUT}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
labels = {
    "final200-compact-ratio0p4": "low",
    "final200-default-text": "medium",
    "final200-high-context-16k": "high",
    "final200-default-images-fixed": "medium-with-image-input",
    "final200-visual-naive-w1": "medium-with-visual-recall",
    "codex-direct-pdf-qa200": "Codex direct PDF",
}
if rows:
    fieldnames = ["experiment_row", *[name for name in rows[0] if name != "experiment_row"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row["experiment_row"] = labels.get(row.get("run_id", ""), row.get("run_id", ""))
            writer.writerow(row)
PY
