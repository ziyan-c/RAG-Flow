#!/usr/bin/env bash
set -euo pipefail

: "${RAG_FLOW_COMPILE_JOBS:=20}"
: "${RAG_FLOW_RUNTIME_THREADS:=8}"
: "${RAG_FLOW_INIT_BASHRC:=$HOME/.bashrc}"

if grep -q "RAG Flow CPU threading" "$RAG_FLOW_INIT_BASHRC" 2>/dev/null; then
  echo "CPU threading settings already exist in $RAG_FLOW_INIT_BASHRC"
  exit 0
fi

cat >> "$RAG_FLOW_INIT_BASHRC" <<EOF

# --- RAG Flow CPU threading ---
export MAX_JOBS=$RAG_FLOW_COMPILE_JOBS
export MAKEFLAGS="-j$RAG_FLOW_COMPILE_JOBS"
export NINJA_JOBS=$RAG_FLOW_COMPILE_JOBS
export OMP_NUM_THREADS=$RAG_FLOW_RUNTIME_THREADS
export MKL_NUM_THREADS=$RAG_FLOW_RUNTIME_THREADS
export OPENBLAS_NUM_THREADS=$RAG_FLOW_RUNTIME_THREADS
export VECLIB_MAXIMUM_THREADS=$RAG_FLOW_RUNTIME_THREADS
export NUMEXPR_NUM_THREADS=$RAG_FLOW_RUNTIME_THREADS
# --------------------------------
EOF
