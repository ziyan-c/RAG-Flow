#!/usr/bin/env bash
set -euo pipefail

: "${RAG_FLOW_COMPILE_JOBS:=20}"
: "${RAG_FLOW_RUNTIME_THREADS:=8}"
: "${RAG_FLOW_INIT_BASHRC:=$HOME/.bashrc}"

write_block() {
  cat <<EOF

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
}

tmp_file="$(mktemp)"
mkdir -p "$(dirname "$RAG_FLOW_INIT_BASHRC")"
touch "$RAG_FLOW_INIT_BASHRC"
awk '
  /# --- RAG Flow CPU threading ---/ { skip = 1; next }
  /# --------------------------------/ && skip { skip = 0; next }
  !skip { print }
' "$RAG_FLOW_INIT_BASHRC" > "$tmp_file"
write_block >> "$tmp_file"
mv "$tmp_file" "$RAG_FLOW_INIT_BASHRC"
echo "Wrote CPU threading settings to $RAG_FLOW_INIT_BASHRC"
