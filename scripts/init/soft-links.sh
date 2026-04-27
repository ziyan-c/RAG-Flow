#!/usr/bin/env bash
set -euo pipefail

: "${RAG_FLOW_RUNTIME_ROOT:=/root/autodl-tmp}"
: "${RAG_FLOW_DATA_DISK:=$RAG_FLOW_RUNTIME_ROOT}"

move_and_link() {
  local src=$1
  local target_parent=$2
  local dir_name
  local target_path
  dir_name=$(basename "$src")
  target_path="$target_parent/$dir_name"

  mkdir -p "$target_parent"
  if [ ! -d "$target_path" ]; then
    if [ -d "$src" ] && [ ! -L "$src" ]; then
      mv "$src" "$target_parent/"
    else
      mkdir -p "$target_path"
    fi
  fi

  rm -rf "$src"
  ln -s "$target_path" "$src"
  echo "$src -> $target_path"
}

move_and_link ~/.cache "$RAG_FLOW_DATA_DISK"
move_and_link ~/.local "$RAG_FLOW_DATA_DISK"

CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
if [ -d "$CONDA_BASE/pkgs" ] || [ -L "$CONDA_BASE/pkgs" ]; then
  move_and_link "$CONDA_BASE/pkgs" "$RAG_FLOW_DATA_DISK/miniconda3"
fi
if [ -d "$CONDA_BASE/envs" ] || [ -L "$CONDA_BASE/envs" ]; then
  move_and_link "$CONDA_BASE/envs" "$RAG_FLOW_DATA_DISK/miniconda3"
fi
