#!/usr/bin/env bash
set -euo pipefail

: "${RAG_FLOW_RUNTIME_ROOT:=/root/autodl-tmp}"
: "${RAG_FLOW_DATA_DISK:=$RAG_FLOW_RUNTIME_ROOT}"
: "${RAG_FLOW_SOFT_LINK_MERGE_EXISTING:=1}"
: "${RAG_FLOW_SOFT_LINK_BACKUP_DIR:=$RAG_FLOW_DATA_DISK/.rag-flow-backups}"

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

move_and_link() {
  local src=$1
  local target_parent=$2
  local dir_name
  local target_path
  local current_target
  local backup_path
  dir_name=$(basename "$src")
  target_path="$target_parent/$dir_name"

  mkdir -p "$target_parent"

  if [ -L "$src" ]; then
    current_target="$(readlink "$src")"
    mkdir -p "$target_path"
    if [ "$current_target" = "$target_path" ]; then
      echo "$src -> $target_path"
      return
    fi
    rm "$src"
  elif [ -d "$src" ]; then
    if [ -d "$target_path" ]; then
      if [[ -n "$(find "$src" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        if ! truthy "$RAG_FLOW_SOFT_LINK_MERGE_EXISTING"; then
          echo "Refuse to replace non-empty $src because $target_path already exists." >&2
          echo "Move or merge $src into $target_path, then rerun." >&2
          exit 1
        fi
        mkdir -p "$RAG_FLOW_SOFT_LINK_BACKUP_DIR" "$target_path"
        cp -an "$src"/. "$target_path"/
        backup_path="$RAG_FLOW_SOFT_LINK_BACKUP_DIR/$dir_name.$(date +%Y%m%d%H%M%S)"
        mv "$src" "$backup_path"
        echo "Merged missing files from $src into $target_path"
        echo "Backed up original $src to $backup_path"
      else
        rmdir "$src"
      fi
    else
      mv "$src" "$target_parent/"
    fi
  else
    rm -rf "$src"
    mkdir -p "$target_path"
  fi

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
