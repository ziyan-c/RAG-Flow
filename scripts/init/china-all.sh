#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

resolve_env_file() {
  local preferred="${RAG_FLOW_ENV_FILE:-$REPO_ROOT/.local/rag-flow.env}"
  local parent
  local fallback_dir
  local backup_path
  parent="$(dirname "$preferred")"
  if [[ -d "$parent" || ( ! -e "$parent" && ! -L "$parent" ) ]]; then
    echo "$preferred"
    return
  fi
  fallback_dir="${RAG_FLOW_RUNTIME_ROOT:-$HOME/autodl-tmp}/.local"
  mkdir -p "$fallback_dir"
  if [[ -L "$parent" ]]; then
    rm "$parent"
    ln -s "$fallback_dir" "$parent"
    echo "Repaired env directory symlink: $parent -> $fallback_dir" >&2
  else
    backup_path="$parent.bak.$(date +%Y%m%d%H%M%S)"
    mv "$parent" "$backup_path"
    ln -s "$fallback_dir" "$parent"
    echo "Moved unusable env directory path to: $backup_path" >&2
    echo "Created env directory symlink: $parent -> $fallback_dir" >&2
  fi
  echo "$preferred"
}

ENV_FILE="$(resolve_env_file)"

load_env_file() {
  local file="$1"
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done < "$file"
}

if [[ -f "$ENV_FILE" ]]; then
  load_env_file "$ENV_FILE"
fi

echo "Initialize China machine layout and mirrors."
echo "Using env file: $ENV_FILE"
RAG_FLOW_ENV_FILE="$ENV_FILE" "$SCRIPT_DIR/soft-links.sh"
RAG_FLOW_ENV_FILE="$ENV_FILE" "$SCRIPT_DIR/cpu-cores.sh"
RAG_FLOW_ENV_FILE="$ENV_FILE" "$SCRIPT_DIR/china-source.sh"
