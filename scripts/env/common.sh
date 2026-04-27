#!/usr/bin/env bash
set -euo pipefail

RAG_FLOW_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_FLOW_REPO_ROOT="$(cd "$RAG_FLOW_ENV_SCRIPT_DIR/../.." && pwd)"

resolve_env_file() {
  local preferred="${RAG_FLOW_ENV_FILE:-$RAG_FLOW_REPO_ROOT/.local/rag-flow.env}"
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

RAG_FLOW_ENV_FILE="$(resolve_env_file)"

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

if [[ -f "$RAG_FLOW_ENV_FILE" ]]; then
  load_env_file "$RAG_FLOW_ENV_FILE"
fi

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
}

: "${RAG_FLOW_ENV_MANAGER:=auto}"
: "${RAG_FLOW_CONDA_EXE:=micromamba}"
: "${RAG_FLOW_RUNTIME_ROOT:=$HOME/autodl-tmp}"
: "${RAG_FLOW_ENV_ROOT:=$RAG_FLOW_RUNTIME_ROOT/envs}"
: "${RAG_FLOW_CORE_ENV:=rag-flow-core}"
: "${RAG_FLOW_MINERU_ENV:=rag-flow-mineru}"
: "${RAG_FLOW_GPU_ENV:=rag-flow-gpu}"
: "${RAG_FLOW_LLM_ENV:=rag-flow-llm}"
: "${RAG_FLOW_CORE_PYTHON:=3.12}"
: "${RAG_FLOW_MINERU_PYTHON_VERSION:=3.12}"
: "${RAG_FLOW_GPU_PYTHON:=3.12}"
: "${RAG_FLOW_LLM_PYTHON:=3.12}"
: "${RAG_FLOW_PIP_INDEX_URL:=${RAG_FLOW_INIT_PIP_INDEX_URL:-}}"
: "${RAG_FLOW_UV_INDEX_URL:=${RAG_FLOW_INIT_UV_INDEX_URL:-$RAG_FLOW_PIP_INDEX_URL}}"
: "${RAG_FLOW_PIP_CACHE_DIR:=$RAG_FLOW_RUNTIME_ROOT/.cache/pip}"
: "${RAG_FLOW_UV_CACHE_DIR:=$RAG_FLOW_RUNTIME_ROOT/.cache/uv}"
: "${RAG_FLOW_CONDA_PKGS_DIRS:=$RAG_FLOW_RUNTIME_ROOT/conda-pkgs}"
: "${RAG_FLOW_UPDATE_ENV_FILE:=1}"

export PIP_CACHE_DIR="$RAG_FLOW_PIP_CACHE_DIR"
export UV_CACHE_DIR="$RAG_FLOW_UV_CACHE_DIR"
export CONDA_PKGS_DIRS="$RAG_FLOW_CONDA_PKGS_DIRS"

resolve_conda_exe() {
  if [[ -n "${RAG_FLOW_CONDA_EXE:-}" ]] && command -v "$RAG_FLOW_CONDA_EXE" >/dev/null 2>&1; then
    command -v "$RAG_FLOW_CONDA_EXE"
    return
  fi
  for candidate in micromamba mamba conda; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done
  return 1
}

resolve_manager() {
  case "$RAG_FLOW_ENV_MANAGER" in
    uv)
      require_command uv
      echo "uv"
      ;;
    conda|micromamba|mamba)
      resolve_conda_exe >/dev/null || {
        echo "No conda-compatible executable found. Install micromamba/conda or set RAG_FLOW_ENV_MANAGER=uv." >&2
        exit 1
      }
      echo "conda"
      ;;
    auto)
      if resolve_conda_exe >/dev/null; then
        echo "conda"
      else
        require_command uv
        echo "uv"
      fi
      ;;
    *)
      echo "Unknown RAG_FLOW_ENV_MANAGER=$RAG_FLOW_ENV_MANAGER. Use auto, uv, conda, micromamba, or mamba." >&2
      exit 1
      ;;
  esac
}

create_python_env() {
  local env_name="$1"
  local python_version="$2"
  local manager
  manager="$(resolve_manager)"
  RAG_FLOW_SELECTED_ENV_MANAGER="$manager"

  if [[ "$manager" == "uv" ]]; then
    require_command uv
    local env_dir="$RAG_FLOW_ENV_ROOT/$env_name"
    mkdir -p "$RAG_FLOW_ENV_ROOT"
    if [[ ! -x "$env_dir/bin/python" ]]; then
      uv venv --python "$python_version" "$env_dir"
    fi
    RAG_FLOW_ENV_PYTHON="$env_dir/bin/python"
    return
  fi

  local conda_exe
  local env_dir
  conda_exe="$(resolve_conda_exe)"
  env_dir="$RAG_FLOW_ENV_ROOT/$env_name"
  mkdir -p "$RAG_FLOW_ENV_ROOT" "$RAG_FLOW_CONDA_PKGS_DIRS"
  if [[ ! -x "$env_dir/bin/python" ]]; then
    "$conda_exe" create -y -p "$env_dir" "python=$python_version"
  fi
  RAG_FLOW_ENV_PYTHON="$env_dir/bin/python"
}

pip_install() {
  local package_env=()
  [[ -n "$RAG_FLOW_PIP_INDEX_URL" ]] && package_env+=("PIP_INDEX_URL=$RAG_FLOW_PIP_INDEX_URL")
  [[ -n "$RAG_FLOW_UV_INDEX_URL" ]] && package_env+=("UV_INDEX_URL=$RAG_FLOW_UV_INDEX_URL")
  [[ -n "$RAG_FLOW_PIP_CACHE_DIR" ]] && package_env+=("PIP_CACHE_DIR=$RAG_FLOW_PIP_CACHE_DIR")
  [[ -n "$RAG_FLOW_UV_CACHE_DIR" ]] && package_env+=("UV_CACHE_DIR=$RAG_FLOW_UV_CACHE_DIR")
  if command -v uv >/dev/null 2>&1; then
    env "${package_env[@]}" uv pip install --python "$RAG_FLOW_ENV_PYTHON" "$@"
  else
    env "${package_env[@]}" "$RAG_FLOW_ENV_PYTHON" -m pip install "$@"
  fi
}

pip_install_from_index() {
  local index_url="$1"
  shift
  local package_env=()
  [[ -n "$RAG_FLOW_PIP_CACHE_DIR" ]] && package_env+=("PIP_CACHE_DIR=$RAG_FLOW_PIP_CACHE_DIR")
  [[ -n "$RAG_FLOW_UV_CACHE_DIR" ]] && package_env+=("UV_CACHE_DIR=$RAG_FLOW_UV_CACHE_DIR")
  if command -v uv >/dev/null 2>&1; then
    env "${package_env[@]}" uv pip install --python "$RAG_FLOW_ENV_PYTHON" --index-url "$index_url" "$@"
  else
    env "${package_env[@]}" "$RAG_FLOW_ENV_PYTHON" -m pip install --index-url "$index_url" "$@"
  fi
}

set_env_var() {
  local key="$1"
  local value="$2"
  local file="${3:-$RAG_FLOW_ENV_FILE}"
  local tmp_file
  mkdir -p "$(dirname "$file")"
  touch "$file"
  tmp_file="$(mktemp)"
  awk -v key="$key" -v value="$value" '
    BEGIN { replaced = 0 }
    $0 ~ "^" key "=" {
      print key "=" value
      replaced = 1
      next
    }
    { print }
    END {
      if (!replaced) {
        print key "=" value
      }
    }
  ' "$file" > "$tmp_file"
  mv "$tmp_file" "$file"
}

print_env_summary() {
  local env_name="$1"
  echo "Environment: $env_name"
  echo "Manager: ${RAG_FLOW_SELECTED_ENV_MANAGER:-unknown}"
  echo "Python: $RAG_FLOW_ENV_PYTHON"
  echo "Runtime root: $RAG_FLOW_RUNTIME_ROOT"
  echo "Environment root: $RAG_FLOW_ENV_ROOT"
}
