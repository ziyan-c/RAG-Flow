#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${RAG_FLOW_ENV_FILE:-$REPO_ROOT/.local/rag-flow.env}"

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

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

: "${RAG_FLOW_INIT_REWRITE_APT:=1}"
: "${RAG_FLOW_INIT_APT_MIRROR:=mirrors.aliyun.com}"
: "${RAG_FLOW_INIT_APT_UPDATE:=1}"
: "${RAG_FLOW_INIT_CONFIGURE_LOCALE:=1}"
: "${RAG_FLOW_INIT_LOCALE:=en_US.UTF-8}"
: "${RAG_FLOW_INIT_WRITE_BASHRC:=1}"
: "${RAG_FLOW_INIT_BASHRC:=$HOME/.bashrc}"
: "${RAG_FLOW_RUNTIME_ROOT:=$HOME/autodl-tmp}"
: "${RAG_FLOW_INIT_HF_ENDPOINT:=https://hf-mirror.com}"
: "${RAG_FLOW_INIT_HF_HOME:=$RAG_FLOW_RUNTIME_ROOT/.cache/huggingface}"
: "${RAG_FLOW_INIT_VLLM_USE_MODELSCOPE:=True}"
: "${RAG_FLOW_INIT_MINERU_MODEL_SOURCE:=${RAG_FLOW_MINERU_MODEL_SOURCE:-modelscope}}"
: "${RAG_FLOW_INIT_PIP_INDEX_URL:=https://mirrors.aliyun.com/pypi/simple/}"
: "${RAG_FLOW_INIT_UV_INDEX_URL:=$RAG_FLOW_INIT_PIP_INDEX_URL}"
: "${RAG_FLOW_INIT_PIP_CACHE_DIR:=$RAG_FLOW_RUNTIME_ROOT/.cache/pip}"
: "${RAG_FLOW_INIT_UV_CACHE_DIR:=$RAG_FLOW_RUNTIME_ROOT/.cache/uv}"
: "${RAG_FLOW_INIT_TORCH_HOME:=$RAG_FLOW_RUNTIME_ROOT/.cache/torch}"
: "${RAG_FLOW_INIT_MODELSCOPE_CACHE:=$RAG_FLOW_RUNTIME_ROOT/.cache/modelscope}"
: "${RAG_FLOW_INIT_WRITE_CONDARC:=1}"
: "${RAG_FLOW_INIT_CONDARC:=$HOME/.condarc}"
: "${RAG_FLOW_INIT_CONDA_PKGS_DIRS:=$RAG_FLOW_RUNTIME_ROOT/conda-pkgs}"
: "${RAG_FLOW_INIT_CONDA_MAIN_CHANNEL:=https://mirrors.aliyun.com/anaconda/pkgs/main}"
: "${RAG_FLOW_INIT_CONDA_R_CHANNEL:=https://mirrors.aliyun.com/anaconda/pkgs/r}"
: "${RAG_FLOW_INIT_CONDA_MSYS2_CHANNEL:=https://mirrors.aliyun.com/anaconda/pkgs/msys2}"
: "${RAG_FLOW_INIT_CONDA_FORGE_CHANNEL:=https://mirrors.aliyun.com/anaconda/cloud}"
: "${RAG_FLOW_INIT_CONDA_PYTORCH_CHANNEL:=https://mirrors.aliyun.com/anaconda/cloud}"
: "${RAG_FLOW_INIT_CONDA_CLEAN_INDEX:=1}"

echo "Configure local mirrors and runtime environment."
echo "Using env file: $ENV_FILE"
mkdir -p \
  "$RAG_FLOW_RUNTIME_ROOT" \
  "$RAG_FLOW_INIT_HF_HOME" \
  "$RAG_FLOW_INIT_PIP_CACHE_DIR" \
  "$RAG_FLOW_INIT_UV_CACHE_DIR" \
  "$RAG_FLOW_INIT_TORCH_HOME" \
  "$RAG_FLOW_INIT_MODELSCOPE_CACHE" \
  "$RAG_FLOW_INIT_CONDA_PKGS_DIRS"

if truthy "$RAG_FLOW_INIT_REWRITE_APT"; then
  if [[ -f /etc/apt/sources.list ]]; then
    cp /etc/apt/sources.list "/etc/apt/sources.list.bak.$(date +%Y%m%d%H%M%S)"
    sed -i -E "s#archive.ubuntu.com|security.ubuntu.com|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|repo.huaweicloud.com#$RAG_FLOW_INIT_APT_MIRROR#g" /etc/apt/sources.list
    if truthy "$RAG_FLOW_INIT_APT_UPDATE"; then
      apt-get update -y
    fi
  else
    echo "Skip apt mirror rewrite: /etc/apt/sources.list not found."
  fi
fi

if truthy "$RAG_FLOW_INIT_CONFIGURE_LOCALE"; then
  if command -v locale-gen >/dev/null 2>&1; then
    locale-gen "$RAG_FLOW_INIT_LOCALE"
  fi
  if command -v update-locale >/dev/null 2>&1; then
    update-locale LANG="$RAG_FLOW_INIT_LOCALE" LC_ALL="$RAG_FLOW_INIT_LOCALE"
  fi
fi

if truthy "$RAG_FLOW_INIT_WRITE_BASHRC"; then
  if ! grep -q "RAG Flow AutoDL Environment" "$RAG_FLOW_INIT_BASHRC" 2>/dev/null; then
    cat >> "$RAG_FLOW_INIT_BASHRC" <<EOF

# --- RAG Flow AutoDL Environment ---
export LC_ALL=$RAG_FLOW_INIT_LOCALE
export LANG=$RAG_FLOW_INIT_LOCALE
export LANGUAGE=en_US:en
export HF_ENDPOINT=$RAG_FLOW_INIT_HF_ENDPOINT
export HF_HOME=$RAG_FLOW_INIT_HF_HOME
export MODELSCOPE_CACHE=$RAG_FLOW_INIT_MODELSCOPE_CACHE
export TORCH_HOME=$RAG_FLOW_INIT_TORCH_HOME
export VLLM_USE_MODELSCOPE=$RAG_FLOW_INIT_VLLM_USE_MODELSCOPE
export MINERU_MODEL_SOURCE=$RAG_FLOW_INIT_MINERU_MODEL_SOURCE
export PIP_INDEX_URL=$RAG_FLOW_INIT_PIP_INDEX_URL
export UV_INDEX_URL=$RAG_FLOW_INIT_UV_INDEX_URL
export PIP_CACHE_DIR=$RAG_FLOW_INIT_PIP_CACHE_DIR
export UV_CACHE_DIR=$RAG_FLOW_INIT_UV_CACHE_DIR
export CONDA_PKGS_DIRS=$RAG_FLOW_INIT_CONDA_PKGS_DIRS
# -----------------------------------
EOF
  fi
fi

if truthy "$RAG_FLOW_INIT_WRITE_CONDARC"; then
  cat > "$RAG_FLOW_INIT_CONDARC" <<EOF
channels:
  - defaults
show_channel_urls: true
pkgs_dirs:
  - $RAG_FLOW_INIT_CONDA_PKGS_DIRS
default_channels:
  - $RAG_FLOW_INIT_CONDA_MAIN_CHANNEL
  - $RAG_FLOW_INIT_CONDA_R_CHANNEL
  - $RAG_FLOW_INIT_CONDA_MSYS2_CHANNEL
custom_channels:
  conda-forge: $RAG_FLOW_INIT_CONDA_FORGE_CHANNEL
  pytorch: $RAG_FLOW_INIT_CONDA_PYTORCH_CHANNEL
EOF
fi

if truthy "$RAG_FLOW_INIT_CONDA_CLEAN_INDEX" && command -v conda >/dev/null 2>&1; then
  conda clean -i -y >/dev/null 2>&1
fi

if truthy "$RAG_FLOW_INIT_WRITE_BASHRC"; then
  echo "Done. Run: source $RAG_FLOW_INIT_BASHRC"
else
  echo "Done."
fi
