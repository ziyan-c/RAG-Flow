#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${RAG_FLOW_LLM_INSTALL_SPEC:=sglang[all]}"
: "${RAG_FLOW_LLM_FIX_CUDNN:=1}"
: "${RAG_FLOW_LLM_CUDNN_PACKAGE:=nvidia-cudnn-cu12==9.16.0.29}"
: "${RAG_FLOW_LLM_EXTRA_PACKAGES:=}"
: "${RAG_FLOW_CREATE_LLM_INSTALL_UV:=$RAG_FLOW_USE_UV}"

if truthy "$RAG_FLOW_CREATE_LLM_INSTALL_UV"; then
  "$SCRIPT_DIR/install-uv.sh"
fi

create_python_env "$RAG_FLOW_LLM_ENV" "$RAG_FLOW_LLM_PYTHON"
pip_install "$RAG_FLOW_LLM_INSTALL_SPEC"
if truthy "$RAG_FLOW_LLM_FIX_CUDNN"; then
  pip_install "$RAG_FLOW_LLM_CUDNN_PACKAGE"
fi
if [[ -n "$RAG_FLOW_LLM_EXTRA_PACKAGES" ]]; then
  # shellcheck disable=SC2206
  extra_packages=($RAG_FLOW_LLM_EXTRA_PACKAGES)
  pip_install "${extra_packages[@]}"
fi

if truthy "$RAG_FLOW_UPDATE_ENV_FILE"; then
  set_env_var RAG_FLOW_LLM_PYTHON_BIN "$RAG_FLOW_ENV_PYTHON"
  set_env_var RAG_FLOW_SGLANG_PYTHON "$RAG_FLOW_ENV_PYTHON"
fi

print_env_summary "$RAG_FLOW_LLM_ENV"
