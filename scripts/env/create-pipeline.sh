#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${RAG_FLOW_PIPELINE_INSTALL_TORCH:=1}"
: "${RAG_FLOW_PIPELINE_TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu128}"
: "${RAG_FLOW_PIPELINE_TORCH_PACKAGES:=torch torchvision torchaudio}"

create_python_env "$RAG_FLOW_PIPELINE_ENV" "$RAG_FLOW_PIPELINE_PYTHON"

if truthy "$RAG_FLOW_PIPELINE_INSTALL_TORCH"; then
  read -r -a torch_packages <<< "$RAG_FLOW_PIPELINE_TORCH_PACKAGES"
  pip_install_from_index "$RAG_FLOW_PIPELINE_TORCH_INDEX_URL" "${torch_packages[@]}"
fi

pip_install -e "$RAG_FLOW_REPO_ROOT[retrieval,preprocess]"
if truthy "$RAG_FLOW_UPDATE_ENV_FILE"; then
  set_env_var RAG_FLOW_PIPELINE_PYTHON_BIN "$RAG_FLOW_ENV_PYTHON"
fi
print_env_summary "$RAG_FLOW_PIPELINE_ENV"
