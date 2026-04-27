#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${RAG_FLOW_GPU_INSTALL_TORCH:=1}"
: "${RAG_FLOW_GPU_TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu128}"
: "${RAG_FLOW_GPU_TORCH_PACKAGES:=torch torchvision torchaudio}"

create_python_env "$RAG_FLOW_GPU_ENV" "$RAG_FLOW_GPU_PYTHON"

if truthy "$RAG_FLOW_GPU_INSTALL_TORCH"; then
  read -r -a torch_packages <<< "$RAG_FLOW_GPU_TORCH_PACKAGES"
  pip_install_from_index "$RAG_FLOW_GPU_TORCH_INDEX_URL" "${torch_packages[@]}"
fi

pip_install -e "$RAG_FLOW_REPO_ROOT[retrieval,preprocess]"
print_env_summary "$RAG_FLOW_GPU_ENV"
