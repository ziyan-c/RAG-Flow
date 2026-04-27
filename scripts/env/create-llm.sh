#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${RAG_FLOW_LLM_INSTALL_SPEC:=sglang[all]}"

create_python_env "$RAG_FLOW_LLM_ENV" "$RAG_FLOW_LLM_PYTHON"
pip_install "$RAG_FLOW_LLM_INSTALL_SPEC"
print_env_summary "$RAG_FLOW_LLM_ENV"
