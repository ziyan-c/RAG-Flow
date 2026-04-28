#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

: "${RAG_FLOW_CREATE_MINERU_INSTALL_UV:=$RAG_FLOW_USE_UV}"

if truthy "$RAG_FLOW_CREATE_MINERU_INSTALL_UV"; then
  "$SCRIPT_DIR/install-uv.sh"
fi

create_python_env "$RAG_FLOW_MINERU_ENV" "$RAG_FLOW_MINERU_PYTHON_VERSION"
pip_install -e "$RAG_FLOW_REPO_ROOT[mineru]"

mineru_command="$(dirname "$RAG_FLOW_ENV_PYTHON")/mineru"
if [[ ! -x "$mineru_command" ]]; then
  mineru_command="mineru"
fi

if truthy "$RAG_FLOW_UPDATE_ENV_FILE"; then
  set_env_var RAG_FLOW_MINERU_PYTHON "$RAG_FLOW_ENV_PYTHON"
  set_env_var RAG_FLOW_MINERU_COMMAND "$mineru_command"
fi

print_env_summary "$RAG_FLOW_MINERU_ENV"
echo "MinerU command: $mineru_command"
