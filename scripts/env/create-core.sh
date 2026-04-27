#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

create_python_env "$RAG_FLOW_CORE_ENV" "$RAG_FLOW_CORE_PYTHON"
pip_install -e "$RAG_FLOW_REPO_ROOT[dev]"
print_env_summary "$RAG_FLOW_CORE_ENV"
