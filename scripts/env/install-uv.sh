#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

: "${RAG_FLOW_FORCE_INSTALL_UV:=0}"

python_for_pip() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  return 1
}

if ! truthy "$RAG_FLOW_FORCE_INSTALL_UV" && command -v uv >/dev/null 2>&1; then
  echo "uv already installed: $(command -v uv)"
  exit 0
fi

python_bin="$(python_for_pip)" || {
  echo "Cannot install uv: python3/python not found." >&2
  exit 1
}

echo "Install uv with $python_bin"
package_env=()
[[ -n "$RAG_FLOW_PIP_INDEX_URL" ]] && package_env+=("PIP_INDEX_URL=$RAG_FLOW_PIP_INDEX_URL")
[[ -n "$RAG_FLOW_PIP_CACHE_DIR" ]] && package_env+=("PIP_CACHE_DIR=$RAG_FLOW_PIP_CACHE_DIR")
env "${package_env[@]}" "$python_bin" -m pip install -U uv
