#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${RAG_FLOW_ENV_FILE:-$REPO_ROOT/.local/rag-flow.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

: "${RAG_FLOW_REMOTE_SSH_HOST:?Set RAG_FLOW_REMOTE_SSH_HOST in .local/rag-flow.env}"
: "${RAG_FLOW_REMOTE_SSH_PORT:=22}"
: "${RAG_FLOW_REMOTE_SSH_USER:=root}"

exec ssh -p "$RAG_FLOW_REMOTE_SSH_PORT" "$RAG_FLOW_REMOTE_SSH_USER@$RAG_FLOW_REMOTE_SSH_HOST" "$@"
