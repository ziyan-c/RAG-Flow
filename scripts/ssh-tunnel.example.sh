#!/usr/bin/env bash
set -euo pipefail

: "${RAG_FLOW_REMOTE_HOST:?Set RAG_FLOW_REMOTE_HOST, for example connect.example.com}"
: "${RAG_FLOW_REMOTE_PORT:?Set RAG_FLOW_REMOTE_PORT}"
: "${RAG_FLOW_REMOTE_USER:=root}"
: "${RAG_FLOW_LOCAL_BIND:=127.0.0.1}"

ssh \
  -o StrictHostKeyChecking=accept-new \
  -L "${RAG_FLOW_LOCAL_BIND}:8000:localhost:8000" \
  -L "${RAG_FLOW_LOCAL_BIND}:8080:localhost:8080" \
  "${RAG_FLOW_REMOTE_USER}@${RAG_FLOW_REMOTE_HOST}" \
  -p "$RAG_FLOW_REMOTE_PORT"
