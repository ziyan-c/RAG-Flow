#!/usr/bin/env bash
set -euo pipefail

python -m rag_flow.retrieval_client "$@"
