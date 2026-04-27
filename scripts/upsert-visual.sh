#!/usr/bin/env bash
set -euo pipefail

python -m rag_flow.indexing visual "$@"
