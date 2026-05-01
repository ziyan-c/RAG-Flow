#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/create-mineru.sh"
"$SCRIPT_DIR/create-pipeline.sh"
"$SCRIPT_DIR/create-llm.sh"
