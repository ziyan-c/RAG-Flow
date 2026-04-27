#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/create-core.sh"
"$SCRIPT_DIR/create-mineru.sh"
"$SCRIPT_DIR/create-gpu.sh"
"$SCRIPT_DIR/create-llm.sh"
