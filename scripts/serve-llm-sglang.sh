#!/usr/bin/env bash
set -euo pipefail

: "${RAG_FLOW_SGLANG_MODEL_PATH:=/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}"
: "${RAG_FLOW_SGLANG_PORT:=8080}"
: "${RAG_FLOW_SGLANG_CONTEXT_LENGTH:=100000}"
: "${RAG_FLOW_SGLANG_MEM_FRACTION:=0.85}"

python -m sglang.launch_server \
  --model-path "$RAG_FLOW_SGLANG_MODEL_PATH" \
  --port "$RAG_FLOW_SGLANG_PORT" \
  --tp-size 1 \
  --mem-fraction-static "$RAG_FLOW_SGLANG_MEM_FRACTION" \
  --context-length "$RAG_FLOW_SGLANG_CONTEXT_LENGTH" \
  --reasoning-parser qwen3 \
  --quantization moe_wna16 \
  --attention-backend triton \
  --kv-cache-dtype fp8_e5m2
