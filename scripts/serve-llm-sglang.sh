#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env/common.sh
source "$SCRIPT_DIR/env/common.sh"

dry_run=0
extra_args=()
profile_overridden=0
model_path_overridden=0
served_model_name_overridden=0
require_option_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --profile)
      require_option_value "$@"
      RAG_FLOW_SGLANG_MODEL_PROFILE="$2"
      profile_overridden=1
      shift 2
      ;;
    --model-path)
      require_option_value "$@"
      RAG_FLOW_SGLANG_MODEL_PATH="$2"
      model_path_overridden=1
      shift 2
      ;;
    --served-model-name)
      require_option_value "$@"
      RAG_FLOW_SGLANG_SERVED_MODEL_NAME="$2"
      served_model_name_overridden=1
      shift 2
      ;;
    --port)
      require_option_value "$@"
      RAG_FLOW_SGLANG_PORT="$2"
      shift 2
      ;;
    --context-length)
      require_option_value "$@"
      RAG_FLOW_SGLANG_CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --mem-fraction-static|--mem-fraction)
      require_option_value "$@"
      RAG_FLOW_SGLANG_MEM_FRACTION_STATIC="$2"
      shift 2
      ;;
    --tp-size)
      require_option_value "$@"
      RAG_FLOW_SGLANG_TP_SIZE="$2"
      shift 2
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

: "${RAG_FLOW_SGLANG_MODEL_PROFILE:=qwen3.6-35b-a3b-gptq-int4}"
: "${RAG_FLOW_SGLANG_PORT:=8080}"
: "${RAG_FLOW_SGLANG_TP_SIZE:=1}"
: "${RAG_FLOW_SGLANG_CONTEXT_LENGTH:=100000}"
: "${RAG_FLOW_SGLANG_MEM_FRACTION_STATIC:=${RAG_FLOW_SGLANG_MEM_FRACTION:-0.85}}"
: "${RAG_FLOW_SGLANG_REASONING_PARSER:=qwen3}"
: "${RAG_FLOW_SGLANG_QUANTIZATION:=moe_wna16}"
: "${RAG_FLOW_SGLANG_ATTENTION_BACKEND:=triton}"
: "${RAG_FLOW_SGLANG_KV_CACHE_DTYPE:=fp8_e5m2}"
: "${RAG_FLOW_SGLANG_PYTHON:=${RAG_FLOW_LLM_PYTHON_BIN:-python}}"
: "${RAG_FLOW_SGLANG_EXTRA_ARGS:=}"
: "${RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT:=$RAG_FLOW_RUNTIME_ROOT/models}"

if [[ "$profile_overridden" == "1" && "$model_path_overridden" != "1" ]]; then
  unset RAG_FLOW_SGLANG_MODEL_PATH
fi
if [[ "$profile_overridden" == "1" && "$served_model_name_overridden" != "1" ]]; then
  unset RAG_FLOW_SGLANG_SERVED_MODEL_NAME
fi

profile_key="$(printf '%s' "$RAG_FLOW_SGLANG_MODEL_PROFILE" | tr '[:upper:]' '[:lower:]')"
default_model_id=""
default_model_path=""
default_modelscope_path=""
default_hf_path=""
default_served_model_name=""
case "$profile_key" in
  qwen3.6-35b-a3b-gptq-int4|qwen3.6|palmfuture/qwen3.6-35b-a3b-gptq-int4)
    default_model_id="palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
    default_modelscope_path="/root/.cache/modelscope/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
    default_hf_path="/root/.cache/huggingface/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
    default_served_model_name="palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
    ;;
  qwen3.5-35b-a3b-gptq-int4|qwen3.5|qwen/qwen3.5-35b-a3b-gptq-int4)
    default_model_id="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    default_modelscope_path="/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    default_hf_path="/root/.cache/huggingface/hub/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    default_served_model_name="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    ;;
  custom)
    default_model_path=""
    ;;
  *)
    if [[ -z "${RAG_FLOW_SGLANG_MODEL_PATH:-}" ]]; then
      echo "Unknown RAG_FLOW_SGLANG_MODEL_PROFILE=$RAG_FLOW_SGLANG_MODEL_PROFILE" >&2
      echo "Use qwen3.6-35b-a3b-gptq-int4, qwen3.5-35b-a3b-gptq-int4, custom, or set RAG_FLOW_SGLANG_MODEL_PATH." >&2
      exit 1
    fi
    ;;
esac

if [[ -z "$default_model_path" && -n "$default_modelscope_path" ]]; then
  default_model_path="$default_modelscope_path"
fi
if [[ -z "${RAG_FLOW_SGLANG_MODEL_ID:-}" && -n "$default_model_id" ]]; then
  RAG_FLOW_SGLANG_MODEL_ID="$default_model_id"
fi
if [[ -z "${RAG_FLOW_SGLANG_MODEL_ID:-}" && -n "${RAG_FLOW_LLM_MODEL:-}" ]]; then
  RAG_FLOW_SGLANG_MODEL_ID="$RAG_FLOW_LLM_MODEL"
fi
if [[ -z "$default_modelscope_path" && -n "${RAG_FLOW_SGLANG_MODEL_ID:-}" ]]; then
  default_modelscope_path="/root/.cache/modelscope/hub/models/$RAG_FLOW_SGLANG_MODEL_ID"
fi
if [[ -z "$default_hf_path" && -n "${RAG_FLOW_SGLANG_MODEL_ID:-}" ]]; then
  default_hf_path="/root/.cache/huggingface/hub/models/$RAG_FLOW_SGLANG_MODEL_ID"
fi

if [[ "$model_path_overridden" != "1" ]]; then
  manual_model_candidates=()
  add_manual_model_candidate() {
    local candidate="$1"
    local existing
    [[ -z "$candidate" ]] && return
    if [[ ${#manual_model_candidates[@]} -gt 0 ]]; then
      for existing in "${manual_model_candidates[@]}"; do
        [[ "$existing" == "$candidate" ]] && return
      done
    fi
    manual_model_candidates+=("$candidate")
  }

  if [[ -n "${RAG_FLOW_SGLANG_MODEL_ID:-}" ]]; then
    model_basename="${RAG_FLOW_SGLANG_MODEL_ID##*/}"
    add_manual_model_candidate "$RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT/$RAG_FLOW_SGLANG_MODEL_ID"
    add_manual_model_candidate "$RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT/$model_basename"
  fi
  add_manual_model_candidate "$RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT/$RAG_FLOW_SGLANG_MODEL_PROFILE"

  if [[ -n "${RAG_FLOW_SGLANG_MODEL_PATH:-}" && -d "$RAG_FLOW_SGLANG_MODEL_PATH" ]]; then
    :
  else
    if [[ ${#manual_model_candidates[@]} -gt 0 ]]; then
      for candidate in "${manual_model_candidates[@]}"; do
        if [[ -d "$candidate" ]]; then
          RAG_FLOW_SGLANG_MODEL_PATH="$candidate"
          break
        fi
      done
    fi
  fi
  if [[ ( -z "${RAG_FLOW_SGLANG_MODEL_PATH:-}" || ! -d "$RAG_FLOW_SGLANG_MODEL_PATH" ) && -d "$default_modelscope_path" ]]; then
    RAG_FLOW_SGLANG_MODEL_PATH="$default_modelscope_path"
  fi
  if [[ ( -z "${RAG_FLOW_SGLANG_MODEL_PATH:-}" || ! -d "$RAG_FLOW_SGLANG_MODEL_PATH" ) && -d "$default_hf_path" ]]; then
    RAG_FLOW_SGLANG_MODEL_PATH="$default_hf_path"
  fi
fi

: "${RAG_FLOW_SGLANG_MODEL_PATH:=$default_model_path}"
if [[ -z "$RAG_FLOW_SGLANG_MODEL_PATH" ]]; then
  echo "Set RAG_FLOW_SGLANG_MODEL_PATH when RAG_FLOW_SGLANG_MODEL_PROFILE=custom." >&2
  exit 1
fi
if [[ -z "${RAG_FLOW_SGLANG_SERVED_MODEL_NAME:-}" ]]; then
  if [[ -n "$default_served_model_name" ]]; then
    RAG_FLOW_SGLANG_SERVED_MODEL_NAME="$default_served_model_name"
  else
    : "${RAG_FLOW_SGLANG_SERVED_MODEL_NAME:=${RAG_FLOW_LLM_MODEL:-}}"
  fi
fi

command=(
  "$RAG_FLOW_SGLANG_PYTHON"
  -m sglang.launch_server
  --model-path "$RAG_FLOW_SGLANG_MODEL_PATH"
  --port "$RAG_FLOW_SGLANG_PORT"
  --tp-size "$RAG_FLOW_SGLANG_TP_SIZE"
  --mem-fraction-static "$RAG_FLOW_SGLANG_MEM_FRACTION_STATIC"
  --context-length "$RAG_FLOW_SGLANG_CONTEXT_LENGTH"
)

[[ -n "${RAG_FLOW_SGLANG_SERVED_MODEL_NAME:-}" ]] && command+=(--served-model-name "$RAG_FLOW_SGLANG_SERVED_MODEL_NAME")
[[ -n "$RAG_FLOW_SGLANG_REASONING_PARSER" ]] && command+=(--reasoning-parser "$RAG_FLOW_SGLANG_REASONING_PARSER")
[[ -n "$RAG_FLOW_SGLANG_QUANTIZATION" ]] && command+=(--quantization "$RAG_FLOW_SGLANG_QUANTIZATION")
[[ -n "$RAG_FLOW_SGLANG_ATTENTION_BACKEND" ]] && command+=(--attention-backend "$RAG_FLOW_SGLANG_ATTENTION_BACKEND")
[[ -n "$RAG_FLOW_SGLANG_KV_CACHE_DTYPE" ]] && command+=(--kv-cache-dtype "$RAG_FLOW_SGLANG_KV_CACHE_DTYPE")

if [[ -n "$RAG_FLOW_SGLANG_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  env_extra_args=($RAG_FLOW_SGLANG_EXTRA_ARGS)
  command+=("${env_extra_args[@]}")
fi
if [[ ${#extra_args[@]} -gt 0 ]]; then
  command+=("${extra_args[@]}")
fi

echo "SGLang profile: $RAG_FLOW_SGLANG_MODEL_PROFILE"
echo "SGLang python: $RAG_FLOW_SGLANG_PYTHON"
echo "SGLang model path: $RAG_FLOW_SGLANG_MODEL_PATH"
echo "SGLang served model: ${RAG_FLOW_SGLANG_SERVED_MODEL_NAME:-}"
echo "SGLang port: $RAG_FLOW_SGLANG_PORT"
if [[ "$dry_run" == "1" ]]; then
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
