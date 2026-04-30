#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env/common.sh
source "$SCRIPT_DIR/../env/common.sh"

dry_run=0
profile_overridden=0
model_id_overridden=0
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
    --model-id|--model)
      require_option_value "$@"
      RAG_FLOW_SGLANG_MODEL_ID="$2"
      model_id_overridden=1
      shift 2
      ;;
    --model-path|--local-dir)
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
    --revision)
      require_option_value "$@"
      RAG_FLOW_SGLANG_MODEL_REVISION="$2"
      shift 2
      ;;
    --python)
      require_option_value "$@"
      RAG_FLOW_SGLANG_PYTHON="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

: "${RAG_FLOW_SGLANG_MODEL_PROFILE:=qwen3.6-35b-a3b-gptq-int4}"
: "${RAG_FLOW_SGLANG_PYTHON:=${RAG_FLOW_LLM_PYTHON_BIN:-python}}"
: "${RAG_FLOW_SGLANG_MODEL_REVISION:=}"
: "${RAG_FLOW_SGLANG_DOWNLOAD_INSTALL_MODELSCOPE:=1}"

if [[ "$profile_overridden" == "1" && "$model_id_overridden" != "1" ]]; then
  unset RAG_FLOW_SGLANG_MODEL_ID
fi
if [[ "$profile_overridden" == "1" && "$model_path_overridden" != "1" ]]; then
  unset RAG_FLOW_SGLANG_MODEL_PATH
fi
if [[ "$profile_overridden" == "1" && "$served_model_name_overridden" != "1" ]]; then
  unset RAG_FLOW_SGLANG_SERVED_MODEL_NAME
fi

profile_key="$(printf '%s' "$RAG_FLOW_SGLANG_MODEL_PROFILE" | tr '[:upper:]' '[:lower:]')"
default_model_id=""
default_model_path=""
default_served_model_name=""
case "$profile_key" in
  qwen3.6-35b-a3b-gptq-int4|qwen3.6|palmfuture/qwen3.6-35b-a3b-gptq-int4)
    default_model_id="palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
    default_model_path="/root/.cache/modelscope/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
    default_served_model_name="palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
    ;;
  qwen3.5-35b-a3b-gptq-int4|qwen3.5|qwen/qwen3.5-35b-a3b-gptq-int4)
    default_model_id="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    default_model_path="/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    default_served_model_name="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
    ;;
  custom)
    default_model_id=""
    default_model_path=""
    ;;
  *)
    if [[ -z "${RAG_FLOW_SGLANG_MODEL_ID:-}" || -z "${RAG_FLOW_SGLANG_MODEL_PATH:-}" ]]; then
      echo "Unknown RAG_FLOW_SGLANG_MODEL_PROFILE=$RAG_FLOW_SGLANG_MODEL_PROFILE" >&2
      echo "Use qwen3.6-35b-a3b-gptq-int4, qwen3.5-35b-a3b-gptq-int4, custom, or set model id and path." >&2
      exit 1
    fi
    ;;
esac

: "${RAG_FLOW_SGLANG_MODEL_ID:=$default_model_id}"
: "${RAG_FLOW_SGLANG_MODEL_PATH:=$default_model_path}"
if [[ -z "$RAG_FLOW_SGLANG_MODEL_ID" || -z "$RAG_FLOW_SGLANG_MODEL_PATH" ]]; then
  echo "Set RAG_FLOW_SGLANG_MODEL_ID and RAG_FLOW_SGLANG_MODEL_PATH when RAG_FLOW_SGLANG_MODEL_PROFILE=custom." >&2
  exit 1
fi
if [[ "$model_id_overridden" == "1" && "$served_model_name_overridden" != "1" ]]; then
  unset RAG_FLOW_SGLANG_SERVED_MODEL_NAME
  default_served_model_name="$RAG_FLOW_SGLANG_MODEL_ID"
fi
if [[ -z "${RAG_FLOW_SGLANG_SERVED_MODEL_NAME:-}" ]]; then
  if [[ -n "$default_served_model_name" ]]; then
    RAG_FLOW_SGLANG_SERVED_MODEL_NAME="$default_served_model_name"
  else
    RAG_FLOW_SGLANG_SERVED_MODEL_NAME="$RAG_FLOW_SGLANG_MODEL_ID"
  fi
fi

download_env=(
  "RAG_FLOW_DOWNLOAD_MODEL_ID=$RAG_FLOW_SGLANG_MODEL_ID"
  "RAG_FLOW_DOWNLOAD_LOCAL_DIR=$RAG_FLOW_SGLANG_MODEL_PATH"
  "RAG_FLOW_DOWNLOAD_REVISION=$RAG_FLOW_SGLANG_MODEL_REVISION"
)
download_code='import os
from modelscope import snapshot_download
kwargs = {"local_dir": os.environ["RAG_FLOW_DOWNLOAD_LOCAL_DIR"]}
revision = os.environ.get("RAG_FLOW_DOWNLOAD_REVISION", "")
if revision:
    kwargs["revision"] = revision
path = snapshot_download(os.environ["RAG_FLOW_DOWNLOAD_MODEL_ID"], **kwargs)
print(f"Downloaded model to: {path}")'

echo "LLM model profile: $RAG_FLOW_SGLANG_MODEL_PROFILE"
echo "ModelScope model id: $RAG_FLOW_SGLANG_MODEL_ID"
echo "Local model path: $RAG_FLOW_SGLANG_MODEL_PATH"
echo "Served model name: $RAG_FLOW_SGLANG_SERVED_MODEL_NAME"
echo "Download python: $RAG_FLOW_SGLANG_PYTHON"
if [[ -n "$RAG_FLOW_SGLANG_MODEL_REVISION" ]]; then
  echo "Model revision: $RAG_FLOW_SGLANG_MODEL_REVISION"
fi

if [[ "$dry_run" == "1" ]]; then
  printf 'Command:'
  printf ' %q' env "${download_env[@]}" "$RAG_FLOW_SGLANG_PYTHON" -c "$download_code"
  printf '\n'
  exit 0
fi

if ! "$RAG_FLOW_SGLANG_PYTHON" -c 'import modelscope' >/dev/null 2>&1; then
  if truthy "$RAG_FLOW_SGLANG_DOWNLOAD_INSTALL_MODELSCOPE"; then
    echo "modelscope is not installed in $RAG_FLOW_SGLANG_PYTHON; installing it first."
    package_env=()
    [[ -n "$RAG_FLOW_PIP_INDEX_URL" ]] && package_env+=("PIP_INDEX_URL=$RAG_FLOW_PIP_INDEX_URL")
    [[ -n "$RAG_FLOW_UV_INDEX_URL" ]] && package_env+=("UV_INDEX_URL=$RAG_FLOW_UV_INDEX_URL")
    [[ -n "$RAG_FLOW_PIP_CACHE_DIR" ]] && package_env+=("PIP_CACHE_DIR=$RAG_FLOW_PIP_CACHE_DIR")
    [[ -n "$RAG_FLOW_UV_CACHE_DIR" ]] && package_env+=("UV_CACHE_DIR=$RAG_FLOW_UV_CACHE_DIR")
    if truthy "$RAG_FLOW_USE_UV" && command -v uv >/dev/null 2>&1; then
      env "${package_env[@]}" uv pip install --python "$RAG_FLOW_SGLANG_PYTHON" modelscope
    else
      env "${package_env[@]}" "$RAG_FLOW_SGLANG_PYTHON" -m pip install modelscope
    fi
  else
    echo "modelscope is not installed in $RAG_FLOW_SGLANG_PYTHON." >&2
    echo "Set RAG_FLOW_SGLANG_DOWNLOAD_INSTALL_MODELSCOPE=1 or install modelscope manually." >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "$RAG_FLOW_SGLANG_MODEL_PATH")"
env "${download_env[@]}" "$RAG_FLOW_SGLANG_PYTHON" -c "$download_code"

if truthy "$RAG_FLOW_UPDATE_ENV_FILE"; then
  set_env_var RAG_FLOW_SGLANG_MODEL_PROFILE "$RAG_FLOW_SGLANG_MODEL_PROFILE"
  set_env_var RAG_FLOW_SGLANG_MODEL_ID "$RAG_FLOW_SGLANG_MODEL_ID"
  set_env_var RAG_FLOW_SGLANG_MODEL_PATH "$RAG_FLOW_SGLANG_MODEL_PATH"
  set_env_var RAG_FLOW_SGLANG_SERVED_MODEL_NAME "$RAG_FLOW_SGLANG_SERVED_MODEL_NAME"
  set_env_var RAG_FLOW_LLM_MODEL "$RAG_FLOW_SGLANG_SERVED_MODEL_NAME"
fi
