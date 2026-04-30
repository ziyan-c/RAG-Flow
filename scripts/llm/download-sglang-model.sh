#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../env/common.sh
source "$SCRIPT_DIR/../env/common.sh"

dry_run=0
source_overridden=0
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
    --source)
      require_option_value "$@"
      RAG_FLOW_SGLANG_DOWNLOAD_SOURCE="$2"
      source_overridden=1
      shift 2
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

: "${RAG_FLOW_SGLANG_DOWNLOAD_SOURCE:=auto}"
: "${RAG_FLOW_SGLANG_MODEL_PROFILE:=qwen3.6-35b-a3b-gptq-int4}"
: "${RAG_FLOW_SGLANG_PYTHON:=${RAG_FLOW_LLM_PYTHON_BIN:-python}}"
: "${RAG_FLOW_SGLANG_MODEL_REVISION:=}"
: "${RAG_FLOW_SGLANG_DOWNLOAD_INSTALL_MODELSCOPE:=1}"
: "${RAG_FLOW_SGLANG_DOWNLOAD_INSTALL_HUGGINGFACE_HUB:=1}"

download_source_key="$(printf '%s' "$RAG_FLOW_SGLANG_DOWNLOAD_SOURCE" | tr '[:upper:]' '[:lower:]')"
case "$download_source_key" in
  auto)
    RAG_FLOW_SGLANG_DOWNLOAD_SOURCE="auto"
    ;;
  modelscope)
    RAG_FLOW_SGLANG_DOWNLOAD_SOURCE="modelscope"
    ;;
  hf|huggingface|huggingface-hub)
    RAG_FLOW_SGLANG_DOWNLOAD_SOURCE="hf"
    ;;
  *)
    echo "Unknown RAG_FLOW_SGLANG_DOWNLOAD_SOURCE=$RAG_FLOW_SGLANG_DOWNLOAD_SOURCE. Use auto, modelscope, or hf." >&2
    exit 1
    ;;
esac

if [[ "$profile_overridden" == "1" && "$model_id_overridden" != "1" ]]; then
  unset RAG_FLOW_SGLANG_MODEL_ID
fi
if [[ ( "$profile_overridden" == "1" || "$source_overridden" == "1" ) && "$model_path_overridden" != "1" ]]; then
  unset RAG_FLOW_SGLANG_MODEL_PATH
fi
if [[ "$profile_overridden" == "1" && "$served_model_name_overridden" != "1" ]]; then
  unset RAG_FLOW_SGLANG_SERVED_MODEL_NAME
fi

profile_key="$(printf '%s' "$RAG_FLOW_SGLANG_MODEL_PROFILE" | tr '[:upper:]' '[:lower:]')"
default_model_id=""
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
    ;;
  *)
    if [[ -z "${RAG_FLOW_SGLANG_MODEL_ID:-}" ]]; then
      echo "Unknown RAG_FLOW_SGLANG_MODEL_PROFILE=$RAG_FLOW_SGLANG_MODEL_PROFILE" >&2
      echo "Use qwen3.6-35b-a3b-gptq-int4, qwen3.5-35b-a3b-gptq-int4, custom, or set model id and path." >&2
      exit 1
    fi
    ;;
esac

: "${RAG_FLOW_SGLANG_MODEL_ID:=$default_model_id}"
if [[ -z "$RAG_FLOW_SGLANG_MODEL_ID" ]]; then
  echo "Set RAG_FLOW_SGLANG_MODEL_ID when RAG_FLOW_SGLANG_MODEL_PROFILE=custom." >&2
  exit 1
fi

if [[ -z "$default_modelscope_path" ]]; then
  default_modelscope_path="/root/.cache/modelscope/hub/models/$RAG_FLOW_SGLANG_MODEL_ID"
fi
if [[ -z "$default_hf_path" ]]; then
  default_hf_path="/root/.cache/huggingface/hub/models/$RAG_FLOW_SGLANG_MODEL_ID"
fi

model_path_explicit=0
if [[ -n "${RAG_FLOW_SGLANG_MODEL_PATH:-}" ]]; then
  model_path_explicit=1
fi
if [[ "$model_path_explicit" != "1" ]]; then
  if [[ "$RAG_FLOW_SGLANG_DOWNLOAD_SOURCE" == "hf" ]]; then
    RAG_FLOW_SGLANG_MODEL_PATH="$default_hf_path"
  else
    RAG_FLOW_SGLANG_MODEL_PATH="$default_modelscope_path"
  fi
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

prepared_source=""
prepared_label=""
prepared_package_module=""
prepared_package_name=""
prepared_local_dir=""
prepared_download_code=""

prepare_download_source() {
  local source="$1"
  prepared_source="$source"
  case "$source" in
    modelscope)
      prepared_label="ModelScope"
      prepared_package_module="modelscope"
      prepared_package_name="modelscope"
      if [[ "$model_path_explicit" == "1" ]]; then
        prepared_local_dir="$RAG_FLOW_SGLANG_MODEL_PATH"
      else
        prepared_local_dir="$default_modelscope_path"
      fi
      prepared_download_code='import os
from modelscope import snapshot_download
kwargs = {"local_dir": os.environ["RAG_FLOW_DOWNLOAD_LOCAL_DIR"]}
revision = os.environ.get("RAG_FLOW_DOWNLOAD_REVISION", "")
if revision:
    kwargs["revision"] = revision
path = snapshot_download(os.environ["RAG_FLOW_DOWNLOAD_MODEL_ID"], **kwargs)
print(f"Downloaded model to: {path}")'
      ;;
    hf)
      prepared_label="Hugging Face"
      prepared_package_module="huggingface_hub"
      prepared_package_name="huggingface_hub"
      if [[ "$model_path_explicit" == "1" ]]; then
        prepared_local_dir="$RAG_FLOW_SGLANG_MODEL_PATH"
      else
        prepared_local_dir="$default_hf_path"
      fi
      prepared_download_code='import os
from huggingface_hub import snapshot_download
kwargs = {"local_dir": os.environ["RAG_FLOW_DOWNLOAD_LOCAL_DIR"]}
revision = os.environ.get("RAG_FLOW_DOWNLOAD_REVISION", "")
if revision:
    kwargs["revision"] = revision
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("RAG_FLOW_SGLANG_HF_TOKEN")
if token:
    kwargs["token"] = token
path = snapshot_download(os.environ["RAG_FLOW_DOWNLOAD_MODEL_ID"], **kwargs)
print(f"Downloaded model to: {path}")'
      ;;
    *)
      echo "Unknown download source: $source" >&2
      return 1
      ;;
  esac
}

print_download_plan() {
  local download_env=(
    "RAG_FLOW_DOWNLOAD_MODEL_ID=$RAG_FLOW_SGLANG_MODEL_ID"
    "RAG_FLOW_DOWNLOAD_LOCAL_DIR=$prepared_local_dir"
    "RAG_FLOW_DOWNLOAD_REVISION=$RAG_FLOW_SGLANG_MODEL_REVISION"
  )
  echo "$prepared_label model id: $RAG_FLOW_SGLANG_MODEL_ID"
  echo "$prepared_label local model path: $prepared_local_dir"
  echo "Local model path: $prepared_local_dir"
  printf 'Command:'
  printf ' %q' env "${download_env[@]}" "$RAG_FLOW_SGLANG_PYTHON" -c "$prepared_download_code"
  printf '\n'
}

install_downloader_if_needed() {
  if "$RAG_FLOW_SGLANG_PYTHON" -c "import $prepared_package_module" >/dev/null 2>&1; then
    return 0
  fi

  local install_downloader=0
  if [[ "$prepared_source" == "modelscope" ]] && truthy "$RAG_FLOW_SGLANG_DOWNLOAD_INSTALL_MODELSCOPE"; then
    install_downloader=1
  elif [[ "$prepared_source" == "hf" ]] && truthy "$RAG_FLOW_SGLANG_DOWNLOAD_INSTALL_HUGGINGFACE_HUB"; then
    install_downloader=1
  fi

  if [[ "$install_downloader" != "1" ]]; then
    echo "$prepared_package_module is not installed in $RAG_FLOW_SGLANG_PYTHON." >&2
    echo "Enable the matching auto-install flag or install $prepared_package_name manually." >&2
    return 1
  fi

  echo "$prepared_package_module is not installed in $RAG_FLOW_SGLANG_PYTHON; installing it first."
  local package_env=()
  [[ -n "$RAG_FLOW_PIP_INDEX_URL" ]] && package_env+=("PIP_INDEX_URL=$RAG_FLOW_PIP_INDEX_URL")
  [[ -n "$RAG_FLOW_UV_INDEX_URL" ]] && package_env+=("UV_INDEX_URL=$RAG_FLOW_UV_INDEX_URL")
  [[ -n "$RAG_FLOW_PIP_CACHE_DIR" ]] && package_env+=("PIP_CACHE_DIR=$RAG_FLOW_PIP_CACHE_DIR")
  [[ -n "$RAG_FLOW_UV_CACHE_DIR" ]] && package_env+=("UV_CACHE_DIR=$RAG_FLOW_UV_CACHE_DIR")
  if truthy "$RAG_FLOW_USE_UV" && command -v uv >/dev/null 2>&1; then
    env "${package_env[@]}" uv pip install --python "$RAG_FLOW_SGLANG_PYTHON" "$prepared_package_name"
  else
    env "${package_env[@]}" "$RAG_FLOW_SGLANG_PYTHON" -m pip install "$prepared_package_name"
  fi
}

run_download_source() {
  local source="$1"
  prepare_download_source "$source"
  echo "Trying download source: $prepared_label"
  echo "$prepared_label model id: $RAG_FLOW_SGLANG_MODEL_ID"
  echo "$prepared_label local model path: $prepared_local_dir"
  echo "Local model path: $prepared_local_dir"

  install_downloader_if_needed || return 1
  mkdir -p "$(dirname "$prepared_local_dir")"
  local download_env=(
    "RAG_FLOW_DOWNLOAD_MODEL_ID=$RAG_FLOW_SGLANG_MODEL_ID"
    "RAG_FLOW_DOWNLOAD_LOCAL_DIR=$prepared_local_dir"
    "RAG_FLOW_DOWNLOAD_REVISION=$RAG_FLOW_SGLANG_MODEL_REVISION"
  )
  env "${download_env[@]}" "$RAG_FLOW_SGLANG_PYTHON" -c "$prepared_download_code" || return 1
  RAG_FLOW_SGLANG_DOWNLOAD_SOURCE="$source"
  RAG_FLOW_SGLANG_MODEL_PATH="$prepared_local_dir"
  return 0
}

download_source_display="$RAG_FLOW_SGLANG_DOWNLOAD_SOURCE"
case "$RAG_FLOW_SGLANG_DOWNLOAD_SOURCE" in
  modelscope) download_source_display="ModelScope" ;;
  hf) download_source_display="Hugging Face" ;;
esac

echo "LLM model profile: $RAG_FLOW_SGLANG_MODEL_PROFILE"
echo "Download source: $download_source_display"
echo "Served model name: $RAG_FLOW_SGLANG_SERVED_MODEL_NAME"
echo "Download python: $RAG_FLOW_SGLANG_PYTHON"
if [[ -n "$RAG_FLOW_SGLANG_MODEL_REVISION" ]]; then
  echo "Model revision: $RAG_FLOW_SGLANG_MODEL_REVISION"
fi

sources=()
case "$RAG_FLOW_SGLANG_DOWNLOAD_SOURCE" in
  auto)
    sources=(modelscope hf)
    ;;
  modelscope|hf)
    sources=("$RAG_FLOW_SGLANG_DOWNLOAD_SOURCE")
    ;;
esac

if [[ "$dry_run" == "1" ]]; then
  if [[ "$RAG_FLOW_SGLANG_DOWNLOAD_SOURCE" == "auto" ]]; then
    echo "Download order: ModelScope, then Hugging Face"
  fi
  for source in "${sources[@]}"; do
    prepare_download_source "$source"
    print_download_plan
  done
  exit 0
fi

download_ok=0
for source in "${sources[@]}"; do
  if run_download_source "$source"; then
    download_ok=1
    break
  fi
  if [[ "$RAG_FLOW_SGLANG_DOWNLOAD_SOURCE" == "auto" && "$source" == "modelscope" ]]; then
    echo "ModelScope download failed; trying Hugging Face fallback." >&2
  fi
done

if [[ "$download_ok" != "1" ]]; then
  echo "LLM model download failed for all configured sources." >&2
  exit 1
fi

if truthy "$RAG_FLOW_UPDATE_ENV_FILE"; then
  set_env_var RAG_FLOW_SGLANG_DOWNLOAD_SOURCE "$RAG_FLOW_SGLANG_DOWNLOAD_SOURCE"
  set_env_var RAG_FLOW_SGLANG_MODEL_PROFILE "$RAG_FLOW_SGLANG_MODEL_PROFILE"
  set_env_var RAG_FLOW_SGLANG_MODEL_ID "$RAG_FLOW_SGLANG_MODEL_ID"
  set_env_var RAG_FLOW_SGLANG_MODEL_PATH "$RAG_FLOW_SGLANG_MODEL_PATH"
  set_env_var RAG_FLOW_SGLANG_SERVED_MODEL_NAME "$RAG_FLOW_SGLANG_SERVED_MODEL_NAME"
  set_env_var RAG_FLOW_LLM_MODEL "$RAG_FLOW_SGLANG_SERVED_MODEL_NAME"
fi
