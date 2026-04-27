#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

resolve_env_file() {
  local preferred="${RAG_FLOW_ENV_FILE:-$REPO_ROOT/.local/rag-flow.env}"
  local parent
  local fallback_dir
  local backup_path
  parent="$(dirname "$preferred")"
  if [[ -d "$parent" || ( ! -e "$parent" && ! -L "$parent" ) ]]; then
    echo "$preferred"
    return
  fi
  fallback_dir="${RAG_FLOW_RUNTIME_ROOT:-$HOME/autodl-tmp}/.local"
  mkdir -p "$fallback_dir"
  if [[ -L "$parent" ]]; then
    rm "$parent"
    ln -s "$fallback_dir" "$parent"
    echo "Repaired env directory symlink: $parent -> $fallback_dir" >&2
  else
    backup_path="$parent.bak.$(date +%Y%m%d%H%M%S)"
    mv "$parent" "$backup_path"
    ln -s "$fallback_dir" "$parent"
    echo "Moved unusable env directory path to: $backup_path" >&2
    echo "Created env directory symlink: $parent -> $fallback_dir" >&2
  fi
  echo "$preferred"
}

ENV_FILE="$(resolve_env_file)"

load_env_file() {
  local file="$1"
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done < "$file"
}

if [[ -f "$ENV_FILE" ]]; then
  load_env_file "$ENV_FILE"
fi

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

set_env_var() {
  local key="$1"
  local value="$2"
  local file="${3:-$ENV_FILE}"
  local tmp_file
  mkdir -p "$(dirname "$file")"
  touch "$file"
  tmp_file="$(mktemp)"
  awk -v key="$key" -v value="$value" '
    BEGIN { replaced = 0 }
    $0 ~ "^" key "=" {
      print key "=" value
      replaced = 1
      next
    }
    { print }
    END {
      if (!replaced) {
        print key "=" value
      }
    }
  ' "$file" > "$tmp_file"
  mv "$tmp_file" "$file"
}

normalize_mirror_profile() {
  case "${1:-}" in
    ali|aliyun|alicloud) echo "aliyun" ;;
    qq|qcloud|tencent|tencentyun) echo "tencent" ;;
    qinghua|tsinghua|tuna) echo "tuna" ;;
    bfsu) echo "bfsu" ;;
    sjtu|sjtug) echo "sjtug" ;;
    *) echo "${1:-}" ;;
  esac
}

profile_apt_mirror() {
  case "$(normalize_mirror_profile "$1")" in
    aliyun) echo "mirrors.aliyun.com" ;;
    tencent) echo "mirrors.cloud.tencent.com" ;;
    tuna) echo "mirrors.tuna.tsinghua.edu.cn" ;;
    bfsu) echo "mirrors.bfsu.edu.cn" ;;
    sjtug) echo "mirror.sjtu.edu.cn" ;;
    *) echo "$1" ;;
  esac
}

profile_pip_index() {
  case "$(normalize_mirror_profile "$1")" in
    aliyun) echo "https://mirrors.aliyun.com/pypi/simple/" ;;
    tencent) echo "https://mirrors.cloud.tencent.com/pypi/simple/" ;;
    tuna) echo "https://pypi.tuna.tsinghua.edu.cn/simple/" ;;
    bfsu) echo "https://mirrors.bfsu.edu.cn/pypi/web/simple/" ;;
    sjtug) echo "https://mirror.sjtu.edu.cn/pypi/web/simple/" ;;
    *) echo "$1" ;;
  esac
}

profile_conda_base() {
  case "$(normalize_mirror_profile "$1")" in
    aliyun) echo "https://mirrors.aliyun.com/anaconda" ;;
    tencent) echo "https://mirrors.cloud.tencent.com/anaconda" ;;
    tuna) echo "https://mirrors.tuna.tsinghua.edu.cn/anaconda" ;;
    bfsu) echo "https://mirrors.bfsu.edu.cn/anaconda" ;;
    sjtug) echo "https://mirror.sjtu.edu.cn/anaconda" ;;
    *) echo "$1" ;;
  esac
}

profile_probe_url() {
  local kind="$1"
  local profile="$2"
  case "$kind" in
    apt) echo "https://$(profile_apt_mirror "$profile")/ubuntu/" ;;
    pip) echo "$(profile_pip_index "$profile")" ;;
    conda) echo "$(profile_conda_base "$profile")/pkgs/main/linux-64/repodata.json" ;;
    *) return 1 ;;
  esac
}

probe_url() {
  local url="$1"
  local timeout="$2"
  local python_bin
  python_bin="$(command -v python3 || command -v python || true)"
  if [[ -z "$python_bin" ]]; then
    return 0
  fi
  "$python_bin" - "$url" "$timeout" <<'PY'
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
timeout = float(sys.argv[2])

def request(method):
    req = urllib.request.Request(url, method=method, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return 200 <= response.status < 400

try:
    ok = request("HEAD")
except urllib.error.HTTPError as exc:
    if exc.code not in {403, 405}:
        raise SystemExit(1)
    try:
        ok = request("GET")
    except Exception:
        raise SystemExit(1)
except Exception:
    raise SystemExit(1)

raise SystemExit(0 if ok else 1)
PY
}

profile_forced_bad() {
  local profile="$1"
  local -a raw=()
  local candidate
  [[ -z "${RAG_FLOW_INIT_MIRROR_FAIL_PROFILES:-}" ]] && return 1
  IFS=', ' read -r -a raw <<< "${RAG_FLOW_INIT_MIRROR_FAIL_PROFILES:-}"
  for candidate in "${raw[@]}"; do
    [[ -z "$candidate" ]] && continue
    if [[ "$(normalize_mirror_profile "$candidate")" == "$profile" ]]; then
      return 0
    fi
  done
  return 1
}

profile_available() {
  local kind="$1"
  local profile="$2"
  local url
  if profile_forced_bad "$profile"; then
    echo "Mirror profile disabled for $kind: $profile" >&2
    return 1
  fi
  if ! truthy "$RAG_FLOW_INIT_MIRROR_PROBE"; then
    return 0
  fi
  url="$(profile_probe_url "$kind" "$profile")"
  if probe_url "$url" "$RAG_FLOW_INIT_MIRROR_PROBE_TIMEOUT"; then
    return 0
  fi
  echo "Mirror probe failed for $kind profile '$profile': $url" >&2
  return 1
}

select_mirror_profile() {
  local kind="$1"
  local fallback=""
  local raw
  local profile
  local -a candidates=()
  IFS=', ' read -r -a candidates <<< "$RAG_FLOW_INIT_MIRROR_ORDER"
  for raw in "${candidates[@]}"; do
    [[ -z "$raw" ]] && continue
    profile="$(normalize_mirror_profile "$raw")"
    fallback="$profile"
    if profile_available "$kind" "$profile"; then
      echo "$profile"
      return 0
    fi
  done
  echo "${fallback:-aliyun}"
}

managed_mirror_value() {
  local kind="$1"
  local value="${2:-}"
  [[ -z "$value" ]] && return 0
  case "$kind:$value" in
    apt:mirrors.aliyun.com|apt:mirrors.cloud.tencent.com|apt:mirrors.tencent.com|apt:mirrors.tuna.tsinghua.edu.cn|apt:mirrors.bfsu.edu.cn|apt:mirror.sjtu.edu.cn)
      return 0
      ;;
  esac
  case "$kind" in
    pip)
      case "$value" in
        https://mirrors.aliyun.com/pypi/simple*|https://mirrors.cloud.tencent.com/pypi/simple*|https://mirrors.tencent.com/pypi/simple*|https://pypi.tuna.tsinghua.edu.cn/simple*|https://mirrors.bfsu.edu.cn/pypi/web/simple*|https://mirror.sjtu.edu.cn/pypi/web/simple*)
          return 0
          ;;
      esac
      ;;
    conda)
      case "$value" in
        https://mirrors.aliyun.com/anaconda*|https://mirrors.cloud.tencent.com/anaconda*|https://mirrors.tencent.com/anaconda*|https://mirrors.tuna.tsinghua.edu.cn/anaconda*|https://mirrors.bfsu.edu.cn/anaconda*|https://mirror.sjtu.edu.cn/anaconda*)
          return 0
          ;;
      esac
      ;;
  esac
  return 1
}

choose_managed_value() {
  local kind="$1"
  local current="$2"
  local selected="$3"
  if managed_mirror_value "$kind" "$current"; then
    echo "$selected"
    return
  fi
  echo "$current"
}

: "${RAG_FLOW_INIT_MIRROR_ORDER:=aliyun,tencent,tuna}"
: "${RAG_FLOW_INIT_MIRROR_PROBE:=1}"
: "${RAG_FLOW_INIT_MIRROR_PROBE_TIMEOUT:=5}"
: "${RAG_FLOW_INIT_UPDATE_ENV_FILE:=1}"
: "${RAG_FLOW_INIT_REWRITE_APT:=1}"
: "${RAG_FLOW_INIT_APT_MIRROR:=mirrors.aliyun.com}"
: "${RAG_FLOW_INIT_APT_UPDATE:=1}"
: "${RAG_FLOW_INIT_CONFIGURE_LOCALE:=1}"
: "${RAG_FLOW_INIT_LOCALE:=en_US.UTF-8}"
: "${RAG_FLOW_INIT_WRITE_BASHRC:=1}"
: "${RAG_FLOW_INIT_BASHRC:=$HOME/.bashrc}"
: "${RAG_FLOW_RUNTIME_ROOT:=$HOME/autodl-tmp}"
: "${RAG_FLOW_INIT_HF_ENDPOINT:=https://hf-mirror.com}"
: "${RAG_FLOW_INIT_HF_HOME:=$RAG_FLOW_RUNTIME_ROOT/.cache/huggingface}"
: "${RAG_FLOW_INIT_VLLM_USE_MODELSCOPE:=True}"
: "${RAG_FLOW_INIT_MINERU_MODEL_SOURCE:=${RAG_FLOW_MINERU_MODEL_SOURCE:-modelscope}}"
: "${RAG_FLOW_INIT_PIP_INDEX_URL:=https://mirrors.aliyun.com/pypi/simple/}"
: "${RAG_FLOW_INIT_UV_INDEX_URL:=$RAG_FLOW_INIT_PIP_INDEX_URL}"
: "${RAG_FLOW_INIT_PIP_CACHE_DIR:=$RAG_FLOW_RUNTIME_ROOT/.cache/pip}"
: "${RAG_FLOW_INIT_UV_CACHE_DIR:=$RAG_FLOW_RUNTIME_ROOT/.cache/uv}"
: "${RAG_FLOW_INIT_TORCH_HOME:=$RAG_FLOW_RUNTIME_ROOT/.cache/torch}"
: "${RAG_FLOW_INIT_MODELSCOPE_CACHE:=$RAG_FLOW_RUNTIME_ROOT/.cache/modelscope}"
: "${RAG_FLOW_INIT_WRITE_CONDARC:=1}"
: "${RAG_FLOW_INIT_CONDARC:=$HOME/.condarc}"
: "${RAG_FLOW_INIT_CONDA_PKGS_DIRS:=$RAG_FLOW_RUNTIME_ROOT/conda-pkgs}"
: "${RAG_FLOW_INIT_CONDA_MAIN_CHANNEL:=https://mirrors.aliyun.com/anaconda/pkgs/main}"
: "${RAG_FLOW_INIT_CONDA_R_CHANNEL:=https://mirrors.aliyun.com/anaconda/pkgs/r}"
: "${RAG_FLOW_INIT_CONDA_MSYS2_CHANNEL:=https://mirrors.aliyun.com/anaconda/pkgs/msys2}"
: "${RAG_FLOW_INIT_CONDA_FORGE_CHANNEL:=https://mirrors.aliyun.com/anaconda/cloud}"
: "${RAG_FLOW_INIT_CONDA_PYTORCH_CHANNEL:=https://mirrors.aliyun.com/anaconda/cloud}"
: "${RAG_FLOW_INIT_CONDA_CLEAN_INDEX:=1}"

apt_mirror_profile="$(select_mirror_profile apt)"
pip_mirror_profile="$(select_mirror_profile pip)"
conda_mirror_profile="$(select_mirror_profile conda)"
selected_apt_mirror="$(profile_apt_mirror "$apt_mirror_profile")"
selected_pip_index="$(profile_pip_index "$pip_mirror_profile")"
selected_conda_base="$(profile_conda_base "$conda_mirror_profile")"

RAG_FLOW_INIT_APT_MIRROR="$(choose_managed_value apt "$RAG_FLOW_INIT_APT_MIRROR" "$selected_apt_mirror")"
RAG_FLOW_INIT_PIP_INDEX_URL="$(choose_managed_value pip "$RAG_FLOW_INIT_PIP_INDEX_URL" "$selected_pip_index")"
RAG_FLOW_INIT_UV_INDEX_URL="$(choose_managed_value pip "$RAG_FLOW_INIT_UV_INDEX_URL" "$selected_pip_index")"
RAG_FLOW_INIT_CONDA_MAIN_CHANNEL="$(choose_managed_value conda "$RAG_FLOW_INIT_CONDA_MAIN_CHANNEL" "$selected_conda_base/pkgs/main")"
RAG_FLOW_INIT_CONDA_R_CHANNEL="$(choose_managed_value conda "$RAG_FLOW_INIT_CONDA_R_CHANNEL" "$selected_conda_base/pkgs/r")"
RAG_FLOW_INIT_CONDA_MSYS2_CHANNEL="$(choose_managed_value conda "$RAG_FLOW_INIT_CONDA_MSYS2_CHANNEL" "$selected_conda_base/pkgs/msys2")"
RAG_FLOW_INIT_CONDA_FORGE_CHANNEL="$(choose_managed_value conda "$RAG_FLOW_INIT_CONDA_FORGE_CHANNEL" "$selected_conda_base/cloud")"
RAG_FLOW_INIT_CONDA_PYTORCH_CHANNEL="$(choose_managed_value conda "$RAG_FLOW_INIT_CONDA_PYTORCH_CHANNEL" "$selected_conda_base/cloud")"

echo "Configure local mirrors and runtime environment."
echo "Using env file: $ENV_FILE"
echo "Mirror profiles: apt=$apt_mirror_profile pip=$pip_mirror_profile conda=$conda_mirror_profile"
mkdir -p \
  "$RAG_FLOW_RUNTIME_ROOT" \
  "$RAG_FLOW_INIT_HF_HOME" \
  "$RAG_FLOW_INIT_PIP_CACHE_DIR" \
  "$RAG_FLOW_INIT_UV_CACHE_DIR" \
  "$RAG_FLOW_INIT_TORCH_HOME" \
  "$RAG_FLOW_INIT_MODELSCOPE_CACHE" \
  "$RAG_FLOW_INIT_CONDA_PKGS_DIRS"

if truthy "$RAG_FLOW_INIT_REWRITE_APT"; then
  if [[ -f /etc/apt/sources.list ]]; then
    apt_sources_changed=0
    tmp_sources="$(mktemp)"
    sed -E "s#archive.ubuntu.com|security.ubuntu.com|ports.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|repo.huaweicloud.com#$RAG_FLOW_INIT_APT_MIRROR#g" /etc/apt/sources.list > "$tmp_sources"
    if cmp -s "$tmp_sources" /etc/apt/sources.list; then
      echo "Apt sources already configured."
      rm -f "$tmp_sources"
    else
      apt_sources_changed=1
      cp /etc/apt/sources.list "/etc/apt/sources.list.bak.$(date +%Y%m%d%H%M%S)"
      cat "$tmp_sources" > /etc/apt/sources.list
      rm -f "$tmp_sources"
    fi
    if truthy "$RAG_FLOW_INIT_APT_UPDATE" && [[ "$apt_sources_changed" == "1" ]]; then
      apt-get update -y
    fi
  else
    echo "Skip apt mirror rewrite: /etc/apt/sources.list not found."
  fi
fi

if truthy "$RAG_FLOW_INIT_CONFIGURE_LOCALE"; then
  if command -v locale-gen >/dev/null 2>&1; then
    locale-gen "$RAG_FLOW_INIT_LOCALE"
  fi
  if command -v update-locale >/dev/null 2>&1; then
    update-locale LANG="$RAG_FLOW_INIT_LOCALE" LC_ALL="$RAG_FLOW_INIT_LOCALE"
  fi
fi

if truthy "$RAG_FLOW_INIT_WRITE_BASHRC"; then
  write_bashrc_block() {
    cat <<EOF

# --- RAG Flow AutoDL Environment ---
export LC_ALL=$RAG_FLOW_INIT_LOCALE
export LANG=$RAG_FLOW_INIT_LOCALE
export LANGUAGE=en_US:en
export HF_ENDPOINT=$RAG_FLOW_INIT_HF_ENDPOINT
export HF_HOME=$RAG_FLOW_INIT_HF_HOME
export MODELSCOPE_CACHE=$RAG_FLOW_INIT_MODELSCOPE_CACHE
export TORCH_HOME=$RAG_FLOW_INIT_TORCH_HOME
export VLLM_USE_MODELSCOPE=$RAG_FLOW_INIT_VLLM_USE_MODELSCOPE
export MINERU_MODEL_SOURCE=$RAG_FLOW_INIT_MINERU_MODEL_SOURCE
export PIP_INDEX_URL=$RAG_FLOW_INIT_PIP_INDEX_URL
export UV_INDEX_URL=$RAG_FLOW_INIT_UV_INDEX_URL
export PIP_CACHE_DIR=$RAG_FLOW_INIT_PIP_CACHE_DIR
export UV_CACHE_DIR=$RAG_FLOW_INIT_UV_CACHE_DIR
export CONDA_PKGS_DIRS=$RAG_FLOW_INIT_CONDA_PKGS_DIRS
# -----------------------------------
EOF
  }
  tmp_bashrc="$(mktemp)"
  mkdir -p "$(dirname "$RAG_FLOW_INIT_BASHRC")"
  touch "$RAG_FLOW_INIT_BASHRC"
  awk '
    /# --- RAG Flow AutoDL Environment ---/ { skip = 1; next }
    /# -----------------------------------/ && skip { skip = 0; next }
    !skip { print }
  ' "$RAG_FLOW_INIT_BASHRC" > "$tmp_bashrc"
  write_bashrc_block >> "$tmp_bashrc"
  mv "$tmp_bashrc" "$RAG_FLOW_INIT_BASHRC"
fi

if truthy "$RAG_FLOW_INIT_WRITE_CONDARC"; then
  cat > "$RAG_FLOW_INIT_CONDARC" <<EOF
channels:
  - defaults
show_channel_urls: true
pkgs_dirs:
  - $RAG_FLOW_INIT_CONDA_PKGS_DIRS
default_channels:
  - $RAG_FLOW_INIT_CONDA_MAIN_CHANNEL
  - $RAG_FLOW_INIT_CONDA_R_CHANNEL
  - $RAG_FLOW_INIT_CONDA_MSYS2_CHANNEL
custom_channels:
  conda-forge: $RAG_FLOW_INIT_CONDA_FORGE_CHANNEL
  pytorch: $RAG_FLOW_INIT_CONDA_PYTORCH_CHANNEL
EOF
fi

if truthy "$RAG_FLOW_INIT_UPDATE_ENV_FILE"; then
  set_env_var "RAG_FLOW_INIT_APT_MIRROR" "$RAG_FLOW_INIT_APT_MIRROR"
  set_env_var "RAG_FLOW_INIT_PIP_INDEX_URL" "$RAG_FLOW_INIT_PIP_INDEX_URL"
  set_env_var "RAG_FLOW_INIT_UV_INDEX_URL" "$RAG_FLOW_INIT_UV_INDEX_URL"
  set_env_var "RAG_FLOW_INIT_CONDA_MAIN_CHANNEL" "$RAG_FLOW_INIT_CONDA_MAIN_CHANNEL"
  set_env_var "RAG_FLOW_INIT_CONDA_R_CHANNEL" "$RAG_FLOW_INIT_CONDA_R_CHANNEL"
  set_env_var "RAG_FLOW_INIT_CONDA_MSYS2_CHANNEL" "$RAG_FLOW_INIT_CONDA_MSYS2_CHANNEL"
  set_env_var "RAG_FLOW_INIT_CONDA_FORGE_CHANNEL" "$RAG_FLOW_INIT_CONDA_FORGE_CHANNEL"
  set_env_var "RAG_FLOW_INIT_CONDA_PYTORCH_CHANNEL" "$RAG_FLOW_INIT_CONDA_PYTORCH_CHANNEL"
  set_env_var "RAG_FLOW_PIP_INDEX_URL" "$RAG_FLOW_INIT_PIP_INDEX_URL"
  set_env_var "RAG_FLOW_UV_INDEX_URL" "$RAG_FLOW_INIT_UV_INDEX_URL"
  set_env_var "RAG_FLOW_CONDA_PKGS_DIRS" "$RAG_FLOW_INIT_CONDA_PKGS_DIRS"
  set_env_var "RAG_FLOW_MINERU_MODEL_SOURCE" "$RAG_FLOW_INIT_MINERU_MODEL_SOURCE"
fi

if truthy "$RAG_FLOW_INIT_CONDA_CLEAN_INDEX" && command -v conda >/dev/null 2>&1; then
  conda clean -i -y >/dev/null 2>&1
fi

if truthy "$RAG_FLOW_INIT_WRITE_BASHRC"; then
  echo "Done. Run: source $RAG_FLOW_INIT_BASHRC"
else
  echo "Done."
fi
