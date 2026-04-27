#!/usr/bin/env bash
set -euo pipefail

echo "Configure Aliyun mirrors for apt, pip, uv, Hugging Face, and conda."

cp /etc/apt/sources.list /etc/apt/sources.list.bak
sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
sed -i 's/ports.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
sed -i 's/mirrors.tuna.tsinghua.edu.cn/mirrors.aliyun.com/g' /etc/apt/sources.list
apt-get update -y

locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

if ! grep -q "RAG Flow AutoDL Environment" ~/.bashrc; then
  cat >> ~/.bashrc <<'EOF'

# --- RAG Flow AutoDL Environment ---
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export MINERU_MODEL_SOURCE=modelscope
export VLLM_USE_MODELSCOPE=True
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
# -----------------------------------
EOF
fi

cat > ~/.condarc <<'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.aliyun.com/anaconda/pkgs/main
  - https://mirrors.aliyun.com/anaconda/pkgs/r
  - https://mirrors.aliyun.com/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.aliyun.com/anaconda/cloud
  pytorch: https://mirrors.aliyun.com/anaconda/cloud
EOF

if command -v conda >/dev/null 2>&1; then
  conda clean -i -y >/dev/null 2>&1
fi

echo "Done. Run: source ~/.bashrc"
