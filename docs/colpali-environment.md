uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

uv pip install --upgrade transformers peft accelerate colpali-engine

uv pip install --upgrade transformers accelerate qwen-vl-utils

# 强制更新所有 nvidia-cuda 相关的库
uv pip install --upgrade nvidia-nccl-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 --index-url https://download.pytorch.org/whl/nightly/cu128

uv pip uninstall  torch torchvision torchaudio

uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128 


uv pip install --upgrade transformers peft accelerate colpali-engine

uv pip install transformers==4.46.3 peft==0.13.2

uv pip install colpali-engine==0.3.1

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 