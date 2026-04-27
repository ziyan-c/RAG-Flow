# Migration From The Old Workspace

The original workspace was a collection of focused scripts:

- `post-mineru...` became `src/rag_flow/preprocessing/`
- `content-list-json-chunking/chunking.py` became `src/rag_flow/chunking.py`
- `qdrant/*.py` became `src/rag_flow/indexing.py` and `src/rag_flow/retrieval.py`
- `qdrant/qdrant-test-retrieval-server.py` became `src/rag_flow/retrieval_client.py`
- `chatbot/qwen3.5.py` became `src/rag_flow/chat_cli.py`
- `chatbot/qwen3.5-sglang.sh` became `scripts/serve-llm-sglang.sh`
- `chatbot/ssh-tunnel.sh` became `scripts/ssh-tunnel.example.sh`
- `初始化小脚本/` became `scripts/init/`

Hard-coded paths are now environment variables. Hard-coded credentials were
removed rather than migrated.

## Compatibility Notes

The exported conda environments are preserved in `envs/` because this stack
depends on CUDA/PyTorch/Transformers/ColPali/SGLang compatibility. Use those
files as reproducibility references, but prefer a fresh lockfile once the target
GPU runtime is stable.
