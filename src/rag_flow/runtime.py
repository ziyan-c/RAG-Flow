from __future__ import annotations


DEFAULT_TRUSTED_REMOTE_CODE_MODELS = ("Qwen/Qwen3.5-9B",)


def get_torch_device(*, require_cuda: bool = False, feature: str = "This command") -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if require_cuda:
        raise RuntimeError(
            f"{feature} requires a CUDA GPU, but torch.cuda.is_available() is false. "
            "Run it on a GPU machine or disable the CUDA-only option."
        )
    return "cpu"


def require_trusted_remote_code_model(
    model_name: str,
    *,
    allowed_models: tuple[str, ...] = DEFAULT_TRUSTED_REMOTE_CODE_MODELS,
) -> None:
    if model_name in allowed_models:
        return
    allowed = ", ".join(allowed_models) or "<none>"
    raise ValueError(
        f"Refusing to load remote model code for {model_name!r}. "
        f"Allowed remote-code models: {allowed}. "
        "Set RAG_FLOW_TRUSTED_REMOTE_CODE_MODELS only for model repositories you trust."
    )
