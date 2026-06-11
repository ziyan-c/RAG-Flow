from __future__ import annotations

from pathlib import Path

from rag_flow.config import ModelServerConfig
from rag_flow.model_server import _build_sglang_env


def test_build_sglang_env_uses_base_url_port_and_model_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_FLOW_RUNTIME_ROOT", str(tmp_path))
    server_config = ModelServerConfig(
        sglang_model_profile="custom",
        sglang_model_id="palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4",
        sglang_model_path=tmp_path / "models" / "qwen",
        sglang_served_model_name="palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4",
        sglang_local_model_root=Path("/models"),
        sglang_mem_fraction_static="0.5",
        sglang_context_length="32768",
        sglang_tp_size="1",
        sglang_quantization="none",
    )

    env = _build_sglang_env(
        base_url="http://127.0.0.1:8081/v1",
        model="palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4",
        server_config=server_config,
    )

    assert env["RAG_FLOW_SGLANG_PORT"] == "8081"
    assert env["RAG_FLOW_LLM_MODEL"] == "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
    assert env["RAG_FLOW_SGLANG_MODEL_PROFILE"] == "custom"
    assert env["RAG_FLOW_SGLANG_MODEL_ID"] == "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
    assert env["RAG_FLOW_SGLANG_MODEL_PATH"] == str(tmp_path / "models" / "qwen")
    assert env["RAG_FLOW_SGLANG_SERVED_MODEL_NAME"] == "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
    assert env["RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT"] == "/models"
    assert env["RAG_FLOW_SGLANG_MEM_FRACTION_STATIC"] == "0.5"
    assert env["RAG_FLOW_SGLANG_CONTEXT_LENGTH"] == "32768"
    assert env["RAG_FLOW_SGLANG_TP_SIZE"] == "1"
    assert env["RAG_FLOW_SGLANG_QUANTIZATION"] == "none"
