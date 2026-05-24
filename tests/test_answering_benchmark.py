from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rag_flow.benchmark import answering
from rag_flow.config import (
    AppConfig,
    CaptioningConfig,
    ChunkingConfig,
    IndexingConfig,
    MinerUConfig,
    ModelConfig,
    PatchingConfig,
    PathsConfig,
    RetrievalConfig,
    ServerConfig,
)


def _config(tmp_path: Path) -> AppConfig:
    paths = PathsConfig(
        base_dir=tmp_path,
        source_name="manual.pdf",
        source_pdf=tmp_path / "manual.pdf",
        content_json=tmp_path / "content.json",
        sectioned_json=tmp_path / "sectioned.json",
        patched_json=tmp_path / "patched.json",
        captioned_json=tmp_path / "captioned.json",
        chunks_json=tmp_path / "chunks.json",
        db_path=tmp_path / "db",
        collection_name="manuals",
    )
    models = ModelConfig(
        dense_model="dense",
        sparse_model="sparse",
        colpali_model="colpali",
        vlm_model="vlm",
        vlm_model_revision="",
        trusted_remote_code_models=(),
        llm_base_url="http://localhost:8080/v1",
        llm_api_key="EMPTY",
        llm_model="qwen",
        llm_max_tokens=20000,
    )
    return AppConfig(
        paths=paths,
        models=models,
        retrieval=RetrievalConfig(
            retrieval_k=150,
            final_top_k=80,
            rrf_k=10,
            visual_weight=2.5,
            quantized_colpali=True,
            max_context_tokens=10000,
        ),
        server=ServerConfig(
            host="127.0.0.1",
            port=8000,
            retriever_url="http://127.0.0.1:8000/retrieve",
            retriever_api_key="",
            max_query_chars=4000,
        ),
        mineru=MinerUConfig(
            command="mineru",
            input_path=tmp_path / "manual.pdf",
            output_dir=tmp_path,
            backend="",
            model_source="",
            lang="",
            extra_args="",
            package="mineru",
            version="3.0.9",
            extra="all",
            python="",
            auto_install=False,
        ),
        patching=PatchingConfig(max_new_tokens=8000, llm_timeout=120),
        captioning=CaptioningConfig(max_new_tokens=8000, max_context_tokens=2000, batch_size=32),
        chunking=ChunkingConfig(),
        indexing=IndexingConfig(),
    )


def test_answering_messages_use_retrieval_final_output_content():
    content = [
        {"type": "text", "text": "retrieved context"},
        {"type": "image_url", "image_url": {"url": "/tmp/evidence.png"}},
    ]

    messages = answering._answering_messages(query="What is shown?", final_output_content=content)

    user_content = messages[1]["content"]
    assert user_content[1:3] == content
    assert "retrieved material" in user_content[0]["text"]


def test_final_output_content_falls_back_to_context():
    result = SimpleNamespace(final_output=None, context="context only")

    assert answering._final_output_content(result) == [{"type": "text", "text": "context only"}]


def test_answering_benchmark_dry_run_records_retrieval_parameters(tmp_path, monkeypatch, capsys):
    query_set = tmp_path / "queries.jsonl"
    query_set.write_text('{"query_id":"q1","query":"Question?"}\n', encoding="utf-8")
    config = _config(tmp_path)
    monkeypatch.setattr(answering.AppConfig, "from_env", classmethod(lambda cls: config))

    answering.run_answering_benchmark(
        query_set=query_set,
        output_dir=tmp_path / "runs",
        run_id="dry",
        context_cap=16000,
        retrieval_k=120,
        final_top_k=40,
        rrf_k=30,
        min_score_ratio=0.4,
        final_output_images=True,
        enable_thinking=True,
        route_mode="text-visual-bbox",
        visual_bonus="page-bbox",
        visual_weight=1.75,
        max_tokens=6000,
        llm_base_url="http://localhost:8080/v1",
        llm_model="qwen",
        llm_api_key="EMPTY",
        request_timeout=180,
        limit=None,
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert '"retrieval_k": 120' in output
    assert '"final_top_k": 40' in output
    assert '"rrf_k": 30' in output
    assert '"min_score_ratio": 0.4' in output
    assert '"final_output_images": true' in output
