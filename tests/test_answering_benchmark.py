from __future__ import annotations

from contextlib import contextmanager
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
        tagged_json=tmp_path / "tagged.json",
        db_path=tmp_path / "db",
        collection_name="manuals",
    )
    models = ModelConfig(
        dense_model="dense",
        dense_vector_size=1024,
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


def test_answering_benchmark_skip_answering_does_not_start_llm(tmp_path, monkeypatch):
    query_set = tmp_path / "queries.jsonl"
    query_set.write_text('{"query_id":"q1","query":"Question?"}\n', encoding="utf-8")
    config = _config(tmp_path)
    monkeypatch.setattr(answering.AppConfig, "from_env", classmethod(lambda cls: config))

    class FakeEngine:
        def __init__(self, config_arg):
            self.config_arg = config_arg

        def load(self):
            return None

        def retrieve(self, query):
            return SimpleNamespace(hit_page=1, all_hits=[], context="retrieved context", final_output=None)

    def fail_managed_model_server(*args, **kwargs):
        raise AssertionError("LLM server should not start when answering is skipped")

    monkeypatch.setattr(answering, "RetrievalEngine", FakeEngine)
    monkeypatch.setattr(answering, "managed_model_server", fail_managed_model_server)

    run_dir = answering.run_answering_benchmark(
        query_set=query_set,
        output_dir=tmp_path / "runs",
        run_id="skip",
        context_cap=10000,
        retrieval_k=10,
        final_top_k=3,
        rrf_k=10,
        min_score_ratio=1.0,
        final_output_images=False,
        enable_thinking=False,
        route_mode="text",
        visual_bonus="none",
        visual_weight=1.0,
        max_tokens=4000,
        llm_base_url="http://localhost:8080/v1",
        llm_model="qwen",
        llm_api_key="EMPTY",
        request_timeout=180,
        limit=None,
        dry_run=False,
        skip_answering=True,
    )

    assert (run_dir / "summary.json").exists()


def test_answering_benchmark_starts_llm_for_answers(tmp_path, monkeypatch):
    query_set = tmp_path / "queries.jsonl"
    query_set.write_text('{"query_id":"q1","query":"Question?"}\n', encoding="utf-8")
    config = _config(tmp_path)
    monkeypatch.setattr(answering.AppConfig, "from_env", classmethod(lambda cls: config))
    events = []

    class FakeEngine:
        def __init__(self, config_arg):
            self.config_arg = config_arg

        def load(self):
            return None

        def retrieve(self, query):
            return SimpleNamespace(hit_page=1, all_hits=[], context="retrieved context", final_output=None)

    @contextmanager
    def fake_managed_model_server(config_arg, kind, **kwargs):
        events.append(("enter", kind, kwargs["model"]))
        yield
        events.append(("exit", kind, kwargs["model"]))

    def fake_call_answering_llm(**kwargs):
        return "answer", "", {"total_tokens": 12}, {"id": "response"}

    monkeypatch.setattr(answering, "RetrievalEngine", FakeEngine)
    monkeypatch.setattr(answering, "managed_model_server", fake_managed_model_server)
    monkeypatch.setattr(answering, "_call_answering_llm", fake_call_answering_llm)

    answering.run_answering_benchmark(
        query_set=query_set,
        output_dir=tmp_path / "runs",
        run_id="answer",
        context_cap=10000,
        retrieval_k=10,
        final_top_k=3,
        rrf_k=10,
        min_score_ratio=1.0,
        final_output_images=False,
        enable_thinking=False,
        route_mode="text",
        visual_bonus="none",
        visual_weight=1.0,
        max_tokens=4000,
        llm_base_url="http://localhost:8080/v1",
        llm_model="qwen",
        llm_api_key="EMPTY",
        request_timeout=180,
        limit=None,
        dry_run=False,
        skip_answering=False,
    )

    assert events == [("enter", "llm", "qwen"), ("exit", "llm", "qwen")]
