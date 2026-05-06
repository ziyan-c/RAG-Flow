from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")
TestClient = testclient.TestClient

from rag_flow.api import create_app
from rag_flow.config import (
    AppConfig,
    CaptioningConfig,
    ChunkingConfig,
    MinerUConfig,
    ModelConfig,
    PatchingConfig,
    PathsConfig,
    RetrievalConfig,
    ServerConfig,
)
from rag_flow.retrieval import RetrievalResult


def make_config(tmp_path: Path, *, api_key: str = "", max_query_chars: int = 4000) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            base_dir=tmp_path,
            source_name="manual.pdf",
            source_pdf=tmp_path / "manual.pdf",
            content_json=tmp_path / "content.json",
            sectioned_json=tmp_path / "sectioned.json",
            patched_json=tmp_path / "patched.json",
            captioned_json=tmp_path / "captioned.json",
            chunks_json=tmp_path / "chunks.json",
            db_path=tmp_path / "qdrant",
            collection_name="manuals",
        ),
        models=ModelConfig(
            dense_model="dense",
            sparse_model="sparse",
            colpali_model="colpali",
            vlm_model="vlm",
            vlm_model_revision="",
            trusted_remote_code_models=("vlm",),
            llm_base_url="http://localhost:8080/v1",
            llm_api_key="EMPTY",
            llm_model="llm",
            llm_max_tokens=1000,
        ),
        retrieval=RetrievalConfig(
            retrieval_k=10,
            final_top_k=3,
            rrf_k=60,
            visual_weight=1.5,
            quantized_colpali=False,
        ),
        server=ServerConfig(
            host="127.0.0.1",
            port=8000,
            retriever_url="http://127.0.0.1:8000/retrieve",
            retriever_api_key=api_key,
            max_query_chars=max_query_chars,
        ),
        mineru=MinerUConfig(
            command="mineru",
            input_path=tmp_path / "manual.pdf",
            output_dir=tmp_path / "mineru-output",
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
        patching=PatchingConfig(max_new_tokens=8000, llm_timeout=120.0),
        captioning=CaptioningConfig(max_new_tokens=8000, max_context_tokens=10000, batch_size=4),
        chunking=ChunkingConfig(),
    )


def test_retrieve_requires_configured_api_key(tmp_path, monkeypatch):
    from rag_flow import api

    monkeypatch.setattr(api.RetrievalEngine, "load", lambda self: None)
    monkeypatch.setattr(
        api.RetrievalEngine,
        "retrieve",
        lambda self, query: RetrievalResult(hit_page=1, all_hits=[], context="ok"),
    )

    app = create_app(make_config(tmp_path, api_key="secret"))
    with TestClient(app) as client:
        unauthorized = client.post("/retrieve", json={"query": "hello"})
        authorized = client.post(
            "/retrieve",
            json={"query": "hello"},
            headers={"Authorization": "Bearer secret"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json()["context"] == "ok"


def test_retrieve_rejects_overlong_query(tmp_path, monkeypatch):
    from rag_flow import api

    monkeypatch.setattr(api.RetrievalEngine, "load", lambda self: None)
    app = create_app(make_config(tmp_path, max_query_chars=5))

    with TestClient(app) as client:
        response = client.post("/retrieve", json={"query": "too long"})

    assert response.status_code == 422
