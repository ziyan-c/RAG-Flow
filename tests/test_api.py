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
from rag_flow.retrieval import RetrievalResult, RetrievedImage


def make_config(
    tmp_path: Path,
    *,
    api_key: str = "",
    max_query_chars: int = 4000,
    final_output_images: bool = False,
) -> AppConfig:
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
            final_output_images=final_output_images,
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


def test_retrieve_returns_selected_image_references(tmp_path, monkeypatch):
    from rag_flow import api

    image = RetrievedImage(
        hit_rank=1,
        chunk_id="DSS/manual.pdf::manual-chunk-00001",
        source_relpath="DSS/manual.pdf",
        img_path="images/login.png",
        image_path=str(tmp_path / "images" / "login.png"),
        image_exists=False,
        page_idx=2,
        page_number=3,
        bbox=[1.0, 2.0, 3.0, 4.0],
        image_answering_policy="image_recommended",
        image_answering_confidence="high",
        image_answering_reason="Visible labels matter.",
        image_caption="Login",
        image_description_vlm="A login screen.",
    )
    monkeypatch.setattr(api.RetrievalEngine, "load", lambda self: None)
    monkeypatch.setattr(
        api.RetrievalEngine,
        "retrieve",
        lambda self, query: RetrievalResult(hit_page=3, all_hits=[], context="ok", images=(image,)),
    )

    app = create_app(make_config(tmp_path))
    with TestClient(app) as client:
        response = client.post("/retrieve", json={"query": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["images"][0]["image_path"] == str(tmp_path / "images" / "login.png")
    assert body["images"][0]["chunk_id"] == "DSS/manual.pdf::manual-chunk-00001"
    assert body["images"][0]["image_answering_policy"] == "image_recommended"
    assert body["images"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert body["final_output"]["mode"] == "context_only"
    assert body["final_output"]["content"] == [{"type": "text", "text": "ok"}]
    assert body["final_output"]["images"] == []


def test_retrieve_final_output_adds_recommended_existing_images_when_enabled(tmp_path, monkeypatch):
    from rag_flow import api

    optional = RetrievedImage(
        hit_rank=1,
        chunk_id="DSS/manual.pdf::manual-chunk-00001",
        source_relpath="DSS/manual.pdf",
        img_path="images/optional.png",
        image_path=str(tmp_path / "images" / "optional.png"),
        image_exists=True,
        page_idx=0,
        page_number=1,
        bbox=[],
        image_answering_policy="image_optional",
    )
    recommended = RetrievedImage(
        hit_rank=1,
        chunk_id="DSS/manual.pdf::manual-chunk-00001",
        source_relpath="DSS/manual.pdf",
        img_path="images/recommended.png",
        image_path=str(tmp_path / "images" / "recommended.png"),
        image_exists=True,
        page_idx=1,
        page_number=2,
        bbox=[],
        image_answering_policy="image_recommended",
    )
    missing_required = RetrievedImage(
        hit_rank=2,
        chunk_id="DSS/manual.pdf::manual-chunk-00002",
        source_relpath="DSS/manual.pdf",
        img_path="images/missing.png",
        image_path=str(tmp_path / "images" / "missing.png"),
        image_exists=False,
        page_idx=2,
        page_number=3,
        bbox=[],
        image_answering_policy="image_required",
    )
    monkeypatch.setattr(api.RetrievalEngine, "load", lambda self: None)
    monkeypatch.setattr(
        api.RetrievalEngine,
        "retrieve",
        lambda self, query: RetrievalResult(
            hit_page=2,
            all_hits=[],
            context="retrieved context",
            images=(optional, recommended, missing_required),
        ),
    )

    app = create_app(make_config(tmp_path, final_output_images=True))
    with TestClient(app) as client:
        response = client.post("/retrieve", json={"query": "hello"})

    assert response.status_code == 200
    final_output = response.json()["final_output"]
    assert final_output["mode"] == "openai_compatible_multimodal"
    assert final_output["content"] == [
        {"type": "text", "text": "retrieved context"},
        {"type": "image_url", "image_url": {"url": str(tmp_path / "images" / "recommended.png")}},
    ]
    assert [image["image_path"] for image in final_output["images"]] == [
        str(tmp_path / "images" / "recommended.png")
    ]
