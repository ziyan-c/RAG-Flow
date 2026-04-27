from __future__ import annotations

import json
from pathlib import Path

from rag_flow.config import AppConfig, MinerUConfig, ModelConfig, PathsConfig, RetrievalConfig, ServerConfig
from rag_flow.mineru import build_mineru_command, find_content_json, infer_artifacts, mineru_install_spec
from rag_flow.pipeline import run_ingest


def make_config(tmp_path: Path, *, command: str = "mineru", input_path: Path | None = None) -> AppConfig:
    base_dir = tmp_path / "manual" / "hybrid_auto"
    source_pdf = tmp_path / "source" / "manual.pdf"
    return AppConfig(
        paths=PathsConfig(
            base_dir=base_dir,
            source_name="manual.pdf",
            source_pdf=source_pdf,
            content_json=base_dir / "manual_content_list.json",
            small_icon_json=base_dir / "manual_content_list_small-icon-fixed.json",
            captioned_json=base_dir / "manual_content_list_small-icon-fixed_image-with-captions.json",
            chunks_json=base_dir / "manual_page_level_chunks.json",
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
            retriever_api_key="",
            max_query_chars=4000,
        ),
        mineru=MinerUConfig(
            command=command,
            input_path=input_path or source_pdf,
            output_dir=tmp_path / "mineru-output",
            backend="pipeline",
            model_source="modelscope",
            lang="en",
            extra_args="--debug false",
            package="mineru",
            version="3.0.9",
            extra="all",
            python="/envs/mineru/bin/python",
            auto_install=False,
        ),
    )


def test_mineru_install_spec_pins_package_and_extra(tmp_path):
    config = make_config(tmp_path)

    assert mineru_install_spec(config) == "mineru[all]==3.0.9"


def test_build_mineru_command_uses_default_cli_flags(tmp_path):
    config = make_config(tmp_path)

    assert build_mineru_command(config) == [
        "mineru",
        "-p",
        str(config.paths.source_pdf),
        "-o",
        str(config.mineru.output_dir),
        "-b",
        "pipeline",
        "-l",
        "en",
        "--debug",
        "false",
    ]


def test_build_mineru_command_uses_configured_input_path(tmp_path):
    input_path = tmp_path / "inputs" / "configured.pdf"
    config = make_config(tmp_path, input_path=input_path)

    assert build_mineru_command(config)[2] == str(input_path)


def test_build_mineru_command_supports_templates(tmp_path):
    config = make_config(tmp_path, command="mineru-wrapper -p {input_path} -o {output_dir} -b {backend}")

    assert build_mineru_command(config) == [
        "mineru-wrapper",
        "-p",
        str(config.paths.source_pdf),
        "-o",
        str(config.mineru.output_dir),
        "-b",
        "pipeline",
    ]


def test_infer_artifacts_from_discovered_content_list(tmp_path):
    config = make_config(tmp_path)
    content_json = tmp_path / "mineru-output" / "manual" / "auto" / "manual_content_list.json"
    content_json.parent.mkdir(parents=True)
    content_json.write_text("[]", encoding="utf-8")

    artifacts = infer_artifacts(config)

    assert artifacts.base_dir == content_json.parent
    assert artifacts.content_json == content_json
    assert artifacts.small_icon_json == content_json.parent / "manual_content_list_small-icon-fixed.json"
    assert artifacts.captioned_json == (
        content_json.parent / "manual_content_list_small-icon-fixed_image-with-captions.json"
    )
    assert artifacts.chunks_json == content_json.parent / "manual_page_level_chunks.json"


def test_find_content_json_ignores_other_pdf_outputs(tmp_path):
    config = make_config(tmp_path)
    old_content_json = tmp_path / "mineru-output" / "old-manual" / "auto" / "old-manual_content_list.json"
    old_content_json.parent.mkdir(parents=True)
    old_content_json.write_text("[]", encoding="utf-8")

    assert find_content_json(config, source_pdf=config.paths.source_pdf) is None


def test_find_content_json_matches_current_pdf_stem(tmp_path):
    config = make_config(tmp_path)
    content_json = tmp_path / "mineru-output" / "manual" / "auto" / "manual_content_list.json"
    content_json.parent.mkdir(parents=True)
    content_json.write_text("[]", encoding="utf-8")

    assert find_content_json(config, source_pdf=config.paths.source_pdf) == content_json


def test_run_ingest_uses_pdf_override_for_chunk_source(tmp_path):
    config = make_config(tmp_path)
    source_pdf = tmp_path / "source" / "other.pdf"
    content_json = tmp_path / "mineru-output" / "other" / "auto" / "other_content_list.json"
    captioned_json = content_json.parent / "other_content_list_small-icon-fixed_image-with-captions.json"
    captioned_json.parent.mkdir(parents=True)
    content_json.write_text("[]", encoding="utf-8")
    captioned_json.write_text(json.dumps([{"type": "text", "page_idx": 0, "text": "hello"}]), encoding="utf-8")

    artifacts = run_ingest(
        config,
        pdf_path=source_pdf,
        from_stage="chunks",
        to_stage="chunks",
        skip_existing=False,
    )

    chunks = json.loads(artifacts.chunks_json.read_text(encoding="utf-8"))
    assert chunks[0]["metadata"]["source"] == "other.pdf"
