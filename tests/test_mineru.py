from __future__ import annotations

import json
from pathlib import Path

from rag_flow.config import (
    AppConfig,
    CaptioningConfig,
    MinerUConfig,
    ModelConfig,
    PatchingConfig,
    ChunkingConfig,
    PathsConfig,
    RetrievalConfig,
    ServerConfig,
)
from rag_flow.mineru import (
    build_mineru_command,
    find_content_json,
    infer_artifacts,
    iter_input_pdfs,
    mineru_batch_items,
    mineru_install_spec,
    run_mineru_batch,
)
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
            sectioned_json=base_dir / "manual_content_list_SECTIONED.json",
            patched_json=base_dir / "manual_content_list_SECTIONED_PATCHED.json",
            captioned_json=base_dir / "manual_content_list_SECTIONED_PATCHED_CAPTIONED.json",
            chunks_json=base_dir / "manual_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED.json",
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
        patching=PatchingConfig(max_new_tokens=8000, llm_timeout=120.0),
        captioning=CaptioningConfig(max_new_tokens=8000, max_context_tokens=10000, batch_size=4),
        chunking=ChunkingConfig(),
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


def test_iter_input_pdfs_recurses_and_sorts_case_insensitive(tmp_path):
    docs = tmp_path / "docs"
    (docs / "nested").mkdir(parents=True)
    (docs / "b.PDF").write_text("b", encoding="utf-8")
    (docs / "nested" / "a.pdf").write_text("a", encoding="utf-8")
    (docs / "note.txt").write_text("ignore", encoding="utf-8")

    assert [path.relative_to(docs) for path in iter_input_pdfs(docs)] == [
        Path("b.PDF"),
        Path("nested/a.pdf"),
    ]


def test_iter_input_pdfs_can_skip_nested_directories(tmp_path):
    docs = tmp_path / "docs"
    (docs / "nested").mkdir(parents=True)
    (docs / "root.pdf").write_text("root", encoding="utf-8")
    (docs / "nested" / "nested.pdf").write_text("nested", encoding="utf-8")

    assert [path.relative_to(docs) for path in iter_input_pdfs(docs, recursive=False)] == [
        Path("root.pdf"),
    ]


def test_mineru_batch_items_mirror_input_tree(tmp_path):
    docs = tmp_path / "docs"
    output = tmp_path / "mineru-output"
    (docs / "product" / "network").mkdir(parents=True)
    (docs / "quickstart.pdf").write_text("quickstart", encoding="utf-8")
    (docs / "product" / "network" / "admin.pdf").write_text("admin", encoding="utf-8")

    items = mineru_batch_items(docs, output)

    assert [(item.input_pdf.relative_to(docs), item.output_dir.relative_to(output)) for item in items] == [
        (Path("product/network/admin.pdf"), Path("product/network")),
        (Path("quickstart.pdf"), Path(".")),
    ]


def test_run_mineru_batch_dry_run_uses_mirrored_output_dirs(tmp_path, monkeypatch, capsys):
    config = make_config(tmp_path)
    docs = tmp_path / "docs"
    output = tmp_path / "mineru-output"
    (docs / "nested").mkdir(parents=True)
    (docs / "manual.pdf").write_text("manual", encoding="utf-8")
    (docs / "nested" / "guide.pdf").write_text("guide", encoding="utf-8")

    monkeypatch.setattr(
        "rag_flow.mineru.mineru_status",
        lambda _config: type(
            "Status",
            (),
            {"command_path": "/envs/mineru/bin/mineru", "command": "mineru"},
        )(),
    )

    commands = run_mineru_batch(config, input_path=docs, output_dir=output, dry_run=True)

    assert [command[2] for command in commands] == [
        str(docs / "manual.pdf"),
        str(docs / "nested" / "guide.pdf"),
    ]
    assert [command[4] for command in commands] == [
        str(output),
        str(output / "nested"),
    ]
    assert not output.exists()
    printed = capsys.readouterr().out
    assert f"MinerU output dir: {output / 'nested'}" in printed


def test_infer_artifacts_from_discovered_content_list(tmp_path):
    config = make_config(tmp_path)
    content_json = tmp_path / "mineru-output" / "manual" / "auto" / "manual_content_list.json"
    content_json.parent.mkdir(parents=True)
    content_json.write_text("[]", encoding="utf-8")

    artifacts = infer_artifacts(config)

    assert artifacts.base_dir == content_json.parent
    assert artifacts.content_json == content_json
    assert artifacts.sectioned_json == content_json.parent / "manual_content_list_SECTIONED.json"
    assert artifacts.sectioning_audit_json == content_json.parent / "manual_SECTIONING_AUDIT.json"
    assert artifacts.patched_json == content_json.parent / "manual_content_list_SECTIONED_PATCHED.json"
    assert artifacts.captioned_json == (
        content_json.parent / "manual_content_list_SECTIONED_PATCHED_CAPTIONED.json"
    )
    assert artifacts.chunks_json == (
        content_json.parent / "manual_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED.json"
    )


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


def test_run_ingest_sectioning_stage_recovers_pdf_outline(tmp_path):
    fitz = __import__("fitz")
    config = make_config(tmp_path)
    config.paths.source_pdf.parent.mkdir(parents=True)
    config.paths.base_dir.mkdir(parents=True)

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1 Overview")
    page.insert_text((72, 120), "Body")
    doc.set_toc([[1, "1 Overview", 1]])
    doc.save(config.paths.source_pdf)
    doc.close()

    config.paths.content_json.write_text(
        json.dumps(
            [
                {"type": "text", "page_idx": 0, "text": "1 Overview", "bbox": [100, 100, 300, 130]},
                {"type": "text", "page_idx": 0, "text": "Body"},
            ]
        ),
        encoding="utf-8",
    )

    artifacts = run_ingest(
        config,
        from_stage="sectioning",
        to_stage="sectioning",
        skip_existing=False,
    )

    sectioned = json.loads(artifacts.sectioned_json.read_text(encoding="utf-8"))
    audit = json.loads(artifacts.sectioning_audit_json.read_text(encoding="utf-8"))
    assert sectioned[1]["section_path"] == ["1 Overview"]
    assert sectioned[1]["source_relpath"] == "manual.pdf"
    assert sectioned[1]["source_filename"] == "manual.pdf"
    assert sectioned[1]["breadcrumb"] == "manual.pdf > 1 Overview"
    assert audit["source_name"] == "manual.pdf"
    assert audit["stats"]["section_event_count"] == 1


def test_run_ingest_uses_pdf_override_for_chunk_source(tmp_path):
    config = make_config(tmp_path)
    source_pdf = tmp_path / "source" / "other.pdf"
    content_json = tmp_path / "mineru-output" / "other" / "auto" / "other_content_list.json"
    captioned_json = content_json.parent / "other_content_list_SECTIONED_PATCHED_CAPTIONED.json"
    captioned_json.parent.mkdir(parents=True)
    content_json.write_text("[]", encoding="utf-8")
    captioned_json.write_text(json.dumps([{"type": "text", "page_idx": 0, "text": "hello"}]), encoding="utf-8")

    artifacts = run_ingest(
        config,
        pdf_path=source_pdf,
        from_stage="chunking",
        to_stage="chunking",
        skip_existing=False,
    )

    chunks = json.loads(artifacts.chunks_json.read_text(encoding="utf-8"))
    assert chunks[0]["metadata"]["source_relpath"] == "other.pdf"
    assert "source" not in chunks[0]["metadata"]


def test_run_ingest_uses_source_root_relative_chunk_source(tmp_path):
    docs = tmp_path / "source-pdfs"
    config = make_config(tmp_path, input_path=docs)
    source_pdf = docs / "DSS" / "manual.pdf"
    content_json = tmp_path / "mineru-output" / "DSS" / "manual" / "auto" / "manual_content_list.json"
    captioned_json = content_json.parent / "manual_content_list_SECTIONED_PATCHED_CAPTIONED.json"
    captioned_json.parent.mkdir(parents=True)
    content_json.write_text("[]", encoding="utf-8")
    captioned_json.write_text(json.dumps([{"type": "text", "page_idx": 0, "text": "hello"}]), encoding="utf-8")

    artifacts = run_ingest(
        config,
        pdf_path=source_pdf,
        from_stage="chunking",
        to_stage="chunking",
        skip_existing=False,
    )

    chunks = json.loads(artifacts.chunks_json.read_text(encoding="utf-8"))
    assert chunks[0]["metadata"]["source_relpath"] == "DSS/manual.pdf"
    assert chunks[0]["metadata"]["source_filename"] == "manual.pdf"
    assert "source" not in chunks[0]["metadata"]


def test_run_ingest_uses_manual_source_root_override(tmp_path):
    config = make_config(tmp_path)
    source_root = tmp_path / "pdfs"
    source_pdf = source_root / "DSS" / "manual.pdf"
    content_json = tmp_path / "mineru-output" / "DSS" / "manual" / "auto" / "manual_content_list.json"
    captioned_json = content_json.parent / "manual_content_list_SECTIONED_PATCHED_CAPTIONED.json"
    captioned_json.parent.mkdir(parents=True)
    content_json.write_text("[]", encoding="utf-8")
    captioned_json.write_text(json.dumps([{"type": "text", "page_idx": 0, "text": "hello"}]), encoding="utf-8")

    artifacts = run_ingest(
        config,
        pdf_path=source_pdf,
        source_root=source_root,
        from_stage="chunking",
        to_stage="chunking",
        skip_existing=False,
    )

    chunks = json.loads(artifacts.chunks_json.read_text(encoding="utf-8"))
    assert chunks[0]["metadata"]["source_relpath"] == "DSS/manual.pdf"
    assert chunks[0]["metadata"]["breadcrumb"] == "DSS > manual.pdf"
