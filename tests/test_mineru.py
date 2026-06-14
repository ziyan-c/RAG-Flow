from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from rag_flow.config import (
    AppConfig,
    CaptioningConfig,
    MinerUConfig,
    ModelConfig,
    PatchingConfig,
    ChunkingConfig,
    IndexingConfig,
    TaggingConfig,
    PathsConfig,
    RetrievalConfig,
    ServerConfig,
)
from rag_flow.mineru import (
    build_mineru_command,
    expected_content_json,
    find_content_json,
    infer_artifacts,
    iter_input_pdfs,
    mineru_batch_items,
    mineru_install_spec,
    mineru_output_dir_for_pdf,
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
            tagged_json=base_dir / "manual_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED_TAGGED.json",
            db_path=tmp_path / "qdrant",
            collection_name="manuals",
        ),
        models=ModelConfig(
            dense_model="dense",
            dense_vector_size=1024,
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


def test_single_pdf_artifacts_preserve_source_root_subfolders(tmp_path):
    docs = tmp_path / "pdfs" / "source"
    output = tmp_path / "pdfs" / "output"
    pdf = docs / "DSS Professional 8.7" / "manual.pdf"
    base_config = make_config(tmp_path, input_path=docs)
    config = replace(
        base_config,
        mineru=replace(base_config.mineru, output_dir=output),
    )

    assert mineru_output_dir_for_pdf(config, source_pdf=pdf) == output / "DSS Professional 8.7"
    assert expected_content_json(config, source_pdf=pdf) == (
        output / "DSS Professional 8.7" / "manual" / "auto" / "manual_content_list.json"
    )


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
    assert artifacts.tagged_json == (
        content_json.parent / "manual_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED_TAGGED.json"
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


def test_run_ingest_default_runs_through_indexing_without_tagging(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.paths.base_dir.mkdir(parents=True)
    config.paths.content_json.write_text("[]", encoding="utf-8")
    calls = []

    @contextmanager
    def fake_managed_model_server(config_arg, kind):
        calls.append(("server-enter", kind))
        yield
        calls.append(("server-exit", kind))

    class FakeSectioningResult:
        stats = {"section_event_count": 0}

    def fake_section(**kwargs):
        calls.append(("sectioning", Path(kwargs["output_json"])))
        Path(kwargs["output_json"]).write_text("[]", encoding="utf-8")
        Path(kwargs["audit_json"]).write_text("{}", encoding="utf-8")
        return FakeSectioningResult()

    def fake_patch(**kwargs):
        calls.append(("patching", Path(kwargs["output_json"])))
        Path(kwargs["output_json"]).write_text("[]", encoding="utf-8")

    def fake_caption(**kwargs):
        calls.append(("captioning", Path(kwargs["output_json"])))
        Path(kwargs["output_json"]).write_text("[]", encoding="utf-8")

    def fake_create_chunks(*args, **kwargs):
        calls.append(("chunking",))
        return [{"chunk_content": "hello", "metadata": {"source_relpath": "manual.pdf", "page_idx": 0}}]

    def fake_write_chunks(chunks, output_path):
        Path(output_path).write_text(json.dumps(chunks), encoding="utf-8")

    def fake_tag(**kwargs):
        calls.append(("tagging", Path(kwargs["output_json"])))
        Path(kwargs["output_json"]).write_text(Path(kwargs["chunks_json"]).read_text(encoding="utf-8"), encoding="utf-8")
        return SimpleNamespace(metadata_yaml=None, chunk_count=1, tagged_count=0, missing_metadata_count=0)

    def fake_index(config_arg, chunks_path, *, batch_size):
        calls.append(("indexing", Path(chunks_path), batch_size))

    monkeypatch.setattr("rag_flow.pipeline.managed_model_server", fake_managed_model_server)
    monkeypatch.setattr("rag_flow.pipeline.write_sectioned_json", fake_section)
    monkeypatch.setattr("rag_flow.pipeline.add_small_icon_text", fake_patch)
    monkeypatch.setattr("rag_flow.pipeline.add_image_descriptions", fake_caption)
    monkeypatch.setattr("rag_flow.pipeline.create_chunks", fake_create_chunks)
    monkeypatch.setattr("rag_flow.pipeline.write_chunks", fake_write_chunks)
    monkeypatch.setattr("rag_flow.pipeline.write_tagged_chunks", fake_tag)
    monkeypatch.setattr("rag_flow.pipeline.upsert_text_vectors", fake_index)

    artifacts = run_ingest(config)

    assert artifacts.chunks_json.exists()
    assert not artifacts.tagged_json.exists()
    assert [call[0] for call in calls] == [
        "sectioning",
        "server-enter",
        "patching",
        "captioning",
        "server-exit",
        "chunking",
        "indexing",
    ]
    assert calls[-1] == ("indexing", artifacts.chunks_json, config.indexing.text_batch_size)


def test_run_ingest_enabled_tagging_runs_before_indexing(tmp_path, monkeypatch):
    config = replace(make_config(tmp_path), tagging=TaggingConfig(enabled=True))
    config.paths.base_dir.mkdir(parents=True)
    config.paths.content_json.write_text("[]", encoding="utf-8")
    calls = []

    @contextmanager
    def fake_managed_model_server(config_arg, kind):
        calls.append(("server-enter", kind))
        yield
        calls.append(("server-exit", kind))

    class FakeSectioningResult:
        stats = {"section_event_count": 0}

    def fake_section(**kwargs):
        calls.append(("sectioning", Path(kwargs["output_json"])))
        Path(kwargs["output_json"]).write_text("[]", encoding="utf-8")
        Path(kwargs["audit_json"]).write_text("{}", encoding="utf-8")
        return FakeSectioningResult()

    def fake_patch(**kwargs):
        calls.append(("patching", Path(kwargs["output_json"])))
        Path(kwargs["output_json"]).write_text("[]", encoding="utf-8")

    def fake_caption(**kwargs):
        calls.append(("captioning", Path(kwargs["output_json"])))
        Path(kwargs["output_json"]).write_text("[]", encoding="utf-8")

    def fake_create_chunks(*args, **kwargs):
        calls.append(("chunking",))
        return [{"chunk_content": "hello", "metadata": {"source_relpath": "manual.pdf", "page_idx": 0}}]

    def fake_write_chunks(chunks, output_path):
        Path(output_path).write_text(json.dumps(chunks), encoding="utf-8")

    def fake_tag(**kwargs):
        calls.append(("tagging", Path(kwargs["output_json"]), kwargs["require_metadata"]))
        Path(kwargs["output_json"]).write_text(Path(kwargs["chunks_json"]).read_text(encoding="utf-8"), encoding="utf-8")
        return SimpleNamespace(metadata_yaml=tmp_path / "source" / "metadata.yml", chunk_count=1, tagged_count=1, missing_metadata_count=0)

    def fake_index(config_arg, chunks_path, *, batch_size):
        calls.append(("indexing", Path(chunks_path), batch_size))

    monkeypatch.setattr("rag_flow.pipeline.managed_model_server", fake_managed_model_server)
    monkeypatch.setattr("rag_flow.pipeline.write_sectioned_json", fake_section)
    monkeypatch.setattr("rag_flow.pipeline.add_small_icon_text", fake_patch)
    monkeypatch.setattr("rag_flow.pipeline.add_image_descriptions", fake_caption)
    monkeypatch.setattr("rag_flow.pipeline.create_chunks", fake_create_chunks)
    monkeypatch.setattr("rag_flow.pipeline.write_chunks", fake_write_chunks)
    monkeypatch.setattr("rag_flow.pipeline.write_tagged_chunks", fake_tag)
    monkeypatch.setattr("rag_flow.pipeline.upsert_text_vectors", fake_index)

    artifacts = run_ingest(config)

    assert artifacts.chunks_json.exists()
    assert artifacts.tagged_json.exists()
    assert [call[0] for call in calls] == [
        "sectioning",
        "server-enter",
        "patching",
        "captioning",
        "server-exit",
        "chunking",
        "tagging",
        "indexing",
    ]
    assert calls[-2] == ("tagging", artifacts.tagged_json, True)
    assert calls[-1] == ("indexing", artifacts.tagged_json, config.indexing.text_batch_size)


def test_run_ingest_rejects_explicit_tagging_when_disabled(tmp_path):
    config = make_config(tmp_path)
    config.paths.base_dir.mkdir(parents=True)
    config.paths.content_json.write_text("[]", encoding="utf-8")

    try:
        run_ingest(
            config,
            from_stage="tagging",
            to_stage="tagging",
            skip_existing=False,
        )
    except ValueError as exc:
        assert "RAG_FLOW_TAGGING_ENABLED=1" in str(exc)
    else:
        raise AssertionError("Expected disabled explicit tagging stage to fail")


def test_run_ingest_uses_pdf_override_for_chunk_source(tmp_path):
    config = make_config(tmp_path)
    source_pdf = tmp_path / "source" / "other.pdf"
    content_json = tmp_path / "mineru-output" / "other" / "auto" / "other_content_list.json"
    captioned_json = content_json.parent / "other_content_list_SECTIONED_PATCHED_CAPTIONED.json"
    captioned_json.parent.mkdir(parents=True)
    content_json.write_text("[]", encoding="utf-8")
    captioned_json.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "page_idx": 0,
                    "text": "hello",
                    "source_relpath": "other.pdf",
                    "source_filename": "other.pdf",
                    "breadcrumb": "other.pdf",
                }
            ]
        ),
        encoding="utf-8",
    )

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
    docs = tmp_path / ".local/CUSTOM_DATA" / "pdfs" / "source"
    config = make_config(tmp_path, input_path=docs)
    source_pdf = docs / "DSS" / "manual.pdf"
    content_json = tmp_path / "mineru-output" / "DSS" / "manual" / "auto" / "manual_content_list.json"
    captioned_json = content_json.parent / "manual_content_list_SECTIONED_PATCHED_CAPTIONED.json"
    captioned_json.parent.mkdir(parents=True)
    content_json.write_text("[]", encoding="utf-8")
    captioned_json.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "page_idx": 0,
                    "text": "hello",
                    "source_relpath": "DSS/manual.pdf",
                    "source_filename": "manual.pdf",
                    "breadcrumb": "DSS > manual.pdf",
                }
            ]
        ),
        encoding="utf-8",
    )

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
    captioned_json.write_text(
        json.dumps(
            [
                {
                    "type": "text",
                    "page_idx": 0,
                    "text": "hello",
                    "source_relpath": "DSS/manual.pdf",
                    "source_filename": "manual.pdf",
                    "breadcrumb": "DSS > manual.pdf",
                }
            ]
        ),
        encoding="utf-8",
    )

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


def test_run_ingest_preprocessing_uses_vlm_model(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config = replace(
        config,
        models=replace(
            config.models,
            vlm_base_url="http://vlm.local/v1",
            vlm_api_key="vlm-key",
            llm_base_url="http://llm.local/v1",
            llm_api_key="llm-key",
        ),
    )
    config.paths.base_dir.mkdir(parents=True)
    config.paths.content_json.write_text("[]", encoding="utf-8")

    calls = []

    def fake_patch(**kwargs):
        calls.append(("patch", kwargs["llm_base_url"], kwargs["llm_api_key"], kwargs["llm_model"]))
        Path(kwargs["output_json"]).write_text("[]", encoding="utf-8")

    def fake_caption(**kwargs):
        calls.append(("caption", kwargs["llm_base_url"], kwargs["llm_api_key"], kwargs["model_name"]))
        Path(kwargs["output_json"]).write_text("[]", encoding="utf-8")

    monkeypatch.setattr("rag_flow.pipeline.add_small_icon_text", fake_patch)
    monkeypatch.setattr("rag_flow.pipeline.add_image_descriptions", fake_caption)

    run_ingest(
        config,
        from_stage="patching",
        to_stage="captioning",
        skip_existing=False,
    )

    assert config.models.vlm_model != config.models.llm_model
    assert calls == [
        ("patch", "http://vlm.local/v1", "vlm-key", "vlm"),
        ("caption", "http://vlm.local/v1", "vlm-key", "vlm"),
    ]


def test_run_ingest_starts_vlm_only_around_preprocessing(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.paths.base_dir.mkdir(parents=True)
    config.paths.content_json.write_text("[]", encoding="utf-8")
    events = []

    @contextmanager
    def fake_managed_model_server(config_arg, kind):
        events.append(("enter", kind, config_arg.models.vlm_model))
        yield
        events.append(("exit", kind, config_arg.models.vlm_model))

    def fake_patch(**kwargs):
        events.append(("patch", kwargs["llm_model"]))
        Path(kwargs["output_json"]).write_text("[]", encoding="utf-8")

    def fake_caption(**kwargs):
        events.append(("caption", kwargs["model_name"]))
        Path(kwargs["output_json"]).write_text("[]", encoding="utf-8")

    monkeypatch.setattr("rag_flow.pipeline.managed_model_server", fake_managed_model_server)
    monkeypatch.setattr("rag_flow.pipeline.add_small_icon_text", fake_patch)
    monkeypatch.setattr("rag_flow.pipeline.add_image_descriptions", fake_caption)

    run_ingest(
        config,
        from_stage="patching",
        to_stage="captioning",
        skip_existing=False,
    )

    assert events == [
        ("enter", "vlm", "vlm"),
        ("patch", "vlm"),
        ("caption", "vlm"),
        ("exit", "vlm", "vlm"),
    ]


def test_run_ingest_indexing_both_mode_passes_current_chunks_to_visual(tmp_path, monkeypatch):
    config = replace(
        make_config(tmp_path),
        indexing=IndexingConfig(mode="both", text_batch_size=11, visual_batch_size=12, visual_dpi=220),
    )
    config.paths.base_dir.mkdir(parents=True)
    config.paths.content_json.write_text("[]", encoding="utf-8")

    calls = []

    def fake_text(config_arg, chunks_path, *, batch_size):
        calls.append(("text", Path(chunks_path), batch_size))

    def fake_visual(config_arg, **kwargs):
        calls.append(("visual", Path(kwargs["chunks_path"]), kwargs["batch_size"], kwargs["dpi"]))

    monkeypatch.setattr("rag_flow.pipeline.upsert_text_vectors", fake_text)
    monkeypatch.setattr("rag_flow.pipeline.upsert_colpali_vectors", fake_visual)

    artifacts = run_ingest(
        config,
        from_stage="indexing",
        to_stage="indexing",
        skip_existing=False,
    )

    assert calls == [
        ("text", artifacts.chunks_json, 11),
        ("visual", artifacts.chunks_json, 12, 220),
    ]


def test_run_ingest_enabled_tagging_indexing_both_uses_tagged_chunks(tmp_path, monkeypatch):
    config = replace(
        make_config(tmp_path),
        indexing=IndexingConfig(mode="both", text_batch_size=11, visual_batch_size=12, visual_dpi=220),
        tagging=TaggingConfig(enabled=True),
    )
    config.paths.base_dir.mkdir(parents=True)
    config.paths.content_json.write_text("[]", encoding="utf-8")

    calls = []

    def fake_text(config_arg, chunks_path, *, batch_size):
        calls.append(("text", Path(chunks_path), batch_size))

    def fake_visual(config_arg, **kwargs):
        calls.append(("visual", Path(kwargs["chunks_path"]), kwargs["batch_size"], kwargs["dpi"]))

    monkeypatch.setattr("rag_flow.pipeline.upsert_text_vectors", fake_text)
    monkeypatch.setattr("rag_flow.pipeline.upsert_colpali_vectors", fake_visual)

    artifacts = run_ingest(
        config,
        from_stage="indexing",
        to_stage="indexing",
        skip_existing=False,
    )

    assert calls == [
        ("text", artifacts.tagged_json, 11),
        ("visual", artifacts.tagged_json, 12, 220),
    ]


def test_run_ingest_rejects_visual_only_index_mode(tmp_path):
    config = replace(make_config(tmp_path), indexing=IndexingConfig(mode="visual"))
    config.paths.base_dir.mkdir(parents=True)
    config.paths.content_json.write_text("[]", encoding="utf-8")

    try:
        run_ingest(
            config,
            from_stage="indexing",
            to_stage="indexing",
            skip_existing=False,
        )
    except ValueError as exc:
        assert "RAG_FLOW_INDEX_MODE must be one of: text, both" in str(exc)
    else:
        raise AssertionError("visual-only index mode should be rejected")
