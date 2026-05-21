from __future__ import annotations

from rag_flow.config import AppConfig, resolve_env_file


def test_resolve_env_file_finds_local_env_from_child_directory(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text("RAG_FLOW_COLLECTION=test\n", encoding="utf-8")
    child = tmp_path / "nested" / "workdir"
    child.mkdir(parents=True)

    monkeypatch.chdir(child)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    assert resolve_env_file() == env_file


def test_resolve_env_file_prefers_explicit_env_var(tmp_path, monkeypatch):
    explicit = tmp_path / "private.env"
    monkeypatch.setenv("RAG_FLOW_ENV_FILE", str(explicit))

    assert resolve_env_file() == str(explicit)


def test_resolve_env_file_uses_runtime_fallback(tmp_path, monkeypatch):
    runtime_root = tmp_path / "runtime"
    env_file = runtime_root / ".local" / "rag-flow.env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("RAG_FLOW_COLLECTION=test\n", encoding="utf-8")
    child = tmp_path / "nested" / "workdir"
    child.mkdir(parents=True)

    monkeypatch.chdir(child)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)
    monkeypatch.setenv("RAG_FLOW_RUNTIME_ROOT", str(runtime_root))

    assert resolve_env_file() == env_file


def test_relative_paths_in_local_env_resolve_from_repo_root(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_SOURCE_PDF=.local/source-documents/manual.pdf",
                "RAG_FLOW_SOURCE_ROOT=.local/source-documents",
                "RAG_FLOW_MINERU_INPUT_PATH=.local/source-documents/mineru-input.pdf",
                "RAG_FLOW_BASE_DIR=runtime/manual",
            ]
        ),
        encoding="utf-8",
    )
    child = tmp_path / "nested" / "workdir"
    child.mkdir(parents=True)

    monkeypatch.chdir(child)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.paths.source_pdf == tmp_path / ".local" / "source-documents" / "manual.pdf"
    assert config.paths.source_root == tmp_path / ".local" / "source-documents"
    assert config.mineru.input_path == tmp_path / ".local" / "source-documents" / "mineru-input.pdf"
    assert config.paths.base_dir == tmp_path / "runtime" / "manual"


def test_relative_paths_imported_from_env_file_still_resolve_from_repo_root(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_SOURCE_PDF=.local/source-documents/manual.pdf",
                "RAG_FLOW_BASE_DIR=runtime/manual",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path / ".local")
    monkeypatch.setenv("RAG_FLOW_ENV_FILE", str(env_file))
    monkeypatch.setenv("RAG_FLOW_SOURCE_PDF", ".local/source-documents/manual.pdf")
    monkeypatch.setenv("RAG_FLOW_BASE_DIR", "runtime/manual")

    config = AppConfig.from_env()

    root = tmp_path.resolve()
    assert config.paths.source_pdf == root / ".local" / "source-documents" / "manual.pdf"
    assert config.paths.base_dir == root / "runtime" / "manual"


def test_default_stage_paths_follow_source_name(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_BASE_DIR=runtime/manual/hybrid_auto",
                "RAG_FLOW_SOURCE_NAME=manual.pdf",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.paths.content_json.name == "manual_content_list.json"
    assert config.paths.sectioned_json.name == "manual_content_list_SECTIONED.json"
    assert config.paths.patched_json.name == "manual_content_list_SECTIONED_PATCHED.json"
    assert config.paths.captioned_json.name == "manual_content_list_SECTIONED_PATCHED_CAPTIONED.json"
    assert config.paths.chunks_json.name == "manual_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED.json"


def test_patch_max_new_tokens_defaults_to_8000(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.patching.max_new_tokens == 8000
    assert config.patching.llm_timeout == 120.0
    assert config.patching.batch_size == 512
    assert config.patching.concurrency == 8
    assert config.patching.checkpoint_interval == 1
    assert config.patching.invalid_retry_limit == 0
    assert config.patching.dpi == 250
    assert config.patching.page_window_size == 200


def test_patching_reads_local_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_PATCH_MAX_NEW_TOKENS=4000",
                "RAG_FLOW_PATCH_LLM_TIMEOUT=30.5",
                "RAG_FLOW_PATCH_BATCH_SIZE=12",
                "RAG_FLOW_PATCH_CONCURRENCY=2",
                "RAG_FLOW_PATCH_CHECKPOINT_INTERVAL=5",
                "RAG_FLOW_PATCH_INVALID_RETRY_LIMIT=4",
                "RAG_FLOW_PATCH_DPI=180",
                "RAG_FLOW_PATCH_PAGE_WINDOW_SIZE=50",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.patching.max_new_tokens == 4000
    assert config.patching.llm_timeout == 30.5
    assert config.patching.batch_size == 12
    assert config.patching.concurrency == 2
    assert config.patching.checkpoint_interval == 5
    assert config.patching.invalid_retry_limit == 4
    assert config.patching.dpi == 180
    assert config.patching.page_window_size == 50


def test_captioning_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.captioning.max_new_tokens == 8000
    assert config.captioning.max_context_tokens == 2000
    assert config.captioning.max_image_side == 2048
    assert config.captioning.batch_size == 32
    assert config.captioning.concurrency == 6
    assert config.captioning.checkpoint_interval == 1
    assert config.captioning.llm_timeout == 120.0


def test_captioning_reads_local_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_CAPTION_MAX_NEW_TOKENS=6000",
                "RAG_FLOW_CAPTION_MAX_CONTEXT_TOKENS=7000",
                "RAG_FLOW_CAPTION_MAX_IMAGE_SIDE=1536",
                "RAG_FLOW_CAPTION_BATCH_SIZE=2",
                "RAG_FLOW_CAPTION_CONCURRENCY=3",
                "RAG_FLOW_CAPTION_CHECKPOINT_INTERVAL=7",
                "RAG_FLOW_CAPTION_LLM_TIMEOUT=45.5",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.captioning.max_new_tokens == 6000
    assert config.captioning.max_context_tokens == 7000
    assert config.captioning.max_image_side == 1536
    assert config.captioning.batch_size == 2
    assert config.captioning.concurrency == 3
    assert config.captioning.checkpoint_interval == 7
    assert config.captioning.llm_timeout == 45.5


def test_chunking_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.chunking.mode == "auto"
    assert config.chunking.max_tokens == 5000
    assert config.chunking.overlap_tokens == 500
    assert config.chunking.min_tokens == 200


def test_chunking_reads_local_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_CHUNK_MODE=token",
                "RAG_FLOW_CHUNK_MAX_TOKENS=900",
                "RAG_FLOW_CHUNK_OVERLAP_TOKENS=100",
                "RAG_FLOW_CHUNK_MIN_TOKENS=80",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.chunking.mode == "token"
    assert config.chunking.max_tokens == 900
    assert config.chunking.overlap_tokens == 100
    assert config.chunking.min_tokens == 80


def test_retrieval_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)
    monkeypatch.delenv("RAG_FLOW_PRESET", raising=False)

    config = AppConfig.from_env()

    assert config.models.colpali_model == "vidore/colpali-v1.3-merged"
    assert config.models.colpali_model_path is None
    assert config.models.colpali_local_model_root.as_posix() == "/root/autodl-tmp/models"
    assert config.retrieval.enable_visual is False
    assert config.retrieval.device == "auto"
    assert config.retrieval.route_mode == "text"
    assert config.retrieval.candidate_mode == "direct"
    assert config.retrieval.quantized_colpali is True
    assert config.retrieval.retrieval_k == 150
    assert config.retrieval.final_top_k == 80
    assert config.retrieval.rrf_k == 10
    assert config.retrieval.visual_weight == 2.5
    assert config.retrieval.max_context_tokens == 10000
    assert config.retrieval.min_score_ratio == 1.0


def test_qdrant_server_env_is_optional(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_QDRANT_URL=http://127.0.0.1:6333",
                "RAG_FLOW_QDRANT_API_KEY=secret",
                "RAG_FLOW_QDRANT_PREFER_GRPC=1",
                "RAG_FLOW_QDRANT_TIMEOUT=12.5",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.qdrant.url == "http://127.0.0.1:6333"
    assert config.qdrant.api_key == "secret"
    assert config.qdrant.prefer_grpc is True
    assert config.qdrant.timeout == 12.5


def test_retrieval_preset_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text("RAG_FLOW_PRESET=visualroute\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)
    monkeypatch.delenv("RAG_FLOW_PRESET", raising=False)

    config = AppConfig.from_env()

    assert config.retrieval.enable_visual is True
    assert config.retrieval.route_mode == "visual-naive"
    assert config.retrieval.candidate_mode == "visual-page-local-naive"
    assert config.retrieval.retrieval_k == 150
    assert config.retrieval.final_top_k == 80
    assert config.retrieval.visual_weight == 2.5
    assert config.retrieval.max_context_tokens == 16000
    assert config.indexing.visual_dpi == 200
    assert config.indexing.visual_batch_size == 8


def test_precise_preset_uses_smaller_context_cap(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text("RAG_FLOW_PRESET=precise\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)
    monkeypatch.delenv("RAG_FLOW_PRESET", raising=False)

    config = AppConfig.from_env()

    assert config.retrieval.enable_visual is False
    assert config.retrieval.route_mode == "text"
    assert config.retrieval.candidate_mode == "direct"
    assert config.retrieval.retrieval_k == 150
    assert config.retrieval.final_top_k == 80
    assert config.retrieval.max_context_tokens == 5000
    assert config.retrieval.min_score_ratio == 1.0


def test_tiny_preset_uses_very_small_context_cap(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text("RAG_FLOW_PRESET=tiny\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)
    monkeypatch.delenv("RAG_FLOW_PRESET", raising=False)

    config = AppConfig.from_env()

    assert config.retrieval.enable_visual is False
    assert config.retrieval.route_mode == "text"
    assert config.retrieval.candidate_mode == "direct"
    assert config.retrieval.retrieval_k == 150
    assert config.retrieval.final_top_k == 80
    assert config.retrieval.max_context_tokens == 3000
    assert config.retrieval.min_score_ratio == 1.0


def test_retrieval_preset_values_can_be_overridden(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_PRESET=enhanced",
                "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS=12000",
                "RAG_FLOW_RETRIEVAL_ENABLE_VISUAL=1",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)
    monkeypatch.delenv("RAG_FLOW_PRESET", raising=False)

    config = AppConfig.from_env()

    assert config.retrieval.max_context_tokens == 12000
    assert config.retrieval.enable_visual is True
    assert config.retrieval.route_mode == "text"


def test_colpali_local_model_paths_read_local_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_COLPALI_MODEL=owner/model",
                "RAG_FLOW_COLPALI_MODEL_PATH=models/colpali",
                "RAG_FLOW_COLPALI_LOCAL_MODEL_ROOT=models",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.models.colpali_model == "owner/model"
    assert config.models.colpali_model_path == tmp_path / "models" / "colpali"
    assert config.models.colpali_local_model_root == tmp_path / "models"


def test_retrieval_reads_visual_mode_from_local_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_RETRIEVAL_ENABLE_VISUAL=0",
                "RAG_FLOW_RETRIEVAL_DEVICE=cpu",
                "RAG_FLOW_QUANTIZED_COLPALI=0",
                "RAG_FLOW_VISUAL_WEIGHT=0.75",
                "RAG_FLOW_RETRIEVAL_CANDIDATE_MODE=direct",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.retrieval.enable_visual is False
    assert config.retrieval.device == "cpu"
    assert config.retrieval.candidate_mode == "direct"
    assert config.retrieval.quantized_colpali is False
    assert config.retrieval.visual_weight == 0.75


def test_indexing_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.indexing.text_batch_size == 256
    assert config.indexing.visual_batch_size == 8
    assert config.indexing.visual_dpi == 200


def test_indexing_reads_local_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_INDEX_TEXT_BATCH_SIZE=128",
                "RAG_FLOW_INDEX_VISUAL_BATCH_SIZE=32",
                "RAG_FLOW_INDEX_VISUAL_DPI=250",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.indexing.text_batch_size == 128
    assert config.indexing.visual_batch_size == 32
    assert config.indexing.visual_dpi == 250
