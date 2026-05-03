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


def test_patch_max_new_tokens_defaults_to_8000(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)

    config = AppConfig.from_env()

    assert config.patching.max_new_tokens == 8000
    assert config.patching.llm_timeout == 120.0
    assert config.patching.batch_size == 140
    assert config.patching.concurrency == 10
    assert config.patching.checkpoint_interval == 30
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
    assert config.captioning.max_context_tokens == 10000
    assert config.captioning.batch_size == 4
    assert config.captioning.concurrency == 1
    assert config.captioning.llm_timeout == 120.0


def test_captioning_reads_local_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".local" / "rag-flow.env"
    env_file.parent.mkdir()
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_CAPTION_MAX_NEW_TOKENS=6000",
                "RAG_FLOW_CAPTION_MAX_CONTEXT_TOKENS=7000",
                "RAG_FLOW_CAPTION_BATCH_SIZE=2",
                "RAG_FLOW_CAPTION_CONCURRENCY=3",
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
    assert config.captioning.batch_size == 2
    assert config.captioning.concurrency == 3
    assert config.captioning.llm_timeout == 45.5
