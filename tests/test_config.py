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
