from __future__ import annotations

import pytest

from rag_flow.model_paths import resolve_model_location


def write_model_dir(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def test_explicit_model_path_resolves_snapshot(tmp_path):
    snapshot = write_model_dir(tmp_path / "manual-model" / "snapshots" / "abc123")

    assert resolve_model_location("owner/model", explicit_path=tmp_path / "manual-model") == str(snapshot)


def test_missing_explicit_model_path_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model_location("owner/model", explicit_path=tmp_path / "missing")


def test_local_model_root_finds_huggingface_cache_layout(tmp_path):
    snapshot = write_model_dir(tmp_path / "models" / "models--owner--model" / "snapshots" / "abc123")

    assert resolve_model_location("owner/model", local_root=tmp_path / "models") == str(snapshot)


def test_local_model_root_finds_repo_basename_layout(tmp_path):
    model_dir = write_model_dir(tmp_path / "models" / "model")

    assert resolve_model_location("owner/model", local_root=tmp_path / "models") == str(model_dir)


def test_missing_local_model_returns_original_id(tmp_path):
    assert resolve_model_location("owner/model", local_root=tmp_path / "models") == "owner/model"
