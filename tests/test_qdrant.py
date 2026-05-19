from __future__ import annotations

import sys
from types import SimpleNamespace

from rag_flow.config import AppConfig
from rag_flow.qdrant import create_qdrant_client


def _clear_qdrant_env(monkeypatch):
    for key in (
        "RAG_FLOW_ENV_FILE",
        "RAG_FLOW_QDRANT_URL",
        "RAG_FLOW_QDRANT_API_KEY",
        "RAG_FLOW_QDRANT_PREFER_GRPC",
        "RAG_FLOW_QDRANT_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_create_qdrant_client_uses_local_path_by_default(tmp_path, monkeypatch):
    calls = []

    class FakeQdrantClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=FakeQdrantClient))
    monkeypatch.chdir(tmp_path)
    _clear_qdrant_env(monkeypatch)

    config = AppConfig.from_env()
    client = create_qdrant_client(config)

    assert client.kwargs == {"path": str(config.paths.db_path)}
    assert calls == [client.kwargs]


def test_create_qdrant_client_uses_server_url_when_configured(tmp_path, monkeypatch):
    calls = []

    class FakeQdrantClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)

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

    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=FakeQdrantClient))
    monkeypatch.chdir(tmp_path)
    _clear_qdrant_env(monkeypatch)

    config = AppConfig.from_env()
    client = create_qdrant_client(config)

    assert client.kwargs == {
        "url": "http://127.0.0.1:6333",
        "api_key": "secret",
        "prefer_grpc": True,
        "timeout": 12.5,
    }
    assert calls == [client.kwargs]
