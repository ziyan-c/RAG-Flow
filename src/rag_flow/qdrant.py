from __future__ import annotations

from typing import Any

from .config import AppConfig


def create_qdrant_client(config: AppConfig) -> Any:
    from qdrant_client import QdrantClient

    if config.qdrant.url.strip():
        return QdrantClient(
            url=config.qdrant.url,
            api_key=config.qdrant.api_key or None,
            prefer_grpc=config.qdrant.prefer_grpc,
            timeout=config.qdrant.timeout,
        )
    return QdrantClient(path=str(config.paths.db_path))
