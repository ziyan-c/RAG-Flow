from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_BASE_DIR = Path(
    "/root/autodl-tmp/manuals/public/"
    "Dahua-DSS-Professional_User-Manual_V8.7.0/hybrid_auto"
)
DEFAULT_SOURCE_NAME = "Dahua-DSS-Professional_User-Manual_V8.7.0.pdf"


def load_env_file(path: str | os.PathLike[str] | None) -> dict[str, str]:
    """Load a simple KEY=VALUE env file without overriding process env."""
    if not path:
        return {}
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class EnvView:
    def __init__(self, file_values: Mapping[str, str]):
        self.file_values = file_values

    def get(self, key: str, default: str) -> str:
        return os.environ.get(key, self.file_values.get(key, default))

    def int(self, key: str, default: int) -> int:
        return int(self.get(key, str(default)))

    def float(self, key: str, default: float) -> float:
        return float(self.get(key, str(default)))

    def path(self, key: str, default: str | Path) -> Path:
        return Path(self.get(key, str(default))).expanduser()


@dataclass(frozen=True)
class PathsConfig:
    base_dir: Path
    source_name: str
    source_pdf: Path
    content_json: Path
    small_icon_json: Path
    captioned_json: Path
    chunks_json: Path
    db_path: Path
    collection_name: str


@dataclass(frozen=True)
class ModelConfig:
    dense_model: str
    sparse_model: str
    colpali_model: str
    vlm_model: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_max_tokens: int


@dataclass(frozen=True)
class RetrievalConfig:
    retrieval_k: int
    final_top_k: int
    rrf_k: int
    visual_weight: float
    quantized_colpali: bool


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    retriever_url: str


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    models: ModelConfig
    retrieval: RetrievalConfig
    server: ServerConfig

    @classmethod
    def from_env(cls, env_file: str | os.PathLike[str] | None = None) -> "AppConfig":
        file_values = load_env_file(env_file or os.environ.get("RAG_FLOW_ENV_FILE"))
        env = EnvView(file_values)

        base_dir = env.path("RAG_FLOW_BASE_DIR", DEFAULT_BASE_DIR)
        source_name = env.get("RAG_FLOW_SOURCE_NAME", DEFAULT_SOURCE_NAME)

        paths = PathsConfig(
            base_dir=base_dir,
            source_name=source_name,
            source_pdf=env.path(
                "RAG_FLOW_SOURCE_PDF",
                base_dir / "Dahua-DSS-Professional_User-Manual_V8.7.0_origin.pdf",
            ),
            content_json=env.path(
                "RAG_FLOW_CONTENT_JSON",
                base_dir / "Dahua-DSS-Professional_User-Manual_V8.7.0_content_list.json",
            ),
            small_icon_json=env.path(
                "RAG_FLOW_SMALL_ICON_JSON",
                base_dir / "Dahua-DSS-Professional_User-Manual_V8.7.0_content_list_small-icon-fixed.json",
            ),
            captioned_json=env.path(
                "RAG_FLOW_CAPTIONED_JSON",
                base_dir / "Dahua-DSS-Professional_User-Manual_V8.7.0_content_list_small-icon-fixed_image-with-captions.json",
            ),
            chunks_json=env.path(
                "RAG_FLOW_CHUNKS_JSON",
                base_dir / "Dahua-DSS-Professional_User-Manual_V8.7.0_page_level_chunks.json",
            ),
            db_path=env.path("RAG_FLOW_DB_PATH", "/root/qdrant-dbs/dahua-db"),
            collection_name=env.get("RAG_FLOW_COLLECTION", "dahua-manuals"),
        )

        models = ModelConfig(
            dense_model=env.get("RAG_FLOW_DENSE_MODEL", "intfloat/multilingual-e5-large"),
            sparse_model=env.get("RAG_FLOW_SPARSE_MODEL", "Qdrant/bm25"),
            colpali_model=env.get("RAG_FLOW_COLPALI_MODEL", "vidore/colpali-v1.3-merged"),
            vlm_model=env.get("RAG_FLOW_VLM_MODEL", "Qwen/Qwen3.5-9B"),
            llm_base_url=env.get("RAG_FLOW_LLM_BASE_URL", "http://localhost:8080/v1"),
            llm_api_key=env.get("RAG_FLOW_LLM_API_KEY", "EMPTY"),
            llm_model=env.get("RAG_FLOW_LLM_MODEL", "/root/autodl-tmp/models/Qwen3.5-35B-A3B-GPTQ-Int4"),
            llm_max_tokens=env.int("RAG_FLOW_LLM_MAX_TOKENS", 2048),
        )

        retrieval = RetrievalConfig(
            retrieval_k=env.int("RAG_FLOW_RETRIEVAL_K", 50),
            final_top_k=env.int("RAG_FLOW_FINAL_TOP_K", 10),
            rrf_k=env.int("RAG_FLOW_RRF_K", 60),
            visual_weight=env.float("RAG_FLOW_VISUAL_WEIGHT", 1.5),
            quantized_colpali=env.get("RAG_FLOW_QUANTIZED_COLPALI", "1") not in {"0", "false", "False"},
        )

        server = ServerConfig(
            host=env.get("RAG_FLOW_RETRIEVER_HOST", "127.0.0.1"),
            port=env.int("RAG_FLOW_RETRIEVER_PORT", 8000),
            retriever_url=env.get("RAG_FLOW_RETRIEVER_URL", "http://127.0.0.1:8000/retrieve"),
        )

        return cls(paths=paths, models=models, retrieval=retrieval, server=server)
