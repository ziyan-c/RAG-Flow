from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .presets import get_preset


DEFAULT_SOURCE_NAME = "example-technical-manual.pdf"
DEFAULT_SOURCE_ROOT = Path(".local/CUSTOM_DATA/pdfs/source")
DEFAULT_OUTPUT_ROOT = Path(".local/CUSTOM_DATA/pdfs/output")
DEFAULT_BASE_DIR = DEFAULT_OUTPUT_ROOT / Path(DEFAULT_SOURCE_NAME).stem / "auto"
DEFAULT_DB_PATH = Path(".local/CUSTOM_DATA/qdrant-db")
DEFAULT_LOCAL_ENV_FILE = Path(".local/rag-flow.env")
DEFAULT_RUNTIME_ENV_FILE = Path("autodl-tmp/.local/rag-flow.env")
DEFAULT_CONFIG_PRESET = "low"


def find_upwards(relative_path: str | Path, start: Path | None = None) -> Path | None:
    base = (start or Path.cwd()).resolve()
    for directory in (base, *base.parents):
        candidate = directory / relative_path
        if candidate.exists():
            return candidate
    return None


def resolve_env_file(env_file: str | os.PathLike[str] | None = None) -> str | os.PathLike[str] | None:
    if env_file:
        return env_file
    if os.environ.get("RAG_FLOW_ENV_FILE"):
        return os.environ["RAG_FLOW_ENV_FILE"]
    local_env = find_upwards(DEFAULT_LOCAL_ENV_FILE)
    if local_env:
        return local_env
    runtime_root = os.environ.get("RAG_FLOW_RUNTIME_ROOT")
    runtime_candidates = []
    if runtime_root:
        runtime_candidates.append(Path(runtime_root).expanduser() / ".local" / "rag-flow.env")
    runtime_candidates.append(Path.home() / DEFAULT_RUNTIME_ENV_FILE)
    for candidate in runtime_candidates:
        if candidate.exists():
            return candidate
    return None


def env_path_base(env_file: str | os.PathLike[str] | None) -> Path | None:
    if not env_file:
        return None
    env_path = Path(env_file).expanduser()
    if not env_path.is_absolute():
        env_path = Path.cwd() / env_path
    if env_path.parent.name == ".local":
        return env_path.parent.parent
    return env_path.parent


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


def apply_preset_defaults(file_values: Mapping[str, str]) -> dict[str, str]:
    preset_name = os.environ.get("RAG_FLOW_PRESET") or file_values.get("RAG_FLOW_PRESET") or DEFAULT_CONFIG_PRESET
    preset = get_preset(preset_name)
    return {**preset.env, **file_values, "RAG_FLOW_PRESET": preset.name}


class EnvView:
    def __init__(self, file_values: Mapping[str, str], *, path_base: Path | None = None):
        self.file_values = file_values
        self.path_base = path_base

    def get(self, key: str, default: str) -> str:
        return os.environ.get(key, self.file_values.get(key, default))

    def int(self, key: str, default: int) -> int:
        return int(self.get(key, str(default)))

    def float(self, key: str, default: float) -> float:
        return float(self.get(key, str(default)))

    def bool(self, key: str, default: bool) -> bool:
        value = self.get(key, "1" if default else "0").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def csv(self, key: str, default: str) -> tuple[str, ...]:
        return tuple(item.strip() for item in self.get(key, default).split(",") if item.strip())

    def path(self, key: str, default: str | Path) -> Path:
        if key in os.environ:
            value = os.environ[key]
            base = self.path_base if self.file_values.get(key) == value else None
        elif key in self.file_values:
            value = self.file_values[key]
            base = self.path_base
        else:
            value = str(default)
            base = None

        path = Path(value).expanduser()
        if base is not None and not path.is_absolute():
            return base / path
        return path


@dataclass(frozen=True)
class PathsConfig:
    base_dir: Path
    source_name: str
    source_pdf: Path
    content_json: Path
    sectioned_json: Path
    patched_json: Path
    captioned_json: Path
    chunks_json: Path
    tagged_json: Path
    db_path: Path
    collection_name: str
    source_root: Path | None = None


@dataclass(frozen=True)
class QdrantConfig:
    url: str = ""
    api_key: str = ""
    prefer_grpc: bool = False
    timeout: float = 30.0


@dataclass(frozen=True)
class ModelConfig:
    dense_model: str
    dense_vector_size: int
    sparse_model: str
    colpali_model: str
    vlm_model: str
    vlm_model_revision: str
    trusted_remote_code_models: tuple[str, ...]
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_max_tokens: int
    vlm_base_url: str = "http://localhost:8080/v1"
    vlm_api_key: str = "EMPTY"
    colpali_model_path: Path | None = None
    colpali_local_model_root: Path = Path("/root/autodl-tmp/models")


@dataclass(frozen=True)
class RetrievalConfig:
    retrieval_k: int
    final_top_k: int
    rrf_k: int
    visual_weight: float
    quantized_colpali: bool
    device: str = "auto"
    route_mode: str = "auto"
    visual_bonus: str = "none"
    candidate_scroll_page_size: int = 30
    max_context_tokens: int = 0
    context_chars_per_token: float = 4.0
    min_candidate_score: float = 0.0
    min_score_ratio: float = 1.0
    final_output_images: bool = False


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    retriever_url: str
    retriever_api_key: str
    max_query_chars: int


@dataclass(frozen=True)
class MinerUConfig:
    command: str
    input_path: Path
    output_dir: Path
    backend: str
    model_source: str
    lang: str
    extra_args: str
    package: str
    version: str
    extra: str
    python: str
    auto_install: bool


@dataclass(frozen=True)
class PatchingConfig:
    max_new_tokens: int
    llm_timeout: float
    batch_size: int = 9
    concurrency: int = 3
    checkpoint_interval: int = 30
    invalid_retry_limit: int = 0
    dpi: int = 250
    page_window_size: int = 200


@dataclass(frozen=True)
class CaptioningConfig:
    max_new_tokens: int
    max_context_tokens: int
    batch_size: int
    max_image_side: int = 0
    concurrency: int = 1
    checkpoint_interval: int = 1
    llm_timeout: float = 120.0


@dataclass(frozen=True)
class ChunkingConfig:
    mode: str = "auto"
    max_tokens: int = 1500
    overlap_tokens: int = 150
    min_tokens: int = 150


@dataclass(frozen=True)
class TaggingConfig:
    enabled: bool = False


@dataclass(frozen=True)
class IndexingConfig:
    mode: str = "text"
    text_batch_size: int = 256
    visual_batch_size: int = 8
    visual_dpi: int = 200


@dataclass(frozen=True)
class ModelServerConfig:
    auto_start: bool = False
    stop_after: bool = True
    command: str = ""
    log_path: Path | None = None
    startup_timeout: float = 900.0
    poll_interval: float = 5.0
    sglang_model_profile: str = "custom"
    sglang_model_id: str = ""
    sglang_model_path: Path | None = None
    sglang_served_model_name: str = ""
    sglang_python: str = ""
    sglang_local_model_root: Path = Path("/root/autodl-tmp/models")
    sglang_mem_fraction_static: str = ""
    sglang_context_length: str = ""
    sglang_tp_size: str = ""
    sglang_quantization: str = ""
    sglang_reasoning_parser: str = ""
    sglang_attention_backend: str = ""
    sglang_kv_cache_dtype: str = ""
    sglang_extra_args: str = ""


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    models: ModelConfig
    retrieval: RetrievalConfig
    server: ServerConfig
    mineru: MinerUConfig
    patching: PatchingConfig
    captioning: CaptioningConfig
    chunking: ChunkingConfig
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    tagging: TaggingConfig = field(default_factory=TaggingConfig)
    vlm_server: ModelServerConfig = field(default_factory=ModelServerConfig)
    llm_server: ModelServerConfig = field(default_factory=ModelServerConfig)

    @classmethod
    def from_env(cls, env_file: str | os.PathLike[str] | None = None) -> "AppConfig":
        resolved_env_file = resolve_env_file(env_file)
        file_values = apply_preset_defaults(load_env_file(resolved_env_file))
        env = EnvView(file_values, path_base=env_path_base(resolved_env_file))

        base_dir = env.path("RAG_FLOW_BASE_DIR", DEFAULT_BASE_DIR)
        source_name = env.get("RAG_FLOW_SOURCE_NAME", DEFAULT_SOURCE_NAME)
        source_stem = Path(source_name).stem

        source_root_value = env.get("RAG_FLOW_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT)).strip()
        source_root = env.path("RAG_FLOW_SOURCE_ROOT", DEFAULT_SOURCE_ROOT) if source_root_value else None

        paths = PathsConfig(
            base_dir=base_dir,
            source_name=source_name,
            source_root=source_root,
            source_pdf=env.path(
                "RAG_FLOW_SOURCE_PDF",
                DEFAULT_SOURCE_ROOT / DEFAULT_SOURCE_NAME,
            ),
            content_json=env.path(
                "RAG_FLOW_CONTENT_JSON",
                base_dir / f"{source_stem}_content_list.json",
            ),
            sectioned_json=env.path(
                "RAG_FLOW_SECTIONED_JSON",
                base_dir / f"{source_stem}_content_list_SECTIONED.json",
            ),
            patched_json=env.path(
                "RAG_FLOW_PATCHED_JSON",
                env.path(
                    "RAG_FLOW_SMALL_ICON_JSON",
                    base_dir / f"{source_stem}_content_list_SECTIONED_PATCHED.json",
                ),
            ),
            captioned_json=env.path(
                "RAG_FLOW_CAPTIONED_JSON",
                base_dir / f"{source_stem}_content_list_SECTIONED_PATCHED_CAPTIONED.json",
            ),
            chunks_json=env.path(
                "RAG_FLOW_CHUNKS_JSON",
                base_dir / f"{source_stem}_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED.json",
            ),
            tagged_json=env.path(
                "RAG_FLOW_TAGGED_JSON",
                base_dir / f"{source_stem}_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED_TAGGED.json",
            ),
            db_path=env.path("RAG_FLOW_DB_PATH", DEFAULT_DB_PATH),
            collection_name=env.get("RAG_FLOW_COLLECTION", "technical-manuals"),
        )

        qdrant = QdrantConfig(
            url=env.get("RAG_FLOW_QDRANT_URL", ""),
            api_key=env.get("RAG_FLOW_QDRANT_API_KEY", ""),
            prefer_grpc=env.bool("RAG_FLOW_QDRANT_PREFER_GRPC", False),
            timeout=env.float("RAG_FLOW_QDRANT_TIMEOUT", 30.0),
        )

        models = ModelConfig(
            dense_model=env.get("RAG_FLOW_DENSE_MODEL", "intfloat/multilingual-e5-large"),
            dense_vector_size=env.int("RAG_FLOW_DENSE_VECTOR_SIZE", 1024),
            sparse_model=env.get("RAG_FLOW_SPARSE_MODEL", "Qdrant/bm25"),
            colpali_model=env.get("RAG_FLOW_COLPALI_MODEL", "vidore/colpali-v1.3-merged"),
            colpali_model_path=env.path("RAG_FLOW_COLPALI_MODEL_PATH", "")
            if env.get("RAG_FLOW_COLPALI_MODEL_PATH", "")
            else None,
            colpali_local_model_root=env.path("RAG_FLOW_COLPALI_LOCAL_MODEL_ROOT", "/root/autodl-tmp/models"),
            vlm_model=env.get("RAG_FLOW_VLM_MODEL", "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"),
            vlm_model_revision=env.get("RAG_FLOW_VLM_MODEL_REVISION", ""),
            trusted_remote_code_models=env.csv(
                "RAG_FLOW_TRUSTED_REMOTE_CODE_MODELS",
                "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4",
            ),
            vlm_base_url=env.get("RAG_FLOW_VLM_BASE_URL", env.get("RAG_FLOW_LLM_BASE_URL", "http://localhost:8080/v1")),
            vlm_api_key=env.get("RAG_FLOW_VLM_API_KEY", env.get("RAG_FLOW_LLM_API_KEY", "EMPTY")),
            llm_base_url=env.get("RAG_FLOW_LLM_BASE_URL", "http://localhost:8080/v1"),
            llm_api_key=env.get("RAG_FLOW_LLM_API_KEY", "EMPTY"),
            llm_model=env.get("RAG_FLOW_LLM_MODEL", "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"),
            llm_max_tokens=env.int("RAG_FLOW_LLM_MAX_TOKENS", 20000),
        )

        retrieval = RetrievalConfig(
            retrieval_k=env.int("RAG_FLOW_RETRIEVAL_K", 80),
            final_top_k=env.int("RAG_FLOW_FINAL_TOP_K", 20),
            rrf_k=env.int("RAG_FLOW_RRF_K", 10),
            visual_weight=env.float("RAG_FLOW_VISUAL_WEIGHT", 2.5),
            quantized_colpali=env.get("RAG_FLOW_QUANTIZED_COLPALI", "1") not in {"0", "false", "False"},
            device=env.get("RAG_FLOW_RETRIEVAL_DEVICE", "auto"),
            route_mode=env.get("RAG_FLOW_RETRIEVAL_ROUTE_MODE", "text"),
            visual_bonus=env.get("RAG_FLOW_RETRIEVAL_VISUAL_BONUS", "none"),
            candidate_scroll_page_size=env.int("RAG_FLOW_RETRIEVAL_CANDIDATE_SCROLL_PAGE_SIZE", 30),
            max_context_tokens=env.int("RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS", 10000),
            context_chars_per_token=env.float("RAG_FLOW_RETRIEVAL_CONTEXT_CHARS_PER_TOKEN", 4.0),
            min_candidate_score=env.float("RAG_FLOW_RETRIEVAL_MIN_CANDIDATE_SCORE", 0.0),
            min_score_ratio=env.float("RAG_FLOW_RETRIEVAL_MIN_SCORE_RATIO", 1.0),
            final_output_images=env.bool("RAG_FLOW_RETRIEVAL_FINAL_OUTPUT_IMAGES", False),
        )

        server = ServerConfig(
            host=env.get("RAG_FLOW_RETRIEVER_HOST", "127.0.0.1"),
            port=env.int("RAG_FLOW_RETRIEVER_PORT", 8000),
            retriever_url=env.get("RAG_FLOW_RETRIEVER_URL", "http://127.0.0.1:8000/retrieve"),
            retriever_api_key=env.get("RAG_FLOW_RETRIEVER_API_KEY", ""),
            max_query_chars=env.int("RAG_FLOW_RETRIEVER_MAX_QUERY_CHARS", 4000),
        )

        mineru = MinerUConfig(
            command=env.get("RAG_FLOW_MINERU_COMMAND", "mineru"),
            input_path=env.path("RAG_FLOW_MINERU_INPUT_PATH", paths.source_pdf),
            output_dir=env.path("RAG_FLOW_MINERU_OUTPUT_DIR", DEFAULT_OUTPUT_ROOT),
            backend=env.get("RAG_FLOW_MINERU_BACKEND", ""),
            model_source=env.get("RAG_FLOW_MINERU_MODEL_SOURCE", ""),
            lang=env.get("RAG_FLOW_MINERU_LANG", ""),
            extra_args=env.get("RAG_FLOW_MINERU_EXTRA_ARGS", ""),
            package=env.get("RAG_FLOW_MINERU_PACKAGE", "mineru"),
            version=env.get("RAG_FLOW_MINERU_VERSION", "3.0.9"),
            extra=env.get("RAG_FLOW_MINERU_EXTRA", "all"),
            python=env.get("RAG_FLOW_MINERU_PYTHON", ""),
            auto_install=env.bool("RAG_FLOW_AUTO_INSTALL_MINERU", False),
        )

        patching = PatchingConfig(
            max_new_tokens=env.int("RAG_FLOW_PATCH_MAX_NEW_TOKENS", 8000),
            llm_timeout=env.float("RAG_FLOW_PATCH_LLM_TIMEOUT", 120.0),
            batch_size=env.int("RAG_FLOW_PATCH_BATCH_SIZE", 512),
            concurrency=env.int("RAG_FLOW_PATCH_CONCURRENCY", 8),
            checkpoint_interval=env.int("RAG_FLOW_PATCH_CHECKPOINT_INTERVAL", 1),
            invalid_retry_limit=env.int("RAG_FLOW_PATCH_INVALID_RETRY_LIMIT", 0),
            dpi=env.int("RAG_FLOW_PATCH_DPI", 250),
            page_window_size=env.int("RAG_FLOW_PATCH_PAGE_WINDOW_SIZE", 200),
        )

        captioning = CaptioningConfig(
            max_new_tokens=env.int("RAG_FLOW_CAPTION_MAX_NEW_TOKENS", 8000),
            max_context_tokens=env.int("RAG_FLOW_CAPTION_MAX_CONTEXT_TOKENS", 2000),
            batch_size=env.int("RAG_FLOW_CAPTION_BATCH_SIZE", 32),
            max_image_side=env.int("RAG_FLOW_CAPTION_MAX_IMAGE_SIDE", 2048),
            concurrency=env.int("RAG_FLOW_CAPTION_CONCURRENCY", 3),
            checkpoint_interval=env.int("RAG_FLOW_CAPTION_CHECKPOINT_INTERVAL", 1),
            llm_timeout=env.float("RAG_FLOW_CAPTION_LLM_TIMEOUT", 120.0),
        )

        chunking = ChunkingConfig(
            mode=env.get("RAG_FLOW_CHUNK_MODE", "auto"),
            max_tokens=env.int("RAG_FLOW_CHUNK_MAX_TOKENS", 1500),
            overlap_tokens=env.int("RAG_FLOW_CHUNK_OVERLAP_TOKENS", 150),
            min_tokens=env.int("RAG_FLOW_CHUNK_MIN_TOKENS", 150),
        )

        tagging = TaggingConfig(
            enabled=env.bool("RAG_FLOW_TAGGING_ENABLED", False),
        )

        indexing = IndexingConfig(
            mode=env.get("RAG_FLOW_INDEX_MODE", "text").strip().lower(),
            text_batch_size=env.int("RAG_FLOW_INDEX_TEXT_BATCH_SIZE", 256),
            visual_batch_size=env.int("RAG_FLOW_INDEX_VISUAL_BATCH_SIZE", 8),
            visual_dpi=env.int("RAG_FLOW_INDEX_VISUAL_DPI", 200),
        )

        def server_config(prefix: str, *, default_profile: str, default_model_id: str, default_quantization: str) -> ModelServerConfig:
            raw_model_path = env.get(f"RAG_FLOW_{prefix}_SGLANG_MODEL_PATH", "")
            raw_log_path = env.get(f"RAG_FLOW_{prefix}_SERVER_LOG_PATH", "")
            return ModelServerConfig(
                auto_start=env.bool(f"RAG_FLOW_{prefix}_SERVER_AUTO_START", False),
                stop_after=env.bool(f"RAG_FLOW_{prefix}_SERVER_STOP_AFTER", True),
                command=env.get(f"RAG_FLOW_{prefix}_SERVER_COMMAND", ""),
                log_path=env.path(f"RAG_FLOW_{prefix}_SERVER_LOG_PATH", raw_log_path) if raw_log_path else None,
                startup_timeout=env.float(f"RAG_FLOW_{prefix}_SERVER_STARTUP_TIMEOUT", 900.0),
                poll_interval=env.float(f"RAG_FLOW_{prefix}_SERVER_POLL_INTERVAL", 5.0),
                sglang_model_profile=env.get(f"RAG_FLOW_{prefix}_SGLANG_MODEL_PROFILE", default_profile),
                sglang_model_id=env.get(f"RAG_FLOW_{prefix}_SGLANG_MODEL_ID", default_model_id),
                sglang_model_path=env.path(f"RAG_FLOW_{prefix}_SGLANG_MODEL_PATH", raw_model_path)
                if raw_model_path
                else None,
                sglang_served_model_name=env.get(f"RAG_FLOW_{prefix}_SGLANG_SERVED_MODEL_NAME", default_model_id),
                sglang_python=env.get(f"RAG_FLOW_{prefix}_SGLANG_PYTHON", ""),
                sglang_local_model_root=env.path(
                    f"RAG_FLOW_{prefix}_SGLANG_LOCAL_MODEL_ROOT",
                    "/root/autodl-tmp/models",
                ),
                sglang_mem_fraction_static=env.get(f"RAG_FLOW_{prefix}_SGLANG_MEM_FRACTION_STATIC", ""),
                sglang_context_length=env.get(f"RAG_FLOW_{prefix}_SGLANG_CONTEXT_LENGTH", ""),
                sglang_tp_size=env.get(f"RAG_FLOW_{prefix}_SGLANG_TP_SIZE", ""),
                sglang_quantization=env.get(f"RAG_FLOW_{prefix}_SGLANG_QUANTIZATION", default_quantization),
                sglang_reasoning_parser=env.get(f"RAG_FLOW_{prefix}_SGLANG_REASONING_PARSER", ""),
                sglang_attention_backend=env.get(f"RAG_FLOW_{prefix}_SGLANG_ATTENTION_BACKEND", ""),
                sglang_kv_cache_dtype=env.get(f"RAG_FLOW_{prefix}_SGLANG_KV_CACHE_DTYPE", ""),
                sglang_extra_args=env.get(f"RAG_FLOW_{prefix}_SGLANG_EXTRA_ARGS", ""),
            )

        vlm_server = server_config(
            "VLM",
            default_profile="qwen3.6-35b-a3b-gptq-int4",
            default_model_id=models.vlm_model,
            default_quantization="moe_wna16",
        )
        llm_server = server_config(
            "LLM",
            default_profile="qwen3.6-35b-a3b-gptq-int4",
            default_model_id=models.llm_model,
            default_quantization="moe_wna16",
        )

        return cls(
            paths=paths,
            qdrant=qdrant,
            models=models,
            retrieval=retrieval,
            server=server,
            mineru=mineru,
            patching=patching,
            captioning=captioning,
            chunking=chunking,
            tagging=tagging,
            indexing=indexing,
            vlm_server=vlm_server,
            llm_server=llm_server,
        )
