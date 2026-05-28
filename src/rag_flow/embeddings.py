from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from .runtime import get_torch_device


QWEN3_QUERY_INSTRUCTION = (
    "Given a customer support technical-manual question, retrieve relevant passages "
    "from product manuals that answer the question."
)


@dataclass(frozen=True)
class SparseEmbeddingVector:
    indices: np.ndarray
    values: np.ndarray


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    return value


def _batched(items: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def sparse_embedding_parts(sparse_vec: Any) -> tuple[list[int], list[float]]:
    indices = sparse_vec.indices
    values = sparse_vec.values
    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(index) for index in indices], [float(value) for value in values]


class FastEmbedDenseEmbedding:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)

    def embed(self, documents: Iterable[str]) -> Iterable[Any]:
        return self.model.embed(documents)

    def query_embed(self, query: str | Iterable[str]) -> Iterable[Any]:
        return self.model.query_embed(query)


class FastEmbedSparseEmbedding:
    def __init__(self, model_name: str):
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding(model_name)

    def embed(self, documents: Iterable[str]) -> Iterable[Any]:
        return self.model.embed(documents)

    def query_embed(self, query: str | Iterable[str]) -> Iterable[Any]:
        return self.model.query_embed(query)


class OffDenseEmbedding:
    def __init__(self, *, vector_size: int | None):
        self.vector_size = int(vector_size or 1024)
        if self.vector_size <= 0:
            raise ValueError("Dense off-vector size must be positive")

    def _vector(self) -> np.ndarray:
        vector = np.zeros(self.vector_size, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def embed(self, documents: Iterable[str]) -> Iterable[np.ndarray]:
        for _ in documents:
            yield self._vector()

    def query_embed(self, query: str | Iterable[str]) -> Iterable[np.ndarray]:
        queries = [query] if isinstance(query, str) else list(query)
        for _ in queries:
            yield self._vector()


class OffSparseEmbedding:
    def _vector(self) -> SparseEmbeddingVector:
        return SparseEmbeddingVector(
            indices=np.array([0], dtype=np.uint32),
            values=np.array([1e-12], dtype=np.float32),
        )

    def embed(self, documents: Iterable[str]) -> Iterable[SparseEmbeddingVector]:
        for _ in documents:
            yield self._vector()

    def query_embed(self, query: str | Iterable[str]) -> Iterable[SparseEmbeddingVector]:
        queries = [query] if isinstance(query, str) else list(query)
        for _ in queries:
            yield self._vector()


class Qwen3DenseEmbedding:
    """Transformers adapter for Qwen3 embedding models.

    Qwen3 embedding model cards require last-token pooling, normalized vectors, and
    an instruction prefix on the query side. Documents are embedded without the
    instruction prefix.
    """

    def __init__(self, model_name: str, *, vector_size: int | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.vector_size = int(vector_size or 0) or None
        self.max_length = _env_int("RAG_FLOW_EMBEDDING_MAX_LENGTH", 8192)
        self.batch_size = _env_int("RAG_FLOW_DENSE_EMBEDDING_BATCH_SIZE", 8)
        self.query_instruction = os.environ.get("RAG_FLOW_QWEN3_QUERY_INSTRUCTION", QWEN3_QUERY_INSTRUCTION)
        self.device = get_torch_device(
            feature=f"{model_name} dense embedding",
            preferred=os.environ.get("RAG_FLOW_DENSE_EMBEDDING_DEVICE", "auto"),
        )
        dtype_name = os.environ.get("RAG_FLOW_DENSE_EMBEDDING_DTYPE", "auto").strip().lower()
        dtype = None
        if dtype_name == "auto":
            dtype = torch.float16 if self.device == "cuda" else torch.float32
        elif dtype_name in {"float16", "fp16"}:
            dtype = torch.float16
        elif dtype_name in {"bfloat16", "bf16"}:
            dtype = torch.bfloat16
        elif dtype_name in {"float32", "fp32"}:
            dtype = torch.float32
        else:
            raise ValueError(
                "RAG_FLOW_DENSE_EMBEDDING_DTYPE must be one of auto, float16, bfloat16, or float32"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()

    @staticmethod
    def _last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
        import torch

        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _query_text(self, query: str) -> str:
        return f"Instruct: {self.query_instruction}\nQuery:{query}"

    def _encode(self, texts: Sequence[str], *, is_query: bool) -> Iterator[np.ndarray]:
        import torch
        import torch.nn.functional as F

        prepared = [self._query_text(text) if is_query else text for text in texts]
        with torch.no_grad():
            for batch_texts in _batched(prepared, self.batch_size):
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                embeddings = self._last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
                if self.vector_size is not None:
                    if embeddings.shape[1] < self.vector_size:
                        raise ValueError(
                            f"{self.model_name} produced {embeddings.shape[1]} dimensions, "
                            f"but RAG_FLOW_DENSE_VECTOR_SIZE={self.vector_size}"
                        )
                    if embeddings.shape[1] > self.vector_size:
                        embeddings = embeddings[:, : self.vector_size]
                embeddings = F.normalize(embeddings, p=2, dim=1)
                for row in embeddings.float().cpu().numpy():
                    yield row

    def embed(self, documents: Iterable[str]) -> Iterable[np.ndarray]:
        return self._encode(list(documents), is_query=False)

    def query_embed(self, query: str | Iterable[str]) -> Iterable[np.ndarray]:
        queries = [query] if isinstance(query, str) else list(query)
        return self._encode(queries, is_query=True)


class BGEM3SparseEmbedding:
    def __init__(self, model_name: str):
        from FlagEmbedding import BGEM3FlagModel

        self.model_name = model_name
        self.batch_size = _env_int("RAG_FLOW_SPARSE_EMBEDDING_BATCH_SIZE", 16)
        self.max_length = _env_int("RAG_FLOW_SPARSE_EMBEDDING_MAX_LENGTH", 8192)
        kwargs: dict[str, Any] = {"use_fp16": os.environ.get("RAG_FLOW_SPARSE_EMBEDDING_FP16", "1") != "0"}
        devices = os.environ.get("RAG_FLOW_SPARSE_EMBEDDING_DEVICES", "").strip()
        if devices:
            kwargs["devices"] = [item.strip() for item in devices.split(",") if item.strip()]
        self.model = BGEM3FlagModel(self._model_location(model_name), **kwargs)

    @staticmethod
    def _model_location(model_name: str) -> str:
        explicit_path = os.environ.get("RAG_FLOW_BGE_M3_MODEL_PATH", "").strip()
        if explicit_path:
            return explicit_path
        if model_name.strip().lower() != "baai/bge-m3":
            return model_name

        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=model_name,
            ignore_patterns=("*.DS_Store", "imgs/.DS_Store"),
        )

    @staticmethod
    def _weights_to_sparse(weights: dict[Any, Any]) -> SparseEmbeddingVector:
        pairs: list[tuple[int, float]] = []
        for key, value in weights.items():
            try:
                index = int(key)
                score = float(value)
            except (TypeError, ValueError):
                continue
            if index >= 0 and score != 0.0:
                pairs.append((index, score))
        pairs.sort(key=lambda item: item[0])
        return SparseEmbeddingVector(
            indices=np.array([index for index, _ in pairs], dtype=np.uint32),
            values=np.array([value for _, value in pairs], dtype=np.float32),
        )

    def _encode(self, texts: Sequence[str]) -> Iterator[SparseEmbeddingVector]:
        for batch_texts in _batched(texts, self.batch_size):
            output = self.model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                max_length=self.max_length,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            weights_list = output.get("lexical_weights", [])
            for weights in weights_list:
                yield self._weights_to_sparse(weights if isinstance(weights, dict) else {})

    def embed(self, documents: Iterable[str]) -> Iterable[SparseEmbeddingVector]:
        return self._encode(list(documents))

    def query_embed(self, query: str | Iterable[str]) -> Iterable[SparseEmbeddingVector]:
        queries = [query] if isinstance(query, str) else list(query)
        return self._encode(queries)


def create_dense_embedding(model_name: str, *, vector_size: int | None = None) -> Any:
    normalized = model_name.strip().lower()
    if normalized in {"off", "none", "disabled"}:
        return OffDenseEmbedding(vector_size=vector_size)
    if normalized.startswith("qwen/qwen3-embedding-"):
        return Qwen3DenseEmbedding(model_name, vector_size=vector_size)
    return FastEmbedDenseEmbedding(model_name)


def create_sparse_embedding(model_name: str) -> Any:
    normalized = model_name.strip().lower()
    if normalized in {"off", "none", "disabled"}:
        return OffSparseEmbedding()
    if normalized == "baai/bge-m3":
        return BGEM3SparseEmbedding(model_name)
    return FastEmbedSparseEmbedding(model_name)
