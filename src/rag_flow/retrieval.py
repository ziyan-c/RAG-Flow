from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .indexing import PAGE_COLPALI_VECTOR_NAME, TEXT_DENSE_VECTOR_NAME, TEXT_SPARSE_VECTOR_NAME
from .runtime import get_torch_device


@dataclass(frozen=True)
class HitDetail:
    rank: int
    page_idx: int
    page_number: int
    score: float
    is_continuation: bool


@dataclass(frozen=True)
class RetrievalResult:
    hit_page: int
    all_hits: list[HitDetail]
    context: str


class RetrievalEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = None
        self.dense_model = None
        self.sparse_model = None
        self.colpali_processor = None
        self.colpali_model = None
        self.device = None

    def load(self) -> None:
        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor
        from fastembed import SparseTextEmbedding, TextEmbedding
        from qdrant_client import QdrantClient
        from transformers import BitsAndBytesConfig

        self.device = get_torch_device(
            require_cuda=self.config.retrieval.quantized_colpali,
            feature="Quantized ColPali retrieval",
        )
        self.client = QdrantClient(path=str(self.config.paths.db_path))
        self.dense_model = TextEmbedding(self.config.models.dense_model)
        self.sparse_model = SparseTextEmbedding(self.config.models.sparse_model)
        self.colpali_processor = ColPaliProcessor.from_pretrained(self.config.models.colpali_model)

        kwargs: dict[str, Any] = {"device_map": self.device}
        if self.config.retrieval.quantized_colpali:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.colpali_model = ColPali.from_pretrained(self.config.models.colpali_model, **kwargs).eval()

    def retrieve(self, query_text: str) -> RetrievalResult:
        if self.client is None:
            self.load()

        import torch
        from qdrant_client import models

        dense_query = list(self.dense_model.query_embed(query_text))[0].tolist()
        sparse_query_obj = list(self.sparse_model.query_embed(query_text))[0]
        sparse_indices = (
            sparse_query_obj.indices.tolist()
            if hasattr(sparse_query_obj.indices, "tolist")
            else list(sparse_query_obj.indices)
        )
        sparse_values = (
            sparse_query_obj.values.tolist()
            if hasattr(sparse_query_obj.values, "tolist")
            else list(sparse_query_obj.values)
        )

        with torch.no_grad():
            batch_query = self.colpali_processor.process_queries([query_text]).to(self.device)
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            batch_query = {
                key: value.to(dtype) if value.is_floating_point() else value
                for key, value in batch_query.items()
            }
            visual_query = self.colpali_model(**batch_query)[0].cpu().float().tolist()

        collection = self.config.paths.collection_name
        retrieval_k = self.config.retrieval.retrieval_k
        dense_hits = self.client.query_points(
            collection_name=collection,
            query=dense_query,
            using=TEXT_DENSE_VECTOR_NAME,
            limit=retrieval_k,
        ).points
        sparse_hits = self.client.query_points(
            collection_name=collection,
            query=models.SparseVector(indices=sparse_indices, values=sparse_values),
            using=TEXT_SPARSE_VECTOR_NAME,
            limit=retrieval_k,
        ).points
        visual_hits = self.client.query_points(
            collection_name=collection,
            query=visual_query,
            using=PAGE_COLPALI_VECTOR_NAME,
            limit=retrieval_k,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=10.0,
                ),
                hnsw_ef=128,
            ),
        ).points

        final_ranking = self._compute_rrf(dense_hits, sparse_hits, visual_hits)
        top_hits = final_ranking[: self.config.retrieval.final_top_k]

        if not top_hits:
            return RetrievalResult(
                hit_page=1,
                all_hits=[],
                context="No relevant information found in the manual.",
            )

        seen_records: set[tuple[str, str]] = set()
        context_blocks: list[str] = []
        hit_details: list[HitDetail] = []

        for rank, hit in enumerate(top_hits):
            payload = hit["payload"]
            original_hit_page = int(payload["page_idx"])
            source_pdf = payload["source"]
            is_continuation = bool(payload.get("is_table_continuation", False))
            hit_details.append(
                HitDetail(
                    rank=rank + 1,
                    page_idx=original_hit_page,
                    page_number=original_hit_page + 1,
                    score=float(hit["score"]),
                    is_continuation=is_continuation,
                )
            )

            logical_center_page = int(payload.get("parent_page_idx", original_hit_page))
            should_conditions = [
                models.FieldCondition(key="page_idx", match=models.MatchValue(value=logical_center_page)),
                models.FieldCondition(key="page_indices", match=models.MatchValue(value=logical_center_page)),
                models.FieldCondition(key="parent_page_idx", match=models.MatchValue(value=logical_center_page)),
            ]
            if rank == 0:
                for offset in [-2, -1, 1, 2]:
                    page = logical_center_page + offset
                    should_conditions.append(
                        models.FieldCondition(key="page_idx", match=models.MatchValue(value=page))
                    )
                    should_conditions.append(
                        models.FieldCondition(key="page_indices", match=models.MatchValue(value=page))
                    )
            elif rank <= 2:
                for offset in [-1, 1]:
                    page = logical_center_page + offset
                    should_conditions.append(
                        models.FieldCondition(key="page_idx", match=models.MatchValue(value=page))
                    )
                    should_conditions.append(
                        models.FieldCondition(key="page_indices", match=models.MatchValue(value=page))
                    )

            records, _ = self.client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchValue(value=source_pdf))],
                    should=should_conditions,
                ),
                limit=30,
                with_payload=True,
                with_vectors=False,
            )

            unique_records = []
            for record in records:
                page_idx = int(record.payload["page_idx"])
                key = (source_pdf, str(record.payload.get("chunk_id") or record.id))
                if page_idx >= 0 and key not in seen_records:
                    unique_records.append(record)
                    seen_records.add(key)
            unique_records.sort(
                key=lambda item: (
                    int(item.payload.get("page_idx", item.payload.get("page_start", 0))),
                    int(item.payload.get("chunk_idx", -1)),
                    str(item.payload.get("chunk_id", "")),
                )
            )

            for record in unique_records:
                page_idx = int(record.payload["page_idx"])
                page_start = int(record.payload.get("page_start", page_idx))
                page_end = int(record.payload.get("page_end", page_idx))
                page_label = f"{page_start + 1}" if page_start == page_end else f"{page_start + 1}-{page_end + 1}"
                section = record.payload.get("section_title")
                section_line = f", Section: {section}" if section else ""
                note_prefix = ""
                if page_idx == original_hit_page and is_continuation:
                    note_prefix = "[Target Visual Match] "
                context_blocks.append(
                    f"[Source: {source_pdf}, Page: {page_label}{section_line}]\n"
                    f"{note_prefix}{record.payload.get('chunk_content', '')}"
                )

        final_context = (
            "Based on the following retrieved context from the official manual, "
            "please answer the user's question.\n"
            "If the answer is not contained within the context, state that clearly.\n"
            "Please always cite the Source and Page number when providing facts.\n\n"
            "--- START OF CONTEXT ---\n\n"
            + "\n\n".join(context_blocks)
            + "\n\n--- END OF CONTEXT ---\n"
        )

        return RetrievalResult(
            hit_page=int(top_hits[0]["payload"]["page_idx"]) + 1,
            all_hits=hit_details,
            context=final_context,
        )

    def _compute_rrf(self, dense_res: list[Any], sparse_res: list[Any], visual_res: list[Any]) -> list[dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}

        def add_to_rrf(results: list[Any], weight: float) -> None:
            for rank, hit in enumerate(results):
                if hit.id not in scores:
                    scores[hit.id] = {"score": 0.0, "payload": hit.payload}
                scores[hit.id]["score"] += weight * (1.0 / (self.config.retrieval.rrf_k + rank + 1))

        add_to_rrf(dense_res, 1.0)
        add_to_rrf(sparse_res, 1.0)
        add_to_rrf(visual_res, self.config.retrieval.visual_weight)
        return sorted(scores.values(), key=lambda item: item["score"], reverse=True)
