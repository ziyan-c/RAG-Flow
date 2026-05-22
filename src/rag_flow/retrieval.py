from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import AppConfig
from .indexing import COLPALI_VECTOR_SIZE, PAGE_IMAGE_COLPALI_VECTOR_NAME, TEXT_DENSE_VECTOR_NAME, TEXT_SPARSE_VECTOR_NAME
from .model_paths import resolve_model_location
from .qdrant import create_qdrant_client
from .runtime import get_torch_device
from .source_paths import normalize_source_name, source_breadcrumb


@dataclass(frozen=True)
class RetrievedImage:
    hit_rank: int
    chunk_id: str
    source_relpath: str
    img_path: str
    image_path: str
    image_exists: bool
    page_idx: int
    page_number: int
    bbox: list[float]
    image_answering_policy: str
    image_answering_confidence: str = ""
    image_answering_reason: str = ""
    image_caption: str = ""
    image_description_vlm: str = ""


@dataclass(frozen=True)
class HitDetail:
    rank: int
    page_idx: int
    page_number: int
    page_indices: list[int]
    page_numbers: list[int]
    score: float
    is_continuation: bool
    chunk_id: str = ""
    visual_page_prior: float = 0.0
    visual_alignment_score: float = 0.0
    dense_rrf_score: float = 0.0
    sparse_rrf_score: float = 0.0
    visual_rrf_score: float = 0.0
    direct_text_rrf_score: float = 0.0
    image_references: tuple[RetrievedImage, ...] = ()


@dataclass(frozen=True)
class FinalOutput:
    mode: str
    context: str
    content: tuple[dict[str, Any], ...]
    images: tuple[RetrievedImage, ...] = ()


@dataclass(frozen=True)
class RetrievalResult:
    hit_page: int
    all_hits: list[HitDetail]
    context: str
    images: tuple[RetrievedImage, ...] = ()
    final_output: FinalOutput | None = None


_FINAL_OUTPUT_IMAGE_POLICIES = {
    "image_recommended",
    "image_required",
    "recommended",
    "required",
}


def _is_final_output_image(image: RetrievedImage) -> bool:
    policy = image.image_answering_policy.strip().lower().replace("-", "_")
    return bool(image.image_exists and policy in _FINAL_OUTPUT_IMAGE_POLICIES)


def build_final_output(
    *,
    context: str,
    images: tuple[RetrievedImage, ...] = (),
    include_images: bool = False,
) -> FinalOutput:
    selected_images = tuple(image for image in images if include_images and _is_final_output_image(image))
    content: list[dict[str, Any]] = [{"type": "text", "text": context}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image.image_path}}
        for image in selected_images
    )
    return FinalOutput(
        mode="openai_compatible_multimodal" if selected_images else "context_only",
        context=context,
        content=tuple(content),
        images=selected_images,
    )


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _payload_chunk_key(payload: dict[str, Any]) -> tuple[str, str]:
    source = _payload_source_relpath(payload)
    chunk_id = payload.get("chunk_id")
    if chunk_id is None:
        chunk_id = payload.get("id")
    if chunk_id is None:
        chunk_id = f"chunk_idx:{payload.get('chunk_idx', '')}"
    return source, str(chunk_id)


def _payload_source_relpath(payload: dict[str, Any]) -> str:
    return normalize_source_name(payload.get("source_relpath") or payload.get("source") or "")


def _payload_page_indices(payload: dict[str, Any]) -> list[int]:
    pages = payload.get("page_indices")
    if isinstance(pages, list):
        resolved = []
        for page in pages:
            try:
                resolved.append(int(page))
            except (TypeError, ValueError):
                continue
        if resolved:
            return sorted(set(resolved))
    page_start = _as_int(payload.get("page_start", payload.get("page_idx", 0)))
    page_end = _as_int(payload.get("page_end", page_start), page_start)
    if page_end < page_start:
        page_start, page_end = page_end, page_start
    return list(range(page_start, page_end + 1))


def _payload_bboxes_by_page(payload: dict[str, Any]) -> dict[int, list[tuple[float, float, float, float]]]:
    raw = payload.get("bboxes_by_page", {})
    if not isinstance(raw, dict):
        return {}
    resolved: dict[int, list[tuple[float, float, float, float]]] = {}
    for page_key, bboxes in raw.items():
        try:
            page_idx = int(page_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(bboxes, list):
            continue
        page_bboxes = []
        for bbox in bboxes:
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x0, y0, x1, y1 = (float(value) for value in bbox)
            except (TypeError, ValueError):
                continue
            if x1 > x0 and y1 > y0:
                page_bboxes.append((x0, y0, x1, y1))
        if page_bboxes:
            resolved[page_idx] = page_bboxes
    return resolved


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _page_bbox_area_fraction(payload: dict[str, Any], page_idx: int) -> float:
    bboxes_by_page = _payload_bboxes_by_page(payload)
    total_area = sum(_bbox_area(bbox) for bboxes in bboxes_by_page.values() for bbox in bboxes)
    if total_area <= 0:
        return 0.0
    page_area = sum(_bbox_area(bbox) for bbox in bboxes_by_page.get(page_idx, []))
    return max(0.0, min(1.0, page_area / total_area))


def _payload_position_key(payload: dict[str, Any], preferred_page_idx: int | None = None) -> tuple[int, float, float, int, str]:
    pages = _payload_page_indices(payload)
    page_idx = preferred_page_idx if preferred_page_idx in pages else (pages[0] if pages else 0)
    bboxes = _payload_bboxes_by_page(payload).get(page_idx, [])
    if bboxes:
        top = min(bbox[1] for bbox in bboxes)
        left = min(bbox[0] for bbox in bboxes if bbox[1] == top) if any(bbox[1] == top for bbox in bboxes) else min(
            bbox[0] for bbox in bboxes
        )
    else:
        top = 10**9
        left = 10**9
    return (
        page_idx,
        top,
        left,
        _as_int(payload.get("chunk_idx"), 10**9),
        str(payload.get("chunk_id", "")),
    )


def _section_path(payload: dict[str, Any]) -> tuple[str, ...]:
    value = payload.get("section_path", [])
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _visual_chunk_alignment_score(chunk_payload: dict[str, Any], visual_payload: dict[str, Any]) -> float:
    """Conservatively distribute a page-level visual hit to chunk candidates.

    This is not a ColPali patch heatmap yet. It uses the indexed visual page's
    chunk pointers plus chunk bbox coverage on that page, so visual evidence is
    kept page-local and cross-page chunks are attenuated by how much of their
    bbox area actually appears on the visual hit page.
    """

    if not visual_payload.get("is_visual_page"):
        return 0.0
    page_idx = _as_int(visual_payload.get("page_idx", visual_payload.get("page_start", 0)))
    if page_idx not in _payload_page_indices(chunk_payload):
        return 0.0

    chunk_id = str(chunk_payload.get("chunk_id", ""))
    raw_chunk_ids = visual_payload.get("chunk_ids_on_page", [])
    chunk_ids_on_page = {str(item) for item in raw_chunk_ids if str(item)}
    listed_on_visual_page = bool(chunk_id and chunk_id in chunk_ids_on_page)
    bboxes_by_page = _payload_bboxes_by_page(chunk_payload)
    has_bbox_on_visual_page = bool(bboxes_by_page.get(page_idx))
    if not listed_on_visual_page and not has_bbox_on_visual_page:
        return 0.0

    area_fraction = _page_bbox_area_fraction(chunk_payload, page_idx)
    if area_fraction <= 0:
        return 0.65 if listed_on_visual_page else 0.5

    base = 1.0 if listed_on_visual_page else 0.75
    # Single-page chunks keep the full page prior; cross-page chunks get a
    # smaller share when only a small part of the chunk is on the visual page.
    return min(1.0, base * (0.5 + 0.5 * area_fraction))


def _visual_chunk_naive_score(chunk_payload: dict[str, Any], visual_payload: dict[str, Any]) -> float:
    if not visual_payload.get("is_visual_page"):
        return 0.0
    page_idx = _as_int(visual_payload.get("page_idx", visual_payload.get("page_start", 0)))
    return 1.0 if page_idx in _payload_page_indices(chunk_payload) else 0.0


def _visual_page_query_filter(models: Any) -> Any:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="is_visual_page",
                match=models.MatchValue(value=True),
            )
        ]
    )


def _candidate_min_score(*, best_score: float, min_candidate_score: float, min_score_ratio: float) -> float:
    """Return the minimum candidate score after final ranking.

    ``min_score_ratio`` is interpreted as an allowed drop from the best score:
    0.2 keeps candidates scoring at least 80% of the best candidate. Use 1.0
    to disable relative filtering for non-negative RRF-style scores.
    """

    allowed_drop = min(max(0.0, min_score_ratio), 1.0)
    relative_min_score = best_score * (1.0 - allowed_drop)
    return max(min_candidate_score, relative_min_score)


def _evidence_bbox(evidence: dict[str, Any]) -> list[float]:
    raw = evidence.get("bbox")
    if not isinstance(raw, list) or len(raw) != 4:
        return []
    try:
        return [float(value) for value in raw]
    except (TypeError, ValueError):
        return []


def _named_vector(record_vector: Any, vector_name: str) -> Any:
    if isinstance(record_vector, dict):
        return record_vector.get(vector_name)
    return record_vector


def _colpali_maxsim_score(query_vectors: Any, page_vectors: Any) -> float | None:
    if not query_vectors or not page_vectors:
        return None

    try:
        import numpy as np

        query_array = np.asarray(query_vectors, dtype=np.float32)
        page_array = np.asarray(page_vectors, dtype=np.float32)
        if query_array.ndim != 2 or page_array.ndim != 2 or query_array.shape[1] != page_array.shape[1]:
            return None
        query_norm = np.linalg.norm(query_array, axis=1, keepdims=True)
        page_norm = np.linalg.norm(page_array, axis=1, keepdims=True)
        query_norm[query_norm == 0] = 1.0
        page_norm[page_norm == 0] = 1.0
        similarities = (query_array / query_norm) @ (page_array / page_norm).T
        return float(similarities.max(axis=1).sum())
    except ImportError:
        pass

    query = [list(map(float, row)) for row in query_vectors if isinstance(row, list)]
    page = [list(map(float, row)) for row in page_vectors if isinstance(row, list)]
    if not query or not page:
        return None
    dims = len(query[0])
    if dims <= 0 or any(len(row) != dims for row in query) or any(len(row) != dims for row in page):
        return None

    normalized_page = []
    for page_row in page:
        norm = sum(value * value for value in page_row) ** 0.5 or 1.0
        normalized_page.append([value / norm for value in page_row])

    score = 0.0
    for query_row in query:
        norm = sum(value * value for value in query_row) ** 0.5 or 1.0
        normalized_query = [value / norm for value in query_row]
        score += max(
            sum(query_value * page_value for query_value, page_value in zip(normalized_query, page_row))
            for page_row in normalized_page
        )
    return score


class RetrievalEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = None
        self.dense_model = None
        self.sparse_model = None
        self.colpali_processor = None
        self.colpali_model = None
        self.device = None
        self._visual_page_cache = None

    def _route_mode(self) -> str:
        mode = (self.config.retrieval.route_mode or "auto").strip().lower()
        if mode == "auto":
            return "visual-bbox" if self.config.retrieval.enable_visual else "text"
        aliases = {
            "dense-only": "dense",
            "sparse-only": "sparse",
            "text-only": "text",
            "visual": "visual-bbox",
            "visual_bbox": "visual-bbox",
            "visual-naive": "visual-naive",
            "visual_naive": "visual-naive",
            "visual-only": "visual-only-bbox",
            "visual_only": "visual-only-bbox",
            "visual-only-bbox": "visual-only-bbox",
            "visual_only_bbox": "visual-only-bbox",
            "visual-only-naive": "visual-only-naive",
            "visual_only_naive": "visual-only-naive",
            "dense-visual": "dense-visual-bbox",
            "dense_visual": "dense-visual-bbox",
            "dense-visual-bbox": "dense-visual-bbox",
            "dense_visual_bbox": "dense-visual-bbox",
            "dense-visual-naive": "dense-visual-naive",
            "dense_visual_naive": "dense-visual-naive",
            "sparse-visual": "sparse-visual-bbox",
            "sparse_visual": "sparse-visual-bbox",
            "sparse-visual-bbox": "sparse-visual-bbox",
            "sparse_visual_bbox": "sparse-visual-bbox",
            "sparse-visual-naive": "sparse-visual-naive",
            "sparse_visual_naive": "sparse-visual-naive",
        }
        return aliases.get(mode, mode)

    def _candidate_mode(self) -> str:
        mode = (self.config.retrieval.candidate_mode or "direct").strip().lower()
        aliases = {
            "auto": "direct",
            "default": "direct",
            "direct-rank": "direct",
            "direct_rank": "direct",
            "page-local-bbox": "visual-page-local-bbox",
            "page_local_bbox": "visual-page-local-bbox",
            "visual_page_local_bbox": "visual-page-local-bbox",
            "page-local-naive": "visual-page-local-naive",
            "page_local_naive": "visual-page-local-naive",
            "visual_page_local_naive": "visual-page-local-naive",
        }
        resolved = aliases.get(mode, mode)
        if resolved not in {"direct", "visual-page-local-bbox", "visual-page-local-naive"}:
            raise ValueError(
                "Unsupported retrieval candidate mode "
                f"'{self.config.retrieval.candidate_mode}'. Seed expansion has been removed; "
                "use 'direct', 'visual-page-local-bbox', or 'visual-page-local-naive'."
            )
        return resolved

    def _uses_dense_route(self) -> bool:
        return self._route_mode() in {
            "dense",
            "text",
            "visual-bbox",
            "visual-naive",
            "dense-visual-bbox",
            "dense-visual-naive",
        }

    def _uses_sparse_route(self) -> bool:
        return self._route_mode() in {
            "sparse",
            "text",
            "visual-bbox",
            "visual-naive",
            "sparse-visual-bbox",
            "sparse-visual-naive",
        }

    def _uses_visual_route(self) -> bool:
        return self.config.retrieval.enable_visual and self._route_mode() in {
            "visual-bbox",
            "visual-naive",
            "visual-only-bbox",
            "visual-only-naive",
            "dense-visual-bbox",
            "dense-visual-naive",
            "sparse-visual-bbox",
            "sparse-visual-naive",
        }

    def load(self) -> None:
        from fastembed import SparseTextEmbedding, TextEmbedding

        self.client = create_qdrant_client(self.config)
        if self._uses_dense_route():
            self.dense_model = TextEmbedding(self.config.models.dense_model)
        if self._uses_sparse_route():
            self.sparse_model = SparseTextEmbedding(self.config.models.sparse_model)
        if not self._uses_visual_route():
            self.device = None
            self.colpali_processor = None
            self.colpali_model = None
            return

        import torch
        from colpali_engine.models import ColPali, ColPaliProcessor
        from transformers import BitsAndBytesConfig

        self.device = get_torch_device(
            feature="ColPali retrieval query encoding",
            preferred=self.config.retrieval.device,
        )
        colpali_model_location = resolve_model_location(
            self.config.models.colpali_model,
            explicit_path=self.config.models.colpali_model_path,
            local_root=self.config.models.colpali_local_model_root,
        )
        self.colpali_processor = ColPaliProcessor.from_pretrained(colpali_model_location)

        kwargs: dict[str, Any] = {"device_map": self.device}
        if self.config.retrieval.quantized_colpali and self.device == "cuda":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.colpali_model = ColPali.from_pretrained(colpali_model_location, **kwargs).eval()

    def retrieve(self, query_text: str) -> RetrievalResult:
        if self.client is None:
            self.load()

        from qdrant_client import models

        collection = self.config.paths.collection_name
        retrieval_k = self.config.retrieval.retrieval_k
        dense_hits = []
        sparse_hits = []
        if self._uses_dense_route():
            dense_query = list(self.dense_model.query_embed(query_text))[0].tolist()
            dense_hits = self.client.query_points(
                collection_name=collection,
                query=dense_query,
                using=TEXT_DENSE_VECTOR_NAME,
                limit=retrieval_k,
            ).points
        if self._uses_sparse_route():
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
            sparse_hits = self.client.query_points(
                collection_name=collection,
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using=TEXT_SPARSE_VECTOR_NAME,
                limit=retrieval_k,
            ).points
        visual_hits = []
        if self._uses_visual_route():
            import torch

            with torch.no_grad():
                batch_query = self.colpali_processor.process_queries([query_text]).to(self.device)
                dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                batch_query = {
                    key: value.to(dtype) if value.is_floating_point() else value
                    for key, value in batch_query.items()
                }
                visual_query = self.colpali_model(**batch_query)[0].cpu().float().tolist()

            visual_hits = self._query_visual_pages_by_scroll(
                collection_name=collection,
                visual_query=visual_query,
                limit=retrieval_k,
                models=models,
            )

        final_ranking = self._compute_rrf(dense_hits, sparse_hits, visual_hits)
        candidate_mode = self._candidate_mode()
        if candidate_mode == "direct":
            scored_candidates = self._direct_rank_candidates(final_ranking)
        elif candidate_mode in {"visual-page-local-bbox", "visual-page-local-naive"}:
            scored_candidates = self._visual_page_local_candidates(
                collection_name=collection,
                final_ranking=final_ranking,
                visual_hits=visual_hits,
                candidate_mode=candidate_mode,
                models=models,
            )
        else:
            raise ValueError(
                f"Unsupported retrieval candidate mode '{candidate_mode}'. "
                "Use 'direct', 'visual-page-local-bbox', or 'visual-page-local-naive'."
            )

        if not scored_candidates:
            context = "No relevant information found in the manual."
            return RetrievalResult(
                hit_page=1,
                all_hits=[],
                context=context,
                final_output=build_final_output(context=context),
            )
        scored_candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                _payload_position_key(item["payload"], item["preferred_page_idx"]),
            )
        )
        best_score = float(scored_candidates[0]["score"])
        min_score = _candidate_min_score(
            best_score=best_score,
            min_candidate_score=float(self.config.retrieval.min_candidate_score),
            min_score_ratio=float(self.config.retrieval.min_score_ratio),
        )
        context_token_budget = max(0, int(self.config.retrieval.max_context_tokens))
        selected_candidates = self._select_context_candidates(
            scored_candidates,
            min_score=min_score,
            context_token_budget=context_token_budget,
        )

        if not selected_candidates:
            context = "No relevant information found in the manual."
            return RetrievalResult(
                hit_page=1,
                all_hits=[],
                context=context,
                final_output=build_final_output(context=context),
            )

        context_blocks: list[str] = []
        hit_details: list[HitDetail] = []
        retrieved_images: list[RetrievedImage] = []
        seen_image_paths: set[str] = set()
        for rank, (candidate, context_block) in enumerate(selected_candidates, start=1):
            payload = candidate["payload"]
            page_idx = int(payload["page_idx"])
            page_start = int(payload.get("page_start", page_idx))
            page_end = int(payload.get("page_end", page_idx))
            page_indices = _payload_page_indices(payload)
            image_references = self._image_references_for_payload(payload, hit_rank=rank)
            for image_reference in image_references:
                if image_reference.image_path in seen_image_paths:
                    continue
                seen_image_paths.add(image_reference.image_path)
                retrieved_images.append(image_reference)
            context_blocks.append(context_block)
            hit_details.append(
                HitDetail(
                    rank=rank,
                    page_idx=page_idx,
                    page_number=page_idx + 1,
                    page_indices=page_indices,
                    page_numbers=[page + 1 for page in page_indices],
                    score=float(candidate["score"]),
                    is_continuation=bool(candidate["is_continuation"]),
                    chunk_id=str(payload.get("chunk_id", "")),
                    visual_page_prior=float(candidate["visual_page_prior"]),
                    visual_alignment_score=float(candidate["visual_alignment_score"]),
                    dense_rrf_score=float(candidate["dense_rrf_score"]),
                    sparse_rrf_score=float(candidate["sparse_rrf_score"]),
                    visual_rrf_score=float(candidate["visual_rrf_score"]),
                    direct_text_rrf_score=float(candidate["direct_text_rrf_score"]),
                    image_references=image_references,
                )
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
            hit_page=hit_details[0].page_number if hit_details else 1,
            all_hits=hit_details,
            context=final_context,
            images=tuple(retrieved_images),
            final_output=build_final_output(
                context=final_context,
                images=tuple(retrieved_images),
                include_images=bool(self.config.retrieval.final_output_images),
            ),
        )

    def _format_context_block(self, candidate: dict[str, Any]) -> str:
        payload = candidate["payload"]
        page_idx = int(payload["page_idx"])
        page_start = int(payload.get("page_start", page_idx))
        page_end = int(payload.get("page_end", page_idx))
        page_label = f"{page_start + 1}" if page_start == page_end else f"{page_start + 1}-{page_end + 1}"
        note_prefix = "[Visual Page Match] " if candidate["visual_alignment_score"] > 0 else ""
        chunk_content = str(payload.get("chunk_content", ""))
        breadcrumb = str(
            payload.get("breadcrumb")
            or source_breadcrumb(_payload_source_relpath(payload), _section_path(payload))
        )
        breadcrumb_line = "" if chunk_content.lstrip().startswith("[Breadcrumb:") else f"[Breadcrumb: {breadcrumb}]\n"
        return (
            f"[Source: {_payload_source_relpath(payload)}, Page: {page_label}]\n"
            f"{note_prefix}{breadcrumb_line}{chunk_content}"
        )

    def _estimate_context_tokens(self, text: str) -> int:
        chars_per_token = max(1.0, float(self.config.retrieval.context_chars_per_token))
        return max(1, math.ceil(len(text) / chars_per_token))

    def _select_context_candidates(
        self,
        scored_candidates: list[dict[str, Any]],
        *,
        min_score: float,
        context_token_budget: int,
    ) -> list[tuple[dict[str, Any], str]]:
        selected_candidates: list[tuple[dict[str, Any], str]] = []
        used_context_tokens = 0
        for candidate in scored_candidates:
            if len(selected_candidates) >= self.config.retrieval.final_top_k:
                break
            if float(candidate["score"]) < min_score:
                continue
            context_block = self._format_context_block(candidate)
            selected_candidates.append((candidate, context_block))
            used_context_tokens += self._estimate_context_tokens(context_block)
            if context_token_budget and used_context_tokens > context_token_budget:
                break
        return selected_candidates

    def _image_base_dir_for_payload(self, payload: dict[str, Any]) -> Path:
        raw_base = str(payload.get("image_base_dir") or "").strip()
        if raw_base:
            return Path(raw_base).expanduser()
        return Path(self.config.paths.base_dir).expanduser()

    def _image_references_for_payload(self, payload: dict[str, Any], *, hit_rank: int) -> tuple[RetrievedImage, ...]:
        raw_evidence = payload.get("image_answering_evidence", [])
        if not isinstance(raw_evidence, list):
            return ()

        base_dir = self._image_base_dir_for_payload(payload)
        source_relpath = _payload_source_relpath(payload)
        chunk_id = str(payload.get("chunk_id", ""))
        references: list[RetrievedImage] = []
        seen: set[str] = set()
        for evidence in raw_evidence:
            if not isinstance(evidence, dict):
                continue
            policy = str(evidence.get("image_answering_policy", "") or "").strip()
            img_path = str(evidence.get("img_path", "") or "").strip()
            if not img_path:
                continue
            image_path = Path(img_path).expanduser()
            if not image_path.is_absolute():
                image_path = base_dir / image_path
            resolved_image_path = str(image_path)
            if resolved_image_path in seen:
                continue
            seen.add(resolved_image_path)
            page_idx = _as_int(evidence.get("page_idx", payload.get("page_idx", 0)))
            references.append(
                RetrievedImage(
                    hit_rank=hit_rank,
                    chunk_id=chunk_id,
                    source_relpath=source_relpath,
                    img_path=img_path,
                    image_path=resolved_image_path,
                    image_exists=image_path.exists(),
                    page_idx=page_idx,
                    page_number=page_idx + 1,
                    bbox=_evidence_bbox(evidence),
                    image_answering_policy=policy,
                    image_answering_confidence=str(evidence.get("image_answering_confidence", "") or ""),
                    image_answering_reason=str(evidence.get("image_answering_reason", "") or ""),
                    image_caption=str(evidence.get("image_caption", "") or ""),
                    image_description_vlm=str(evidence.get("image_description_vlm", "") or ""),
                )
            )
        return tuple(references)

    def _new_candidate(
        self,
        *,
        payload: dict[str, Any],
        preferred_page_idx: int,
        source_pdf: str,
        record_id: Any = "",
    ) -> dict[str, Any]:
        return {
            "payload": payload,
            "direct_score": 0.0,
            "visual_score": 0.0,
            "visual_page_prior": 0.0,
            "visual_alignment_score": 0.0,
            "dense_rrf_score": 0.0,
            "sparse_rrf_score": 0.0,
            "visual_rrf_score": 0.0,
            "direct_text_rrf_score": 0.0,
            "preferred_page_idx": preferred_page_idx,
            "is_continuation": bool(payload.get("is_table_continuation", False)),
            "key": (source_pdf, str(payload.get("chunk_id") or record_id)),
        }

    def _score_candidates(self, candidates: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
        scored_candidates = []
        for candidate in candidates.values():
            candidate["score"] = (
                candidate["direct_score"]
                + candidate["visual_score"]
            )
            scored_candidates.append(candidate)
        return scored_candidates

    def _direct_rank_candidates(self, final_ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        for hit in final_ranking:
            payload = hit["payload"]
            if payload.get("is_visual_page"):
                continue
            source_pdf = _payload_source_relpath(payload)
            preferred_page_idx = _payload_page_indices(payload)[0] if _payload_page_indices(payload) else _as_int(payload.get("page_idx", 0))
            key = _payload_chunk_key(payload)
            candidate = candidates.setdefault(
                key,
                self._new_candidate(
                    payload=payload,
                    preferred_page_idx=preferred_page_idx,
                    source_pdf=source_pdf,
                    record_id=hit.get("id", ""),
                ),
            )
            dense_score = float(hit["routes"].get("dense", 0.0))
            sparse_score = float(hit["routes"].get("sparse", 0.0))
            direct_text_score = dense_score + sparse_score
            candidate["dense_rrf_score"] = max(candidate["dense_rrf_score"], dense_score)
            candidate["sparse_rrf_score"] = max(candidate["sparse_rrf_score"], sparse_score)
            candidate["direct_text_rrf_score"] = max(candidate["direct_text_rrf_score"], direct_text_score)
            candidate["direct_score"] = max(candidate["direct_score"], direct_text_score)
        return self._score_candidates(candidates)

    def _visual_page_local_candidates(
        self,
        *,
        collection_name: str,
        final_ranking: list[dict[str, Any]],
        visual_hits: list[Any],
        candidate_mode: str,
        models: Any,
    ) -> list[dict[str, Any]]:
        candidates: dict[tuple[str, str], dict[str, Any]] = {
            candidate["key"]: candidate for candidate in self._direct_rank_candidates(final_ranking)
        }
        use_naive = candidate_mode == "visual-page-local-naive"
        visual_limit = self.config.retrieval.final_top_k

        for rank, hit in enumerate(visual_hits[:visual_limit]):
            visual_payload = hit.payload
            if not visual_payload.get("is_visual_page"):
                continue
            source_pdf = _payload_source_relpath(visual_payload)
            page_idx = _as_int(visual_payload.get("page_idx", visual_payload.get("page_start", 0)))
            route_score = self.config.retrieval.visual_weight * (
                1.0 / (self.config.retrieval.rrf_k + rank + 1)
            )
            records, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="source_relpath", match=models.MatchValue(value=source_pdf))],
                    should=[
                        models.FieldCondition(key="page_idx", match=models.MatchValue(value=page_idx)),
                        models.FieldCondition(key="page_indices", match=models.MatchValue(value=page_idx)),
                    ],
                ),
                limit=max(1, self.config.retrieval.candidate_scroll_limit),
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                if record.payload.get("is_visual_page"):
                    continue
                if page_idx not in _payload_page_indices(record.payload):
                    continue
                key = _payload_chunk_key(record.payload)
                candidate = candidates.setdefault(
                    key,
                    self._new_candidate(
                        payload=record.payload,
                        preferred_page_idx=page_idx,
                        source_pdf=source_pdf,
                        record_id=record.id,
                    ),
                )
                alignment_score = (
                    _visual_chunk_naive_score(record.payload, visual_payload)
                    if use_naive
                    else _visual_chunk_alignment_score(record.payload, visual_payload)
                )
                candidate["visual_page_prior"] = max(candidate["visual_page_prior"], route_score)
                candidate["visual_alignment_score"] = max(candidate["visual_alignment_score"], alignment_score)
                candidate["visual_rrf_score"] = max(candidate["visual_rrf_score"], route_score)
                candidate["visual_score"] = max(candidate["visual_score"], route_score * alignment_score)
        return self._score_candidates(candidates)

    def _query_visual_pages_by_scroll(
        self,
        *,
        collection_name: str,
        visual_query: list[list[float]],
        limit: int,
        models: Any,
    ) -> list[Any]:
        cached_hits = self._query_cached_visual_pages(
            collection_name=collection_name,
            visual_query=visual_query,
            limit=limit,
            models=models,
        )
        if cached_hits is not None:
            return cached_hits

        scored_hits: list[Any] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=_visual_page_query_filter(models),
                limit=64,
                with_payload=True,
                with_vectors=[PAGE_IMAGE_COLPALI_VECTOR_NAME],
                offset=offset,
            )
            for record in records:
                page_vectors = _named_vector(record.vector, PAGE_IMAGE_COLPALI_VECTOR_NAME)
                score = _colpali_maxsim_score(visual_query, page_vectors)
                if score is None:
                    continue
                scored_hits.append(SimpleNamespace(id=record.id, payload=record.payload, score=score))
            if offset is None:
                break
        scored_hits.sort(key=lambda hit: hit.score, reverse=True)
        return scored_hits[:limit]

    def _query_cached_visual_pages(
        self,
        *,
        collection_name: str,
        visual_query: list[list[float]],
        limit: int,
        models: Any,
    ) -> list[Any] | None:
        try:
            import numpy as np
        except ImportError:
            return None

        cache = self._load_visual_page_cache(collection_name=collection_name, models=models, np=np)
        if not cache or cache["patch_matrix"].size == 0:
            return []

        query_array = np.asarray(visual_query, dtype=np.float32)
        if query_array.ndim != 2 or query_array.shape[1] != cache["patch_matrix"].shape[1]:
            return None
        query_norm = np.linalg.norm(query_array, axis=1, keepdims=True)
        query_norm[query_norm == 0] = 1.0
        query_array = query_array / query_norm

        similarities = query_array @ cache["patch_matrix"].T
        page_count = len(cache["entries"])
        max_by_page = np.full((query_array.shape[0], page_count), -np.inf, dtype=np.float32)
        for query_idx in range(query_array.shape[0]):
            np.maximum.at(max_by_page[query_idx], cache["patch_page_indices"], similarities[query_idx])
        scores = np.where(np.isfinite(max_by_page), max_by_page, 0.0).sum(axis=0)
        if page_count <= limit:
            top_indices = np.argsort(-scores)
        else:
            partial = np.argpartition(-scores, limit - 1)[:limit]
            top_indices = partial[np.argsort(-scores[partial])]
        return [
            SimpleNamespace(
                id=cache["entries"][int(index)]["id"],
                payload=cache["entries"][int(index)]["payload"],
                score=float(scores[int(index)]),
            )
            for index in top_indices[:limit]
        ]

    def _load_visual_page_cache(self, *, collection_name: str, models: Any, np: Any) -> dict[str, Any] | None:
        paths = getattr(self.config, "paths", None)
        cache_key = (
            collection_name,
            getattr(paths, "db_path", None),
            getattr(paths, "collection_name", collection_name),
        )
        if self._visual_page_cache and self._visual_page_cache.get("cache_key") == cache_key:
            return self._visual_page_cache

        entries: list[dict[str, Any]] = []
        patch_matrices = []
        patch_page_indices = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=_visual_page_query_filter(models),
                limit=64,
                with_payload=True,
                with_vectors=[PAGE_IMAGE_COLPALI_VECTOR_NAME],
                offset=offset,
            )
            for record in records:
                page_vectors = _named_vector(record.vector, PAGE_IMAGE_COLPALI_VECTOR_NAME)
                if not page_vectors:
                    continue
                page_array = np.asarray(page_vectors, dtype=np.float32)
                if page_array.ndim != 2 or page_array.shape[1] <= 0:
                    continue
                page_norm = np.linalg.norm(page_array, axis=1, keepdims=True)
                page_norm[page_norm == 0] = 1.0
                page_array = page_array / page_norm
                page_index = len(entries)
                entries.append({"id": record.id, "payload": record.payload})
                patch_matrices.append(page_array)
                patch_page_indices.append(np.full(page_array.shape[0], page_index, dtype=np.int32))
            if offset is None:
                break

        if patch_matrices:
            patch_matrix = np.vstack(patch_matrices)
            patch_indices = np.concatenate(patch_page_indices)
        else:
            patch_matrix = np.empty((0, COLPALI_VECTOR_SIZE), dtype=np.float32)
            patch_indices = np.empty((0,), dtype=np.int32)
        self._visual_page_cache = {
            "cache_key": cache_key,
            "entries": entries,
            "patch_matrix": patch_matrix,
            "patch_page_indices": patch_indices,
        }
        return self._visual_page_cache

    def _compute_rrf(self, dense_res: list[Any], sparse_res: list[Any], visual_res: list[Any]) -> list[dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}

        def add_to_rrf(results: list[Any], weight: float, route: str) -> None:
            for rank, hit in enumerate(results):
                if hit.id not in scores:
                    scores[hit.id] = {"id": hit.id, "score": 0.0, "payload": hit.payload, "routes": {}}
                route_score = weight * (1.0 / (self.config.retrieval.rrf_k + rank + 1))
                scores[hit.id]["score"] += route_score
                scores[hit.id]["routes"][route] = route_score

        add_to_rrf(dense_res, 1.0, "dense")
        add_to_rrf(sparse_res, 1.0, "sparse")
        add_to_rrf(visual_res, self.config.retrieval.visual_weight, "visual")
        return sorted(scores.values(), key=lambda item: item["score"], reverse=True)
