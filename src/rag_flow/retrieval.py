from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .config import AppConfig
from .indexing import PAGE_IMAGE_COLPALI_VECTOR_NAME, TEXT_DENSE_VECTOR_NAME, TEXT_SPARSE_VECTOR_NAME
from .model_paths import resolve_model_location
from .runtime import get_torch_device


SECTION_EXACT_BONUS = 0.02
SECTION_RELATED_BONUS = 0.01
PAGE_SAME_BONUS = 0.02
PAGE_NEAR_BONUS = 0.01
PAGE_FAR_BONUS = 0.005


@dataclass(frozen=True)
class HitDetail:
    rank: int
    page_idx: int
    page_number: int
    score: float
    is_continuation: bool
    chunk_id: str = ""
    visual_page_prior: float = 0.0
    visual_alignment_score: float = 0.0
    section_bonus: float = 0.0
    page_bonus: float = 0.0
    dense_rrf_score: float = 0.0
    sparse_rrf_score: float = 0.0
    visual_rrf_score: float = 0.0
    direct_text_rrf_score: float = 0.0
    is_visual_seed: bool = False
    seed_page_idx: int = 0
    seed_source_route: str = ""
    candidate_page_distance: int = 0


@dataclass(frozen=True)
class RetrievalResult:
    hit_page: int
    all_hits: list[HitDetail]
    context: str


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _payload_chunk_key(payload: dict[str, Any]) -> tuple[str, str]:
    source = str(payload.get("source", ""))
    chunk_id = payload.get("chunk_id")
    if chunk_id is None:
        chunk_id = payload.get("id")
    if chunk_id is None:
        chunk_id = f"chunk_idx:{payload.get('chunk_idx', '')}"
    return source, str(chunk_id)


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


def _page_distance(payload: dict[str, Any], target_page_idx: int) -> int:
    pages = _payload_page_indices(payload)
    if not pages:
        return 10**9
    return min(abs(page - target_page_idx) for page in pages)


def _page_proximity_bonus(payload: dict[str, Any], target_page_idx: int) -> float:
    distance = _page_distance(payload, target_page_idx)
    if distance == 0:
        return PAGE_SAME_BONUS
    if distance == 1:
        return PAGE_NEAR_BONUS
    if distance == 2:
        return PAGE_FAR_BONUS
    return 0.0


def _section_path(payload: dict[str, Any]) -> tuple[str, ...]:
    value = payload.get("section_path", [])
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _section_proximity_bonus(payload: dict[str, Any], reference_payload: dict[str, Any]) -> float:
    left = _section_path(payload)
    right = _section_path(reference_payload)
    if not left or not right:
        return 0.0
    if left == right:
        return SECTION_EXACT_BONUS
    prefix_len = min(len(left), len(right))
    if left[:prefix_len] == right[:prefix_len]:
        return SECTION_RELATED_BONUS
    return 0.0


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
        }
        return aliases.get(mode, mode)

    def _candidate_mode(self) -> str:
        mode = (self.config.retrieval.candidate_mode or "seed").strip().lower()
        aliases = {
            "auto": "seed",
            "default": "seed",
            "seed-expansion": "seed",
            "seed_expansion": "seed",
            "direct-rank": "direct",
            "direct_rank": "direct",
            "no-seed": "direct",
            "no_seed": "direct",
            "page-local-bbox": "visual-page-local-bbox",
            "page_local_bbox": "visual-page-local-bbox",
            "visual_page_local_bbox": "visual-page-local-bbox",
            "page-local-naive": "visual-page-local-naive",
            "page_local_naive": "visual-page-local-naive",
            "visual_page_local_naive": "visual-page-local-naive",
            "current-visual-seed": "visual-seed",
            "visual_seed": "visual-seed",
        }
        return aliases.get(mode, mode)

    def _uses_dense_route(self) -> bool:
        return self._route_mode() in {"dense", "text", "visual-bbox", "visual-naive"}

    def _uses_sparse_route(self) -> bool:
        return self._route_mode() in {"sparse", "text", "visual-bbox", "visual-naive"}

    def _uses_visual_route(self) -> bool:
        return self.config.retrieval.enable_visual and self._route_mode() in {"visual-bbox", "visual-naive"}

    def load(self) -> None:
        from fastembed import SparseTextEmbedding, TextEmbedding
        from qdrant_client import QdrantClient

        self.client = QdrantClient(path=str(self.config.paths.db_path))
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

        import torch
        from qdrant_client import models

        collection = self.config.paths.collection_name
        retrieval_k = self.config.retrieval.retrieval_k
        route_mode = self._route_mode()
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
            top_hits = self._candidate_seed_hits(final_ranking, visual_hits)
            if not top_hits:
                return RetrievalResult(
                    hit_page=1,
                    all_hits=[],
                    context="No relevant information found in the manual.",
                )
            scored_candidates = self._seed_expansion_candidates(
                collection_name=collection,
                top_hits=top_hits,
                route_mode=route_mode,
                models=models,
            )

        if not scored_candidates:
            return RetrievalResult(
                hit_page=1,
                all_hits=[],
                context="No relevant information found in the manual.",
            )
        scored_candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                _payload_position_key(item["payload"], item["seed_page_idx"]),
            )
        )
        selected_candidates = scored_candidates[: self.config.retrieval.final_top_k]

        context_blocks: list[str] = []
        hit_details: list[HitDetail] = []
        for rank, candidate in enumerate(selected_candidates, start=1):
            payload = candidate["payload"]
            page_idx = int(payload["page_idx"])
            page_start = int(payload.get("page_start", page_idx))
            page_end = int(payload.get("page_end", page_idx))
            page_label = f"{page_start + 1}" if page_start == page_end else f"{page_start + 1}-{page_end + 1}"
            section = payload.get("section_title")
            section_line = f", Section: {section}" if section else ""
            note_prefix = ""
            if candidate["visual_alignment_score"] > 0:
                note_prefix = "[Visual Page Match] "
            context_blocks.append(
                f"[Source: {payload.get('source', '')}, Page: {page_label}{section_line}]\n"
                f"{note_prefix}{payload.get('chunk_content', '')}"
            )
            hit_details.append(
                HitDetail(
                    rank=rank,
                    page_idx=page_idx,
                    page_number=page_idx + 1,
                    score=float(candidate["score"]),
                    is_continuation=bool(candidate["is_continuation"]),
                    chunk_id=str(payload.get("chunk_id", "")),
                    visual_page_prior=float(candidate["visual_page_prior"]),
                    visual_alignment_score=float(candidate["visual_alignment_score"]),
                    section_bonus=float(candidate["section_bonus"]),
                    page_bonus=float(candidate["page_bonus"]),
                    dense_rrf_score=float(candidate["dense_rrf_score"]),
                    sparse_rrf_score=float(candidate["sparse_rrf_score"]),
                    visual_rrf_score=float(candidate["visual_rrf_score"]),
                    direct_text_rrf_score=float(candidate["direct_text_rrf_score"]),
                    is_visual_seed=bool(candidate["is_visual_seed"]),
                    seed_page_idx=int(candidate["seed_page_idx"]),
                    seed_source_route="|".join(sorted(candidate["seed_source_routes"])),
                    candidate_page_distance=int(candidate["candidate_page_distance"]),
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
        )

    def _new_candidate(
        self,
        *,
        payload: dict[str, Any],
        seed_page_idx: int,
        source_pdf: str,
        record_id: Any = "",
    ) -> dict[str, Any]:
        return {
            "payload": payload,
            "direct_score": 0.0,
            "visual_score": 0.0,
            "visual_page_prior": 0.0,
            "visual_alignment_score": 0.0,
            "section_bonus": 0.0,
            "page_bonus": 0.0,
            "dense_rrf_score": 0.0,
            "sparse_rrf_score": 0.0,
            "visual_rrf_score": 0.0,
            "direct_text_rrf_score": 0.0,
            "is_visual_seed": False,
            "seed_page_idx": seed_page_idx,
            "seed_source_routes": set(),
            "candidate_page_distance": _page_distance(payload, seed_page_idx),
            "is_continuation": bool(payload.get("is_table_continuation", False)),
            "key": (source_pdf, str(payload.get("chunk_id") or record_id)),
        }

    def _score_candidates(self, candidates: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
        scored_candidates = []
        for candidate in candidates.values():
            candidate["score"] = (
                candidate["direct_score"]
                + candidate["visual_score"]
                + candidate["section_bonus"]
                + candidate["page_bonus"]
            )
            scored_candidates.append(candidate)
        return scored_candidates

    def _direct_rank_candidates(self, final_ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        for hit in final_ranking:
            payload = hit["payload"]
            if payload.get("is_visual_page"):
                continue
            source_pdf = str(payload.get("source", ""))
            seed_page_idx = _payload_page_indices(payload)[0] if _payload_page_indices(payload) else _as_int(payload.get("page_idx", 0))
            key = _payload_chunk_key(payload)
            candidate = candidates.setdefault(
                key,
                self._new_candidate(
                    payload=payload,
                    seed_page_idx=seed_page_idx,
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
            candidate["seed_source_routes"].update(route for route in ("dense", "sparse") if hit["routes"].get(route))
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
        seed_limit = self.config.retrieval.seed_k or self.config.retrieval.final_top_k

        for rank, hit in enumerate(visual_hits[:seed_limit]):
            visual_payload = hit.payload
            if not visual_payload.get("is_visual_page"):
                continue
            source_pdf = str(visual_payload.get("source", ""))
            page_idx = _as_int(visual_payload.get("page_idx", visual_payload.get("page_start", 0)))
            route_score = self.config.retrieval.visual_weight * (
                1.0 / (self.config.retrieval.rrf_k + rank + 1)
            )
            records, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchValue(value=source_pdf))],
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
                        seed_page_idx=page_idx,
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
                if alignment_score > 0:
                    candidate["is_visual_seed"] = True
                    candidate["seed_source_routes"].add("visual")
                candidate["candidate_page_distance"] = min(
                    candidate["candidate_page_distance"],
                    _page_distance(record.payload, page_idx),
                )
        return self._score_candidates(candidates)

    def _seed_expansion_candidates(
        self,
        *,
        collection_name: str,
        top_hits: list[dict[str, Any]],
        route_mode: str,
        models: Any,
    ) -> list[dict[str, Any]]:
        candidates: dict[tuple[str, str], dict[str, Any]] = {}

        for rank, hit in enumerate(top_hits):
            payload = hit["payload"]
            original_hit_page = int(payload["page_idx"])
            source_pdf = payload["source"]

            logical_center_page = int(payload.get("parent_page_idx", original_hit_page))
            should_conditions = [
                models.FieldCondition(key="page_idx", match=models.MatchValue(value=logical_center_page)),
                models.FieldCondition(key="page_indices", match=models.MatchValue(value=logical_center_page)),
                models.FieldCondition(key="parent_page_idx", match=models.MatchValue(value=logical_center_page)),
            ]
            neighbor_window = max(0, self.config.retrieval.neighbor_window)
            if rank == 0:
                for offset in [*range(-neighbor_window, 0), *range(1, neighbor_window + 1)]:
                    page = logical_center_page + offset
                    should_conditions.append(
                        models.FieldCondition(key="page_idx", match=models.MatchValue(value=page))
                    )
                    should_conditions.append(
                        models.FieldCondition(key="page_indices", match=models.MatchValue(value=page))
                    )
            elif rank <= 2:
                secondary_window = max(0, neighbor_window - 1)
                for offset in [*range(-secondary_window, 0), *range(1, secondary_window + 1)]:
                    page = logical_center_page + offset
                    should_conditions.append(
                        models.FieldCondition(key="page_idx", match=models.MatchValue(value=page))
                    )
                    should_conditions.append(
                        models.FieldCondition(key="page_indices", match=models.MatchValue(value=page))
                    )

            records, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="source", match=models.MatchValue(value=source_pdf))],
                    should=should_conditions,
                ),
                limit=max(1, self.config.retrieval.candidate_scroll_limit),
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                if record.payload.get("is_visual_page"):
                    continue
                page_idx = int(record.payload["page_idx"])
                key = _payload_chunk_key(record.payload)
                if page_idx < 0:
                    continue
                candidate = candidates.setdefault(
                    key,
                    self._new_candidate(
                        payload=record.payload,
                        seed_page_idx=logical_center_page,
                        source_pdf=source_pdf,
                        record_id=record.id,
                    ),
                )
                if not payload.get("is_visual_page") and _payload_chunk_key(record.payload) == _payload_chunk_key(payload):
                    dense_score = float(hit["routes"].get("dense", 0.0))
                    sparse_score = float(hit["routes"].get("sparse", 0.0))
                    direct_text_score = dense_score + sparse_score
                    candidate["dense_rrf_score"] = max(candidate["dense_rrf_score"], dense_score)
                    candidate["sparse_rrf_score"] = max(candidate["sparse_rrf_score"], sparse_score)
                    candidate["direct_text_rrf_score"] = max(candidate["direct_text_rrf_score"], direct_text_score)
                    candidate["direct_score"] = max(candidate["direct_score"], direct_text_score)
                    candidate["seed_source_routes"].update(route for route in ("dense", "sparse") if hit["routes"].get(route))
                if payload.get("is_visual_page"):
                    visual_page_prior = float(hit["routes"].get("visual", hit["score"]))
                    alignment_score = (
                        _visual_chunk_naive_score(record.payload, payload)
                        if route_mode == "visual-naive"
                        else _visual_chunk_alignment_score(record.payload, payload)
                    )
                    candidate["visual_page_prior"] = max(candidate["visual_page_prior"], visual_page_prior)
                    candidate["visual_alignment_score"] = max(candidate["visual_alignment_score"], alignment_score)
                    candidate["visual_rrf_score"] = max(candidate["visual_rrf_score"], visual_page_prior)
                    candidate["visual_score"] = max(
                        candidate["visual_score"],
                        visual_page_prior * alignment_score,
                    )
                    if alignment_score > 0:
                        candidate["is_visual_seed"] = True
                        candidate["seed_source_routes"].add("visual")
                candidate["section_bonus"] = max(
                    candidate["section_bonus"],
                    _section_proximity_bonus(record.payload, payload)
                    * self.config.retrieval.section_bonus_scale,
                )
                candidate["page_bonus"] = max(
                    candidate["page_bonus"],
                    _page_proximity_bonus(record.payload, logical_center_page)
                    * self.config.retrieval.page_bonus_scale,
                )
                candidate["candidate_page_distance"] = min(
                    candidate["candidate_page_distance"],
                    _page_distance(record.payload, logical_center_page),
                )
                if _page_distance(record.payload, logical_center_page) < _page_distance(
                    candidate["payload"], candidate["seed_page_idx"]
                ):
                    candidate["seed_page_idx"] = logical_center_page

        return self._score_candidates(candidates)

    def _query_visual_pages_by_scroll(
        self,
        *,
        collection_name: str,
        visual_query: list[list[float]],
        limit: int,
        models: Any,
    ) -> list[Any]:
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

    def _candidate_seed_hits(self, final_ranking: list[dict[str, Any]], visual_hits: list[Any]) -> list[dict[str, Any]]:
        seed_limit = self.config.retrieval.seed_k or self.config.retrieval.final_top_k
        seeds: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add_seed(seed: dict[str, Any]) -> None:
            seed_id = str(seed.get("id") or _payload_chunk_key(seed["payload"]))
            if seed_id in seen:
                return
            seen.add(seed_id)
            seeds.append(seed)

        for seed in final_ranking[:seed_limit]:
            add_seed(seed)

        for rank, hit in enumerate(visual_hits[:seed_limit]):
            route_score = self.config.retrieval.visual_weight * (
                1.0 / (self.config.retrieval.rrf_k + rank + 1)
            )
            add_seed(
                {
                    "id": hit.id,
                    "score": route_score,
                    "payload": hit.payload,
                    "routes": {"visual": route_score},
                }
            )

        return seeds

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
