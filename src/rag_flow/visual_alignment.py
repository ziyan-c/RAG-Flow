from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Sequence

PAGE_COORD_SIZE = 1000.0
EPSILON = 1e-9

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class PatchEvidence:
    token_idx: int
    patch_idx: int
    score: float
    bbox: BBox


@dataclass(frozen=True)
class ChunkAlignment:
    chunk_id: str
    chunk_idx: int | None
    page_idx: int
    chunk_score: float
    token_coverage: float
    density_score: float
    raw_score: float
    area_fraction: float
    matched_tokens: int
    evidence_count: int
    metadata: dict[str, Any]

    @property
    def sort_key(self) -> tuple[float, float, float]:
        return (self.chunk_score, self.token_coverage, self.density_score)


def bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def intersection_area(left: BBox, right: BBox) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def patch_bbox_for_index(patch_idx: int, *, grid_width: int, grid_height: int) -> BBox:
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError("grid_width and grid_height must be positive")
    if patch_idx < 0 or patch_idx >= grid_width * grid_height:
        raise ValueError(f"patch_idx {patch_idx} is outside {grid_width}x{grid_height} grid")
    row, col = divmod(patch_idx, grid_width)
    cell_width = PAGE_COORD_SIZE / grid_width
    cell_height = PAGE_COORD_SIZE / grid_height
    return (
        col * cell_width,
        row * cell_height,
        (col + 1) * cell_width,
        (row + 1) * cell_height,
    )


def cosine_similarity_matrix(
    query_vectors: Sequence[Sequence[float]],
    patch_vectors: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    if not query_vectors or not patch_vectors:
        return ()
    dimension = len(query_vectors[0])
    if dimension <= 0:
        raise ValueError("embedding vectors must not be empty")
    patch_norms = []
    for patch_idx, vector in enumerate(patch_vectors):
        if len(vector) != dimension:
            raise ValueError(f"patch vector {patch_idx} has {len(vector)} dims; expected {dimension}")
        patch_norms.append(sqrt(sum(float(value) * float(value) for value in vector)))

    rows = []
    for token_idx, query_vector in enumerate(query_vectors):
        if len(query_vector) != dimension:
            raise ValueError(f"query vector {token_idx} has {len(query_vector)} dims; expected {dimension}")
        query_norm = sqrt(sum(float(value) * float(value) for value in query_vector))
        row = []
        for patch_vector, patch_norm in zip(patch_vectors, patch_norms):
            denom = query_norm * patch_norm
            if denom <= EPSILON:
                row.append(0.0)
                continue
            dot = sum(float(left) * float(right) for left, right in zip(query_vector, patch_vector))
            row.append(dot / denom)
        rows.append(tuple(row))
    return tuple(rows)


def token_topk_patch_evidence(
    similarity: Sequence[Sequence[float]],
    *,
    grid_width: int,
    grid_height: int,
    top_k: int = 5,
    min_score: float | None = None,
) -> tuple[PatchEvidence, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    evidence = []
    patch_count = grid_width * grid_height
    for token_idx, row in enumerate(similarity):
        if len(row) != patch_count:
            raise ValueError(
                f"similarity row {token_idx} has {len(row)} patches; expected {patch_count}"
            )
        ranked = sorted(enumerate(row), key=lambda item: float(item[1]), reverse=True)
        for patch_idx, score in ranked[:top_k]:
            score = float(score)
            if min_score is not None and score < min_score:
                continue
            evidence.append(
                PatchEvidence(
                    token_idx=token_idx,
                    patch_idx=patch_idx,
                    score=score,
                    bbox=patch_bbox_for_index(
                        patch_idx,
                        grid_width=grid_width,
                        grid_height=grid_height,
                    ),
                )
            )
    return tuple(evidence)


def _chunk_page_bboxes(metadata: dict[str, Any], page_idx: int) -> tuple[BBox, ...]:
    raw = metadata.get("bboxes_by_page", {})
    if not isinstance(raw, dict):
        return ()
    page_bboxes = raw.get(str(page_idx), raw.get(page_idx, []))
    if not isinstance(page_bboxes, list):
        return ()
    bboxes = []
    for bbox in page_bboxes:
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if x1 > x0 and y1 > y0:
            bboxes.append((x0, y0, x1, y1))
    return tuple(bboxes)


def _patch_overlap_fraction(patch_bbox: BBox, chunk_bboxes: tuple[BBox, ...]) -> float:
    patch_area = bbox_area(patch_bbox)
    if patch_area <= 0:
        return 0.0
    # Use max overlap so overlapping block boxes inside one chunk do not double count the same patch.
    overlap = max((intersection_area(patch_bbox, bbox) for bbox in chunk_bboxes), default=0.0)
    return min(1.0, overlap / patch_area)


def _chunk_area_fraction(chunk_bboxes: tuple[BBox, ...]) -> float:
    # A simple upper-bound area estimate. It is intentionally conservative for density tie-breaking.
    area = sum(bbox_area(bbox) for bbox in chunk_bboxes)
    return min(1.0, area / (PAGE_COORD_SIZE * PAGE_COORD_SIZE))


def score_chunk_alignment(
    chunk: dict[str, Any],
    evidence: Sequence[PatchEvidence],
    *,
    page_idx: int,
    total_query_tokens: int,
) -> ChunkAlignment | None:
    metadata = dict(chunk.get("metadata", {}))
    chunk_bboxes = _chunk_page_bboxes(metadata, page_idx)
    if not chunk_bboxes:
        return None

    total_score = sum(max(0.0, item.score) for item in evidence)
    if total_score <= 0:
        return None

    raw_score = 0.0
    matched_tokens: set[int] = set()
    evidence_count = 0
    for item in evidence:
        overlap = _patch_overlap_fraction(item.bbox, chunk_bboxes)
        if overlap <= 0:
            continue
        raw_score += max(0.0, item.score) * overlap
        matched_tokens.add(item.token_idx)
        evidence_count += 1

    if raw_score <= 0:
        return None

    area_fraction = _chunk_area_fraction(chunk_bboxes)
    chunk_score = raw_score / total_score
    token_coverage = len(matched_tokens) / max(1, total_query_tokens)
    density_score = chunk_score / max(area_fraction, EPSILON)
    chunk_idx = metadata.get("chunk_idx")
    try:
        chunk_idx = int(chunk_idx) if chunk_idx is not None else None
    except (TypeError, ValueError):
        chunk_idx = None
    return ChunkAlignment(
        chunk_id=str(metadata.get("chunk_id", chunk_idx if chunk_idx is not None else "")),
        chunk_idx=chunk_idx,
        page_idx=page_idx,
        chunk_score=chunk_score,
        token_coverage=token_coverage,
        density_score=density_score,
        raw_score=raw_score,
        area_fraction=area_fraction,
        matched_tokens=len(matched_tokens),
        evidence_count=evidence_count,
        metadata=metadata,
    )


def rank_chunk_alignments(
    chunks: Sequence[dict[str, Any]],
    evidence: Sequence[PatchEvidence],
    *,
    page_idx: int,
    total_query_tokens: int,
) -> list[ChunkAlignment]:
    alignments = [
        alignment
        for chunk in chunks
        if (alignment := score_chunk_alignment(
            chunk,
            evidence,
            page_idx=page_idx,
            total_query_tokens=total_query_tokens,
        ))
        is not None
    ]
    return sorted(
        alignments,
        key=lambda item: (
            item.chunk_score,
            item.token_coverage,
            item.density_score,
            -(item.chunk_idx if item.chunk_idx is not None else 10**9),
        ),
        reverse=True,
    )
