from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TABLE_CONTINUATION_INDICES_KEY = "rag_flow_table_continuation_indices"
TABLE_CONTINUATION_MASTER_IDX_KEY = "rag_flow_table_continuation_master_idx"


@dataclass(frozen=True)
class TableContinuationRegion:
    block_idx: int
    page_idx: int
    bbox: tuple[float, float, float, float]


def join_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def block_page_idx(block: dict[str, Any], *, default: int = 0) -> int:
    try:
        return int(block.get("page_idx", default))
    except (TypeError, ValueError):
        return default


def block_bbox(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = block.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def is_table_continuation_block(block: dict[str, Any]) -> bool:
    if block.get("type") != "table":
        return False
    return (
        not join_text(block.get("table_body", "")).strip()
        and not join_text(block.get("table_caption", "")).strip()
        and not join_text(block.get("table_footnote", "")).strip()
        and not str(block.get("img_path", "")).strip()
    )


def is_table_master_block(block: dict[str, Any]) -> bool:
    return block.get("type") == "table" and bool(join_text(block.get("table_body", "")).strip())


def is_table_continuation_for_master(master: dict[str, Any], block: dict[str, Any]) -> bool:
    if not is_table_continuation_block(block):
        return False

    master_bbox = block_bbox(master)
    continuation_bbox = block_bbox(block)
    if master_bbox is None or continuation_bbox is None:
        return False

    master_page = block_page_idx(master)
    block_page = block_page_idx(block)
    if block_page < master_page:
        return False
    if block_page == master_page:
        return continuation_bbox[1] >= master_bbox[3]
    return continuation_bbox[1] <= 180


def build_table_continuation_map(content_data: list[dict[str, Any]]) -> dict[int, list[int]]:
    continuations: dict[int, list[int]] = {}
    current_master_idx: int | None = None

    for idx, block in enumerate(content_data):
        if not isinstance(block, dict) or block.get("type") != "table":
            continue

        if is_table_master_block(block):
            current_master_idx = idx
            continuations.setdefault(idx, [])
            continue

        if current_master_idx is None:
            continue
        master = content_data[current_master_idx]
        if is_table_continuation_for_master(master, block):
            continuations.setdefault(current_master_idx, []).append(idx)

    return {master_idx: indices for master_idx, indices in continuations.items() if indices}


def table_master_by_continuation(table_continuations: dict[int, list[int]]) -> dict[int, int]:
    return {
        continuation_idx: master_idx
        for master_idx, continuation_indices in table_continuations.items()
        for continuation_idx in continuation_indices
    }


def table_continuation_indices(table_continuations: dict[int, list[int]]) -> set[int]:
    return {idx for indices in table_continuations.values() for idx in indices}


def table_visual_regions(
    content_data: list[dict[str, Any]],
    *,
    master_idx: int,
    continuation_indices: list[int],
) -> tuple[TableContinuationRegion, ...]:
    regions = []
    for idx in [master_idx, *continuation_indices]:
        if idx < 0 or idx >= len(content_data):
            continue
        block = content_data[idx]
        if not isinstance(block, dict):
            continue
        bbox = block_bbox(block)
        if bbox is None:
            continue
        regions.append(
            TableContinuationRegion(
                block_idx=idx,
                page_idx=block_page_idx(block),
                bbox=bbox,
            )
        )
    return tuple(regions)
