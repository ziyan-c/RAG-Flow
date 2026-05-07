from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from io import BytesIO
import json
import math
import re
import shutil
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_flow.config import AppConfig
from rag_flow.table_continuations import (
    TABLE_CONTINUATION_INDICES_KEY,
    TABLE_CONTINUATION_MASTER_IDX_KEY,
    build_table_continuation_map as _build_table_continuation_map,
    is_table_continuation_block as _shared_is_table_continuation_block,
    is_table_continuation_for_master as _shared_is_table_continuation_for_master,
    table_continuation_indices as _shared_table_continuation_indices,
    table_master_by_continuation as _shared_table_master_by_continuation,
)


IGNORE_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "equation",
    "seal",
    "chart",
}

TEXT_FIELD_MAP = {
    "text": ["text"],
    "list": ["list_items"],
    "table": ["table_footnote", "table_body"],
    "image": ["image_caption", "image_footnote"],
    "code": ["code_caption", "code_footnote"],
}

TEXT_FIELD_KEYS = {
    "text",
    "list_items",
    "table_caption",
    "table_footnote",
    "table_body",
    "image_caption",
    "image_footnote",
    "code_caption",
    "code_footnote",
}
NON_PATCH_FIELD_KEYS = {
    "table_caption",
    "image_caption",
    "image_description_vlm",
}

INLINE_ICON_KEY = "vlm-small-icon-inline-icon"
INLINE_ICON_CANDIDATE_KEY = "vlm-small-icon-inline-candidate"
INLINE_ICON_TARGET_IDX_KEY = "vlm-small-icon-inline-target-idx"
INLINE_ICON_TARGET_FIELD_KEY = "vlm-small-icon-inline-target-field"
INLINE_ICON_TARGET_TYPE_KEY = "vlm-small-icon-inline-target-type"
INLINE_ICON_SCORE_KEY = "vlm-small-icon-inline-score"
INLINE_ICON_KEYS = {
    INLINE_ICON_KEY,
    INLINE_ICON_CANDIDATE_KEY,
    INLINE_ICON_TARGET_IDX_KEY,
    INLINE_ICON_TARGET_FIELD_KEY,
    INLINE_ICON_TARGET_TYPE_KEY,
    INLINE_ICON_SCORE_KEY,
}

METADATA_KEYS = {
    "type",
    "bbox",
    "page_idx",
    "text_level",
    "img_path",
    "sub_type",
    "section_path",
    "section_title",
    "section_level",
    "section_source",
    "section_confidence",
    "section_match_score",
    "section_outline_index",
    TABLE_CONTINUATION_INDICES_KEY,
    TABLE_CONTINUATION_MASTER_IDX_KEY,
    *INLINE_ICON_KEYS,
}

CHECKED_FIELDS_KEY = "vlm-small-icon-checked-fields"
PATCHED_FIELDS_KEY = "vlm-small-icon-patched-fields"
NO_MISSING_SENTINEL = "NO_MISSING_SAFE"
BENCHMARK_ORIGINAL_INDEX_KEY = "rag-flow-benchmark-original-index"


@dataclass(frozen=True)
class IconPatchArtifacts:
    artifact_dir: Path
    content_json: Path
    origin_pdf: Path
    output_json: Path


@dataclass
class IconPatchStats:
    blocks_seen: int = 0
    fields_seen: int = 0
    requests_submitted: int = 0
    invalid_retries: int = 0
    invalid_fallbacks: int = 0
    invalid_rejected: int = 0
    checked_count: int = 0
    patched_count: int = 0
    no_missing_count: int = 0
    skipped_ignored_blocks: int = 0
    skipped_no_bbox: int = 0
    skipped_no_fields: int = 0
    skipped_empty_fields: int = 0
    table_continuation_blocks: int = 0
    table_continuation_crops: int = 0
    inline_icon_candidates: int = 0
    inline_icon_linked: int = 0
    inline_icon_unlinked: int = 0
    windows_processed: int = 0
    batches_processed: int = 0
    checkpoints_written: int = 0


@dataclass(frozen=True)
class InlineIconLink:
    icon_idx: int
    target_idx: int
    target_field: str
    target_type: str
    score: float


@dataclass(frozen=True)
class InlineIconLinks:
    by_target: dict[int, list[InlineIconLink]]
    by_icon: dict[int, InlineIconLink]
    candidates: tuple[int, ...]


def _single_candidate(candidates: list[Path], *, label: str, artifact_dir: Path) -> Path:
    if not candidates:
        raise FileNotFoundError(f"Cannot find {label} under MinerU artifact dir: {artifact_dir}")
    if len(candidates) > 1:
        names = ", ".join(str(path.name) for path in candidates)
        raise ValueError(f"Found multiple {label} files under {artifact_dir}: {names}")
    return candidates[0]


def _content_stem(content_json: Path, artifact_dir: Path) -> str:
    if content_json.name.endswith("_content_list_SECTIONED.json"):
        return content_json.name[: -len("_content_list_SECTIONED.json")]
    if content_json.name.endswith("_content_list.json"):
        return content_json.name[: -len("_content_list.json")]
    if content_json.name == "content_list_SECTIONED.json":
        return artifact_dir.name
    if content_json.name == "content_list.json":
        return artifact_dir.name
    return content_json.stem


def _patched_json_path_for(content_json: Path, artifact_dir: Path) -> Path:
    if content_json.name.endswith("_content_list_SECTIONED.json"):
        prefix = content_json.name[: -len("_content_list_SECTIONED.json")]
        return artifact_dir / f"{prefix}_content_list_SECTIONED_PATCHED.json"
    if content_json.name == "content_list_SECTIONED.json":
        return artifact_dir / f"{artifact_dir.name}_content_list_SECTIONED_PATCHED.json"
    stem = _content_stem(content_json, artifact_dir)
    return artifact_dir / f"{stem}_content_list_PATCHED.json"


def resolve_icon_patch_artifacts(
    artifact_dir: str | Path,
    *,
    content_json: str | Path | None = None,
    origin_pdf: str | Path | None = None,
    output_json: str | Path | None = None,
) -> IconPatchArtifacts:
    resolved_dir = Path(artifact_dir).expanduser()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"MinerU artifact dir does not exist: {resolved_dir}")

    if content_json:
        resolved_content = Path(content_json).expanduser()
    else:
        content_candidates = sorted(
            path
            for path in resolved_dir.glob("*_content_list_SECTIONED.json")
            if "PATCHED" not in path.name and "CAPTIONED" not in path.name
        )
        if not content_candidates:
            content_candidates = sorted(path for path in resolved_dir.glob("content_list_SECTIONED.json"))
        if not content_candidates:
            content_candidates = sorted(
                path
                for path in resolved_dir.glob("*_content_list.json")
                if "small-icon" not in path.name and "caption" not in path.name
            )
        if not content_candidates:
            content_candidates = sorted(path for path in resolved_dir.glob("content_list.json"))
        resolved_content = _single_candidate(
            content_candidates,
            label="MinerU content_list JSON",
            artifact_dir=resolved_dir,
        )

    stem = _content_stem(resolved_content, resolved_dir)
    if origin_pdf:
        resolved_pdf = Path(origin_pdf).expanduser()
    else:
        exact_origin = resolved_dir / f"{stem}_origin.pdf"
        if exact_origin.exists():
            resolved_pdf = exact_origin
        else:
            resolved_pdf = _single_candidate(
                sorted(resolved_dir.glob("*_origin.pdf")),
                label="MinerU origin PDF",
                artifact_dir=resolved_dir,
            )

    resolved_output = Path(output_json).expanduser() if output_json else _patched_json_path_for(
        resolved_content,
        resolved_dir,
    )
    return IconPatchArtifacts(
        artifact_dir=resolved_dir,
        content_json=resolved_content,
        origin_pdf=resolved_pdf,
        output_json=resolved_output,
    )


def resolve_icon_patch_batch(
    artifact_dir: str | Path,
    *,
    recursive: bool = True,
) -> list[IconPatchArtifacts]:
    root = Path(artifact_dir).expanduser()
    try:
        return [resolve_icon_patch_artifacts(root)]
    except (FileNotFoundError, ValueError):
        pass

    sectioned_candidates = (
        root.rglob("*_content_list_SECTIONED.json") if recursive else root.glob("*_content_list_SECTIONED.json")
    )
    raw_candidates = root.rglob("*_content_list.json") if recursive else root.glob("*_content_list.json")
    artifacts = []
    seen_dirs: set[Path] = set()
    for content_json in sorted([*sectioned_candidates, *raw_candidates]):
        if "small-icon" in content_json.name or "caption" in content_json.name:
            continue
        if "PATCHED" in content_json.name or "CAPTIONED" in content_json.name:
            continue
        artifact_parent = content_json.parent
        if artifact_parent in seen_dirs:
            continue
        try:
            artifacts.append(resolve_icon_patch_artifacts(artifact_parent, content_json=content_json))
        except FileNotFoundError:
            continue
        seen_dirs.add(artifact_parent)

    if not artifacts:
        raise FileNotFoundError(f"Cannot find MinerU artifact folders under: {root}")
    return artifacts


def checkpoint_path_for(output_json: str | Path) -> Path:
    output = Path(output_json)
    return output.with_name(f"{output.stem}.checkpoint{output.suffix}")


def _checked_fields(block: dict[str, Any]) -> set[str]:
    value = block.get(CHECKED_FIELDS_KEY, [])
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def _mark_checked(block: dict[str, Any], key: str) -> None:
    fields = sorted({*_checked_fields(block), key})
    block[CHECKED_FIELDS_KEY] = fields


def _mark_patched(block: dict[str, Any], key: str) -> None:
    value = block.get(PATCHED_FIELDS_KEY, [])
    fields = set(str(item) for item in value) if isinstance(value, list) else set()
    fields.add(key)
    block[PATCHED_FIELDS_KEY] = sorted(fields)
    block["vlm-small-icon-patched"] = True


def _patch_field_keys(block: dict[str, Any]) -> list[str]:
    block_type = block.get("type")
    if block_type in IGNORE_TYPES or _is_table_continuation_block(block):
        return []

    keys: list[str] = []
    for key in TEXT_FIELD_MAP.get(str(block_type), []):
        if key in NON_PATCH_FIELD_KEYS:
            continue
        if key in block and _join(block.get(key, "")).strip():
            keys.append(key)
    for key, value in block.items():
        if key in NON_PATCH_FIELD_KEYS:
            continue
        if key in keys or key in METADATA_KEYS or key.startswith("vlm-small-icon-"):
            continue
        if (key in TEXT_FIELD_KEYS or isinstance(value, str) or _is_text_list(value)) and _join(value).strip():
            keys.append(key)
    return keys


def _is_text_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def is_inline_icon_block(block: dict[str, Any]) -> bool:
    return bool(block.get(INLINE_ICON_KEY) or block.get(INLINE_ICON_CANDIDATE_KEY))


def _clear_inline_icon_metadata(block: dict[str, Any]) -> None:
    for key in INLINE_ICON_KEYS:
        block.pop(key, None)


def _block_page_idx(block: dict[str, Any], *, default: int = 0) -> int:
    try:
        return int(block.get("page_idx", default))
    except (TypeError, ValueError):
        return default


def _block_bbox(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw_bbox = block.get("bbox")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _bbox_size(bbox: tuple[float, float, float, float]) -> tuple[float, float, float]:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height, width * height


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_contains_point(bbox: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _bbox_overlap_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _bbox_vertical_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    shortest = max(1.0, min(first[3] - first[1], second[3] - second[1]))
    return overlap / shortest


def _bbox_horizontal_gap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    if first[2] < second[0]:
        return second[0] - first[2]
    if second[2] < first[0]:
        return first[0] - second[2]
    return 0.0


def _bbox_distance(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    dx = _bbox_horizontal_gap(first, second)
    if first[3] < second[1]:
        dy = second[1] - first[3]
    elif second[3] < first[1]:
        dy = first[1] - second[3]
    else:
        dy = 0.0
    return math.hypot(dx, dy)


def _bbox_union(
    boxes: list[tuple[float, float, float, float]],
    *,
    padding: float = 0.0,
) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (
        max(0.0, min(box[0] for box in boxes) - padding),
        max(0.0, min(box[1] for box in boxes) - padding),
        min(1000.0, max(box[2] for box in boxes) + padding),
        min(1000.0, max(box[3] for box in boxes) + padding),
    )


def is_inline_icon_candidate(block: dict[str, Any]) -> bool:
    if block.get("type") != "image":
        return False
    if _join(block.get("image_caption", "")).strip() or _join(block.get("image_footnote", "")).strip():
        return False
    bbox = _block_bbox(block)
    if bbox is None:
        return False
    width, height, area = _bbox_size(bbox)
    return (width <= 80 and height <= 80) or area <= 5000


def _table_master_by_continuation(table_continuations: dict[int, list[int]]) -> dict[int, int]:
    return _shared_table_master_by_continuation(table_continuations)


def _primary_inline_target_field(block: dict[str, Any]) -> str | None:
    block_type = block.get("type")
    if block_type == "text" and _join(block.get("text", "")).strip():
        return "text"
    if block_type == "list" and _join(block.get("list_items", "")).strip():
        return "list_items"
    if block_type == "table" and _join(block.get("table_body", "")).strip():
        return "table_body"
    return None


def _target_text_for_field(block: dict[str, Any], field: str) -> str:
    return _join(block.get(field, "")).strip()


def _has_missing_inline_icon_pattern(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:click|select|choose|tap|press|double-click|right-click)\s+(?:[.,;:]|and\b|or\b|to\b)",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\b(?:click|select|choose|tap|press|double-click|right-click)\s*$",
            text,
            flags=re.IGNORECASE,
        )
    )


def _score_inline_target(
    *,
    icon: dict[str, Any],
    target: dict[str, Any],
    target_field: str,
) -> float | None:
    icon_bbox = _block_bbox(icon)
    target_bbox = _block_bbox(target)
    if icon_bbox is None or target_bbox is None:
        return None

    score = _bbox_distance(icon_bbox, target_bbox)
    icon_center = _bbox_center(icon_bbox)
    target_center = _bbox_center(target_bbox)
    score += abs(icon_center[1] - target_center[1]) * 0.10
    vertical_overlap = _bbox_vertical_overlap_ratio(icon_bbox, target_bbox)
    if _bbox_contains_point(target_bbox, icon_center):
        score -= 220
    if vertical_overlap > 0:
        score -= 70
    if vertical_overlap > 0 and _bbox_horizontal_gap(icon_bbox, target_bbox) <= 80:
        score -= 90
    if target.get("type") in {"text", "list"} and _has_missing_inline_icon_pattern(
        _target_text_for_field(target, target_field)
    ):
        score -= 140
    return score


def _accept_inline_link(
    *,
    icon: dict[str, Any],
    target: dict[str, Any],
    target_field: str,
    score: float,
) -> bool:
    icon_bbox = _block_bbox(icon)
    target_bbox = _block_bbox(target)
    if icon_bbox is None or target_bbox is None:
        return False
    distance = _bbox_distance(icon_bbox, target_bbox)
    icon_center = _bbox_center(icon_bbox)
    if _bbox_contains_point(target_bbox, icon_center):
        return True
    if target.get("type") in {"text", "list"} and _has_missing_inline_icon_pattern(
        _target_text_for_field(target, target_field)
    ):
        return distance <= 260
    if (
        _bbox_vertical_overlap_ratio(icon_bbox, target_bbox) > 0
        and _bbox_horizontal_gap(icon_bbox, target_bbox) <= 90
    ):
        return True
    return score <= 130


def _best_inline_icon_link(
    *,
    content_data: list[dict[str, Any]],
    icon_idx: int,
    continuation_to_master: dict[int, int],
) -> InlineIconLink | None:
    icon = content_data[icon_idx]
    icon_bbox = _block_bbox(icon)
    if icon_bbox is None:
        return None
    icon_page = _block_page_idx(icon)
    icon_center = _bbox_center(icon_bbox)
    _, _, icon_area = _bbox_size(icon_bbox)

    for table_idx, table in enumerate(content_data):
        if not isinstance(table, dict) or table.get("type") != "table":
            continue
        if _block_page_idx(table, default=-1) != icon_page:
            continue
        table_bbox = _block_bbox(table)
        if table_bbox is None:
            continue
        overlap_ratio = _bbox_overlap_area(icon_bbox, table_bbox) / max(1.0, icon_area)
        if _bbox_contains_point(table_bbox, icon_center) or overlap_ratio >= 0.5:
            master_idx = continuation_to_master.get(table_idx, table_idx)
            master = content_data[master_idx]
            if _join(master.get("table_body", "")).strip():
                return InlineIconLink(
                    icon_idx=icon_idx,
                    target_idx=master_idx,
                    target_field="table_body",
                    target_type="table",
                    score=-1000 + _bbox_distance(icon_bbox, table_bbox),
                )

    best_link: InlineIconLink | None = None
    for target_idx, target in enumerate(content_data):
        if target_idx == icon_idx or not isinstance(target, dict):
            continue
        if _block_page_idx(target, default=-1) != icon_page:
            continue
        target_field = _primary_inline_target_field(target)
        if not target_field:
            continue
        score = _score_inline_target(icon=icon, target=target, target_field=target_field)
        if score is None:
            continue
        if best_link is None or score < best_link.score:
            best_link = InlineIconLink(
                icon_idx=icon_idx,
                target_idx=target_idx,
                target_field=target_field,
                target_type=str(target.get("type", "")),
                score=score,
            )

    if best_link is None:
        return None
    target = content_data[best_link.target_idx]
    if not _accept_inline_link(
        icon=icon,
        target=target,
        target_field=best_link.target_field,
        score=best_link.score,
    ):
        return None
    return best_link


def build_inline_icon_links(
    content_data: list[dict[str, Any]],
    table_continuations: dict[int, list[int]],
) -> InlineIconLinks:
    for block in content_data:
        if isinstance(block, dict):
            _clear_inline_icon_metadata(block)

    continuation_to_master = _table_master_by_continuation(table_continuations)
    candidates = tuple(
        idx
        for idx, block in enumerate(content_data)
        if isinstance(block, dict) and is_inline_icon_candidate(block)
    )
    by_target: dict[int, list[InlineIconLink]] = defaultdict(list)
    by_icon: dict[int, InlineIconLink] = {}

    for icon_idx in candidates:
        icon = content_data[icon_idx]
        link = _best_inline_icon_link(
            content_data=content_data,
            icon_idx=icon_idx,
            continuation_to_master=continuation_to_master,
        )
        if link is None:
            icon[INLINE_ICON_CANDIDATE_KEY] = True
            continue

        icon[INLINE_ICON_KEY] = True
        icon[INLINE_ICON_TARGET_IDX_KEY] = link.target_idx
        icon[INLINE_ICON_TARGET_FIELD_KEY] = link.target_field
        icon[INLINE_ICON_TARGET_TYPE_KEY] = link.target_type
        icon[INLINE_ICON_SCORE_KEY] = round(link.score, 3)
        by_icon[icon_idx] = link
        by_target[link.target_idx].append(link)

    return InlineIconLinks(
        by_target={target_idx: links for target_idx, links in by_target.items()},
        by_icon=by_icon,
        candidates=candidates,
    )


def _is_table_continuation_block(block: dict[str, Any]) -> bool:
    return _shared_is_table_continuation_block(block)


def _is_table_continuation_for_master(master: dict[str, Any], block: dict[str, Any]) -> bool:
    return _shared_is_table_continuation_for_master(master, block)


def build_table_continuation_map(content_data: list[dict[str, Any]]) -> dict[int, list[int]]:
    return _build_table_continuation_map(content_data)


def _table_continuation_indices(table_continuations: dict[int, list[int]]) -> set[int]:
    return _shared_table_continuation_indices(table_continuations)


def _window_visual_page_end(
    *,
    content_data: list[dict[str, Any]],
    table_continuations: dict[int, list[int]],
    page_start: int,
    page_end: int,
    max_page_idx: int,
) -> int:
    visual_page_end = page_end
    for master_idx, continuation_indices in table_continuations.items():
        master = content_data[master_idx]
        master_page = int(master.get("page_idx", 0))
        if page_start <= master_page <= page_end:
            for continuation_idx in continuation_indices:
                continuation = content_data[continuation_idx]
                visual_page_end = max(visual_page_end, int(continuation.get("page_idx", 0)))
    return min(visual_page_end, max_page_idx)


def _iter_page_windows(
    *,
    content_data: list[dict[str, Any]],
    table_continuations: dict[int, list[int]],
    max_page_idx: int,
    page_window_size: int,
    page_indices: set[int] | None,
) -> Iterator[tuple[int, int, int]]:
    if page_indices is None:
        for page_start in range(0, max_page_idx + 1, max(1, page_window_size)):
            scan_page_end = min(max_page_idx, page_start + max(1, page_window_size) - 1)
            visual_page_end = _window_visual_page_end(
                content_data=content_data,
                table_continuations=table_continuations,
                page_start=page_start,
                page_end=scan_page_end,
                max_page_idx=max_page_idx,
            )
            yield page_start, scan_page_end, visual_page_end
        return

    pages = sorted(page for page in page_indices if 0 <= page <= max_page_idx)
    if not pages:
        return

    max_window = max(1, page_window_size)
    group_start = pages[0]
    previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1 and page - group_start + 1 <= max_window:
            previous = page
            continue
        visual_page_end = _window_visual_page_end(
            content_data=content_data,
            table_continuations=table_continuations,
            page_start=group_start,
            page_end=previous,
            max_page_idx=max_page_idx,
        )
        yield group_start, previous, visual_page_end
        group_start = previous = page

    visual_page_end = _window_visual_page_end(
        content_data=content_data,
        table_continuations=table_continuations,
        page_start=group_start,
        page_end=previous,
        max_page_idx=max_page_idx,
    )
    yield group_start, previous, visual_page_end


def _emit_metric(metrics_sink: Any | None, kind: str, payload: dict[str, Any]) -> None:
    if metrics_sink is None:
        return
    emit = getattr(metrics_sink, "emit", None)
    if emit is None:
        return
    emit(kind, payload)


def _write_json(path: Path, content_data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(content_data, f, ensure_ascii=False, indent=2)


def _strip_checkpoint_fields(content_data: list[dict[str, Any]]) -> None:
    for block in content_data:
        if isinstance(block, dict):
            block.pop(CHECKED_FIELDS_KEY, None)


def _print_stats(stats: IconPatchStats, output_json: Path) -> None:
    print("Icon patching stats:")
    print(f"  blocks seen: {stats.blocks_seen}")
    print(f"  fields seen: {stats.fields_seen}")
    print(f"  requests submitted: {stats.requests_submitted}")
    print(f"  invalid retries: {stats.invalid_retries}")
    print(f"  invalid fallbacks: {stats.invalid_fallbacks}")
    print(f"  invalid rejected: {stats.invalid_rejected}")
    print(f"  checked: {stats.checked_count}")
    print(f"  patched: {stats.patched_count}")
    print(f"  no missing: {stats.no_missing_count}")
    print(f"  skipped ignored blocks: {stats.skipped_ignored_blocks}")
    print(f"  skipped without bbox: {stats.skipped_no_bbox}")
    print(f"  skipped without text fields: {stats.skipped_no_fields}")
    print(f"  skipped empty fields: {stats.skipped_empty_fields}")
    print(f"  table continuation blocks: {stats.table_continuation_blocks}")
    print(f"  table continuation crops: {stats.table_continuation_crops}")
    print(f"  inline icon candidates: {stats.inline_icon_candidates}")
    print(f"  inline icons linked: {stats.inline_icon_linked}")
    print(f"  inline icons unlinked: {stats.inline_icon_unlinked}")
    print(f"  page windows: {stats.windows_processed}")
    print(f"  LLM batches: {stats.batches_processed}")
    print(f"  checkpoints written: {stats.checkpoints_written}")
    print(f"  output: {output_json}")


def _join(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def build_icon_patch_prompt(*, original_text: str, field_key: str) -> str:
    if field_key == "table_body" and "<table" in original_text.lower():
        return (
            "Patch missing small icons in this OCR table.\n"
            "Keep the complete original table content.\n"
            "Insert each missing icon at its missing position.\n"
            "Return only the complete patched table content, including inserted icons.\n"
            "Use `[Icon: name]` for inserted icons.\n"
            f"If no icon is missing, return exactly without any other text or icons: {NO_MISSING_SENTINEL}\n\n"
            f"Original table content:\n{original_text}"
        )

    return (
        "Patch missing small icons in this OCR text.\n"
        "Keep the complete original text.\n"
        "Insert each missing icon at its missing position.\n"
        "Return only the complete patched text, including inserted icons.\n"
        "Use `[Icon: name]` for inserted icons.\n"
        f"If no icon is missing, return exactly without any other text or icons: {NO_MISSING_SENTINEL}\n\n"
        f"Original text:\n{original_text}"
    )


def image_to_data_url(image: Any) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def strip_reasoning_text(text: str) -> str:
    output = text.strip()
    if "</think>" in output:
        output = output.split("</think>")[-1].strip()
    return output


def make_patching_llm_client(*, base_url: str, api_key: str, timeout: float) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def assert_patching_llm_available(client: Any, *, base_url: str) -> None:
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    try:
        client.models.list()
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(
            f"Cannot reach the patching LLM service at {base_url}. "
            "Start it first with `rag-flow serve llm-sglang`."
        ) from exc
    except APIStatusError as exc:
        if exc.status_code in {404, 405}:
            return
        raise RuntimeError(
            f"The patching LLM service at {base_url} is reachable but not ready "
            f"(HTTP {exc.status_code})."
        ) from exc


def request_icon_patch_from_llm(
    *,
    client: Any,
    model: str,
    image: Any,
    prompt: str,
    max_tokens: int,
) -> str:
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError
    except ModuleNotFoundError:
        class _OpenAIUnavailableError(Exception):
            pass

        APIConnectionError = APIStatusError = APITimeoutError = _OpenAIUnavailableError

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "separate_reasoning": True,
            },
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(
            "Cannot reach the patching LLM service. Start it first with "
            "`rag-flow serve llm-sglang`."
        ) from exc
    except APIStatusError as exc:
        raise RuntimeError(f"Patching LLM request failed with HTTP {exc.status_code}: {exc}") from exc

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Patching LLM returned an empty response.")
    return strip_reasoning_text(content)


ICON_MARKER_RE = re.compile(r"\[icon\s*:[^\]]+\]", re.IGNORECASE)


def extract_icon_markers(text: str) -> list[str]:
    markers: list[str] = []
    for match in ICON_MARKER_RE.finditer(text):
        raw = match.group(0)
        name = re.sub(r"^\[icon\s*:\s*", "", raw[:-1], flags=re.IGNORECASE).strip()
        if name:
            markers.append(f"[Icon: {name}]")
    return markers


def _without_icon_markers(text: str) -> str:
    return ICON_MARKER_RE.sub(" ", text)


def is_only_icon_output(*, original_text: str, patched_text: str) -> bool:
    if not original_text.strip() or not re.search(r"\[icon\s*:", patched_text, flags=re.IGNORECASE):
        return False
    remainder = _without_icon_markers(patched_text)
    remainder = re.sub(r"[\s`\"'.:;!,，。；：！/\\|()\[\]{}<>_-]+", "", remainder)
    return not remainder


def invalid_icon_patch_reason(*, original_text: str, patched_text: str, field_key: str) -> str | None:
    if is_no_missing_response(patched_text):
        return None
    if field_key != "table_body" and is_only_icon_output(original_text=original_text, patched_text=patched_text):
        return "only icon output"
    return None


def build_icon_patch_retry_prompt(*, prompt: str, invalid_reason: str) -> str:
    if invalid_reason == "only icon output":
        retry_hint = "Previous answer dropped the original text."
    else:
        retry_hint = "Previous answer did not follow the output rules."
    return f"{prompt}\n\nRetry: {retry_hint} Return only the complete patched content."


def fallback_only_icon_output(*, original_text: str, icon_only_output: str) -> str | None:
    markers = extract_icon_markers(icon_only_output)
    if not original_text.strip() or not markers:
        return None

    icon_text = " ".join(markers)

    before_punctuation = re.search(r"\s+([,.;:!?，。；：！？])", original_text)
    if before_punctuation:
        start, end = before_punctuation.span()
        punctuation = before_punctuation.group(1)
        return f"{original_text[:start]} {icon_text}{punctuation}{original_text[end:]}"

    inner_gap = re.search(r"[ \t]{2,}", original_text)
    if inner_gap:
        start, end = inner_gap.span()
        return f"{original_text[:start]} {icon_text} {original_text[end:]}"

    trailing = original_text[len(original_text.rstrip()) :]
    return f"{original_text.rstrip()} {icon_text}{trailing}"


def request_valid_icon_patch_from_llm(
    *,
    client: Any,
    model: str,
    image: Any,
    prompt: str,
    original_text: str,
    field_key: str,
    max_tokens: int,
    invalid_retry_limit: int,
) -> tuple[str, int, str | None]:
    retry_count = 0
    current_prompt = prompt
    while True:
        output = request_icon_patch_from_llm(
            client=client,
            model=model,
            image=image,
            prompt=current_prompt,
            max_tokens=max_tokens,
        )
        invalid_reason = invalid_icon_patch_reason(
            original_text=original_text,
            patched_text=output,
            field_key=field_key,
        )
        if invalid_reason is None:
            return output, retry_count, None
        if retry_count >= invalid_retry_limit:
            return output, retry_count, invalid_reason
        retry_count += 1
        current_prompt = build_icon_patch_retry_prompt(prompt=prompt, invalid_reason=invalid_reason)


def _request_metric_base(req: dict[str, Any], *, batch_id: int) -> dict[str, Any]:
    image = req.get("image")
    width = getattr(image, "width", None)
    height = getattr(image, "height", None)
    pixels = int(width) * int(height) if isinstance(width, int) and isinstance(height, int) else None
    original_index = req.get("original_idx", req.get("idx"))
    return {
        "batch_id": batch_id,
        "field_id": f"{original_index}:{req.get('key')}",
        "block_idx": req.get("idx"),
        "original_block_idx": original_index,
        "page_idx": req.get("page_idx"),
        "pdf_page": int(req.get("page_idx", 0)) + 1 if req.get("page_idx") is not None else None,
        "block_type": req.get("block_type"),
        "field": req.get("key"),
        "image_width": width,
        "image_height": height,
        "image_pixels": pixels,
        "original_text_chars": len(str(req.get("original_text", ""))),
    }


def _run_icon_patch_request(
    req: dict[str, Any],
    *,
    client: Any,
    model: str,
    max_tokens: int,
    invalid_retry_limit: int,
    batch_id: int,
    metrics_sink: Any | None,
) -> tuple[str, int, str | None, dict[str, Any]]:
    started_at = time.time()
    started_perf = time.perf_counter()
    event = _request_metric_base(req, batch_id=batch_id)
    event["started_at"] = started_at
    try:
        output, retries, invalid_reason = request_valid_icon_patch_from_llm(
            client=client,
            model=model,
            image=req["image"],
            prompt=req["prompt"],
            original_text=req["original_text"],
            field_key=req["key"],
            max_tokens=max_tokens,
            invalid_retry_limit=invalid_retry_limit,
        )
    except Exception as exc:
        ended_at = time.time()
        event.update(
            {
                "ended_at": ended_at,
                "duration_s": time.perf_counter() - started_perf,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "written": False,
            }
        )
        _emit_metric(metrics_sink, "request", event)
        raise

    ended_at = time.time()
    event.update(
        {
            "ended_at": ended_at,
            "duration_s": time.perf_counter() - started_perf,
            "status": "ok",
            "error_type": None,
            "error": None,
            "retries": retries,
            "invalid_reason": invalid_reason,
            "output_chars": len(output),
        }
    )
    return output, retries, invalid_reason, event


def iter_icon_patch_results(
    requests: list[dict[str, Any]],
    *,
    client: Any,
    model: str,
    max_tokens: int,
    concurrency: int,
    invalid_retry_limit: int,
    batch_id: int = 0,
    metrics_sink: Any | None = None,
) -> Iterator[tuple[dict[str, Any], str, int, str | None, dict[str, Any]]]:
    if concurrency <= 1 or len(requests) <= 1:
        for req in requests:
            output, retries, invalid_reason, event = _run_icon_patch_request(
                req,
                client=client,
                model=model,
                max_tokens=max_tokens,
                invalid_retry_limit=invalid_retry_limit,
                batch_id=batch_id,
                metrics_sink=metrics_sink,
            )
            yield req, output, retries, invalid_reason, event
        return

    max_workers = min(concurrency, len(requests))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_request = {
            executor.submit(
                _run_icon_patch_request,
                req,
                client=client,
                model=model,
                max_tokens=max_tokens,
                invalid_retry_limit=invalid_retry_limit,
                batch_id=batch_id,
                metrics_sink=metrics_sink,
            ): req
            for req in requests
        }
        try:
            for future in as_completed(future_to_request):
                output, retries, invalid_reason, event = future.result()
                yield future_to_request[future], output, retries, invalid_reason, event
        except Exception:
            for future in future_to_request:
                future.cancel()
            raise


def is_no_missing_response(text: str) -> bool:
    if NO_MISSING_SENTINEL.lower() in text.lower():
        return True
    normalized = re.sub(r"[\s`\"'.:;!,，。；：！]+", " ", text.strip().lower()).strip()
    return normalized in {"no missing", "no missing icon", "no missing icons"} or normalized.startswith(
        "no missing "
    )


def should_apply_icon_patch(*, original_text: str, patched_text: str, field_key: str) -> bool:
    if is_no_missing_response(patched_text):
        return False
    if invalid_icon_patch_reason(original_text=original_text, patched_text=patched_text, field_key=field_key):
        return False
    return bool(patched_text.strip())


def crop_image_from_bbox(
    *,
    page_idx: int,
    bbox: tuple[float, float, float, float],
    pdf_images: list[Any],
    page_offset: int = 0,
) -> Any | None:
    local_page_idx = page_idx - page_offset
    if local_page_idx < 0 or local_page_idx >= len(pdf_images):
        return None

    image = pdf_images[local_page_idx]
    norm_x0, norm_y0, norm_x1, norm_y1 = bbox
    rx0 = (norm_x0 / 1000.0) * image.width
    ry0 = (norm_y0 / 1000.0) * image.height
    rx1 = (norm_x1 / 1000.0) * image.width
    ry1 = (norm_y1 / 1000.0) * image.height
    rx0, ry0 = max(0, rx0), max(0, ry0)
    rx1, ry1 = min(image.width, rx1), min(image.height, ry1)
    return image.crop((rx0, ry0, rx1, ry1))


def crop_image_from_block(block: dict[str, Any], pdf_images: list[Any], *, page_offset: int = 0) -> Any | None:
    bbox = _block_bbox(block)
    if bbox is None:
        return None
    return crop_image_from_bbox(
        page_idx=_block_page_idx(block),
        bbox=bbox,
        pdf_images=pdf_images,
        page_offset=page_offset,
    )


def crop_image_from_block_with_inline_icons(
    *,
    block: dict[str, Any],
    content_data: list[dict[str, Any]],
    inline_icon_links: list[InlineIconLink],
    pdf_images: list[Any],
    page_offset: int = 0,
) -> Any | None:
    block_bbox = _block_bbox(block)
    if block_bbox is None:
        return None
    page_idx = _block_page_idx(block)
    boxes = [block_bbox]
    for link in inline_icon_links:
        icon = content_data[link.icon_idx]
        if _block_page_idx(icon, default=-1) != page_idx:
            continue
        icon_bbox = _block_bbox(icon)
        if icon_bbox is not None:
            boxes.append(icon_bbox)
    union_bbox = _bbox_union(boxes)
    if union_bbox is None:
        return None
    return crop_image_from_bbox(
        page_idx=page_idx,
        bbox=union_bbox,
        pdf_images=pdf_images,
        page_offset=page_offset,
    )


def concat_images_vertically(images: list[Any]) -> Any | None:
    if not images:
        return None
    if len(images) == 1:
        return images[0]

    from PIL import Image

    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    combined = Image.new("RGB", (width, height), (255, 255, 255))
    y_offset = 0
    for image in images:
        combined.paste(image, (0, y_offset))
        y_offset += image.height
    return combined


def build_table_footnote_crop(
    *,
    content_data: list[dict[str, Any]],
    pdf_images: list[Any],
    block_idx: int,
    page_offset: int = 0,
) -> Any | None:
    block = content_data[block_idx]
    block_bbox = _block_bbox(block)
    if block_bbox is None:
        return None
    last_idx = block_idx
    lookahead_idx = block_idx + 1

    while lookahead_idx < len(content_data):
        next_block = content_data[lookahead_idx]
        next_type = next_block.get("type")
        if next_type in IGNORE_TYPES:
            lookahead_idx += 1
            continue
        if next_type != block.get("type"):
            break
        next_text = _join(next_block.get("table_footnote", "")).strip()
        if next_text:
            break
        last_idx = lookahead_idx
        lookahead_idx += 1

    last_block = content_data[last_idx]
    page_idx = int(last_block.get("page_idx", 0))
    local_page_idx = page_idx - page_offset
    if local_page_idx < 0 or local_page_idx >= len(pdf_images) or "bbox" not in last_block:
        return None

    image = pdf_images[local_page_idx]
    x_padding_norm = 12.0
    x0_norm = max(0.0, block_bbox[0] - x_padding_norm)
    x1_norm = min(1000.0, block_bbox[2] + x_padding_norm)
    y0_norm = last_block["bbox"][3]
    y1_norm = 1000

    for idx in range(last_idx + 1, len(content_data)):
        next_block = content_data[idx]
        if next_block.get("page_idx") != page_idx:
            break
        if next_block.get("type") not in {"header", "footer", "page_number"} and "bbox" in next_block:
            next_y0 = next_block["bbox"][1]
            if next_y0 > y0_norm:
                y1_norm = next_y0
                break

    rx0 = (x0_norm / 1000.0) * image.width
    ry0 = (y0_norm / 1000.0) * image.height
    rx1 = (x1_norm / 1000.0) * image.width
    ry1 = (y1_norm / 1000.0) * image.height
    min_height = int(image.width / 190.0) + 1
    if (ry1 - ry0) < min_height:
        ry1 = ry0 + min_height
        if ry1 > image.height:
            ry1 = image.height
            ry0 = max(0, image.height - min_height)
    return image.crop((rx0, ry0, rx1, ry1))


def build_table_body_crop(
    *,
    content_data: list[dict[str, Any]],
    pdf_images: list[Any],
    block_idx: int,
    continuation_indices: list[int],
    inline_icon_links: list[InlineIconLink] | None = None,
    page_offset: int = 0,
) -> Any | None:
    crops = []
    block = content_data[block_idx]
    block_crop = crop_image_from_block_with_inline_icons(
        block=block,
        content_data=content_data,
        inline_icon_links=inline_icon_links or [],
        pdf_images=pdf_images,
        page_offset=page_offset,
    )
    if block_crop is not None:
        crops.append(block_crop)

    for continuation_idx in continuation_indices:
        continuation = content_data[continuation_idx]
        continuation_crop = crop_image_from_block_with_inline_icons(
            block=continuation,
            content_data=content_data,
            inline_icon_links=inline_icon_links or [],
            pdf_images=pdf_images,
            page_offset=page_offset,
        )
        if continuation_crop is not None:
            crops.append(continuation_crop)

    return concat_images_vertically(crops)


def add_small_icon_text(
    *,
    input_json: str | Path,
    output_json: str | Path,
    pdf_path: str | Path,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
    dpi: int = 250,
    batch_size: int = 16,
    max_new_tokens: int = 8000,
    llm_timeout: float = 120.0,
    page_window_size: int = 200,
    checkpoint_interval: int = 10,
    invalid_retry_limit: int = 0,
    concurrency: int = 3,
    checkpoint_json: str | Path | None = None,
    resume: bool = True,
    write_patching_view: bool = True,
    patching_view_pdf: str | Path | None = None,
    page_indices: set[int] | None = None,
    metrics_sink: Any | None = None,
) -> None:
    try:
        from pdf2image import convert_from_path
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        missing = exc.name or "required patching dependency"
        raise RuntimeError(
            f"Missing Python package for patching: {missing}. "
            "Run `rag-flow env create-pipeline`, then retry `rag-flow patch`."
        ) from exc

    missing_poppler = [name for name in ("pdfinfo", "pdftoppm") if shutil.which(name) is None]
    if missing_poppler:
        raise RuntimeError(
            "Missing PDF rendering command(s): "
            f"{', '.join(missing_poppler)}. Install Poppler, for example "
            "`apt-get install -y poppler-utils`, or run `rag-flow init china-all` "
            "after updating the project."
        )

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if page_window_size < 1:
        raise ValueError("page_window_size must be >= 1.")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be >= 0.")
    if invalid_retry_limit < 0:
        raise ValueError("invalid_retry_limit must be >= 0.")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1.")

    llm_client = make_patching_llm_client(
        base_url=llm_base_url,
        api_key=llm_api_key,
        timeout=llm_timeout,
    )
    assert_patching_llm_available(llm_client, base_url=llm_base_url)

    input_path = Path(input_json)
    output_path = Path(output_json)
    checkpoint_path = Path(checkpoint_json) if checkpoint_json else checkpoint_path_for(output_path)
    if resume and checkpoint_path.exists():
        print(f"Resuming icon patching from checkpoint: {checkpoint_path}")
        source_json = checkpoint_path
    else:
        source_json = input_path

    with source_json.open("r", encoding="utf-8") as f:
        content_data: list[dict[str, Any]] = json.load(f)

    stats = IconPatchStats()
    max_page_idx = max(
        (int(block.get("page_idx", 0)) for block in content_data if isinstance(block, dict) and "page_idx" in block),
        default=-1,
    )
    table_continuations = build_table_continuation_map(content_data)
    table_continuation_indices = _table_continuation_indices(table_continuations)
    stats.table_continuation_blocks = len(table_continuation_indices)
    inline_icon_links = build_inline_icon_links(content_data, table_continuations)
    stats.inline_icon_candidates = len(inline_icon_links.candidates)
    stats.inline_icon_linked = len(inline_icon_links.by_icon)
    stats.inline_icon_unlinked = stats.inline_icon_candidates - stats.inline_icon_linked

    def write_checkpoint(reason: str) -> None:
        started_at = time.time()
        started_perf = time.perf_counter()
        _write_json(checkpoint_path, content_data)
        stats.checkpoints_written += 1
        _emit_metric(
            metrics_sink,
            "checkpoint",
            {
                "reason": reason,
                "path": str(checkpoint_path),
                "checkpoint_index": stats.checkpoints_written,
                "batches_processed": stats.batches_processed,
                "requests_submitted": stats.requests_submitted,
                "started_at": started_at,
                "ended_at": time.time(),
                "duration_s": time.perf_counter() - started_perf,
                "file_size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
            },
        )

    def process_batch(requests: list[dict[str, Any]]) -> None:
        if not requests:
            return
        stats.requests_submitted += len(requests)
        stats.batches_processed += 1

        batch_id = stats.batches_processed
        for req, output, retries, invalid_reason, request_event in iter_icon_patch_results(
            requests,
            client=llm_client,
            model=llm_model,
            max_tokens=max_new_tokens,
            concurrency=concurrency,
            invalid_retry_limit=invalid_retry_limit,
            batch_id=batch_id,
            metrics_sink=metrics_sink,
        ):
            stats.invalid_retries += retries
            idx = req["idx"]
            key = req["key"]
            block = content_data[idx]
            _mark_checked(block, key)
            stats.checked_count += 1
            decision = "unchanged"
            written = False
            if invalid_reason is not None:
                fallback_output = None
                if invalid_reason == "only icon output":
                    fallback_output = fallback_only_icon_output(
                        original_text=req["original_text"],
                        icon_only_output=output,
                    )
                if fallback_output and should_apply_icon_patch(
                    original_text=req["original_text"],
                    patched_text=fallback_output,
                    field_key=req["key"],
                ):
                    output = fallback_output
                    stats.invalid_fallbacks += 1
                    decision = "fallback"
                else:
                    stats.invalid_rejected += 1
                    request_event.update({"decision": "invalid_rejected", "written": False})
                    _emit_metric(metrics_sink, "request", request_event)
                    continue
            if is_no_missing_response(output):
                stats.no_missing_count += 1
                request_event.update({"decision": "no_missing", "written": False})
                _emit_metric(metrics_sink, "request", request_event)
                continue

            if should_apply_icon_patch(
                original_text=req["original_text"],
                patched_text=output,
                field_key=req["key"],
            ):
                content_data[idx][key] = output.split("\n") if req["is_list"] else output
                _mark_patched(content_data[idx], key)
                stats.patched_count += 1
                decision = "patched" if decision == "unchanged" else f"{decision}_patched"
                written = True

            request_event.update({"decision": decision, "written": written})
            _emit_metric(metrics_sink, "request", request_event)

        gc.collect()
        if checkpoint_interval > 0 and stats.batches_processed % checkpoint_interval == 0:
            write_checkpoint("interval")

    try:
        if max_page_idx < 0:
            _strip_checkpoint_fields(content_data)
            _write_json(output_path, content_data)
            _print_stats(stats, output_path)
            return

        for page_start, scan_page_end, visual_page_end in _iter_page_windows(
            content_data=content_data,
            table_continuations=table_continuations,
            max_page_idx=max_page_idx,
            page_window_size=page_window_size,
            page_indices=page_indices,
        ):
            render_started_at = time.time()
            render_started_perf = time.perf_counter()
            pdf_images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                first_page=page_start + 1,
                last_page=visual_page_end + 1,
            )
            stats.windows_processed += 1
            _emit_metric(
                metrics_sink,
                "render",
                {
                    "window_index": stats.windows_processed,
                    "page_start_idx": page_start,
                    "scan_page_end_idx": scan_page_end,
                    "visual_page_end_idx": visual_page_end,
                    "first_pdf_page": page_start + 1,
                    "last_scan_pdf_page": scan_page_end + 1,
                    "last_visual_pdf_page": visual_page_end + 1,
                    "pages_rendered": len(pdf_images),
                    "dpi": dpi,
                    "started_at": render_started_at,
                    "ended_at": time.time(),
                    "duration_s": time.perf_counter() - render_started_perf,
                },
            )

            batch: list[dict[str, Any]] = []
            for idx in tqdm(range(len(content_data)), desc=f"Scanning pages {page_start + 1}-{scan_page_end + 1}"):
                block = content_data[idx]
                if not isinstance(block, dict):
                    continue
                page_idx = int(block.get("page_idx", 0))
                if page_idx < page_start or page_idx > scan_page_end:
                    continue
                stats.blocks_seen += 1
                if block.get("type") in IGNORE_TYPES or idx in table_continuation_indices:
                    stats.skipped_ignored_blocks += 1
                    continue
                if "bbox" not in block:
                    stats.skipped_no_bbox += 1
                    continue

                field_keys = _patch_field_keys(block)
                if not field_keys:
                    stats.skipped_no_fields += 1
                    continue

                checked_fields = _checked_fields(block)
                for key in field_keys:
                    if key in checked_fields:
                        continue
                    stats.fields_seen += 1
                    value = block.get(key, "")
                    is_list = isinstance(value, list)
                    original_text = _join(value).strip()
                    if not original_text:
                        _mark_checked(block, key)
                        stats.skipped_empty_fields += 1
                        continue

                    if key == "table_body":
                        continuation_indices = table_continuations.get(idx, [])
                        final_image = build_table_body_crop(
                            content_data=content_data,
                            pdf_images=pdf_images,
                            block_idx=idx,
                            continuation_indices=continuation_indices,
                            inline_icon_links=inline_icon_links.by_target.get(idx, []),
                            page_offset=page_start,
                        )
                        stats.table_continuation_crops += len(continuation_indices)
                    elif key == "table_footnote":
                        final_image = build_table_footnote_crop(
                            content_data=content_data,
                            pdf_images=pdf_images,
                            block_idx=idx,
                            page_offset=page_start,
                        )
                    else:
                        final_image = crop_image_from_block_with_inline_icons(
                            block=block,
                            content_data=content_data,
                            inline_icon_links=[
                                link
                                for link in inline_icon_links.by_target.get(idx, [])
                                if link.target_field == key
                            ],
                            pdf_images=pdf_images,
                            page_offset=page_start,
                        )
                    if final_image is None:
                        continue

                    prompt = build_icon_patch_prompt(original_text=original_text, field_key=key)
                    batch.append(
                        {
                            "idx": idx,
                            "key": key,
                            "block_type": block.get("type"),
                            "page_idx": page_idx,
                            "original_idx": block.get(BENCHMARK_ORIGINAL_INDEX_KEY, idx),
                            "original_text": original_text,
                            "is_list": is_list,
                            "image": final_image,
                            "prompt": prompt,
                        }
                    )
                    if len(batch) >= batch_size:
                        process_batch(batch)
                        batch = []

            if batch:
                process_batch(batch)

            del pdf_images
            gc.collect()
            write_checkpoint("page_window")

    except Exception:
        write_checkpoint("failure")
        print(f"Icon patching checkpoint saved before failure: {checkpoint_path}")
        raise

    _strip_checkpoint_fields(content_data)
    _write_json(output_path, content_data)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if write_patching_view:
        from rag_flow.preprocessing.patching_view import write_patching_view_pdf

        view_stats = write_patching_view_pdf(
            content_json=output_path,
            pdf_path=pdf_path,
            output_pdf=patching_view_pdf,
        )
        print(f"Generated patching view PDF at {view_stats.output_pdf}")
        print(f"  overlays: {view_stats.region_count}")
        print(f"  inline icons linked: {view_stats.inline_icons_linked}/{view_stats.inline_icon_candidates}")
    _print_stats(stats, output_path)


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(
        description="Use a local OpenAI-compatible vision LLM to patch small icon text missing from MinerU JSON."
    )
    parser.add_argument("--artifact-dir", help="MinerU output folder containing *_content_list.json and *_origin.pdf.")
    parser.add_argument("--input", default=None, help="Input MinerU content_list JSON.")
    parser.add_argument("--output", default=None, help="Output patched content_list JSON.")
    parser.add_argument("--pdf", default=None, help="PDF used for bbox crops. Defaults to *_origin.pdf in --artifact-dir.")
    parser.add_argument("--model", "--llm-model", dest="llm_model", default=config.models.llm_model)
    parser.add_argument("--llm-base-url", default=config.models.llm_base_url)
    parser.add_argument("--api-key", default=config.models.llm_api_key)
    parser.add_argument("--dpi", type=int, default=config.patching.dpi)
    parser.add_argument("--batch-size", type=int, default=config.patching.batch_size)
    parser.add_argument("--concurrency", type=int, default=config.patching.concurrency)
    parser.add_argument("--max-new-tokens", type=int, default=config.patching.max_new_tokens)
    parser.add_argument("--request-timeout", type=float, default=config.patching.llm_timeout)
    parser.add_argument("--page-window-size", type=int, default=config.patching.page_window_size)
    parser.add_argument("--checkpoint-interval", type=int, default=config.patching.checkpoint_interval)
    parser.add_argument("--invalid-retry-limit", type=int, default=config.patching.invalid_retry_limit)
    parser.add_argument("--checkpoint-json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--patching-view-pdf", help="Output PDF that visualizes patching LLM crop regions.")
    parser.add_argument("--no-patching-view", action="store_true", help="Do not write the PATCHING_VIEW PDF.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved inputs without calling the LLM.")
    args = parser.parse_args(argv)

    if args.artifact_dir:
        if args.input or args.pdf or args.output:
            artifacts_list = [
                resolve_icon_patch_artifacts(
                    args.artifact_dir,
                    content_json=args.input,
                    origin_pdf=args.pdf,
                    output_json=args.output,
                )
            ]
        else:
            artifacts_list = resolve_icon_patch_batch(args.artifact_dir, recursive=not args.no_recursive)
    else:
        artifacts_list = [
            IconPatchArtifacts(
                artifact_dir=config.paths.base_dir,
                content_json=Path(args.input).expanduser() if args.input else config.paths.content_json,
                output_json=Path(args.output).expanduser() if args.output else config.paths.patched_json,
                origin_pdf=Path(args.pdf).expanduser() if args.pdf else config.paths.source_pdf,
            )
        ]

    if len(artifacts_list) > 1 and args.checkpoint_json:
        parser.error("--checkpoint-json can only be used with a single patching job.")
    if len(artifacts_list) > 1 and args.patching_view_pdf:
        parser.error("--patching-view-pdf can only be used with a single patching job.")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1.")
    if args.page_window_size < 1:
        parser.error("--page-window-size must be >= 1.")
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint-interval must be >= 0.")
    if args.invalid_retry_limit < 0:
        parser.error("--invalid-retry-limit must be >= 0.")

    if args.dry_run:
        from rag_flow.preprocessing.patching_view import patching_view_path_for

        print(f"Icon patching jobs: {len(artifacts_list)}")
        for artifacts in artifacts_list:
            print("Icon patching inputs:")
            print(f"  artifact_dir: {artifacts.artifact_dir}")
            print(f"  input_json: {artifacts.content_json}")
            print(f"  pdf: {artifacts.origin_pdf}")
            print(f"  output_json: {artifacts.output_json}")
            if args.no_patching_view:
                print("  patching_view_pdf: disabled")
            else:
                print(f"  patching_view_pdf: {args.patching_view_pdf or patching_view_path_for(artifacts.output_json)}")
            print(f"  checkpoint_json: {args.checkpoint_json or checkpoint_path_for(artifacts.output_json)}")
            print(f"  dpi: {args.dpi}")
            print(f"  page_window_size: {args.page_window_size}")
            print(f"  batch_size: {args.batch_size}")
            print(f"  concurrency: {args.concurrency}")
            print(f"  checkpoint_interval: {args.checkpoint_interval}")
            print(f"  invalid_retry_limit: {args.invalid_retry_limit}")
            print(f"  max_new_tokens: {args.max_new_tokens}")
            print(f"  llm_base_url: {args.llm_base_url}")
            print(f"  llm_model: {args.llm_model}")
            print(f"  request_timeout: {args.request_timeout}")
        return

    for job_idx, artifacts in enumerate(artifacts_list, start=1):
        print(f"Icon patching job {job_idx}/{len(artifacts_list)}: {artifacts.artifact_dir}")
        add_small_icon_text(
            input_json=artifacts.content_json,
            output_json=artifacts.output_json,
            pdf_path=artifacts.origin_pdf,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.api_key,
            llm_model=args.llm_model,
            dpi=args.dpi,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
            llm_timeout=args.request_timeout,
            page_window_size=args.page_window_size,
            checkpoint_interval=args.checkpoint_interval,
            invalid_retry_limit=args.invalid_retry_limit,
            checkpoint_json=args.checkpoint_json,
            resume=not args.no_resume,
            write_patching_view=not args.no_patching_view,
            patching_view_pdf=args.patching_view_pdf,
        )


if __name__ == "__main__":
    main()
