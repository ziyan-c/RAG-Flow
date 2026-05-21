from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .source_paths import source_breadcrumb, source_payload_fields
from .table_continuations import (
    build_table_continuation_map,
    table_continuation_indices,
    table_visual_regions,
)

INLINE_ICON_KEYS = ("vlm-small-icon-inline-icon", "vlm-small-icon-inline-candidate")
IGNORED_BLOCK_TYPES = {"header", "footer", "page_number"}
SUPPORTED_CHUNK_MODES = ("auto", "section", "token", "page")


@dataclass(frozen=True)
class VisualRegion:
    block_idx: int
    page_idx: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ChunkItem:
    text: str
    page_idx: int
    block_idx: int
    bbox: tuple[float, float, float, float] | None = None
    visual_regions: tuple[VisualRegion, ...] = ()
    images: tuple[str, ...] = ()
    image_answering_evidence: tuple[dict[str, Any], ...] = ()
    tables: tuple[str, ...] = ()
    section_path: tuple[str, ...] = ()
    section_level: int | None = None
    section_source: str = ""
    table_continuation_block_indices: tuple[int, ...] = ()
    source_relpath: str = ""
    source_filename: str = ""
    breadcrumb: str = ""


def _join_field(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def _has_inline_icon_marker(block: dict[str, Any]) -> bool:
    return any(block.get(key) for key in INLINE_ICON_KEYS)


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def estimate_token_count(text: str) -> int:
    """Small dependency-free token estimate for chunk budgeting."""
    if not text:
        return 0
    cjk_chars = re.findall(r"[\u3400-\u9fff]", text)
    other = re.sub(r"[\u3400-\u9fff]", " ", text)
    words = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", other)
    return len(cjk_chars) + len(words)


def _section_path(block: dict[str, Any]) -> tuple[str, ...]:
    value = block.get("section_path", [])
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _block_page_idx(block: dict[str, Any]) -> int:
    try:
        return int(block.get("page_idx", 0))
    except (TypeError, ValueError):
        return 0


def _block_bbox(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
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


def _source_fields_for_block(block: dict[str, Any], fallback_source_name: str) -> dict[str, str]:
    source_relpath = str(block.get("source_relpath", "") or "").strip()
    source_filename = str(block.get("source_filename", "") or "").strip()
    fields = source_payload_fields(source_relpath or fallback_source_name)
    if source_filename:
        fields["source_filename"] = source_filename
    return fields


def _source_fields_for_items(source_name: str, items: list[ChunkItem]) -> dict[str, str]:
    for item in items:
        if item.source_relpath:
            fields = source_payload_fields(item.source_relpath)
            if item.source_filename:
                fields["source_filename"] = item.source_filename
            return fields
    return source_payload_fields(source_name)


def _breadcrumb_for_items(
    source_name: str,
    items: list[ChunkItem],
    section_path: tuple[str, ...],
) -> str:
    if section_path:
        for item in items:
            if item.breadcrumb and item.section_path == section_path:
                return item.breadcrumb
    for item in items:
        if item.breadcrumb:
            return item.breadcrumb
    source_fields = _source_fields_for_items(source_name, items)
    return source_breadcrumb(source_fields["source_relpath"], section_path)


def _visual_region_for_block(block: dict[str, Any], block_idx: int) -> tuple[VisualRegion, ...]:
    bbox = _block_bbox(block)
    if bbox is None:
        return ()
    return (
        VisualRegion(
            block_idx=block_idx,
            page_idx=_block_page_idx(block),
            bbox=bbox,
        ),
    )


def _image_answering_evidence_for_block(
    block: dict[str, Any],
    block_idx: int,
    *,
    page_idx: int,
    bbox: tuple[float, float, float, float] | None,
    caption: str,
) -> dict[str, Any] | None:
    if not block.get("img_path") or _has_inline_icon_marker(block):
        return None
    evidence: dict[str, Any] = {
        "img_path": str(block["img_path"]),
        "block_idx": block_idx,
        "page_idx": page_idx,
    }
    if bbox is not None:
        evidence["bbox"] = [round(value, 3) for value in bbox]
    if caption:
        evidence["image_caption"] = caption
    for key in (
        "image_answering_policy",
        "image_answering_confidence",
        "image_answering_reason",
    ):
        value = str(block.get(key, "") or "").strip()
        if value:
            evidence[key] = value
    return evidence


def _block_text_item(
    block: dict[str, Any],
    block_idx: int,
    *,
    content_data: list[dict[str, Any]],
    table_continuations: dict[int, list[int]],
) -> ChunkItem | None:
    page_idx = _block_page_idx(block)
    bbox = _block_bbox(block)
    visual_regions = _visual_region_for_block(block, block_idx)
    block_type = block.get("type")
    section_path = _section_path(block)
    section_level = block.get("section_level")
    try:
        section_level = int(section_level) if section_level is not None else None
    except (TypeError, ValueError):
        section_level = None
    section_source = str(block.get("section_source", "") or "")
    source_relpath = str(block.get("source_relpath", "") or "").strip()
    source_filename = str(block.get("source_filename", "") or "").strip()
    breadcrumb = str(block.get("breadcrumb", "") or "").strip()

    if block_type in IGNORED_BLOCK_TYPES:
        return None

    if block_type == "image":
        description = str(block.get("image_description_vlm", "")).strip()
        caption = _join_field(block.get("image_caption", [])).strip()
        footnote = _join_field(block.get("image_footnote", [])).strip()
        parts = []
        if description or caption:
            label = f"[Image with illustration: {caption}]" if caption else "[Image with illustration]"
            parts.append(f"{label}\n{description}".strip())
        if footnote:
            parts.append(f"[Image footnote: {footnote}]")
        text = "\n".join(parts).strip()
        images = (str(block["img_path"]),) if block.get("img_path") and not _has_inline_icon_marker(block) else ()
        evidence = _image_answering_evidence_for_block(
            block,
            block_idx,
            page_idx=page_idx,
            bbox=bbox,
            caption=caption,
        )
        image_answering_evidence = (evidence,) if evidence else ()
        if not text and not images:
            return None
        return ChunkItem(
            text=text,
            page_idx=page_idx,
            block_idx=block_idx,
            bbox=bbox,
            visual_regions=visual_regions,
            images=images,
            image_answering_evidence=image_answering_evidence,
            section_path=section_path,
            section_level=section_level,
            section_source=section_source,
            source_relpath=source_relpath,
            source_filename=source_filename,
            breadcrumb=breadcrumb,
        )

    if block_type == "table":
        caption = _join_field(block.get("table_caption", [])).strip()
        body = _join_field(block.get("table_body", [])).strip()
        footnote = _join_field(block.get("table_footnote", [])).strip()
        parts = []
        if caption:
            parts.append(f"[Table: {caption}]")
        if body:
            parts.append(body)
        if footnote:
            parts.append(f"[Footnote: {footnote}]")
        text = "\n".join(parts).strip()
        tables = (str(block["img_path"]),) if block.get("img_path") else ()
        if not text and not tables:
            return None
        continuation_indices = tuple(table_continuations.get(block_idx, []))
        table_regions = table_visual_regions(
            content_data,
            master_idx=block_idx,
            continuation_indices=list(continuation_indices),
        )
        visual_regions = tuple(
            VisualRegion(region.block_idx, region.page_idx, region.bbox) for region in table_regions
        )
        return ChunkItem(
            text=text,
            page_idx=page_idx,
            block_idx=block_idx,
            bbox=bbox,
            visual_regions=visual_regions,
            tables=tables,
            section_path=section_path,
            section_level=section_level,
            section_source=section_source,
            table_continuation_block_indices=continuation_indices,
            source_relpath=source_relpath,
            source_filename=source_filename,
            breadcrumb=breadcrumb,
        )

    if block_type in {"text", "list"}:
        key = "list_items" if block_type == "list" else "text"
        text = _join_field(block.get(key, [])).strip()
        if not text:
            return None
        return ChunkItem(
            text=text,
            page_idx=page_idx,
            block_idx=block_idx,
            bbox=bbox,
            visual_regions=visual_regions,
            section_path=section_path,
            section_level=section_level,
            section_source=section_source,
            source_relpath=source_relpath,
            source_filename=source_filename,
            breadcrumb=breadcrumb,
        )

    return None


def content_items(content_data: list[dict[str, Any]]) -> list[ChunkItem]:
    items = []
    table_continuations = build_table_continuation_map(content_data)
    continuation_indices = table_continuation_indices(table_continuations)
    for idx, block in enumerate(content_data):
        if not isinstance(block, dict):
            continue
        if idx in continuation_indices:
            continue
        item = _block_text_item(
            block,
            idx,
            content_data=content_data,
            table_continuations=table_continuations,
        )
        if item is not None:
            items.append(item)
    return items


def _tail_overlap(items: list[ChunkItem], overlap_tokens: int) -> list[ChunkItem]:
    if overlap_tokens <= 0:
        return []
    selected = []
    total = 0
    for item in reversed(items):
        selected.append(item)
        total += estimate_token_count(item.text)
        if total >= overlap_tokens:
            break
    return list(reversed(selected))


def _chunk_metadata(
    *,
    source_name: str,
    chunk_idx: int,
    mode: str,
    items: list[ChunkItem],
    token_count: int,
    section_path: tuple[str, ...] = (),
    section_level: int | None = None,
    section_source: str = "",
) -> dict[str, Any]:
    pages = sorted(
        {item.page_idx for item in items}
        | {region.page_idx for item in items for region in item.visual_regions}
    )
    page_start = pages[0] if pages else 0
    page_end = pages[-1] if pages else page_start
    images = _unique([image for item in items for image in item.images])
    image_answering_evidence = [
        evidence
        for item in items
        for evidence in item.image_answering_evidence
        if evidence.get("img_path")
    ]
    tables = _unique([table for item in items for table in item.tables])
    bboxes_by_page: dict[str, list[list[float]]] = defaultdict(list)
    block_indices: list[int] = []
    table_continuations = []
    for item in items:
        if item.block_idx not in block_indices:
            block_indices.append(item.block_idx)
        for continuation_idx in item.table_continuation_block_indices:
            if continuation_idx not in block_indices:
                block_indices.append(continuation_idx)
        if item.table_continuation_block_indices:
            table_continuations.append(
                {
                    "master_block_idx": item.block_idx,
                    "continuation_block_indices": list(item.table_continuation_block_indices),
                    "continuation_page_indices": sorted(
                        {
                            region.page_idx
                            for region in item.visual_regions
                            if region.block_idx in item.table_continuation_block_indices
                        }
                    ),
                }
            )
        for region in item.visual_regions:
            bboxes_by_page[str(region.page_idx)].append([round(value, 3) for value in region.bbox])
    source_fields = _source_fields_for_items(source_name, items)
    metadata: dict[str, Any] = {
        **source_fields,
        "chunk_idx": chunk_idx,
        "chunk_id": f"{Path(source_name).stem}-chunk-{chunk_idx:05d}",
        "chunk_mode": mode,
        "token_count": token_count,
        "block_indices": block_indices,
        "bboxes_by_page": dict(bboxes_by_page),
        "page_idx": page_start,
        "page_start": page_start,
        "page_end": page_end,
        "page_indices": pages,
        "images_on_page": images,
        "tables_on_page": tables,
    }
    metadata["breadcrumb"] = _breadcrumb_for_items(source_name, items, section_path)
    if image_answering_evidence:
        metadata["image_answering_evidence"] = image_answering_evidence
    if table_continuations:
        metadata["table_continuations"] = table_continuations
    if section_path:
        metadata["section_path"] = list(section_path)
        metadata["section_title"] = section_path[-1]
        if section_level is not None:
            metadata["section_level"] = section_level
        if section_source:
            metadata["section_source"] = section_source
    return metadata


def _make_chunk(
    *,
    source_name: str,
    chunk_idx: int,
    mode: str,
    items: list[ChunkItem],
    section_path: tuple[str, ...] = (),
    section_level: int | None = None,
    section_source: str = "",
) -> dict[str, Any]:
    breadcrumb = _breadcrumb_for_items(source_name, items, section_path)
    parts = [f"[Breadcrumb: {breadcrumb}]"]
    if section_path:
        heading = " > ".join(section_path)
        parts.append(f"[Section: {heading}]")
    parts.extend(item.text for item in items if item.text)
    chunk_content = "\n\n".join(parts).strip()
    token_count = estimate_token_count(chunk_content)
    return {
        "chunk_content": chunk_content,
        "metadata": _chunk_metadata(
            source_name=source_name,
            chunk_idx=chunk_idx,
            mode=mode,
            items=items,
            token_count=token_count,
            section_path=section_path,
            section_level=section_level,
            section_source=section_source,
        ),
    }


def _token_window_chunks(
    items: list[ChunkItem],
    source_name: str,
    *,
    mode: str,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    start_chunk_idx: int = 0,
    section_path: tuple[str, ...] = (),
    section_level: int | None = None,
    section_source: str = "",
) -> list[dict[str, Any]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")
    overlap_tokens = min(overlap_tokens, max(0, max_tokens - 1))
    chunks = []
    current: list[ChunkItem] = []
    current_tokens = 0
    chunk_idx = start_chunk_idx

    for item in items:
        item_tokens = max(1, estimate_token_count(item.text))
        should_flush = current and current_tokens + item_tokens > max_tokens and current_tokens >= min_tokens
        if should_flush:
            chunk = _make_chunk(
                source_name=source_name,
                chunk_idx=chunk_idx,
                mode=mode,
                items=current,
                section_path=section_path,
                section_level=section_level,
                section_source=section_source,
            )
            if chunk["chunk_content"]:
                chunks.append(chunk)
                chunk_idx += 1
            current = _tail_overlap(current, overlap_tokens)
            current_tokens = sum(max(1, estimate_token_count(overlap.text)) for overlap in current)

        current.append(item)
        current_tokens += item_tokens

    if current:
        chunk = _make_chunk(
            source_name=source_name,
            chunk_idx=chunk_idx,
            mode=mode,
            items=current,
            section_path=section_path,
            section_level=section_level,
            section_source=section_source,
        )
        if chunk["chunk_content"]:
            chunks.append(chunk)
    return chunks


def _section_window_chunks(
    items: list[ChunkItem],
    source_name: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
) -> list[dict[str, Any]]:
    chunks = []
    group: list[ChunkItem] = []
    current_path: tuple[str, ...] = ()
    current_level: int | None = None
    current_source = ""

    def flush() -> None:
        nonlocal chunks, group, current_path, current_level, current_source
        if not group:
            return
        chunks.extend(
            _token_window_chunks(
                group,
                source_name,
                mode="section",
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                min_tokens=min_tokens,
                start_chunk_idx=len(chunks),
                section_path=current_path,
                section_level=current_level,
                section_source=current_source,
            )
        )
        group = []

    for item in items:
        item_path = item.section_path or current_path
        if not current_path:
            current_path = item_path
            current_level = item.section_level
            current_source = item.section_source
        elif item_path != current_path:
            flush()
            current_path = item_path
            current_level = item.section_level
            current_source = item.section_source
        group.append(item)
    flush()
    return chunks


def create_page_level_chunks(
    json_path: str | Path,
    source_name: str,
) -> list[dict[str, Any]]:
    with Path(json_path).open("r", encoding="utf-8") as f:
        content_data = json.load(f)

    chunk_contents_by_page: dict[int, list[str]] = defaultdict(list)
    page_images: dict[int, list[str]] = defaultdict(list)
    page_image_answering_evidence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    page_tables: dict[int, list[str]] = defaultdict(list)
    page_block_indices: dict[int, list[int]] = defaultdict(list)
    page_bboxes: dict[int, list[list[float]]] = defaultdict(list)
    page_sections: dict[int, tuple[tuple[str, ...], int | None, str]] = {}
    page_source_fields: dict[int, dict[str, str]] = {}
    page_breadcrumbs: dict[int, str] = {}
    table_continuations = build_table_continuation_map(content_data)
    continuation_indices = table_continuation_indices(table_continuations)

    def add_page_metadata(page_idx: int, block: dict[str, Any], block_idx: int) -> None:
        if block_idx not in page_block_indices[page_idx]:
            page_block_indices[page_idx].append(block_idx)
        if page_idx not in page_source_fields:
            page_source_fields[page_idx] = _source_fields_for_block(block, source_name)
        if page_idx not in page_breadcrumbs:
            breadcrumb = str(block.get("breadcrumb", "") or "").strip()
            if breadcrumb:
                page_breadcrumbs[page_idx] = breadcrumb
        bbox = _block_bbox(block)
        if bbox is not None:
            page_bboxes[page_idx].append([round(value, 3) for value in bbox])
        if page_idx not in page_sections:
            section_path = _section_path(block)
            section_level = block.get("section_level")
            try:
                section_level = int(section_level) if section_level is not None else None
            except (TypeError, ValueError):
                section_level = None
            section_source = str(block.get("section_source", "") or "")
            if section_path:
                page_sections[page_idx] = (section_path, section_level, section_source)

    for block_idx, block in enumerate(content_data):
        if not isinstance(block, dict):
            continue
        if block_idx in continuation_indices:
            continue
        page_idx = int(block.get("page_idx", 0))
        block_type = block.get("type")

        if block_type in IGNORED_BLOCK_TYPES:
            continue

        if block_type == "image":
            description = str(block.get("image_description_vlm", "")).strip()
            caption = _join_field(block.get("image_caption", [])).strip()
            footnote = _join_field(block.get("image_footnote", [])).strip()
            if description or caption or footnote:
                parts = []
                label = f"[Image with illustration: {caption}]" if caption else "[Image with illustration]"
                if description or caption:
                    parts.append(f"\n{label}\n{description}".strip())
                if footnote:
                    parts.append(f"[Image footnote: {footnote}]")
                chunk_contents_by_page[page_idx].append("\n".join(parts))
                add_page_metadata(page_idx, block, block_idx)
            if block.get("img_path") and not _has_inline_icon_marker(block):
                page_images[page_idx].append(block["img_path"])
                evidence = _image_answering_evidence_for_block(
                    block,
                    block_idx,
                    page_idx=page_idx,
                    bbox=_block_bbox(block),
                    caption=caption,
                )
                if evidence:
                    page_image_answering_evidence[page_idx].append(evidence)

        elif block_type == "table":
            caption = _join_field(block.get("table_caption", [])).strip()
            body = _join_field(block.get("table_body", [])).strip()
            footnote = _join_field(block.get("table_footnote", [])).strip()
            parts = []
            if caption:
                parts.append(f"[Table: {caption}]")
            if body:
                parts.append(body)
            if footnote:
                parts.append(f"[Footnote: {footnote}]")
            if parts:
                table_text = "\n".join(parts)
                chunk_contents_by_page[page_idx].append(table_text)
                add_page_metadata(page_idx, block, block_idx)
                for continuation_idx in table_continuations.get(block_idx, []):
                    continuation = content_data[continuation_idx]
                    continuation_page_idx = _block_page_idx(continuation)
                    chunk_contents_by_page[continuation_page_idx].append(
                        f"[Continuation of table from page {page_idx + 1}]\n{table_text}"
                    )
                    add_page_metadata(continuation_page_idx, continuation, continuation_idx)
            if block.get("img_path"):
                page_tables[page_idx].append(block["img_path"])

        elif block_type in {"text", "list"}:
            key = "list_items" if block_type == "list" else "text"
            text = _join_field(block.get(key, [])).strip()
            if text:
                chunk_contents_by_page[page_idx].append(text)
                add_page_metadata(page_idx, block, block_idx)

    chunks: list[dict[str, Any]] = []
    for page_idx in sorted(chunk_contents_by_page):
        section_path, section_level, section_source = page_sections.get(page_idx, ((), None, ""))
        source_fields = page_source_fields.get(page_idx, source_payload_fields(source_name))
        breadcrumb = page_breadcrumbs.get(page_idx) or source_breadcrumb(source_fields["source_relpath"], section_path)
        content_parts = [f"[Breadcrumb: {breadcrumb}]"]
        if section_path:
            content_parts.append(f"[Section: {' > '.join(section_path)}]")
        content_parts.extend(chunk_contents_by_page[page_idx])
        chunk_content = "\n\n".join(content_parts).strip()
        if not chunk_content:
            continue
        chunk_idx = len(chunks)
        metadata: dict[str, Any] = {
            **source_fields,
            "chunk_idx": chunk_idx,
            "chunk_id": f"{Path(source_name).stem}-chunk-{chunk_idx:05d}",
            "chunk_mode": "page",
            "token_count": estimate_token_count(chunk_content),
            "block_indices": page_block_indices[page_idx],
            "bboxes_by_page": {str(page_idx): page_bboxes[page_idx]} if page_bboxes[page_idx] else {},
            "page_idx": page_idx,
            "page_start": page_idx,
            "page_end": page_idx,
            "page_indices": [page_idx],
            "images_on_page": _unique(page_images[page_idx]),
            "tables_on_page": _unique(page_tables[page_idx]),
        }
        metadata["breadcrumb"] = breadcrumb
        if page_image_answering_evidence[page_idx]:
            metadata["image_answering_evidence"] = page_image_answering_evidence[page_idx]
        if section_path:
            metadata["section_path"] = list(section_path)
            metadata["section_title"] = section_path[-1]
            if section_level is not None:
                metadata["section_level"] = section_level
            if section_source:
                metadata["section_source"] = section_source
        chunks.append(
            {
                "chunk_content": chunk_content,
                "metadata": metadata,
            }
        )
    return chunks


def create_chunks(
    json_path: str | Path,
    source_name: str,
    *,
    mode: str = "auto",
    max_tokens: int = 5000,
    overlap_tokens: int = 500,
    min_tokens: int = 200,
) -> list[dict[str, Any]]:
    if mode not in SUPPORTED_CHUNK_MODES:
        raise ValueError(f"Unsupported chunk mode {mode!r}. Choose one of: {', '.join(SUPPORTED_CHUNK_MODES)}")
    if mode == "page":
        return create_page_level_chunks(json_path, source_name)

    with Path(json_path).open("r", encoding="utf-8") as f:
        content_data = json.load(f)
    if not isinstance(content_data, list):
        raise ValueError(f"Expected a list in content JSON: {json_path}")

    items = content_items(content_data)
    has_sections = any(item.section_path for item in items)
    resolved_mode = "section" if mode == "auto" and has_sections else "token" if mode == "auto" else mode
    if resolved_mode == "section" and not has_sections:
        resolved_mode = "token"

    if resolved_mode == "section":
        return _section_window_chunks(
            items,
            source_name,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_tokens=min_tokens,
        )
    return _token_window_chunks(
        items,
        source_name,
        mode="token",
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
    )


def write_chunks(chunks: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Build retrieval chunks from MinerU content_list JSON.")
    parser.add_argument("--input", default=str(config.paths.captioned_json), help="Input enriched content_list JSON.")
    parser.add_argument("--output", default=str(config.paths.chunks_json), help="Output chunks JSON.")
    parser.add_argument("--source-name", default=config.paths.source_name, help="Source PDF name stored in metadata.")
    parser.add_argument("--mode", choices=SUPPORTED_CHUNK_MODES, default=config.chunking.mode)
    parser.add_argument("--max-tokens", type=int, default=config.chunking.max_tokens)
    parser.add_argument("--overlap-tokens", type=int, default=config.chunking.overlap_tokens)
    parser.add_argument("--min-tokens", type=int, default=config.chunking.min_tokens)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved chunking settings without reading JSON.")
    args = parser.parse_args(argv)

    if args.dry_run:
        print("Chunking inputs:")
        print(f"  input_json: {Path(args.input).expanduser()}")
        print(f"  output_json: {Path(args.output).expanduser()}")
        print(f"  source_name: {args.source_name}")
        print(f"  mode: {args.mode}")
        print(f"  max_tokens: {args.max_tokens}")
        print(f"  overlap_tokens: {args.overlap_tokens}")
        print(f"  min_tokens: {args.min_tokens}")
        return

    chunks = create_chunks(
        args.input,
        args.source_name,
        mode=args.mode,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        min_tokens=args.min_tokens,
    )
    write_chunks(chunks, args.output)
    print(f"Created {len(chunks)} {args.mode} chunks at {args.output}")


if __name__ == "__main__":
    main()
