#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
ASSETS = ROOT / "assets"
SOURCE_PDF = REPO_ROOT / ".local" / "source-documents" / "technical-manual.pdf"
CONTENT_JSON = Path("/Users/ziyan/MINE/technical-manual/hybrid_auto/technical-manual_content_list.json")
PATCHED_JSON = Path("/Users/ziyan/MINE/technical-manual/hybrid_auto/technical-manual_content_list_PATCHED.json")
DPIS = (200, 250, 300)

sys.path.insert(0, str(REPO_ROOT / "src"))
from rag_flow.preprocessing import small_icons  # noqa: E402
from rag_flow.preprocessing.patching_view import FIELD_COLORS, collect_patching_view_regions  # noqa: E402


@dataclass(frozen=True)
class CaseRow:
    label: str
    block_idx: int
    field_key: str
    visual_reference: str


@dataclass(frozen=True)
class PageCase:
    chrome_page: int
    page_idx: int
    rows: tuple[CaseRow, ...]


PAGE_CASES = (
    PageCase(
        chrome_page=313,
        page_idx=312,
        rows=(
            CaseRow(
                "tracks/favorites icons",
                4313,
                "text",
                "Click [Icon: star] ... click [Icon: starred record with red badge] at the upper-right corner ...",
            ),
            CaseRow(
                "record operation list",
                4316,
                "list_items",
                "Add [Icon: face arming group], [Icon: track playback], [Icon: filter/search], [Icon: vehicle arming group], and [Icon: delete] at the missing positions.",
            ),
            CaseRow(
                "resume upload action",
                4319,
                "text",
                "Click [Icon: resume/upload] at the upper right, then enter the platform login password.",
            ),
        ),
    ),
    PageCase(
        chrome_page=320,
        page_idx=319,
        rows=(
            CaseRow(
                "case add/back actions",
                4433,
                "list_items",
                "Click [Icon: add/plus] next to the record; click [Icon: back/left arrow] to go back.",
            ),
            CaseRow(
                "attachment action",
                4434,
                "text",
                "Step 8 Click [Icon: attachment/link], then click Add under Attachment.",
            ),
            CaseRow(
                "related operation icons",
                4439,
                "list_items",
                "Recover [Icon: view], [Icon: delete/minus], [Icon: upload], [Icon: search], [Icon: download], [Icon: trash], and [Icon: toggle].",
            ),
        ),
    ),
    PageCase(
        chrome_page=435,
        page_idx=434,
        rows=(
            CaseRow(
                "interface operation table",
                6210,
                "table_body",
                "Replace the corrupted Icon/Function cells with semantic labels such as close-all, split-screen, snapshot, close-window, stop/pause, speed, frame-by-frame, and search-by-snapshot.",
            ),
            CaseRow(
                "plain settings paragraph",
                6212,
                "text",
                "No missing icons. Keep the original paragraph unchanged.",
            ),
        ),
    ),
)


def norm_to_rect(
    bbox: tuple[float, float, float, float],
    *,
    page_rect: fitz.Rect,
) -> fitz.Rect:
    x0, y0, x1, y1 = bbox
    return fitz.Rect(
        max(0.0, x0 / 1000.0 * page_rect.width),
        max(0.0, y0 / 1000.0 * page_rect.height),
        min(page_rect.width, x1 / 1000.0 * page_rect.width),
        min(page_rect.height, y1 / 1000.0 * page_rect.height),
    )


def union_bbox(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def render_crop_png(
    pdf: fitz.Document,
    *,
    page_idx: int,
    bbox: tuple[float, float, float, float],
    dpi: int,
) -> tuple[bytes, int, int]:
    page = pdf[page_idx]
    clip = norm_to_rect(bbox, page_rect=page.rect)
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    return pix.tobytes("png"), pix.width, pix.height


def linked_crop_bbox(
    *,
    content_data: list[dict],
    inline_links: small_icons.InlineIconPlan,
    block_idx: int,
    field_key: str,
) -> tuple[float, float, float, float]:
    block = content_data[block_idx]
    block_bbox = small_icons._block_bbox(block)
    if block_bbox is None:
        raise ValueError(f"block {block_idx} has no bbox")
    page_idx = small_icons._block_page_idx(block)
    boxes = [block_bbox]
    for link in inline_links.by_target.get(block_idx, []):
        if link.target_field != field_key:
            continue
        icon = content_data[link.icon_idx]
        if small_icons._block_page_idx(icon, default=-1) != page_idx:
            continue
        icon_bbox = small_icons._block_bbox(icon)
        if icon_bbox is not None:
            boxes.append(icon_bbox)
    return union_bbox(boxes)


def save_patching_view_crop(content_data: list[dict], page_case: PageCase) -> None:
    plan = collect_patching_view_regions(content_data)
    page_regions = [region for region in plan.regions if region.page_idx == page_case.page_idx]
    crop_bbox = union_bbox([region.bbox for region in page_regions])

    with fitz.open(SOURCE_PDF) as pdf:
        page = pdf[page_case.page_idx]
        for region in page_regions:
            rect = norm_to_rect(region.bbox, page_rect=page.rect)
            color = FIELD_COLORS.get(region.field, (0.2, 0.2, 0.2))
            page.draw_rect(
                rect,
                color=color,
                fill=color,
                width=0.8,
                overlay=True,
                fill_opacity=0.35 if region.field == "inline_icon" else 0.18,
                stroke_opacity=0.95,
            )

        clip = norm_to_rect(crop_bbox, page_rect=page.rect)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), clip=clip, alpha=False)
    pix.save(ASSETS / f"reference_page_{page_case.chrome_page}_patching_view_regions.png")

    counts = Counter(region.field for region in page_regions)
    print(
        f"Reference page {page_case.chrome_page} / page_idx {page_case.page_idx}: "
        f"{len(page_regions)} patching regions {dict(counts)}"
    )


def row_height_for_bbox(bbox: tuple[float, float, float, float]) -> int:
    height = bbox[3] - bbox[1]
    if height >= 450:
        return 270
    if height >= 180:
        return 220
    if height >= 90:
        return 158
    return 128


def save_dpi_crop_grid(content_data: list[dict], page_case: PageCase) -> None:
    table_continuations = small_icons.build_table_continuation_map(content_data)
    inline_links = small_icons.build_inline_icon_links(content_data, table_continuations)
    row_crops: list[tuple[CaseRow, tuple[float, float, float, float], dict[int, tuple[bytes, int, int]]]] = []

    with fitz.open(SOURCE_PDF) as pdf:
        for row in page_case.rows:
            bbox = linked_crop_bbox(
                content_data=content_data,
                inline_links=inline_links,
                block_idx=row.block_idx,
                field_key=row.field_key,
            )
            crops = {dpi: render_crop_png(pdf, page_idx=page_case.page_idx, bbox=bbox, dpi=dpi) for dpi in DPIS}
            row_crops.append((row, bbox, crops))

    label_w = 188
    cell_w = 224
    margin = 24
    header_h = 58
    row_heights = [row_height_for_bbox(bbox) for _, bbox, _ in row_crops]
    width = margin * 2 + label_w + len(DPIS) * cell_w
    height = margin * 2 + header_h + sum(row_heights)

    grid = fitz.open()
    page = grid.new_page(width=width, height=height)
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1), overlay=True)
    page.insert_text(
        (margin, margin + 8),
        f"Reference page {page_case.chrome_page} / page_idx={page_case.page_idx}: actual VLM crop bboxes",
        fontsize=11.5,
        color=(0.07, 0.09, 0.15),
    )
    for col, dpi in enumerate(DPIS):
        x = margin + label_w + col * cell_w
        page.insert_text((x, margin + 38), f"{dpi} DPI", fontsize=11, color=(0.15, 0.39, 0.92))

    y = margin + header_h
    for row_h, (row, bbox, crops) in zip(row_heights, row_crops):
        cell_h = max(84, row_h - 38)
        page.draw_line(
            (margin, y - 8),
            (width - margin, y - 8),
            color=(0.90, 0.91, 0.92),
            width=0.5,
        )
        page.insert_textbox(
            fitz.Rect(margin, y + 4, margin + label_w - 10, y + 42),
            row.label,
            fontsize=8.4,
            color=(0.07, 0.09, 0.15),
        )
        page.insert_text(
            (margin, y + 54),
            f"{row.block_idx}:{row.field_key}",
            fontsize=7.2,
            color=(0.29, 0.33, 0.39),
        )
        page.insert_textbox(
            fitz.Rect(margin, y + 65, margin + label_w - 10, y + row_h - 8),
            "bbox " + ",".join(str(round(value)) for value in bbox),
            fontsize=6.3,
            color=(0.42, 0.45, 0.50),
        )

        for col, dpi in enumerate(DPIS):
            x = margin + label_w + col * cell_w
            png_bytes, native_w, native_h = crops[dpi]
            img_rect = fitz.Rect(x + 4, y + 2, x + cell_w - 8, y + cell_h)
            page.draw_rect(img_rect, color=(0.86, 0.88, 0.90), width=0.4)
            page.insert_image(img_rect, stream=png_bytes, keep_proportion=True)
            page.insert_text(
                (x + 6, y + cell_h + 13),
                f"native {native_w}x{native_h}px",
                fontsize=6.6,
                color=(0.29, 0.33, 0.39),
            )
        y += row_h

    pix = page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), alpha=False)
    pix.save(ASSETS / f"reference_page_{page_case.chrome_page}_vlm_crop_dpi_grid.png")
    grid.close()


def save_reference_crop_overview(content_data: list[dict], *, dpi: int = 250) -> None:
    table_continuations = small_icons.build_table_continuation_map(content_data)
    inline_links = small_icons.build_inline_icon_links(content_data, table_continuations)
    page_rows: list[tuple[PageCase, list[tuple[CaseRow, tuple[float, float, float, float], bytes, int, int]]]] = []

    with fitz.open(SOURCE_PDF) as pdf:
        for page_case in PAGE_CASES:
            crops = []
            for row in page_case.rows:
                if row.block_idx == 6212:
                    continue
                bbox = linked_crop_bbox(
                    content_data=content_data,
                    inline_links=inline_links,
                    block_idx=row.block_idx,
                    field_key=row.field_key,
                )
                png_bytes, native_w, native_h = render_crop_png(pdf, page_idx=page_case.page_idx, bbox=bbox, dpi=dpi)
                crops.append((row, bbox, png_bytes, native_w, native_h))
            page_rows.append((page_case, crops))

    width = 980
    margin = 28
    label_w = 168
    gap = 14
    row_h = 202
    height = margin * 2 + 44 + len(page_rows) * row_h
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1), overlay=True)
    page.insert_text(
        (margin, margin + 6),
        f"Target technical manual reference-page crops ({dpi} DPI)",
        fontsize=14,
        color=(0.07, 0.09, 0.15),
    )
    page.insert_text(
        (margin, margin + 26),
        "Representative VLM crop regions used for icon recovery; DPI variants use the same logical bboxes.",
        fontsize=9,
        color=(0.29, 0.33, 0.39),
    )

    y = margin + 52
    for page_case, crops in page_rows:
        page.draw_line((margin, y - 8), (width - margin, y - 8), color=(0.90, 0.91, 0.92), width=0.5)
        page.insert_textbox(
            fitz.Rect(margin, y + 8, margin + label_w - 12, y + row_h - 16),
            f"Reference page {page_case.chrome_page}\npage_idx={page_case.page_idx}",
            fontsize=10.2,
            color=(0.07, 0.09, 0.15),
        )
        crop_area_w = width - margin * 2 - label_w
        cell_w = (crop_area_w - gap * (len(crops) - 1)) / len(crops)
        x = margin + label_w
        for row, _bbox, png_bytes, native_w, native_h in crops:
            img_rect = fitz.Rect(x, y + 4, x + cell_w, y + row_h - 38)
            page.draw_rect(img_rect, color=(0.86, 0.88, 0.90), width=0.4)
            page.insert_image(img_rect, stream=png_bytes, keep_proportion=True)
            page.insert_textbox(
                fitz.Rect(x, y + row_h - 30, x + cell_w, y + row_h - 6),
                f"{row.block_idx}:{row.field_key}  {native_w}x{native_h}px",
                fontsize=7.0,
                color=(0.29, 0.33, 0.39),
            )
            x += cell_w + gap
        y += row_h

    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
    pix.save(ASSETS / f"target_manual_reference_page_crops_{dpi}dpi.png")
    doc.close()


def shorten(text: str, limit: int = 260) -> str:
    text = " ".join(text.replace("\n", " / ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def save_reference_csv(content_data: list[dict], patched_data: list[dict] | None) -> None:
    output = ROOT / "dpi-icon-reference-pages.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "chrome_page",
                "page_idx",
                "block_idx",
                "field",
                "original_ocr",
                "visual_reference",
                "current_patched_output",
            ],
        )
        writer.writeheader()
        for page_case in PAGE_CASES:
            for row in page_case.rows:
                original = small_icons._join(content_data[row.block_idx].get(row.field_key, ""))
                patched = ""
                if patched_data is not None:
                    patched = small_icons._join(patched_data[row.block_idx].get(row.field_key, ""))
                writer.writerow(
                    {
                        "chrome_page": page_case.chrome_page,
                        "page_idx": page_case.page_idx,
                        "block_idx": row.block_idx,
                        "field": row.field_key,
                        "original_ocr": shorten(original),
                        "visual_reference": row.visual_reference,
                        "current_patched_output": shorten(patched),
                    }
                )
    print(f"Wrote {output}")


def main() -> None:
    if not CONTENT_JSON.exists():
        raise SystemExit(f"Missing source JSON: {CONTENT_JSON}")
    ASSETS.mkdir(exist_ok=True)
    content_data: list[dict] = json.loads(CONTENT_JSON.read_text(encoding="utf-8"))
    patched_data: list[dict] | None = None
    if PATCHED_JSON.exists():
        patched_data = json.loads(PATCHED_JSON.read_text(encoding="utf-8"))

    for page_case in PAGE_CASES:
        save_patching_view_crop(content_data, page_case)
        save_dpi_crop_grid(content_data, page_case)
    save_reference_crop_overview(content_data, dpi=250)
    save_reference_csv(content_data, patched_data)


if __name__ == "__main__":
    main()
