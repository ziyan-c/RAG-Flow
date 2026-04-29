from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_flow.config import AppConfig
from rag_flow.preprocessing import small_icons


FIELD_COLORS = {
    "text": (0.15, 0.65, 0.25),
    "list_items": (0.10, 0.35, 0.95),
    "table_body": (0.95, 0.35, 0.05),
    "table_caption": (0.90, 0.70, 0.05),
    "table_footnote": (0.65, 0.20, 0.85),
    "inline_icon": (0.95, 0.05, 0.15),
}


@dataclass(frozen=True)
class PatchingViewRegion:
    page_idx: int
    bbox: tuple[float, float, float, float]
    field: str
    label: str


@dataclass(frozen=True)
class PatchingViewPlan:
    regions: tuple[PatchingViewRegion, ...]
    field_counts: dict[str, int]
    inline_icon_candidates: int
    inline_icons_linked: int


@dataclass(frozen=True)
class PatchingViewStats:
    output_pdf: Path
    page_count: int
    pages_with_regions: int
    region_count: int
    field_counts: dict[str, int]
    inline_icon_candidates: int
    inline_icons_linked: int


def patching_view_path_for(content_json: str | Path) -> Path:
    path = Path(content_json)
    name = path.name
    for suffix in (
        "_content_list_PATCHED_CAPTIONED.json",
        "_content_list_PATCHED.json",
        "_content_list.json",
    ):
        if name.endswith(suffix):
            return path.with_name(f"{name[: -len(suffix)]}_PATCHING_VIEW.pdf")
    return path.with_name(f"{path.stem}_PATCHING_VIEW.pdf")


def _bbox_union_for_block_and_icons(
    *,
    block: dict[str, Any],
    content_data: list[dict[str, Any]],
    inline_icon_links: list[small_icons.InlineIconLink],
) -> tuple[float, float, float, float] | None:
    block_bbox = small_icons._block_bbox(block)
    if block_bbox is None:
        return None
    page_idx = small_icons._block_page_idx(block)
    boxes = [block_bbox]
    for link in inline_icon_links:
        icon = content_data[link.icon_idx]
        if small_icons._block_page_idx(icon, default=-1) != page_idx:
            continue
        icon_bbox = small_icons._block_bbox(icon)
        if icon_bbox is not None:
            boxes.append(icon_bbox)
    return small_icons._bbox_union(boxes)


def _table_footnote_region(
    *,
    content_data: list[dict[str, Any]],
    block_idx: int,
) -> tuple[int, tuple[float, float, float, float]] | None:
    block = content_data[block_idx]
    last_idx = block_idx
    lookahead_idx = block_idx + 1

    while lookahead_idx < len(content_data):
        next_block = content_data[lookahead_idx]
        next_type = next_block.get("type")
        if next_type in small_icons.IGNORE_TYPES:
            lookahead_idx += 1
            continue
        if next_type != block.get("type"):
            break
        next_text = small_icons._join(next_block.get("table_footnote", "")).strip()
        if next_text:
            break
        last_idx = lookahead_idx
        lookahead_idx += 1

    last_block = content_data[last_idx]
    last_bbox = small_icons._block_bbox(last_block)
    if last_bbox is None:
        return None
    page_idx = small_icons._block_page_idx(last_block)
    y0_norm = last_bbox[3]
    y1_norm = 1000.0

    for idx in range(last_idx + 1, len(content_data)):
        next_block = content_data[idx]
        if small_icons._block_page_idx(next_block, default=-1) != page_idx:
            break
        next_bbox = small_icons._block_bbox(next_block)
        if next_block.get("type") not in {"header", "footer", "page_number"} and next_bbox is not None:
            next_y0 = next_bbox[1]
            if next_y0 > y0_norm:
                y1_norm = next_y0
                break

    return page_idx, (0.0, y0_norm, 1000.0, y1_norm)


def collect_patching_view_regions(content_data: list[dict[str, Any]]) -> PatchingViewPlan:
    regions: list[PatchingViewRegion] = []
    field_counts: Counter[str] = Counter()
    table_continuations = small_icons.build_table_continuation_map(content_data)
    table_continuation_indices = small_icons._table_continuation_indices(table_continuations)
    inline_links = small_icons.build_inline_icon_links(content_data, table_continuations)

    def add_region(
        *,
        page_idx: int,
        bbox: tuple[float, float, float, float] | None,
        field: str,
        label: str,
    ) -> None:
        if bbox is None:
            return
        regions.append(PatchingViewRegion(page_idx=page_idx, bbox=bbox, field=field, label=label))
        field_counts[field] += 1

    for idx, block in enumerate(content_data):
        if not isinstance(block, dict):
            continue
        if block.get("type") in small_icons.IGNORE_TYPES or idx in table_continuation_indices:
            continue
        if small_icons._block_bbox(block) is None:
            continue

        field_keys = small_icons._patch_field_keys(block)
        if not field_keys:
            continue

        checked_fields = small_icons._checked_fields(block)
        for key in field_keys:
            if key in checked_fields:
                continue
            original_text = small_icons._join(block.get(key, "")).strip()
            if not original_text:
                continue

            page_idx = small_icons._block_page_idx(block)
            if key == "table_body":
                add_region(
                    page_idx=page_idx,
                    bbox=_bbox_union_for_block_and_icons(
                        block=block,
                        content_data=content_data,
                        inline_icon_links=inline_links.by_target.get(idx, []),
                    ),
                    field=key,
                    label=f"{idx}:{key}",
                )
                for continuation_idx in table_continuations.get(idx, []):
                    continuation = content_data[continuation_idx]
                    add_region(
                        page_idx=small_icons._block_page_idx(continuation),
                        bbox=_bbox_union_for_block_and_icons(
                            block=continuation,
                            content_data=content_data,
                            inline_icon_links=inline_links.by_target.get(idx, []),
                        ),
                        field=key,
                        label=f"{idx}:{key}:continuation",
                    )
                continue

            if key == "table_footnote":
                footnote = _table_footnote_region(content_data=content_data, block_idx=idx)
                if footnote is not None:
                    footnote_page_idx, footnote_bbox = footnote
                    add_region(page_idx=footnote_page_idx, bbox=footnote_bbox, field=key, label=f"{idx}:{key}")
                continue

            add_region(
                page_idx=page_idx,
                bbox=_bbox_union_for_block_and_icons(
                    block=block,
                    content_data=content_data,
                    inline_icon_links=[
                        link for link in inline_links.by_target.get(idx, []) if link.target_field == key
                    ],
                ),
                field=key,
                label=f"{idx}:{key}",
            )

    for link in inline_links.by_icon.values():
        icon = content_data[link.icon_idx]
        add_region(
            page_idx=small_icons._block_page_idx(icon),
            bbox=small_icons._block_bbox(icon),
            field="inline_icon",
            label=f"{link.icon_idx}:inline-icon",
        )

    return PatchingViewPlan(
        regions=tuple(regions),
        field_counts=dict(field_counts),
        inline_icon_candidates=len(inline_links.candidates),
        inline_icons_linked=len(inline_links.by_icon),
    )


def _draw_legend(page: Any, stats: PatchingViewStats) -> None:
    import fitz

    legend_x = 28
    legend_y = 28
    page.draw_rect(
        fitz.Rect(legend_x - 8, legend_y - 12, legend_x + 230, legend_y + 120),
        color=(0.1, 0.1, 0.1),
        fill=(1, 1, 1),
        width=0.5,
        overlay=True,
        fill_opacity=0.85,
        stroke_opacity=0.5,
    )
    page.insert_text((legend_x, legend_y), "PATCHING VIEW", fontsize=10, color=(0, 0, 0), overlay=True)
    legend_items = [
        ("text", "text crop"),
        ("list_items", "list crop"),
        ("table_body", "table body / continuation crop"),
        ("table_caption", "table caption crop"),
        ("table_footnote", "table footnote crop"),
        ("inline_icon", "linked inline icon bbox"),
    ]
    for row, (field, label) in enumerate(legend_items, start=1):
        y = legend_y + row * 16
        color = FIELD_COLORS[field]
        page.draw_rect(
            fitz.Rect(legend_x, y - 9, legend_x + 12, y + 1),
            color=color,
            fill=color,
            overlay=True,
            fill_opacity=0.35,
        )
        page.insert_text(
            (legend_x + 18, y),
            f"{label}: {stats.field_counts.get(field, 0)}",
            fontsize=7,
            color=(0, 0, 0),
            overlay=True,
        )


def write_patching_view_pdf(
    *,
    content_json: str | Path,
    pdf_path: str | Path,
    output_pdf: str | Path | None = None,
) -> PatchingViewStats:
    import fitz

    content_path = Path(content_json)
    output_path = Path(output_pdf).expanduser() if output_pdf else patching_view_path_for(content_path)
    content_data: list[dict[str, Any]] = json.loads(content_path.read_text(encoding="utf-8"))
    plan = collect_patching_view_regions(content_data)
    regions_by_page: dict[int, list[PatchingViewRegion]] = defaultdict(list)
    for region in plan.regions:
        regions_by_page[region.page_idx].append(region)

    doc = fitz.open(Path(pdf_path))
    stats = PatchingViewStats(
        output_pdf=output_path,
        page_count=len(doc),
        pages_with_regions=len(regions_by_page),
        region_count=len(plan.regions),
        field_counts=plan.field_counts,
        inline_icon_candidates=plan.inline_icon_candidates,
        inline_icons_linked=plan.inline_icons_linked,
    )

    for page_idx, page in enumerate(doc):
        width = page.rect.width
        height = page.rect.height
        for region in regions_by_page.get(page_idx, []):
            x0, y0, x1, y1 = region.bbox
            rect = fitz.Rect(
                max(0, x0 / 1000.0 * width),
                max(0, y0 / 1000.0 * height),
                min(width, x1 / 1000.0 * width),
                min(height, y1 / 1000.0 * height),
            )
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
            if region.field == "inline_icon" and rect.width >= 10 and rect.height >= 6:
                page.insert_textbox(
                    rect + (-1, -8, 70, -2),
                    "icon",
                    fontsize=5,
                    color=color,
                    overlay=True,
                )

        if page_idx == 0:
            _draw_legend(page, stats)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    return stats


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Draw patching VLM crop regions over a source PDF.")
    parser.add_argument(
        "--input-json",
        default=str(config.paths.patched_json),
        help="Patched or raw content_list JSON.",
    )
    parser.add_argument("--input-pdf", default=str(config.paths.source_pdf), help="Source PDF used by patching.")
    parser.add_argument("--output", default=None, help="Output PATCHING_VIEW PDF.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths without writing the PDF.")
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser() if args.output else patching_view_path_for(args.input_json)
    if args.dry_run:
        print("Patching view inputs:")
        print(f"  input_json: {Path(args.input_json).expanduser()}")
        print(f"  input_pdf: {Path(args.input_pdf).expanduser()}")
        print(f"  output_pdf: {output}")
        return

    stats = write_patching_view_pdf(content_json=args.input_json, pdf_path=args.input_pdf, output_pdf=output)
    print(f"Generated patching view PDF at {stats.output_pdf}")
    print(f"  pages: {stats.page_count}")
    print(f"  pages with overlays: {stats.pages_with_regions}")
    print(f"  overlays: {stats.region_count}")
    print(f"  inline icons linked: {stats.inline_icons_linked}/{stats.inline_icon_candidates}")


if __name__ == "__main__":
    main()
