from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_flow.config import AppConfig


CHUNK_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.90, 0.12, 0.12),
    (0.10, 0.35, 0.95),
    (0.12, 0.65, 0.25),
    (0.70, 0.20, 0.85),
    (0.95, 0.48, 0.05),
    (0.00, 0.62, 0.62),
    (0.95, 0.10, 0.55),
    (0.45, 0.52, 0.05),
)


@dataclass(frozen=True)
class ChunkingViewRegion:
    page_idx: int
    bbox: tuple[float, float, float, float]
    chunk_idx: int
    chunk_id: str
    label: str
    token_count: int | None = None
    section_title: str = ""


@dataclass(frozen=True)
class ChunkingViewPlan:
    regions: tuple[ChunkingViewRegion, ...]
    chunk_count: int
    chunks_with_regions: int
    pages_with_regions: int
    regions_by_chunk: dict[int, int]


@dataclass(frozen=True)
class ChunkingViewStats:
    output_pdf: Path
    page_count: int
    chunk_count: int
    chunks_with_regions: int
    pages_with_regions: int
    region_count: int
    regions_by_chunk: dict[int, int]


def color_for_chunk(chunk_idx: int) -> tuple[float, float, float]:
    return CHUNK_COLORS[chunk_idx % len(CHUNK_COLORS)]


def chunking_view_path_for(chunks_json: str | Path) -> Path:
    path = Path(chunks_json)
    name = path.name
    for suffix in (
        "_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED.json",
        "_content_list_PATCHED_CAPTIONED_CHUNKED.json",
        "_content_list_CHUNKED.json",
        "_page_level_chunks.json",
        "_chunks.json",
    ):
        if name.endswith(suffix):
            return path.with_name(f"{name[: -len(suffix)]}_CHUNKING_VIEW.pdf")
    return path.with_name(f"{path.stem}_CHUNKING_VIEW.pdf")


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def collect_chunking_view_regions(chunks: list[dict[str, Any]]) -> ChunkingViewPlan:
    regions: list[ChunkingViewRegion] = []
    regions_by_chunk: Counter[int] = Counter()
    chunks_with_regions: set[int] = set()
    pages_with_regions: set[int] = set()

    for fallback_idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        chunk_idx = _coerce_int(metadata.get("chunk_idx"), fallback_idx)
        chunk_id = str(metadata.get("chunk_id") or f"chunk-{chunk_idx:05d}")
        token_count = metadata.get("token_count")
        token_count = _coerce_int(token_count, 0) if token_count is not None else None
        section_title = str(metadata.get("section_title") or "")
        bboxes_by_page = metadata.get("bboxes_by_page", {})
        if not isinstance(bboxes_by_page, dict):
            continue

        for page_key, bboxes in bboxes_by_page.items():
            page_idx = _coerce_int(page_key, -1)
            if page_idx < 0 or not isinstance(bboxes, list):
                continue
            for bbox_idx, bbox_value in enumerate(bboxes):
                bbox = _coerce_bbox(bbox_value)
                if bbox is None:
                    continue
                label = f"chunk {chunk_idx}"
                if bbox_idx == 0 and token_count is not None:
                    label = f"{label} / {token_count} tok"
                regions.append(
                    ChunkingViewRegion(
                        page_idx=page_idx,
                        bbox=bbox,
                        chunk_idx=chunk_idx,
                        chunk_id=chunk_id,
                        label=label,
                        token_count=token_count,
                        section_title=section_title,
                    )
                )
                regions_by_chunk[chunk_idx] += 1
                chunks_with_regions.add(chunk_idx)
                pages_with_regions.add(page_idx)

    return ChunkingViewPlan(
        regions=tuple(sorted(regions, key=lambda item: (item.page_idx, item.chunk_idx, item.bbox))),
        chunk_count=sum(1 for chunk in chunks if isinstance(chunk, dict)),
        chunks_with_regions=len(chunks_with_regions),
        pages_with_regions=len(pages_with_regions),
        regions_by_chunk=dict(regions_by_chunk),
    )


def _draw_legend(page: Any, stats: ChunkingViewStats) -> None:
    import fitz

    legend_x = 28
    legend_y = 28
    page.draw_rect(
        fitz.Rect(legend_x - 8, legend_y - 12, legend_x + 290, legend_y + 126),
        color=(0.1, 0.1, 0.1),
        fill=(1, 1, 1),
        width=0.5,
        overlay=True,
        fill_opacity=0.85,
        stroke_opacity=0.5,
    )
    page.insert_text((legend_x, legend_y), "CHUNKING VIEW", fontsize=10, color=(0, 0, 0), overlay=True)
    page.insert_text(
        (legend_x, legend_y + 15),
        f"chunks: {stats.chunk_count}  with bbox: {stats.chunks_with_regions}",
        fontsize=7,
        color=(0, 0, 0),
        overlay=True,
    )
    page.insert_text(
        (legend_x, legend_y + 28),
        f"pages with overlays: {stats.pages_with_regions}  overlays: {stats.region_count}",
        fontsize=7,
        color=(0, 0, 0),
        overlay=True,
    )
    page.insert_text(
        (legend_x, legend_y + 42),
        "Adjacent chunk colors alternate; repeated colors are non-adjacent palette reuse.",
        fontsize=6,
        color=(0, 0, 0),
        overlay=True,
    )
    for idx, color in enumerate(CHUNK_COLORS):
        row = idx // 4
        col = idx % 4
        x = legend_x + col * 68
        y = legend_y + 62 + row * 24
        page.draw_rect(
            fitz.Rect(x, y - 9, x + 16, y + 3),
            color=color,
            fill=color,
            overlay=True,
            fill_opacity=0.35,
        )
        page.insert_text((x + 20, y), f"chunk {idx}", fontsize=6, color=(0, 0, 0), overlay=True)


def write_chunking_view_pdf(
    *,
    chunks_json: str | Path,
    pdf_path: str | Path,
    output_pdf: str | Path | None = None,
) -> ChunkingViewStats:
    import fitz

    chunks_path = Path(chunks_json)
    output_path = Path(output_pdf).expanduser() if output_pdf else chunking_view_path_for(chunks_path)
    chunks: list[dict[str, Any]] = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a list in chunk JSON: {chunks_path}")

    plan = collect_chunking_view_regions(chunks)
    regions_by_page: dict[int, list[ChunkingViewRegion]] = defaultdict(list)
    for region in plan.regions:
        regions_by_page[region.page_idx].append(region)

    doc = fitz.open(Path(pdf_path))
    stats = ChunkingViewStats(
        output_pdf=output_path,
        page_count=len(doc),
        chunk_count=plan.chunk_count,
        chunks_with_regions=plan.chunks_with_regions,
        pages_with_regions=plan.pages_with_regions,
        region_count=len(plan.regions),
        regions_by_chunk=plan.regions_by_chunk,
    )

    label_seen: set[tuple[int, int]] = set()
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
            color = color_for_chunk(region.chunk_idx)
            page.draw_rect(
                rect,
                color=color,
                fill=color,
                width=0.7,
                overlay=True,
                fill_opacity=0.13,
                stroke_opacity=0.9,
            )
            label_key = (page_idx, region.chunk_idx)
            if label_key not in label_seen and rect.width >= 28 and rect.height >= 8:
                label_seen.add(label_key)
                page.insert_textbox(
                    rect + (2, 2, 120, 17),
                    region.label,
                    fontsize=5.5,
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
    parser = argparse.ArgumentParser(description="Draw chunk bbox regions over a source PDF.")
    parser.add_argument(
        "--input-json",
        default=str(config.paths.chunks_json),
        help="Chunk JSON produced by `rag-flow chunk` or the chunking pipeline stage.",
    )
    parser.add_argument("--input-pdf", default=str(config.paths.source_pdf), help="Source PDF used by chunking.")
    parser.add_argument("--output", default=None, help="Output CHUNKING_VIEW PDF.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths without writing the PDF.")
    args = parser.parse_args(argv)

    output = Path(args.output).expanduser() if args.output else chunking_view_path_for(args.input_json)
    if args.dry_run:
        print("Chunking view inputs:")
        print(f"  input_json: {Path(args.input_json).expanduser()}")
        print(f"  input_pdf: {Path(args.input_pdf).expanduser()}")
        print(f"  output_pdf: {output}")
        return

    stats = write_chunking_view_pdf(chunks_json=args.input_json, pdf_path=args.input_pdf, output_pdf=output)
    print(f"Generated chunking view PDF at {stats.output_pdf}")
    print(f"  pages: {stats.page_count}")
    print(f"  chunks: {stats.chunk_count}")
    print(f"  chunks with overlays: {stats.chunks_with_regions}")
    print(f"  pages with overlays: {stats.pages_with_regions}")
    print(f"  overlays: {stats.region_count}")


if __name__ == "__main__":
    main()
