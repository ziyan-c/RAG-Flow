from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_flow.config import AppConfig
from rag_flow.preprocessing import small_icons
from rag_flow.preprocessing.image_descriptions import (
    DEFAULT_CAPTION_MAX_CONTEXT_TOKENS,
    ApproxTokenBudgeter,
    ImageDescriptionArtifacts,
    TextBudgeter,
    collect_surrounding_context_selection,
    resolve_image_description_batch,
    should_caption_image_block,
)


REGION_COLORS = {
    "caption_target": (0.95, 0.10, 0.10),
    "context_before": (0.10, 0.35, 0.95),
    "context_current": (0.65, 0.20, 0.85),
    "context_after": (0.15, 0.65, 0.25),
}


@dataclass(frozen=True)
class CaptioningViewRegion:
    page_idx: int
    bbox: tuple[float, float, float, float]
    field: str
    label: str
    target_idx: int


@dataclass(frozen=True)
class CaptioningViewPlan:
    regions: tuple[CaptioningViewRegion, ...]
    field_counts: dict[str, int]
    caption_targets: int


@dataclass(frozen=True)
class CaptioningViewStats:
    output_pdf: Path
    page_count: int
    pages_with_regions: int
    region_count: int
    field_counts: dict[str, int]
    caption_targets: int


def captioning_view_path_for(content_json: str | Path) -> Path:
    path = Path(content_json)
    name = path.name
    for suffix in (
        "_content_list_SECTIONED_PATCHED_CAPTIONED.json",
        "_content_list_SECTIONED_PATCHED.json",
        "_content_list_SECTIONED.json",
        "_content_list_PATCHED_CAPTIONED.json",
        "_content_list_PATCHED.json",
        "_content_list.json",
    ):
        if name.endswith(suffix):
            return path.with_name(f"{name[: -len(suffix)]}_CAPTIONING_VIEW.pdf")
    return path.with_name(f"{path.stem}_CAPTIONING_VIEW.pdf")


def _add_region(
    regions: list[CaptioningViewRegion],
    field_counts: Counter[str],
    *,
    content_data: list[dict[str, Any]],
    block_idx: int,
    field: str,
    label: str,
    target_idx: int,
) -> None:
    block = content_data[block_idx]
    bbox = small_icons._block_bbox(block)
    if bbox is None:
        return
    regions.append(
        CaptioningViewRegion(
            page_idx=small_icons._block_page_idx(block),
            bbox=bbox,
            field=field,
            label=label,
            target_idx=target_idx,
        )
    )
    field_counts[field] += 1


def collect_captioning_view_regions(
    content_data: list[dict[str, Any]],
    *,
    max_context_tokens: int = DEFAULT_CAPTION_MAX_CONTEXT_TOKENS,
    budgeter: TextBudgeter | None = None,
) -> CaptioningViewPlan:
    regions: list[CaptioningViewRegion] = []
    field_counts: Counter[str] = Counter()
    caption_targets = 0
    budgeter = budgeter or ApproxTokenBudgeter()

    for idx, block in enumerate(content_data):
        if not isinstance(block, dict) or not should_caption_image_block(block):
            continue
        caption_targets += 1
        _add_region(
            regions,
            field_counts,
            content_data=content_data,
            block_idx=idx,
            field="caption_target",
            label=f"{idx}:image",
            target_idx=idx,
        )
        _context, selection = collect_surrounding_context_selection(
            content_data,
            idx,
            max_context_tokens=max_context_tokens,
            budgeter=budgeter,
        )
        for context_idx in selection.before_indices:
            _add_region(
                regions,
                field_counts,
                content_data=content_data,
                block_idx=context_idx,
                field="context_before",
                label=f"{idx}:before:{context_idx}",
                target_idx=idx,
            )
        for context_idx in selection.current_indices:
            _add_region(
                regions,
                field_counts,
                content_data=content_data,
                block_idx=context_idx,
                field="context_current",
                label=f"{idx}:current:{context_idx}",
                target_idx=idx,
            )
        for context_idx in selection.after_indices:
            _add_region(
                regions,
                field_counts,
                content_data=content_data,
                block_idx=context_idx,
                field="context_after",
                label=f"{idx}:after:{context_idx}",
                target_idx=idx,
            )

    return CaptioningViewPlan(
        regions=tuple(regions),
        field_counts=dict(field_counts),
        caption_targets=caption_targets,
    )


def _draw_legend(page: Any, stats: CaptioningViewStats, *, max_context_tokens: int) -> None:
    import fitz

    legend_x = 28
    legend_y = 28
    page.draw_rect(
        fitz.Rect(legend_x - 8, legend_y - 12, legend_x + 250, legend_y + 108),
        color=(0.1, 0.1, 0.1),
        fill=(1, 1, 1),
        width=0.5,
        overlay=True,
        fill_opacity=0.85,
        stroke_opacity=0.5,
    )
    page.insert_text((legend_x, legend_y), "CAPTIONING VIEW", fontsize=10, color=(0, 0, 0), overlay=True)
    page.insert_text(
        (legend_x, legend_y + 14),
        f"context budget: {max_context_tokens} tokens",
        fontsize=7,
        color=(0, 0, 0),
        overlay=True,
    )
    legend_items = [
        ("caption_target", "caption target image"),
        ("context_before", "nearby context before image"),
        ("context_current", "image caption/footnote text"),
        ("context_after", "nearby context after image"),
    ]
    for row, (field, label) in enumerate(legend_items, start=2):
        y = legend_y + row * 16
        color = REGION_COLORS[field]
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


def write_captioning_view_pdf(
    *,
    content_json: str | Path,
    pdf_path: str | Path,
    output_pdf: str | Path | None = None,
    max_context_tokens: int = DEFAULT_CAPTION_MAX_CONTEXT_TOKENS,
    budgeter: TextBudgeter | None = None,
) -> CaptioningViewStats:
    import fitz

    content_path = Path(content_json)
    output_path = Path(output_pdf).expanduser() if output_pdf else captioning_view_path_for(content_path)
    content_data: list[dict[str, Any]] = json.loads(content_path.read_text(encoding="utf-8"))
    plan = collect_captioning_view_regions(
        content_data,
        max_context_tokens=max_context_tokens,
        budgeter=budgeter,
    )
    regions_by_page: dict[int, list[CaptioningViewRegion]] = defaultdict(list)
    for region in plan.regions:
        regions_by_page[region.page_idx].append(region)

    doc = fitz.open(Path(pdf_path))
    stats = CaptioningViewStats(
        output_pdf=output_path,
        page_count=len(doc),
        pages_with_regions=len(regions_by_page),
        region_count=len(plan.regions),
        field_counts=plan.field_counts,
        caption_targets=plan.caption_targets,
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
            color = REGION_COLORS.get(region.field, (0.2, 0.2, 0.2))
            page.draw_rect(
                rect,
                color=color,
                fill=color,
                width=1.0 if region.field == "caption_target" else 0.6,
                overlay=True,
                fill_opacity=0.24 if region.field == "caption_target" else 0.13,
                stroke_opacity=0.95,
            )
            if region.field == "caption_target" and rect.width >= 24 and rect.height >= 12:
                page.insert_textbox(
                    rect + (2, 2, 110, 16),
                    f"image {region.target_idx}",
                    fontsize=6,
                    color=color,
                    overlay=True,
                )

        if page_idx == 0:
            _draw_legend(page, stats, max_context_tokens=max_context_tokens)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    return stats


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Draw captioning image targets and context blocks over a source PDF.")
    parser.add_argument("--artifact-dir", help="MinerU output folder containing patched content_list JSON.")
    parser.add_argument(
        "--input-json",
        default=None,
        help="Patched content_list JSON used as captioning input.",
    )
    parser.add_argument("--input-pdf", default=None, help="Source PDF used by captioning.")
    parser.add_argument("--output", default=None, help="Output CAPTIONING_VIEW PDF.")
    parser.add_argument("--max-context-tokens", type=int, default=config.captioning.max_context_tokens)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths without writing the PDF.")
    args = parser.parse_args(argv)

    if args.artifact_dir:
        if args.input_json or args.input_pdf:
            parser.error("--artifact-dir cannot be combined with --input-json or --input-pdf.")
        artifacts_list = resolve_image_description_batch(args.artifact_dir, recursive=not args.no_recursive)
    else:
        input_json = Path(args.input_json).expanduser() if args.input_json else config.paths.patched_json
        input_pdf = Path(args.input_pdf).expanduser() if args.input_pdf else config.paths.source_pdf
        artifacts_list = [
            ImageDescriptionArtifacts(
                artifact_dir=input_json.parent,
                base_dir=input_json.parent,
                input_json=input_json,
                output_json=input_json,
                origin_pdf=input_pdf,
            )
        ]

    if len(artifacts_list) > 1 and args.output:
        parser.error("--output can only be used with a single captioning view job.")

    if args.dry_run:
        print(f"Captioning view jobs: {len(artifacts_list)}")
        for artifacts in artifacts_list:
            output = Path(args.output).expanduser() if args.output else captioning_view_path_for(artifacts.input_json)
            print("Captioning view inputs:")
            print(f"  artifact_dir: {artifacts.artifact_dir}")
            print(f"  input_json: {artifacts.input_json}")
            print(f"  input_pdf: {artifacts.origin_pdf}")
            print(f"  output_pdf: {output}")
            print(f"  max_context_tokens: {args.max_context_tokens}")
        return

    for job_idx, artifacts in enumerate(artifacts_list, start=1):
        output = Path(args.output).expanduser() if args.output else captioning_view_path_for(artifacts.input_json)
        print(f"Captioning view job {job_idx}/{len(artifacts_list)}: {artifacts.artifact_dir}")
        stats = write_captioning_view_pdf(
            content_json=artifacts.input_json,
            pdf_path=artifacts.origin_pdf,
            output_pdf=output,
            max_context_tokens=args.max_context_tokens,
        )
        print(f"Generated captioning view PDF at {stats.output_pdf}")
        print(f"  pages: {stats.page_count}")
        print(f"  pages with overlays: {stats.pages_with_regions}")
        print(f"  overlays: {stats.region_count}")
        print(f"  caption targets: {stats.caption_targets}")


if __name__ == "__main__":
    main()
