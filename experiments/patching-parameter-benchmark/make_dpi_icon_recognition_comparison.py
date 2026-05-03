#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH = Path(os.environ.get("RAG_FLOW_DPI_BENCH_DIR", ROOT / "data-icon-pages"))
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
ICON_RE = re.compile(r"\[icon\s*:\s*([^\]]+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class FieldReview:
    reference_page: int
    block_idx: int
    field: str
    label: str
    visual_reference: str
    scope: str
    note: str
    judgements: dict[int, str]


FIELD_REVIEWS = (
    FieldReview(
        313,
        4311,
        "text",
        "search continuation text",
        "No UI icon needs to be inserted.",
        "all",
        "250 and 300 DPI leave the paragraph unchanged; 200 DPI hallucinates a manual-select icon.",
        {200: "wrong", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        313,
        4312,
        "list_items",
        "descriptive bullet list",
        "No UI icon needs to be inserted; bullets should not become icons.",
        "all",
        "All DPI settings hallucinate icons from bullets or surrounding layout.",
        {200: "wrong", 250: "wrong", 300: "wrong"},
    ),
    FieldReview(
        313,
        4313,
        "text",
        "tracks/favorites star icons",
        "Two missing icons: star before add-to-tracks/favorites, and starred record with red notification badge at the upper right.",
        "icon_required",
        "All three DPIs recover the two star icons. 300 DPI gives the most descriptive second label.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        313,
        4314,
        "text",
        "section bullet mistaken as icon",
        "No UI icon needs to be inserted.",
        "all",
        "All DPI settings insert a bullet icon that is not useful for retrieval.",
        {200: "wrong", 250: "wrong", 300: "wrong"},
    ),
    FieldReview(
        313,
        4315,
        "text",
        "temporary records star icons",
        "Two missing star/favorite icons should be inserted around the temporary-record action.",
        "icon_required",
        "All DPI settings recover the star/favorite icon pair.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        313,
        4316,
        "list_items",
        "record operation list",
        "Recover face arming group, track playback, filter/search, vehicle arming group, and delete icons.",
        "icon_required",
        "200 and 300 DPI recover the arming-group semantics more clearly; 250 uses face_add and vehicle_add.",
        {200: "correct", 250: "partial", 300: "correct"},
    ),
    FieldReview(
        313,
        4317,
        "text",
        "deletion limitation note",
        "No UI icon needs to be inserted.",
        "all",
        "All DPI settings correctly leave the paragraph unchanged.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        313,
        4318,
        "text",
        "offline extraction paragraph",
        "No UI icon needs to be inserted; the OCR has broken the word offline.",
        "all",
        "All three DPI settings incorrectly turn the broken word offline into icons.",
        {200: "wrong", 250: "wrong", 300: "wrong"},
    ),
    FieldReview(
        313,
        4319,
        "text",
        "resume upload action",
        "The missing upper-right icon means resume/upload, not a generic menu.",
        "icon_required",
        "250 DPI outputs upload; 200 and 300 call the icon menu and also add markdown styling.",
        {200: "wrong", 250: "correct", 300: "wrong"},
    ),
    FieldReview(
        313,
        4320,
        "list_items",
        "auto extraction bullet list",
        "No UI icon needs to be inserted; list bullets should remain list markers.",
        "all",
        "All DPI settings turn list bullets into diamond icons.",
        {200: "wrong", 250: "wrong", 300: "wrong"},
    ),
    FieldReview(
        313,
        4321,
        "text",
        "extraction configuration sentence",
        "No UI icon needs to be inserted.",
        "all",
        "All DPI settings correctly leave the sentence unchanged.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        320,
        4433,
        "list_items",
        "case add/back actions",
        "Recover add/plus next to the record and back/left-arrow for returning to the case adding page.",
        "icon_required",
        "All three DPIs recover plus and left arrow, and all repair the OCR formula noise around next.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        320,
        4434,
        "text",
        "attachment action",
        "Step 8 should click the attachment/link icon before adding attachments.",
        "icon_required",
        "250 DPI recognizes attachment; 200 and 300 mislabel the icon as refresh.",
        {200: "wrong", 250: "correct", 300: "wrong"},
    ),
    FieldReview(
        320,
        4435,
        "list_items",
        "upload limits and file extensions",
        "No UI icon needs to be inserted; file extension OCR noise should not become icons.",
        "all",
        "All DPI settings hallucinate icons from the corrupted .flv/file-extension text.",
        {200: "wrong", 250: "wrong", 300: "wrong"},
    ),
    FieldReview(
        320,
        4436,
        "text",
        "file-count limit note",
        "No UI icon needs to be inserted.",
        "all",
        "All DPI settings correctly leave the sentence unchanged.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        320,
        4437,
        "text",
        "step 9 ok instruction",
        "No UI icon needs to be inserted.",
        "all",
        "All DPI settings correctly leave the step unchanged.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        320,
        4438,
        "text",
        "related operations heading",
        "No UI icon needs to be inserted.",
        "all",
        "All DPI settings correctly leave the heading unchanged.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        320,
        4439,
        "list_items",
        "case related-operation icons",
        "Recover view, delete/minus, upload, search, view, download, trash/delete, and toggle icons.",
        "icon_required",
        "All three DPIs recover the main operation-icon sequence with nearly identical labels.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        320,
        4440,
        "text",
        "access management section heading",
        "No UI icon needs to be inserted.",
        "all",
        "All DPI settings correctly leave the heading unchanged.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        320,
        4441,
        "text",
        "access management overview",
        "No UI icon needs to be inserted.",
        "all",
        "All DPI settings correctly leave the paragraph unchanged.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        320,
        4442,
        "text",
        "access control subsection heading",
        "No UI icon needs to be inserted.",
        "all",
        "All DPI settings correctly leave the heading unchanged.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        435,
        6209,
        "text",
        "related operations heading",
        "No UI icon needs to be inserted.",
        "all",
        "All DPI settings correctly leave the heading unchanged.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        435,
        6210,
        "table_body",
        "interface operation table",
        "Replace corrupted Icon/Function cells with close-all, split-screen, snapshot, close-window, stop/pause, speed, frame-by-frame, and search-by-snapshot labels.",
        "icon_required",
        "All three recover table icons, but 300 DPI preserves the surrounding table HTML/OCR text most faithfully.",
        {200: "partial", 250: "partial", 300: "correct"},
    ),
    FieldReview(
        435,
        6211,
        "text",
        "local settings section heading",
        "No UI icon needs to be inserted.",
        "all",
        "All DPI settings correctly leave the heading unchanged.",
        {200: "correct", 250: "correct", 300: "correct"},
    ),
    FieldReview(
        435,
        6212,
        "text",
        "local settings overview paragraph",
        "No missing icons; the paragraph should remain unchanged.",
        "all",
        "300 DPI correctly leaves the paragraph unchanged; 200 and 250 hallucinate category icons.",
        {200: "wrong", 250: "wrong", 300: "correct"},
    ),
)


STATUS_LABELS = {
    "correct": "Correct",
    "partial": "Partial",
    "wrong": "Wrong",
}


STATUS_COLORS = {
    "correct": "#16a34a",
    "partial": "#f59e0b",
    "wrong": "#dc2626",
}


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def index_blocks(blocks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for idx, block in enumerate(blocks):
        if isinstance(block, dict):
            indexed[int(block.get("benchmark_global_idx", idx))] = block
    return indexed


def join_text(value: Any) -> str:
    if isinstance(value, list):
        return " / ".join(str(item) for item in value)
    return str(value or "")


def one_line(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def shorten(text: str, limit: int = 220) -> str:
    text = one_line(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def icon_labels(text: str) -> str:
    return "; ".join(label.strip().replace("_", " ") for label in ICON_RE.findall(text))


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "400", fill: str = "#111827") -> str:
    import html

    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{html.escape(text)}</text>"
    )


def summarize_rows(rows: list[dict[str, str]]) -> dict[int, dict[str, int]]:
    summary: dict[int, dict[str, int]] = {
        dpi: {"correct": 0, "partial": 0, "wrong": 0}
        for dpi in (200, 250, 300)
    }
    for row in rows:
        for dpi in (200, 250, 300):
            summary[dpi][row[f"dpi_{dpi}_judgement"]] += 1
    return summary


def write_ratio_outputs(rows: list[dict[str, str]]) -> None:
    summary_groups = {
        "all_submitted_fields": rows,
        "icon_required_fields": [row for row in rows if row["scope"] == "icon_required"],
    }
    ratio_rows: list[dict[str, str]] = []
    summaries = {name: summarize_rows(group_rows) for name, group_rows in summary_groups.items()}
    for scope, by_dpi in summaries.items():
        for dpi in (200, 250, 300):
            total = sum(by_dpi[dpi].values())
            ratio_rows.append(
                {
                    "scope": scope,
                    "dpi": str(dpi),
                    "total_cases": str(total),
                    "correct": str(by_dpi[dpi]["correct"]),
                    "partial": str(by_dpi[dpi]["partial"]),
                    "wrong": str(by_dpi[dpi]["wrong"]),
                    "correct_ratio": f"{by_dpi[dpi]['correct'] / total:.4f}",
                    "partial_ratio": f"{by_dpi[dpi]['partial'] / total:.4f}",
                    "wrong_ratio": f"{by_dpi[dpi]['wrong'] / total:.4f}",
                }
            )
    with (BENCH / "recognition-ratio.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ratio_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ratio_rows)

    width, height = 900, 610
    left, right, top, bottom = 128, 48, 92, 64
    plot_w = width - left - right
    bar_h = 36
    row_gap = 48
    panel_gap = 76
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(24, 32, "DPI Recognition Accuracy by Field Scope", 20, weight="700"),
        svg_text(24, 56, "Manual review on target technical manual reference pages 313, 320, and 435", 12, fill="#4b5563"),
    ]
    panel_specs = (
        ("all_submitted_fields", "Overall field stability (n=25)", "Includes ordinary text fields; tests hallucination control."),
        ("icon_required_fields", "Icon recovery accuracy (n=8)", "Only fields where a missing UI icon should be recovered."),
    )
    y = top
    for scope, title, subtitle in panel_specs:
        by_dpi = summaries[scope]
        out.append(svg_text(24, y - 18, title, 14, weight="700"))
        out.append(svg_text(230, y - 18, subtitle, 11, fill="#4b5563"))
        for i, dpi in enumerate((200, 250, 300)):
            row_y = y + i * row_gap
            out.append(svg_text(left - 18, row_y + bar_h / 2 + 5, f"{dpi} DPI", 12, anchor="end", weight="700"))
            x = left
            total = sum(by_dpi[dpi].values())
            for status in ("correct", "partial", "wrong"):
                count = by_dpi[dpi][status]
                w = plot_w * count / total
                out.append(f'<rect x="{x:.1f}" y="{row_y:.1f}" width="{w:.1f}" height="{bar_h}" fill="{STATUS_COLORS[status]}" rx="4"/>')
                if w >= 56:
                    out.append(svg_text(x + w / 2, row_y + bar_h / 2 + 4, f"{count}/{total}", 12, anchor="middle", weight="700", fill="#ffffff"))
                x += w
        y += 3 * row_gap + panel_gap
    legend_x = left
    legend_y = height - 26
    for i, status in enumerate(("correct", "partial", "wrong")):
        x = legend_x + i * 150
        out.append(f'<rect x="{x}" y="{legend_y - 13}" width="14" height="14" fill="{STATUS_COLORS[status]}" rx="3"/>')
        out.append(svg_text(x + 22, legend_y, STATUS_LABELS[status], 12))
    out.append("</svg>")
    (ASSETS / "chart-dpi-recognition-ratio.svg").write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    raw = index_blocks(load_json(BENCH / "inputs" / "chrome_pages_313_320_435_content_list.json"))
    patched = {
        200: index_blocks(load_json(BENCH / "runs" / "d200" / "d200_PATCHED.json")),
        250: index_blocks(load_json(BENCH / "runs" / "d250" / "d250_PATCHED.json")),
        300: index_blocks(load_json(BENCH / "runs" / "d300" / "d300_PATCHED.json")),
    }

    full_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    for case in FIELD_REVIEWS:
        original = join_text(raw[case.block_idx].get(case.field, ""))
        outputs = {
            dpi: join_text(patched[dpi][case.block_idx].get(case.field, ""))
            for dpi in (200, 250, 300)
        }
        base = {
            "reference_page": str(case.reference_page),
            "block_idx": str(case.block_idx),
            "field": case.field,
            "label": case.label,
            "scope": case.scope,
            "visual_reference": case.visual_reference,
            "recognition_note": case.note,
        }
        full_rows.append(
            {
                **base,
                "original_ocr": original,
                "dpi_200_output": outputs[200],
                "dpi_250_output": outputs[250],
                "dpi_300_output": outputs[300],
                "dpi_200_icons": icon_labels(outputs[200]),
                "dpi_250_icons": icon_labels(outputs[250]),
                "dpi_300_icons": icon_labels(outputs[300]),
                "dpi_200_judgement": case.judgements[200],
                "dpi_250_judgement": case.judgements[250],
                "dpi_300_judgement": case.judgements[300],
            }
        )
        summary_rows.append(
            {
                **base,
                "original_ocr": shorten(original),
                "dpi_200_output": shorten(outputs[200]),
                "dpi_250_output": shorten(outputs[250]),
                "dpi_300_output": shorten(outputs[300]),
                "dpi_200_icons": shorten(icon_labels(outputs[200]), 140),
                "dpi_250_icons": shorten(icon_labels(outputs[250]), 140),
                "dpi_300_icons": shorten(icon_labels(outputs[300]), 140),
                "dpi_200_judgement": case.judgements[200],
                "dpi_250_judgement": case.judgements[250],
                "dpi_300_judgement": case.judgements[300],
            }
        )

    for path, rows in (
        (BENCH / "recognition-comparison-full.csv", full_rows),
        (BENCH / "recognition-comparison-summary.csv", summary_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    write_ratio_outputs(full_rows)


if __name__ == "__main__":
    main()
