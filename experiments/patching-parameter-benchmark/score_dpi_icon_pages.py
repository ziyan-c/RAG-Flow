#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ICON_RE = re.compile(r"\[icon\s*:\s*([^\]]+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class ScoreCase:
    chrome_page: int
    page_idx: int
    block_idx: int
    field_key: str
    label: str
    expected_groups: tuple[tuple[str, ...], ...]
    reference: str


CASES = (
    ScoreCase(
        chrome_page=313,
        page_idx=312,
        block_idx=4313,
        field_key="text",
        label="tracks/favorites icons",
        expected_groups=(("star",), ("badge", "notification", "red")),
        reference="Click [Icon: star] ... click [Icon: starred record with red badge] at the upper-right corner ...",
    ),
    ScoreCase(
        chrome_page=313,
        page_idx=312,
        block_idx=4316,
        field_key="list_items",
        label="record operation list",
        expected_groups=(
            ("face", "arming"),
            ("track", "playback"),
            ("filter", "search"),
            ("vehicle", "arming"),
            ("delete", "trash"),
        ),
        reference="Recover face arming group, track playback, filter/search, vehicle arming group, and delete icons.",
    ),
    ScoreCase(
        chrome_page=313,
        page_idx=312,
        block_idx=4319,
        field_key="text",
        label="resume upload action",
        expected_groups=(("upload", "resume"),),
        reference="Click [Icon: resume/upload] at the upper right.",
    ),
    ScoreCase(
        chrome_page=320,
        page_idx=319,
        block_idx=4433,
        field_key="list_items",
        label="case add/back actions",
        expected_groups=(("add", "plus"), ("back", "left", "arrow")),
        reference="Click [Icon: add/plus] next to the record; click [Icon: back/left arrow] to go back.",
    ),
    ScoreCase(
        chrome_page=320,
        page_idx=319,
        block_idx=4434,
        field_key="text",
        label="attachment action",
        expected_groups=(("attachment", "link"),),
        reference="Step 8 Click [Icon: attachment/link], then click Add under Attachment.",
    ),
    ScoreCase(
        chrome_page=320,
        page_idx=319,
        block_idx=4439,
        field_key="list_items",
        label="related operation icons",
        expected_groups=(
            ("eye", "view"),
            ("delete", "minus"),
            ("upload",),
            ("search",),
            ("eye", "view"),
            ("download",),
            ("trash", "delete"),
            ("toggle", "close", "open"),
        ),
        reference="Recover view, delete/minus, upload, search, download, trash, and toggle icons.",
    ),
    ScoreCase(
        chrome_page=435,
        page_idx=434,
        block_idx=6210,
        field_key="table_body",
        label="interface operation table",
        expected_groups=(
            ("close", "all"),
            ("split", "screen"),
            ("snapshot", "camera"),
            ("close", "window"),
            ("stop", "pause"),
            ("speed", "slow", "fast"),
            ("frame",),
            ("search", "snapshot"),
        ),
        reference="Recover semantic table icons: close-all, split-screen, snapshot, close-window, stop/pause, speed, frame-by-frame, and search-by-snapshot.",
    ),
    ScoreCase(
        chrome_page=435,
        page_idx=434,
        block_idx=6212,
        field_key="text",
        label="plain settings paragraph",
        expected_groups=(),
        reference="No missing icons. Keep the original paragraph unchanged.",
    ),
)


def join_text(value: Any) -> str:
    if isinstance(value, list):
        return " / ".join(str(item) for item in value)
    return str(value or "")


def normalize_text(text: str) -> str:
    text = ICON_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def shorten(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).replace("\n", " / ").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def icon_labels(text: str) -> list[str]:
    return [match.strip().lower().replace("_", " ") for match in ICON_RE.findall(text)]


def group_hit(labels: list[str], group: tuple[str, ...]) -> bool:
    for label in labels:
        if all(term in label for term in group):
            return True
    return False


def read_successful_runs(bench_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jsonl = bench_root / "results.jsonl"
    if not jsonl.exists():
        return rows
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("status", 1)) == 0:
            rows.append(row)
    return rows


def index_blocks(blocks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for idx, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        global_idx = block.get("benchmark_global_idx", idx)
        indexed[int(global_idx)] = block
    return indexed


def score_run(raw_by_idx: dict[int, dict[str, Any]], run: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_path = Path(run["output_json"])
    patched_by_idx = index_blocks(json.loads(output_path.read_text(encoding="utf-8")))
    detail_rows: list[dict[str, Any]] = []
    expected_total = 0
    expected_hits = 0
    unexpected_markers = 0
    no_missing_total = 0
    no_missing_passed = 0
    retention_values: list[float] = []

    for case in CASES:
        raw_block = raw_by_idx[case.block_idx]
        patched_block = patched_by_idx[case.block_idx]
        original = join_text(raw_block.get(case.field_key, ""))
        patched = join_text(patched_block.get(case.field_key, ""))
        labels = icon_labels(patched)
        hits = sum(1 for group in case.expected_groups if group_hit(labels, group))
        expected = len(case.expected_groups)
        extra = max(0, len(labels) - hits)
        retention = difflib.SequenceMatcher(None, normalize_text(original), normalize_text(patched)).ratio()
        no_missing_ok = ""
        if expected == 0:
            no_missing_total += 1
            no_missing_ok = "yes" if len(labels) == 0 and retention >= 0.97 else "no"
            if no_missing_ok == "yes":
                no_missing_passed += 1
        expected_total += expected
        expected_hits += hits
        unexpected_markers += extra
        retention_values.append(retention)
        detail_rows.append(
            {
                "dpi": run["dpi"],
                "name": run["name"],
                "chrome_page": case.chrome_page,
                "page_idx": case.page_idx,
                "block_idx": case.block_idx,
                "field": case.field_key,
                "label": case.label,
                "expected_icons": expected,
                "expected_hits": hits,
                "marker_count": len(labels),
                "unexpected_marker_proxy": extra,
                "text_retention_ratio": round(retention, 4),
                "no_missing_ok": no_missing_ok,
                "reference": case.reference,
                "original_ocr": shorten(original),
                "patched_output": shorten(patched),
                "icon_labels": "; ".join(labels),
            }
        )

    no_missing_violations = no_missing_total - no_missing_passed
    summary = {
        "dpi": run["dpi"],
        "name": run["name"],
        "elapsed_sec": run.get("elapsed_sec", ""),
        "requests_submitted": run.get("requests_submitted", ""),
        "requests_per_sec": run.get("requests_per_sec", ""),
        "expected_icon_groups": expected_total,
        "expected_icon_hits": expected_hits,
        "expected_icon_recall": round(expected_hits / expected_total, 4) if expected_total else 0,
        "unexpected_marker_proxy": unexpected_markers,
        "no_missing_passed": no_missing_passed,
        "no_missing_total": no_missing_total,
        "avg_text_retention": round(sum(retention_values) / len(retention_values), 4),
        "quality_proxy": round(expected_hits - 0.25 * unexpected_markers - 2.0 * no_missing_violations, 3),
        "output_json": run["output_json"],
    }
    return summary, detail_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the three-page DPI icon benchmark outputs.")
    parser.add_argument("bench_root", type=Path)
    args = parser.parse_args()

    dataset = json.loads((args.bench_root / "dataset.json").read_text(encoding="utf-8"))
    raw_blocks = json.loads(Path(dataset["input_json"]).read_text(encoding="utf-8"))
    raw_by_idx = index_blocks(raw_blocks)

    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for run in read_successful_runs(args.bench_root):
        summary, rows = score_run(raw_by_idx, run)
        summaries.append(summary)
        details.extend(rows)

    summaries.sort(key=lambda row: int(row["dpi"]))
    details.sort(key=lambda row: (int(row["dpi"]), int(row["chrome_page"]), int(row["block_idx"])))
    write_csv(args.bench_root / "quality-scores.csv", summaries)
    write_csv(args.bench_root / "quality-details.csv", details)
    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
