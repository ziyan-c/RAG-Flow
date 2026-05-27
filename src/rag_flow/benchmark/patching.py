from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from rag_flow.config import AppConfig
from rag_flow.preprocessing.small_icons import (
    BENCHMARK_ORIGINAL_INDEX_KEY,
    _patch_field_keys,
    add_small_icon_text,
    build_inline_icon_links,
    build_table_continuation_map,
)


DEFAULT_MAIN_PAGES = "50-250"
DEFAULT_QUALITY_PAGES = "52,76,81,90,107,123,137,150,158,173,178,196,213,228,244,249,250,258,287,313,316,340,419"
DEFAULT_OUTPUT_DIR = Path("thesis-v2/04-patching/data/benchmark-runs")
DEFAULT_QUALITY_SAMPLE_SIZE = 120
ACTION_PATTERN = re.compile(
    r"\b(click|select|choose|tap|press|hover|open|close|delete|edit|add|upload|download|import|export|save|send)\b",
    re.IGNORECASE,
)
ICON_TAG_PATTERN = re.compile(r"\[Icon:[^\]]+\]", re.IGNORECASE)
MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*([^*]+)\*\*")


@dataclass(frozen=True)
class RunSpec:
    stage: str
    name: str
    dpi: int
    concurrency: int
    batch_size: int
    checkpoint_interval: int
    repeat_index: int = 1

    @property
    def run_id(self) -> str:
        repeat_suffix = f"_r{self.repeat_index}" if self.repeat_index > 1 else ""
        return (
            f"{self.stage}_{self.name}_"
            f"d{self.dpi}_c{self.concurrency}_b{self.batch_size}_k{self.checkpoint_interval}{repeat_suffix}"
        )


class JsonlMetricsSink:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._lock = threading.Lock()
        self._paths = {
            "request": run_dir / "requests.jsonl",
            "render": run_dir / "render.jsonl",
            "checkpoint": run_dir / "checkpoints.jsonl",
        }

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        path = self._paths.get(kind, self.run_dir / f"{kind}.jsonl")
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


class GpuSampler:
    def __init__(self, output_csv: Path, *, interval_s: float):
        self.output_csv = output_csv
        self.interval_s = max(0.1, interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuSampler":
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sample_time",
                    "nvidia_timestamp",
                    "gpu_index",
                    "gpu_utilization_pct",
                    "memory_used_mib",
                    "memory_total_mib",
                    "power_w",
                    "error",
                ]
            )
        if shutil.which("nvidia-smi") is None:
            self._write_error("nvidia-smi not found")
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _write_error(self, message: str) -> None:
        with self.output_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), "", "", "", "", "", "", message])

    def _run(self) -> None:
        while not self._stop.is_set():
            command = [
                "nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ]
            try:
                result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
                if result.returncode != 0:
                    self._write_error(result.stderr.strip() or f"nvidia-smi exited {result.returncode}")
                else:
                    sample_time = time.time()
                    with self.output_csv.open("a", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        for line in result.stdout.splitlines():
                            parts = [part.strip() for part in line.split(",")]
                            writer.writerow([sample_time, *parts, ""])
            except Exception as exc:  # pragma: no cover - hardware dependent
                self._write_error(str(exc))
            self._stop.wait(self.interval_s)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return ""


def collect_environment_metadata() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]

    def git_value(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    git_status_short = git_value("status", "--short")
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "rag_flow_git_commit": git_value("rev-parse", "HEAD"),
        "rag_flow_git_dirty": bool(git_status_short.strip()),
        "rag_flow_git_status_short_preview": "\n".join(git_status_short.splitlines()[:20]),
        "sglang_version": _package_version("sglang"),
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "openai_version": _package_version("openai"),
    }


def parse_pdf_pages(spec: str) -> set[int]:
    pages: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if start < 1 or end < start:
                raise ValueError(f"Invalid PDF page range: {part}")
            pages.update(range(start - 1, end))
        else:
            page = int(part)
            if page < 1:
                raise ValueError(f"Invalid PDF page: {part}")
            pages.add(page - 1)
    return pages


def parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if part:
            values.append(int(part))
    return values


def parse_batch_size_spec(spec: str, *, concurrency: int) -> list[int]:
    values: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        normalized = part.upper()
        if normalized == "C":
            values.append(concurrency)
        elif normalized.endswith("C"):
            factor = int(normalized[:-1])
            values.append(factor * concurrency)
        else:
            values.append(int(part))
    return sorted(set(values))


def _join_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def _preview(text: str, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _load_content(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in MinerU content JSON: {path}")
    return data


def build_subset(
    content_data: list[dict[str, Any]],
    *,
    page_indices: set[int],
) -> tuple[list[dict[str, Any]], set[int], list[int]]:
    working_data = copy.deepcopy(content_data)
    selected_indices = {
        idx
        for idx, block in enumerate(working_data)
        if isinstance(block, dict) and int(block.get("page_idx", -1)) in page_indices
    }

    table_continuations = build_table_continuation_map(working_data)
    reverse_continuations = {
        continuation_idx: master_idx
        for master_idx, continuation_indices in table_continuations.items()
        for continuation_idx in continuation_indices
    }
    for idx in list(selected_indices):
        selected_indices.update(table_continuations.get(idx, []))
        if idx in reverse_continuations:
            selected_indices.add(reverse_continuations[idx])

    inline_links = build_inline_icon_links(working_data, table_continuations)
    for idx in list(selected_indices):
        for link in inline_links.by_target.get(idx, []):
            selected_indices.add(link.icon_idx)
        if idx in inline_links.by_icon:
            selected_indices.add(inline_links.by_icon[idx].target_idx)

    selected_pages = {
        int(working_data[idx].get("page_idx", 0))
        for idx in selected_indices
        if isinstance(working_data[idx], dict)
    }
    subset: list[dict[str, Any]] = []
    for idx in sorted(selected_indices):
        block = working_data[idx]
        if not isinstance(block, dict):
            continue
        copied = copy.deepcopy(block)
        copied[BENCHMARK_ORIGINAL_INDEX_KEY] = idx
        subset.append(copied)
    return subset, selected_pages, sorted(selected_indices)


def write_subset_json(
    *,
    content_data: list[dict[str, Any]],
    page_indices: set[int],
    output_path: Path,
) -> tuple[list[dict[str, Any]], set[int], list[int]]:
    subset, selected_pages, selected_indices = build_subset(content_data, page_indices=page_indices)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    return subset, selected_pages, selected_indices


def write_quality_scoring_template(
    *,
    content_data: list[dict[str, Any]],
    main_pages: set[int],
    quality_pages: set[int],
    output_csv: Path,
    negative_controls: int,
    seed: int = 20260504,
) -> None:
    rows: list[dict[str, Any]] = []
    negative_pool: list[dict[str, Any]] = []
    for idx, block in enumerate(content_data):
        if not isinstance(block, dict):
            continue
        page_idx = int(block.get("page_idx", -1))
        field_keys = _patch_field_keys(block)
        if not field_keys:
            continue
        for field in field_keys:
            text = _join_text(block.get(field, "")).strip()
            if not text:
                continue
            row = {
                "sample_label": "quality_page",
                "pdf_page": page_idx + 1,
                "page_idx": page_idx,
                "block_idx": idx,
                "field": field,
                "block_type": block.get("type", ""),
                "text_preview": _preview(text),
                "target_icons": "",
                "strict_hits": "",
                "wrong_icons": "",
                "missed_targets": "",
                "false_positives": "",
                "overall_patch_quality_score": "",
                "text_preserved": "",
                "human_notes": "",
            }
            if page_idx in quality_pages:
                rows.append(row)
            elif page_idx in main_pages and not ACTION_PATTERN.search(text):
                negative_pool.append({**row, "sample_label": "negative_control"})

    rng = random.Random(seed)
    rng.shuffle(negative_pool)
    rows.extend(negative_pool[: max(0, negative_controls)])
    rows.sort(key=lambda item: (int(item["pdf_page"]), int(item["block_idx"]), str(item["field"])))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_label",
        "pdf_page",
        "page_idx",
        "block_idx",
        "field",
        "block_type",
        "text_preview",
        "target_icons",
        "strict_hits",
        "wrong_icons",
        "missed_targets",
        "false_positives",
        "overall_patch_quality_score",
        "text_preserved",
        "human_notes",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_quality_rows(
    *,
    content_data: list[dict[str, Any]],
    main_pages: set[int],
    quality_pages: set[int],
    negative_controls: int,
    seed: int = 20260504,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    negative_pool: list[dict[str, Any]] = []
    for idx, block in enumerate(content_data):
        if not isinstance(block, dict):
            continue
        page_idx = int(block.get("page_idx", -1))
        field_keys = _patch_field_keys(block)
        if not field_keys:
            continue
        for field in field_keys:
            text = _join_text(block.get(field, "")).strip()
            if not text:
                continue
            row = {
                "sample_label": "quality_page",
                "pdf_page": page_idx + 1,
                "page_idx": page_idx,
                "block_idx": idx,
                "field": field,
                "block_type": block.get("type", ""),
                "text_preview": _preview(text),
                "target_icons": "",
                "strict_hits": "",
                "wrong_icons": "",
                "missed_targets": "",
                "false_positives": "",
                "overall_patch_quality_score": "",
                "text_preserved": "",
                "human_notes": "",
            }
            if page_idx in quality_pages:
                rows.append(row)
            elif page_idx in main_pages and not ACTION_PATTERN.search(text):
                negative_pool.append({**row, "sample_label": "negative_control"})

    rng = random.Random(seed)
    rng.shuffle(negative_pool)
    rows.extend(negative_pool[: max(0, negative_controls)])
    rows.sort(key=lambda item: (int(item["pdf_page"]), int(item["block_idx"]), str(item["field"])))
    return rows


def write_quality_review_samples(
    *,
    content_data: list[dict[str, Any]],
    main_pages: set[int],
    quality_pages: set[int],
    output_csv: Path,
    score_template_csv: Path,
    negative_controls: int,
    sample_size: int,
    dpis: Sequence[int],
    seed: int = 20260504,
) -> None:
    all_rows = _collect_quality_rows(
        content_data=content_data,
        main_pages=main_pages,
        quality_pages=quality_pages,
        negative_controls=negative_controls,
        seed=seed,
    )
    quality_rows = [row for row in all_rows if row["sample_label"] == "quality_page"]
    negative_rows = [row for row in all_rows if row["sample_label"] == "negative_control"]

    rng = random.Random(seed)
    by_page: dict[int, list[dict[str, Any]]] = {}
    for row in quality_rows:
        by_page.setdefault(int(row["pdf_page"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for page in sorted(by_page):
        page_rows = list(by_page[page])
        rng.shuffle(page_rows)
        selected.extend(page_rows[: max(1, sample_size // max(1, len(by_page)))])
    if len(selected) < sample_size:
        remaining = [row for row in quality_rows if row not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])
    selected = selected[: max(0, sample_size)]
    selected.extend(negative_rows[:negative_controls])
    selected.sort(key=lambda item: (int(item["pdf_page"]), int(item["block_idx"]), str(item["field"])))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    review_fieldnames = [
        "sample_id",
        "sample_label",
        "pdf_page",
        "page_idx",
        "block_idx",
        "field",
        "block_type",
        "crop_path",
        "text_preview",
        "review_notes",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review_fieldnames)
        writer.writeheader()
        for sample_idx, row in enumerate(selected, start=1):
            writer.writerow(
                {
                    "sample_id": f"q{sample_idx:04d}",
                    "sample_label": row["sample_label"],
                    "pdf_page": row["pdf_page"],
                    "page_idx": row["page_idx"],
                    "block_idx": row["block_idx"],
                    "field": row["field"],
                    "block_type": row["block_type"],
                    "crop_path": "",
                    "text_preview": row["text_preview"],
                    "review_notes": "",
                }
            )

    score_fieldnames = [
        "dpi",
        "sample_id",
        "sample_label",
        "pdf_page",
        "block_idx",
        "field",
        "target_icons",
        "strict_hits",
        "wrong_icons",
        "missed_targets",
        "false_positives",
        "overall_patch_quality_score",
        "text_preserved",
        "review_notes",
    ]
    with score_template_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=score_fieldnames)
        writer.writeheader()
        for dpi in dpis:
            for sample_idx, row in enumerate(selected, start=1):
                writer.writerow(
                    {
                        "dpi": dpi,
                        "sample_id": f"q{sample_idx:04d}",
                        "sample_label": row["sample_label"],
                        "pdf_page": row["pdf_page"],
                        "block_idx": row["block_idx"],
                        "field": row["field"],
                        "target_icons": "",
                        "strict_hits": "",
                        "wrong_icons": "",
                        "missed_targets": "",
                        "false_positives": "",
                        "overall_patch_quality_score": "",
                        "text_preserved": "",
                        "review_notes": "",
                    }
                )


def write_quality_review_crops(
    *,
    content_data: list[dict[str, Any]],
    review_csv: Path,
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
) -> None:
    if not review_csv.exists():
        return
    try:
        from pdf2image import convert_from_path
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing pdf2image for quality crop export. Run `rag-flow env create-pipeline` first."
        ) from exc

    rows = list(csv.DictReader(review_csv.open(encoding="utf-8")))
    rows_by_page: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            page_idx = int(row["page_idx"])
        except (KeyError, ValueError):
            continue
        rows_by_page.setdefault(page_idx, []).append(row)

    crop_dir = output_dir / "quality-review-crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for page_idx, page_rows in sorted(rows_by_page.items()):
        pdf_images = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_idx + 1, last_page=page_idx + 1)
        if not pdf_images:
            continue
        image = pdf_images[0]
        for row in page_rows:
            try:
                block = content_data[int(row["block_idx"])]
            except (IndexError, KeyError, ValueError):
                continue
            bbox = block.get("bbox") if isinstance(block, dict) else None
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = (float(value) for value in bbox)
            crop_box = (
                max(0, int(x0 / 1000.0 * image.width)),
                max(0, int(y0 / 1000.0 * image.height)),
                min(image.width, int(x1 / 1000.0 * image.width)),
                min(image.height, int(y1 / 1000.0 * image.height)),
            )
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                continue
            filename = (
                f"{row['sample_id']}_p{int(row['pdf_page']):04d}_"
                f"b{int(row['block_idx']):05d}_{row['field']}.png"
            )
            path = crop_dir / filename
            image.crop(crop_box).save(path)
            row["crop_path"] = str(path.relative_to(output_dir))

    with review_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_cross_page_table_samples(*, content_data: list[dict[str, Any]], output_csv: Path) -> None:
    continuations = build_table_continuation_map(copy.deepcopy(content_data))
    rows: list[dict[str, Any]] = []
    for master_idx, continuation_indices in continuations.items():
        master = content_data[master_idx]
        continuation_pages = [
            int(content_data[idx].get("page_idx", 0)) + 1
            for idx in continuation_indices
            if isinstance(content_data[idx], dict)
        ]
        rows.append(
            {
                "master_block_idx": master_idx,
                "master_pdf_page": int(master.get("page_idx", 0)) + 1,
                "continuation_block_indices": ";".join(str(idx) for idx in continuation_indices),
                "continuation_pdf_pages": ";".join(str(page) for page in continuation_pages),
                "span_pages": 1 + len(set(continuation_pages)),
                "table_caption": _preview(_join_text(master.get("table_caption", ""))),
                "table_body_preview": _preview(_join_text(master.get("table_body", ""))),
                "human_notes": "",
            }
        )
    rows.sort(key=lambda item: (-int(item["span_pages"]), int(item["master_pdf_page"])))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "master_block_idx",
        "master_pdf_page",
        "continuation_block_indices",
        "continuation_pdf_pages",
        "span_pages",
        "table_caption",
        "table_body_preview",
        "human_notes",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, math_round((len(values) - 1) * q)))
    return values[idx]


def math_round(value: float) -> int:
    return int(value + 0.5)


def summarize_run(run_dir: Path, *, elapsed_s: float) -> dict[str, Any]:
    requests = _read_jsonl(run_dir / "requests.jsonl")
    renders = _read_jsonl(run_dir / "render.jsonl")
    checkpoints = _read_jsonl(run_dir / "checkpoints.jsonl")
    durations = [float(row.get("duration_s", 0.0)) for row in requests if row.get("duration_s") is not None]
    ok_requests = [row for row in requests if row.get("status") == "ok"]
    timeout_count = sum(
        1
        for row in requests
        if "timeout" in str(row.get("error_type", "")).lower() or "timeout" in str(row.get("error", "")).lower()
    )
    checkpoint_by_reason: dict[str, list[dict[str, Any]]] = {}
    for row in checkpoints:
        checkpoint_by_reason.setdefault(str(row.get("reason", "unknown")), []).append(row)
    checkpoint_batch_positions = sorted(
        int(row.get("batches_processed", 0) or 0)
        for row in checkpoints
        if row.get("batches_processed") is not None
    )
    checkpoint_gaps = [
        right - left
        for left, right in zip([0, *checkpoint_batch_positions[:-1]], checkpoint_batch_positions, strict=False)
    ]
    summary = {
        "elapsed_s": elapsed_s,
        "request_count": len(requests),
        "ok_request_count": len(ok_requests),
        "error_request_count": len(requests) - len(ok_requests),
        "timeout_count": timeout_count,
        "written_count": sum(1 for row in requests if row.get("written")),
        "fallback_count": sum(1 for row in requests if "fallback" in str(row.get("decision", ""))),
        "rejected_count": sum(1 for row in requests if row.get("decision") == "invalid_rejected"),
        "fields_per_min": (len(requests) / elapsed_s * 60.0) if elapsed_s > 0 else 0.0,
        "request_duration_avg_s": (sum(durations) / len(durations)) if durations else 0.0,
        "request_duration_p50_s": _percentile(durations, 0.50),
        "request_duration_p95_s": _percentile(durations, 0.95),
        "render_count": len(renders),
        "render_duration_total_s": sum(float(row.get("duration_s", 0.0)) for row in renders),
        "checkpoint_count": len(checkpoints),
        "checkpoint_duration_total_s": sum(float(row.get("duration_s", 0.0)) for row in checkpoints),
        "checkpoint_bytes_max": max((int(row.get("file_size_bytes", 0)) for row in checkpoints), default=0),
        "checkpoint_interval_count": len(checkpoint_by_reason.get("interval", [])),
        "checkpoint_interval_duration_total_s": sum(
            float(row.get("duration_s", 0.0)) for row in checkpoint_by_reason.get("interval", [])
        ),
        "checkpoint_page_window_count": len(checkpoint_by_reason.get("page_window", [])),
        "checkpoint_page_window_duration_total_s": sum(
            float(row.get("duration_s", 0.0)) for row in checkpoint_by_reason.get("page_window", [])
        ),
        "checkpoint_failure_count": len(checkpoint_by_reason.get("failure", [])),
        "checkpoint_recovery_batches_max": max(checkpoint_gaps, default=0),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _mean(values: Iterable[float]) -> float:
    values = [value for value in values if value is not None]
    return (sum(values) / len(values)) if values else 0.0


def _normalize_for_text_preservation(text: str) -> str:
    text = ICON_TAG_PATTERN.sub(" ", text)
    text = MARKDOWN_BOLD_PATTERN.sub(r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\s\W_]+", " ", text.lower(), flags=re.UNICODE)
    return " ".join(text.split())


def infer_text_preserved(original_text: str, patched_text: str) -> int:
    original = _normalize_for_text_preservation(original_text)
    patched = _normalize_for_text_preservation(patched_text)
    if not original:
        return 1
    if not patched:
        return 0
    if original in patched or patched in original:
        return 1
    return 1 if SequenceMatcher(None, original, patched).ratio() >= 0.86 else 0


def derive_overall_patch_quality_score(row: dict[str, Any], *, text_preserved: int = 1) -> int:
    if not text_preserved:
        return 0

    target_icons = _safe_int(row.get("target_icons"))
    strict_hits = _safe_int(row.get("strict_hits"))
    wrong_icons = _safe_int(row.get("wrong_icons"))
    missed_targets = _safe_int(row.get("missed_targets"))
    false_positives = _safe_int(row.get("false_positives"))

    if target_icons <= 0:
        if false_positives <= 0:
            return 5
        if false_positives == 1:
            return 3
        if false_positives == 2:
            return 2
        return 1

    score = 5.0 - missed_targets * 2.0 - wrong_icons * 1.5 - false_positives * 1.0
    recall = strict_hits / target_icons if target_icons else 0.0
    if strict_hits == 0 and (wrong_icons or missed_targets):
        score = min(score, 2.0)
    elif recall < 0.5:
        score = min(score, 2.0)
    elif recall < 0.8:
        score = min(score, 3.0)

    return max(0, min(5, int(score + 0.5)))


def _worklist_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("dpi", "")), str(row.get("sample_id", "")))


def backfill_quality_scores(
    *,
    scores_csv: Path,
    worklist_csv: Path | None = None,
    output_csv: Path | None = None,
) -> dict[str, Any]:
    rows = list(csv.DictReader(scores_csv.open(encoding="utf-8")))
    if not rows:
        return {"rows": 0, "text_preserved_zero": 0, "score_avg": 0.0, "output_csv": str(output_csv or scores_csv)}

    worklist_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if worklist_csv and worklist_csv.exists():
        with worklist_csv.open(encoding="utf-8") as f:
            worklist_by_key = {_worklist_key(row): row for row in csv.DictReader(f)}

    updated_rows: list[dict[str, Any]] = []
    text_preserved_zero = 0
    score_total = 0
    for row in rows:
        work = worklist_by_key.get(_worklist_key(row), {})
        text_preserved = infer_text_preserved(
            str(work.get("original_text", "")),
            str(work.get("patched_text", "")),
        ) if work else 1
        score = derive_overall_patch_quality_score(row, text_preserved=text_preserved)
        updated = dict(row)
        updated["overall_patch_quality_score"] = str(score)
        updated["text_preserved"] = str(text_preserved)
        updated_rows.append(updated)
        text_preserved_zero += 1 if text_preserved == 0 else 0
        score_total += score

    output_path = output_csv or scores_csv
    fieldnames = list(rows[0].keys())
    for field in ("overall_patch_quality_score", "text_preserved"):
        if field not in fieldnames:
            fieldnames.append(field)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    return {
        "rows": len(updated_rows),
        "text_preserved_zero": text_preserved_zero,
        "score_avg": score_total / len(updated_rows),
        "output_csv": str(output_path),
    }


def _summarize_gpu_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "gpu_sample_count": 0,
            "gpu_utilization_avg_pct": 0.0,
            "gpu_memory_used_max_mib": 0.0,
            "gpu_memory_used_avg_mib": 0.0,
            "gpu_power_avg_w": 0.0,
        }
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    usable = [row for row in rows if not row.get("error")]
    memory_values = [_safe_float(row.get("memory_used_mib")) for row in usable]
    return {
        "gpu_sample_count": len(usable),
        "gpu_utilization_avg_pct": _mean(_safe_float(row.get("gpu_utilization_pct")) for row in usable),
        "gpu_memory_used_max_mib": max(memory_values, default=0.0),
        "gpu_memory_used_avg_mib": _mean(memory_values),
        "gpu_power_avg_w": _mean(_safe_float(row.get("power_w")) for row in usable),
    }


def _stage_sort_key(stage: str) -> int:
    order = {
        "baseline": 0,
        "dpi": 1,
        "quality": 2,
        "concurrency": 3,
        "batch-size": 4,
        "checkpoint": 5,
        "dpi-confirm": 6,
        "final": 7,
    }
    return order.get(stage, 99)


def _load_benchmark_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(output_dir.glob("*/*/summary.json")):
        run_dir = summary_path.parent
        params_path = run_dir / "run_params.json"
        if not params_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        params = json.loads(params_path.read_text(encoding="utf-8"))
        gpu = _summarize_gpu_csv(run_dir / "gpu.csv")
        row = {
            "run_id": params.get("run_id", run_dir.name),
            "stage": params.get("stage", run_dir.parent.name),
            "dpi": int(params.get("dpi", 0) or 0),
            "concurrency": int(params.get("concurrency", 0) or 0),
            "batch_size": int(params.get("batch_size", 0) or 0),
            "checkpoint_interval": int(params.get("checkpoint_interval", 0) or 0),
            "repeat_index": int(params.get("repeat_index", 1) or 1),
            "subset_blocks": int(params.get("subset_blocks", 0) or 0),
            "request_count": int(summary.get("request_count", 0) or 0),
            "ok_request_count": int(summary.get("ok_request_count", 0) or 0),
            "error_request_count": int(summary.get("error_request_count", 0) or 0),
            "timeout_count": int(summary.get("timeout_count", 0) or 0),
            "written_count": int(summary.get("written_count", 0) or 0),
            "fallback_count": int(summary.get("fallback_count", 0) or 0),
            "rejected_count": int(summary.get("rejected_count", 0) or 0),
            "elapsed_s": _safe_float(summary.get("elapsed_s")),
            "fields_per_min": _safe_float(summary.get("fields_per_min")),
            "request_duration_avg_s": _safe_float(summary.get("request_duration_avg_s")),
            "request_duration_p50_s": _safe_float(summary.get("request_duration_p50_s")),
            "request_duration_p95_s": _safe_float(summary.get("request_duration_p95_s")),
            "render_count": int(summary.get("render_count", 0) or 0),
            "render_duration_total_s": _safe_float(summary.get("render_duration_total_s")),
            "checkpoint_count": int(summary.get("checkpoint_count", 0) or 0),
            "checkpoint_duration_total_s": _safe_float(summary.get("checkpoint_duration_total_s")),
            "checkpoint_bytes_max": int(summary.get("checkpoint_bytes_max", 0) or 0),
            "checkpoint_interval_count": int(summary.get("checkpoint_interval_count", 0) or 0),
            "checkpoint_interval_duration_total_s": _safe_float(
                summary.get("checkpoint_interval_duration_total_s")
            ),
            "checkpoint_page_window_count": int(summary.get("checkpoint_page_window_count", 0) or 0),
            "checkpoint_page_window_duration_total_s": _safe_float(
                summary.get("checkpoint_page_window_duration_total_s")
            ),
            "checkpoint_failure_count": int(summary.get("checkpoint_failure_count", 0) or 0),
            "checkpoint_recovery_batches_max": int(summary.get("checkpoint_recovery_batches_max", 0) or 0),
            "run_dir": str(run_dir),
            **gpu,
        }
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            _stage_sort_key(str(row["stage"])),
            int(row["dpi"]),
            int(row["concurrency"]),
            int(row["batch_size"]),
            int(row["checkpoint_interval"]),
            int(row["repeat_index"]),
            str(row["run_id"]),
        ),
    )


def _write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "run_id",
        "dpi",
        "concurrency",
        "batch_size",
        "checkpoint_interval",
        "repeat_index",
        "subset_blocks",
        "request_count",
        "ok_request_count",
        "error_request_count",
        "timeout_count",
        "written_count",
        "fallback_count",
        "rejected_count",
        "elapsed_s",
        "fields_per_min",
        "request_duration_avg_s",
        "request_duration_p50_s",
        "request_duration_p95_s",
        "render_count",
        "render_duration_total_s",
        "checkpoint_count",
        "checkpoint_duration_total_s",
        "checkpoint_bytes_max",
        "checkpoint_interval_count",
        "checkpoint_interval_duration_total_s",
        "checkpoint_page_window_count",
        "checkpoint_page_window_duration_total_s",
        "checkpoint_failure_count",
        "checkpoint_recovery_batches_max",
        "gpu_sample_count",
        "gpu_utilization_avg_pct",
        "gpu_memory_used_max_mib",
        "gpu_memory_used_avg_mib",
        "gpu_power_avg_w",
        "run_dir",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_metric(rows: list[dict[str, Any]], *, stage: str, x_key: str, y_key: str) -> list[tuple[float, float, int]]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        if row.get("stage") != stage:
            continue
        x = _safe_float(row.get(x_key))
        grouped.setdefault(x, []).append(_safe_float(row.get(y_key)))
    return [(x, _mean(values), len(values)) for x, values in sorted(grouped.items())]


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_line_svg(
    *,
    points: list[tuple[float, float, int]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    stroke: str = "#2563eb",
) -> bool:
    if not points:
        return False
    width, height = 760, 420
    left, right, top, bottom = 74, 28, 54, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = 0.0, max(ys)
    if max_x == min_x:
        max_x += 1
    if max_y == min_y:
        max_y += 1

    def sx(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - min_y) / (max_y - min_y) * plot_h

    polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in points)
    x_ticks = "".join(
        f'<line x1="{sx(x):.1f}" y1="{top + plot_h:.1f}" x2="{sx(x):.1f}" y2="{top + plot_h + 5:.1f}" stroke="#334155"/>'
        f'<text x="{sx(x):.1f}" y="{top + plot_h + 24:.1f}" text-anchor="middle" font-size="12">{x:g}</text>'
        for x in xs
    )
    y_ticks = ""
    for idx in range(5):
        value = min_y + (max_y - min_y) * idx / 4
        y = sy(value)
        y_ticks += (
            f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#334155"/>'
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>'
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{value:.1f}</text>'
        )
    circles = "".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{stroke}">'
        f'<title>x={x:g}, y={y:.3f}, n={count}</title></circle>'
        for x, y, count in points
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700">{_svg_escape(title)}</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>
  {y_ticks}
  {x_ticks}
  <polyline fill="none" stroke="{stroke}" stroke-width="3" points="{polyline}"/>
  {circles}
  <text x="{left + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{_svg_escape(x_label)}</text>
  <text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">{_svg_escape(y_label)}</text>
</svg>
""",
        encoding="utf-8",
    )
    return True


def _write_bar_svg(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    stage: str | None = None,
) -> bool:
    selected = [row for row in rows if stage is None or row.get("stage") == stage]
    if not selected:
        return False
    selected = selected[:24]
    width, height = 980, 460
    left, right, top, bottom = 80, 28, 54, 128
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_total = max(
        (
            int(row["timeout_count"]) + int(row["fallback_count"]) + int(row["rejected_count"]) + int(row["error_request_count"])
            for row in selected
        ),
        default=1,
    )
    if max_total <= 0:
        max_total = 1
    bar_w = max(8, plot_w / max(len(selected), 1) * 0.56)
    colors = {
        "timeout_count": "#ef4444",
        "fallback_count": "#f59e0b",
        "rejected_count": "#8b5cf6",
        "error_request_count": "#64748b",
    }
    bars = []
    labels = []
    for idx, row in enumerate(selected):
        x = left + (idx + 0.5) * plot_w / len(selected) - bar_w / 2
        y_cursor = top + plot_h
        for key in ("timeout_count", "fallback_count", "rejected_count", "error_request_count"):
            value = int(row[key])
            h = value / max_total * plot_h
            if h > 0:
                y_cursor -= h
                bars.append(
                    f'<rect x="{x:.1f}" y="{y_cursor:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[key]}">'
                    f'<title>{_svg_escape(str(row["run_id"]))}: {key}={value}</title></rect>'
                )
        label = str(row["run_id"]).replace(f"{row['stage']}_", "")
        labels.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 18:.1f}" text-anchor="end" transform="rotate(-45 {x + bar_w / 2:.1f} {top + plot_h + 18:.1f})" font-size="10">{_svg_escape(label)}</text>'
        )
    legend = " ".join(
        f'<rect x="{left + i * 170}" y="{height - 26}" width="12" height="12" fill="{color}"/>'
        f'<text x="{left + i * 170 + 18}" y="{height - 15}" font-size="12">{_svg_escape(key.replace("_count", ""))}</text>'
        for i, (key, color) in enumerate(colors.items())
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700">{_svg_escape(title)}</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>
  <text x="{left - 10}" y="{top + 4}" text-anchor="end" font-size="12">{max_total}</text>
  <text x="{left - 10}" y="{top + plot_h + 4}" text-anchor="end" font-size="12">0</text>
  {''.join(bars)}
  {''.join(labels)}
  {legend}
</svg>
""",
        encoding="utf-8",
    )
    return True


def _write_quality_charts(output_dir: Path, report_dir: Path, charts_dir: Path) -> list[tuple[str, str]]:
    source = output_dir / "quality_scores.csv"
    if not source.exists():
        source = report_dir / "quality_scores.csv"
    if not source.exists():
        return []

    with source.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[float, dict[str, float]] = {}
    for row in rows:
        dpi = _safe_float(row.get("dpi"))
        if dpi <= 0:
            continue
        bucket = grouped.setdefault(
            dpi,
            {
                "target_icons": 0.0,
                "strict_hits": 0.0,
                "wrong_icons": 0.0,
                "missed_targets": 0.0,
                "false_positives": 0.0,
                "overall_patch_quality_score_total": 0.0,
                "overall_patch_quality_score_count": 0.0,
                "text_preserved_total": 0.0,
                "text_preserved_count": 0.0,
            },
        )
        for key in ("target_icons", "strict_hits", "wrong_icons", "missed_targets", "false_positives"):
            bucket[key] += _safe_float(row.get(key))
        overall_score = _optional_float(row.get("overall_patch_quality_score"))
        if overall_score is not None:
            bucket["overall_patch_quality_score_total"] += overall_score
            bucket["overall_patch_quality_score_count"] += 1.0
        text_preserved = _optional_float(row.get("text_preserved"))
        if text_preserved is not None:
            bucket["text_preserved_total"] += text_preserved
            bucket["text_preserved_count"] += 1.0
    if not grouped:
        return []

    summary_rows = []
    hit_points = []
    miss_points = []
    overall_score_points = []
    text_preserved_points = []
    for dpi, values in sorted(grouped.items()):
        target_icons = values["target_icons"]
        strict_hit_ratio = values["strict_hits"] / target_icons * 100.0 if target_icons else 0.0
        miss_rate = values["missed_targets"] / target_icons * 100.0 if target_icons else 0.0
        overall_score_count = values["overall_patch_quality_score_count"]
        overall_score_avg = (
            values["overall_patch_quality_score_total"] / overall_score_count
            if overall_score_count
            else 0.0
        )
        text_preserved_count = values["text_preserved_count"]
        text_preserved_ratio = (
            values["text_preserved_total"] / text_preserved_count * 100.0
            if text_preserved_count
            else 0.0
        )
        summary_rows.append(
            {
                "dpi": int(dpi),
                "target_icons": values["target_icons"],
                "strict_hits": values["strict_hits"],
                "wrong_icons": values["wrong_icons"],
                "missed_targets": values["missed_targets"],
                "false_positives": values["false_positives"],
                "strict_hit_ratio_pct": strict_hit_ratio,
                "miss_rate_pct": miss_rate,
                "overall_patch_quality_score_avg": overall_score_avg,
                "text_preserved_ratio_pct": text_preserved_ratio,
                "overall_patch_quality_score_count": overall_score_count,
                "text_preserved_count": text_preserved_count,
            }
        )
        hit_points.append((dpi, strict_hit_ratio, int(target_icons)))
        miss_points.append((dpi, miss_rate, int(target_icons)))
        if overall_score_count:
            overall_score_points.append((dpi, overall_score_avg, int(overall_score_count)))
        if text_preserved_count:
            text_preserved_points.append((dpi, text_preserved_ratio, int(text_preserved_count)))

    with (report_dir / "quality_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dpi",
            "target_icons",
            "strict_hits",
            "wrong_icons",
            "missed_targets",
            "false_positives",
            "strict_hit_ratio_pct",
            "miss_rate_pct",
            "overall_patch_quality_score_avg",
            "overall_patch_quality_score_count",
            "text_preserved_ratio_pct",
            "text_preserved_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    chart_files: list[tuple[str, str]] = []
    if _write_line_svg(
        points=hit_points,
        output_path=charts_dir / "quality_strict_hit_ratio.svg",
        title="DPI 复核质量评分：Strict Hit Ratio",
        x_label="DPI",
        y_label="strict hit ratio (%)",
        stroke="#16a34a",
    ):
        chart_files.append(("quality_strict_hit_ratio.svg", "DPI 复核质量评分：Strict Hit Ratio"))
    if _write_line_svg(
        points=miss_points,
        output_path=charts_dir / "quality_miss_rate.svg",
        title="DPI 复核质量评分：Miss Rate",
        x_label="DPI",
        y_label="miss rate (%)",
        stroke="#dc2626",
    ):
        chart_files.append(("quality_miss_rate.svg", "DPI 复核质量评分：Miss Rate"))
    if _write_line_svg(
        points=overall_score_points,
        output_path=charts_dir / "quality_overall_patch_score.svg",
        title="DPI 复核质量评分：Overall Patch Quality",
        x_label="DPI",
        y_label="overall patch quality (0-5)",
        stroke="#7c3aed",
    ):
        chart_files.append(("quality_overall_patch_score.svg", "DPI 复核质量评分：Overall Patch Quality"))
    if _write_line_svg(
        points=text_preserved_points,
        output_path=charts_dir / "quality_text_preserved_ratio.svg",
        title="DPI 复核质量评分：Text Preserved Ratio",
        x_label="DPI",
        y_label="text preserved (%)",
        stroke="#0f766e",
    ):
        chart_files.append(("quality_text_preserved_ratio.svg", "DPI 复核质量评分：Text Preserved Ratio"))
    return chart_files


def _block_by_original_index(content_data: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    mapped: dict[int, dict[str, Any]] = {}
    for idx, block in enumerate(content_data):
        if not isinstance(block, dict):
            continue
        original_idx = int(block.get(BENCHMARK_ORIGINAL_INDEX_KEY, idx))
        mapped[original_idx] = block
    return mapped


def write_quality_review_worklist(output_dir: Path, report_dir: Path) -> Path | None:
    samples_path = output_dir / "quality_review_samples.csv"
    if not samples_path.exists():
        return None
    samples = list(csv.DictReader(samples_path.open(encoding="utf-8")))
    if not samples:
        return None

    rows: list[dict[str, Any]] = []
    for params_path in sorted((output_dir / "quality").glob("*/run_params.json")):
        params = json.loads(params_path.read_text(encoding="utf-8"))
        input_json = Path(params.get("input_json", ""))
        output_json = Path(params.get("output_json", ""))
        if not input_json.exists() or not output_json.exists():
            continue
        input_by_original = _block_by_original_index(_load_content(input_json))
        output_by_original = _block_by_original_index(_load_content(output_json))
        for sample in samples:
            try:
                original_idx = int(sample["block_idx"])
            except (KeyError, ValueError):
                continue
            field = sample.get("field", "")
            input_block = input_by_original.get(original_idx, {})
            output_block = output_by_original.get(original_idx, {})
            rows.append(
                {
                    "dpi": params.get("dpi", ""),
                    "run_id": params.get("run_id", ""),
                    "sample_id": sample.get("sample_id", ""),
                    "sample_label": sample.get("sample_label", ""),
                    "pdf_page": sample.get("pdf_page", ""),
                    "block_idx": original_idx,
                    "field": field,
                    "block_type": sample.get("block_type", ""),
                    "crop_path": sample.get("crop_path", ""),
                    "original_text": _join_text(input_block.get(field, "")),
                    "patched_text": _join_text(output_block.get(field, "")),
                    "target_icons": "",
                    "strict_hits": "",
                    "wrong_icons": "",
                    "missed_targets": "",
                    "false_positives": "",
                    "overall_patch_quality_score": "",
                    "text_preserved": "",
                    "review_decision": "",
                    "review_notes": "",
                }
            )

    if not rows:
        return None

    output_path = report_dir / "quality_review_worklist.csv"
    fieldnames = [
        "dpi",
        "run_id",
        "sample_id",
        "sample_label",
        "pdf_page",
        "block_idx",
        "field",
        "block_type",
        "crop_path",
        "original_text",
        "patched_text",
        "target_icons",
        "strict_hits",
        "wrong_icons",
        "missed_targets",
        "false_positives",
        "overall_patch_quality_score",
        "text_preserved",
        "review_decision",
        "review_notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def generate_report(output_dir: Path, *, main_pages: str, quality_pages: str) -> None:
    rows = _load_benchmark_rows(output_dir)
    report_dir = output_dir / "report"
    charts_dir = report_dir / "charts"
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(rows, report_dir / "benchmark_summary.csv")
    quality_worklist = write_quality_review_worklist(output_dir, report_dir)

    chart_specs = [
        ("dpi", "dpi", "fields_per_min", "dpi_throughput.svg", "DPI 对吞吐的影响", "DPI", "fields/min", "#2563eb"),
        ("dpi", "dpi", "request_duration_p95_s", "dpi_p95_latency.svg", "DPI 对 p95 latency 的影响", "DPI", "p95 latency (s)", "#dc2626"),
        ("dpi", "dpi", "render_duration_total_s", "dpi_render_time.svg", "DPI 对 PDF 渲染耗时的影响", "DPI", "render time (s)", "#16a34a"),
        ("quality", "dpi", "fields_per_min", "quality_runtime.svg", "质量页 DPI 运行开销", "DPI", "fields/min", "#0f766e"),
        ("concurrency", "concurrency", "fields_per_min", "concurrency_throughput.svg", "Concurrency 吞吐曲线", "concurrency", "fields/min", "#2563eb"),
        ("concurrency", "concurrency", "request_duration_p95_s", "concurrency_p95_latency.svg", "Concurrency p95 latency 曲线", "concurrency", "p95 latency (s)", "#dc2626"),
        ("concurrency", "concurrency", "gpu_utilization_avg_pct", "concurrency_gpu_utilization.svg", "Concurrency 平均 GPU 利用率", "concurrency", "GPU utilization (%)", "#7c3aed"),
        ("batch-size", "batch_size", "fields_per_min", "batch_size_throughput.svg", "Batch Size 吞吐曲线", "batch size", "fields/min", "#2563eb"),
        (
            "checkpoint",
            "checkpoint_interval",
            "checkpoint_interval_duration_total_s",
            "checkpoint_interval_overhead.svg",
            "周期 Checkpoint 写入开销",
            "checkpoint interval",
            "interval checkpoint time (s)",
            "#f59e0b",
        ),
        (
            "checkpoint",
            "checkpoint_interval",
            "checkpoint_recovery_batches_max",
            "checkpoint_recovery_risk.svg",
            "Checkpoint 最大理论重跑 batch 数",
            "checkpoint interval",
            "max replay batches",
            "#7c3aed",
        ),
        ("dpi-confirm", "dpi", "fields_per_min", "dpi_confirm_throughput.svg", "最终部署参数下的 DPI 复验", "DPI", "fields/min", "#2563eb"),
        ("final", "repeat_index", "fields_per_min", "final_repeat_throughput.svg", "最终配置重复运行稳定性", "repeat", "fields/min", "#2563eb"),
    ]
    chart_files: list[tuple[str, str]] = []
    for stage, x_key, y_key, filename, title, x_label, y_label, stroke in chart_specs:
        points = _group_metric(rows, stage=stage, x_key=x_key, y_key=y_key)
        if _write_line_svg(
            points=points,
            output_path=charts_dir / filename,
            title=title,
            x_label=x_label,
            y_label=y_label,
            stroke=stroke,
        ):
            chart_files.append((filename, title))
    if _write_bar_svg(rows=rows, output_path=charts_dir / "failure_counts.svg", title="失败、fallback 与 rejected 统计"):
        chart_files.append(("failure_counts.svg", "失败、fallback 与 rejected 统计"))
    chart_files.extend(_write_quality_charts(output_dir, report_dir, charts_dir))

    best_by_stage: dict[str, dict[str, Any]] = {}
    for row in rows:
        stage = str(row["stage"])
        if stage not in best_by_stage or float(row["fields_per_min"]) > float(best_by_stage[stage]["fields_per_min"]):
            best_by_stage[stage] = row

    chart_typst = "\n".join(
        f'#figure(image("charts/{filename}", width: 92%), caption: [{title}])'
        for filename, title in chart_files
    )
    best_lines = "\n".join(
        f'- `{stage}`: `{row["run_id"]}`，fields/min={float(row["fields_per_min"]):.2f}，p95={float(row["request_duration_p95_s"]):.2f}s。'
        for stage, row in sorted(best_by_stage.items(), key=lambda item: _stage_sort_key(item[0]))
    )
    (report_dir / "report.typ").write_text(
        f"""= Patching Benchmark 自动报告

主测试页：PDF pages {main_pages}

质量检查页：PDF pages {quality_pages}

== 汇总数据

所有结构化结果已汇总到 `benchmark_summary.csv`。图表使用每个参数值下的重复运行均值；如果存在复核后填写的 `quality_scores.csv`，报告会额外生成 `quality_summary.csv` 和 DPI 质量评分图。质量复核工作表：{quality_worklist.name if quality_worklist else "暂无，需先完成 quality 阶段"}。

== 阶段最优吞吐

{best_lines or "- 暂无已完成 run。"}

== 图表

{chart_typst or "暂无可绘制图表。"}
""",
        encoding="utf-8",
    )
    (report_dir / "README.md").write_text(
        "\n".join(
            [
                "# Patching benchmark report",
                "",
                f"- Main pages: {main_pages}",
                f"- Quality pages: {quality_pages}",
                f"- Runs discovered: {len(rows)}",
                f"- Summary CSV: {report_dir / 'benchmark_summary.csv'}",
                f"- Typst report: {report_dir / 'report.typ'}",
                f"- Quality review worklist: {quality_worklist or ''}",
                f"- Optional quality CSV: {output_dir / 'quality_scores.csv'}",
                "",
                "## Charts",
                *(f"- {title}: charts/{filename}" for filename, title in chart_files),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Benchmark runs discovered: {len(rows)}")
    print(f"Summary CSV: {report_dir / 'benchmark_summary.csv'}")
    print(f"Typst report: {report_dir / 'report.typ'}")
    print(f"Charts: {charts_dir}")


def should_stop_concurrency_search(
    *,
    summary: dict[str, Any],
    best_summary: dict[str, Any] | None,
    timeout_ratio: float,
    throughput_drop_ratio: float,
    p95_ratio: float,
) -> tuple[bool, str]:
    request_count = max(1, int(summary.get("request_count", 0) or 0))
    timeout_count = int(summary.get("timeout_count", 0) or 0)
    error_count = int(summary.get("error_request_count", 0) or 0)
    if error_count > 0:
        return True, f"error requests observed: {error_count}"
    if timeout_count / request_count > timeout_ratio:
        return True, f"timeout ratio {timeout_count / request_count:.3f} exceeded {timeout_ratio:.3f}"
    if best_summary is None:
        return False, ""

    fields_per_min = float(summary.get("fields_per_min", 0.0) or 0.0)
    best_fields_per_min = float(best_summary.get("fields_per_min", 0.0) or 0.0)
    if best_fields_per_min > 0 and fields_per_min < best_fields_per_min * (1.0 - throughput_drop_ratio):
        return (
            True,
            f"fields/min {fields_per_min:.2f} dropped more than {throughput_drop_ratio:.0%} "
            f"from best {best_fields_per_min:.2f}",
        )

    p95 = float(summary.get("request_duration_p95_s", 0.0) or 0.0)
    best_p95 = float(best_summary.get("request_duration_p95_s", 0.0) or 0.0)
    if best_p95 > 0 and p95 > best_p95 * p95_ratio:
        return True, f"p95 latency {p95:.2f}s exceeded {p95_ratio:.1f}x best p95 {best_p95:.2f}s"
    return False, ""


def _repeat_specs(spec: RunSpec, repeat: int) -> list[RunSpec]:
    return [
        RunSpec(
            stage=spec.stage,
            name=spec.name,
            dpi=spec.dpi,
            concurrency=spec.concurrency,
            batch_size=spec.batch_size,
            checkpoint_interval=spec.checkpoint_interval,
            repeat_index=idx,
        )
        for idx in range(1, repeat + 1)
    ]


def build_run_specs(args: argparse.Namespace, config: AppConfig) -> list[RunSpec]:
    stage = args.stage
    repeat = args.repeat if args.repeat is not None else (3 if stage == "final" else 1)
    if stage in {"prepare", "report", "score"}:
        return []
    if stage == "baseline":
        return [
            * _repeat_specs(RunSpec(stage, "sanity", 250, 3, 9, 30), repeat),
            * _repeat_specs(RunSpec(stage, "default", config.patching.dpi, config.patching.concurrency, config.patching.batch_size, config.patching.checkpoint_interval), repeat),
        ]
    if stage == "dpi":
        return [
            spec
            for dpi in parse_int_list(args.dpis)
            for spec in _repeat_specs(RunSpec(stage, f"dpi{dpi}", dpi, 3, 9, 30), repeat)
        ]
    if stage == "quality":
        concurrency = args.selected_concurrency or 3
        batch_size = args.selected_batch_size or 9
        checkpoint_interval = args.selected_checkpoint_interval if args.selected_checkpoint_interval is not None else 30
        return [
            spec
            for dpi in parse_int_list(args.dpis)
            for spec in _repeat_specs(
                RunSpec(stage, f"dpi{dpi}", dpi, concurrency, batch_size, checkpoint_interval),
                repeat,
            )
        ]
    if stage == "concurrency":
        _require(args.selected_dpi, "--selected-dpi is required for concurrency stage.")
        return [
            spec
            for concurrency in parse_int_list(args.concurrency_values)
            for spec in _repeat_specs(RunSpec(stage, f"c{concurrency}", args.selected_dpi, concurrency, 140, 30), repeat)
        ]
    if stage == "batch-size":
        _require(args.selected_dpi, "--selected-dpi is required for batch-size stage.")
        _require(args.selected_concurrency, "--selected-concurrency is required for batch-size stage.")
        values = parse_batch_size_spec(args.batch_sizes, concurrency=args.selected_concurrency)
        return [
            spec
            for batch_size in values
            for spec in _repeat_specs(
                RunSpec(stage, f"b{batch_size}", args.selected_dpi, args.selected_concurrency, batch_size, 30),
                repeat,
            )
        ]
    if stage == "checkpoint":
        _require(args.selected_dpi, "--selected-dpi is required for checkpoint stage.")
        _require(args.selected_concurrency, "--selected-concurrency is required for checkpoint stage.")
        _require(args.selected_batch_size, "--selected-batch-size is required for checkpoint stage.")
        return [
            spec
            for checkpoint_interval in parse_int_list(args.checkpoint_intervals)
            for spec in _repeat_specs(
                RunSpec(
                    stage,
                    f"k{checkpoint_interval}",
                    args.selected_dpi,
                    args.selected_concurrency,
                    args.selected_batch_size,
                    checkpoint_interval,
                ),
                repeat,
            )
        ]
    if stage == "dpi-confirm":
        _require(args.selected_concurrency, "--selected-concurrency is required for dpi-confirm stage.")
        _require(args.selected_batch_size, "--selected-batch-size is required for dpi-confirm stage.")
        _require(args.selected_checkpoint_interval, "--selected-checkpoint-interval is required for dpi-confirm stage.")
        return [
            spec
            for dpi in parse_int_list(args.dpis)
            for spec in _repeat_specs(
                RunSpec(
                    stage,
                    f"dpi{dpi}",
                    dpi,
                    args.selected_concurrency,
                    args.selected_batch_size,
                    args.selected_checkpoint_interval,
                ),
                repeat,
            )
        ]
    if stage == "final":
        _require(args.selected_dpi, "--selected-dpi is required for final stage.")
        _require(args.selected_concurrency, "--selected-concurrency is required for final stage.")
        _require(args.selected_batch_size, "--selected-batch-size is required for final stage.")
        _require(args.selected_checkpoint_interval, "--selected-checkpoint-interval is required for final stage.")
        return _repeat_specs(
            RunSpec(
                stage,
                "recommended",
                args.selected_dpi,
                args.selected_concurrency,
                args.selected_batch_size,
                args.selected_checkpoint_interval,
            ),
            repeat,
        )
    raise SystemExit(f"Unknown benchmark stage: {stage}")


def _require(value: Any, message: str) -> None:
    if value is None:
        raise SystemExit(message)


def page_indices_for_stage(
    *,
    stage: str,
    main_pages: set[int],
    quality_pages: set[int],
    include_quality_pages: bool,
) -> set[int]:
    if stage == "quality":
        return set(quality_pages)
    if include_quality_pages:
        return set(main_pages) | set(quality_pages)
    return set(main_pages)


def write_prepare_artifacts(
    *,
    content_data: list[dict[str, Any]],
    main_pages: set[int],
    quality_pages: set[int],
    output_dir: Path,
    negative_controls: int,
    quality_sample_size: int,
    dpis: Sequence[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_quality_scoring_template(
        content_data=content_data,
        main_pages=main_pages,
        quality_pages=quality_pages,
        output_csv=output_dir / "quality_scoring_template.csv",
        negative_controls=negative_controls,
    )
    write_cross_page_table_samples(
        content_data=content_data,
        output_csv=output_dir / "cross_page_table_samples.csv",
    )
    write_quality_review_samples(
        content_data=content_data,
        main_pages=main_pages,
        quality_pages=quality_pages,
        output_csv=output_dir / "quality_review_samples.csv",
        score_template_csv=output_dir / "quality_scores_template.csv",
        negative_controls=negative_controls,
        sample_size=quality_sample_size,
        dpis=dpis,
    )


def run_one(
    spec: RunSpec,
    *,
    content_data: list[dict[str, Any]],
    page_indices: set[int],
    pdf_path: Path,
    output_dir: Path,
    config: AppConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = output_dir / spec.stage / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_json = run_dir / "input_content_list.json"
    output_json = run_dir / "output_content_list_PATCHED.json"
    checkpoint_json = run_dir / "output_content_list_PATCHED.checkpoint.json"
    subset, selected_pages, selected_indices = write_subset_json(
        content_data=content_data,
        page_indices=page_indices,
        output_path=input_json,
    )
    params = {
        "run_id": spec.run_id,
        "stage": spec.stage,
        "dpi": spec.dpi,
        "concurrency": spec.concurrency,
        "batch_size": spec.batch_size,
        "checkpoint_interval": spec.checkpoint_interval,
        "repeat_index": spec.repeat_index,
        "input_json": str(input_json),
        "output_json": str(output_json),
        "pdf_path": str(pdf_path),
        "subset_blocks": len(subset),
        "selected_original_indices": selected_indices,
        "selected_pdf_pages": [page + 1 for page in sorted(selected_pages)],
        "max_new_tokens": args.max_new_tokens,
        "request_timeout": args.request_timeout,
        "page_window_size": args.page_window_size,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "server_profile": {
            "mem_fraction_static": os.environ.get("RAG_FLOW_SGLANG_MEM_FRACTION_STATIC", ""),
            "context_length": os.environ.get("RAG_FLOW_SGLANG_CONTEXT_LENGTH", ""),
            "tp_size": os.environ.get("RAG_FLOW_SGLANG_TP_SIZE", ""),
            "reasoning_parser": os.environ.get("RAG_FLOW_SGLANG_REASONING_PARSER", ""),
            "quantization": os.environ.get("RAG_FLOW_SGLANG_QUANTIZATION", ""),
            "attention_backend": os.environ.get("RAG_FLOW_SGLANG_ATTENTION_BACKEND", ""),
            "kv_cache_dtype": os.environ.get("RAG_FLOW_SGLANG_KV_CACHE_DTYPE", ""),
        },
        "environment": collect_environment_metadata(),
        "started_at": time.time(),
    }
    with (run_dir / "run_params.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"Running {spec.run_id}")
    started = time.perf_counter()
    metrics_sink = JsonlMetricsSink(run_dir)
    sampler_context = (
        GpuSampler(run_dir / "gpu.csv", interval_s=args.gpu_sample_interval)
        if not args.no_gpu_log
        else _NullContext()
    )
    with sampler_context:
        add_small_icon_text(
            input_json=input_json,
            output_json=output_json,
            pdf_path=pdf_path,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.api_key,
            llm_model=args.llm_model,
            dpi=spec.dpi,
            batch_size=spec.batch_size,
            max_new_tokens=args.max_new_tokens,
            llm_timeout=args.request_timeout,
            page_window_size=args.page_window_size,
            checkpoint_interval=spec.checkpoint_interval,
            invalid_retry_limit=args.invalid_retry_limit,
            concurrency=spec.concurrency,
            checkpoint_json=checkpoint_json,
            resume=False,
            write_patching_view=args.write_patching_view,
            patching_view_pdf=run_dir / "PATCHING_VIEW.pdf",
            page_indices=selected_pages,
            metrics_sink=metrics_sink,
        )
    summary = summarize_run(run_dir, elapsed_s=time.perf_counter() - started)
    print(
        f"  requests={summary['request_count']} fields/min={summary['fields_per_min']:.2f} "
        f"p95={summary['request_duration_p95_s']:.2f}s"
    )
    return summary


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def build_parser(config: AppConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run patching parameter benchmark stages.")
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "baseline",
            "dpi",
            "quality",
            "concurrency",
            "batch-size",
            "checkpoint",
            "dpi-confirm",
            "final",
            "score",
            "report",
        ),
    )
    parser.add_argument("--input-json", type=Path, default=config.paths.sectioned_json)
    parser.add_argument("--pdf", type=Path, default=config.paths.source_pdf)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--main-pages", default=DEFAULT_MAIN_PAGES)
    parser.add_argument("--quality-pages", default=DEFAULT_QUALITY_PAGES)
    parser.add_argument("--negative-controls", type=int, default=30)
    parser.add_argument("--quality-sample-size", type=int, default=DEFAULT_QUALITY_SAMPLE_SIZE)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--dpis", default="200,250,300")
    parser.add_argument("--concurrency-values", default="1,2,4,6,8,10,12,16,20,24")
    parser.add_argument("--batch-sizes", default="C,2C,4C,140,256,512")
    parser.add_argument("--checkpoint-intervals", default="0,1,2,5,10,30,60")
    parser.add_argument("--selected-dpi", type=int)
    parser.add_argument("--selected-concurrency", type=int)
    parser.add_argument("--selected-batch-size", type=int)
    parser.add_argument("--selected-checkpoint-interval", type=int)
    parser.add_argument("--llm-base-url", default=config.models.llm_base_url)
    parser.add_argument("--api-key", default=config.models.llm_api_key)
    parser.add_argument("--llm-model", default=config.models.llm_model)
    parser.add_argument("--max-new-tokens", type=int, default=config.patching.max_new_tokens)
    parser.add_argument("--request-timeout", type=float, default=config.patching.llm_timeout)
    parser.add_argument("--page-window-size", type=int, default=config.patching.page_window_size)
    parser.add_argument("--invalid-retry-limit", type=int, default=config.patching.invalid_retry_limit)
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0)
    parser.add_argument("--no-gpu-log", action="store_true")
    parser.add_argument("--include-quality-pages-in-runs", action="store_true")
    parser.add_argument("--write-quality-crops", action="store_true")
    parser.add_argument("--quality-crop-dpi", type=int, default=config.patching.dpi)
    parser.add_argument("--no-auto-stop", action="store_true")
    parser.add_argument("--stop-timeout-ratio", type=float, default=0.02)
    parser.add_argument("--stop-throughput-drop-ratio", type=float, default=0.15)
    parser.add_argument("--stop-p95-ratio", type=float, default=2.0)
    parser.add_argument("--write-patching-view", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write templates and print planned runs without calling the LLM.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = build_parser(config)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.stage == "report":
        generate_report(args.output_dir, main_pages=args.main_pages, quality_pages=args.quality_pages)
        return
    if args.stage == "score":
        summary = backfill_quality_scores(
            scores_csv=args.output_dir / "quality_scores.csv",
            worklist_csv=args.output_dir / "report" / "quality_review_worklist.csv",
        )
        report_dir = args.output_dir / "report"
        charts_dir = report_dir / "charts"
        report_dir.mkdir(parents=True, exist_ok=True)
        charts_dir.mkdir(parents=True, exist_ok=True)
        quality_charts = _write_quality_charts(args.output_dir, report_dir, charts_dir)
        print(f"Backfilled quality scores: {summary['output_csv']}")
        print(f"  rows: {summary['rows']}")
        print(f"  average overall patch quality: {summary['score_avg']:.2f}")
        print(f"  text preservation failures: {summary['text_preserved_zero']}")
        print(f"  refreshed quality charts: {len(quality_charts)}")
        return

    main_pages = parse_pdf_pages(args.main_pages)
    quality_pages = parse_pdf_pages(args.quality_pages)
    content_data = _load_content(args.input_json)

    write_prepare_artifacts(
        content_data=content_data,
        main_pages=main_pages,
        quality_pages=quality_pages,
        output_dir=args.output_dir,
        negative_controls=args.negative_controls,
        quality_sample_size=args.quality_sample_size,
        dpis=parse_int_list(args.dpis),
    )
    if args.write_quality_crops:
        write_quality_review_crops(
            content_data=content_data,
            review_csv=args.output_dir / "quality_review_samples.csv",
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            dpi=args.quality_crop_dpi,
        )

    run_specs = build_run_specs(args, config)
    if args.dry_run or args.stage == "prepare":
        print(f"Benchmark output dir: {args.output_dir}")
        print(f"Input JSON: {args.input_json}")
        print(f"PDF: {args.pdf}")
        print(f"Main pages: {args.main_pages}")
        print(f"Quality pages: {args.quality_pages}")
        print(f"Quality sample size: {args.quality_sample_size}")
        print(f"Run page set: {'main+quality' if args.include_quality_pages_in_runs else ('quality only' if args.stage == 'quality' else 'main only')}")
        print(f"Planned runs: {len(run_specs)}")
        for spec in run_specs:
            print(
                f"  {spec.run_id}: dpi={spec.dpi} concurrency={spec.concurrency} "
                f"batch_size={spec.batch_size} checkpoint_interval={spec.checkpoint_interval}"
            )
        return

    best_concurrency_summary: dict[str, Any] | None = None
    for spec in run_specs:
        run_pages = page_indices_for_stage(
            stage=spec.stage,
            main_pages=main_pages,
            quality_pages=quality_pages,
            include_quality_pages=args.include_quality_pages_in_runs,
        )
        summary = run_one(
            spec,
            content_data=content_data,
            page_indices=run_pages,
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            config=config,
            args=args,
        )
        if spec.stage == "concurrency" and not args.no_auto_stop:
            should_stop, reason = should_stop_concurrency_search(
                summary=summary,
                best_summary=best_concurrency_summary,
                timeout_ratio=args.stop_timeout_ratio,
                throughput_drop_ratio=args.stop_throughput_drop_ratio,
                p95_ratio=args.stop_p95_ratio,
            )
            if best_concurrency_summary is None or summary["fields_per_min"] > best_concurrency_summary["fields_per_min"]:
                best_concurrency_summary = summary
            if should_stop:
                stop_path = args.output_dir / "concurrency" / "auto_stop.json"
                stop_path.parent.mkdir(parents=True, exist_ok=True)
                stop_path.write_text(
                    json.dumps(
                        {
                            "stopped_after_run_id": spec.run_id,
                            "reason": reason,
                            "summary": summary,
                            "best_summary": best_concurrency_summary,
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                print(f"Auto-stopping concurrency search after {spec.run_id}: {reason}")
                break


if __name__ == "__main__":
    main()
