from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_flow.benchmark.patching import (
    GpuSampler,
    JsonlMetricsSink,
    collect_environment_metadata,
    parse_batch_size_spec,
    parse_int_list,
    should_stop_concurrency_search,
)
from rag_flow.config import AppConfig
from rag_flow.preprocessing.captioning_view import write_captioning_view_pdf
from rag_flow.preprocessing.image_descriptions import (
    IMAGE_ANSWERING_CONFIDENCE_KEY,
    IMAGE_ANSWERING_POLICY_KEY,
    IMAGE_ANSWERING_REASON_KEY,
    ApproxTokenBudgeter,
    add_image_descriptions,
    collect_surrounding_context_selection,
    has_complete_image_description,
    resize_image_for_captioning,
    should_caption_image_block,
)


DEFAULT_OUTPUT_DIR = Path("thesis-v2/05-captioning/data/benchmark-runs")
DEFAULT_QUALITY_SAMPLE_SIZE = 80
DEFAULT_REVIEW_CONTEXT_TOKENS = 50000


@dataclass(frozen=True)
class CaptionRunSpec:
    stage: str
    name: str
    max_image_side: int
    max_context_tokens: int
    concurrency: int
    batch_size: int
    checkpoint_interval: int
    repeat_index: int = 1

    @property
    def run_id(self) -> str:
        repeat_suffix = f"_r{self.repeat_index}" if self.repeat_index > 1 else ""
        return (
            f"{self.stage}_{self.name}_"
            f"s{self.max_image_side}_t{self.max_context_tokens}_"
            f"c{self.concurrency}_b{self.batch_size}_k{self.checkpoint_interval}{repeat_suffix}"
        )


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _require(value: Any, message: str) -> None:
    if value is None:
        raise SystemExit(message)


def _load_content(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in MinerU content JSON: {path}")
    return data


def _write_json(path: Path, content_data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(content_data, f, ensure_ascii=False, indent=2)


def _join(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def _preview(text: str, limit: int = 220) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "..."


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Iterable[float]) -> float:
    values = [value for value in values if value is not None]
    return (sum(values) / len(values)) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int((len(values) - 1) * q + 0.5)))
    return values[idx]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _candidate_caption_text(block: dict[str, Any]) -> str:
    return "\n".join(
        text
        for text in (
            _join(block.get("image_caption", "")).strip(),
            _join(block.get("image_footnote", "")).strip(),
            str(block.get("img_path", "")).strip(),
        )
        if text
    )


def classify_image_candidate(block: dict[str, Any]) -> str:
    text = _candidate_caption_text(block).lower()
    if any(word in text for word in ("flow", "workflow", "procedure", "process")):
        return "flowchart"
    if any(word in text for word in ("architecture", "topology", "module", "server", "client")):
        return "architecture"
    if any(word in text for word in ("chart", "graph", "trend", "axis", "legend")):
        return "chart"
    if any(word in text for word in ("table", "matrix", "grid")):
        return "complex_table_or_mixed"
    if any(word in text for word in ("screenshot", "interface", "window", "button", "menu", "dialog")):
        return "ui_screenshot"
    return "other_image"


def _image_size(path: Path) -> tuple[int, int] | tuple[None, None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def collect_caption_candidates(
    content_data: list[dict[str, Any]],
    *,
    base_dir: Path,
    skip_existing: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, block in enumerate(content_data):
        if not isinstance(block, dict) or not should_caption_image_block(block):
            continue
        if skip_existing and has_complete_image_description(block):
            continue
        image_path = base_dir / str(block.get("img_path", ""))
        width, height = _image_size(image_path)
        rows.append(
            {
                "block_idx": idx,
                "page_idx": int(block.get("page_idx", 0) or 0),
                "pdf_page": int(block.get("page_idx", 0) or 0) + 1,
                "img_path": block.get("img_path", ""),
                "image_exists": image_path.exists(),
                "image_width": width,
                "image_height": height,
                "image_type": classify_image_candidate(block),
                "caption_preview": _preview(_candidate_caption_text(block)),
                "has_existing_description": has_complete_image_description(block),
            }
        )
    return rows


def _write_candidates_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "block_idx",
        "pdf_page",
        "page_idx",
        "img_path",
        "image_exists",
        "image_width",
        "image_height",
        "image_type",
        "caption_preview",
        "has_existing_description",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_quality_samples(
    candidates: list[dict[str, Any]],
    *,
    sample_size: int,
    negative_controls: int,
    seed: int,
) -> list[dict[str, Any]]:
    existing = [row for row in candidates if row.get("image_exists")]
    rng = random.Random(seed)

    negative_pool = [row for row in existing if is_negative_control_candidate(row)]
    positive_pool = [row for row in existing if row not in negative_pool]
    rng.shuffle(negative_pool)
    selected_negative = negative_pool[: max(0, min(negative_controls, sample_size))]

    remaining_slots = max(0, sample_size - len(selected_negative))
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in positive_pool:
        by_type.setdefault(str(row["image_type"]), []).append(row)

    selected: list[dict[str, Any]] = []
    for image_type in sorted(by_type):
        if len(selected) >= remaining_slots:
            break
        rows = list(by_type[image_type])
        rng.shuffle(rows)
        selected.extend(rows[:1])

    remaining = [row for row in positive_pool if row not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, remaining_slots - len(selected))])
    if len(selected) < remaining_slots:
        fallback_negative = [row for row in negative_pool if row not in selected_negative]
        selected.extend(fallback_negative[: remaining_slots - len(selected)])

    labeled_samples = [
        {**row, "sample_label": "negative_control"}
        for row in selected_negative
    ] + [
        {**row, "sample_label": "quality_sample"}
        for row in selected
    ]
    return sorted(labeled_samples[:sample_size], key=lambda row: (int(row["pdf_page"]), int(row["block_idx"])))


def is_negative_control_candidate(row: dict[str, Any]) -> bool:
    caption = str(row.get("caption_preview", "")).lower()
    path = str(row.get("img_path", "")).lower()
    if row.get("has_existing_description"):
        return True
    if str(row.get("image_type")) == "other_image":
        return True
    return any(
        keyword in caption or keyword in path
        for keyword in (
            "logo",
            "cover",
            "icon",
            "decorative",
            "placeholder",
            "background",
        )
    )


def write_quality_templates(
    *,
    samples: list[dict[str, Any]],
    base_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    review_image_paths = write_quality_review_images(samples=samples, base_dir=base_dir, output_dir=output_dir)
    review_fieldnames = [
        "sample_id",
        "sample_label",
        "image_type",
        "pdf_page",
        "page_idx",
        "block_idx",
        "img_path",
        "review_image_path",
        "caption_preview",
        "review_notes",
    ]
    score_fieldnames = [
        "sample_id",
        "sample_label",
        "image_type",
        "pdf_page",
        "block_idx",
        "img_path",
        "run_id",
        "max_image_side",
        "max_context_tokens",
        "visual_coverage_score",
        "small_text_readability_score",
        "context_sufficiency_score",
        "context_grounding_score",
        "retrieval_usefulness_score",
        "overall_quality_score",
        "hallucinated",
        "over_contextualized",
        "missing_key_details",
        "unfinished_output",
        "notes",
    ]
    with (output_dir / "quality_review_samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review_fieldnames)
        writer.writeheader()
        for sample_idx, row in enumerate(samples, start=1):
            writer.writerow(
                {
                    "sample_id": f"q{sample_idx:04d}",
                    "sample_label": row.get("sample_label", "quality_sample"),
                    "image_type": row["image_type"],
                    "pdf_page": row["pdf_page"],
                    "page_idx": row["page_idx"],
                    "block_idx": row["block_idx"],
                    "img_path": row["img_path"],
                    "review_image_path": review_image_paths.get(str(row["block_idx"]), ""),
                    "caption_preview": row["caption_preview"],
                    "review_notes": "",
                }
            )

    with (output_dir / "quality_scores_template.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=score_fieldnames)
        writer.writeheader()
        for sample_idx, row in enumerate(samples, start=1):
            writer.writerow(
                {
                    "sample_id": f"q{sample_idx:04d}",
                    "sample_label": row.get("sample_label", "quality_sample"),
                    "image_type": row["image_type"],
                    "pdf_page": row["pdf_page"],
                    "block_idx": row["block_idx"],
                    "img_path": row["img_path"],
                    "run_id": "",
                    "max_image_side": "",
                    "max_context_tokens": "",
                    "visual_coverage_score": "",
                    "small_text_readability_score": "",
                    "context_sufficiency_score": "",
                    "context_grounding_score": "",
                    "retrieval_usefulness_score": "",
                    "overall_quality_score": "",
                    "hallucinated": "",
                    "over_contextualized": "",
                    "missing_key_details": "",
                    "unfinished_output": "",
                    "notes": "",
                }
            )


def write_quality_review_images(
    *,
    samples: list[dict[str, Any]],
    base_dir: Path,
    output_dir: Path,
    thumb_side: int = 360,
) -> dict[str, str]:
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError:
        return {}

    image_dir = output_dir / "quality-review-images"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths_by_block: dict[str, str] = {}
    thumbnails = []
    for sample_idx, row in enumerate(samples, start=1):
        source = base_dir / str(row.get("img_path", ""))
        if not source.exists():
            continue
        try:
            image = Image.open(source).convert("RGB")
        except Exception:
            continue
        sample_id = f"q{sample_idx:04d}"
        filename = f"{sample_id}_p{int(row['pdf_page']):04d}_b{int(row['block_idx']):05d}.png"
        target = image_dir / filename
        image.save(target)
        paths_by_block[str(row["block_idx"])] = str(target.relative_to(output_dir))

        thumb = image.copy()
        thumb.thumbnail((thumb_side, thumb_side))
        canvas = Image.new("RGB", (thumb_side, thumb_side + 42), "white")
        canvas.paste(thumb, ((thumb_side - thumb.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_side + 8), f"{sample_id} p{row['pdf_page']} {row['image_type']}", fill=(20, 20, 20))
        thumbnails.append(canvas)

    if thumbnails:
        columns = min(4, len(thumbnails))
        rows = (len(thumbnails) + columns - 1) // columns
        contact = Image.new("RGB", (columns * thumb_side, rows * (thumb_side + 42)), "white")
        for idx, thumb in enumerate(thumbnails):
            x = (idx % columns) * thumb_side
            y = (idx // columns) * (thumb_side + 42)
            contact.paste(thumb, (x, y))
        contact.save(output_dir / "quality_contact_sheet.jpg", quality=92)
    return paths_by_block


def _split_context_sections(context: str) -> dict[str, str]:
    markers = {
        "before": "### Nearby Text Before Image",
        "current": "### Current Image Caption/Footnote",
        "after": "### Nearby Text After Image",
    }
    positions = sorted((context.find(marker), name, marker) for name, marker in markers.items() if context.find(marker) >= 0)
    parts = {"before": "", "current": "", "after": ""}
    for idx, (start, name, marker) in enumerate(positions):
        end = positions[idx + 1][0] if idx + 1 < len(positions) else len(context)
        parts[name] = context[start + len(marker) : end].strip()
    return parts


def write_run_worklist(
    *,
    content_data: list[dict[str, Any]],
    base_dir: Path,
    output_path: Path,
    max_context_tokens: int,
    review_context_tokens: int,
) -> None:
    budgeter = ApproxTokenBudgeter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in collect_caption_candidates(content_data, base_dir=base_dir, skip_existing=False):
            block_idx = int(row["block_idx"])
            model_context, model_selection = collect_surrounding_context_selection(
                content_data,
                block_idx,
                max_context_tokens=max_context_tokens,
                budgeter=budgeter,
            )
            review_context, review_selection = collect_surrounding_context_selection(
                content_data,
                block_idx,
                max_context_tokens=review_context_tokens,
                budgeter=budgeter,
            )
            model_parts = _split_context_sections(model_context)
            review_parts = _split_context_sections(review_context)
            payload = {
                **row,
                "model_context_tokens": budgeter.count(model_context),
                "review_context_tokens": budgeter.count(review_context),
                "model_context_before_indices": list(model_selection.before_indices),
                "model_context_current_indices": list(model_selection.current_indices),
                "model_context_after_indices": list(model_selection.after_indices),
                "review_context_before_indices": list(review_selection.before_indices),
                "review_context_current_indices": list(review_selection.current_indices),
                "review_context_after_indices": list(review_selection.after_indices),
                "model_context_before": model_parts["before"],
                "model_context_current": model_parts["current"],
                "model_context_after": model_parts["after"],
                "review_context_before": review_parts["before"],
                "review_context_current": review_parts["current"],
                "review_context_after": review_parts["after"],
            }
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def write_prepare_artifacts(
    *,
    content_data: list[dict[str, Any]],
    base_dir: Path,
    pdf_path: Path,
    output_dir: Path,
    quality_sample_size: int,
    negative_controls: int,
    seed: int,
    max_context_tokens: int,
    no_captioning_view: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = collect_caption_candidates(content_data, base_dir=base_dir, skip_existing=False)
    _write_candidates_csv(candidates, output_dir / "captioning_candidates.csv")
    samples = select_quality_samples(
        candidates,
        sample_size=quality_sample_size,
        negative_controls=negative_controls,
        seed=seed,
    )
    write_quality_templates(samples=samples, base_dir=base_dir, output_dir=output_dir)
    if not no_captioning_view and pdf_path.exists():
        input_json = output_dir / "prepare_input_content_list_SECTIONED_PATCHED.json"
        _write_json(input_json, content_data)
        stats = write_captioning_view_pdf(
            content_json=input_json,
            pdf_path=pdf_path,
            output_pdf=output_dir / "CAPTIONING_VIEW.pdf",
            max_context_tokens=max_context_tokens,
        )
        (output_dir / "captioning_view_summary.json").write_text(
            json.dumps(
                {
                    "output_pdf": str(stats.output_pdf),
                    "page_count": stats.page_count,
                    "pages_with_regions": stats.pages_with_regions,
                    "region_count": stats.region_count,
                    "field_counts": stats.field_counts,
                    "caption_targets": stats.caption_targets,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _repeat_specs(spec: CaptionRunSpec, repeat: int) -> list[CaptionRunSpec]:
    return [
        CaptionRunSpec(
            stage=spec.stage,
            name=spec.name,
            max_image_side=spec.max_image_side,
            max_context_tokens=spec.max_context_tokens,
            concurrency=spec.concurrency,
            batch_size=spec.batch_size,
            checkpoint_interval=spec.checkpoint_interval,
            repeat_index=idx,
        )
        for idx in range(1, repeat + 1)
    ]


def build_run_specs(args: argparse.Namespace, config: AppConfig) -> list[CaptionRunSpec]:
    stage = args.stage
    repeat = args.repeat if args.repeat is not None else (3 if stage == "final" else 1)
    if stage in {"prepare", "report"}:
        return []
    if stage == "image-side":
        return [
            spec
            for side in parse_int_list(args.image_sides)
            for spec in _repeat_specs(
                CaptionRunSpec(stage, f"s{side}", side, 10000, 1, 4, 1),
                repeat,
            )
        ]
    if stage == "context-tokens":
        _require(args.selected_image_side, "--selected-image-side is required for context-tokens stage.")
        return [
            spec
            for tokens in parse_int_list(args.context_tokens)
            for spec in _repeat_specs(
                CaptionRunSpec(stage, f"t{tokens}", args.selected_image_side, tokens, 1, 4, 1),
                repeat,
            )
        ]
    if stage == "concurrency":
        _require(args.selected_image_side, "--selected-image-side is required for concurrency stage.")
        _require(args.selected_context_tokens, "--selected-context-tokens is required for concurrency stage.")
        return [
            spec
            for concurrency in parse_int_list(args.concurrency_values)
            for spec in _repeat_specs(
                CaptionRunSpec(stage, f"c{concurrency}", args.selected_image_side, args.selected_context_tokens, concurrency, 16, 1),
                repeat,
            )
        ]
    if stage == "batch-size":
        _require(args.selected_image_side, "--selected-image-side is required for batch-size stage.")
        _require(args.selected_context_tokens, "--selected-context-tokens is required for batch-size stage.")
        _require(args.selected_concurrency, "--selected-concurrency is required for batch-size stage.")
        values = parse_batch_size_spec(args.batch_sizes, concurrency=args.selected_concurrency)
        return [
            spec
            for batch_size in values
            for spec in _repeat_specs(
                CaptionRunSpec(
                    stage,
                    f"b{batch_size}",
                    args.selected_image_side,
                    args.selected_context_tokens,
                    args.selected_concurrency,
                    batch_size,
                    1,
                ),
                repeat,
            )
        ]
    if stage == "checkpoint":
        _require(args.selected_image_side, "--selected-image-side is required for checkpoint stage.")
        _require(args.selected_context_tokens, "--selected-context-tokens is required for checkpoint stage.")
        _require(args.selected_concurrency, "--selected-concurrency is required for checkpoint stage.")
        _require(args.selected_batch_size, "--selected-batch-size is required for checkpoint stage.")
        return [
            spec
            for interval in parse_int_list(args.checkpoint_intervals)
            for spec in _repeat_specs(
                CaptionRunSpec(
                    stage,
                    f"k{interval}",
                    args.selected_image_side,
                    args.selected_context_tokens,
                    args.selected_concurrency,
                    args.selected_batch_size,
                    interval,
                ),
                repeat,
            )
        ]
    if stage == "final":
        _require(args.selected_image_side, "--selected-image-side is required for final stage.")
        _require(args.selected_context_tokens, "--selected-context-tokens is required for final stage.")
        _require(args.selected_concurrency, "--selected-concurrency is required for final stage.")
        _require(args.selected_batch_size, "--selected-batch-size is required for final stage.")
        _require(
            args.selected_checkpoint_interval,
            "--selected-checkpoint-interval is required for final stage.",
        )
        return _repeat_specs(
            CaptionRunSpec(
                stage,
                "recommended",
                args.selected_image_side,
                args.selected_context_tokens,
                args.selected_concurrency,
                args.selected_batch_size,
                args.selected_checkpoint_interval,
            ),
            repeat,
        )
    raise SystemExit(f"Unknown benchmark stage: {stage}")


def _strip_existing_descriptions(content_data: list[dict[str, Any]]) -> None:
    for block in content_data:
        if isinstance(block, dict):
            block.pop("image_description_vlm", None)
            block.pop(IMAGE_ANSWERING_POLICY_KEY, None)
            block.pop(IMAGE_ANSWERING_CONFIDENCE_KEY, None)
            block.pop(IMAGE_ANSWERING_REASON_KEY, None)


def _warmup_captioning_request(
    *,
    content_data: list[dict[str, Any]],
    base_dir: Path,
    config: AppConfig,
    args: argparse.Namespace,
    spec: CaptionRunSpec,
) -> None:
    from PIL import Image

    from rag_flow.preprocessing.image_descriptions import (
        make_captioning_llm_client,
        request_image_description_from_llm,
    )

    budgeter = ApproxTokenBudgeter()
    client = make_captioning_llm_client(base_url=args.llm_base_url, api_key=args.api_key, timeout=args.request_timeout)
    for idx, block in enumerate(content_data):
        if not isinstance(block, dict) or not should_caption_image_block(block):
            continue
        image_path = base_dir / str(block.get("img_path", ""))
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        image = resize_image_for_captioning(image, spec.max_image_side)
        context, _selection = collect_surrounding_context_selection(
            content_data,
            idx,
            max_context_tokens=spec.max_context_tokens,
            budgeter=budgeter,
        )
        prompt = (
            "Warmup captioning request. Describe the image briefly and faithfully, "
            "then judge whether the original image should accompany the description "
            "for future answering.\n\n"
            f"### Text Context:\n{context}"
        )
        request_image_description_from_llm(
            client=client,
            model=args.llm_model,
            image=image,
            prompt=prompt,
            max_tokens=args.max_new_tokens,
        )
        return


def summarize_run(run_dir: Path, *, elapsed_s: float) -> dict[str, Any]:
    requests = _read_jsonl(run_dir / "requests.jsonl")
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
        "images_per_min": (len(requests) / elapsed_s * 60.0) if elapsed_s > 0 else 0.0,
        "request_duration_avg_s": (sum(durations) / len(durations)) if durations else 0.0,
        "request_duration_p50_s": _percentile(durations, 0.50),
        "request_duration_p95_s": _percentile(durations, 0.95),
        "checkpoint_count": len(checkpoints),
        "checkpoint_duration_total_s": sum(float(row.get("duration_s", 0.0)) for row in checkpoints),
        "checkpoint_bytes_max": max((int(row.get("file_size_bytes", 0)) for row in checkpoints), default=0),
        "checkpoint_interval_count": len(checkpoint_by_reason.get("interval", [])),
        "checkpoint_interval_duration_total_s": sum(
            float(row.get("duration_s", 0.0)) for row in checkpoint_by_reason.get("interval", [])
        ),
        "checkpoint_failure_count": len(checkpoint_by_reason.get("failure", [])),
        "checkpoint_recovery_batches_max": max(checkpoint_gaps, default=0),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


def run_one(
    spec: CaptionRunSpec,
    *,
    content_data: list[dict[str, Any]],
    base_dir: Path,
    pdf_path: Path,
    output_dir: Path,
    config: AppConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = output_dir / spec.stage / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_json = run_dir / "input_content_list_SECTIONED_PATCHED.json"
    output_json = run_dir / "output_content_list_SECTIONED_PATCHED_CAPTIONED.json"
    checkpoint_json = run_dir / "output_content_list_SECTIONED_PATCHED_CAPTIONED.checkpoint.json"
    working_data = copy.deepcopy(content_data)
    _strip_existing_descriptions(working_data)
    _write_json(input_json, working_data)
    write_run_worklist(
        content_data=working_data,
        base_dir=base_dir,
        output_path=run_dir / "worklist.jsonl",
        max_context_tokens=spec.max_context_tokens,
        review_context_tokens=args.review_context_tokens,
    )
    candidates = collect_caption_candidates(working_data, base_dir=base_dir, skip_existing=False)
    params = {
        "run_id": spec.run_id,
        "stage": spec.stage,
        "max_image_side": spec.max_image_side,
        "max_context_tokens": spec.max_context_tokens,
        "concurrency": spec.concurrency,
        "batch_size": spec.batch_size,
        "checkpoint_interval": spec.checkpoint_interval,
        "repeat_index": spec.repeat_index,
        "input_json": str(input_json),
        "output_json": str(output_json),
        "pdf_path": str(pdf_path),
        "base_dir": str(base_dir),
        "candidate_count": len(candidates),
        "existing_image_count": sum(1 for row in candidates if row.get("image_exists")),
        "max_new_tokens": args.max_new_tokens,
        "request_timeout": args.request_timeout,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "review_context_tokens": args.review_context_tokens,
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
    (run_dir / "run_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Running {spec.run_id}")
    if args.warmup:
        _warmup_captioning_request(content_data=working_data, base_dir=base_dir, config=config, args=args, spec=spec)

    metrics_sink = JsonlMetricsSink(run_dir)
    sampler_context = (
        GpuSampler(run_dir / "gpu.csv", interval_s=args.gpu_sample_interval)
        if not args.no_gpu_log
        else _NullContext()
    )
    started = time.perf_counter()
    with sampler_context:
        add_image_descriptions(
            base_dir=base_dir,
            input_json=input_json,
            output_json=output_json,
            pdf_path=pdf_path,
            model_name=args.llm_model,
            max_new_tokens=args.max_new_tokens,
            batch_size=spec.batch_size,
            concurrency=spec.concurrency,
            max_context_tokens=spec.max_context_tokens,
            max_image_side=spec.max_image_side,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.api_key,
            llm_timeout=args.request_timeout,
            checkpoint_interval=spec.checkpoint_interval,
            checkpoint_json=checkpoint_json,
            resume=False,
            skip_existing=False,
            write_captioning_view=args.write_captioning_view,
            captioning_view_pdf=run_dir / "CAPTIONING_VIEW.pdf",
            review_context_tokens=args.review_context_tokens,
            metrics_sink=metrics_sink,
        )
    summary = summarize_run(run_dir, elapsed_s=time.perf_counter() - started)
    print(
        f"  requests={summary['request_count']} images/min={summary['images_per_min']:.2f} "
        f"p95={summary['request_duration_p95_s']:.2f}s"
    )
    return summary


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
        "image-side": 1,
        "context-tokens": 2,
        "concurrency": 3,
        "batch-size": 4,
        "checkpoint": 5,
        "final": 6,
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
            "max_image_side": int(params.get("max_image_side", 0) or 0),
            "max_context_tokens": int(params.get("max_context_tokens", 0) or 0),
            "concurrency": int(params.get("concurrency", 0) or 0),
            "batch_size": int(params.get("batch_size", 0) or 0),
            "checkpoint_interval": int(params.get("checkpoint_interval", 0) or 0),
            "repeat_index": int(params.get("repeat_index", 1) or 1),
            "candidate_count": int(params.get("candidate_count", 0) or 0),
            "request_count": int(summary.get("request_count", 0) or 0),
            "ok_request_count": int(summary.get("ok_request_count", 0) or 0),
            "error_request_count": int(summary.get("error_request_count", 0) or 0),
            "timeout_count": int(summary.get("timeout_count", 0) or 0),
            "written_count": int(summary.get("written_count", 0) or 0),
            "elapsed_s": _safe_float(summary.get("elapsed_s")),
            "images_per_min": _safe_float(summary.get("images_per_min")),
            "request_duration_avg_s": _safe_float(summary.get("request_duration_avg_s")),
            "request_duration_p50_s": _safe_float(summary.get("request_duration_p50_s")),
            "request_duration_p95_s": _safe_float(summary.get("request_duration_p95_s")),
            "checkpoint_count": int(summary.get("checkpoint_count", 0) or 0),
            "checkpoint_duration_total_s": _safe_float(summary.get("checkpoint_duration_total_s")),
            "checkpoint_bytes_max": int(summary.get("checkpoint_bytes_max", 0) or 0),
            "checkpoint_interval_count": int(summary.get("checkpoint_interval_count", 0) or 0),
            "checkpoint_interval_duration_total_s": _safe_float(
                summary.get("checkpoint_interval_duration_total_s")
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
            int(row["max_image_side"]),
            int(row["max_context_tokens"]),
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
        "max_image_side",
        "max_context_tokens",
        "concurrency",
        "batch_size",
        "checkpoint_interval",
        "repeat_index",
        "candidate_count",
        "request_count",
        "ok_request_count",
        "error_request_count",
        "timeout_count",
        "written_count",
        "elapsed_s",
        "images_per_min",
        "request_duration_avg_s",
        "request_duration_p50_s",
        "request_duration_p95_s",
        "checkpoint_count",
        "checkpoint_duration_total_s",
        "checkpoint_bytes_max",
        "checkpoint_interval_count",
        "checkpoint_interval_duration_total_s",
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
    width = 760
    height = 420
    left = 76
    right = 32
    top = 48
    bottom = 70
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x = min_x + 1
    if min_y == max_y:
        max_y = min_y + 1

    def sx(value: float) -> float:
        return left + (value - min_x) / (max_x - min_x) * (width - left - right)

    def sy(value: float) -> float:
        return height - bottom - (value - min_y) / (max_y - min_y) * (height - top - bottom)

    polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _count in points)
    point_marks = "\n".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{stroke}"/>'
        f'<text x="{sx(x):.1f}" y="{sy(y)-10:.1f}" text-anchor="middle" font-size="11">{y:.2f}</text>'
        for x, y, _count in points
    )
    tick_marks = "\n".join(
        f'<text x="{sx(x):.1f}" y="{height - bottom + 24}" text-anchor="middle" font-size="11">{x:g}</text>'
        for x in xs
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="26" text-anchor="middle" font-size="18" font-family="sans-serif">{_svg_escape(title)}</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>
<polyline points="{polyline}" fill="none" stroke="{stroke}" stroke-width="2.5"/>
{point_marks}
{tick_marks}
<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="13" font-family="sans-serif">{_svg_escape(x_label)}</text>
<text x="18" y="{height/2}" text-anchor="middle" font-size="13" font-family="sans-serif" transform="rotate(-90 18 {height/2})">{_svg_escape(y_label)}</text>
</svg>
""",
        encoding="utf-8",
    )
    return True


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_quality_summary_and_charts(output_dir: Path, report_dir: Path, charts_dir: Path) -> list[tuple[str, str]]:
    quality_csv = output_dir / "quality_scores.csv"
    if not quality_csv.exists():
        return []

    with quality_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    score_fields = [
        "visual_coverage_score",
        "small_text_readability_score",
        "context_sufficiency_score",
        "context_grounding_score",
        "retrieval_usefulness_score",
        "overall_quality_score",
    ]
    flag_fields = [
        "hallucinated",
        "over_contextualized",
        "missing_key_details",
        "unfinished_output",
    ]

    summary_rows: list[dict[str, Any]] = []
    for group_key in ("max_image_side", "max_context_tokens"):
        grouped: dict[float, list[dict[str, Any]]] = {}
        for row in rows:
            group_value = _optional_float(row.get(group_key))
            if group_value is None:
                continue
            grouped.setdefault(group_value, []).append(row)
        for group_value, group_rows in sorted(grouped.items()):
            summary: dict[str, Any] = {
                "group_key": group_key,
                "group_value": group_value,
                "sample_count": len(group_rows),
            }
            for field in score_fields:
                values = [_optional_float(row.get(field)) for row in group_rows]
                values = [value for value in values if value is not None]
                summary[f"{field}_avg"] = _mean(values)
                summary[f"{field}_p10"] = _percentile(values, 0.10)
                summary[f"{field}_p25"] = _percentile(values, 0.25)
                summary[f"{field}_count"] = len(values)
            for field in flag_fields:
                values = [_optional_float(row.get(field)) for row in group_rows]
                values = [value for value in values if value is not None]
                summary[f"{field}_ratio_pct"] = _mean(values) * 100.0 if values else 0.0
                summary[f"{field}_count"] = len(values)
            summary_rows.append(summary)

    if not summary_rows:
        return []

    fieldnames = ["group_key", "group_value", "sample_count"]
    for field in score_fields:
        fieldnames.extend([f"{field}_avg", f"{field}_p10", f"{field}_p25", f"{field}_count"])
    for field in flag_fields:
        fieldnames.extend([f"{field}_ratio_pct", f"{field}_count"])
    with (report_dir / "quality_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    chart_files: list[tuple[str, str]] = []
    image_side_points = [
        (float(row["group_value"]), _safe_float(row.get("overall_quality_score_avg")), int(row["sample_count"]))
        for row in summary_rows
        if row["group_key"] == "max_image_side" and int(row.get("overall_quality_score_count", 0) or 0) > 0
    ]
    context_points = [
        (float(row["group_value"]), _safe_float(row.get("overall_quality_score_avg")), int(row["sample_count"]))
        for row in summary_rows
        if row["group_key"] == "max_context_tokens" and int(row.get("overall_quality_score_count", 0) or 0) > 0
    ]
    hallucination_points = [
        (float(row["group_value"]), _safe_float(row.get("hallucinated_ratio_pct")), int(row["sample_count"]))
        for row in summary_rows
        if row["group_key"] == "max_context_tokens"
    ]
    if _write_line_svg(
        points=image_side_points,
        output_path=charts_dir / "quality_overall_by_image_side.svg",
        title="Image Side 质量评分",
        x_label="max image side",
        y_label="overall quality (0-5)",
        stroke="#7c3aed",
    ):
        chart_files.append(("quality_overall_by_image_side.svg", "Image Side 质量评分"))
    if _write_line_svg(
        points=context_points,
        output_path=charts_dir / "quality_overall_by_context_tokens.svg",
        title="Context Tokens 质量评分",
        x_label="context tokens",
        y_label="overall quality (0-5)",
        stroke="#7c3aed",
    ):
        chart_files.append(("quality_overall_by_context_tokens.svg", "Context Tokens 质量评分"))
    if _write_line_svg(
        points=hallucination_points,
        output_path=charts_dir / "quality_hallucination_by_context_tokens.svg",
        title="Context Tokens 幻觉比例",
        x_label="context tokens",
        y_label="hallucination (%)",
        stroke="#dc2626",
    ):
        chart_files.append(("quality_hallucination_by_context_tokens.svg", "Context Tokens 幻觉比例"))
    return chart_files


def generate_report(output_dir: Path) -> None:
    rows = _load_benchmark_rows(output_dir)
    report_dir = output_dir / "report"
    charts_dir = report_dir / "charts"
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(rows, report_dir / "benchmark_summary.csv")

    chart_specs = [
        ("image-side", "max_image_side", "images_per_min", "image_side_throughput.svg", "Image Side 对吞吐的影响", "max image side", "images/min", "#2563eb"),
        ("image-side", "max_image_side", "request_duration_p95_s", "image_side_p95_latency.svg", "Image Side 对 p95 latency 的影响", "max image side", "p95 latency (s)", "#dc2626"),
        ("context-tokens", "max_context_tokens", "images_per_min", "context_tokens_throughput.svg", "Context Tokens 对吞吐的影响", "context tokens", "images/min", "#2563eb"),
        ("context-tokens", "max_context_tokens", "request_duration_p95_s", "context_tokens_p95_latency.svg", "Context Tokens 对 p95 latency 的影响", "context tokens", "p95 latency (s)", "#dc2626"),
        ("concurrency", "concurrency", "images_per_min", "concurrency_throughput.svg", "Concurrency 吞吐曲线", "concurrency", "images/min", "#2563eb"),
        ("concurrency", "concurrency", "request_duration_p95_s", "concurrency_p95_latency.svg", "Concurrency p95 latency 曲线", "concurrency", "p95 latency (s)", "#dc2626"),
        ("concurrency", "concurrency", "gpu_utilization_avg_pct", "concurrency_gpu_utilization.svg", "Concurrency 平均 GPU 利用率", "concurrency", "GPU utilization (%)", "#7c3aed"),
        ("batch-size", "batch_size", "images_per_min", "batch_size_throughput.svg", "Batch Size 吞吐曲线", "batch size", "images/min", "#2563eb"),
        ("checkpoint", "checkpoint_interval", "checkpoint_interval_duration_total_s", "checkpoint_interval_overhead.svg", "周期 Checkpoint 写入开销", "checkpoint interval", "interval checkpoint time (s)", "#f59e0b"),
        ("checkpoint", "checkpoint_interval", "checkpoint_recovery_batches_max", "checkpoint_recovery_risk.svg", "Checkpoint 最大理论重跑 batch 数", "checkpoint interval", "max replay batches", "#7c3aed"),
        ("final", "repeat_index", "images_per_min", "final_repeat_throughput.svg", "最终配置重复运行吞吐", "repeat", "images/min", "#2563eb"),
    ]
    chart_files: list[tuple[str, str]] = []
    for stage, x_key, y_key, filename, title, x_label, y_label, stroke in chart_specs:
        if _write_line_svg(
            points=_group_metric(rows, stage=stage, x_key=x_key, y_key=y_key),
            output_path=charts_dir / filename,
            title=title,
            x_label=x_label,
            y_label=y_label,
            stroke=stroke,
        ):
            chart_files.append((filename, title))
    chart_files.extend(_write_quality_summary_and_charts(output_dir, report_dir, charts_dir))

    best_by_stage: dict[str, dict[str, Any]] = {}
    for row in rows:
        stage = str(row["stage"])
        if stage not in best_by_stage or float(row["images_per_min"]) > float(best_by_stage[stage]["images_per_min"]):
            best_by_stage[stage] = row

    best_lines = "\n".join(
        f'- `{stage}`: `{row["run_id"]}`，images/min={float(row["images_per_min"]):.2f}，p95={float(row["request_duration_p95_s"]):.2f}s。'
        for stage, row in sorted(best_by_stage.items(), key=lambda item: _stage_sort_key(item[0]))
    )
    chart_typst = "\n".join(
        f'#figure(image("charts/{filename}", width: 92%), caption: [{title}])'
        for filename, title in chart_files
    )
    (report_dir / "report.typ").write_text(
        f"""= Captioning Benchmark 自动报告

== 汇总数据

所有结构化结果已汇总到 `benchmark_summary.csv`。图表使用每个参数值下的重复运行均值；如果存在复核后填写的 `quality_scores.csv`，报告会额外生成 `quality_summary.csv` 和质量评分图。

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
                "# Captioning benchmark report",
                "",
                f"- Runs discovered: {len(rows)}",
                f"- Summary CSV: {report_dir / 'benchmark_summary.csv'}",
                f"- Typst report: {report_dir / 'report.typ'}",
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


def build_parser(config: AppConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run captioning parameter benchmark stages.")
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "image-side",
            "context-tokens",
            "concurrency",
            "batch-size",
            "checkpoint",
            "final",
            "report",
        ),
    )
    parser.add_argument("--input-json", type=Path, default=config.paths.patched_json)
    parser.add_argument("--base-dir", type=Path, default=config.paths.base_dir)
    parser.add_argument("--pdf", type=Path, default=config.paths.source_pdf)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quality-sample-size", type=int, default=DEFAULT_QUALITY_SAMPLE_SIZE)
    parser.add_argument("--negative-controls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--image-sides", default="1024,1536,2048,0")
    parser.add_argument("--context-tokens", default="0,2000,5000,10000,20000")
    parser.add_argument("--concurrency-values", default="1,2,4,6,8,10,12")
    parser.add_argument("--batch-sizes", default="C,2C,4C,16,32,64")
    parser.add_argument("--checkpoint-intervals", default="0,1,2,5,10,30")
    parser.add_argument("--selected-image-side", type=int)
    parser.add_argument("--selected-context-tokens", type=int)
    parser.add_argument("--selected-concurrency", type=int)
    parser.add_argument("--selected-batch-size", type=int)
    parser.add_argument("--selected-checkpoint-interval", type=int)
    parser.add_argument("--llm-base-url", default=config.models.vlm_base_url)
    parser.add_argument("--api-key", default=config.models.vlm_api_key)
    parser.add_argument("--llm-model", default=config.models.vlm_model)
    parser.add_argument("--max-new-tokens", type=int, default=config.captioning.max_new_tokens)
    parser.add_argument("--request-timeout", type=float, default=config.captioning.llm_timeout)
    parser.add_argument("--review-context-tokens", type=int, default=DEFAULT_REVIEW_CONTEXT_TOKENS)
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0)
    parser.add_argument("--no-gpu-log", action="store_true")
    parser.add_argument("--no-auto-stop", action="store_true")
    parser.add_argument("--stop-timeout-ratio", type=float, default=0.02)
    parser.add_argument("--stop-throughput-drop-ratio", type=float, default=0.15)
    parser.add_argument("--stop-p95-ratio", type=float, default=2.0)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(warmup=True)
    parser.add_argument("--write-captioning-view", action="store_true")
    parser.add_argument("--no-captioning-view", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write templates and print planned runs without calling the LLM.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = build_parser(config)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.stage == "report":
        generate_report(args.output_dir)
        return

    content_data = _load_content(args.input_json)
    write_prepare_artifacts(
        content_data=content_data,
        base_dir=args.base_dir,
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        quality_sample_size=args.quality_sample_size,
        negative_controls=args.negative_controls,
        seed=args.seed,
        max_context_tokens=config.captioning.max_context_tokens,
        no_captioning_view=args.no_captioning_view or args.stage != "prepare",
    )

    run_specs = build_run_specs(args, config)
    if args.dry_run or args.stage == "prepare":
        candidates = collect_caption_candidates(content_data, base_dir=args.base_dir, skip_existing=False)
        print(f"Captioning benchmark output dir: {args.output_dir}")
        print(f"Input JSON: {args.input_json}")
        print(f"Base dir: {args.base_dir}")
        print(f"PDF: {args.pdf}")
        print(f"Captioning candidates: {len(candidates)}")
        print(f"Existing image files: {sum(1 for row in candidates if row.get('image_exists'))}")
        print(f"Quality sample size: {args.quality_sample_size}")
        print(f"Negative controls: {args.negative_controls}")
        print(f"Review context tokens: {args.review_context_tokens}")
        print(f"Planned runs: {len(run_specs)}")
        for spec in run_specs:
            print(
                f"  {spec.run_id}: max_image_side={spec.max_image_side} "
                f"max_context_tokens={spec.max_context_tokens} concurrency={spec.concurrency} "
                f"batch_size={spec.batch_size} checkpoint_interval={spec.checkpoint_interval}"
            )
        return

    best_concurrency_summary: dict[str, Any] | None = None
    for spec in run_specs:
        summary = run_one(
            spec,
            content_data=content_data,
            base_dir=args.base_dir,
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            config=config,
            args=args,
        )
        if spec.stage == "concurrency" and not args.no_auto_stop:
            stop_summary = {**summary, "fields_per_min": summary.get("images_per_min", 0.0)}
            best_stop_summary = (
                {**best_concurrency_summary, "fields_per_min": best_concurrency_summary.get("images_per_min", 0.0)}
                if best_concurrency_summary is not None
                else None
            )
            should_stop, reason = should_stop_concurrency_search(
                summary=stop_summary,
                best_summary=best_stop_summary,
                timeout_ratio=args.stop_timeout_ratio,
                throughput_drop_ratio=args.stop_throughput_drop_ratio,
                p95_ratio=args.stop_p95_ratio,
            )
            if best_concurrency_summary is None or summary["images_per_min"] > best_concurrency_summary["images_per_min"]:
                best_concurrency_summary = summary
            if should_stop:
                print(f"Stopping concurrency search after {spec.run_id}: {reason}")
                break


if __name__ == "__main__":
    main()
