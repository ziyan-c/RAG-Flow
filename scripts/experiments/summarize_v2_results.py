from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], key: str) -> float | None:
    try:
        value = row.get(key)
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def summarize_runs(work_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scored_path in sorted((work_root / "answering-runs").glob("*/answering_metrics_scored.csv")):
        run_id = scored_path.parent.name
        metrics = _read_csv(scored_path)
        scores = [value for row in metrics if (value := _float(row, "answer_score")) is not None]
        total_tokens = [value for row in metrics if (value := _float(row, "usage_total_tokens")) is not None]
        context_tokens = [value for row in metrics if (value := _float(row, "estimated_text_tokens")) is not None]
        run_json = work_root / "runs" / f"{run_id}.json"
        run_meta = json.loads(run_json.read_text(encoding="utf-8")) if run_json.exists() else {}
        summary_json = scored_path.parent / "summary.json"
        answer_summary = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.exists() else {}
        avg_context = statistics.mean(context_tokens) if context_tokens else 0.0
        mean_score = statistics.mean(scores) if scores else 0.0
        rows.append(
            {
                "run_id": run_id,
                "query_count": len(metrics),
                "mean_score": round(mean_score, 4),
                "median_score": round(statistics.median(scores), 4) if scores else 0,
                "high_quality_rate": round(sum(value >= 4 for value in scores) / len(scores), 4) if scores else 0,
                "low_score_rate": round(sum(value <= 2 for value in scores) / len(scores), 4) if scores else 0,
                "avg_total_tokens": round(statistics.mean(total_tokens), 2) if total_tokens else 0,
                "avg_context_tokens": round(avg_context, 2),
                "retrieval_side_efficiency": round(mean_score / (avg_context / 1000), 4) if avg_context else 0,
                "avg_total_latency_seconds": answer_summary.get("avg_total_latency_seconds", ""),
                "chunk_mode": run_meta.get("chunk_mode", ""),
                "chunk_max_tokens": run_meta.get("chunk_max_tokens", ""),
                "chunk_overlap_tokens": run_meta.get("chunk_overlap_tokens", ""),
                "chunk_min_tokens": run_meta.get("chunk_min_tokens", ""),
                "context_cap": run_meta.get("context_cap", ""),
                "retrieval_k": run_meta.get("retrieval_k", ""),
                "final_top_k": run_meta.get("final_top_k", ""),
                "rrf_k": run_meta.get("rrf_k", ""),
                "min_score_ratio": run_meta.get("min_score_ratio", ""),
                "route_mode": run_meta.get("route_mode", ""),
                "visual_bonus": run_meta.get("visual_bonus", ""),
                "visual_weight": run_meta.get("visual_weight", ""),
                "max_tokens": run_meta.get("max_tokens", ""),
                "final_output_images": run_meta.get("final_output_images", ""),
                "enable_thinking": run_meta.get("enable_thinking", ""),
                "text_index_seconds": run_meta.get("text_index_seconds", ""),
                "visual_index_seconds": run_meta.get("visual_index_seconds", ""),
            }
        )
    return rows


def _svg_text(x: float, y: float, text: str, *, size: int = 13, anchor: str = "middle", weight: str = "400") -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}">{escaped}</text>'


def write_chunking_score_svg(path: Path, rows: Sequence[dict[str, Any]], *, title: str) -> None:
    data = [
        row
        for row in rows
        if str(row.get("run_id", "")).startswith("chunk-coarse-") and row.get("chunk_max_tokens") != ""
    ]
    data.sort(key=lambda row: int(row["chunk_max_tokens"]))
    if not data:
        return
    width, height = 1100, 620
    left, right, top, bottom = 90, 40, 80, 130
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_score = max(float(row["mean_score"]) for row in data)
    min_score = min(float(row["mean_score"]) for row in data)
    y_min = max(0.0, min_score - 0.15)
    y_max = min(5.0, max_score + 0.15)
    if y_max <= y_min:
        y_max = y_min + 1
    bar_gap = 24
    bar_w = max(28, (chart_w - bar_gap * (len(data) - 1)) / len(data) * 0.62)
    slot_w = chart_w / len(data)

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * chart_h

    colors = ["#7aa6ff" if float(row["mean_score"]) < max_score else "#22a06b" for row in data]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}</style>',
        _svg_text(width / 2, 38, title, size=24, weight="700"),
        _svg_text(width / 2, 62, "50 QA, Qwen answer scored by three Codex passes", size=13),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#8190a8"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#8190a8"/>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left + chart_w}" y2="{yy:.1f}" stroke="#e8edf5"/>')
        parts.append(_svg_text(left - 12, yy + 4, f"{value:.2f}", size=12, anchor="end"))
    for index, row in enumerate(data):
        score = float(row["mean_score"])
        x_mid = left + slot_w * index + slot_w / 2
        bar_h = top + chart_h - y(score)
        parts.append(
            f'<rect x="{x_mid - bar_w / 2:.1f}" y="{y(score):.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="6" fill="{colors[index]}"/>'
        )
        parts.append(_svg_text(x_mid, y(score) - 9, f"{score:.3f}", size=13, weight="700"))
        label = f"{row['chunk_max_tokens']}/{row['chunk_overlap_tokens']}/{row['chunk_min_tokens']}"
        parts.append(_svg_text(x_mid, top + chart_h + 28, label, size=12))
    parts.append(_svg_text(28, top + chart_h / 2, "mean answer score", size=13, anchor="middle"))
    parts.append(_svg_text(width / 2, height - 42, "chunk max / overlap / min tokens", size=13))
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize v2 experiment outputs.")
    parser.add_argument("--work-root", type=Path, default=Path("thesis-v2/experiments/v2-final"))
    args = parser.parse_args()
    rows = summarize_runs(args.work_root)
    _write_csv(args.work_root / "tables" / "run_summary.csv", rows)
    write_chunking_score_svg(
        args.work_root / "figures" / "chunking_coarse_answer_score.svg",
        rows,
        title="Chunking Coarse Sweep: Answer Score",
    )
    print(f"Wrote {args.work_root / 'tables' / 'run_summary.csv'}")


if __name__ == "__main__":
    main()
