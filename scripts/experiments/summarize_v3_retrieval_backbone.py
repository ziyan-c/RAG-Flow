from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_float(rows: Sequence[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        try:
            if value not in ("", None):
                values.append(float(value))
        except (TypeError, ValueError):
            continue
    return statistics.mean(values) if values else 0.0


def _rate(rows: Sequence[dict[str, Any]], key: str, predicate) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(1 for value in values if predicate(value)) / len(values) if values else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _score_path(run_dir: Path, score_tag: str) -> Path:
    if score_tag == "scored":
        return run_dir / "answering_metrics_scored.csv"
    return run_dir / f"answering_metrics_{score_tag}.csv"


def summarize_run(work_root: Path, run_id: str, *, score_tag: str, require_score: bool = False) -> dict[str, Any] | None:
    meta = _load_json(work_root / "runs" / f"{run_id}.json")
    qwen_dir = work_root / "answering-runs-qwen" / run_id
    scored = _read_csv(_score_path(qwen_dir, score_tag))
    if require_score and not scored:
        return None
    metrics = scored or _read_csv(qwen_dir / "answering_metrics.csv")
    if not metrics:
        metrics = _read_csv(work_root / "answering-runs" / run_id / "answering_metrics.csv")
    if not metrics:
        return None
    return {
        "run_id": run_id,
        "query_count": len(metrics),
        "route_mode": meta.get("route_mode", ""),
        "dense_model": meta.get("dense_model", ""),
        "dense_vector_size": meta.get("dense_vector_size", ""),
        "sparse_model": meta.get("sparse_model", ""),
        "rrf_k": meta.get("rrf_k", ""),
        "retrieval_k": meta.get("retrieval_k", ""),
        "final_top_k": meta.get("final_top_k", ""),
        "context_cap": meta.get("context_cap", ""),
        "text_index_seconds": meta.get("text_index_seconds", ""),
        "db_size_mb": round(float(meta.get("db_size_bytes") or 0) / 1024 / 1024, 2),
        "mean_score": round(_mean_float(metrics, "answer_score"), 4),
        "median_score": round(statistics.median([float(row["answer_score"]) for row in metrics if str(row.get("answer_score", "")).strip()]), 4)
        if any(str(row.get("answer_score", "")).strip() for row in metrics)
        else 0.0,
        "high_quality_rate": round(_rate(metrics, "answer_score", lambda value: value >= 4), 4),
        "low_score_rate": round(_rate(metrics, "answer_score", lambda value: value <= 2), 4),
        "avg_context_tokens": round(_mean_float(metrics, "estimated_text_tokens"), 2),
        "avg_usage_total_tokens": round(_mean_float(metrics, "usage_total_tokens"), 2),
        "avg_total_latency_seconds": round(_mean_float(metrics, "total_latency_seconds"), 4),
        "avg_retrieval_latency_seconds": round(_mean_float(metrics, "retrieval_latency_seconds"), 4),
        "avg_answering_latency_seconds": round(_mean_float(metrics, "answering_latency_seconds"), 4),
        "usage_missing_count": sum(1 for row in metrics if str(row.get("usage_missing", "")).strip() in {"1", "true", "True"}),
        "llm_error_count": sum(1 for row in metrics if str(row.get("llm_error", "")).strip()),
        "reviewer_model": next((row.get("reviewer_model", "") for row in scored if row.get("reviewer_model")), ""),
        "reviewer_reasoning_effort": next(
            (row.get("reviewer_reasoning_effort", "") for row in scored if row.get("reviewer_reasoning_effort")),
            "",
        ),
        "scoring_mode": next((row.get("scoring_mode", "") for row in scored if row.get("scoring_mode")), ""),
        "scoring_passes": next((row.get("scoring_passes", "") for row in scored if row.get("scoring_passes")), ""),
    }


def _run_ids(work_root: Path, patterns: Sequence[str]) -> list[str]:
    ids = sorted(path.stem for path in (work_root / "runs").glob("*.json"))
    if not patterns:
        return ids
    result = []
    for run_id in ids:
        if any(run_id.startswith(pattern) for pattern in patterns):
            result.append(run_id)
    return result


def _write_markdown(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id",
        "mean_score",
        "high_quality_rate",
        "low_score_rate",
        "avg_context_tokens",
        "avg_total_latency_seconds",
        "text_index_seconds",
        "db_size_mb",
        "dense_model",
        "dense_vector_size",
        "sparse_model",
        "rrf_k",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize v3 retrieval-backbone experiment runs.")
    parser.add_argument("--work-root", type=Path, default=Path("thesis-v3/experiments/retrieval-backbone-v3"))
    parser.add_argument("--score-tag", default="content_only_screen1")
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--require-score", action="store_true")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args(argv)

    rows = [
        row
        for run_id in _run_ids(args.work_root, args.prefix)
        if (row := summarize_run(args.work_root, run_id, score_tag=args.score_tag, require_score=args.require_score))
        is not None
    ]
    rows.sort(key=lambda row: (-(float(row.get("mean_score") or 0)), str(row.get("run_id") or "")))
    _write_csv(args.output_csv, rows)
    if args.output_md:
        _write_markdown(args.output_md, rows)


if __name__ == "__main__":
    main()
