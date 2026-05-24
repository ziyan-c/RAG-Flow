from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


FAILURE_TYPES = {
    "none",
    "retrieval_failure",
    "answering_failure",
    "visual_failure",
    "unsupported_answer",
    "partial_answer",
    "wrong_answer",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def _load_answer(run_dir: Path, query_id: str) -> str:
    path = run_dir / "answers" / f"{query_id}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_context(run_dir: Path, query_id: str, *, context_chars: int) -> str:
    path = run_dir / "contexts" / f"{query_id}.txt"
    return _clip(path.read_text(encoding="utf-8"), context_chars) if path.exists() else ""


def _evidence_summary(query: dict[str, Any], *, limit: int = 1600) -> str:
    evidence = query.get("gold_evidence")
    if not isinstance(evidence, list):
        evidence = []
    parts = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        parts.append(
            " | ".join(
                str(value)
                for value in (
                    item.get("pdf", ""),
                    f"page {item.get('page_number', '')}",
                    item.get("modality", ""),
                    item.get("support", ""),
                )
                if str(value).strip()
            )
        )
    return _clip(" ; ".join(parts), limit)


def _prompt_for_batch(items: Sequence[dict[str, Any]], *, pass_id: int) -> str:
    payload = []
    for item in items:
        payload.append(
            {
                "query_id": item["query_id"],
                "question": item["question"],
                "canonical_answer": item["canonical_answer"],
                "gold_evidence_summary": item["gold_evidence_summary"],
                "retrieved_context_excerpt": item["retrieved_context_excerpt"],
                "system_answer": item["system_answer"],
            }
        )
    return (
        "You are Codex acting as an independent thesis evaluator. "
        f"This is scoring pass {pass_id}. Do not infer from previous passes.\n\n"
        "Score each RAG answer from 0 to 5 as an integer:\n"
        "5 = complete and supported by retrieved context; 4 = mostly correct with minor omissions; "
        "3 = partially useful but missing important details; 2 = weak or poorly supported; "
        "1 = mostly wrong; 0 = empty/irrelevant.\n"
        "Use the canonical answer and gold evidence summary as the truth source. "
        "If the answer is correct but the retrieved context excerpt does not support it, mark unsupported_answer. "
        "Use failure_type one of: none, retrieval_failure, answering_failure, visual_failure, "
        "unsupported_answer, partial_answer, wrong_answer.\n\n"
        "Return only valid JSON with this shape:\n"
        "{\"scores\":[{\"query_id\":\"...\",\"score\":5,\"failure_type\":\"none\","
        "\"canonical_comparison\":\"what the answer covers or misses versus the canonical answer\","
        "\"review_notes\":\"one concise sentence\"}]}\n\n"
        f"Items:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Codex output did not contain JSON: {text[:500]}")
    return json.loads(match.group(0))


def _run_codex(prompt: str, *, timeout: int) -> str:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(prompt)
        prompt_path = Path(handle.name)
    try:
        command = [
            "codex",
            "exec",
            "--sandbox",
            "read-only",
            "--cd",
            str(REPO_ROOT),
            "--output-last-message",
            str(prompt_path.with_suffix(".out.txt")),
            "-",
        ]
        result = subprocess.run(
            command,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        output_path = prompt_path.with_suffix(".out.txt")
        if output_path.exists():
            return output_path.read_text(encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
        return result.stdout
    finally:
        try:
            prompt_path.unlink()
        except FileNotFoundError:
            pass


def score_run(
    run_dir: Path,
    *,
    query_lookup: dict[str, dict[str, Any]],
    passes: int,
    batch_size: int,
    context_chars: int,
    timeout: int,
    jobs: int = 1,
) -> tuple[Path, list[dict[str, Any]]]:
    metrics = _read_csv(run_dir / "answering_metrics.csv")
    all_pass_rows: list[dict[str, Any]] = []
    raw_dir = run_dir / "codex-review-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, jobs)

    def score_batch(pass_id: int, start: int) -> list[dict[str, Any]]:
        batch_metrics = metrics[start : start + batch_size]
        items = []
        for row in batch_metrics:
            query_id = row["query_id"]
            query = query_lookup.get(query_id, {})
            items.append(
                {
                    "query_id": query_id,
                    "question": query.get("query") or row.get("query", ""),
                    "canonical_answer": query.get("canonical_answer", ""),
                    "gold_evidence_summary": _evidence_summary(query),
                    "retrieved_context_excerpt": _load_context(run_dir, query_id, context_chars=context_chars),
                    "system_answer": _load_answer(run_dir, query_id),
                }
            )
        batch_id = start // batch_size + 1
        prompt = _prompt_for_batch(items, pass_id=pass_id)
        raw_text = _run_codex(prompt, timeout=timeout)
        raw_path = raw_dir / f"pass-{pass_id:02d}-batch-{batch_id:04d}.txt"
        raw_path.write_text(raw_text, encoding="utf-8")
        payload = _extract_json_object(raw_text)
        scores = payload.get("scores", [])
        if not isinstance(scores, list):
            raise ValueError(f"Missing scores array in {raw_path}")
        rows: list[dict[str, Any]] = []
        for score in scores:
            if not isinstance(score, dict):
                continue
            query_id = str(score.get("query_id") or "")
            try:
                value = int(score.get("score"))
            except (TypeError, ValueError):
                value = 0
            value = max(0, min(5, value))
            failure_type = str(score.get("failure_type") or "none")
            if failure_type not in FAILURE_TYPES:
                failure_type = "none" if value >= 4 else "partial_answer"
            rows.append(
                {
                    "query_id": query_id,
                    "pass_id": pass_id,
                    "score": value,
                    "failure_type": failure_type,
                    "canonical_comparison": str(score.get("canonical_comparison") or ""),
                    "review_notes": str(score.get("review_notes") or ""),
                }
            )
        print(f"{run_dir.name}: scored pass {pass_id} batch {batch_id}", flush=True)
        return rows

    tasks = [
        (pass_id, start)
        for pass_id in range(1, passes + 1)
        for start in range(0, len(metrics), batch_size)
    ]
    if jobs == 1:
        for pass_id, start in tasks:
            all_pass_rows.extend(score_batch(pass_id, start))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(score_batch, pass_id, start): (pass_id, start) for pass_id, start in tasks}
            for future in as_completed(futures):
                all_pass_rows.extend(future.result())

    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in all_pass_rows:
        by_query.setdefault(row["query_id"], []).append(row)

    final_rows: list[dict[str, Any]] = []
    for metric in metrics:
        query_id = metric["query_id"]
        scored = by_query.get(query_id, [])
        values = [float(row["score"]) for row in scored]
        avg = statistics.mean(values) if values else 0.0
        final = dict(metric)
        final.update(
            {
                "score_pass_1": next((row["score"] for row in scored if row["pass_id"] == 1), ""),
                "score_pass_2": next((row["score"] for row in scored if row["pass_id"] == 2), ""),
                "score_pass_3": next((row["score"] for row in scored if row["pass_id"] == 3), ""),
                "answer_score": round(avg, 4),
                "score_disagreement": round(max(values) - min(values), 4) if values else "",
                "failure_type": "|".join(sorted({row["failure_type"] for row in scored if row["failure_type"] != "none"}))
                or "none",
                "canonical_comparison": " / ".join(
                    row["canonical_comparison"] for row in scored if row.get("canonical_comparison")
                ),
                "review_notes": " / ".join(row["review_notes"] for row in scored if row["review_notes"]),
            }
        )
        final_rows.append(final)
    _write_csv(run_dir / "codex_score_passes.csv", all_pass_rows)
    scored_path = run_dir / "answering_metrics_scored.csv"
    _write_csv(scored_path, final_rows)
    return scored_path, final_rows


def summarize_runs(run_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        path = run_dir / "answering_metrics_scored.csv"
        if not path.exists():
            continue
        metrics = _read_csv(path)
        scores = [float(row["answer_score"]) for row in metrics if str(row.get("answer_score", "")).strip()]
        total_tokens = [int(row["usage_total_tokens"]) for row in metrics if str(row.get("usage_total_tokens", "")).isdigit()]
        context_tokens = [
            int(row["estimated_text_tokens"])
            for row in metrics
            if str(row.get("estimated_text_tokens", "")).isdigit()
        ]
        rows.append(
            {
                "run_id": run_dir.name,
                "query_count": len(metrics),
                "mean_score": round(statistics.mean(scores), 4) if scores else 0,
                "median_score": round(statistics.median(scores), 4) if scores else 0,
                "high_quality_rate": round(sum(1 for value in scores if value >= 4) / len(scores), 4) if scores else 0,
                "low_score_rate": round(sum(1 for value in scores if value <= 2) / len(scores), 4) if scores else 0,
                "avg_usage_total_tokens": round(statistics.mean(total_tokens), 2) if total_tokens else 0,
                "avg_context_tokens": round(statistics.mean(context_tokens), 2) if context_tokens else 0,
                "retrieval_side_efficiency": round(
                    statistics.mean(scores) / (statistics.mean(context_tokens) / 1000), 4
                )
                if scores and context_tokens and statistics.mean(context_tokens) > 0
                else 0,
                "answering_side_efficiency": round(
                    statistics.mean(scores) / (statistics.mean(total_tokens) / 1000), 4
                )
                if scores and total_tokens and statistics.mean(total_tokens) > 0
                else 0,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score answering benchmark runs with Codex.")
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--context-chars", type=int, default=6000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args(argv)

    query_lookup = {str(row["query_id"]): row for row in _read_jsonl(args.query_set)}
    run_dirs = [path.expanduser().resolve() for path in args.run_dir]
    for run_dir in run_dirs:
        path, _rows = score_run(
            run_dir,
            query_lookup=query_lookup,
            passes=args.passes,
            batch_size=args.batch_size,
            context_chars=args.context_chars,
            timeout=args.timeout,
            jobs=args.jobs,
        )
        print(f"Scored {run_dir}: {path}")
    if args.summary_output:
        _write_csv(args.summary_output, summarize_runs(run_dirs))
        print(f"Summary: {args.summary_output}")


if __name__ == "__main__":
    main()
