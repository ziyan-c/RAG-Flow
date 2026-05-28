from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openai import OpenAI

from rag_flow.benchmark.answering import _answering_messages, _call_answering_llm, _usage_int
from rag_flow.config import AppConfig
from rag_flow.preprocessing.small_icons import strip_reasoning_text


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def run_answering_from_contexts(
    *,
    source_run_dir: Path,
    output_dir: Path,
    run_id: str,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    max_tokens: int,
    timeout: float,
    enable_thinking: bool,
    jobs: int,
    resume: bool,
) -> Path:
    source_run_dir = source_run_dir.expanduser().resolve()
    source_rows = _read_csv(source_run_dir / "answering_metrics.csv")
    run_dir = output_dir / _safe_slug(run_id)
    answers_dir = run_dir / "answers"
    contexts_dir = run_dir / "contexts"
    retrieval_dir = run_dir / "retrieval"
    responses_dir = run_dir / "llm-responses"
    for path in (answers_dir, contexts_dir, retrieval_dir, responses_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "source_run_dir": str(source_run_dir),
        "source_run_id": source_run_dir.name,
        "mode": "qwen-answer-from-archived-rag-contexts",
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "jobs": jobs,
        "latency_policy": "total_latency_seconds = archived retrieval latency + new answering latency",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    jobs = max(1, jobs)

    def run_one(index: int, source_row: dict[str, Any]) -> dict[str, Any]:
        query_id = str(source_row["query_id"])
        answer_path = answers_dir / f"{query_id}.md"
        context_path = contexts_dir / f"{query_id}.txt"
        response_path = responses_dir / f"{query_id}.json"
        retrieval_path = retrieval_dir / f"{query_id}.json"
        source_context_path = source_run_dir / "contexts" / f"{query_id}.txt"
        source_retrieval_path = source_run_dir / "retrieval" / f"{query_id}.json"
        _copy_if_exists(source_context_path, context_path)
        _copy_if_exists(source_retrieval_path, retrieval_path)
        context = context_path.read_text(encoding="utf-8") if context_path.exists() else ""

        skipped_existing = bool(resume and answer_path.exists() and answer_path.read_text(encoding="utf-8").strip())
        started = time.perf_counter()
        answer = ""
        reasoning = ""
        usage: dict[str, Any] = {}
        raw_response: dict[str, Any] = {}
        llm_error = ""
        if skipped_existing:
            answer = answer_path.read_text(encoding="utf-8")
        else:
            client = OpenAI(api_key=llm_api_key, base_url=llm_base_url, timeout=timeout)
            messages = _answering_messages(
                query=str(source_row.get("query", "")),
                final_output_content=[{"type": "text", "text": context}],
            )
            try:
                answer, reasoning, usage, raw_response = _call_answering_llm(
                    client=client,
                    model=llm_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                )
            except Exception as exc:
                llm_error = str(exc)
            answer = strip_reasoning_text(answer)
            answer_path.write_text(answer, encoding="utf-8")
            response_path.write_text(
                json.dumps(
                    {
                        "answer": answer,
                        "reasoning_content": reasoning,
                        "usage": usage,
                        "raw_response": raw_response,
                        "errors": {"llm_error": llm_error},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        answering_latency = time.perf_counter() - started
        try:
            retrieval_latency = float(source_row.get("retrieval_latency_seconds") or 0.0)
        except ValueError:
            retrieval_latency = 0.0
        row = dict(source_row)
        row.update(
            {
                "_index": index,
                "run_id": run_id,
                "source_retrieval_run_id": source_run_dir.name,
                "answering_latency_seconds": round(answering_latency, 4),
                "total_latency_seconds": round(retrieval_latency + answering_latency, 4),
                "usage_prompt_tokens": _usage_int(usage, "prompt_tokens"),
                "usage_completion_tokens": _usage_int(usage, "completion_tokens"),
                "usage_total_tokens": _usage_int(usage, "total_tokens"),
                "usage_missing": int(not bool(usage)),
                "reasoning_chars": len(reasoning or ""),
                "answer_chars": len(answer or ""),
                "llm_error": llm_error,
                "answer_path": str(answer_path),
                "context_path": str(context_path),
                "retrieval_path": str(retrieval_path),
                "response_path": str(response_path),
                "skipped_existing": int(skipped_existing),
            }
        )
        status = "skip" if skipped_existing else ("error" if llm_error else "ok")
        print(
            f"[{index}/{len(source_rows)}] {query_id}: {status} "
            f"retrieve={retrieval_latency:.2f}s answer={answering_latency:.2f}s",
            flush=True,
        )
        return row

    rows: list[dict[str, Any]] = []
    if jobs == 1:
        for index, source_row in enumerate(source_rows, start=1):
            rows.append(run_one(index, source_row))
            _write_csv(run_dir / "answering_metrics.csv", sorted(rows, key=lambda row: row["_index"]))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(run_one, index, source_row): index
                for index, source_row in enumerate(source_rows, start=1)
            }
            for future in as_completed(futures):
                rows.append(future.result())
                _write_csv(run_dir / "answering_metrics.csv", sorted(rows, key=lambda row: row["_index"]))

    rows = sorted(rows, key=lambda row: row["_index"])
    for row in rows:
        row.pop("_index", None)
    _write_csv(run_dir / "answering_metrics.csv", rows)
    summary = {
        "run_id": run_id,
        "queries": len(rows),
        "errors": sum(1 for row in rows if row.get("llm_error")),
        "avg_total_latency_seconds": round(
            sum(float(row.get("total_latency_seconds") or 0.0) for row in rows) / len(rows), 4
        )
        if rows
        else 0.0,
        "avg_usage_total_tokens": round(
            sum(int(row.get("usage_total_tokens") or 0) for row in rows)
            / max(1, sum(1 for row in rows if int(row.get("usage_total_tokens") or 0) > 0)),
            2,
        )
        if rows
        else 0.0,
        "metrics_path": str(run_dir / "answering_metrics.csv"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Generate Qwen answers from archived RAG retrieval contexts.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--llm-base-url", default=config.models.llm_base_url)
    parser.add_argument("--llm-model", default=config.models.llm_model)
    parser.add_argument("--api-key", default=config.models.llm_api_key)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    run_dir = run_answering_from_contexts(
        source_run_dir=args.source_run_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=args.api_key,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        enable_thinking=args.enable_thinking,
        jobs=args.jobs,
        resume=args.resume,
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
