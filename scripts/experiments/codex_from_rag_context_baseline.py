from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("thesis-v2/experiments/codex-from-rag-context")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_query_set(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = _read_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list or JSONL query set: {path}")
        rows = payload
    normalized = [_normalize_query(row, index=index) for index, row in enumerate(rows, start=1)]
    return normalized[:limit] if limit is not None else normalized


def _normalize_query(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    query_id = str(row.get("query_id") or row.get("id") or f"query-{index:04d}")
    return {
        **row,
        "query_id": query_id,
        "query": str(row.get("query") or row.get("question") or ""),
        "canonical_answer": row.get("canonical_answer") or row.get("answer") or "",
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prompt_for_query(query: dict[str, Any], *, rag_context: str, source_run_id: str) -> str:
    return (
        "You are Codex answering a RAG-context-only baseline question for the RAG-Flow thesis.\n\n"
        "This baseline isolates the value of the archived RAG retrieval context. You may use only "
        "the RAG context provided in this prompt. Do not inspect original PDF files, source-pdf "
        "directories, Qdrant indexes, generated captions outside this prompt, previous experiment "
        "answers, gold answers, thesis text, or web search. Do not call any retrieval or answering "
        "pipeline. If the provided RAG context does not contain enough information, say so clearly.\n\n"
        "Return only the final answer. Cite Source and Page values when the RAG context contains "
        "them. Do not include hidden reasoning or analysis.\n\n"
        f"Question id: {query['query_id']}\n"
        f"Source RAG run: {source_run_id}\n"
        f"Question: {query['query']}\n\n"
        "Archived RAG context:\n"
        f"{rag_context}\n"
    )


def _run_codex(
    prompt: str,
    *,
    answer_path: Path,
    timeout: int,
    model: str,
    profile: str,
    profile_v2: str,
) -> tuple[int, str, str]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(prompt)
        prompt_path = Path(handle.name)
    command = [
        "codex",
        "exec",
        "--sandbox",
        "read-only",
        "--cd",
        str(REPO_ROOT),
        "--output-last-message",
        str(answer_path),
    ]
    if model:
        command.extend(["--model", model])
    if profile:
        command.extend(["--profile", profile])
    if profile_v2:
        command.extend(["--profile-v2", profile_v2])
    command.append("-")
    try:
        result = subprocess.run(
            command,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr
    finally:
        try:
            prompt_path.unlink()
        except FileNotFoundError:
            pass


def run_baseline(
    *,
    query_set: Path,
    source_run_dir: Path,
    output_dir: Path,
    run_id: str,
    limit: int | None,
    timeout: int,
    model: str,
    profile: str,
    profile_v2: str,
    dry_run: bool,
    resume: bool,
    jobs: int,
) -> Path:
    queries = _read_query_set(query_set, limit=limit)
    source_run_dir = source_run_dir.expanduser().resolve()
    source_run_id = source_run_dir.name
    source_metrics_path = source_run_dir / "answering_metrics.csv"
    source_metrics = {
        row["query_id"]: row
        for row in _read_csv(source_metrics_path)
        if row.get("query_id")
    } if source_metrics_path.exists() else {}

    run_dir = output_dir / _safe_slug(run_id)
    answers_dir = run_dir / "answers"
    prompts_dir = run_dir / "prompts"
    raw_dir = run_dir / "codex-raw"
    for path in (answers_dir, prompts_dir, raw_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "baseline_type": "codex-from-archived-rag-context",
        "query_set": str(query_set),
        "query_count": len(queries),
        "source_run_dir": str(source_run_dir),
        "source_run_id": source_run_id,
        "model": model or "codex default",
        "profile": profile,
        "profile_v2": profile_v2,
        "rules": [
            "Codex receives question text and the archived RAG context only.",
            "Original PDFs, source-pdf directories, Qdrant indexes, prior answers, gold answers, thesis text, and web search are forbidden.",
            "This baseline isolates whether the archived RAG context contains enough answer evidence for a strong Codex answerer.",
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    jobs = max(1, jobs)

    def run_one(index: int, query: dict[str, Any]) -> dict[str, Any]:
        query_id = query["query_id"]
        context_path = source_run_dir / "contexts" / f"{query_id}.txt"
        rag_context = context_path.read_text(encoding="utf-8") if context_path.exists() else ""
        prompt = _prompt_for_query(query, rag_context=rag_context, source_run_id=source_run_id)
        prompt_path = prompts_dir / f"{query_id}.txt"
        answer_path = answers_dir / f"{query_id}.md"
        stdout_path = raw_dir / f"{query_id}.stdout.txt"
        stderr_path = raw_dir / f"{query_id}.stderr.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        started = time.perf_counter()
        returncode = 0
        stdout = ""
        stderr = ""
        skipped_existing = bool(resume and answer_path.exists() and answer_path.read_text(encoding="utf-8").strip())
        if skipped_existing:
            answer = answer_path.read_text(encoding="utf-8")
        elif not dry_run:
            returncode, stdout, stderr = _run_codex(
                prompt,
                answer_path=answer_path,
                timeout=timeout,
                model=model,
                profile=profile,
                profile_v2=profile_v2,
            )
        elapsed = time.perf_counter() - started
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        answer = answer if skipped_existing else (answer_path.read_text(encoding="utf-8") if answer_path.exists() else "")
        source_row = source_metrics.get(query_id, {})

        row = {
            "_index": index,
            "query_id": query_id,
            "run_id": run_id,
            "source_run_id": source_run_id,
            "query": query["query"],
            "difficulty": query.get("difficulty", ""),
            "query_type": query.get("question_type") or query.get("query_type", ""),
            "requires_visual": int(bool(query.get("requires_visual", False))),
            "requires_multiple_pages": int(bool(query.get("requires_multiple_pages", False))),
            "requires_multiple_pdfs": int(bool(query.get("requires_multiple_pdfs", False))),
            "context_cap": source_row.get("context_cap", ""),
            "retrieval_k": source_row.get("retrieval_k", ""),
            "final_top_k": source_row.get("final_top_k", ""),
            "rrf_k": source_row.get("rrf_k", ""),
            "min_score_ratio": source_row.get("min_score_ratio", ""),
            "route_mode": "codex-from-archived-rag-context",
            "source_route_mode": source_row.get("route_mode", ""),
            "estimated_text_tokens": source_row.get("estimated_text_tokens", ""),
            "usage_prompt_tokens": 0,
            "usage_completion_tokens": 0,
            "usage_total_tokens": 0,
            "usage_missing": 1,
            "retrieval_latency_seconds": source_row.get("retrieval_latency_seconds", ""),
            "answering_latency_seconds": round(elapsed, 4),
            "total_latency_seconds": round(elapsed, 4),
            "answer_chars": len(answer),
            "codex_returncode": returncode,
            "skipped_existing": int(skipped_existing),
            "context_chars": len(rag_context),
            "source_context_path": str(context_path),
            "llm_error": stderr[-2000:] if returncode else "",
            "answer_path": str(answer_path),
            "prompt_path": str(prompt_path),
            "response_path": str(stdout_path),
            "answer_score": "",
            "failure_type": "",
            "review_notes": "",
        }
        status = "skip" if skipped_existing else f"returncode={returncode}"
        print(f"[{index}/{len(queries)}] {query_id}: {status} elapsed={elapsed:.2f}s", flush=True)
        return row

    if jobs == 1:
        for index, query in enumerate(queries, start=1):
            rows.append(run_one(index, query))
            _write_csv(run_dir / "answering_metrics.csv", sorted(rows, key=lambda row: row["_index"]))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(run_one, index, query): index
                for index, query in enumerate(queries, start=1)
            }
            for future in as_completed(futures):
                rows.append(future.result())
                _write_csv(run_dir / "answering_metrics.csv", sorted(rows, key=lambda row: row["_index"]))

    rows = sorted(rows, key=lambda row: row["_index"])
    for row in rows:
        row.pop("_index", None)
    _write_csv(run_dir / "answering_metrics.csv", rows)
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a Codex baseline constrained to archived RAG context.")
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--profile-v2", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip questions whose answer file already exists.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of concurrent Codex RAG-context calls.")
    args = parser.parse_args(argv)

    run_dir = run_baseline(
        query_set=args.query_set,
        source_run_dir=args.source_run_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        limit=args.limit,
        timeout=args.timeout,
        model=args.model,
        profile=args.profile,
        profile_v2=args.profile_v2,
        dry_run=args.dry_run,
        resume=args.resume,
        jobs=args.jobs,
    )
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
