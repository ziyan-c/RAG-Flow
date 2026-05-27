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
DEFAULT_OUTPUT_DIR = Path("thesis-v2/experiments/codex-direct-pdf")


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
    if limit is not None:
        normalized = normalized[:limit]
    return normalized


def _normalize_query(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    query_id = str(row.get("query_id") or row.get("id") or f"query-{index:04d}")
    question = str(row.get("query") or row.get("question") or "")
    source_pdfs = row.get("source_pdfs")
    if not isinstance(source_pdfs, list):
        source_pdfs = []
    evidence = row.get("gold_evidence")
    if evidence is None:
        evidence = row.get("evidence", [])
    return {
        **row,
        "query_id": query_id,
        "query": question,
        "canonical_answer": row.get("canonical_answer") or row.get("answer") or "",
        "gold_evidence": evidence if isinstance(evidence, list) else [],
        "source_pdfs": [str(pdf) for pdf in source_pdfs],
    }


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


def _prompt_for_query(query: dict[str, Any]) -> str:
    source_pdfs = query.get("source_pdfs") or []
    return (
        "You are Codex answering a direct-PDF baseline question for the RAG-Flow thesis.\n\n"
        "This is a no RAG-Flow baseline. You may inspect only the original PDF files listed below "
        "and may use ordinary PDF/text/image inspection tools on those files. Do not use RAG-Flow "
        "chunks, Qdrant indexes, generated captions, evidence cards, gold answers, thesis text, "
        "previous experiment outputs, or web search. Do not call any RAG retrieval or answering "
        "pipeline. If the listed PDFs do not contain enough information, say so.\n\n"
        "Return only the final answer. Cite the relevant PDF filename and page number when possible. "
        "Do not include hidden reasoning or analysis.\n\n"
        f"Question id: {query['query_id']}\n"
        f"Question: {query['query']}\n\n"
        "Original PDF files allowed for this question:\n"
        f"{json.dumps(source_pdfs, ensure_ascii=False, indent=2)}\n"
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
    run_dir = output_dir / _safe_slug(run_id)
    answers_dir = run_dir / "answers"
    prompts_dir = run_dir / "prompts"
    raw_dir = run_dir / "codex-raw"
    for path in (answers_dir, prompts_dir, raw_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "baseline_type": "codex-direct-original-pdf",
        "query_set": str(query_set),
        "query_count": len(queries),
        "model": model or "codex default",
        "profile": profile,
        "profile_v2": profile_v2,
        "rules": [
            "Codex receives question text and original PDF paths only.",
            "RAG-Flow chunks, Qdrant indexes, generated captions, evidence cards, gold answers, and thesis text are forbidden.",
            "This is a direct-PDF closed-document baseline, not a RAG-Flow retrieval run.",
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    jobs = max(1, jobs)

    def run_one(index: int, query: dict[str, Any]) -> dict[str, Any]:
        query_id = query["query_id"]
        prompt = _prompt_for_query(query)
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

        row = {
            "_index": index,
            "query_id": query_id,
            "run_id": run_id,
            "query": query["query"],
            "difficulty": query.get("difficulty", ""),
            "query_type": query.get("question_type") or query.get("query_type", ""),
            "requires_visual": int(bool(query.get("requires_visual", False))),
            "requires_multiple_pages": int(bool(query.get("requires_multiple_pages", False))),
            "requires_multiple_pdfs": int(bool(query.get("requires_multiple_pdfs", False))),
            "source_pdfs": "|".join(query.get("source_pdfs") or []),
            "context_cap": 0,
            "retrieval_k": 0,
            "final_top_k": 0,
            "rrf_k": 0,
            "min_score_ratio": "",
            "route_mode": "codex-direct-original-pdf",
            "estimated_text_tokens": 0,
            "usage_prompt_tokens": 0,
            "usage_completion_tokens": 0,
            "usage_total_tokens": 0,
            "usage_missing": 1,
            "retrieval_latency_seconds": 0,
            "answering_latency_seconds": round(elapsed, 4),
            "total_latency_seconds": round(elapsed, 4),
            "answer_chars": len(answer),
            "codex_returncode": returncode,
            "skipped_existing": int(skipped_existing),
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
    parser = argparse.ArgumentParser(description="Run a Codex direct-original-PDF no-RAG baseline.")
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--profile-v2", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip questions whose answer file already exists.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of concurrent Codex direct-PDF calls.")
    args = parser.parse_args(argv)

    run_dir = run_baseline(
        query_set=args.query_set,
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
