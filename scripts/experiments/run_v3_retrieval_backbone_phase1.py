from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = REPO_ROOT / "scripts" / "experiments" / "v2_experiment_runner.py"
ANSWER_FROM_CONTEXTS = REPO_ROOT / "scripts" / "experiments" / "qwen_answer_from_archived_contexts.py"


@dataclass(frozen=True)
class ModelRow:
    run_id: str
    family: str
    route_mode: str
    dense_model: str
    dense_size: int
    sparse_model: str
    dense_batch_size: int = 8
    sparse_batch_size: int = 16

    @property
    def collection(self) -> str:
        return f"rag-flow-v3-{self.run_id}"


DENSE_ROWS = (
    ModelRow("p1-dense-e5large", "dense-only", "text-dense-only", "intfloat/multilingual-e5-large", 1024, "off"),
    ModelRow("p1-dense-bgelarge", "dense-only", "text-dense-only", "BAAI/bge-large-en-v1.5", 1024, "off"),
    ModelRow("p1-dense-mxbai", "dense-only", "text-dense-only", "mixedbread-ai/mxbai-embed-large-v1", 1024, "off"),
    ModelRow("p1-dense-arctic", "dense-only", "text-dense-only", "snowflake/snowflake-arctic-embed-l", 1024, "off"),
    ModelRow("p1-dense-qwen3-0p6b", "dense-only", "text-dense-only", "Qwen/Qwen3-Embedding-0.6B", 1024, "off", dense_batch_size=16),
    ModelRow("p1-dense-qwen3-4b", "dense-only", "text-dense-only", "Qwen/Qwen3-Embedding-4B", 2560, "off", dense_batch_size=4),
    ModelRow("p1-dense-qwen3-8b", "dense-only", "text-dense-only", "Qwen/Qwen3-Embedding-8B", 4096, "off", dense_batch_size=2),
)

SPARSE_ROWS = (
    ModelRow("p1-sparse-bm25", "sparse-only", "text-sparse-only", "off", 1024, "Qdrant/bm25"),
    ModelRow("p1-sparse-bm42", "sparse-only", "text-sparse-only", "off", 1024, "Qdrant/bm42-all-minilm-l6-v2-attentions"),
    ModelRow("p1-sparse-splade", "sparse-only", "text-sparse-only", "off", 1024, "prithivida/Splade_PP_en_v1", sparse_batch_size=8),
    ModelRow("p1-sparse-bge-m3", "sparse-only", "text-sparse-only", "off", 1024, "BAAI/bge-m3", sparse_batch_size=8),
)


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _row_env(row: ModelRow) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "RAG_FLOW_DENSE_MODEL": row.dense_model,
            "RAG_FLOW_DENSE_VECTOR_SIZE": str(row.dense_size),
            "RAG_FLOW_SPARSE_MODEL": row.sparse_model,
            "RAG_FLOW_DENSE_EMBEDDING_BATCH_SIZE": str(row.dense_batch_size),
            "RAG_FLOW_SPARSE_EMBEDDING_BATCH_SIZE": str(row.sparse_batch_size),
        }
    )
    return env


def _write_manifest(path: Path, rows: Sequence[ModelRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(row) | {"collection": row.collection} for row in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prepare_qa(args: argparse.Namespace) -> None:
    _run(
        [
            sys.executable,
            str(PIPELINE),
            "prepare-qa",
            "--work-root",
            str(args.work_root),
        ],
        env=dict(os.environ),
    )


def _run_retrieval_row(args: argparse.Namespace, row: ModelRow) -> None:
    summary_path = args.work_root / "runs" / f"{row.run_id}.json"
    run_dir = args.work_root / "answering-runs" / row.run_id
    if args.resume and summary_path.exists() and (run_dir / "answering_metrics.csv").exists():
        print(f"skip existing {row.run_id}", flush=True)
        return
    command = [
        sys.executable,
        str(PIPELINE),
        "run-config",
        "--work-root",
        str(args.work_root),
        "--source-root",
        str(args.source_root),
        "--output-root",
        str(args.output_root),
        "--db-root",
        str(args.db_root),
        "--collection",
        row.collection,
        "--query-set",
        str(args.work_root / "qa" / "qa50.jsonl"),
        "--run-id",
        row.run_id,
        "--chunk-mode",
        "section",
        "--chunk-max-tokens",
        "1500",
        "--chunk-overlap-tokens",
        "150",
        "--chunk-min-tokens",
        "150",
        "--text-batch-size",
        str(args.text_batch_size),
        "--context-cap",
        "10000",
        "--retrieval-k",
        "80",
        "--final-top-k",
        "20",
        "--rrf-k",
        "10",
        "--min-score-ratio",
        "1.0",
        "--route-mode",
        row.route_mode,
        "--max-tokens",
        "8000",
        "--skip-answering",
    ]
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    _run(command, env=_row_env(row))


def _answer_row(args: argparse.Namespace, row: ModelRow) -> None:
    source_run_dir = args.work_root / "answering-runs" / row.run_id
    output_run_dir = args.work_root / "answering-runs-qwen" / row.run_id
    if args.resume and (output_run_dir / "answering_metrics.csv").exists():
        print(f"skip existing qwen answers {row.run_id}", flush=True)
        return
    command = [
        sys.executable,
        str(ANSWER_FROM_CONTEXTS),
        "--source-run-dir",
        str(source_run_dir),
        "--output-dir",
        str(args.work_root / "answering-runs-qwen"),
        "--run-id",
        row.run_id,
        "--max-tokens",
        "8000",
        "--jobs",
        str(args.answer_jobs),
    ]
    if args.resume:
        command.append("--resume")
    _run(command, env=dict(os.environ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v3 phase-1 dense-only and sparse-only retrieval-backbone tests.")
    parser.add_argument("--stage", choices=("prepare", "retrieve", "answer", "all"), required=True)
    parser.add_argument("--work-root", type=Path, default=Path("thesis-v3/experiments/retrieval-backbone-v3"))
    parser.add_argument("--source-root", type=Path, default=Path(".local/CUSTOM_DATA/pdfs/source"))
    parser.add_argument("--output-root", type=Path, default=Path("output-pdfs"))
    parser.add_argument("--db-root", type=Path, default=Path("/root/autodl-tmp/rag-flow-v3-experiments/qdrant"))
    parser.add_argument("--families", choices=("all", "dense", "sparse"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--answer-jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def selected_rows(families: str) -> tuple[ModelRow, ...]:
    if families == "dense":
        return DENSE_ROWS
    if families == "sparse":
        return SPARSE_ROWS
    return (*DENSE_ROWS, *SPARSE_ROWS)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = selected_rows(args.families)
    _write_manifest(args.work_root / "phase1-row-manifest.json", rows)
    if args.stage in {"prepare", "all"}:
        _prepare_qa(args)
    if args.stage in {"retrieve", "all"}:
        for row in rows:
            _run_retrieval_row(args, row)
    if args.stage in {"answer", "all"}:
        for row in rows:
            _answer_row(args, row)


if __name__ == "__main__":
    main()
