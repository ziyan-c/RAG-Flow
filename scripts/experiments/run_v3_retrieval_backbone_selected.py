from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = REPO_ROOT / "scripts" / "experiments" / "v2_experiment_runner.py"
ANSWER_FROM_CONTEXTS = REPO_ROOT / "scripts" / "experiments" / "qwen_answer_from_archived_contexts.py"


@dataclass(frozen=True)
class DenseSpec:
    label: str
    model: str
    size: int
    batch_size: int = 8


@dataclass(frozen=True)
class SparseSpec:
    label: str
    model: str
    batch_size: int = 16


@dataclass(frozen=True)
class RunRow:
    run_id: str
    dense: DenseSpec
    sparse: SparseSpec
    route_mode: str = "text"
    rrf_k: int = 10
    query_set_name: str = "qa50"

    @property
    def collection(self) -> str:
        return f"rag-flow-v3-{self.run_id}"


DENSE: dict[str, DenseSpec] = {
    "e5large": DenseSpec("e5large", "intfloat/multilingual-e5-large", 1024),
    "bgelarge": DenseSpec("bgelarge", "BAAI/bge-large-en-v1.5", 1024),
    "mxbai": DenseSpec("mxbai", "mixedbread-ai/mxbai-embed-large-v1", 1024),
    "arctic": DenseSpec("arctic", "snowflake/snowflake-arctic-embed-l", 1024),
    "qwen3-0p6b": DenseSpec("qwen3-0p6b", "Qwen/Qwen3-Embedding-0.6B", 1024, batch_size=16),
    "qwen3-4b": DenseSpec("qwen3-4b", "Qwen/Qwen3-Embedding-4B", 2560, batch_size=4),
    "qwen3-8b": DenseSpec("qwen3-8b", "Qwen/Qwen3-Embedding-8B", 4096, batch_size=2),
}

SPARSE: dict[str, SparseSpec] = {
    "bm25": SparseSpec("bm25", "Qdrant/bm25"),
    "bm42": SparseSpec("bm42", "Qdrant/bm42-all-minilm-l6-v2-attentions"),
    "splade": SparseSpec("splade", "prithivida/Splade_PP_en_v1", batch_size=8),
    "bge-m3": SparseSpec("bge-m3", "BAAI/bge-m3", batch_size=8),
}


def _run(command: list[str], *, env: dict[str, str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _row_env(row: RunRow) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "RAG_FLOW_DENSE_MODEL": row.dense.model,
            "RAG_FLOW_DENSE_VECTOR_SIZE": str(row.dense.size),
            "RAG_FLOW_SPARSE_MODEL": row.sparse.model,
            "RAG_FLOW_DENSE_EMBEDDING_BATCH_SIZE": str(row.dense.batch_size),
            "RAG_FLOW_SPARSE_EMBEDDING_BATCH_SIZE": str(row.sparse.batch_size),
        }
    )
    return env


def _dedupe(rows: Iterable[RunRow]) -> list[RunRow]:
    seen: set[tuple[str, str, str, int, str]] = set()
    result: list[RunRow] = []
    for row in rows:
        key = (row.dense.label, row.sparse.label, row.route_mode, row.rrf_k, row.query_set_name)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _phase2_rows(*, dense: DenseSpec, sparse: SparseSpec) -> list[RunRow]:
    current_dense = DENSE["e5large"]
    current_sparse = SPARSE["bm25"]
    return _dedupe(
        (
            RunRow("p2-current-current", current_dense, current_sparse),
            RunRow(f"p2-{dense.label}-current", dense, current_sparse),
            RunRow(f"p2-current-{sparse.label}", current_dense, sparse),
            RunRow(f"p2-{dense.label}-{sparse.label}", dense, sparse),
        )
    )


def _rrf_rows(*, dense: DenseSpec, sparse: SparseSpec, values: Sequence[int]) -> list[RunRow]:
    return [
        RunRow(f"p3-rrf{value}-{dense.label}-{sparse.label}", dense, sparse, rrf_k=value)
        for value in values
    ]


def _final200_rows(*, dense: DenseSpec, sparse: SparseSpec, rrf_k: int) -> list[RunRow]:
    return _dedupe(
        (
            RunRow("p4-final200-current-current", DENSE["e5large"], SPARSE["bm25"], rrf_k=10, query_set_name="qa200"),
            RunRow(
                f"p4-final200-{dense.label}-{sparse.label}-rrf{rrf_k}",
                dense,
                sparse,
                rrf_k=rrf_k,
                query_set_name="qa200",
            ),
        )
    )


def selected_rows(args: argparse.Namespace) -> list[RunRow]:
    dense = DENSE[args.dense]
    sparse = SPARSE[args.sparse]
    if args.family == "phase2":
        return _phase2_rows(dense=dense, sparse=sparse)
    if args.family == "rrf":
        return _rrf_rows(dense=dense, sparse=sparse, values=args.rrf_values)
    if args.family == "final200":
        return _final200_rows(dense=dense, sparse=sparse, rrf_k=args.selected_rrf_k)
    raise ValueError(args.family)


def _write_manifest(path: Path, rows: Sequence[RunRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for row in rows:
        payload.append(
            {
                "run_id": row.run_id,
                "collection": row.collection,
                "route_mode": row.route_mode,
                "rrf_k": row.rrf_k,
                "query_set_name": row.query_set_name,
                "dense": asdict(row.dense),
                "sparse": asdict(row.sparse),
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _retrieve_row(args: argparse.Namespace, row: RunRow) -> None:
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
        str(args.work_root / "qa" / f"{row.query_set_name}.jsonl"),
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
        str(row.rrf_k),
        "--min-score-ratio",
        "1.0",
        "--route-mode",
        row.route_mode,
        "--max-tokens",
        "8000",
        "--skip-answering",
    ]
    _run(command, env=_row_env(row))


def _answer_row(args: argparse.Namespace, row: RunRow) -> None:
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
    parser = argparse.ArgumentParser(description="Run selected v3 retrieval-backbone fusion/final rows.")
    parser.add_argument("--stage", choices=("retrieve", "answer", "all"), required=True)
    parser.add_argument("--family", choices=("phase2", "rrf", "final200"), required=True)
    parser.add_argument("--dense", choices=sorted(DENSE), required=True)
    parser.add_argument("--sparse", choices=sorted(SPARSE), required=True)
    parser.add_argument("--selected-rrf-k", type=int, default=10)
    parser.add_argument("--rrf-values", type=int, nargs="+", default=[5, 10, 20, 60])
    parser.add_argument("--work-root", type=Path, default=Path("thesis-v3/experiments/retrieval-backbone-v3"))
    parser.add_argument("--source-root", type=Path, default=Path("source-pdfs"))
    parser.add_argument("--output-root", type=Path, default=Path("output-pdfs"))
    parser.add_argument("--db-root", type=Path, default=Path("/root/autodl-tmp/rag-flow-v3-experiments/qdrant"))
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--answer-jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = selected_rows(args)
    _write_manifest(args.work_root / f"{args.family}-row-manifest.json", rows)
    if args.stage in {"retrieve", "all"}:
        for row in rows:
            _retrieve_row(args, row)
    if args.stage in {"answer", "all"}:
        for row in rows:
            _answer_row(args, row)


if __name__ == "__main__":
    main()
