from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_flow.benchmark.answering import run_answering_benchmark
from rag_flow.chunking import create_chunks, estimate_token_count, write_chunks
from rag_flow.config import AppConfig, load_env_file, resolve_env_file
from rag_flow.indexing import upsert_colpali_vectors, upsert_text_vectors
from rag_flow.source_paths import source_name_for_pdf


DEFAULT_WORK_ROOT = Path("thesis-v2/experiments/v2-final")
DEFAULT_QA50_JSON = Path("qa-goldset/source-pdfs-qa-50.quick-stratified.json")
DEFAULT_QA200_JSON = Path("qa-goldset/source-pdfs-qa-200.codex-reviewed.json")
SUPPORT_ENV_KEYS = (
    "FASTEMBED_CACHE_PATH",
    "HF_HOME",
    "HF_ENDPOINT",
    "HUGGINGFACE_HUB_CACHE",
    "MODELSCOPE_CACHE",
    "PIP_CACHE_DIR",
    "UV_CACHE_DIR",
    "TORCH_HOME",
)


@dataclass(frozen=True)
class SourceDoc:
    source_pdf: Path
    source_name: str
    captioned_json: Path
    relative_key: Path


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", value).strip("-") or "run"


def _split_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _split_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _profile_slug(*, mode: str, max_tokens: int, overlap_tokens: int, min_tokens: int) -> str:
    return _safe_slug(f"{mode}_m{max_tokens}_o{overlap_tokens}_n{min_tokens}")


def _load_json_list(path: Path, label: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of {label}")
    return [item for item in data if isinstance(item, dict)]


def _find_pdf_by_stem(source_root: Path) -> dict[str, Path]:
    by_stem: dict[str, Path] = {}
    for pdf in sorted(source_root.rglob("*.pdf")):
        by_stem.setdefault(pdf.stem, pdf)
    return by_stem


def discover_source_docs(*, source_root: Path, output_root: Path) -> list[SourceDoc]:
    by_stem = _find_pdf_by_stem(source_root)
    docs: list[SourceDoc] = []
    for captioned in sorted(output_root.rglob("*_content_list_SECTIONED_PATCHED_CAPTIONED.json")):
        stem = captioned.name.split("_content_list_", 1)[0]
        pdf = by_stem.get(stem)
        if pdf is None:
            continue
        source_name = source_name_for_pdf(pdf, source_root=source_root)
        docs.append(
            SourceDoc(
                source_pdf=pdf,
                source_name=source_name,
                captioned_json=captioned,
                relative_key=captioned.relative_to(output_root),
            )
        )
    if not docs:
        raise FileNotFoundError(
            f"No captioned JSON files under {output_root} matched PDFs under {source_root}."
        )
    return docs


def prepare_query_set(qa_json: Path, output_jsonl: Path, *, limit: int | None = None) -> Path:
    rows = _load_json_list(qa_json, "QA rows")
    if limit is not None:
        rows = rows[:limit]
    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        evidence = row.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
        page_indices = sorted(
            {
                int(item["page_idx"])
                for item in evidence
                if isinstance(item, dict) and str(item.get("page_idx", "")).lstrip("-").isdigit()
            }
        )
        source_pdfs = row.get("source_pdfs")
        if not isinstance(source_pdfs, list):
            source_pdfs = []
        output_rows.append(
            {
                "query_id": str(row.get("id") or row.get("query_id") or f"qa-{index:04d}"),
                "query": str(row.get("question") or row.get("query") or ""),
                "canonical_answer": str(row.get("answer") or ""),
                "gold_evidence": evidence,
                "gold_page_indices": page_indices,
                "gold_page_numbers": [page + 1 for page in page_indices],
                "gold_source_pdfs": source_pdfs,
                "query_type": str(row.get("question_type") or row.get("query_type") or ""),
                "difficulty": str(row.get("difficulty") or ""),
                "requires_visual": bool(row.get("requires_visual", False)),
                "requires_multiple_pages": bool(row.get("requires_multiple_pages", False)),
                "requires_multiple_pdfs": bool(row.get("requires_multiple_pdfs", False)),
                "notes": str(row.get("notes") or ""),
            }
        )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in output_rows) + "\n",
        encoding="utf-8",
    )
    return output_jsonl


def build_chunks_for_profile(
    docs: Sequence[SourceDoc],
    *,
    output_root: Path,
    mode: str,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
) -> tuple[Path, list[dict[str, Any]]]:
    slug = _profile_slug(
        mode=mode,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
    )
    profile_root = output_root / "chunks" / slug
    manifest: list[dict[str, Any]] = []
    for doc in docs:
        chunks = create_chunks(
            doc.captioned_json,
            doc.source_name,
            mode=mode,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_tokens=min_tokens,
        )
        chunk_path = profile_root / doc.relative_key.with_name(
            doc.captioned_json.name.replace(
                "_content_list_SECTIONED_PATCHED_CAPTIONED.json",
                "_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED.json",
            )
        )
        write_chunks(chunks, chunk_path)
        token_counts = [
            int(dict(chunk.get("metadata", {})).get("token_count") or estimate_token_count(str(chunk.get("chunk_content") or "")))
            for chunk in chunks
        ]
        manifest.append(
            {
                "source_pdf": str(doc.source_pdf),
                "source_name": doc.source_name,
                "captioned_json": str(doc.captioned_json),
                "chunks_json": str(chunk_path),
                "chunk_count": len(chunks),
                "token_count_total": sum(token_counts),
                "token_count_avg": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
                "token_count_max": max(token_counts) if token_counts else 0,
                "over_max_token_chunks": sum(1 for value in token_counts if value > max_tokens),
            }
        )
    profile_root.mkdir(parents=True, exist_ok=True)
    (profile_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile_root, manifest


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _env_override(**values: str) -> dict[str, str]:
    env = dict(os.environ)
    for key, value in values.items():
        env[key] = value
        os.environ[key] = value
    return env


def export_support_env_from_rag_flow_env() -> None:
    """Export cache/mirror env vars that third-party libraries read directly."""
    values = load_env_file(resolve_env_file())
    for key in SUPPORT_ENV_KEYS:
        value = values.get(key)
        if value and not os.environ.get(key):
            os.environ[key] = value


def index_profile(
    docs: Sequence[SourceDoc],
    manifest: Sequence[dict[str, Any]],
    *,
    db_path: Path,
    collection: str,
    text_batch_size: int,
    index_visual: bool,
    visual_batch_size: int,
    visual_dpi: int,
) -> dict[str, Any]:
    if db_path.exists():
        shutil.rmtree(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _env_override(
        RAG_FLOW_DB_PATH=str(db_path),
        RAG_FLOW_COLLECTION=collection,
        RAG_FLOW_QDRANT_URL="",
    )
    config = AppConfig.from_env()
    started = time.perf_counter()
    combined_chunks: list[dict[str, Any]] = []
    for row in manifest:
        combined_chunks.extend(json.loads(Path(row["chunks_json"]).read_text(encoding="utf-8")))
    combined_chunks_path = db_path.parent / f"{db_path.name}.chunks.json"
    combined_chunks_path.write_text(
        json.dumps(combined_chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[index] {db_path.name}: upserting {len(combined_chunks)} text chunks "
        f"from {len(manifest)} documents",
        flush=True,
    )
    upsert_text_vectors(config, combined_chunks_path, batch_size=text_batch_size)
    text_seconds = time.perf_counter() - started
    visual_seconds = 0.0
    if index_visual:
        by_source = {doc.source_name: doc for doc in docs}
        visual_started = time.perf_counter()
        for row in manifest:
            doc = by_source[row["source_name"]]
            upsert_colpali_vectors(
                config,
                pdf_path=doc.source_pdf,
                source_name=doc.source_name,
                chunks_path=row["chunks_json"],
                batch_size=visual_batch_size,
                dpi=visual_dpi,
            )
        visual_seconds = time.perf_counter() - visual_started
    return {
        "db_path": str(db_path),
        "collection": collection,
        "text_index_seconds": round(text_seconds, 3),
        "visual_index_seconds": round(visual_seconds, 3),
        "index_visual": index_visual,
    }


def run_answer_config(
    *,
    query_set: Path,
    output_dir: Path,
    run_id: str,
    db_path: Path,
    collection: str,
    context_cap: int,
    retrieval_k: int,
    final_top_k: int,
    rrf_k: int,
    min_score_ratio: float,
    route_mode: str,
    visual_bonus: str,
    visual_weight: float,
    max_tokens: int,
    final_output_images: bool,
    enable_thinking: bool,
    limit: int | None,
) -> Path:
    _env_override(
        RAG_FLOW_DB_PATH=str(db_path),
        RAG_FLOW_COLLECTION=collection,
        RAG_FLOW_QDRANT_URL="",
    )
    return run_answering_benchmark(
        query_set=query_set,
        output_dir=output_dir,
        run_id=run_id,
        context_cap=context_cap,
        retrieval_k=retrieval_k,
        final_top_k=final_top_k,
        rrf_k=rrf_k,
        min_score_ratio=min_score_ratio,
        final_output_images=final_output_images,
        enable_thinking=enable_thinking,
        route_mode=route_mode,
        visual_bonus=visual_bonus,
        visual_weight=visual_weight,
        max_tokens=max_tokens,
        llm_base_url=AppConfig.from_env().models.llm_base_url,
        llm_model=AppConfig.from_env().models.llm_model,
        llm_api_key=AppConfig.from_env().models.llm_api_key,
        request_timeout=240.0,
        limit=limit,
        dry_run=False,
    )


def cmd_prepare_qa(args: argparse.Namespace) -> None:
    qa50 = prepare_query_set(args.qa50_json, args.work_root / "qa" / "qa50.jsonl")
    qa200 = prepare_query_set(args.qa200_json, args.work_root / "qa" / "qa200.jsonl")
    print(f"Wrote {qa50}")
    print(f"Wrote {qa200}")


def cmd_run_config(args: argparse.Namespace) -> None:
    docs = discover_source_docs(source_root=args.source_root, output_root=args.output_root)
    profile_root, manifest = build_chunks_for_profile(
        docs,
        output_root=args.work_root,
        mode=args.chunk_mode,
        max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.chunk_overlap_tokens,
        min_tokens=args.chunk_min_tokens,
    )
    slug = _profile_slug(
        mode=args.chunk_mode,
        max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.chunk_overlap_tokens,
        min_tokens=args.chunk_min_tokens,
    )
    db_path = args.db_root / f"{args.run_id}-{slug}"
    index_summary = index_profile(
        docs,
        manifest,
        db_path=db_path,
        collection=args.collection,
        text_batch_size=args.text_batch_size,
        index_visual=args.index_visual,
        visual_batch_size=args.visual_batch_size,
        visual_dpi=args.visual_dpi,
    )
    query_set = args.query_set
    if not query_set.exists():
        query_set = prepare_query_set(args.qa50_json, args.work_root / "qa" / "qa50.jsonl")
    answer_run = run_answer_config(
        query_set=query_set,
        output_dir=args.work_root / "answering-runs",
        run_id=args.run_id,
        db_path=db_path,
        collection=args.collection,
        context_cap=args.context_cap,
        retrieval_k=args.retrieval_k,
        final_top_k=args.final_top_k,
        rrf_k=args.rrf_k,
        min_score_ratio=args.min_score_ratio,
        route_mode=args.route_mode,
        visual_bonus=args.visual_bonus,
        visual_weight=args.visual_weight,
        max_tokens=args.max_tokens,
        final_output_images=args.final_output_images,
        enable_thinking=args.enable_thinking,
        limit=args.limit,
    )
    summary = {
        "run_id": args.run_id,
        "profile_slug": slug,
        "profile_root": str(profile_root),
        "answer_run": str(answer_run),
        "doc_count": len(docs),
        "chunk_mode": args.chunk_mode,
        "chunk_max_tokens": args.chunk_max_tokens,
        "chunk_overlap_tokens": args.chunk_overlap_tokens,
        "chunk_min_tokens": args.chunk_min_tokens,
        "context_cap": args.context_cap,
        "retrieval_k": args.retrieval_k,
        "final_top_k": args.final_top_k,
        "rrf_k": args.rrf_k,
        "min_score_ratio": args.min_score_ratio,
        "route_mode": args.route_mode,
        "visual_bonus": args.visual_bonus,
        "visual_weight": args.visual_weight,
        "max_tokens": args.max_tokens,
        "final_output_images": args.final_output_images,
        "enable_thinking": args.enable_thinking,
        **index_summary,
    }
    run_summary_path = args.work_root / "runs" / f"{_safe_slug(args.run_id)}.json"
    run_summary_path.parent.mkdir(parents=True, exist_ok=True)
    run_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows_csv(args.work_root / "chunk_profiles" / f"{slug}.csv", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_index_profile(args: argparse.Namespace) -> None:
    docs = discover_source_docs(source_root=args.source_root, output_root=args.output_root)
    profile_root, manifest = build_chunks_for_profile(
        docs,
        output_root=args.work_root,
        mode=args.chunk_mode,
        max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.chunk_overlap_tokens,
        min_tokens=args.chunk_min_tokens,
    )
    slug = _profile_slug(
        mode=args.chunk_mode,
        max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.chunk_overlap_tokens,
        min_tokens=args.chunk_min_tokens,
    )
    db_path = args.db_root / f"{args.run_id}-{slug}"
    index_summary = index_profile(
        docs,
        manifest,
        db_path=db_path,
        collection=args.collection,
        text_batch_size=args.text_batch_size,
        index_visual=args.index_visual,
        visual_batch_size=args.visual_batch_size,
        visual_dpi=args.visual_dpi,
    )
    summary = {
        "run_id": args.run_id,
        "profile_slug": slug,
        "profile_root": str(profile_root),
        "doc_count": len(docs),
        "chunk_mode": args.chunk_mode,
        "chunk_max_tokens": args.chunk_max_tokens,
        "chunk_overlap_tokens": args.chunk_overlap_tokens,
        "chunk_min_tokens": args.chunk_min_tokens,
        **index_summary,
    }
    run_summary_path = args.work_root / "runs" / f"{_safe_slug(args.run_id)}.json"
    run_summary_path.parent.mkdir(parents=True, exist_ok=True)
    run_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows_csv(args.work_root / "chunk_profiles" / f"{slug}.csv", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_answer_config(args: argparse.Namespace) -> None:
    query_set = args.query_set
    if not query_set.exists():
        query_set = prepare_query_set(args.qa50_json, args.work_root / "qa" / "qa50.jsonl")
    answer_run = run_answer_config(
        query_set=query_set,
        output_dir=args.work_root / "answering-runs",
        run_id=args.run_id,
        db_path=args.db_path,
        collection=args.collection,
        context_cap=args.context_cap,
        retrieval_k=args.retrieval_k,
        final_top_k=args.final_top_k,
        rrf_k=args.rrf_k,
        min_score_ratio=args.min_score_ratio,
        route_mode=args.route_mode,
        visual_bonus=args.visual_bonus,
        visual_weight=args.visual_weight,
        max_tokens=args.max_tokens,
        final_output_images=args.final_output_images,
        enable_thinking=args.enable_thinking,
        limit=args.limit,
    )
    summary = {
        "run_id": args.run_id,
        "answer_run": str(answer_run),
        "db_path": str(args.db_path),
        "collection": args.collection,
        "context_cap": args.context_cap,
        "retrieval_k": args.retrieval_k,
        "final_top_k": args.final_top_k,
        "rrf_k": args.rrf_k,
        "min_score_ratio": args.min_score_ratio,
        "route_mode": args.route_mode,
        "visual_bonus": args.visual_bonus,
        "visual_weight": args.visual_weight,
        "max_tokens": args.max_tokens,
        "final_output_images": args.final_output_images,
        "enable_thinking": args.enable_thinking,
        "index_visual": args.index_visual,
    }
    run_summary_path = args.work_root / "runs" / f"{_safe_slug(args.run_id)}.json"
    run_summary_path.parent.mkdir(parents=True, exist_ok=True)
    run_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run v2 thesis chunking/retrieval/answering experiment configs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    qa_parser = subparsers.add_parser("prepare-qa")
    qa_parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    qa_parser.add_argument("--qa50-json", type=Path, default=DEFAULT_QA50_JSON)
    qa_parser.add_argument("--qa200-json", type=Path, default=DEFAULT_QA200_JSON)

    run_parser = subparsers.add_parser("run-config")
    run_parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    run_parser.add_argument("--source-root", type=Path, default=Path("source-pdfs"))
    run_parser.add_argument("--output-root", type=Path, default=Path("output-pdfs"))
    run_parser.add_argument("--db-root", type=Path, default=Path("/root/autodl-tmp/rag-flow-v2-experiments/qdrant"))
    run_parser.add_argument("--collection", default="rag-flow-v2")
    run_parser.add_argument("--qa50-json", type=Path, default=DEFAULT_QA50_JSON)
    run_parser.add_argument("--query-set", type=Path, default=DEFAULT_WORK_ROOT / "qa" / "qa50.jsonl")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--chunk-mode", choices=("auto", "section", "token"), default="auto")
    run_parser.add_argument("--chunk-max-tokens", type=int, required=True)
    run_parser.add_argument("--chunk-overlap-tokens", type=int, required=True)
    run_parser.add_argument("--chunk-min-tokens", type=int, required=True)
    run_parser.add_argument("--text-batch-size", type=int, default=256)
    run_parser.add_argument("--index-visual", action="store_true")
    run_parser.add_argument("--visual-batch-size", type=int, default=8)
    run_parser.add_argument("--visual-dpi", type=int, default=200)
    run_parser.add_argument("--context-cap", type=int, default=16000)
    run_parser.add_argument("--retrieval-k", type=int, default=150)
    run_parser.add_argument("--final-top-k", type=int, default=80)
    run_parser.add_argument("--rrf-k", type=int, default=10)
    run_parser.add_argument("--min-score-ratio", type=float, default=1.0)
    run_parser.add_argument("--route-mode", choices=("text", "text-visual-naive", "text-visual-bbox"), default="text")
    run_parser.add_argument("--visual-bonus", choices=("none", "page-naive", "page-bbox"), default="none")
    run_parser.add_argument("--visual-weight", type=float, default=2.5)
    run_parser.add_argument("--max-tokens", type=int, default=4000)
    run_parser.add_argument("--final-output-images", action="store_true")
    run_parser.add_argument("--enable-thinking", action="store_true")
    run_parser.add_argument("--limit", type=int)

    index_parser = subparsers.add_parser("index-profile")
    index_parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    index_parser.add_argument("--source-root", type=Path, default=Path("source-pdfs"))
    index_parser.add_argument("--output-root", type=Path, default=Path("output-pdfs"))
    index_parser.add_argument("--db-root", type=Path, default=Path("/root/autodl-tmp/rag-flow-v2-experiments/qdrant"))
    index_parser.add_argument("--collection", default="rag-flow-v2")
    index_parser.add_argument("--run-id", required=True)
    index_parser.add_argument("--chunk-mode", choices=("auto", "section", "token"), default="auto")
    index_parser.add_argument("--chunk-max-tokens", type=int, required=True)
    index_parser.add_argument("--chunk-overlap-tokens", type=int, required=True)
    index_parser.add_argument("--chunk-min-tokens", type=int, required=True)
    index_parser.add_argument("--text-batch-size", type=int, default=8)
    index_parser.add_argument("--index-visual", action="store_true")
    index_parser.add_argument("--visual-batch-size", type=int, default=8)
    index_parser.add_argument("--visual-dpi", type=int, default=200)

    answer_parser = subparsers.add_parser("answer-config")
    answer_parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    answer_parser.add_argument("--db-path", type=Path, required=True)
    answer_parser.add_argument("--collection", default="rag-flow-v2")
    answer_parser.add_argument("--qa50-json", type=Path, default=DEFAULT_QA50_JSON)
    answer_parser.add_argument("--query-set", type=Path, default=DEFAULT_WORK_ROOT / "qa" / "qa50.jsonl")
    answer_parser.add_argument("--run-id", required=True)
    answer_parser.add_argument("--context-cap", type=int, default=16000)
    answer_parser.add_argument("--retrieval-k", type=int, default=150)
    answer_parser.add_argument("--final-top-k", type=int, default=80)
    answer_parser.add_argument("--rrf-k", type=int, default=10)
    answer_parser.add_argument("--min-score-ratio", type=float, default=1.0)
    answer_parser.add_argument("--route-mode", choices=("text", "text-visual-naive", "text-visual-bbox"), default="text")
    answer_parser.add_argument("--visual-bonus", choices=("none", "page-naive", "page-bbox"), default="none")
    answer_parser.add_argument("--visual-weight", type=float, default=2.5)
    answer_parser.add_argument("--max-tokens", type=int, default=4000)
    answer_parser.add_argument("--final-output-images", action="store_true")
    answer_parser.add_argument("--enable-thinking", action="store_true")
    answer_parser.add_argument("--index-visual", action="store_true")
    answer_parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    export_support_env_from_rag_flow_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-qa":
        cmd_prepare_qa(args)
    elif args.command == "run-config":
        cmd_run_config(args)
    elif args.command == "index-profile":
        cmd_index_profile(args)
    elif args.command == "answer-config":
        cmd_answer_config(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
