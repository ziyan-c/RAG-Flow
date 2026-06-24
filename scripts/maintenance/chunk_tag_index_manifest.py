from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from rag_flow.chunking import create_chunks, write_chunks
from rag_flow.config import AppConfig
from rag_flow.indexing import upsert_text_vectors
from rag_flow.source_paths import source_breadcrumb, source_payload_fields
from rag_flow.tagging import write_tagged_chunks


def stage_path_for(captioned_json: Path, stage: str) -> Path:
    if captioned_json.suffix != ".json":
        raise ValueError(f"Expected .json path: {captioned_json}")
    return captioned_json.with_name(f"{captioned_json.stem}_{stage}.json")


def read_manifest(path: Path) -> list[str]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line)
    return rows


def json_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")
    return len(data)


def append_json_array_items(output, items: list[dict[str, Any]], *, first: bool) -> bool:
    for item in items:
        if not first:
            output.write(",\n")
        json.dump(item, output, ensure_ascii=False)
        first = False
    return first


def _replace_breadcrumb_prefix(content: str, breadcrumb: str) -> str:
    marker = "[Breadcrumb: "
    if not content.startswith(marker):
        return content
    closing = content.find("]")
    if closing < 0:
        return content
    return f"[Breadcrumb: {breadcrumb}]{content[closing + 1:]}"


def _replace_legacy_pending_reference(value: str, *, source_filename: str, breadcrumb: str) -> str:
    marker = "pending_2026-06-23_downloaded_pdfs"
    if marker not in value:
        return value

    result = value
    while marker in result:
        start = result.find(marker)
        end = result.find(source_filename, start)
        if end >= 0:
            end += len(source_filename)
            result = f"{result[:start]}{breadcrumb}{result[end:]}"
        else:
            result = result.replace(marker, breadcrumb)
            break
    return result


def _replace_legacy_pending_references(value: Any, *, source_filename: str, breadcrumb: str) -> Any:
    if isinstance(value, str):
        return _replace_legacy_pending_reference(
            value,
            source_filename=source_filename,
            breadcrumb=breadcrumb,
        )
    if isinstance(value, list):
        return [
            _replace_legacy_pending_references(
                item,
                source_filename=source_filename,
                breadcrumb=breadcrumb,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _replace_legacy_pending_references(
                item,
                source_filename=source_filename,
                breadcrumb=breadcrumb,
            )
            for key, item in value.items()
        }
    return value


def normalize_chunk_source(chunks: list[dict[str, Any]], source_name: str) -> None:
    source_fields = source_payload_fields(source_name)
    source_relpath = source_fields["source_relpath"]
    source_filename = source_fields["source_filename"]
    filename_stem = Path(source_filename).stem

    for fallback_idx, chunk in enumerate(chunks):
        metadata = chunk.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Chunk metadata must be an object.")

        chunk_idx = metadata.get("chunk_idx", fallback_idx)
        try:
            chunk_idx_int = int(chunk_idx)
        except (TypeError, ValueError):
            chunk_idx_int = fallback_idx

        section_path = metadata.get("section_path") or ()
        if not isinstance(section_path, (list, tuple)):
            section_path = ()
        breadcrumb = source_breadcrumb(source_relpath, section_path)

        metadata.update(source_fields)
        metadata["breadcrumb"] = breadcrumb
        metadata["chunk_id"] = f"{source_relpath}::{filename_stem}-chunk-{chunk_idx_int:05d}"
        chunk["metadata"] = _replace_legacy_pending_references(
            metadata,
            source_filename=source_filename,
            breadcrumb=breadcrumb,
        )

        content = str(chunk.get("chunk_content") or "")
        content = _replace_breadcrumb_prefix(content, breadcrumb)
        chunk["chunk_content"] = _replace_legacy_pending_reference(
            content,
            source_filename=source_filename,
            breadcrumb=breadcrumb,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk, metadata-tag, and index a manifest of already-captioned PDFs."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--combined-json", required=True, type=Path)
    parser.add_argument("--stats-json", required=True, type=Path)
    parser.add_argument("--reset-db", action="store_true")
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = AppConfig.from_env()
    source_root = (args.source_root or config.paths.source_root or Path(".local/CUSTOM_DATA/pdfs/source")).expanduser()
    output_root = (args.output_root or config.mineru.output_dir).expanduser()
    source_root = source_root.resolve()
    output_root = output_root.resolve()

    relpaths = read_manifest(args.manifest)
    if not relpaths:
        raise SystemExit("Manifest is empty.")

    if args.reset_db:
        db_path = config.paths.db_path.expanduser()
        if db_path.exists():
            print(f"Removing existing Qdrant local DB: {db_path}", flush=True)
            shutil.rmtree(db_path)

    args.combined_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)

    stats: list[dict[str, Any]] = []
    combined_chunk_count = 0
    first = True

    with args.combined_json.open("w", encoding="utf-8") as combined:
        combined.write("[\n")
        for index, relpath in enumerate(relpaths, start=1):
            rel = Path(relpath)
            source_pdf = source_root / rel
            captioned_json = output_root / rel.with_suffix("") / "hybrid_auto" / (
                f"{rel.stem}_content_list_SECTIONED_PATCHED_CAPTIONED.json"
            )
            chunks_json = stage_path_for(captioned_json, "CHUNKED")
            tagged_json = stage_path_for(chunks_json, "TAGGED")

            if not source_pdf.exists():
                raise FileNotFoundError(source_pdf)
            if not captioned_json.exists():
                raise FileNotFoundError(captioned_json)

            if args.force or not chunks_json.exists():
                chunks = create_chunks(
                    captioned_json,
                    relpath,
                    mode=config.chunking.mode,
                    max_tokens=config.chunking.max_tokens,
                    overlap_tokens=config.chunking.overlap_tokens,
                    min_tokens=config.chunking.min_tokens,
                )
                normalize_chunk_source(chunks, relpath)
                write_chunks(chunks, chunks_json)
                chunk_count = len(chunks)
                chunk_action = "written"
            else:
                chunk_count = json_count(chunks_json)
                chunk_action = "existing"

            if args.force or not tagged_json.exists():
                tag_stats = write_tagged_chunks(
                    chunks_json=chunks_json,
                    output_json=tagged_json,
                    source_pdf=source_pdf,
                    source_name=relpath,
                    source_root=source_root,
                    require_metadata=True,
                    strict=True,
                )
                tagged_count = tag_stats.tagged_count
                tag_action = "written"
            else:
                tagged_count = json_count(tagged_json)
                tag_action = "existing"

            with tagged_json.open("r", encoding="utf-8") as handle:
                tagged_chunks = json.load(handle)
            if not isinstance(tagged_chunks, list):
                raise ValueError(f"Expected list JSON: {tagged_json}")
            first = append_json_array_items(combined, tagged_chunks, first=first)
            combined_chunk_count += len(tagged_chunks)

            row = {
                "index": index,
                "source_relpath": relpath,
                "captioned_json": str(captioned_json),
                "chunks_json": str(chunks_json),
                "tagged_json": str(tagged_json),
                "chunk_count": chunk_count,
                "tagged_count": tagged_count,
                "chunk_action": chunk_action,
                "tag_action": tag_action,
            }
            stats.append(row)
            if index == 1 or index % 25 == 0 or index == len(relpaths):
                print(
                    f"[chunk/tag] {index}/{len(relpaths)} docs, "
                    f"{combined_chunk_count} tagged chunks combined",
                    flush=True,
                )

        combined.write("\n]\n")

    summary = {
        "manifest": str(args.manifest),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "combined_json": str(args.combined_json),
        "document_count": len(relpaths),
        "combined_chunk_count": combined_chunk_count,
        "db_path": str(config.paths.db_path),
        "collection": config.paths.collection_name,
        "rows": stats,
    }
    args.stats_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[chunk/tag] complete: {len(relpaths)} docs, {combined_chunk_count} chunks, "
        f"stats={args.stats_json}",
        flush=True,
    )

    if args.no_index:
        print("[index] skipped by --no-index", flush=True)
        return

    print(f"[index] combined tagged chunks: {args.combined_json}", flush=True)
    upsert_text_vectors(config, args.combined_json, batch_size=config.indexing.text_batch_size)
    print("[index] complete", flush=True)


if __name__ == "__main__":
    main()
