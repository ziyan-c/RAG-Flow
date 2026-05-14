from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any


EVIDENCE_ANCHOR_VERSION = 1
SNIPPET_CHAR_LIMIT = 320


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: query row must be an object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _load_json_list(path: Path, *, label: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list of {label}")
    rows = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{idx}: {label} item must be an object")
        rows.append(item)
    return rows


def _list_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = value.split("|") if "|" in value else [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        values = [value]
    result = []
    seen = set()
    for item in values:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _list_ints(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = value.split("|") if "|" in value else re.split(r"[\s,]+", value)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        raw_values = list(value)
    else:
        raw_values = [value]
    result = []
    seen = set()
    for item in raw_values:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number not in seen:
            seen.add(number)
            result.append(number)
    return result


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _chunk_id(chunk: dict[str, Any], fallback_idx: int) -> str:
    metadata = dict(chunk.get("metadata", {}))
    return str(metadata.get("chunk_id") or metadata.get("chunk_idx") or f"chunk-{fallback_idx:05d}")


def _chunk_pages(chunk: dict[str, Any]) -> list[int]:
    metadata = dict(chunk.get("metadata", {}))
    pages = _list_ints(metadata.get("page_indices"))
    if pages:
        return sorted(set(pages))
    page_idx = metadata.get("page_idx", metadata.get("page_start"))
    return sorted(set(_list_ints(page_idx)))


def _chunk_blocks(chunk: dict[str, Any]) -> list[int]:
    metadata = dict(chunk.get("metadata", {}))
    return sorted(set(_list_ints(metadata.get("block_indices"))))


def _chunk_token_count(chunk: dict[str, Any]) -> int:
    metadata = dict(chunk.get("metadata", {}))
    try:
        return int(metadata.get("token_count") or 0)
    except (TypeError, ValueError):
        return 0


def _chunk_content(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_content") or "")


def _content_snippet(text: str) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= SNIPPET_CHAR_LIMIT:
        return normalized
    return normalized[:SNIPPET_CHAR_LIMIT].rstrip()


def _chunk_lookup(chunks: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_chunk_id(chunk, idx): chunk for idx, chunk in enumerate(chunks)}


def _source_chunk_ids(query: dict[str, Any]) -> list[str]:
    ordered = []
    primary = str(query.get("primary_gold_chunk_id") or "").strip()
    if primary:
        ordered.append(primary)
    ordered.extend(_list_strings(query.get("strict_gold_chunk_ids")))
    ordered.extend(_list_strings(query.get("gold_chunk_ids", query.get("expected_chunk_ids"))))
    return _list_strings(ordered)


def _gold_pages(query: dict[str, Any]) -> list[int]:
    pages = _list_ints(query.get("gold_page_indices", query.get("expected_page_indices")))
    if pages:
        return sorted(set(pages))
    page_numbers = _list_ints(query.get("gold_page_numbers", query.get("expected_pages")))
    return sorted({page - 1 for page in page_numbers if page > 0})


def build_evidence_anchors(
    *,
    query_set: Path,
    chunks_path: Path,
    output: Path,
    audit_output: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    queries = read_jsonl(query_set)
    chunks = _load_json_list(chunks_path, label="chunks")
    by_id = _chunk_lookup(chunks)
    anchored_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for query in queries:
        source_ids = _source_chunk_ids(query)
        source_chunks = [by_id[chunk_id] for chunk_id in source_ids if chunk_id in by_id]
        primary_id = str(query.get("primary_gold_chunk_id") or "").strip()
        primary_chunk = by_id.get(primary_id) if primary_id else (source_chunks[0] if source_chunks else None)
        strict_ids = _list_strings(query.get("strict_gold_chunk_ids"))
        strict_chunks = [by_id[chunk_id] for chunk_id in strict_ids if chunk_id in by_id]

        evidence_blocks = sorted({block for chunk in source_chunks for block in _chunk_blocks(chunk)})
        primary_blocks = sorted(_chunk_blocks(primary_chunk)) if primary_chunk else []
        strict_blocks = sorted({block for chunk in strict_chunks for block in _chunk_blocks(chunk)})
        evidence_pages = sorted(
            set(_gold_pages(query)) | {page for chunk in source_chunks for page in _chunk_pages(chunk)}
        )
        snippets = [
            {
                "source_chunk_id": chunk_id,
                "snippet": _content_snippet(_chunk_content(by_id[chunk_id])),
            }
            for chunk_id in source_ids
            if chunk_id in by_id and _content_snippet(_chunk_content(by_id[chunk_id]))
        ]

        anchored = dict(query)
        anchored.update(
            {
                "evidence_anchor_version": EVIDENCE_ANCHOR_VERSION,
                "evidence_source_chunk_ids": source_ids,
                "evidence_missing_source_chunk_ids": [chunk_id for chunk_id in source_ids if chunk_id not in by_id],
                "evidence_block_indices": evidence_blocks,
                "primary_evidence_block_indices": primary_blocks,
                "strict_evidence_block_indices": strict_blocks,
                "evidence_page_indices": evidence_pages,
                "evidence_page_numbers": [page + 1 for page in evidence_pages],
                "evidence_snippets": snippets,
            }
        )
        anchored_rows.append(anchored)
        audit_rows.append(
            {
                "query_id": query.get("query_id", ""),
                "source_chunk_count": len(source_ids),
                "matched_source_chunk_count": len(source_chunks),
                "missing_source_chunk_count": len(source_ids) - len(source_chunks),
                "evidence_block_count": len(evidence_blocks),
                "evidence_page_count": len(evidence_pages),
                "snippet_count": len(snippets),
            }
        )

    write_jsonl(output, anchored_rows)
    if audit_output is not None:
        _write_csv(audit_output, audit_rows)
    return output, audit_rows


def _records_for_chunks(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for idx, chunk in enumerate(chunks):
        chunk_id = _chunk_id(chunk, idx)
        records.append(
            {
                "chunk_id": chunk_id,
                "chunk_idx": idx,
                "blocks": set(_chunk_blocks(chunk)),
                "pages": set(_chunk_pages(chunk)),
                "token_count": _chunk_token_count(chunk),
                "content": _normalize_text(_chunk_content(chunk)),
            }
        )
    return records


def _chunk_ids_for_blocks(records: Sequence[dict[str, Any]], target_blocks: set[int]) -> list[str]:
    if not target_blocks:
        return []
    scored = []
    for record in records:
        overlap = len(record["blocks"] & target_blocks)
        if overlap:
            scored.append((record["chunk_idx"], str(record["chunk_id"])))
    return [chunk_id for _, chunk_id in sorted(scored)]


def _chunk_ids_for_snippets(records: Sequence[dict[str, Any]], snippets: Sequence[dict[str, Any]]) -> list[str]:
    scored: list[tuple[int, str]] = []
    for record in records:
        content = str(record["content"])
        if not content:
            continue
        matched = False
        for snippet_row in snippets:
            snippet = _normalize_text(str(snippet_row.get("snippet") or ""))
            if len(snippet) >= 40 and snippet in content:
                matched = True
                break
        if matched:
            scored.append((record["chunk_idx"], str(record["chunk_id"])))
    return [chunk_id for _, chunk_id in sorted(scored)]


def _best_primary_chunk(
    records: Sequence[dict[str, Any]],
    *,
    primary_blocks: set[int],
    evidence_blocks: set[int],
    evidence_pages: set[int],
    candidate_ids: set[str],
) -> str:
    candidates = []
    for record in records:
        chunk_id = str(record["chunk_id"])
        if candidate_ids and chunk_id not in candidate_ids:
            continue
        primary_overlap = len(record["blocks"] & primary_blocks)
        evidence_overlap = len(record["blocks"] & evidence_blocks)
        page_overlap = len(record["pages"] & evidence_pages)
        if not (primary_overlap or evidence_overlap or page_overlap):
            continue
        candidates.append(
            (
                primary_overlap,
                evidence_overlap,
                page_overlap,
                -int(record["token_count"] or 0),
                -int(record["chunk_idx"]),
                chunk_id,
            )
        )
    if not candidates:
        return ""
    return max(candidates)[-1]


def remap_query_set_to_chunks(
    *,
    query_set: Path,
    chunks_path: Path,
    output: Path,
    audit_output: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    queries = read_jsonl(query_set)
    chunks = _load_json_list(chunks_path, label="chunks")
    records = _records_for_chunks(chunks)
    remapped_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for query in queries:
        evidence_blocks = set(_list_ints(query.get("evidence_block_indices")))
        primary_blocks = set(_list_ints(query.get("primary_evidence_block_indices")))
        strict_blocks = set(_list_ints(query.get("strict_evidence_block_indices")))
        evidence_pages = set(_list_ints(query.get("evidence_page_indices")) or _gold_pages(query))
        snippets = query.get("evidence_snippets")
        if not isinstance(snippets, list):
            snippets = []

        remap_source = "none"
        gold_chunk_ids = _chunk_ids_for_blocks(records, evidence_blocks)
        if gold_chunk_ids:
            remap_source = "block_overlap"
        elif snippets:
            gold_chunk_ids = _chunk_ids_for_snippets(records, snippets)
            if gold_chunk_ids:
                remap_source = "snippet"

        strict_gold_chunk_ids = _chunk_ids_for_blocks(records, strict_blocks) if strict_blocks else []
        primary_gold_chunk_id = _best_primary_chunk(
            records,
            primary_blocks=primary_blocks or strict_blocks or evidence_blocks,
            evidence_blocks=evidence_blocks,
            evidence_pages=evidence_pages,
            candidate_ids=set(gold_chunk_ids),
        )
        warning = ""
        if evidence_blocks and not gold_chunk_ids:
            warning = "block_anchor_not_mapped"
        elif not evidence_blocks and not gold_chunk_ids and evidence_pages:
            warning = "page_only_gold"
            remap_source = "page_fallback"

        remapped = dict(query)
        remapped.update(
            {
                "gold_chunk_ids": gold_chunk_ids,
                "strict_gold_chunk_ids": strict_gold_chunk_ids,
                "primary_gold_chunk_id": primary_gold_chunk_id,
                "gold_page_indices": sorted(evidence_pages),
                "gold_page_numbers": [page + 1 for page in sorted(evidence_pages)],
                "gold_remap_source": remap_source,
                "gold_remap_warning": warning,
                "gold_remap_chunk_count": len(gold_chunk_ids),
            }
        )
        remapped_rows.append(remapped)
        audit_rows.append(
            {
                "query_id": query.get("query_id", ""),
                "evidence_block_count": len(evidence_blocks),
                "evidence_page_count": len(evidence_pages),
                "gold_chunk_count": len(gold_chunk_ids),
                "strict_gold_chunk_count": len(strict_gold_chunk_ids),
                "primary_gold_chunk_id": primary_gold_chunk_id,
                "remap_source": remap_source,
                "warning": warning,
            }
        )

    write_jsonl(output, remapped_rows)
    if audit_output is not None:
        _write_csv(audit_output, audit_rows)
    return output, audit_rows
