from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .config import AppConfig
from .mineru import find_content_json
from .source_paths import source_breadcrumb, source_name_for_pdf, source_payload_fields, source_root_from_input_path
from .table_continuations import (
    TABLE_CONTINUATION_INDICES_KEY,
    TABLE_CONTINUATION_MASTER_IDX_KEY,
    build_table_continuation_map,
)


IGNORE_TYPES = {"header", "footer", "page_number"}
SECTION_FIELD_KEYS = (
    "section_path",
    "section_title",
    "section_level",
    "section_source",
    "section_confidence",
    "section_match_score",
    "section_outline_index",
)


@dataclass(frozen=True)
class OutlineEntry:
    outline_index: int
    level: int
    title: str
    page_idx: int
    section_path: tuple[str, ...]
    dest_y: float | None = None
    page_height: float | None = None


@dataclass(frozen=True)
class SectionEvent:
    outline_index: int
    target_block_idx: int
    title: str
    level: int
    section_path: tuple[str, ...]
    section_source: str
    section_confidence: float
    section_match_score: float


@dataclass(frozen=True)
class SectionAuditEntry:
    outline_index: int
    outline_title: str
    outline_level: int
    outline_page: int
    section_path: tuple[str, ...]
    match_type: str
    matched_block_idx: int | None
    matched_text: str
    confidence: float
    match_score: float
    y_distance: float | None
    fallback_reason: str | None


@dataclass(frozen=True)
class SectioningResult:
    content_data: list[dict[str, Any]]
    events: tuple[SectionEvent, ...]
    audit_entries: tuple[SectionAuditEntry, ...]
    stats: dict[str, int]


def sectioned_path_for(content_json: str | Path) -> Path:
    path = Path(content_json)
    name = path.name
    for suffix in (
        "_content_list_PATCHED_CAPTIONED.json",
        "_content_list_PATCHED.json",
        "_content_list.json",
        "content_list.json",
    ):
        if not name.endswith(suffix):
            continue
        if suffix == "content_list.json":
            return path.with_name("content_list_SECTIONED.json")
        prefix = name[: -len(suffix)]
        stage_suffix = suffix[len("_content_list") : -len(".json")]
        return path.with_name(f"{prefix}_content_list_SECTIONED{stage_suffix}.json")
    return path.with_name(f"{path.stem}_SECTIONED.json")


def sectioning_audit_path_for(content_json: str | Path) -> Path:
    path = Path(content_json)
    name = path.name
    for suffix in (
        "_content_list_PATCHED_CAPTIONED.json",
        "_content_list_PATCHED.json",
        "_content_list.json",
        "content_list.json",
    ):
        if not name.endswith(suffix):
            continue
        if suffix == "content_list.json":
            return path.with_name("SECTIONING_AUDIT.json")
        prefix = name[: -len(suffix)]
        return path.with_name(f"{prefix}_SECTIONING_AUDIT.json")
    return path.with_name(f"{path.stem}_SECTIONING_AUDIT.json")


def _plain_compare_text(text: Any) -> str:
    return re.sub(r"[ \t\r\n]+", " ", str(text or "").strip())


def display_title(text: Any) -> str:
    cleaned = str(text or "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", cleaned.strip())


def normalize_title(text: Any) -> str:
    return display_title(text).casefold()


def heading_number(text: str) -> str | None:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)\b", text)
    return match.group(1) if match else None


def _block_text(block: dict[str, Any]) -> str:
    value = block.get("text", "")
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def _is_valid_block(block: dict[str, Any]) -> bool:
    return isinstance(block, dict) and block.get("type") not in IGNORE_TYPES


def _is_heading_candidate(block: dict[str, Any]) -> bool:
    return _is_valid_block(block) and block.get("type") == "text" and bool(_block_text(block).strip())


def _bbox_y0(block: dict[str, Any]) -> float | None:
    bbox = block.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        return float(bbox[1])
    except (TypeError, ValueError):
        return None


def _outline_y_candidates(entry: OutlineEntry) -> tuple[float, ...]:
    if entry.dest_y is None or not entry.page_height:
        return ()
    if entry.page_height <= 0:
        return ()
    direct = entry.dest_y / entry.page_height * 1000.0
    flipped = (entry.page_height - entry.dest_y) / entry.page_height * 1000.0
    candidates = []
    for value in (direct, flipped):
        if 0 <= value <= 1000 and all(abs(value - existing) > 0.001 for existing in candidates):
            candidates.append(value)
    return tuple(candidates)


def _y_distance(entry: OutlineEntry, block: dict[str, Any]) -> float | None:
    block_y = _bbox_y0(block)
    candidates = _outline_y_candidates(entry)
    if block_y is None or not candidates:
        return None
    return min(abs(block_y - value) for value in candidates)


def _candidate_too_long(outline_title: str, block_text: str, *, max_chars: int) -> bool:
    outline_len = len(normalize_title(outline_title))
    block_len = len(normalize_title(block_text))
    return block_len > max_chars or block_len > max(outline_len * 3, outline_len + 100)


def _fuzzy_score(outline_title: str, block_text: str, *, max_candidate_chars: int) -> float:
    if _candidate_too_long(outline_title, block_text, max_chars=max_candidate_chars):
        return 0.0
    outline_norm = normalize_title(outline_title)
    block_norm = normalize_title(block_text)
    outline_number = heading_number(outline_norm)
    block_number = heading_number(block_norm)
    if outline_number and block_number and outline_number != block_number:
        return 0.0
    return SequenceMatcher(None, outline_norm, block_norm).ratio()


def read_pdf_outline(pdf_path: str | Path) -> list[OutlineEntry]:
    import fitz

    doc = fitz.open(Path(pdf_path).expanduser())
    raw_toc = doc.get_toc(simple=False)
    page_heights = {idx: float(page.rect.height) for idx, page in enumerate(doc)}
    stack: list[str] = []
    entries: list[OutlineEntry] = []
    for outline_index, row in enumerate(raw_toc):
        if len(row) < 3:
            continue
        level = max(1, int(row[0]))
        title = display_title(row[1])
        if not title:
            continue
        page_idx = max(0, int(row[2]) - 1)
        dest = row[3] if len(row) > 3 and isinstance(row[3], dict) else {}
        if isinstance(dest.get("page"), int) and dest["page"] >= 0:
            page_idx = int(dest["page"])
        dest_y = None
        point = dest.get("to")
        if point is not None and hasattr(point, "y"):
            try:
                dest_y = float(point.y)
            except (TypeError, ValueError):
                dest_y = None

        stack = stack[: level - 1]
        stack.append(title)
        entries.append(
            OutlineEntry(
                outline_index=outline_index,
                level=level,
                title=title,
                page_idx=page_idx,
                section_path=tuple(stack),
                dest_y=dest_y,
                page_height=page_heights.get(page_idx),
            )
        )
    doc.close()
    return entries


def _page_indices(content_data: list[dict[str, Any]]) -> dict[int, list[int]]:
    by_page: dict[int, list[int]] = defaultdict(list)
    for idx, block in enumerate(content_data):
        if not isinstance(block, dict):
            continue
        try:
            page_idx = int(block.get("page_idx", 0))
        except (TypeError, ValueError):
            page_idx = 0
        by_page[page_idx].append(idx)
    return by_page


def _first_valid_block_on_page(content_data: list[dict[str, Any]], block_indices: list[int]) -> int | None:
    for idx in block_indices:
        block = content_data[idx]
        if _is_valid_block(block):
            return idx
    return None


def _fallback_block_for_outline(
    entry: OutlineEntry,
    content_data: list[dict[str, Any]],
    block_indices: list[int],
) -> tuple[int | None, float | None]:
    valid_blocks = [
        (idx, content_data[idx])
        for idx in block_indices
        if _is_valid_block(content_data[idx])
    ]
    candidates = _outline_y_candidates(entry)
    if valid_blocks and candidates:
        y_blocks = [
            (idx, block, y0)
            for idx, block in valid_blocks
            if (y0 := _bbox_y0(block)) is not None
        ]
        if y_blocks:
            after_choices = []
            nearest_choices = []
            for candidate_y in candidates:
                after = [
                    (abs(y0 - candidate_y), y0, idx)
                    for idx, _block, y0 in y_blocks
                    if y0 >= candidate_y
                ]
                if after:
                    after_choices.append(min(after, key=lambda item: (item[0], item[1], item[2])))
                nearest_choices.extend((abs(y0 - candidate_y), y0, idx) for idx, _block, y0 in y_blocks)
            if after_choices:
                distance, _y0, idx = min(after_choices, key=lambda item: (item[0], item[1], item[2]))
                return idx, distance
            if nearest_choices:
                distance, _y0, idx = min(nearest_choices, key=lambda item: (item[0], item[1], item[2]))
                return idx, distance

    return _first_valid_block_on_page(content_data, block_indices), None


def _best_by_y_or_order(
    entry: OutlineEntry,
    candidates: list[tuple[int, dict[str, Any], float]],
) -> tuple[int, dict[str, Any], float, float | None]:
    scored = []
    for idx, block, score in candidates:
        distance = _y_distance(entry, block)
        scored.append((distance is None, distance if distance is not None else float("inf"), idx, block, score))
    _missing_y, distance, idx, block, score = min(scored, key=lambda item: (item[0], item[1], item[2]))
    return idx, block, score, None if distance == float("inf") else distance


def _match_outline_entry(
    entry: OutlineEntry,
    content_data: list[dict[str, Any]],
    block_indices: list[int],
    *,
    fuzzy_threshold: float,
    max_fuzzy_candidate_chars: int,
    y_close_threshold: float,
) -> tuple[SectionEvent | None, SectionAuditEntry]:
    heading_indices = [idx for idx in block_indices if _is_heading_candidate(content_data[idx])]
    title_raw = _plain_compare_text(entry.title)
    title_norm = normalize_title(entry.title)

    raw_matches = [
        (idx, content_data[idx], 1.0)
        for idx in heading_indices
        if _plain_compare_text(_block_text(content_data[idx])) == title_raw
    ]
    if raw_matches:
        idx, block, score, y_distance = _best_by_y_or_order(entry, raw_matches)
        y_close = y_distance is not None and y_distance <= y_close_threshold
        return _event_and_audit(
            entry,
            idx,
            block,
            match_type="exact_y" if y_close else "exact",
            source="pdf_outline_exact_y" if y_close else "pdf_outline_exact",
            confidence=1.0 if y_close else 0.95,
            match_score=score,
            y_distance=y_distance,
            fallback_reason=None,
        )

    normalized_matches = [
        (idx, content_data[idx], 1.0)
        for idx in heading_indices
        if normalize_title(_block_text(content_data[idx])) == title_norm
    ]
    if normalized_matches:
        idx, block, score, y_distance = _best_by_y_or_order(entry, normalized_matches)
        y_close = y_distance is not None and y_distance <= y_close_threshold
        return _event_and_audit(
            entry,
            idx,
            block,
            match_type="normalized_exact_y" if y_close else "normalized_exact",
            source="pdf_outline_normalized_exact_y" if y_close else "pdf_outline_normalized_exact",
            confidence=0.98 if y_close else 0.95,
            match_score=score,
            y_distance=y_distance,
            fallback_reason=None,
        )

    fuzzy_matches = []
    for idx in heading_indices:
        block = content_data[idx]
        score = _fuzzy_score(entry.title, _block_text(block), max_candidate_chars=max_fuzzy_candidate_chars)
        if score >= fuzzy_threshold:
            fuzzy_matches.append((idx, block, score))
    if fuzzy_matches:
        idx, block, score, y_distance = _best_by_y_or_order(entry, fuzzy_matches)
        y_close = y_distance is not None and y_distance <= y_close_threshold
        return _event_and_audit(
            entry,
            idx,
            block,
            match_type="fuzzy_y" if y_close else "fuzzy",
            source="pdf_outline_fuzzy_y" if y_close else "pdf_outline_fuzzy",
            confidence=0.88 if y_close else 0.85,
            match_score=score,
            y_distance=y_distance,
            fallback_reason=None,
        )

    fallback_idx, fallback_y_distance = _fallback_block_for_outline(entry, content_data, block_indices)
    if fallback_idx is not None:
        match_type = "page_fallback_y" if fallback_y_distance is not None else "page_fallback"
        return _event_and_audit(
            entry,
            fallback_idx,
            content_data[fallback_idx],
            match_type=match_type,
            source=f"pdf_outline_{match_type}",
            confidence=0.78 if fallback_y_distance is not None else 0.75,
            match_score=0.0,
            y_distance=fallback_y_distance,
            fallback_reason="no heading text block matched outline title on target page",
        )

    audit = SectionAuditEntry(
        outline_index=entry.outline_index,
        outline_title=entry.title,
        outline_level=entry.level,
        outline_page=entry.page_idx + 1,
        section_path=entry.section_path,
        match_type="unmatched",
        matched_block_idx=None,
        matched_text="",
        confidence=0.0,
        match_score=0.0,
        y_distance=None,
        fallback_reason="no valid content block found on target page",
    )
    return None, audit


def _event_and_audit(
    entry: OutlineEntry,
    block_idx: int,
    block: dict[str, Any],
    *,
    match_type: str,
    source: str,
    confidence: float,
    match_score: float,
    y_distance: float | None,
    fallback_reason: str | None,
) -> tuple[SectionEvent, SectionAuditEntry]:
    event = SectionEvent(
        outline_index=entry.outline_index,
        target_block_idx=block_idx,
        title=entry.title,
        level=entry.level,
        section_path=entry.section_path,
        section_source=source,
        section_confidence=confidence,
        section_match_score=match_score,
    )
    audit = SectionAuditEntry(
        outline_index=entry.outline_index,
        outline_title=entry.title,
        outline_level=entry.level,
        outline_page=entry.page_idx + 1,
        section_path=entry.section_path,
        match_type=match_type,
        matched_block_idx=block_idx,
        matched_text=_block_text(block),
        confidence=confidence,
        match_score=match_score,
        y_distance=None if y_distance is None else round(y_distance, 3),
        fallback_reason=fallback_reason,
    )
    return event, audit


def section_content(
    content_data: list[dict[str, Any]],
    outline_entries: list[OutlineEntry],
    *,
    fuzzy_threshold: float = 0.92,
    max_fuzzy_candidate_chars: int = 260,
    y_close_threshold: float = 50.0,
) -> SectioningResult:
    indices_by_page = _page_indices(content_data)
    events: list[SectionEvent] = []
    audit_entries: list[SectionAuditEntry] = []
    for entry in outline_entries:
        event, audit = _match_outline_entry(
            entry,
            content_data,
            indices_by_page.get(entry.page_idx, []),
            fuzzy_threshold=fuzzy_threshold,
            max_fuzzy_candidate_chars=max_fuzzy_candidate_chars,
            y_close_threshold=y_close_threshold,
        )
        if event is not None:
            events.append(event)
        audit_entries.append(audit)

    events = sorted(events, key=lambda item: (item.target_block_idx, item.outline_index))
    events_by_block: dict[int, list[SectionEvent]] = defaultdict(list)
    for event in events:
        events_by_block[event.target_block_idx].append(event)

    annotated: list[dict[str, Any]] = []
    current_event: SectionEvent | None = None
    for block_idx, block in enumerate(content_data):
        for event in events_by_block.get(block_idx, []):
            current_event = event

        if not isinstance(block, dict):
            annotated.append(block)
            continue

        new_block = dict(block)
        if current_event is not None:
            new_block.update(
                {
                    "section_path": list(current_event.section_path),
                    "section_title": current_event.title,
                    "section_level": current_event.level,
                    "section_source": current_event.section_source,
                    "section_confidence": current_event.section_confidence,
                    "section_match_score": current_event.section_match_score,
                    "section_outline_index": current_event.outline_index,
                }
            )
        else:
            for key in SECTION_FIELD_KEYS:
                new_block.pop(key, None)
        annotated.append(new_block)

    table_continuations = build_table_continuation_map(annotated)
    for master_idx, continuation_indices in table_continuations.items():
        if master_idx >= len(annotated) or not isinstance(annotated[master_idx], dict):
            continue
        master = annotated[master_idx]
        master[TABLE_CONTINUATION_INDICES_KEY] = list(continuation_indices)
        for continuation_idx in continuation_indices:
            if continuation_idx >= len(annotated) or not isinstance(annotated[continuation_idx], dict):
                continue
            continuation = annotated[continuation_idx]
            for key in SECTION_FIELD_KEYS:
                if key in master:
                    continuation[key] = master[key]
                else:
                    continuation.pop(key, None)
            continuation[TABLE_CONTINUATION_MASTER_IDX_KEY] = master_idx

    blocks_annotated = sum(
        1 for block in annotated if isinstance(block, dict) and bool(block.get("section_path"))
    )

    counts = Counter(entry.match_type for entry in audit_entries)
    stats = {
        "outline_entry_count": len(outline_entries),
        "section_event_count": len(events),
        "blocks_total": len(content_data),
        "blocks_annotated": blocks_annotated,
        "exact_matches": counts.get("exact", 0),
        "exact_y_matches": counts.get("exact_y", 0),
        "normalized_exact_matches": counts.get("normalized_exact", 0),
        "normalized_exact_y_matches": counts.get("normalized_exact_y", 0),
        "fuzzy_matches": counts.get("fuzzy", 0),
        "fuzzy_y_matches": counts.get("fuzzy_y", 0),
        "page_fallbacks": counts.get("page_fallback", 0) + counts.get("page_fallback_y", 0),
        "page_fallback_y": counts.get("page_fallback_y", 0),
        "unmatched": counts.get("unmatched", 0),
        "table_continuation_groups": len(table_continuations),
        "table_continuation_blocks": sum(len(indices) for indices in table_continuations.values()),
    }
    return SectioningResult(
        content_data=annotated,
        events=tuple(events),
        audit_entries=tuple(audit_entries),
        stats=stats,
    )


def add_source_metadata(
    content_data: list[dict[str, Any]],
    *,
    source_name: str,
) -> list[dict[str, Any]]:
    source_fields = source_payload_fields(source_name)
    annotated: list[dict[str, Any]] = []
    for block in content_data:
        if not isinstance(block, dict):
            annotated.append(block)
            continue
        new_block = dict(block)
        section_path = new_block.get("section_path", [])
        if not isinstance(section_path, list):
            section_path = []
        new_block.update(source_fields)
        new_block["breadcrumb"] = source_breadcrumb(source_fields["source_relpath"], section_path)
        annotated.append(new_block)
    return annotated


def write_sectioning_audit(
    *,
    audit_path: str | Path,
    result: SectioningResult,
    source_pdf: str | Path,
    input_json: str | Path,
    output_json: str | Path,
    source_name: str,
) -> None:
    payload = {
        "source_pdf": str(source_pdf),
        "source_name": source_name,
        "input_json": str(input_json),
        "output_json": str(output_json),
        "stats": result.stats,
        "entries": [asdict(entry) for entry in result.audit_entries],
    }
    output = Path(audit_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_sectioned_json(
    *,
    input_json: str | Path,
    input_pdf: str | Path,
    output_json: str | Path,
    audit_json: str | Path,
    fuzzy_threshold: float = 0.92,
    max_fuzzy_candidate_chars: int = 260,
    y_close_threshold: float = 50.0,
    source_name: str | None = None,
) -> SectioningResult:
    input_path = Path(input_json).expanduser()
    output_path = Path(output_json).expanduser()
    content_data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(content_data, list):
        raise ValueError(f"Expected a list in content JSON: {input_path}")
    outline_entries = read_pdf_outline(input_pdf)
    resolved_source_name = source_name or source_name_for_pdf(input_pdf)
    result = section_content(
        content_data,
        outline_entries,
        fuzzy_threshold=fuzzy_threshold,
        max_fuzzy_candidate_chars=max_fuzzy_candidate_chars,
        y_close_threshold=y_close_threshold,
    )
    source_annotated = add_source_metadata(result.content_data, source_name=resolved_source_name)
    result = SectioningResult(
        content_data=source_annotated,
        events=result.events,
        audit_entries=result.audit_entries,
        stats={
            **result.stats,
            "source_metadata_blocks": sum(1 for block in source_annotated if isinstance(block, dict)),
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.content_data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sectioning_audit(
        audit_path=audit_json,
        result=result,
        source_pdf=input_pdf,
        source_name=resolved_source_name,
        input_json=input_path,
        output_json=output_path,
    )
    return result


def _default_input_json(config: AppConfig) -> Path:
    if config.paths.content_json.exists():
        return config.paths.content_json
    discovered = find_content_json(config, source_pdf=config.paths.source_pdf)
    if discovered is not None:
        return discovered
    return config.paths.content_json


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    default_input_json = _default_input_json(config)

    parser = argparse.ArgumentParser(description="Annotate MinerU content_list blocks with PDF outline sections.")
    parser.add_argument("--input-json", default=str(default_input_json), help="Input MinerU content_list JSON.")
    parser.add_argument(
        "--input-pdf",
        default=str(config.paths.source_pdf),
        help="Original source PDF with outline/bookmarks.",
    )
    parser.add_argument("--output-json", default=None, help="Output SECTIONED content_list JSON.")
    parser.add_argument("--audit-json", default=None, help="Output SECTIONING_AUDIT JSON.")
    parser.add_argument(
        "--source-name",
        default=None,
        help="Source relative path stored in sectioned blocks; defaults to sourcepdfs-relative path when available.",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Directory treated as the source-relative root, e.g. /root/pdfs -> DSS/manual.pdf.",
    )
    parser.add_argument("--fuzzy-threshold", type=float, default=0.92)
    parser.add_argument("--max-fuzzy-candidate-chars", type=int, default=260)
    parser.add_argument("--y-close-threshold", type=float, default=50.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    input_json = Path(args.input_json).expanduser()
    if args.output_json:
        output_json = Path(args.output_json).expanduser()
    elif input_json == config.paths.content_json:
        output_json = config.paths.sectioned_json
    else:
        output_json = sectioned_path_for(input_json)
    audit_json = (
        Path(args.audit_json).expanduser()
        if args.audit_json
        else output_json.parent / sectioning_audit_path_for(input_json).name
    )
    input_pdf = Path(args.input_pdf).expanduser()
    resolved_source_name = args.source_name or source_name_for_pdf(
        input_pdf,
        configured_source_pdf=config.paths.source_pdf,
        configured_source_name=config.paths.source_name,
        source_root=args.source_root
        or config.paths.source_root
        or source_root_from_input_path(config.mineru.input_path),
    )

    if args.dry_run:
        print("Sectioning inputs:")
        print(f"  input_json: {input_json}")
        print(f"  input_pdf: {input_pdf}")
        print(f"  source_name: {resolved_source_name}")
        print(f"  output_json: {output_json}")
        print(f"  audit_json: {audit_json}")
        print(f"  fuzzy_threshold: {args.fuzzy_threshold}")
        print(f"  y_close_threshold: {args.y_close_threshold}")
        return

    result = write_sectioned_json(
        input_json=input_json,
        input_pdf=input_pdf,
        output_json=output_json,
        audit_json=audit_json,
        fuzzy_threshold=args.fuzzy_threshold,
        max_fuzzy_candidate_chars=args.max_fuzzy_candidate_chars,
        y_close_threshold=args.y_close_threshold,
        source_name=resolved_source_name,
    )
    print(f"Wrote sectioned content JSON at {output_json}")
    print(f"Wrote sectioning audit JSON at {audit_json}")
    print(f"  outline entries: {result.stats['outline_entry_count']}")
    print(f"  section events: {result.stats['section_event_count']}")
    print(f"  blocks annotated: {result.stats['blocks_annotated']}/{result.stats['blocks_total']}")
    print(
        "  matches: "
        f"exact={result.stats['exact_matches'] + result.stats['exact_y_matches']} "
        f"normalized_exact={result.stats['normalized_exact_matches'] + result.stats['normalized_exact_y_matches']} "
        f"fuzzy={result.stats['fuzzy_matches'] + result.stats['fuzzy_y_matches']} "
        f"fallback={result.stats['page_fallbacks']} "
        f"unmatched={result.stats['unmatched']}"
    )


if __name__ == "__main__":
    main()
