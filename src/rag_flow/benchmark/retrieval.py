from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_flow.benchmark.evidence_remap import build_evidence_anchors, remap_query_set_to_chunks
from rag_flow.config import AppConfig


DEFAULT_OUTPUT_DIR = Path("thesis/07-retrieval-serving/data/benchmark-runs")
DEFAULT_PILOT_QUERY_SET = Path("thesis/07-retrieval-serving/data/pilot_query_set.jsonl")
DEFAULT_FULL_QUERY_SET = Path("thesis/07-retrieval-serving/data/full_query_set.jsonl")
DEFAULT_RECALL_KS = (1, 3, 5, 10)
FULL_QUERY_VISUAL_PAGE_RATIO = 0.15

PILOT_QUERIES: tuple[dict[str, Any], ...] = (
    {
        "query_id": "pilot-001",
        "query": "What is DSS Professional used for in centralized security management?",
        "query_type": "text",
        "gold_page_indices": [10],
        "gold_page_numbers": [11],
        "gold_chunk_ids": ["technical-manual-chunk-00007"],
        "evidence_type": "text",
        "notes": "Introduction section describes centralized video monitoring, access control, intercom, alarms, and POS search.",
    },
    {
        "query_id": "pilot-002",
        "query": "Which ports should be considered when configuring router mapping for LAN or WAN?",
        "query_type": "text",
        "gold_page_indices": [35],
        "gold_page_numbers": [36],
        "gold_chunk_ids": ["technical-manual-chunk-00035"],
        "evidence_type": "text_icon",
        "notes": "Router configuration points to Appendix 1 and says WAN ports should be consistent with LAN ports.",
    },
    {
        "query_id": "pilot-003",
        "query": "How do I add video retrieval plans one by one from the DSS Client?",
        "query_type": "operation",
        "gold_page_indices": [56, 57, 58],
        "gold_page_numbers": [57, 58, 59],
        "gold_chunk_ids": ["technical-manual-chunk-00067"],
        "evidence_type": "text_icon",
        "notes": "Procedure uses App Config/Basic Config device path and retrieval plan steps.",
    },
    {
        "query_id": "pilot-004",
        "query": "How do I add file retrieval plans in batches?",
        "query_type": "operation",
        "gold_page_indices": [59, 60],
        "gold_page_numbers": [60, 61],
        "gold_chunk_ids": ["technical-manual-chunk-00070"],
        "evidence_type": "text_icon",
        "notes": "Batch file retrieval plan workflow.",
    },
    {
        "query_id": "pilot-005",
        "query": "What happens when storage space runs out for platform videos?",
        "query_type": "text",
        "gold_page_indices": [61, 62],
        "gold_page_numbers": [62, 63],
        "gold_chunk_ids": ["technical-manual-chunk-00072"],
        "evidence_type": "text",
        "notes": "Video retention period section says new recorded videos cover oldest videos automatically.",
    },
    {
        "query_id": "pilot-006",
        "query": "How can alarm video pre-recording be configured for a device?",
        "query_type": "operation",
        "gold_page_indices": [92],
        "gold_page_numbers": [93],
        "gold_chunk_ids": ["technical-manual-chunk-00105"],
        "evidence_type": "text",
        "notes": "Alarm video pre-recording mode and linked alarm behavior.",
    },
    {
        "query_id": "pilot-007",
        "query": "What is a face arming list used for?",
        "query_type": "text",
        "gold_page_indices": [124],
        "gold_page_numbers": [125],
        "gold_chunk_ids": ["technical-manual-chunk-00140"],
        "evidence_type": "text",
        "notes": "Face arming list is sent to devices for face recognition and alarms.",
    },
    {
        "query_id": "pilot-008",
        "query": "Which credential security options can be enabled for access control?",
        "query_type": "image_ui",
        "gold_page_indices": [151],
        "gold_page_numbers": [152],
        "gold_chunk_ids": ["technical-manual-chunk-00175"],
        "evidence_type": "text_icon",
        "notes": "DESFire Card, Block NFC Cards, and fingerprint uniqueness verification icons.",
    },
    {
        "query_id": "pilot-009",
        "query": "What information does a parking lot configuration include?",
        "query_type": "text",
        "gold_page_indices": [172],
        "gold_page_numbers": [173],
        "gold_chunk_ids": ["technical-manual-chunk-00210"],
        "evidence_type": "text",
        "notes": "Parking spaces, entrances/exits, barrier control rules, ANPR camera, and VTO.",
    },
    {
        "query_id": "pilot-010",
        "query": "Where is the AR system name displayed after it is configured?",
        "query_type": "operation",
        "gold_page_indices": [202, 203],
        "gold_page_numbers": [203, 204],
        "gold_chunk_ids": ["technical-manual-chunk-00245"],
        "evidence_type": "text_icon",
        "notes": "AR system name appears on the AR homepage in the monitoring center.",
    },
    {
        "query_id": "pilot-011",
        "query": "What are private view groups used for in video monitoring?",
        "query_type": "text",
        "gold_page_indices": [247],
        "gold_page_numbers": [248],
        "gold_chunk_ids": ["technical-manual-chunk-00280"],
        "evidence_type": "text",
        "notes": "Private view groups organize private views and allow one level of sub groups.",
    },
    {
        "query_id": "pilot-012",
        "query": "During playback, how can I search for a selected target in DeepXplore?",
        "query_type": "mixed",
        "gold_page_indices": [271],
        "gold_page_numbers": [272],
        "gold_chunk_ids": ["technical-manual-chunk-00315"],
        "evidence_type": "text",
        "notes": "Searching for Targets section mentions manual selection or AcuPick recognized targets.",
    },
    {
        "query_id": "pilot-013",
        "query": "Why would I use temporary disarming in Event Center?",
        "query_type": "operation",
        "gold_page_indices": [304, 305],
        "gold_page_numbers": [305, 306],
        "gold_chunk_ids": ["technical-manual-chunk-00350"],
        "evidence_type": "text",
        "notes": "Temporary disarming avoids unnecessary alarms, for example during a celebration activity.",
    },
    {
        "query_id": "pilot-014",
        "query": "What is the visitor application process after an appointment?",
        "query_type": "operation",
        "gold_page_indices": [332],
        "gold_page_numbers": [333],
        "gold_chunk_ids": ["technical-manual-chunk-00385"],
        "evidence_type": "text",
        "notes": "Visitor makes appointment, confirms information through check-in, then accesses quickly.",
    },
    {
        "query_id": "pilot-015",
        "query": "What statistics can Metadata Analysis show and export?",
        "query_type": "table",
        "gold_page_indices": [364, 365],
        "gold_page_numbers": [365, 366],
        "gold_chunk_ids": ["technical-manual-chunk-00420"],
        "evidence_type": "text_table",
        "notes": "Metadata analysis statistics include faces, human body, motor vehicles, and non-motor vehicles.",
    },
    {
        "query_id": "pilot-016",
        "query": "How long is the video linked to a vehicle snapshot record?",
        "query_type": "text",
        "gold_page_indices": [385],
        "gold_page_numbers": [386],
        "gold_chunk_ids": ["technical-manual-chunk-00455"],
        "evidence_type": "text",
        "notes": "Each video is 20 seconds long, with 10 seconds before and after capture.",
    },
    {
        "query_id": "pilot-017",
        "query": "Why do I need to activate a license before using features or channels?",
        "query_type": "text",
        "gold_page_indices": [406, 407],
        "gold_page_numbers": [407, 408],
        "gold_chunk_ids": ["technical-manual-chunk-00490"],
        "evidence_type": "text_icon",
        "notes": "License activation unlocks desired features or number of channels.",
    },
    {
        "query_id": "pilot-018",
        "query": "What are local default settings used for after restoring local settings?",
        "query_type": "text",
        "gold_page_indices": [435],
        "gold_page_numbers": [436],
        "gold_chunk_ids": ["technical-manual-chunk-00525"],
        "evidence_type": "text",
        "notes": "Configured default settings are applied after Local Settings are restored to default.",
    },
    {
        "query_id": "pilot-019",
        "query": "What kinds of devices can be added when managing devices?",
        "query_type": "table",
        "gold_page_indices": [40],
        "gold_page_numbers": [41],
        "gold_chunk_ids": ["technical-manual-chunk-00046"],
        "evidence_type": "text_table",
        "notes": "Device examples include encoder, decoder, ANPR, access control, displays, emergency assistance, alarm box, radar, and intercom.",
    },
    {
        "query_id": "pilot-020",
        "query": "How do I configure linkage between radar and PTZ cameras?",
        "query_type": "mixed",
        "gold_page_indices": [100, 101],
        "gold_page_numbers": [101, 102],
        "gold_chunk_ids": ["technical-manual-chunk-00115"],
        "evidence_type": "text_image",
        "notes": "Radar-PTZ linkage section describes configuring linkage when an alarm is triggered.",
    },
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]], *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_pilot_query_set(path: Path = DEFAULT_PILOT_QUERY_SET, *, force: bool = False) -> Path:
    _write_jsonl(path, PILOT_QUERIES, force=force)
    return path


def _strip_section_number(title: str) -> str:
    return re.sub(r"^\s*\d+(?:\.\d+)*\s+", "", title).strip()


def _clean_note(text: str, *, limit: int = 220) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[Icon:[^\]]+\]", " ", text)
    text = re.sub(
        r"\[(?:Breadcrumb|Section|Table|Table caption|Table footnote|Image with illustration|Image caption|Image description|Image VLM description|Image footnote|Image answering policy|Footnote):[^\]]*\]",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].strip()
    return clipped + "."


def _load_json_list(path: Path, *, label: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of {label}")
    rows = []
    for item in data:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _pdf_page_excerpt(pdf_path: Path | None, page_idx: int, *, limit: int = 260) -> str:
    if pdf_path is None:
        return ""
    try:
        import fitz
    except ImportError:
        return ""
    try:
        with fitz.open(pdf_path) as doc:
            if page_idx < 0 or page_idx >= len(doc):
                return ""
            text = doc[page_idx].get_text("text")
    except Exception:
        return ""
    return _clean_note(text, limit=limit)


def _extract_label(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for item in value.values():
            text = _first_text(item)
            if text:
                return text
        return ""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _question_for_action(title: str) -> str:
    label = _strip_section_number(title)
    if not label:
        return "How is this operation performed in the manual?"
    replacements = (
        ("Configuring ", "configure "),
        ("Adding ", "add "),
        ("Creating ", "create "),
        ("Managing ", "manage "),
        ("Viewing ", "view "),
        ("Searching ", "search "),
        ("Setting ", "set "),
        ("Editing ", "edit "),
        ("Importing ", "import "),
        ("Exporting ", "export "),
        ("Enabling ", "enable "),
        ("Disabling ", "disable "),
        ("Installing ", "install "),
        ("Logging ", "log "),
    )
    for prefix, verb in replacements:
        if label.startswith(prefix):
            return f"How do I {verb}{label[len(prefix):]}?"
    return f"How do I use {label}?"


def _query_type_for_chunk(text: str) -> str:
    has_table = "[Table caption:" in text or "[Table:" in text or "<table" in text
    has_image = (
        "[Image caption:" in text
        or "[Image VLM description:" in text
        or "[Image description:" in text
        or "[Image with illustration" in text
    )
    has_steps = bool(re.search(r"\bStep\s+\d+", text))
    if has_table:
        return "table"
    if has_image and has_steps:
        return "mixed"
    if has_image:
        return "image_ui"
    if has_steps:
        return "operation"
    return "text"


def _question_for_chunk(chunk: dict[str, Any]) -> str:
    text = str(chunk.get("chunk_content", ""))
    metadata = dict(chunk.get("metadata", {}))
    title = str(metadata.get("section_title") or "")
    query_type = _query_type_for_chunk(text)
    if query_type == "table":
        caption = _extract_label(text, r"\[Table caption:\s*([^\]]+)\]") or _extract_label(
            text,
            r"\[Table:\s*([^\]]+)\]",
        )
        subject = caption or _strip_section_number(title) or "this table"
        return f"What parameters or information are described in {subject}?"
    if query_type == "image_ui":
        caption = _extract_label(text, r"\[Image caption:\s*([^\]]+)\]") or _extract_label(
            text,
            r"\[Image with illustration:\s*([^\]]+)\]",
        )
        subject = caption or _strip_section_number(title) or "this interface"
        return f"What does {subject} show in the manual?"
    if query_type == "mixed":
        return f"What operation and visual information are described in {_strip_section_number(title) or 'this section'}?"
    if query_type == "operation":
        return _question_for_action(title)
    subject = _strip_section_number(title)
    if subject:
        return f"What does the manual say about {subject}?"
    return "What information is provided in this section of the manual?"


def _query_row_from_chunk(chunk: dict[str, Any], query_id: str) -> dict[str, Any]:
    text = str(chunk.get("chunk_content", ""))
    metadata = dict(chunk.get("metadata", {}))
    pages = _list_ints(metadata.get("page_indices"))
    if not pages:
        pages = _list_ints(metadata.get("page_idx"))
    chunk_id = str(metadata.get("chunk_id") or query_id)
    section_path = metadata.get("section_path", [])
    if not isinstance(section_path, list):
        section_path = []
    query_type = _query_type_for_chunk(text)
    evidence_type = {
        "table": "text_table",
        "image_ui": "text_image",
        "mixed": "text_image",
        "operation": "text_icon" if "[Icon:" in text else "text",
    }.get(query_type, "text_icon" if "[Icon:" in text else "text")
    return {
        "query_id": query_id,
        "query": _question_for_chunk(chunk),
        "query_type": query_type,
        "gold_page_indices": pages,
        "gold_page_numbers": [page + 1 for page in pages],
        "gold_chunk_ids": [chunk_id],
        "gold_section_path": [str(item) for item in section_path if str(item).strip()],
        "evidence_type": evidence_type,
        "evidence_source": ["chunked_json"],
        "review_status": "chunk_text_grounded",
        "requires_pdf_review": False,
        "notes": _clean_note(text),
    }


def _eligible_query_chunk(chunk: dict[str, Any]) -> bool:
    text = str(chunk.get("chunk_content", ""))
    metadata = dict(chunk.get("metadata", {}))
    query_type = _query_type_for_chunk(text)
    if query_type == "text" and len(_clean_note(text, limit=80)) < 40:
        return False
    pages = _list_ints(metadata.get("page_indices") or metadata.get("page_idx"))
    if pages and min(pages) < 10:
        return False
    title = str(metadata.get("section_title") or "")
    if title.lower() in {"foreword", "contents"}:
        return False
    if not metadata.get("chunk_id"):
        return False
    return True


def generate_full_query_set_from_chunks(
    chunks_path: Path,
    *,
    target_size: int = 100,
) -> list[dict[str, Any]]:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError(f"{chunks_path} must contain a list of chunks")

    pools: dict[str, list[dict[str, Any]]] = {key: [] for key in ("text", "operation", "table", "image_ui", "mixed")}
    seen_sections: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict) or not _eligible_query_chunk(chunk):
            continue
        metadata = dict(chunk.get("metadata", {}))
        section_key = str(metadata.get("section_title") or metadata.get("chunk_id"))
        query_type = _query_type_for_chunk(str(chunk.get("chunk_content", "")))
        # Keep broad coverage first; repeated sections can still appear later
        # if a pool otherwise runs short.
        if section_key not in seen_sections:
            pools[query_type].append(chunk)
            seen_sections.add(section_key)
    if sum(len(pool) for pool in pools.values()) < target_size:
        for chunk in chunks:
            if not isinstance(chunk, dict) or not _eligible_query_chunk(chunk):
                continue
            query_type = _query_type_for_chunk(str(chunk.get("chunk_content", "")))
            if chunk not in pools[query_type]:
                pools[query_type].append(chunk)

    quotas = {
        "text": round(target_size * 0.30),
        "operation": round(target_size * 0.25),
        "table": round(target_size * 0.20),
        "image_ui": round(target_size * 0.15),
    }
    quotas["mixed"] = target_size - sum(quotas.values())

    selected: list[dict[str, Any]] = []
    used_chunk_ids: set[str] = set()

    def take_from_pool(query_type: str, count: int) -> None:
        for chunk in pools[query_type]:
            chunk_id = str(dict(chunk.get("metadata", {})).get("chunk_id", ""))
            if chunk_id in used_chunk_ids:
                continue
            selected.append(chunk)
            used_chunk_ids.add(chunk_id)
            if sum(1 for item in selected if _query_type_for_chunk(str(item.get("chunk_content", ""))) == query_type) >= count:
                break

    for query_type, count in quotas.items():
        take_from_pool(query_type, count)

    if len(selected) < target_size:
        for query_type in ("operation", "table", "image_ui", "mixed", "text"):
            for chunk in pools[query_type]:
                chunk_id = str(dict(chunk.get("metadata", {})).get("chunk_id", ""))
                if chunk_id in used_chunk_ids:
                    continue
                selected.append(chunk)
                used_chunk_ids.add(chunk_id)
                if len(selected) >= target_size:
                    break
            if len(selected) >= target_size:
                break

    rows = [_query_row_from_chunk(chunk, f"full-{idx:03d}") for idx, chunk in enumerate(selected[:target_size], start=1)]
    if len(rows) < target_size:
        raise ValueError(f"Only generated {len(rows)} query rows from {chunks_path}; requested {target_size}")
    return rows


def _chunks_by_page(chunks: Sequence[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_page: dict[int, list[dict[str, Any]]] = {}
    for chunk in chunks:
        metadata = dict(chunk.get("metadata", {}))
        pages = _list_ints(metadata.get("page_indices") or metadata.get("page_idx"))
        for page in pages:
            by_page.setdefault(page, []).append(chunk)
    return by_page


def _image_query_subject(block: dict[str, Any]) -> str:
    caption = _first_text(block.get("image_caption"))
    if caption:
        return re.sub(r"\s+", " ", caption)
    section_title = _first_text(block.get("section_title"))
    if section_title:
        return _strip_section_number(section_title) or section_title
    try:
        return f"page {int(block.get('page_idx')) + 1}"
    except (TypeError, ValueError):
        return "this page"


def _visual_page_question(block: dict[str, Any]) -> str:
    subject = _image_query_subject(block)
    if subject.lower().startswith("figure"):
        return f"What does {subject} show?"
    return f"What visual information is shown in {subject}?"


def _visual_page_candidate_score(block: dict[str, Any], image_count_by_page: dict[int, int]) -> tuple[int, int, int]:
    page_idx = _list_ints(block.get("page_idx"))
    page = page_idx[0] if page_idx else -1
    has_description = bool(str(block.get("image_description_vlm") or "").strip())
    has_caption = bool(str(block.get("image_caption") or "").strip())
    return (
        image_count_by_page.get(page, 0),
        int(has_description) + int(has_caption),
        page,
    )


def generate_visual_page_queries(
    *,
    captioned_json_path: Path,
    chunks_path: Path,
    pdf_path: Path | None = None,
    target_size: int = 15,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    blocks = _load_json_list(captioned_json_path, label="captioned MinerU blocks")
    chunks = _load_json_list(chunks_path, label="chunks")
    chunks_by_page = _chunks_by_page(chunks)
    image_blocks = [
        block
        for block in blocks
        if block.get("type") == "image" and _list_ints(block.get("page_idx")) and _list_ints(block.get("page_idx"))[0] >= 10
    ]
    image_count_by_page: dict[int, int] = {}
    for block in image_blocks:
        page = _list_ints(block.get("page_idx"))[0]
        image_count_by_page[page] = image_count_by_page.get(page, 0) + 1

    image_blocks.sort(key=lambda block: _visual_page_candidate_score(block, image_count_by_page), reverse=True)

    def build_row(block: dict[str, Any], row_index: int) -> dict[str, Any]:
        page_idx = _list_ints(block.get("page_idx"))[0]
        subject = _image_query_subject(block)
        section_path = block.get("section_path", [])
        if not isinstance(section_path, list):
            section_path = []
        bboxes_by_page = {}
        bbox = block.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            bboxes_by_page[str(page_idx)] = [bbox]
        row = {
            "query_id": f"full-visual-{start_index + row_index:03d}",
            "query": _visual_page_question(block),
            "query_type": "visual_page",
            "gold_page_indices": [page_idx],
            "gold_page_numbers": [page_idx + 1],
            # Intentionally empty: visual-page queries evaluate whether the
            # system can recover the right original PDF page even when a
            # precise chunk id is not the gold label.
            "gold_chunk_ids": [],
            "gold_section_path": [str(item) for item in section_path if str(item).strip()],
            "evidence_type": "page_visual",
            "evidence_source": ["origin_pdf", "captioned_json"],
            "evidence_bboxes_by_page": bboxes_by_page,
            "review_status": "pdf_page_grounded",
            "requires_pdf_review": True,
            "candidate_chunk_ids_on_page": [
                str(dict(chunk.get("metadata", {})).get("chunk_id"))
                for chunk in chunks_by_page.get(page_idx, [])
                if dict(chunk.get("metadata", {})).get("chunk_id")
            ],
            "notes": _clean_note(
                _first_text(block.get("image_description_vlm") or block.get("image_caption"))
                or _pdf_page_excerpt(pdf_path, page_idx)
            ),
        }
        return row

    rows: list[dict[str, Any]] = []
    selected_block_ids: set[int] = set()
    used_pages: set[int] = set()
    used_subjects: set[tuple[int, str]] = set()

    for allow_duplicate_page in (False, True):
        for block in image_blocks:
            block_id = id(block)
            if block_id in selected_block_ids:
                continue
            page_idx = _list_ints(block.get("page_idx"))[0]
            subject_key = _image_query_subject(block).lower()
            if not allow_duplicate_page and page_idx in used_pages:
                continue
            if (page_idx, subject_key) in used_subjects:
                continue
            rows.append(build_row(block, len(rows)))
            selected_block_ids.add(block_id)
            used_pages.add(page_idx)
            used_subjects.add((page_idx, subject_key))
            if len(rows) >= target_size:
                return rows
    return rows


def _renumber_queries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    renumbered = []
    for index, row in enumerate(rows, start=1):
        updated = dict(row)
        updated["query_id"] = f"full-{index:03d}"
        renumbered.append(updated)
    return renumbered


def generate_pdf_grounded_full_query_set(
    chunks_path: Path,
    *,
    captioned_json_path: Path | None = None,
    pdf_path: Path | None = None,
    target_size: int = 100,
    visual_page_count: int | None = None,
) -> list[dict[str, Any]]:
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if visual_page_count is None:
        visual_page_count = round(target_size * FULL_QUERY_VISUAL_PAGE_RATIO)
    visual_page_count = max(0, min(target_size, visual_page_count))

    visual_rows: list[dict[str, Any]] = []
    if captioned_json_path is not None and captioned_json_path.exists() and visual_page_count:
        visual_rows = generate_visual_page_queries(
            captioned_json_path=captioned_json_path,
            chunks_path=chunks_path,
            pdf_path=pdf_path if pdf_path and pdf_path.exists() else None,
            target_size=visual_page_count,
            start_index=1,
        )

    chunk_target = target_size - len(visual_rows)
    chunk_rows = generate_full_query_set_from_chunks(chunks_path, target_size=chunk_target)
    return _renumber_queries([*chunk_rows, *visual_rows])


def write_full_query_set(
    chunks_path: Path,
    path: Path = DEFAULT_FULL_QUERY_SET,
    *,
    captioned_json_path: Path | None = None,
    pdf_path: Path | None = None,
    target_size: int = 100,
    visual_page_count: int | None = None,
    force: bool = False,
) -> Path:
    rows = generate_pdf_grounded_full_query_set(
        chunks_path,
        captioned_json_path=captioned_json_path,
        pdf_path=pdf_path,
        target_size=target_size,
        visual_page_count=visual_page_count,
    )
    _write_jsonl(path, rows, force=force)
    return path


def _list_ints(value: Any) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return sorted(set(result))


def _list_strings(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return sorted({str(item) for item in values if str(item).strip()})


def _gold_page_indices(query: dict[str, Any]) -> list[int]:
    pages = _list_ints(query.get("gold_page_indices", query.get("expected_page_indices")))
    if pages:
        return pages
    page_numbers = _list_ints(query.get("gold_page_numbers", query.get("expected_pages")))
    return sorted({page - 1 for page in page_numbers if page > 0})


def _gold_chunk_ids(query: dict[str, Any]) -> list[str]:
    return _list_strings(query.get("gold_chunk_ids", query.get("expected_chunk_ids")))


def _strict_gold_chunk_ids(query: dict[str, Any]) -> list[str]:
    return _list_strings(query.get("strict_gold_chunk_ids"))


def _primary_gold_chunk_id(query: dict[str, Any]) -> str:
    primary = str(query.get("primary_gold_chunk_id") or "").strip()
    if primary:
        return primary
    strict = _strict_gold_chunk_ids(query)
    if strict:
        return strict[0]
    gold = _gold_chunk_ids(query)
    return gold[0] if gold else ""


def _hit_chunk_id(hit: dict[str, Any]) -> str:
    return str(hit.get("chunk_id") or hit.get("id") or "")


def _first_correct_rank(query: dict[str, Any], hits: Sequence[dict[str, Any]]) -> int | None:
    gold_chunks = set(_gold_chunk_ids(query))
    gold_pages = set(_gold_page_indices(query))
    for rank, hit in enumerate(hits, start=1):
        chunk_id = _hit_chunk_id(hit)
        if gold_chunks and chunk_id in gold_chunks:
            return rank
        try:
            page_idx = int(hit.get("page_idx"))
        except (TypeError, ValueError):
            page_idx = None
        if not gold_chunks and page_idx in gold_pages:
            return rank
    return None


def _first_chunk_rank(chunk_ids: set[str], hits: Sequence[dict[str, Any]]) -> int | None:
    if not chunk_ids:
        return None
    for rank, hit in enumerate(hits, start=1):
        if _hit_chunk_id(hit) in chunk_ids:
            return rank
    return None


def score_query_result(
    query: dict[str, Any],
    response: dict[str, Any],
    *,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
) -> dict[str, Any]:
    hits = response.get("all_hits", [])
    if not isinstance(hits, list):
        hits = []
    gold_pages = set(_gold_page_indices(query))
    gold_chunks = set(_gold_chunk_ids(query))
    strict_chunks = set(_strict_gold_chunk_ids(query))
    primary_chunk = _primary_gold_chunk_id(query)
    first_rank = _first_correct_rank(query, hits)
    primary_rank = _first_chunk_rank({primary_chunk} if primary_chunk else set(), hits)
    strict_rank = _first_chunk_rank(strict_chunks, hits)
    row: dict[str, Any] = {
        "query_id": query.get("query_id", ""),
        "query_type": query.get("query_type", ""),
        "primary_gold_chunk_id": primary_chunk,
        "gold_chunk_count": len(gold_chunks),
        "strict_gold_chunk_count": len(strict_chunks),
        "first_correct_rank": first_rank or "",
        "reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
        "primary_first_rank": primary_rank or "",
        "primary_reciprocal_rank": (1.0 / primary_rank) if primary_rank else 0.0,
        "strict_first_rank": strict_rank or "",
        "strict_reciprocal_rank": (1.0 / strict_rank) if strict_rank else 0.0,
        "top_hit_page": response.get("hit_page", ""),
        "returned_chunk_ids": "|".join(_hit_chunk_id(hit) for hit in hits if _hit_chunk_id(hit)),
        "returned_page_indices": "|".join(
            str(hit.get("page_idx")) for hit in hits if hit.get("page_idx") is not None
        ),
    }
    for k in recall_ks:
        top_hits = hits[:k]
        returned_gold_chunks = {_hit_chunk_id(hit) for hit in top_hits if _hit_chunk_id(hit) in gold_chunks}
        returned_strict_chunks = {
            _hit_chunk_id(hit) for hit in top_hits if _hit_chunk_id(hit) in strict_chunks
        }
        page_hit = any(
            hit.get("page_idx") is not None and int(hit.get("page_idx")) in gold_pages
            for hit in top_hits
            if str(hit.get("page_idx", "")).lstrip("-").isdigit()
        )
        chunk_hit = any(_hit_chunk_id(hit) in gold_chunks for hit in top_hits) if gold_chunks else False
        primary_hit = bool(primary_chunk) and any(_hit_chunk_id(hit) == primary_chunk for hit in top_hits)
        evidence_hit = chunk_hit if gold_chunks else page_hit
        row[f"page_recall@{k}"] = int(page_hit)
        row[f"chunk_recall@{k}"] = int(chunk_hit)
        row[f"primary_hit@{k}"] = int(primary_hit)
        row[f"strict_hit@{k}"] = int(bool(returned_strict_chunks))
        row[f"strict_coverage@{k}"] = (
            round(len(returned_strict_chunks) / len(strict_chunks), 6) if strict_chunks else 0.0
        )
        row[f"gold_coverage@{k}"] = round(len(returned_gold_chunks) / len(gold_chunks), 6) if gold_chunks else 0.0
        row[f"recall@{k}"] = int(evidence_hit)
        relevant_count = max(1, len(gold_chunks or gold_pages))
        ideal_count = min(k, relevant_count)
        ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
        dcg = 0.0
        seen_evidence: set[str] = set()
        for rank, hit in enumerate(top_hits, start=1):
            chunk_id = _hit_chunk_id(hit)
            evidence_key = ""
            if gold_chunks and chunk_id in gold_chunks:
                evidence_key = f"chunk:{chunk_id}"
            elif not gold_chunks and hit.get("page_idx") is not None:
                try:
                    page_idx = int(hit.get("page_idx"))
                except (TypeError, ValueError):
                    page_idx = None
                if page_idx in gold_pages:
                    evidence_key = f"page:{page_idx}"
            if not evidence_key or evidence_key in seen_evidence:
                continue
            seen_evidence.add(evidence_key)
            dcg += 1.0 / math.log2(rank + 1)
        row[f"ndcg@{k}"] = round(dcg / ideal_dcg, 6) if ideal_dcg else 0.0
    return row


def _call_retriever(
    *,
    url: str,
    query: str,
    api_key: str,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    import requests

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    start = time.perf_counter()
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(url, json={"query": query}, headers=headers, timeout=timeout)
            elapsed = time.perf_counter() - start
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= 3:
                raise
            time.sleep(2.0 * attempt)
    else:
        raise RuntimeError("Retriever request failed") from last_exc
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Retriever response must be a JSON object")
    return payload, elapsed


def _retrieval_result_response(result: Any) -> dict[str, Any]:
    def image_response(image: Any) -> dict[str, Any]:
        return {
            "hit_rank": image.hit_rank,
            "chunk_id": image.chunk_id,
            "source_relpath": image.source_relpath,
            "img_path": image.img_path,
            "image_path": image.image_path,
            "image_exists": image.image_exists,
            "page_idx": image.page_idx,
            "page_number": image.page_number,
            "bbox": list(image.bbox),
            "image_answering_policy": image.image_answering_policy,
            "image_answering_confidence": image.image_answering_confidence,
            "image_answering_reason": image.image_answering_reason,
            "image_caption": image.image_caption,
            "image_description_vlm": image.image_description_vlm,
        }

    final_output = getattr(result, "final_output", None)
    return {
        "hit_page": result.hit_page,
        "all_hits": [
            {
                "rank": hit.rank,
                "page_idx": hit.page_idx,
                "page_number": hit.page_number,
                "score": hit.score,
                "is_continuation": hit.is_continuation,
                "chunk_id": hit.chunk_id,
                "visual_page_prior": hit.visual_page_prior,
                "visual_alignment_score": hit.visual_alignment_score,
                "dense_rrf_score": hit.dense_rrf_score,
                "sparse_rrf_score": hit.sparse_rrf_score,
                "visual_rrf_score": hit.visual_rrf_score,
                "direct_text_rrf_score": hit.direct_text_rrf_score,
                "image_references": [image_response(image) for image in hit.image_references],
            }
            for hit in result.all_hits
        ],
        "context": result.context,
        "images": [image_response(image) for image in result.images],
        "final_output": {
            "mode": final_output.mode,
            "context": final_output.context,
            "content": [dict(item) for item in final_output.content],
            "images": [image_response(image) for image in final_output.images],
        }
        if final_output
        else None,
    }


def _hit_float(hit: dict[str, Any], key: str) -> float:
    try:
        return float(hit.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _hit_int(hit: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(hit.get(key))
    except (TypeError, ValueError):
        return default


def _response_feature_summary(query: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    hits = response.get("all_hits", [])
    if not isinstance(hits, list):
        hits = []
    top_hit = hits[0] if hits else {}
    gold_pages = set(_gold_page_indices(query))
    primary_chunk = _primary_gold_chunk_id(query)
    final_k = len(hits)
    top_final = hits[:final_k]
    returned_pages = {
        _hit_int(hit, "page_idx", -1)
        for hit in top_final
        if str(hit.get("page_idx", "")).lstrip("-").isdigit()
    }
    visual_hits = [hit for hit in top_final if _hit_float(hit, "visual_page_prior") > 0]
    visual_pages = {
        _hit_int(hit, "page_idx", -1)
        for hit in visual_hits
        if str(hit.get("page_idx", "")).lstrip("-").isdigit()
    }
    primary_hit_final = bool(primary_chunk) and any(_hit_chunk_id(hit) == primary_chunk for hit in top_final)
    same_page_misallocation = bool(gold_pages & returned_pages) and not primary_hit_final
    visual_top = bool(_hit_float(top_hit, "visual_page_prior") > 0)
    visual_overweight_failure = (
        visual_top
        and not primary_hit_final
        and _hit_float(top_hit, "visual_rrf_score") > _hit_float(top_hit, "direct_text_rrf_score")
    )
    return {
        "top_dense_rrf_score": round(_hit_float(top_hit, "dense_rrf_score"), 8),
        "top_sparse_rrf_score": round(_hit_float(top_hit, "sparse_rrf_score"), 8),
        "top_visual_rrf_score": round(_hit_float(top_hit, "visual_rrf_score"), 8),
        "top_direct_text_rrf_score": round(_hit_float(top_hit, "direct_text_rrf_score"), 8),
        "top_visual_alignment_score": round(_hit_float(top_hit, "visual_alignment_score"), 6),
        "visual_hit_count": len(visual_hits),
        "visual_gold_page_hit": int(bool(gold_pages & visual_pages)),
        "same_page_misallocation": int(same_page_misallocation),
        "visual_overweight_failure": int(visual_overweight_failure),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_review_worklist(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    base_fieldnames = [
        "query_id",
        "query_type",
        "query",
        "primary_gold_chunk_id",
        "gold_page_indices",
        "gold_chunk_ids",
        "returned_chunk_ids",
        "returned_page_indices",
        "first_correct_rank",
        "context_path",
        "response_path",
        "context_usefulness_score",
        "evidence_precision_score",
        "visual_attribution_score",
        "failure_type",
        "review_notes",
    ]
    extras = sorted({key for row in rows for key in row if key not in base_fieldnames})
    fieldnames = base_fieldnames[:9] + extras + base_fieldnames[9:]
    _write_csv(path, rows, fieldnames)


def run_retrieval_benchmark(
    *,
    query_set: Path,
    output_dir: Path,
    url: str,
    config: AppConfig | None = None,
    api_key: str = "",
    timeout: float = 120.0,
    run_id: str | None = None,
    limit: int | None = None,
    direct: bool = False,
    dry_run: bool = False,
) -> Path:
    queries = _read_jsonl(query_set)
    if limit is not None:
        queries = queries[:limit]
    run_id = run_id or datetime.now().strftime("retrieval-%Y%m%d-%H%M%S")
    run_dir = output_dir / run_id
    if dry_run:
        print(f"Retrieval benchmark run dir: {run_dir}")
        print(f"Query set: {query_set}")
        print(f"Queries: {len(queries)}")
        print(f"Mode: {'direct engine' if direct else 'HTTP API'}")
        print(f"Retriever URL: {url}")
        print(f"API key configured: {bool(api_key)}")
        return run_dir

    run_config = config or AppConfig.from_env()
    recall_ks = tuple(sorted({*DEFAULT_RECALL_KS, run_config.retrieval.final_top_k}))
    responses_dir = run_dir / "responses"
    contexts_dir = run_dir / "contexts"
    responses_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    engine = None
    if direct:
        from rag_flow.retrieval import RetrievalEngine

        engine = RetrievalEngine(run_config)
        engine.load()

    for index, query in enumerate(queries, start=1):
        query_id = str(query.get("query_id") or f"query-{index:04d}")
        if engine is not None:
            start = time.perf_counter()
            response = _retrieval_result_response(engine.retrieve(str(query.get("query", ""))))
            elapsed = time.perf_counter() - start
        else:
            response, elapsed = _call_retriever(
                url=url,
                query=str(query.get("query", "")),
                api_key=api_key,
                timeout=timeout,
            )
        response_path = responses_dir / f"{query_id}.json"
        context_path = contexts_dir / f"{query_id}.txt"
        response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        context_path.write_text(str(response.get("context", "")), encoding="utf-8")
        score_row = score_query_result(query, response, recall_ks=recall_ks)
        feature_row = _response_feature_summary(query, response)
        score_row.update(
            {
                "query": query.get("query", ""),
                "requires_visual": int(bool(query.get("requires_visual", False))),
                "requires_table": int(bool(query.get("requires_table", False))),
                "difficulty": query.get("difficulty", ""),
                "expected_context_granularity": query.get("expected_context_granularity", ""),
                "gold_page_indices": "|".join(str(page) for page in _gold_page_indices(query)),
                "gold_chunk_ids": "|".join(_gold_chunk_ids(query)),
                "strict_gold_chunk_ids": "|".join(_strict_gold_chunk_ids(query)),
                "latency_seconds": round(elapsed, 4),
                "hit_count": len(response.get("all_hits", [])) if isinstance(response.get("all_hits"), list) else 0,
                "context_chars": len(str(response.get("context", ""))),
                "response_path": str(response_path),
                "context_path": str(context_path),
                "route_mode": run_config.retrieval.route_mode,
                "candidate_mode": run_config.retrieval.candidate_mode,
                "retrieval_k": run_config.retrieval.retrieval_k,
                "final_top_k": run_config.retrieval.final_top_k,
                "rrf_k": run_config.retrieval.rrf_k,
                "visual_weight": run_config.retrieval.visual_weight,
                "enable_visual": int(run_config.retrieval.enable_visual),
                "candidate_scroll_limit": run_config.retrieval.candidate_scroll_limit,
                **feature_row,
            }
        )
        result_rows.append(score_row)
        review_rows.append(
            {
                "query_id": query_id,
                "query_type": query.get("query_type", ""),
                "query": query.get("query", ""),
                "primary_gold_chunk_id": _primary_gold_chunk_id(query),
                "gold_page_indices": "|".join(str(page) for page in _gold_page_indices(query)),
                "gold_chunk_ids": "|".join(_gold_chunk_ids(query)),
                "strict_gold_chunk_ids": "|".join(_strict_gold_chunk_ids(query)),
                "returned_chunk_ids": score_row["returned_chunk_ids"],
                "returned_page_indices": score_row["returned_page_indices"],
                "first_correct_rank": score_row["first_correct_rank"],
                "primary_first_rank": score_row["primary_first_rank"],
                "strict_first_rank": score_row["strict_first_rank"],
                f"primary_hit@{run_config.retrieval.final_top_k}": score_row[
                    f"primary_hit@{run_config.retrieval.final_top_k}"
                ],
                f"gold_coverage@{run_config.retrieval.final_top_k}": score_row[
                    f"gold_coverage@{run_config.retrieval.final_top_k}"
                ],
                f"strict_coverage@{run_config.retrieval.final_top_k}": score_row[
                    f"strict_coverage@{run_config.retrieval.final_top_k}"
                ],
                "route_mode": run_config.retrieval.route_mode,
                "candidate_mode": run_config.retrieval.candidate_mode,
                "retrieval_k": run_config.retrieval.retrieval_k,
                "final_top_k": run_config.retrieval.final_top_k,
                "rrf_k": run_config.retrieval.rrf_k,
                "visual_weight": run_config.retrieval.visual_weight,
                "enable_visual": int(run_config.retrieval.enable_visual),
                "candidate_scroll_limit": run_config.retrieval.candidate_scroll_limit,
                "top_dense_rrf_score": feature_row["top_dense_rrf_score"],
                "top_sparse_rrf_score": feature_row["top_sparse_rrf_score"],
                "top_visual_rrf_score": feature_row["top_visual_rrf_score"],
                "top_visual_alignment_score": feature_row["top_visual_alignment_score"],
                "visual_hit_count": feature_row["visual_hit_count"],
                "visual_gold_page_hit": feature_row["visual_gold_page_hit"],
                "same_page_misallocation": feature_row["same_page_misallocation"],
                "visual_overweight_failure": feature_row["visual_overweight_failure"],
                "context_path": str(context_path),
                "response_path": str(response_path),
                "context_usefulness_score": "",
                "evidence_precision_score": "",
                "visual_attribution_score": "",
                "failure_type": "",
                "review_notes": query.get("notes", ""),
            }
        )
        print(f"[{index}/{len(queries)}] {query_id}: {elapsed:.2f}s rank={score_row['first_correct_rank'] or 'miss'}")

    result_fieldnames = [
        "query_id",
        "query_type",
        "query",
        "requires_visual",
        "requires_table",
        "difficulty",
        "expected_context_granularity",
        "primary_gold_chunk_id",
        "gold_page_indices",
        "gold_chunk_ids",
        "strict_gold_chunk_ids",
        "gold_chunk_count",
        "strict_gold_chunk_count",
        "first_correct_rank",
        "reciprocal_rank",
        "primary_first_rank",
        "primary_reciprocal_rank",
        "strict_first_rank",
        "strict_reciprocal_rank",
        "top_hit_page",
        "returned_chunk_ids",
        "returned_page_indices",
        "latency_seconds",
        "hit_count",
        "context_chars",
        "route_mode",
        "candidate_mode",
        "retrieval_k",
        "final_top_k",
        "rrf_k",
        "visual_weight",
        "enable_visual",
        "candidate_scroll_limit",
        "top_dense_rrf_score",
        "top_sparse_rrf_score",
        "top_visual_rrf_score",
        "top_direct_text_rrf_score",
        "top_visual_alignment_score",
        "visual_hit_count",
        "visual_gold_page_hit",
        "same_page_misallocation",
        "visual_overweight_failure",
        "response_path",
        "context_path",
        *[f"page_recall@{k}" for k in recall_ks],
        *[f"chunk_recall@{k}" for k in recall_ks],
        *[f"primary_hit@{k}" for k in recall_ks],
        *[f"strict_hit@{k}" for k in recall_ks],
        *[f"strict_coverage@{k}" for k in recall_ks],
        *[f"gold_coverage@{k}" for k in recall_ks],
        *[f"recall@{k}" for k in recall_ks],
        *[f"ndcg@{k}" for k in recall_ks],
    ]
    _write_csv(run_dir / "run_summary.csv", result_rows, result_fieldnames)
    _write_review_worklist(run_dir / "review_worklist.csv", review_rows)
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in result_rows) + "\n",
        encoding="utf-8",
    )
    print(f"Run summary: {run_dir / 'run_summary.csv'}")
    print(f"Review worklist: {run_dir / 'review_worklist.csv'}")
    return run_dir


def build_parser(config: AppConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval benchmark workflows.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    init_parser = subparsers.add_parser("init-pilot", help="Write the pilot query set JSONL template.")
    init_parser.add_argument("--output", type=Path, default=DEFAULT_PILOT_QUERY_SET)
    init_parser.add_argument("--force", action="store_true")

    full_parser = subparsers.add_parser(
        "init-full",
        help="Generate a PDF-grounded full query set from chunked and captioned outputs.",
    )
    full_parser.add_argument("--chunks", type=Path, default=config.paths.chunks_json)
    full_parser.add_argument("--captioned-json", type=Path, default=config.paths.captioned_json)
    full_parser.add_argument("--pdf", type=Path, default=config.paths.source_pdf)
    full_parser.add_argument("--output", type=Path, default=DEFAULT_FULL_QUERY_SET)
    full_parser.add_argument("--target-size", type=int, default=100)
    full_parser.add_argument(
        "--visual-page-count",
        type=int,
        help="Number of page-level visual questions to include; defaults to about 15 percent.",
    )
    full_parser.add_argument("--force", action="store_true")

    anchor_parser = subparsers.add_parser(
        "build-evidence-anchors",
        help="Convert chunk-id gold labels into stable block/page evidence anchors.",
    )
    anchor_parser.add_argument("--query-set", type=Path, required=True)
    anchor_parser.add_argument("--chunks", type=Path, default=config.paths.chunks_json)
    anchor_parser.add_argument("--output", type=Path, required=True)
    anchor_parser.add_argument("--audit-output", type=Path)

    remap_parser = subparsers.add_parser(
        "remap-query-set",
        help="Map stable evidence anchors onto a freshly chunked JSON file.",
    )
    remap_parser.add_argument("--query-set", type=Path, required=True)
    remap_parser.add_argument("--chunks", type=Path, default=config.paths.chunks_json)
    remap_parser.add_argument("--output", type=Path, required=True)
    remap_parser.add_argument("--audit-output", type=Path)

    run_parser = subparsers.add_parser("run", help="Run a query set against the retrieval API.")
    run_parser.add_argument("--query-set", type=Path, default=DEFAULT_PILOT_QUERY_SET)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--url", default=config.server.retriever_url)
    run_parser.add_argument("--api-key", default=config.server.retriever_api_key)
    run_parser.add_argument("--timeout", type=float, default=120.0)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--dry-run", action="store_true")
    direct_parser = subparsers.add_parser(
        "run-direct",
        help="Run a query set through RetrievalEngine without starting the HTTP API.",
    )
    direct_parser.add_argument("--query-set", type=Path, default=DEFAULT_PILOT_QUERY_SET)
    direct_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    direct_parser.add_argument("--run-id")
    direct_parser.add_argument("--timeout", type=float, default=120.0)
    direct_parser.add_argument("--limit", type=int)
    direct_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = build_parser(config)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.stage == "init-pilot":
        path = write_pilot_query_set(args.output, force=args.force)
        print(f"Wrote pilot query set: {path}")
        print(f"Queries: {len(PILOT_QUERIES)}")
        return

    if args.stage == "init-full":
        path = write_full_query_set(
            args.chunks,
            args.output,
            captioned_json_path=args.captioned_json,
            pdf_path=args.pdf,
            target_size=args.target_size,
            visual_page_count=args.visual_page_count,
            force=args.force,
        )
        print(f"Wrote full query set: {path}")
        print(f"Queries: {args.target_size}")
        return

    if args.stage == "build-evidence-anchors":
        path, audit_rows = build_evidence_anchors(
            query_set=args.query_set,
            chunks_path=args.chunks,
            output=args.output,
            audit_output=args.audit_output,
        )
        print(f"Wrote evidence-anchored query set: {path}")
        print(f"Queries: {len(audit_rows)}")
        return

    if args.stage == "remap-query-set":
        path, audit_rows = remap_query_set_to_chunks(
            query_set=args.query_set,
            chunks_path=args.chunks,
            output=args.output,
            audit_output=args.audit_output,
        )
        page_only = sum(1 for row in audit_rows if row.get("warning") == "page_only_gold")
        unmapped = sum(1 for row in audit_rows if row.get("warning") == "block_anchor_not_mapped")
        print(f"Wrote remapped query set: {path}")
        print(f"Queries: {len(audit_rows)}")
        print(f"Page-only fallback: {page_only}")
        print(f"Unmapped block anchors: {unmapped}")
        return

    if args.stage in {"run", "run-direct"}:
        run_retrieval_benchmark(
            query_set=args.query_set,
            output_dir=args.output_dir,
            url=getattr(args, "url", config.server.retriever_url),
            config=config,
            api_key=getattr(args, "api_key", ""),
            timeout=args.timeout,
            run_id=args.run_id,
            limit=args.limit,
            direct=args.stage == "run-direct",
            dry_run=args.dry_run,
        )
        return

    raise SystemExit(f"Unknown retrieval benchmark stage: {args.stage}")


if __name__ == "__main__":
    main()
