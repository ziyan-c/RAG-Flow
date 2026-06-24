from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / ".local" / "CUSTOM_DATA" / "pdfs" / "source"
OUTPUT_ROOT = REPO_ROOT / ".local" / "CUSTOM_DATA" / "pdfs" / "output"
REPORT_ROOT = REPO_ROOT / ".local" / "CUSTOM_DATA" / "reports" / "dahua_pdf_review"


def _sidecar_path(pdf: Path) -> Path:
    return pdf.with_name(f"{pdf.stem}_metadata.yml")


def _content_json_for_pdf(pdf: Path) -> Path | None:
    rel_parent = pdf.relative_to(SOURCE_ROOT).parent
    docdir = OUTPUT_ROOT / rel_parent / pdf.stem
    candidates = [
        path
        for path in sorted(docdir.rglob("*_content_list.json"))
        if not any(suffix in path.name for suffix in ("SECTIONED", "PATCHED", "CAPTIONED", "CHUNKED", "TAGGED"))
    ]
    if candidates:
        return candidates[0]
    return None


def _text_from_item(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    raw = item.get("text") or item.get("caption") or item.get("image_caption")
    if isinstance(raw, list):
        raw = " ".join(str(part) for part in raw)
    if raw:
        return re.sub(r"\s+", " ", str(raw)).strip()
    content = item.get("content")
    if isinstance(content, str):
        return re.sub(r"\s+", " ", content).strip()
    if isinstance(content, dict):
        parts: list[str] = []
        for value in content.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(part) for part in value if isinstance(part, (str, int, float)))
        return re.sub(r"\s+", " ", " ".join(parts)).strip()
    return ""


def _extract_evidence(content_json: Path | None) -> dict[str, Any]:
    if content_json is None or not content_json.exists():
        return {
            "content_json": None,
            "block_count": 0,
            "titles": [],
            "headings": [],
            "front_text": [],
            "tail_text": [],
            "model_context": [],
            "caption_context": [],
        }

    try:
        data = json.loads(content_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "content_json": str(content_json),
            "error": str(exc),
            "block_count": 0,
            "titles": [],
            "headings": [],
            "front_text": [],
            "tail_text": [],
            "model_context": [],
            "caption_context": [],
        }

    if not isinstance(data, list):
        return {
            "content_json": str(content_json),
            "block_count": 0,
            "titles": [],
            "headings": [],
            "front_text": [],
            "tail_text": [],
            "model_context": [],
            "caption_context": [],
        }

    titles: list[str] = []
    headings: list[str] = []
    front_text: list[str] = []
    tail_text: list[str] = []
    model_context: list[str] = []
    caption_context: list[str] = []
    model_pattern = re.compile(r"\b(?:(?:DH|DHI)-)?[A-Z]{2,6}[A-Z0-9][A-Z0-9._()+/-]{2,45}\b")

    for idx, item in enumerate(data):
        text = _text_from_item(item)
        if not text:
            continue
        item_type = str(item.get("type") or "") if isinstance(item, dict) else ""
        text_level = item.get("text_level") if isinstance(item, dict) else None
        if text_level == 1 and len(titles) < 20:
            titles.append(text)
        elif isinstance(text_level, int) and text_level <= 3 and len(headings) < 80:
            headings.append(text)
        if len(front_text) < 80:
            front_text.append(text)
        if idx >= max(0, len(data) - 40):
            tail_text.append(text)
        if model_pattern.search(text) and len(model_context) < 80:
            model_context.append(text)
        if item_type in {"image", "table"} and len(caption_context) < 40:
            caption_context.append(text)

    return {
        "content_json": str(content_json.relative_to(REPO_ROOT)),
        "block_count": len(data),
        "titles": titles,
        "headings": headings,
        "front_text": front_text,
        "tail_text": tail_text[-40:],
        "model_context": model_context,
        "caption_context": caption_context,
    }


def build_packets() -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for pdf in sorted(SOURCE_ROOT.rglob("*.pdf"), key=lambda item: item.relative_to(SOURCE_ROOT).as_posix().lower()):
        sidecar = _sidecar_path(pdf)
        content_json = _content_json_for_pdf(pdf)
        packets.append(
            {
                "source_relpath": pdf.relative_to(SOURCE_ROOT).as_posix(),
                "filename": pdf.name,
                "sidecar": sidecar.relative_to(SOURCE_ROOT).as_posix() if sidecar.exists() else None,
                "current_sidecar_text": sidecar.read_text(encoding="utf-8") if sidecar.exists() else "",
                "evidence": _extract_evidence(content_json),
            }
        )
    return packets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-PDF content evidence packets for human metadata review.")
    parser.add_argument("--jsonl", default=str(REPORT_ROOT / "review_packets.jsonl"))
    parser.add_argument("--summary", default=str(REPORT_ROOT / "summary.json"))
    parser.add_argument("--sample-md", default=str(REPORT_ROOT / "sample.md"))
    parser.add_argument("--sample-count", type=int, default=30)
    args = parser.parse_args()

    packets = build_packets()
    jsonl_path = Path(args.jsonl)
    summary_path = Path(args.summary)
    sample_path = Path(args.sample_md)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for packet in packets:
            fh.write(json.dumps(packet, ensure_ascii=False) + "\n")

    summary = {
        "packet_count": len(packets),
        "missing_sidecar_count": sum(1 for packet in packets if not packet["sidecar"]),
        "missing_content_json_count": sum(1 for packet in packets if not packet["evidence"].get("content_json")),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sample_lines: list[str] = ["# Dahua PDF Metadata Review Sample", ""]
    for packet in packets[: args.sample_count]:
        evidence = packet["evidence"]
        sample_lines.extend(
            [
                f"## {packet['source_relpath']}",
                "",
                "### Titles",
                *[f"- {item}" for item in evidence["titles"][:8]],
                "",
                "### Headings",
                *[f"- {item}" for item in evidence["headings"][:12]],
                "",
                "### Model Context",
                *[f"- {item}" for item in evidence["model_context"][:8]],
                "",
                "### Current Sidecar",
                "```yaml",
                packet["current_sidecar_text"].strip(),
                "```",
                "",
            ]
        )
    sample_path.write_text("\n".join(sample_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"jsonl={jsonl_path}")
    print(f"sample={sample_path}")


if __name__ == "__main__":
    main()
