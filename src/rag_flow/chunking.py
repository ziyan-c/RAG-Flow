from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import AppConfig


def _join_field(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def create_page_level_chunks(
    json_path: str | Path,
    source_name: str,
) -> list[dict[str, Any]]:
    with Path(json_path).open("r", encoding="utf-8") as f:
        content_data = json.load(f)

    page_contents: dict[int, list[str]] = defaultdict(list)
    page_images: dict[int, list[str]] = defaultdict(list)

    for block in content_data:
        page_idx = int(block.get("page_idx", 0))
        block_type = block.get("type")

        if block_type == "image":
            description = str(block.get("image_description_vlm", "")).strip()
            caption = _join_field(block.get("image_caption", [])).strip()
            footnote = _join_field(block.get("image_footnote", [])).strip()
            if description or caption or footnote:
                parts = []
                label = f"[Image with illustration: {caption}]" if caption else "[Image with illustration]"
                if description or caption:
                    parts.append(f"\n{label}\n{description}".strip())
                if footnote:
                    parts.append(f"[Image footnote: {footnote}]")
                page_contents[page_idx].append("\n".join(parts))
            if block.get("img_path"):
                page_images[page_idx].append(block["img_path"])

        elif block_type == "table":
            caption = _join_field(block.get("table_caption", [])).strip()
            body = _join_field(block.get("table_body", [])).strip()
            footnote = _join_field(block.get("table_footnote", [])).strip()
            parts = []
            if caption:
                parts.append(f"[Table: {caption}]")
            if body:
                parts.append(body)
            if footnote:
                parts.append(f"[Footnote: {footnote}]")
            if parts:
                page_contents[page_idx].append("\n".join(parts))

        elif block_type in {"text", "list"}:
            key = "list_items" if block_type == "list" else "text"
            text = _join_field(block.get(key, [])).strip()
            if text:
                page_contents[page_idx].append(text)

    chunks: list[dict[str, Any]] = []
    for page_idx in sorted(page_contents):
        page_content = "\n\n".join(page_contents[page_idx]).strip()
        if not page_content:
            continue
        chunks.append(
            {
                "page_content": page_content,
                "metadata": {
                    "source": source_name,
                    "page_idx": page_idx,
                    "images_on_page": page_images[page_idx],
                },
            }
        )
    return chunks


def write_chunks(chunks: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Build page-level chunks from MinerU content_list JSON.")
    parser.add_argument("--input", default=str(config.paths.captioned_json), help="Input enriched content_list JSON.")
    parser.add_argument("--output", default=str(config.paths.chunks_json), help="Output page-level chunks JSON.")
    parser.add_argument("--source-name", default=config.paths.source_name, help="Source PDF name stored in metadata.")
    args = parser.parse_args(argv)

    chunks = create_page_level_chunks(args.input, args.source_name)
    write_chunks(chunks, args.output)
    print(f"Created {len(chunks)} page-level chunks at {args.output}")


if __name__ == "__main__":
    main()
