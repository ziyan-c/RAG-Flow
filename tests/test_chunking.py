from __future__ import annotations

import json

from rag_flow.chunking import create_page_level_chunks


def test_create_page_level_chunks(tmp_path):
    content = [
        {"type": "text", "page_idx": 0, "text": "Overview"},
        {"type": "list", "page_idx": 0, "list_items": ["One", "Two"]},
        {
            "type": "table",
            "page_idx": 1,
            "table_caption": ["Ports"],
            "table_body": ["Port 8000: retriever"],
            "table_footnote": ["Local only"],
        },
        {
            "type": "image",
            "page_idx": 1,
            "image_caption": ["Login"],
            "image_description_vlm": "A login screen.",
            "img_path": "images/login.png",
        },
    ]
    input_path = tmp_path / "content.json"
    input_path.write_text(json.dumps(content), encoding="utf-8")

    chunks = create_page_level_chunks(input_path, "manual.pdf")

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["page_idx"] == 0
    assert "Overview" in chunks[0]["page_content"]
    assert "Port 8000" in chunks[1]["page_content"]
    assert chunks[1]["metadata"]["images_on_page"] == ["images/login.png"]
