from __future__ import annotations

import json

from rag_flow.chunking import create_chunks, create_page_level_chunks, main


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
            "img_path": "tables/ports.png",
        },
        {
            "type": "image",
            "page_idx": 1,
            "image_caption": ["Login"],
            "image_footnote": ["Step 3 Click OK."],
            "image_description_vlm": "A login screen.",
            "image_answering_policy": "image_recommended",
            "image_answering_confidence": "medium",
            "image_answering_reason": "Visible labels may matter.",
            "img_path": "images/login.png",
            "bbox": [100, 100, 400, 300],
        },
        {
            "type": "image",
            "page_idx": 1,
            "img_path": "images/tiny-icon.png",
            "vlm-small-icon-inline-icon": True,
        },
        {"type": "header", "page_idx": 1, "text": "User Manual"},
        {"type": "footer", "page_idx": 1, "text": "Contact us"},
        {"type": "page_number", "page_idx": 1, "text": "2"},
    ]
    input_path = tmp_path / "content.json"
    input_path.write_text(json.dumps(content), encoding="utf-8")

    chunks = create_page_level_chunks(input_path, "manual.pdf")

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["page_idx"] == 0
    assert chunks[0]["metadata"]["chunk_id"] == "manual-chunk-00000"
    assert chunks[0]["metadata"]["chunk_mode"] == "page"
    assert chunks[0]["metadata"]["breadcrumb"] == "manual.pdf"
    assert chunks[0]["metadata"]["page_indices"] == [0]
    assert chunks[0]["metadata"]["block_indices"] == [0, 1]
    assert chunks[0]["chunk_content"].startswith("[Breadcrumb: manual.pdf]")
    assert "Overview" in chunks[0]["chunk_content"]
    assert "Port 8000" in chunks[1]["chunk_content"]
    assert "Step 3 Click OK." in chunks[1]["chunk_content"]
    assert "User Manual" not in chunks[1]["chunk_content"]
    assert "Contact us" not in chunks[1]["chunk_content"]
    assert chunks[1]["metadata"]["images_on_page"] == ["images/login.png"]
    assert chunks[1]["metadata"]["image_answering_evidence"] == [
        {
            "img_path": "images/login.png",
            "block_idx": 3,
            "page_idx": 1,
            "bbox": [100.0, 100.0, 400.0, 300.0],
            "image_caption": "Login",
            "image_answering_policy": "image_recommended",
            "image_answering_confidence": "medium",
            "image_answering_reason": "Visible labels may matter.",
        }
    ]
    assert chunks[1]["metadata"]["tables_on_page"] == ["tables/ports.png"]
    assert chunks[1]["metadata"]["block_indices"] == [2, 3]


def test_auto_chunks_without_sections_use_token_windows(tmp_path):
    content = [
        {"type": "text", "page_idx": 0, "text": "alpha beta gamma delta", "bbox": [10, 10, 100, 30]},
        {"type": "text", "page_idx": 1, "text": "epsilon zeta eta theta", "bbox": [20, 20, 120, 40]},
        {"type": "text", "page_idx": 2, "text": "iota kappa lambda mu", "bbox": [30, 30, 130, 50]},
    ]
    input_path = tmp_path / "content.json"
    input_path.write_text(json.dumps(content), encoding="utf-8")

    chunks = create_chunks(input_path, "manual.pdf", mode="auto", max_tokens=5, overlap_tokens=0, min_tokens=1)

    assert len(chunks) == 3
    assert all(chunk["metadata"]["chunk_mode"] == "token" for chunk in chunks)
    assert chunks[0]["metadata"]["breadcrumb"] == "manual.pdf"
    assert chunks[0]["chunk_content"].startswith("[Breadcrumb: manual.pdf]")
    assert chunks[0]["metadata"]["page_indices"] == [0]
    assert chunks[0]["metadata"]["bboxes_by_page"] == {"0": [[10.0, 10.0, 100.0, 30.0]]}
    assert chunks[0]["metadata"]["block_indices"] == [0]
    assert chunks[1]["metadata"]["page_indices"] == [1]
    assert chunks[2]["metadata"]["chunk_id"] == "manual-chunk-00002"


def test_auto_chunks_with_sections_keep_section_boundaries(tmp_path):
    content = [
        {
            "type": "text",
            "page_idx": 0,
            "text": "1 Overview",
            "section_path": ["1 Overview"],
            "section_level": 1,
            "section_source": "pdf_outline_exact",
            "source_relpath": "DSS/manual.pdf",
            "source_filename": "manual.pdf",
            "breadcrumb": "DSS > manual.pdf > 1 Overview",
            "bbox": [10, 10, 200, 30],
        },
        {
            "type": "text",
            "page_idx": 0,
            "text": "overview body",
            "section_path": ["1 Overview"],
            "source_relpath": "DSS/manual.pdf",
            "source_filename": "manual.pdf",
            "breadcrumb": "DSS > manual.pdf > 1 Overview",
            "bbox": [10, 40, 200, 80],
        },
        {
            "type": "text",
            "page_idx": 1,
            "text": "2 Setup",
            "section_path": ["2 Setup"],
            "section_level": 1,
            "section_source": "pdf_outline_exact",
            "source_relpath": "DSS/manual.pdf",
            "source_filename": "manual.pdf",
            "breadcrumb": "DSS > manual.pdf > 2 Setup",
            "bbox": [10, 10, 220, 30],
        },
        {
            "type": "text",
            "page_idx": 1,
            "text": "setup body",
            "section_path": ["2 Setup"],
            "source_relpath": "DSS/manual.pdf",
            "source_filename": "manual.pdf",
            "breadcrumb": "DSS > manual.pdf > 2 Setup",
            "bbox": [10, 40, 220, 80],
        },
    ]
    input_path = tmp_path / "content.json"
    input_path.write_text(json.dumps(content), encoding="utf-8")

    chunks = create_chunks(input_path, "wrong.pdf", mode="auto", max_tokens=100, overlap_tokens=0, min_tokens=1)

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["chunk_mode"] == "section"
    assert chunks[0]["metadata"]["source_relpath"] == "DSS/manual.pdf"
    assert chunks[0]["metadata"]["source_filename"] == "manual.pdf"
    assert chunks[0]["metadata"]["section_path"] == ["1 Overview"]
    assert chunks[0]["metadata"]["breadcrumb"] == "DSS > manual.pdf > 1 Overview"
    assert chunks[0]["chunk_content"].startswith("[Breadcrumb: DSS > manual.pdf > 1 Overview]")
    assert "[Section: 1 Overview]" in chunks[0]["chunk_content"]
    assert chunks[0]["metadata"]["bboxes_by_page"]["0"] == [[10.0, 10.0, 200.0, 30.0], [10.0, 40.0, 200.0, 80.0]]
    assert chunks[1]["metadata"]["section_path"] == ["2 Setup"]
    assert "setup body" not in chunks[0]["chunk_content"]


def test_chunking_attaches_table_continuation_regions_to_master_chunk(tmp_path):
    content = [
        {
            "type": "table",
            "page_idx": 0,
            "bbox": [100, 500, 900, 900],
            "table_caption": ["Table 1"],
            "table_body": "<table><tr><td>Record Mode</td></tr></table>",
            "table_footnote": [],
            "section_path": ["1 Overview"],
        },
        {
            "type": "table",
            "page_idx": 1,
            "bbox": [100, 80, 900, 300],
            "table_caption": [],
            "table_body": "",
            "table_footnote": [],
            "img_path": "",
            "section_path": ["1 Overview"],
        },
    ]
    input_path = tmp_path / "content.json"
    input_path.write_text(json.dumps(content), encoding="utf-8")

    chunks = create_chunks(input_path, "manual.pdf", mode="auto", max_tokens=100, overlap_tokens=0, min_tokens=1)

    assert len(chunks) == 1
    assert chunks[0]["chunk_content"].count("Record Mode") == 1
    metadata = chunks[0]["metadata"]
    assert metadata["page_indices"] == [0, 1]
    assert metadata["page_start"] == 0
    assert metadata["page_end"] == 1
    assert metadata["block_indices"] == [0, 1]
    assert metadata["bboxes_by_page"] == {
        "0": [[100.0, 500.0, 900.0, 900.0]],
        "1": [[100.0, 80.0, 900.0, 300.0]],
    }
    assert metadata["table_continuations"] == [
        {
            "master_block_idx": 0,
            "continuation_block_indices": [1],
            "continuation_page_indices": [1],
        }
    ]


def test_page_level_chunks_copy_master_table_text_to_continuation_page(tmp_path):
    content = [
        {
            "type": "table",
            "page_idx": 0,
            "bbox": [100, 500, 900, 900],
            "table_caption": ["Table 1"],
            "table_body": "<table><tr><td>Record Mode</td></tr></table>",
            "table_footnote": [],
        },
        {
            "type": "table",
            "page_idx": 1,
            "bbox": [100, 80, 900, 300],
            "table_caption": [],
            "table_body": "",
            "table_footnote": [],
            "img_path": "",
        },
    ]
    input_path = tmp_path / "content.json"
    input_path.write_text(json.dumps(content), encoding="utf-8")

    chunks = create_page_level_chunks(input_path, "manual.pdf")

    assert len(chunks) == 2
    assert chunks[0]["metadata"]["page_idx"] == 0
    assert chunks[1]["metadata"]["page_idx"] == 1
    assert chunks[0]["metadata"]["block_indices"] == [0]
    assert chunks[1]["metadata"]["block_indices"] == [1]
    assert chunks[1]["metadata"]["bboxes_by_page"] == {"1": [[100.0, 80.0, 900.0, 300.0]]}
    assert "[Continuation of table from page 1]" in chunks[1]["chunk_content"]
    assert "Record Mode" in chunks[1]["chunk_content"]


def test_chunking_main_dry_run_prints_settings(capsys):
    main(["--input", "missing.json", "--mode", "token", "--max-tokens", "900", "--dry-run"])

    output = capsys.readouterr().out
    assert "Chunking inputs:" in output
    assert "mode: token" in output
    assert "max_tokens: 900" in output
