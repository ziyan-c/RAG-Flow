from __future__ import annotations

import json
from pathlib import Path

from rag_flow.sectioning import (
    OutlineEntry,
    add_source_metadata,
    main,
    normalize_title,
    section_content,
    sectioned_path_for,
    sectioning_audit_path_for,
    write_sectioned_json,
)
from rag_flow.table_continuations import (
    TABLE_CONTINUATION_INDICES_KEY,
    TABLE_CONTINUATION_MASTER_IDX_KEY,
)


def test_section_paths_insert_stage_suffixes():
    assert sectioned_path_for(Path("/tmp/manual_content_list.json")).name == "manual_content_list_SECTIONED.json"
    assert (
        sectioned_path_for(Path("/tmp/manual_content_list_PATCHED.json")).name
        == "manual_content_list_SECTIONED_PATCHED.json"
    )
    assert sectioning_audit_path_for(Path("/tmp/manual_content_list.json")).name == "manual_SECTIONING_AUDIT.json"


def test_normalize_title_handles_nbsp_and_spacing():
    assert normalize_title(" 2.1\u00a0\u00a0Standalone   Deployment ") == "2.1 standalone deployment"


def test_add_source_metadata_builds_block_breadcrumbs():
    content = [
        {"type": "text", "page_idx": 0, "text": "Cover"},
        {"type": "text", "page_idx": 1, "text": "1 Overview", "section_path": ["1 Overview"]},
    ]

    result = add_source_metadata(content, source_name="DSS/manual.pdf")

    assert result[0]["source_relpath"] == "DSS/manual.pdf"
    assert result[0]["source_filename"] == "manual.pdf"
    assert result[0]["breadcrumb"] == "DSS > manual.pdf"
    assert result[1]["breadcrumb"] == "DSS > manual.pdf > 1 Overview"


def test_section_content_matches_exact_fuzzy_and_page_fallback():
    content = [
        {"type": "header", "page_idx": 0, "text": "Header"},
        {"type": "text", "page_idx": 0, "text": "1 Overview", "bbox": [10, 100, 200, 120]},
        {"type": "text", "page_idx": 0, "text": "Intro body"},
        {"type": "text", "page_idx": 0, "text": "1.1 Introduction", "bbox": [10, 300, 220, 320]},
        {"type": "text", "page_idx": 0, "text": "Introduction body"},
        {"type": "text", "page_idx": 1, "text": "2 Installing into DSS Client", "bbox": [10, 100, 300, 120]},
        {"type": "text", "page_idx": 1, "text": "Install body"},
        {"type": "footer", "page_idx": 2, "text": "Footer"},
        {"type": "text", "page_idx": 2, "text": "Fallback body"},
    ]
    entries = [
        OutlineEntry(
            outline_index=0,
            level=1,
            title="1\u00a0Overview",
            page_idx=0,
            section_path=("1\u00a0Overview",),
            dest_y=900,
            page_height=1000,
        ),
        OutlineEntry(
            outline_index=1,
            level=2,
            title="1.1 Introduction",
            page_idx=0,
            section_path=("1\u00a0Overview", "1.1 Introduction"),
        ),
        OutlineEntry(
            outline_index=2,
            level=1,
            title="2 Installing in DSS Client",
            page_idx=1,
            section_path=("2 Installing in DSS Client",),
        ),
        OutlineEntry(
            outline_index=3,
            level=1,
            title="3 Missing Heading",
            page_idx=2,
            section_path=("3 Missing Heading",),
        ),
    ]

    result = section_content(content, entries)

    assert result.stats["outline_entry_count"] == 4
    assert result.stats["normalized_exact_y_matches"] == 1
    assert result.stats["exact_matches"] == 1
    assert result.stats["fuzzy_matches"] == 1
    assert result.stats["page_fallbacks"] == 1
    assert result.content_data[1]["section_title"] == "1\u00a0Overview"
    assert result.content_data[3]["section_path"] == ["1\u00a0Overview", "1.1 Introduction"]
    assert result.content_data[5]["section_source"] == "pdf_outline_fuzzy"
    assert result.content_data[8]["section_source"] == "pdf_outline_page_fallback"
    assert result.content_data[8]["section_confidence"] == 0.75


def test_sectioning_keeps_table_continuation_under_master_section():
    content = [
        {"type": "text", "page_idx": 0, "text": "1 Overview", "bbox": [10, 100, 200, 120]},
        {
            "type": "table",
            "page_idx": 0,
            "bbox": [100, 500, 900, 900],
            "table_caption": ["Table 1"],
            "table_body": "<table><tr><td>Port</td></tr></table>",
            "table_footnote": [],
        },
        {"type": "text", "page_idx": 1, "text": "2 Setup", "bbox": [10, 250, 200, 280]},
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
    entries = [
        OutlineEntry(
            outline_index=0,
            level=1,
            title="1 Overview",
            page_idx=0,
            section_path=("1 Overview",),
        ),
        OutlineEntry(
            outline_index=1,
            level=1,
            title="2 Setup",
            page_idx=1,
            section_path=("2 Setup",),
        ),
    ]

    result = section_content(content, entries)

    assert result.content_data[1]["section_path"] == ["1 Overview"]
    assert result.content_data[1][TABLE_CONTINUATION_INDICES_KEY] == [3]
    assert result.content_data[3]["section_path"] == ["1 Overview"]
    assert result.content_data[3][TABLE_CONTINUATION_MASTER_IDX_KEY] == 1
    assert result.stats["table_continuation_groups"] == 1
    assert result.stats["table_continuation_blocks"] == 1


def test_write_sectioned_json_reads_pdf_outline(tmp_path):
    fitz = __import__("fitz")
    pdf_path = tmp_path / "manual.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1 Overview")
    page.insert_text((72, 120), "Body")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "2 Setup")
    page2.insert_text((72, 120), "Setup body")
    doc.set_toc([[1, "1 Overview", 1], [1, "2 Setup", 2]])
    doc.save(pdf_path)
    doc.close()

    input_json = tmp_path / "manual_content_list.json"
    input_json.write_text(
        json.dumps(
            [
                {"type": "text", "page_idx": 0, "text": "1 Overview", "bbox": [100, 100, 300, 130]},
                {"type": "text", "page_idx": 0, "text": "Body"},
                {"type": "text", "page_idx": 1, "text": "2 Setup", "bbox": [100, 100, 300, 130]},
                {"type": "text", "page_idx": 1, "text": "Setup body"},
            ]
        ),
        encoding="utf-8",
    )
    output_json = sectioned_path_for(input_json)
    audit_json = sectioning_audit_path_for(input_json)

    result = write_sectioned_json(
        input_json=input_json,
        input_pdf=pdf_path,
        output_json=output_json,
        audit_json=audit_json,
    )

    assert result.stats["outline_entry_count"] == 2
    sectioned = json.loads(output_json.read_text(encoding="utf-8"))
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    assert sectioned[1]["section_path"] == ["1 Overview"]
    assert sectioned[1]["source_relpath"] == "manual.pdf"
    assert sectioned[1]["source_filename"] == "manual.pdf"
    assert sectioned[1]["breadcrumb"] == "manual.pdf > 1 Overview"
    assert sectioned[3]["section_path"] == ["2 Setup"]
    assert sectioned[3]["breadcrumb"] == "manual.pdf > 2 Setup"
    assert audit["source_name"] == "manual.pdf"
    assert audit["stats"]["section_event_count"] == 2


def test_sectioning_main_places_default_audit_next_to_custom_output(tmp_path, capsys):
    output_json = tmp_path / "out" / "manual_content_list_SECTIONED.json"

    main(
        [
            "--input-json",
            str(tmp_path / "manual_content_list.json"),
            "--input-pdf",
            str(tmp_path / "manual.pdf"),
            "--output-json",
            str(output_json),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert f"output_json: {output_json}" in output
    assert f"audit_json: {tmp_path / 'out' / 'manual_SECTIONING_AUDIT.json'}" in output
    assert "source_name: manual.pdf" in output


def test_sectioning_main_uses_manual_source_root(tmp_path, capsys):
    source_root = tmp_path / "pdfs"
    source_pdf = source_root / "DSS" / "manual.pdf"

    main(
        [
            "--input-json",
            str(tmp_path / "manual_content_list.json"),
            "--input-pdf",
            str(source_pdf),
            "--source-root",
            str(source_root),
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert "source_name: DSS/manual.pdf" in output
