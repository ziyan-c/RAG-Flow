from __future__ import annotations

from pathlib import Path

from rag_flow.source_paths import source_breadcrumb, source_name_for_pdf, source_payload_fields, source_root_from_input_path


def test_source_name_uses_configured_source_root(tmp_path):
    source_root = tmp_path / ".local/CUSTOM_DATA" / "pdfs" / "source"
    pdf = source_root / "DSS" / "DHI-DSS Pro-User's Manual.pdf"

    assert source_name_for_pdf(pdf, source_root=source_root) == "DSS/DHI-DSS Pro-User's Manual.pdf"


def test_source_name_detects_source_pdfs_ancestor(tmp_path):
    pdf = tmp_path / "source-pdfs" / "HAC-HF3805G" / "manual.pdf"

    assert source_name_for_pdf(pdf) == "HAC-HF3805G/manual.pdf"


def test_source_name_detects_pdfs_source_ancestor(tmp_path):
    pdf = tmp_path / "pdfs" / "source" / "DSS" / "manual.pdf"

    assert source_name_for_pdf(pdf) == "DSS/manual.pdf"


def test_source_name_falls_back_to_configured_source_name(tmp_path):
    pdf = tmp_path / "private" / "manual.pdf"

    assert (
        source_name_for_pdf(
            pdf,
            configured_source_pdf=pdf,
            configured_source_name="configured-manual.pdf",
        )
        == "configured-manual.pdf"
    )


def test_source_payload_fields_split_root_relative_source():
    fields = source_payload_fields(Path("DSS/manual.pdf"))

    assert fields == {
        "source_relpath": "DSS/manual.pdf",
        "source_filename": "manual.pdf",
        "breadcrumb": "DSS > manual.pdf",
    }


def test_source_breadcrumb_joins_source_relpath_and_section_path():
    assert source_breadcrumb("DSS/manual.pdf", ["1 Overview", "1.1 Login"]) == (
        "DSS > manual.pdf > 1 Overview > 1.1 Login"
    )


def test_source_root_from_input_path_ignores_single_pdf():
    assert source_root_from_input_path(".local/CUSTOM_DATA/pdfs/source") == Path(".local/CUSTOM_DATA/pdfs/source")
    assert source_root_from_input_path(".local/CUSTOM_DATA/pdfs/source/manual.pdf") is None
