from __future__ import annotations

from types import SimpleNamespace

from rag_flow.indexing import _page_payloads_from_chunks, point_id, uses_idf_modifier


def test_point_id_uses_chunk_id_when_present():
    page_id = point_id("manual.pdf", 3)
    first_chunk = point_id("manual.pdf", 3, chunk_id="manual-chunk-00001")
    second_chunk = point_id("manual.pdf", 3, chunk_id="manual-chunk-00002")

    assert page_id != first_chunk
    assert first_chunk != second_chunk


def test_page_payloads_from_chunks_carry_section_metadata():
    chunks = [
        {
            "chunk_content": "Section text",
            "metadata": {
                "source": "manual.pdf",
                "chunk_id": "manual-chunk-00001",
                "page_idx": 2,
                "page_indices": [2],
                "section_path": ["1 Overview", "1.1 Login"],
                "section_title": "1.1 Login",
                "section_level": 2,
                "section_source": "pdf_outline_exact",
            },
        }
    ]

    payloads = _page_payloads_from_chunks(chunks, source_name="manual.pdf")

    assert payloads[2]["section_path"] == ["1 Overview", "1.1 Login"]
    assert payloads[2]["section_title"] == "1.1 Login"
    assert payloads[2]["chunk_ids_on_page"] == ["manual-chunk-00001"]
    assert "Section text" in payloads[2]["chunk_content"]


def test_sparse_schema_requires_idf_modifier():
    fake_models = SimpleNamespace(Modifier=SimpleNamespace(IDF="idf"))

    assert uses_idf_modifier(SimpleNamespace(modifier="idf"), fake_models)
    assert not uses_idf_modifier(SimpleNamespace(modifier=None), fake_models)
    assert not uses_idf_modifier(SimpleNamespace(modifier="none"), fake_models)
