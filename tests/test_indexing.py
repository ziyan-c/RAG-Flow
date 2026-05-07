from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_flow.indexing import (
    PAYLOAD_INDEX_SPECS,
    _delete_existing_points_for_sources,
    _page_batches,
    _page_payloads_from_chunks,
    _visual_page_payload,
    point_id,
    uses_idf_modifier,
    visual_point_id,
)


def test_point_id_uses_chunk_id_when_present():
    page_id = point_id("manual.pdf", 3)
    first_chunk = point_id("manual.pdf", 3, chunk_id="manual-chunk-00001")
    second_chunk = point_id("manual.pdf", 3, chunk_id="manual-chunk-00002")
    visual_page = visual_point_id("manual.pdf", 3)

    assert page_id != first_chunk
    assert first_chunk != second_chunk
    assert visual_page not in {page_id, first_chunk, second_chunk}


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
    assert "chunk_content" not in payloads[2]


def test_visual_page_payload_is_page_only_payload():
    payload = _visual_page_payload({}, source_name="manual.pdf", page_idx=5)

    assert payload["source"] == "manual.pdf"
    assert payload["page_idx"] == 5
    assert payload["page_start"] == 5
    assert payload["page_end"] == 5
    assert payload["page_indices"] == [5]
    assert payload["is_visual_page"] is True
    assert payload["chunk_ids_on_page"] == []
    assert "chunk_content" not in payload
    assert "parent_page_idx" not in payload
    assert "is_table_continuation" not in payload


def test_page_batches_use_pdf_one_based_ranges():
    assert _page_batches(10, 4) == [(0, 1, 4), (4, 5, 8), (8, 9, 10)]


def test_page_batches_reject_non_positive_batch_size():
    try:
        _page_batches(10, 0)
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("Expected batch_size validation to fail")


def test_sparse_schema_requires_idf_modifier():
    fake_models = SimpleNamespace(Modifier=SimpleNamespace(IDF="idf"))

    assert uses_idf_modifier(SimpleNamespace(modifier="idf"), fake_models)
    assert not uses_idf_modifier(SimpleNamespace(modifier=None), fake_models)
    assert not uses_idf_modifier(SimpleNamespace(modifier="none"), fake_models)


def test_page_indices_payload_index_is_declared():
    assert ("page_indices", "integer") in PAYLOAD_INDEX_SPECS


def test_delete_existing_points_for_sources_separates_text_and_visual(tmp_path):
    qdrant_client = pytest.importorskip("qdrant_client")
    QdrantClient = qdrant_client.QdrantClient
    models = qdrant_client.models

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection(
        collection_name="c",
        vectors_config={"v": models.VectorParams(size=2, distance=models.Distance.COSINE)},
    )
    client.upsert(
        collection_name="c",
        points=[
            models.PointStruct(
                id=1,
                vector={"v": [1.0, 0.0]},
                payload={"source": "manual.pdf", "chunk_id": "manual-chunk-00001"},
            ),
            models.PointStruct(
                id=2,
                vector={"v": [0.0, 1.0]},
                payload={"source": "manual.pdf", "is_visual_page": True},
            ),
        ],
    )

    _delete_existing_points_for_sources(
        client,
        "c",
        models,
        source_names={"manual.pdf"},
        visual=False,
    )
    records, _ = client.scroll(collection_name="c", limit=10, with_payload=True)
    assert [record.id for record in records] == [2]

    _delete_existing_points_for_sources(
        client,
        "c",
        models,
        source_names={"manual.pdf"},
        visual=True,
    )
    records, _ = client.scroll(collection_name="c", limit=10, with_payload=True)
    assert records == []
