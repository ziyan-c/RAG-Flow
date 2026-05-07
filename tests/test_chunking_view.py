from pathlib import Path

from rag_flow.preprocessing.chunking_view import (
    CHUNK_COLORS,
    chunking_view_path_for,
    collect_chunking_view_regions,
    color_for_chunk,
)


def test_chunking_view_path_removes_chunk_suffixes():
    assert (
        chunking_view_path_for(Path("/tmp/manual_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED.json")).name
        == "manual_CHUNKING_VIEW.pdf"
    )
    assert chunking_view_path_for(Path("/tmp/manual_page_level_chunks.json")).name == "manual_CHUNKING_VIEW.pdf"
    assert chunking_view_path_for(Path("/tmp/manual_chunks.json")).name == "manual_CHUNKING_VIEW.pdf"


def test_color_for_chunk_alternates_adjacent_chunks():
    assert color_for_chunk(0) != color_for_chunk(1)
    assert color_for_chunk(1) != color_for_chunk(2)
    assert color_for_chunk(0) == color_for_chunk(len(CHUNK_COLORS))


def test_collect_chunking_view_regions_uses_chunk_metadata_bboxes():
    chunks = [
        {
            "chunk_content": "First chunk",
            "metadata": {
                "chunk_idx": 0,
                "chunk_id": "manual-chunk-00000",
                "token_count": 12,
                "section_title": "Overview",
                "bboxes_by_page": {
                    "0": [[10, 10, 100, 30], [10, 40, 100, 60]],
                    "1": [[20, 20, 120, 50]],
                },
            },
        },
        {
            "chunk_content": "Second chunk",
            "metadata": {
                "chunk_idx": 1,
                "chunk_id": "manual-chunk-00001",
                "token_count": 8,
                "bboxes_by_page": {"1": [[140, 20, 260, 50]]},
            },
        },
        {
            "chunk_content": "No bbox chunk",
            "metadata": {
                "chunk_idx": 2,
                "chunk_id": "manual-chunk-00002",
            },
        },
    ]

    plan = collect_chunking_view_regions(chunks)

    assert plan.chunk_count == 3
    assert plan.chunks_with_regions == 2
    assert plan.pages_with_regions == 2
    assert plan.regions_by_chunk == {0: 3, 1: 1}
    assert len(plan.regions) == 4
    assert plan.regions[0].label == "chunk 0 / 12 tok"
    assert plan.regions[-1].chunk_idx == 1
