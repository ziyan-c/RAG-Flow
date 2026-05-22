from __future__ import annotations

import json

from rag_flow.benchmark.evidence_remap import build_evidence_anchors, remap_query_set_to_chunks
from rag_flow.benchmark.retrieval import (
    PILOT_QUERIES,
    generate_full_query_set_from_chunks,
    generate_pdf_grounded_full_query_set,
    run_retrieval_benchmark,
    score_query_result,
    write_full_query_set,
    write_pilot_query_set,
)


def test_write_pilot_query_set_creates_twenty_queries(tmp_path):
    output = tmp_path / "pilot.jsonl"

    write_pilot_query_set(output)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    assert len(PILOT_QUERIES) == 20
    assert rows[0]["query_id"] == "pilot-001"
    assert rows[0]["gold_chunk_ids"]
    assert rows[0]["gold_page_indices"]


def _sample_chunk(chunk_id: str, chunk_type: str, page_idx: int) -> dict:
    if chunk_type == "table":
        content = "[Table caption: Alarm parameter description]\n<table><tr><td>Parameter</td></tr></table>"
        section = "4.1 Alarm Parameters"
    elif chunk_type == "image_ui":
        content = "[Image caption: Figure 4-1 Add device]\n[Image VLM description: The interface shows the Add Device dialog.]"
        section = "4.2 Add Device"
    elif chunk_type == "operation":
        content = "Step 1 Open the client.\nStep 2 Click Save."
        section = "4.3 Configuring Rules"
    elif chunk_type == "mixed":
        content = "[Image caption: Figure 4-2 Link camera]\nStep 1 Select a camera.\nStep 2 Click OK."
        section = "4.4 Adding Linkage"
    else:
        content = "This section describes how the platform organizes monitoring resources and permissions."
        section = "4.5 Resource Permissions"
    return {
        "chunk_content": f"[Section: 4 Businesses > {section}]\n\n{section}\n\n{content}",
        "metadata": {
            "chunk_id": chunk_id,
            "page_indices": [page_idx],
            "section_path": ["4 Businesses", section],
            "section_title": section,
        },
    }


def test_generate_full_query_set_from_chunks_balances_query_types(tmp_path):
    chunks = []
    kinds = ["text", "operation", "table", "image_ui", "mixed"]
    for idx in range(20):
        chunks.append(_sample_chunk(f"chunk-{idx:03d}", kinds[idx % len(kinds)], page_idx=20 + idx))
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

    rows = generate_full_query_set_from_chunks(chunks_path, target_size=10)

    assert len(rows) == 10
    assert {row["query_type"] for row in rows} >= {"text", "operation", "table", "image_ui", "mixed"}
    assert all(row["gold_chunk_ids"] for row in rows)
    assert all(row["gold_page_indices"] for row in rows)


def test_write_full_query_set_writes_jsonl(tmp_path):
    chunks = [_sample_chunk(f"chunk-{idx:03d}", "text", page_idx=20 + idx) for idx in range(5)]
    chunks_path = tmp_path / "chunks.json"
    output = tmp_path / "full.jsonl"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

    write_full_query_set(chunks_path, output, target_size=5)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert rows[0]["query_id"] == "full-001"


def test_generate_pdf_grounded_full_query_set_includes_visual_page_rows(tmp_path):
    chunks = [_sample_chunk(f"chunk-{idx:03d}", "text", page_idx=20 + idx) for idx in range(6)]
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
    captioned = [
        {
            "type": "image",
            "page_idx": 22,
            "bbox": [10, 20, 100, 160],
            "image_caption": "Figure 3-1 Device list",
            "image_description_vlm": "The interface shows a device list with action buttons.",
            "section_path": ["3 Basic Configurations", "3.1 Managing Devices"],
        }
    ]
    captioned_path = tmp_path / "captioned.json"
    captioned_path.write_text(json.dumps(captioned), encoding="utf-8")

    rows = generate_pdf_grounded_full_query_set(
        chunks_path,
        captioned_json_path=captioned_path,
        target_size=6,
        visual_page_count=1,
    )

    visual_rows = [row for row in rows if row["query_type"] == "visual_page"]
    assert len(rows) == 6
    assert len(visual_rows) == 1
    assert visual_rows[0]["gold_page_indices"] == [22]
    assert visual_rows[0]["gold_chunk_ids"] == []
    assert visual_rows[0]["requires_pdf_review"] is True


def test_score_query_result_prefers_gold_chunk_rank():
    query = {
        "query_id": "q1",
        "gold_page_indices": [1],
        "gold_chunk_ids": ["chunk-b"],
    }
    response = {
        "hit_page": 2,
        "all_hits": [
            {"rank": 1, "page_idx": 1, "chunk_id": "chunk-a"},
            {"rank": 2, "page_idx": 1, "chunk_id": "chunk-b"},
        ],
    }

    row = score_query_result(query, response, recall_ks=(1, 2))

    assert row["first_correct_rank"] == 2
    assert row["page_recall@1"] == 1
    assert row["chunk_recall@1"] == 0
    assert row["recall@1"] == 0
    assert row["chunk_recall@2"] == 1
    assert row["recall@2"] == 1


def test_score_query_result_falls_back_to_page_when_no_gold_chunk():
    query = {"query_id": "q1", "gold_page_numbers": [5]}
    response = {
        "all_hits": [
            {"rank": 1, "page_idx": 2, "chunk_id": "chunk-a"},
            {"rank": 2, "page_idx": 4, "chunk_id": "chunk-b"},
        ],
    }

    row = score_query_result(query, response, recall_ks=(1, 2))

    assert row["first_correct_rank"] == 2
    assert row["page_recall@1"] == 0
    assert row["page_recall@2"] == 1
    assert row["recall@2"] == 1


def test_evidence_anchors_remap_gold_chunks_after_rechunking(tmp_path):
    original_chunks = [
        {
            "chunk_content": "alpha beta",
            "metadata": {"chunk_id": "manual-chunk-00000", "block_indices": [1, 2], "page_indices": [0]},
        },
        {
            "chunk_content": "gamma",
            "metadata": {"chunk_id": "manual-chunk-00001", "block_indices": [3], "page_indices": [1]},
        },
    ]
    new_chunks = [
        {
            "chunk_content": "alpha",
            "metadata": {"chunk_id": "manual-chunk-00000", "block_indices": [1], "page_indices": [0]},
        },
        {
            "chunk_content": "beta gamma",
            "metadata": {"chunk_id": "manual-chunk-00001", "block_indices": [2, 3], "page_indices": [0, 1]},
        },
    ]
    query_set = tmp_path / "queries.jsonl"
    query_set.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "Where are alpha and beta described?",
                "gold_chunk_ids": ["manual-chunk-00000"],
                "primary_gold_chunk_id": "manual-chunk-00000",
                "gold_page_indices": [0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    original_chunks_path = tmp_path / "original_chunks.json"
    new_chunks_path = tmp_path / "new_chunks.json"
    original_chunks_path.write_text(json.dumps(original_chunks), encoding="utf-8")
    new_chunks_path.write_text(json.dumps(new_chunks), encoding="utf-8")

    anchored_path = tmp_path / "anchored.jsonl"
    remapped_path = tmp_path / "remapped.jsonl"
    build_evidence_anchors(query_set=query_set, chunks_path=original_chunks_path, output=anchored_path)
    remap_query_set_to_chunks(query_set=anchored_path, chunks_path=new_chunks_path, output=remapped_path)

    row = json.loads(remapped_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["evidence_block_indices"] == [1, 2]
    assert row["gold_chunk_ids"] == ["manual-chunk-00000", "manual-chunk-00001"]
    assert row["primary_gold_chunk_id"] == "manual-chunk-00000"
    assert row["gold_remap_source"] == "block_overlap"


def test_evidence_remap_keeps_page_only_queries_as_page_gold(tmp_path):
    chunks = [
        {"chunk_content": "page five", "metadata": {"chunk_id": "chunk-a", "block_indices": [10], "page_indices": [4]}}
    ]
    query_set = tmp_path / "queries.jsonl"
    query_set.write_text(json.dumps({"query_id": "q1", "query": "What is on page five?", "gold_page_numbers": [5]}) + "\n")
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

    anchored_path = tmp_path / "anchored.jsonl"
    remapped_path = tmp_path / "remapped.jsonl"
    build_evidence_anchors(query_set=query_set, chunks_path=chunks_path, output=anchored_path)
    remap_query_set_to_chunks(query_set=anchored_path, chunks_path=chunks_path, output=remapped_path)

    row = json.loads(remapped_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["gold_chunk_ids"] == []
    assert row["gold_page_indices"] == [4]
    assert row["gold_remap_warning"] == "page_only_gold"


def test_run_retrieval_benchmark_dry_run_prints_plan(tmp_path, capsys):
    query_set = tmp_path / "queries.jsonl"
    query_set.write_text(json.dumps({"query_id": "q1", "query": "hello"}) + "\n", encoding="utf-8")

    run_dir = run_retrieval_benchmark(
        query_set=query_set,
        output_dir=tmp_path / "runs",
        url="http://127.0.0.1:8000/retrieve",
        run_id="dry",
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert str(run_dir).endswith("runs/dry")
    assert "Queries: 1" in output
    assert "Retriever URL: http://127.0.0.1:8000/retrieve" in output
