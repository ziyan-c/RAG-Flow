from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from rag_flow.benchmark.patching import (
    build_run_specs,
    build_subset,
    generate_report,
    page_indices_for_stage,
    parse_batch_size_spec,
    parse_pdf_pages,
    should_stop_concurrency_search,
    write_cross_page_table_samples,
    write_quality_review_worklist,
    write_quality_review_samples,
    write_quality_scoring_template,
)


def test_parse_pdf_pages_uses_reader_page_numbers():
    assert parse_pdf_pages("1,3-5") == {0, 2, 3, 4}


def test_parse_batch_size_spec_expands_concurrency_symbol():
    assert parse_batch_size_spec("C,2C,4C,140,140,512", concurrency=20) == [20, 40, 80, 140, 512]


def test_build_subset_keeps_inline_icon_dependency():
    content_data = [
        {
            "type": "image",
            "bbox": [240, 105, 260, 125],
            "page_idx": 49,
            "image_caption": [],
            "image_footnote": [],
        },
        {"type": "text", "bbox": [100, 100, 230, 130], "page_idx": 49, "text": "Click ."},
        {"type": "text", "bbox": [100, 200, 230, 230], "page_idx": 50, "text": "Other page."},
    ]

    subset, pages, indices = build_subset(content_data, page_indices={49})

    assert indices == [0, 1]
    assert pages == {49}
    assert [block["rag-flow-benchmark-original-index"] for block in subset] == [0, 1]


def test_build_subset_keeps_table_continuation_dependency():
    content_data = [
        {
            "type": "table",
            "bbox": [100, 100, 500, 700],
            "page_idx": 9,
            "table_body": "<table><tr><td>Icon</td></tr></table>",
        },
        {
            "type": "table",
            "bbox": [100, 80, 500, 300],
            "page_idx": 10,
            "table_body": "",
            "table_caption": [],
            "table_footnote": [],
        },
    ]

    subset, pages, indices = build_subset(content_data, page_indices={9})

    assert indices == [0, 1]
    assert pages == {9, 10}
    assert [block["rag-flow-benchmark-original-index"] for block in subset] == [0, 1]


def test_quality_template_adds_negative_controls(tmp_path):
    content_data = [
        {"type": "text", "bbox": [0, 0, 100, 20], "page_idx": 51, "text": "Click ."},
        {"type": "text", "bbox": [0, 30, 100, 50], "page_idx": 60, "text": "Plain description."},
    ]
    output = tmp_path / "quality.csv"

    write_quality_scoring_template(
        content_data=content_data,
        main_pages={60},
        quality_pages={51},
        output_csv=output,
        negative_controls=1,
    )

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert [row["sample_label"] for row in rows] == ["quality_page", "negative_control"]


def test_cross_page_table_samples_are_written(tmp_path):
    content_data = [
        {
            "type": "table",
            "bbox": [100, 100, 500, 700],
            "page_idx": 9,
            "table_body": "<table><tr><td>Icon</td></tr></table>",
            "table_caption": ["Table A"],
        },
        {
            "type": "table",
            "bbox": [100, 80, 500, 300],
            "page_idx": 10,
            "table_body": "",
            "table_caption": [],
            "table_footnote": [],
        },
    ]
    output = tmp_path / "tables.csv"

    write_cross_page_table_samples(content_data=content_data, output_csv=output)

    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows[0]["master_block_idx"] == "0"
    assert rows[0]["continuation_block_indices"] == "1"


def test_run_specs_require_selected_values_for_later_stages():
    args = SimpleNamespace(
        stage="batch-size",
        repeat=None,
        selected_dpi=250,
        selected_concurrency=20,
        selected_batch_size=None,
        selected_checkpoint_interval=None,
        batch_sizes="C,2C,4C,140,256,512",
    )
    config = SimpleNamespace()

    specs = build_run_specs(args, config)

    assert [spec.batch_size for spec in specs] == [20, 40, 80, 140, 256, 512]


def test_dpi_confirm_uses_dpi_list():
    args = SimpleNamespace(
        stage="dpi-confirm",
        repeat=None,
        dpis="200,250,300",
        selected_concurrency=6,
        selected_batch_size=140,
        selected_checkpoint_interval=10,
    )
    config = SimpleNamespace()

    specs = build_run_specs(args, config)

    assert [spec.dpi for spec in specs] == [200, 250, 300]
    assert {spec.concurrency for spec in specs} == {6}
    assert {spec.batch_size for spec in specs} == {140}
    assert {spec.checkpoint_interval for spec in specs} == {10}


def test_prepare_artifacts_json_round_trip(tmp_path):
    data = [{"type": "text", "bbox": [0, 0, 100, 20], "page_idx": 0, "text": "Click ."}]
    path = tmp_path / "content.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert json.loads(path.read_text(encoding="utf-8")) == data


def test_page_indices_for_stage_separates_quality_pages():
    main_pages = {49, 50}
    quality_pages = {79, 80}

    assert page_indices_for_stage(
        stage="dpi",
        main_pages=main_pages,
        quality_pages=quality_pages,
        include_quality_pages=False,
    ) == main_pages
    assert page_indices_for_stage(
        stage="quality",
        main_pages=main_pages,
        quality_pages=quality_pages,
        include_quality_pages=False,
    ) == quality_pages
    assert page_indices_for_stage(
        stage="dpi",
        main_pages=main_pages,
        quality_pages=quality_pages,
        include_quality_pages=True,
    ) == main_pages | quality_pages


def test_concurrency_auto_stop_detects_throughput_drop():
    should_stop, reason = should_stop_concurrency_search(
        summary={"request_count": 10, "timeout_count": 0, "error_request_count": 0, "fields_per_min": 70, "request_duration_p95_s": 3},
        best_summary={"fields_per_min": 100, "request_duration_p95_s": 2},
        timeout_ratio=0.02,
        throughput_drop_ratio=0.15,
        p95_ratio=2.0,
    )

    assert should_stop
    assert "fields/min" in reason


def test_quality_review_samples_write_score_template(tmp_path):
    content_data = [
        {"type": "text", "bbox": [0, 0, 100, 20], "page_idx": 51, "text": "Click ."},
        {"type": "text", "bbox": [0, 30, 100, 50], "page_idx": 60, "text": "Plain description."},
    ]

    write_quality_review_samples(
        content_data=content_data,
        main_pages={60},
        quality_pages={51},
        output_csv=tmp_path / "review.csv",
        score_template_csv=tmp_path / "scores.csv",
        negative_controls=1,
        sample_size=1,
        dpis=[200, 250],
    )

    review_rows = list(csv.DictReader((tmp_path / "review.csv").open(encoding="utf-8")))
    score_rows = list(csv.DictReader((tmp_path / "scores.csv").open(encoding="utf-8")))
    assert len(review_rows) == 2
    assert {row["dpi"] for row in score_rows} == {"200", "250"}


def test_generate_report_writes_summary_and_charts(tmp_path):
    run_dir = tmp_path / "dpi" / "dpi_dpi250_d250_c3_b9_k30"
    run_dir.mkdir(parents=True)
    (run_dir / "run_params.json").write_text(
        json.dumps(
            {
                "run_id": "dpi_dpi250_d250_c3_b9_k30",
                "stage": "dpi",
                "dpi": 250,
                "concurrency": 3,
                "batch_size": 9,
                "checkpoint_interval": 30,
                "repeat_index": 1,
                "subset_blocks": 10,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "request_count": 4,
                "ok_request_count": 4,
                "fields_per_min": 12.5,
                "request_duration_p95_s": 1.2,
                "render_duration_total_s": 3.4,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "quality_scores.csv").write_text(
        "dpi,target_icons,strict_hits,wrong_icons,missed_targets,false_positives\n"
        "250,10,8,1,2,0\n",
        encoding="utf-8",
    )

    generate_report(tmp_path, main_pages="50-250", quality_pages="52,76")

    assert (tmp_path / "report" / "benchmark_summary.csv").exists()
    assert (tmp_path / "report" / "quality_summary.csv").exists()
    assert (tmp_path / "report" / "report.typ").exists()
    assert (tmp_path / "report" / "charts" / "dpi_throughput.svg").exists()
    assert (tmp_path / "report" / "charts" / "quality_strict_hit_ratio.svg").exists()


def test_quality_review_worklist_merges_quality_outputs(tmp_path):
    (tmp_path / "quality_review_samples.csv").write_text(
        "sample_id,sample_label,pdf_page,page_idx,block_idx,field,block_type,crop_path,text_preview,review_notes\n"
        "q0001,quality_page,52,51,10,text,text,quality-review-crops/q0001.png,Click ., \n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "quality" / "quality_dpi250_d250_c3_b9_k30"
    run_dir.mkdir(parents=True)
    input_json = run_dir / "input_content_list.json"
    output_json = run_dir / "output_content_list_PATCHED.json"
    input_json.write_text(
        json.dumps([{"rag-flow-benchmark-original-index": 10, "type": "text", "text": "Click ."}]),
        encoding="utf-8",
    )
    output_json.write_text(
        json.dumps([{"rag-flow-benchmark-original-index": 10, "type": "text", "text": "Click [Icon: camera]."}]),
        encoding="utf-8",
    )
    (run_dir / "run_params.json").write_text(
        json.dumps(
            {
                "run_id": "quality_dpi250_d250_c3_b9_k30",
                "stage": "quality",
                "dpi": 250,
                "input_json": str(input_json),
                "output_json": str(output_json),
            }
        ),
        encoding="utf-8",
    )
    report_dir = tmp_path / "report"
    report_dir.mkdir()

    worklist = write_quality_review_worklist(tmp_path, report_dir)

    assert worklist == report_dir / "quality_review_worklist.csv"
    rows = list(csv.DictReader(worklist.open(encoding="utf-8")))
    assert rows[0]["original_text"] == "Click ."
    assert rows[0]["patched_text"] == "Click [Icon: camera]."
