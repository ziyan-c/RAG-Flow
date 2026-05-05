from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from rag_flow.benchmark.captioning import (
    build_run_specs,
    collect_caption_candidates,
    generate_report,
    select_quality_samples,
    write_prepare_artifacts,
    write_run_worklist,
)


def test_captioning_context_token_specs_use_selected_image_side():
    args = SimpleNamespace(
        stage="context-tokens",
        repeat=None,
        selected_image_side=1536,
        context_tokens="0,2000,10000",
    )

    specs = build_run_specs(args, SimpleNamespace())

    assert [spec.max_context_tokens for spec in specs] == [0, 2000, 10000]
    assert {spec.max_image_side for spec in specs} == {1536}


def test_captioning_batch_specs_expand_concurrency_symbol():
    args = SimpleNamespace(
        stage="batch-size",
        repeat=None,
        selected_image_side=1536,
        selected_context_tokens=10000,
        selected_concurrency=6,
        batch_sizes="C,2C,4C,16,16,64",
    )

    specs = build_run_specs(args, SimpleNamespace())

    assert [spec.batch_size for spec in specs] == [6, 12, 16, 24, 64]


def test_prepare_artifacts_write_candidates_and_quality_templates(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    (tmp_path / "images").mkdir()
    Image.new("RGB", (320, 180), "white").save(tmp_path / "images" / "figure.png")
    content = [
        {
            "type": "image",
            "page_idx": 2,
            "img_path": "images/figure.png",
            "image_caption": ["Architecture overview"],
        }
    ]

    write_prepare_artifacts(
        content_data=content,
        base_dir=tmp_path,
        pdf_path=tmp_path / "missing.pdf",
        output_dir=tmp_path / "runs",
        quality_sample_size=80,
        negative_controls=10,
        seed=1,
        max_context_tokens=10000,
        no_captioning_view=True,
    )

    candidates = list(csv.DictReader((tmp_path / "runs" / "captioning_candidates.csv").open(encoding="utf-8")))
    scores = list(csv.DictReader((tmp_path / "runs" / "quality_scores_template.csv").open(encoding="utf-8")))
    assert candidates[0]["image_type"] == "architecture"
    assert candidates[0]["image_width"] == "320"
    assert scores[0]["sample_label"] == "quality_sample"
    assert scores[0]["context_sufficiency_score"] == ""


def test_write_run_worklist_makes_zero_model_context_empty(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "figure.png").write_text("not-an-image", encoding="utf-8")
    content = [
        {"type": "text", "page_idx": 0, "text": "Before context."},
        {"type": "image", "page_idx": 0, "img_path": "images/figure.png", "image_caption": ["Figure 1"]},
        {"type": "text", "page_idx": 0, "text": "After context."},
    ]

    write_run_worklist(
        content_data=content,
        base_dir=tmp_path,
        output_path=tmp_path / "worklist.jsonl",
        max_context_tokens=0,
        review_context_tokens=1000,
    )

    row = json.loads((tmp_path / "worklist.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["model_context_tokens"] == 0
    assert row["model_context_before"] == ""
    assert "Before context" in row["review_context_before"]


def test_captioning_report_writes_summary_and_chart(tmp_path):
    run_dir = tmp_path / "image-side" / "image-side_s1536_s1536_t10000_c1_b4_k1"
    run_dir.mkdir(parents=True)
    (run_dir / "run_params.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "stage": "image-side",
                "max_image_side": 1536,
                "max_context_tokens": 10000,
                "concurrency": 1,
                "batch_size": 4,
                "checkpoint_interval": 1,
                "repeat_index": 1,
                "candidate_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "request_count": 2,
                "ok_request_count": 2,
                "written_count": 2,
                "images_per_min": 12.5,
                "request_duration_p95_s": 1.2,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "quality_scores.csv").write_text(
        "sample_id,max_image_side,max_context_tokens,overall_quality_score,hallucinated,"
        "visual_coverage_score,small_text_readability_score,context_sufficiency_score,"
        "context_grounding_score,retrieval_usefulness_score,over_contextualized,"
        "missing_key_details,unfinished_output\n"
        "q0001,1536,10000,4,0,4,4,5,4,4,0,0,0\n",
        encoding="utf-8",
    )

    generate_report(tmp_path)

    assert (tmp_path / "report" / "benchmark_summary.csv").exists()
    assert (tmp_path / "report" / "quality_summary.csv").exists()
    assert (tmp_path / "report" / "report.typ").exists()
    assert (tmp_path / "report" / "charts" / "image_side_throughput.svg").exists()
    assert (tmp_path / "report" / "charts" / "quality_overall_by_image_side.svg").exists()
    quality_rows = list(csv.DictReader((tmp_path / "report" / "quality_summary.csv").open(encoding="utf-8")))
    assert quality_rows[0]["overall_quality_score_p10"] == "4.0"
    assert quality_rows[0]["overall_quality_score_p25"] == "4.0"


def test_quality_sampling_keeps_available_images_by_type():
    candidates = [
        {"block_idx": 1, "pdf_page": 1, "image_type": "ui_screenshot", "image_exists": True},
        {"block_idx": 2, "pdf_page": 2, "image_type": "flowchart", "image_exists": True},
        {"block_idx": 3, "pdf_page": 3, "image_type": "other_image", "image_exists": True},
        {"block_idx": 4, "pdf_page": 4, "image_type": "missing", "image_exists": False},
    ]

    samples = select_quality_samples(candidates, sample_size=10, negative_controls=1, seed=1)

    assert [sample["block_idx"] for sample in samples] == [1, 2, 3]
    assert samples[-1]["sample_label"] == "negative_control"
