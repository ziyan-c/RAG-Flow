from pathlib import Path

import pytest

from rag_flow.preprocessing.captioning_view import (
    captioning_view_path_for,
    collect_captioning_view_regions,
    main,
)


def test_captioning_view_path_removes_content_list_suffix():
    assert captioning_view_path_for(Path("/tmp/manual_content_list.json")).name == "manual_CAPTIONING_VIEW.pdf"
    assert captioning_view_path_for(Path("/tmp/manual_content_list_PATCHED.json")).name == (
        "manual_CAPTIONING_VIEW.pdf"
    )
    assert captioning_view_path_for(Path("/tmp/manual_content_list_PATCHED_CAPTIONED.json")).name == (
        "manual_CAPTIONING_VIEW.pdf"
    )
    assert captioning_view_path_for(Path("/tmp/manual_content_list_SECTIONED_PATCHED.json")).name == (
        "manual_CAPTIONING_VIEW.pdf"
    )


def test_collect_captioning_view_regions_includes_image_and_context_blocks():
    content_data = [
        {"type": "text", "bbox": [100, 100, 900, 150], "page_idx": 0, "text": "Before the figure."},
        {
            "type": "image",
            "bbox": [200, 200, 700, 500],
            "page_idx": 0,
            "img_path": "images/figure.jpg",
            "image_caption": ["Figure 1"],
        },
        {"type": "text", "bbox": [100, 560, 900, 610], "page_idx": 0, "text": "After the figure."},
    ]

    plan = collect_captioning_view_regions(content_data, max_context_tokens=200)

    assert plan.caption_targets == 1
    assert plan.field_counts["caption_target"] == 1
    assert plan.field_counts["context_before"] == 1
    assert plan.field_counts["context_current"] == 1
    assert plan.field_counts["context_after"] == 1


def test_collect_captioning_view_regions_skips_inline_icons():
    content_data = [
        {
            "type": "image",
            "bbox": [200, 200, 220, 220],
            "page_idx": 0,
            "img_path": "images/icon.jpg",
            "vlm-small-icon-inline-icon": True,
        }
    ]

    plan = collect_captioning_view_regions(content_data)

    assert plan.caption_targets == 0
    assert plan.regions == ()


def test_captioning_view_artifact_dir_dry_run_uses_patched_json(tmp_path, capsys):
    artifact_dir = tmp_path / "hybrid_auto"
    artifact_dir.mkdir()
    (artifact_dir / "manual_content_list.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "manual_content_list_PATCHED.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "manual_content_list_PATCHED_CAPTIONED.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "manual_origin.pdf").write_text("pdf", encoding="utf-8")

    main(["--artifact-dir", str(artifact_dir), "--dry-run"])

    output = capsys.readouterr().out
    assert f"input_json: {artifact_dir / 'manual_content_list_PATCHED.json'}" in output
    assert "manual_content_list_PATCHED_CAPTIONED.json" not in output


def test_captioning_view_artifact_dir_rejects_input_json(tmp_path):
    artifact_dir = tmp_path / "hybrid_auto"
    artifact_dir.mkdir()

    with pytest.raises(SystemExit):
        main(["--artifact-dir", str(artifact_dir), "--input-json", "manual_content_list_PATCHED.json"])
