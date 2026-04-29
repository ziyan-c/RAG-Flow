import pytest

from rag_flow.preprocessing.small_icons import (
    _patch_field_keys,
    _window_visual_page_end,
    build_icon_patch_prompt,
    build_table_continuation_map,
    checkpoint_path_for,
    is_no_missing_response,
    resolve_icon_patch_batch,
    resolve_icon_patch_artifacts,
    should_apply_icon_patch,
)


def test_resolve_icon_patch_artifacts_from_mineru_output_dir(tmp_path):
    artifact_dir = tmp_path / "hybrid_auto"
    artifact_dir.mkdir()
    content_json = artifact_dir / "manual_content_list.json"
    origin_pdf = artifact_dir / "manual_origin.pdf"
    content_json.write_text("[]", encoding="utf-8")
    origin_pdf.write_text("pdf", encoding="utf-8")

    artifacts = resolve_icon_patch_artifacts(artifact_dir)

    assert artifacts.artifact_dir == artifact_dir
    assert artifacts.content_json == content_json
    assert artifacts.origin_pdf == origin_pdf
    assert artifacts.output_json == artifact_dir / "manual_content_list_PATCHED.json"


def test_resolve_icon_patch_artifacts_ignores_v2_content_list(tmp_path):
    artifact_dir = tmp_path / "hybrid_auto"
    artifact_dir.mkdir()
    (artifact_dir / "manual_content_list_v2.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "manual_origin.pdf").write_text("pdf", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        resolve_icon_patch_artifacts(artifact_dir)


def test_resolve_icon_patch_artifacts_rejects_ambiguous_content_lists(tmp_path):
    artifact_dir = tmp_path / "hybrid_auto"
    artifact_dir.mkdir()
    (artifact_dir / "a_content_list.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "b_content_list.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "a_origin.pdf").write_text("pdf", encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_icon_patch_artifacts(artifact_dir)


def test_resolve_icon_patch_batch_finds_artifacts_recursively(tmp_path):
    first = tmp_path / "manuals" / "a" / "auto"
    second = tmp_path / "manuals" / "nested" / "b" / "auto"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "a_content_list.json").write_text("[]", encoding="utf-8")
    (first / "a_origin.pdf").write_text("pdf", encoding="utf-8")
    (second / "b_content_list.json").write_text("[]", encoding="utf-8")
    (second / "b_origin.pdf").write_text("pdf", encoding="utf-8")

    artifacts = resolve_icon_patch_batch(tmp_path / "manuals")

    assert [item.content_json.name for item in artifacts] == ["a_content_list.json", "b_content_list.json"]


def test_checkpoint_path_uses_output_stem():
    assert checkpoint_path_for("/tmp/manual_content_list_PATCHED.json").name == (
        "manual_content_list_PATCHED.checkpoint.json"
    )


def test_patch_field_keys_skips_non_content_block_types():
    block = {
        "type": "header",
        "bbox": [0, 0, 1000, 50],
        "page_idx": 0,
        "text": "Alarm settings",
        "extra_ocr": ["Mode", "Status"],
    }

    assert _patch_field_keys(block) == []


def test_patch_field_keys_skips_empty_table_continuation_blocks():
    block = {
        "type": "table",
        "bbox": [171, 95, 905, 356],
        "page_idx": 20,
        "img_path": "",
        "table_caption": [],
        "table_footnote": [],
    }

    assert _patch_field_keys(block) == []


def test_table_continuations_group_under_previous_body_table():
    content_data = [
        {
            "type": "table",
            "bbox": [171, 497, 905, 782],
            "page_idx": 18,
            "img_path": "images/table.jpg",
            "table_caption": ["Table 2-2"],
            "table_footnote": [],
            "table_body": "<table><tr><td>1</td></tr></table>",
        },
        {"type": "header", "text": "User Manual", "bbox": [808, 66, 905, 78], "page_idx": 18},
        {
            "type": "table",
            "bbox": [171, 93, 905, 904],
            "page_idx": 19,
            "img_path": "",
            "table_caption": [],
            "table_footnote": [],
        },
        {
            "type": "table",
            "bbox": [171, 95, 905, 356],
            "page_idx": 20,
            "img_path": "",
            "table_caption": [],
            "table_footnote": [],
        },
        {
            "type": "table",
            "bbox": [171, 663, 905, 787],
            "page_idx": 20,
            "img_path": "images/second.jpg",
            "table_caption": ["Table 2-3"],
            "table_footnote": [],
            "table_body": "<table><tr><td>2</td></tr></table>",
        },
    ]

    assert build_table_continuation_map(content_data) == {0: [2, 3]}


def test_visual_page_window_extends_for_table_continuations():
    content_data = [
        {"type": "table", "bbox": [171, 497, 905, 782], "page_idx": 18, "table_body": "<table></table>"},
        {"type": "table", "bbox": [171, 93, 905, 904], "page_idx": 19},
        {"type": "table", "bbox": [171, 95, 905, 356], "page_idx": 20},
    ]

    assert (
        _window_visual_page_end(
            content_data=content_data,
            table_continuations={0: [1, 2]},
            page_start=0,
            page_end=18,
            max_page_idx=100,
        )
        == 20
    )


def test_table_icon_prompt_requires_html_preservation():
    prompt = build_icon_patch_prompt(
        original_text="<table><tr><td></td><td>View details.</td></tr></table>",
        field_key="table_body",
    )

    assert "Preserve the full HTML table structure" in prompt
    assert "Only insert `[Icon: ...]` tokens" in prompt


def test_table_icon_patch_accepts_icon_output_without_extra_validation():
    assert should_apply_icon_patch(
        original_text="<table><tr><td></td><td>View details.</td></tr></table>",
        patched_text="[Icon: eye] View details.",
        field_key="table_body",
    )


def test_table_icon_patch_accepts_html_output():
    assert should_apply_icon_patch(
        original_text="<table><tr><td></td><td>View details.</td></tr></table>",
        patched_text="<table><tr><td>[Icon: eye]</td><td>View details.</td></tr></table>",
        field_key="table_body",
    )


def test_no_missing_response_allows_simple_variants():
    assert is_no_missing_response("No missing.")
    assert is_no_missing_response("`No missing icons`")
    assert not should_apply_icon_patch(
        original_text="Open settings",
        patched_text="No missing icons.",
        field_key="text",
    )
