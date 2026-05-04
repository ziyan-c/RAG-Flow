from types import SimpleNamespace

import pytest

from rag_flow.preprocessing.small_icons import (
    INLINE_ICON_CANDIDATE_KEY,
    INLINE_ICON_KEY,
    InlineIconLink,
    _patch_field_keys,
    _window_visual_page_end,
    build_icon_patch_prompt,
    build_icon_patch_retry_prompt,
    build_table_footnote_crop,
    build_inline_icon_links,
    build_table_continuation_map,
    checkpoint_path_for,
    crop_image_from_block_with_inline_icons,
    fallback_only_icon_output,
    image_to_data_url,
    iter_icon_patch_results,
    is_inline_icon_candidate,
    is_no_missing_response,
    request_icon_patch_from_llm,
    resolve_icon_patch_batch,
    resolve_icon_patch_artifacts,
    should_apply_icon_patch,
    strip_reasoning_text,
)
from rag_flow.preprocessing.image_descriptions import should_caption_image_block


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


def test_inline_icon_candidate_requires_small_uncaptioned_image():
    block = {
        "type": "image",
        "bbox": [590, 550, 616, 565],
        "page_idx": 20,
        "image_caption": [],
        "image_footnote": [],
    }

    assert is_inline_icon_candidate(block)
    assert not is_inline_icon_candidate({**block, "bbox": [100, 100, 500, 500]})
    assert not is_inline_icon_candidate({**block, "image_caption": ["Figure 1"]})


def test_inline_icon_links_to_nearby_missing_click_text():
    content_data = [
        {
            "type": "image",
            "bbox": [590, 550, 616, 565],
            "page_idx": 20,
            "image_caption": [],
            "image_footnote": [],
        },
        {
            "type": "text",
            "bbox": [170, 540, 660, 575],
            "page_idx": 20,
            "text": "Press the Windows key, type dxdiag, and then click .",
        },
    ]

    links = build_inline_icon_links(content_data, {})

    assert links.by_icon[0].target_idx == 1
    assert links.by_icon[0].target_field == "text"
    assert content_data[0][INLINE_ICON_KEY] is True


def test_inline_icon_inside_table_links_to_table_body():
    content_data = [
        {
            "type": "table",
            "bbox": [100, 100, 900, 900],
            "page_idx": 0,
            "table_body": "<table><tr><td>Mode</td></tr></table>",
        },
        {
            "type": "image",
            "bbox": [200, 300, 222, 322],
            "page_idx": 0,
            "image_caption": [],
            "image_footnote": [],
        },
    ]

    links = build_inline_icon_links(content_data, {})

    assert links.by_icon[1].target_idx == 0
    assert links.by_icon[1].target_field == "table_body"
    assert content_data[1][INLINE_ICON_KEY] is True


def test_unlinked_inline_icon_candidate_is_marked_for_caption_skip():
    content_data = [
        {
            "type": "image",
            "bbox": [20, 20, 40, 40],
            "page_idx": 0,
            "image_caption": [],
            "image_footnote": [],
        }
    ]

    links = build_inline_icon_links(content_data, {})

    assert links.by_icon == {}
    assert content_data[0][INLINE_ICON_CANDIDATE_KEY] is True


def test_text_crop_expands_to_linked_inline_icon():
    class FakePage:
        width = 1000
        height = 1000

        def crop(self, box):
            return box

    content_data = [
        {"type": "text", "bbox": [100, 100, 200, 120], "page_idx": 0, "text": "Click ."},
        {"type": "image", "bbox": [240, 105, 260, 125], "page_idx": 0},
    ]
    image = crop_image_from_block_with_inline_icons(
        block=content_data[0],
        content_data=content_data,
        inline_icon_links=[
            InlineIconLink(
                icon_idx=1,
                target_idx=0,
                target_field="text",
                target_type="text",
                score=0,
            )
        ],
        pdf_images=[FakePage()],
    )

    assert image == (100.0, 100.0, 260.0, 125.0)


def test_table_footnote_crop_uses_table_width_with_padding():
    class FakePage:
        width = 1000
        height = 1000

        def crop(self, box):
            return box

    content_data = [
        {
            "type": "table",
            "bbox": [171, 502, 905, 656],
            "page_idx": 0,
            "table_caption": ["Table 8-10"],
            "table_body": "<table><tr><td>Map Flash</td></tr></table>",
            "table_footnote": ["Step 3 Click Save."],
        },
        {
            "type": "text",
            "bbox": [89, 690, 594, 715],
            "page_idx": 0,
            "text": "8.7.6 Configure File Storage Settings",
        },
    ]

    crop = build_table_footnote_crop(
        content_data=content_data,
        pdf_images=[FakePage()],
        block_idx=0,
    )

    assert crop == (159.0, 656.0, 917.0, 690.0)


def test_captioning_skips_inline_icon_blocks():
    assert not should_caption_image_block(
        {"type": "image", "img_path": "images/icon.jpg", INLINE_ICON_KEY: True}
    )
    assert not should_caption_image_block(
        {"type": "image", "img_path": "images/icon.jpg", INLINE_ICON_CANDIDATE_KEY: True}
    )
    assert should_caption_image_block({"type": "image", "img_path": "images/screenshot.jpg"})


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

    assert "Keep the complete original table content" in prompt
    assert "Insert each missing icon at its missing position" in prompt
    assert "Return only the complete patched table content, including inserted icons" in prompt
    assert "Use `[Icon: name]` for inserted icons" in prompt
    assert "return exactly without any other text or icons: NO_MISSING_SAFE" in prompt


def test_text_icon_prompt_requires_icon_format():
    prompt = build_icon_patch_prompt(
        original_text="Click .",
        field_key="text",
    )

    assert "Keep the complete original text" in prompt
    assert "Insert each missing icon at its missing position" in prompt
    assert "Return only the complete patched text, including inserted icons" in prompt
    assert "Use `[Icon: name]` for inserted icons" in prompt
    assert "return exactly without any other text or icons: NO_MISSING_SAFE" in prompt


def test_retry_prompt_explains_only_icon_failure():
    prompt = build_icon_patch_retry_prompt(prompt="Base prompt", invalid_reason="only icon output")

    assert "Base prompt" in prompt
    assert "dropped the original text" in prompt
    assert "Return only the complete patched content" in prompt


def test_patch_field_keys_skips_table_caption_by_default():
    block = {
        "type": "table",
        "bbox": [171, 502, 905, 656],
        "page_idx": 0,
        "table_caption": ["Table 8-10"],
        "table_body": "<table><tr><td>Map Flash</td></tr></table>",
        "table_footnote": ["Step 3 Click Save."],
    }

    assert _patch_field_keys(block) == ["table_footnote", "table_body"]


def test_patch_field_keys_patches_image_footnote_but_not_caption():
    block = {
        "type": "image",
        "bbox": [169, 458, 899, 871],
        "page_idx": 79,
        "img_path": "images/disk.jpg",
        "image_caption": ["Figure 3-26 Format a disk"],
        "image_footnote": ["Step 5 Select a disk, and then click ."],
        "image_description_vlm": "A disk management screenshot.",
    }

    assert _patch_field_keys(block) == ["image_footnote"]


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
    assert is_no_missing_response("NO_MISSING_SAFE")
    assert is_no_missing_response("Explanation... NO_MISSING_SAFE [Icon: name]")
    assert is_no_missing_response("No missing.")
    assert is_no_missing_response("`No missing icons`")
    assert not should_apply_icon_patch(
        original_text="Open settings",
        patched_text="NO_MISSING_SAFE [Icon: name]",
        field_key="text",
    )


def test_icon_patch_rejects_only_icon_output_for_text():
    assert not should_apply_icon_patch(
        original_text="Step 7 Click and select a target.",
        patched_text="[Icon: search]",
        field_key="text",
    )
    assert should_apply_icon_patch(
        original_text="Step 7 Click and select a target.",
        patched_text="Step 7 Click [Icon: search] and select a target.",
        field_key="text",
    )


def test_icon_patch_accepts_non_icon_marker_output():
    assert should_apply_icon_patch(
        original_text="Step 7 Click and select a target.",
        patched_text="Step 7 Click the search icon and select a target.",
        field_key="text",
    )


def test_only_icon_fallback_inserts_icon_before_punctuation_gap():
    assert fallback_only_icon_output(original_text="Click .", icon_only_output="[Icon: search]") == (
        "Click [Icon: search]."
    )


def test_only_icon_fallback_inserts_icon_into_inner_gap():
    assert fallback_only_icon_output(original_text="Select  option", icon_only_output="[Icon: gear]") == (
        "Select [Icon: gear] option"
    )


def test_only_icon_fallback_appends_when_no_gap_is_obvious():
    assert fallback_only_icon_output(original_text="Click Search.", icon_only_output="[icon: search]") == (
        "Click Search. [Icon: search]"
    )


def test_icon_patch_allows_icon_name_quality_issues_when_text_is_preserved():
    assert should_apply_icon_patch(
        original_text="Open settings",
        patched_text="[Icon: bullet] Open settings",
        field_key="text",
    )
    assert should_apply_icon_patch(
        original_text="Step 2 Click Add.",
        patched_text="[Icon: Step 2] Step 2 Click Add.",
        field_key="text",
    )


def test_strip_reasoning_text_removes_thinking_suffix():
    assert strip_reasoning_text("<think>inspect</think>\nOpen [Icon: gear] settings") == (
        "Open [Icon: gear] settings"
    )


def test_request_icon_patch_from_llm_sends_openai_vision_payload():
    class FakeImage:
        def save(self, buffer, format):
            assert format == "PNG"
            buffer.write(b"fake-image")

    class FakeCompletions:
        def __init__(self):
            self.kwargs = {}

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="<think>skip</think>\nClick [Icon: save].")
                    )
                ]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    output = request_icon_patch_from_llm(
        client=client,
        model="local-vlm",
        image=FakeImage(),
        prompt="Patch icons",
        max_tokens=8000,
    )

    assert output == "Click [Icon: save]."
    assert completions.kwargs["model"] == "local-vlm"
    assert completions.kwargs["max_tokens"] == 8000
    assert completions.kwargs["temperature"] == 0
    assert completions.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    content = completions.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == image_to_data_url(FakeImage())
    assert content[1] == {"type": "text", "text": "Patch icons"}


def test_iter_icon_patch_results_sends_multiple_requests():
    class FakeImage:
        def save(self, buffer, format):
            buffer.write(b"fake-image")

    class FakeCompletions:
        def create(self, **kwargs):
            prompt = kwargs["messages"][0]["content"][1]["text"]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"{prompt} done"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    requests = [
        {"image": FakeImage(), "prompt": "first", "original_text": "first", "key": "text"},
        {"image": FakeImage(), "prompt": "second", "original_text": "second", "key": "text"},
    ]

    results = list(
        iter_icon_patch_results(
            requests,
            client=client,
            model="local-vlm",
            max_tokens=8000,
            concurrency=2,
            invalid_retry_limit=3,
        )
    )

    assert {output for _, output, _, _, _ in results} == {"first done", "second done"}
    assert {event["status"] for *_, event in results} == {"ok"}
