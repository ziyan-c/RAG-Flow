import json
from types import SimpleNamespace

import pytest

from rag_flow.preprocessing.image_descriptions import (
    ImageDescriptionLLMOutput,
    captioned_json_path_for,
    checkpoint_path_for,
    collect_context_token_stats,
    collect_image_description_stats,
    get_surrounding_text_context,
    image_description_response_format,
    request_image_description_from_llm,
    resize_image_for_captioning,
    resolve_image_description_artifacts,
    should_caption_image_block,
)
from rag_flow.preprocessing.small_icons import image_to_data_url


def test_captioned_json_path_for_content_list_names():
    assert captioned_json_path_for("/tmp/manual_content_list.json").name == (
        "manual_content_list_PATCHED_CAPTIONED.json"
    )
    assert captioned_json_path_for("/tmp/manual_content_list_PATCHED.json").name == (
        "manual_content_list_PATCHED_CAPTIONED.json"
    )
    assert captioned_json_path_for("/tmp/manual_content_list_PATCHED_CAPTIONED.json").name == (
        "manual_content_list_PATCHED_CAPTIONED.json"
    )
    assert captioned_json_path_for("/tmp/manual_content_list_SECTIONED_PATCHED.json").name == (
        "manual_content_list_SECTIONED_PATCHED_CAPTIONED.json"
    )
    assert captioned_json_path_for("/tmp/manual_content_list_SECTIONED_PATCHED_CAPTIONED.json").name == (
        "manual_content_list_SECTIONED_PATCHED_CAPTIONED.json"
    )


def test_image_caption_checkpoint_path_uses_output_stem():
    assert checkpoint_path_for("/tmp/manual_content_list_PATCHED_CAPTIONED.json").name == (
        "manual_content_list_PATCHED_CAPTIONED.checkpoint.json"
    )


def test_resolve_image_description_artifacts_from_mineru_output_dir(tmp_path):
    artifact_dir = tmp_path / "hybrid_auto"
    artifact_dir.mkdir()
    (artifact_dir / "manual_content_list.json").write_text("[]", encoding="utf-8")
    (artifact_dir / "manual_origin.pdf").write_text("pdf", encoding="utf-8")

    artifacts = resolve_image_description_artifacts(artifact_dir)

    assert artifacts.artifact_dir == artifact_dir
    assert artifacts.base_dir == artifact_dir
    assert artifacts.input_json == artifact_dir / "manual_content_list_PATCHED.json"
    assert artifacts.output_json == artifact_dir / "manual_content_list_PATCHED_CAPTIONED.json"
    assert artifacts.origin_pdf == artifact_dir / "manual_origin.pdf"


def test_captioning_stats_count_inline_missing_and_existing_images(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "figure.jpg").write_text("jpg", encoding="utf-8")
    content = [
        {"type": "image", "page_idx": 0, "img_path": "images/figure.jpg"},
        {"type": "image", "page_idx": 0, "img_path": "images/missing.jpg"},
        {"type": "image", "page_idx": 0, "img_path": "images/icon.jpg", "vlm-small-icon-inline-icon": True},
        {"type": "image", "page_idx": 0},
        {
            "type": "image",
            "page_idx": 0,
            "img_path": "images/already.jpg",
            "image_description_vlm": "Already captioned.",
        },
    ]

    stats = collect_image_description_stats(content, base_dir=tmp_path)

    assert stats.images_seen == 5
    assert stats.caption_candidates == 2
    assert stats.missing_image_files == 1
    assert stats.skipped_inline_icons == 1
    assert stats.skipped_without_img_path == 1
    assert stats.skipped_existing == 1


def test_should_caption_image_block_skips_inline_candidate():
    assert not should_caption_image_block(
        {"type": "image", "img_path": "images/icon.jpg", "vlm-small-icon-inline-candidate": True}
    )
    assert should_caption_image_block({"type": "image", "img_path": "images/figure.jpg"})


def test_surrounding_text_context_limits_length_and_keeps_nearby_blocks():
    content = [
        {"type": "text", "page_idx": 0, "text": "far previous " * 100},
        {"type": "text", "page_idx": 0, "text": "near-before-important " * 100},
        {"type": "image", "page_idx": 0, "img_path": "images/figure.jpg", "image_caption": ["Figure 1"]},
        {"type": "text", "page_idx": 0, "text": "near-after-important " * 100},
        {"type": "text", "page_idx": 0, "text": "far after " * 100},
    ]

    context = get_surrounding_text_context(content, 2, max_context_tokens=140)

    assert "near-before-important" in context
    assert "Figure 1" in context
    assert "near-after-important" in context


def test_surrounding_text_context_uses_table_master_for_continuation_blocks():
    content = [
        {
            "type": "table",
            "page_idx": 0,
            "bbox": [100, 500, 900, 900],
            "table_caption": ["Table 1"],
            "table_body": "<table><tr><td>Record Mode</td></tr></table>",
            "table_footnote": [],
        },
        {
            "type": "table",
            "page_idx": 1,
            "bbox": [100, 80, 900, 300],
            "table_caption": [],
            "table_body": "",
            "table_footnote": [],
            "img_path": "",
        },
        {"type": "image", "page_idx": 1, "img_path": "images/figure.jpg", "image_caption": ["Figure 1"]},
    ]

    context = get_surrounding_text_context(content, 2, max_context_tokens=400)

    assert "continuation of table block 0" in context
    assert "Table 1" in context
    assert "Record Mode" in context


def test_surrounding_text_context_zero_budget_is_empty():
    content = [
        {"type": "text", "page_idx": 0, "text": "near-before-important " * 20},
        {"type": "image", "page_idx": 0, "img_path": "images/figure.jpg", "image_caption": ["Figure 1"]},
        {"type": "text", "page_idx": 0, "text": "near-after-important " * 20},
    ]

    context = get_surrounding_text_context(content, 1, max_context_tokens=0)

    assert context == ""


def test_context_token_stats_reports_budget_hits():
    content = [
        {"type": "text", "page_idx": 0, "text": "before " * 100},
        {"type": "image", "page_idx": 0, "img_path": "images/figure.jpg"},
        {"type": "text", "page_idx": 0, "text": "after " * 100},
    ]

    stats = collect_context_token_stats(content, max_context_tokens=20)

    assert stats.contexts == 1
    assert stats.max_tokens <= 20
    assert stats.contexts_at_budget == 1


def test_resize_image_for_captioning_preserves_aspect_ratio():
    Image = pytest.importorskip("PIL.Image")

    image = Image.new("RGB", (4000, 2000), "white")
    resized = resize_image_for_captioning(image, 1000)

    assert resized.size == (1000, 500)
    assert resize_image_for_captioning(image, 0).size == (4000, 2000)
    assert resize_image_for_captioning(Image.new("RGB", (800, 400), "white"), 1000).size == (800, 400)


def test_dry_run_stats_can_load_json(tmp_path):
    content = [{"type": "image", "page_idx": 0, "img_path": "images/figure.jpg"}]
    input_json = tmp_path / "manual_content_list_PATCHED.json"
    input_json.write_text(json.dumps(content), encoding="utf-8")

    loaded = json.loads(input_json.read_text(encoding="utf-8"))
    stats = collect_image_description_stats(loaded, base_dir=tmp_path)

    assert stats.caption_candidates == 1


def test_request_image_description_from_llm_sends_openai_vision_payload():
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
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "image_description_vlm": "A button toolbar screenshot.",
                                    "image_answering_policy": "image_recommended",
                                    "image_answering_confidence": "high",
                                    "image_answering_reason": "Visible button labels matter for answering.",
                                }
                            )
                        )
                    )
                ]
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    output = request_image_description_from_llm(
        client=client,
        model="local-vlm",
        image=FakeImage(),
        prompt="Describe image",
        max_tokens=8000,
    )

    assert output.image_description_vlm == "A button toolbar screenshot."
    assert output.image_answering_policy == "image_recommended"
    assert output.image_answering_confidence == "high"
    assert completions.kwargs["model"] == "local-vlm"
    assert completions.kwargs["max_tokens"] == 8000
    assert completions.kwargs["temperature"] == 0
    assert completions.kwargs["response_format"] == image_description_response_format()
    assert completions.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    content = completions.kwargs["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == image_to_data_url(FakeImage())
    assert content[1] == {"type": "text", "text": "Describe image"}


def test_image_description_llm_output_validates_structured_json():
    output = ImageDescriptionLLMOutput.model_validate_json(
        """
        {
          "image_description_vlm": "A dense UI screenshot with tabs and action buttons.",
          "image_answering_policy": "image_recommended",
          "image_answering_confidence": "medium",
          "image_answering_reason": "Visible labels and layout may matter for answering."
        }
        """
    )

    assert output.image_description_vlm == "A dense UI screenshot with tabs and action buttons."
    assert output.image_answering_policy == "image_recommended"
    assert output.image_answering_confidence == "medium"
    assert output.image_answering_reason == "Visible labels and layout may matter for answering."
