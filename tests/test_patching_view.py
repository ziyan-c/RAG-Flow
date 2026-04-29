from pathlib import Path

from rag_flow.preprocessing.patching_view import collect_patching_view_regions, patching_view_path_for


def test_patching_view_path_removes_content_list_suffix():
    assert patching_view_path_for(Path("/tmp/manual_content_list.json")).name == "manual_PATCHING_VIEW.pdf"
    assert patching_view_path_for(Path("/tmp/manual_content_list_PATCHED.json")).name == "manual_PATCHING_VIEW.pdf"
    assert (
        patching_view_path_for(Path("/tmp/manual_content_list_PATCHED_CAPTIONED.json")).name
        == "manual_PATCHING_VIEW.pdf"
    )


def test_collect_patching_view_regions_includes_linked_inline_icon_and_expanded_text_bbox():
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

    plan = collect_patching_view_regions(content_data)

    assert plan.inline_icon_candidates == 1
    assert plan.inline_icons_linked == 1
    assert plan.field_counts == {"text": 1, "inline_icon": 1}
    text_region = next(region for region in plan.regions if region.field == "text")
    assert text_region.bbox == (158.0, 528.0, 672.0, 587.0)


def test_collect_patching_view_regions_includes_table_continuation_crop():
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
        {
            "type": "table",
            "bbox": [171, 93, 905, 904],
            "page_idx": 19,
            "img_path": "",
            "table_caption": [],
            "table_footnote": [],
        },
    ]

    plan = collect_patching_view_regions(content_data)

    assert plan.field_counts["table_body"] == 2
    body_pages = [region.page_idx for region in plan.regions if region.field == "table_body"]
    assert body_pages == [18, 19]
