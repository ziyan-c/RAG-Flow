from rag_flow.preprocessing.small_icons import build_icon_patch_prompt, should_apply_icon_patch


def test_table_icon_prompt_requires_html_preservation():
    prompt = build_icon_patch_prompt(
        original_text="<table><tr><td></td><td>View details.</td></tr></table>",
        field_key="table_body",
    )

    assert "Preserve the full HTML table structure" in prompt
    assert "Do not convert the table to Markdown or prose" in prompt


def test_table_icon_patch_rejects_non_html_output():
    assert not should_apply_icon_patch(
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

