import pytest

from rag_flow.chat_cli import _build_answering_user_content, _count_image_items


def test_build_answering_user_content_uses_retriever_final_output():
    response = {
        "context": "legacy context should not be read directly",
        "final_output": {
            "content": [
                {"type": "text", "text": "retrieved context"},
                {"type": "image_url", "image_url": {"url": "/tmp/figure.png"}},
            ]
        },
    }

    content = _build_answering_user_content(response, "How do I repair the database?")

    assert content[0]["type"] == "text"
    assert "How do I repair the database?" in content[0]["text"]
    assert "legacy context should not be read directly" not in content[0]["text"]
    assert content[1:] == response["final_output"]["content"]
    assert _count_image_items(content) == 1


def test_build_answering_user_content_requires_final_output():
    with pytest.raises(ValueError, match="final_output"):
        _build_answering_user_content({"context": "context only"}, "Question?")
