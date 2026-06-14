from __future__ import annotations

import json

from rag_flow.tagging import load_document_metadata, tagged_json_path_for, write_tagged_chunks


def test_tagged_json_path_appends_tagged_suffix(tmp_path):
    chunks = tmp_path / "manual_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED.json"

    assert tagged_json_path_for(chunks).name == "manual_content_list_SECTIONED_PATCHED_CAPTIONED_CHUNKED_TAGGED.json"


def test_write_tagged_chunks_adds_document_metadata(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    metadata_yaml = source_root / "metadata.yml"
    metadata_yaml.write_text(
        """
metadata_schema_version: 1
documents:
  "DSS/manual.pdf":
    filename: "manual.pdf"
    product_families:
      - dss
    product_subfamilies:
      - dss_professional
    doc_type: user_manual
    version: "8.7.0"
    models: []
    language: en
    topic_tags:
      - add_device
      - onvif
""".lstrip(),
        encoding="utf-8",
    )
    chunks_json = tmp_path / "chunks.json"
    chunks_json.write_text(
        json.dumps(
            [
                {
                    "chunk_content": "hello",
                    "metadata": {
                        "source_relpath": "DSS/manual.pdf",
                        "source_filename": "manual.pdf",
                        "page_idx": 0,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "tagged.json"

    stats = write_tagged_chunks(
        chunks_json=chunks_json,
        output_json=output_json,
        source_pdf=source_root / "DSS" / "manual.pdf",
        source_name="DSS/manual.pdf",
    )

    tagged = json.loads(output_json.read_text(encoding="utf-8"))
    metadata = tagged[0]["metadata"]
    assert stats.tagged_count == 1
    assert stats.missing_metadata_count == 0
    assert metadata["product_families"] == ["dss"]
    assert metadata["product_subfamilies"] == ["dss_professional"]
    assert metadata["doc_type"] == "user_manual"
    assert metadata["version"] == "8.7.0"
    assert metadata["models"] == []
    assert metadata["language"] == "en"
    assert metadata["topic_tags"] == ["add_device", "onvif"]
    assert metadata["source_filename"] == "manual.pdf"


def test_write_tagged_chunks_copies_when_metadata_is_absent(tmp_path):
    chunks_json = tmp_path / "chunks.json"
    chunks = [{"chunk_content": "hello", "metadata": {"source_relpath": "manual.pdf"}}]
    chunks_json.write_text(json.dumps(chunks), encoding="utf-8")
    output_json = tmp_path / "tagged.json"

    stats = write_tagged_chunks(
        chunks_json=chunks_json,
        output_json=output_json,
        source_pdf=tmp_path / "manual.pdf",
    )

    assert stats.metadata_yaml is None
    assert stats.tagged_count == 0
    assert json.loads(output_json.read_text(encoding="utf-8")) == chunks


def test_write_tagged_chunks_requires_metadata_when_enabled(tmp_path):
    chunks_json = tmp_path / "chunks.json"
    chunks_json.write_text(json.dumps([{"chunk_content": "hello", "metadata": {"source_relpath": "manual.pdf"}}]), encoding="utf-8")

    try:
        write_tagged_chunks(
            chunks_json=chunks_json,
            output_json=tmp_path / "tagged.json",
            source_pdf=tmp_path / "manual.pdf",
            require_metadata=True,
        )
    except FileNotFoundError as exc:
        assert "metadata.yml" in str(exc)
    else:
        raise AssertionError("Expected missing metadata.yml to fail when metadata is required")


def test_load_document_metadata_reads_curated_shape(tmp_path):
    metadata_yaml = tmp_path / "metadata.yml"
    metadata_yaml.write_text(
        """
metadata_schema_version: 1
documents:
  "manual.pdf":
    filename: "manual.pdf"
    product_families: []
    product_subfamilies: []
    doc_type: datasheet
    version: null
    models:
      - IPC-HDBW3849R1-ZAS-PV-PRO
    language: en
    topic_tags:
      - installation
""".lstrip(),
        encoding="utf-8",
    )

    documents = load_document_metadata(metadata_yaml)

    assert documents["manual.pdf"]["version"] is None
    assert documents["manual.pdf"]["models"] == ["IPC-HDBW3849R1-ZAS-PV-PRO"]
