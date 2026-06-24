from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .source_paths import normalize_source_name, source_name_for_pdf, source_root_from_input_path

METADATA_FILENAME = "metadata.yml"
METADATA_SIDECAR_SUFFIX = "_metadata.yml"
DOCUMENT_METADATA_FIELDS = (
    "filename",
    "product_families",
    "product_subfamilies",
    "doc_type",
    "version",
    "models",
    "language",
    "lifecycle_status",
    "topic_tags",
)


@dataclass(frozen=True)
class TaggingStats:
    input_json: Path
    output_json: Path
    metadata_yaml: Path | None
    chunk_count: int
    tagged_count: int
    missing_metadata_count: int


def tagged_json_path_for(chunks_json: str | Path) -> Path:
    path = Path(chunks_json)
    if path.suffix == ".json":
        return path.with_name(f"{path.stem}_TAGGED.json")
    return path.with_name(f"{path.name}_TAGGED")


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if char == "#" and not in_single and not in_double:
            return value[:index].rstrip()
    return value.strip()


def _parse_scalar(value: str) -> Any:
    raw = _strip_inline_comment(value)
    if raw in {"", "null", "~"}:
        return None
    if raw == "[]":
        return []
    if raw == "{}":
        return {}
    if raw.startswith('"') and raw.endswith('"'):
        return json.loads(raw)
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1].replace("''", "'")
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_document_metadata_yaml(path: str | Path) -> dict[str, dict[str, Any]]:
    metadata_path = Path(path)
    documents: dict[str, dict[str, Any]] = {}
    in_documents = False
    current_doc: str | None = None
    current_field: str | None = None

    for line_number, raw_line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        if not in_documents:
            if line == "documents:":
                in_documents = True
            continue

        if line.startswith("  ") and not line.startswith("    "):
            stripped = line.strip()
            if not stripped.endswith(":"):
                raise ValueError(f"{metadata_path}:{line_number}: expected document key")
            current_doc = normalize_source_name(_parse_scalar(stripped[:-1]))
            documents[current_doc] = {}
            current_field = None
            continue

        if current_doc is None:
            raise ValueError(f"{metadata_path}:{line_number}: field appears before document key")

        if line.startswith("    ") and not line.startswith("      "):
            stripped = line.strip()
            if ":" not in stripped:
                raise ValueError(f"{metadata_path}:{line_number}: expected field mapping")
            field, raw_value = stripped.split(":", 1)
            field = field.strip()
            value_text = raw_value.strip()
            if value_text:
                documents[current_doc][field] = _parse_scalar(value_text)
                current_field = None
            else:
                documents[current_doc][field] = []
                current_field = field
            continue

        if line.startswith("      - "):
            if current_field is None:
                raise ValueError(f"{metadata_path}:{line_number}: list item appears outside a list field")
            value = _parse_scalar(line.strip()[2:].strip())
            current_value = documents[current_doc].setdefault(current_field, [])
            if not isinstance(current_value, list):
                raise ValueError(f"{metadata_path}:{line_number}: field {current_field!r} is not a list")
            current_value.append(value)
            continue

        raise ValueError(f"{metadata_path}:{line_number}: unsupported metadata YAML shape")

    return documents


def _parse_sidecar_metadata_yaml(path: str | Path) -> tuple[str, dict[str, Any]]:
    metadata_path = Path(path)
    metadata: dict[str, Any] = {}
    current_field: str | None = None

    for line_number, raw_line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        if not line.startswith(" "):
            if ":" not in line:
                raise ValueError(f"{metadata_path}:{line_number}: expected field mapping")
            field, raw_value = line.split(":", 1)
            field = field.strip()
            value_text = raw_value.strip()
            if value_text:
                metadata[field] = _parse_scalar(value_text)
                current_field = None
            else:
                metadata[field] = []
                current_field = field
            continue

        if line.startswith("  - "):
            if current_field is None:
                raise ValueError(f"{metadata_path}:{line_number}: list item appears outside a list field")
            current_value = metadata.setdefault(current_field, [])
            if not isinstance(current_value, list):
                raise ValueError(f"{metadata_path}:{line_number}: field {current_field!r} is not a list")
            current_value.append(_parse_scalar(line.strip()[2:].strip()))
            continue

        raise ValueError(f"{metadata_path}:{line_number}: unsupported sidecar metadata YAML shape")

    source_relpath = metadata.get("source_relpath")
    if not isinstance(source_relpath, str) or not source_relpath.strip():
        raise ValueError(f"{metadata_path}: missing required source_relpath")
    return normalize_source_name(source_relpath), metadata


def load_document_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    metadata_path = Path(path)
    text = metadata_path.read_text(encoding="utf-8")
    if any(line.rstrip() == "documents:" for line in text.splitlines()):
        documents = _parse_document_metadata_yaml(metadata_path)
    else:
        source_relpath, metadata = _parse_sidecar_metadata_yaml(metadata_path)
        documents = {source_relpath: metadata}
    normalized: dict[str, dict[str, Any]] = {}
    for source_relpath, metadata in documents.items():
        normalized[source_relpath] = {field: metadata.get(field) for field in DOCUMENT_METADATA_FIELDS}
    return normalized


def metadata_yaml_for_source(
    source_pdf: str | Path | None,
    *,
    source_name: str | None = None,
    source_root: str | Path | None = None,
    metadata_yaml: str | Path | None = None,
) -> Path | None:
    if metadata_yaml:
        return Path(metadata_yaml)
    if not source_pdf:
        return None

    pdf_path = Path(source_pdf).expanduser()
    return pdf_path.with_name(f"{pdf_path.stem}{METADATA_SIDECAR_SUFFIX}")


def tag_chunks(
    chunks: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    *,
    strict: bool = True,
) -> tuple[list[dict[str, Any]], int, int]:
    tagged_chunks: list[dict[str, Any]] = []
    tagged_count = 0
    missing_count = 0

    for chunk in chunks:
        tagged = dict(chunk)
        metadata = dict(tagged.get("metadata") or {})
        source_relpath = normalize_source_name(metadata.get("source_relpath") or metadata.get("source") or "")
        document_metadata = documents.get(source_relpath)
        if document_metadata is None:
            missing_count += 1
            if strict:
                raise KeyError(f"No document metadata for source_relpath={source_relpath!r}")
        else:
            metadata.update(document_metadata)
            tagged_count += 1
        tagged["metadata"] = metadata
        tagged_chunks.append(tagged)

    return tagged_chunks, tagged_count, missing_count


def write_tagged_chunks(
    *,
    chunks_json: str | Path,
    output_json: str | Path | None = None,
    metadata_yaml: str | Path | None = None,
    source_pdf: str | Path | None = None,
    source_name: str | None = None,
    source_root: str | Path | None = None,
    require_metadata: bool = False,
    strict: bool = True,
) -> TaggingStats:
    input_path = Path(chunks_json)
    output_path = Path(output_json) if output_json else tagged_json_path_for(input_path)
    resolved_metadata = metadata_yaml_for_source(
        source_pdf,
        source_name=source_name,
        source_root=source_root,
        metadata_yaml=metadata_yaml,
    )

    chunks = json.loads(input_path.read_text(encoding="utf-8"))
    if resolved_metadata is None or not resolved_metadata.exists():
        if require_metadata:
            expected = resolved_metadata or Path(METADATA_FILENAME)
            raise FileNotFoundError(f"Tagging is enabled, but metadata.yml was not found at {expected}")
        tagged_chunks = chunks
        tagged_count = 0
        missing_count = 0
        used_metadata = None
    else:
        documents = load_document_metadata(resolved_metadata)
        tagged_chunks, tagged_count, missing_count = tag_chunks(chunks, documents, strict=strict)
        used_metadata = resolved_metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tagged_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return TaggingStats(
        input_json=input_path,
        output_json=output_path,
        metadata_yaml=used_metadata,
        chunk_count=len(chunks),
        tagged_count=tagged_count,
        missing_metadata_count=missing_count,
    )


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Tag chunk JSON with document metadata.yml fields.")
    parser.add_argument("--chunks", default=str(config.paths.chunks_json), help="Input chunk JSON.")
    parser.add_argument("--output", default=str(config.paths.tagged_json), help="Output tagged chunk JSON.")
    parser.add_argument("--metadata-yaml", help="metadata.yml path. Defaults to source root when available.")
    parser.add_argument("--source-pdf", default=str(config.paths.source_pdf), help="Source PDF for metadata.yml discovery.")
    parser.add_argument("--source-root", default=str(config.paths.source_root or ""), help="Source root containing metadata.yml.")
    parser.add_argument("--source-name", default=config.paths.source_name, help="Source relpath used in chunk metadata.")
    parser.add_argument("--require-metadata", action="store_true", help="Fail if metadata.yml is not found.")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail when metadata.yml lacks a chunk source.")
    args = parser.parse_args(argv)

    source_root = args.source_root or source_root_from_input_path(config.mineru.input_path)
    source_name = source_name_for_pdf(
        args.source_pdf,
        configured_source_pdf=config.paths.source_pdf,
        configured_source_name=args.source_name,
        source_root=source_root,
    )
    stats = write_tagged_chunks(
        chunks_json=args.chunks,
        output_json=args.output,
        metadata_yaml=args.metadata_yaml,
        source_pdf=args.source_pdf,
        source_name=source_name,
        source_root=source_root,
        require_metadata=args.require_metadata or config.tagging.enabled,
        strict=not args.allow_missing,
    )
    if stats.metadata_yaml:
        print(f"Tagged {stats.tagged_count}/{stats.chunk_count} chunks using {stats.metadata_yaml}")
    else:
        print(f"No metadata.yml found; copied {stats.chunk_count} chunks unchanged")
    print(f"tagged_json={stats.output_json}")


if __name__ == "__main__":
    main()
