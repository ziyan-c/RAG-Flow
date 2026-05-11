from __future__ import annotations

import argparse
import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import AppConfig
from .model_paths import resolve_model_location
from .runtime import get_torch_device

DENSE_VECTOR_SIZE = 1024
COLPALI_VECTOR_SIZE = 128
TEXT_INDEX_BATCH_SIZE = 256
VISUAL_INDEX_BATCH_SIZE = 64
VISUAL_INDEX_DPI = 200
TEXT_DENSE_VECTOR_NAME = "chunk-text-dense"
TEXT_SPARSE_VECTOR_NAME = "chunk-text-sparse"
PAGE_IMAGE_COLPALI_VECTOR_NAME = "page-image-colpali"
PAYLOAD_INDEX_SPECS = (
    ("source", "keyword"),
    ("page_idx", "integer"),
    ("page_start", "integer"),
    ("page_end", "integer"),
    ("page_indices", "integer"),
    ("chunk_id", "keyword"),
    ("section_title", "keyword"),
)


def point_id(source_name: str, page_idx: int, chunk_id: str | int | None = None) -> str:
    key = f"{source_name}_{chunk_id}" if chunk_id is not None else f"{source_name}_{page_idx}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def visual_point_id(source_name: str, page_idx: int) -> str:
    return point_id(source_name, page_idx, chunk_id=f"__visual_page__:{page_idx}")


def enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def uses_idf_modifier(sparse_params: Any, models: Any) -> bool:
    modifier = getattr(sparse_params, "modifier", None)
    return enum_value(modifier) == enum_value(models.Modifier.IDF)


def _payload_schema(models: Any, schema: str) -> Any:
    if schema == "keyword":
        return models.PayloadSchemaType.KEYWORD
    if schema == "integer":
        return models.PayloadSchemaType.INTEGER
    raise ValueError(f"Unsupported payload schema: {schema}")


def ensure_payload_indexes(client: Any, collection_name: str, models: Any) -> None:
    for field_name, schema in PAYLOAD_INDEX_SPECS:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=_payload_schema(models, schema),
            )
        except Exception as exc:
            message = str(exc).lower()
            if "already exists" not in message and "exists" not in message:
                raise


def validate_collection_schema(config: AppConfig) -> None:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(path=str(config.paths.db_path))
    info = client.get_collection(config.paths.collection_name)
    vectors = info.config.params.vectors
    sparse_vectors = getattr(info.config.params, "sparse_vectors", {}) or {}
    errors = []

    if not isinstance(vectors, dict):
        errors.append("collection must use named vectors")
    else:
        dense = vectors.get(TEXT_DENSE_VECTOR_NAME)
        colpali = vectors.get(PAGE_IMAGE_COLPALI_VECTOR_NAME)
        if dense is None:
            errors.append(f"missing {TEXT_DENSE_VECTOR_NAME} vector")
        elif dense.size != DENSE_VECTOR_SIZE or enum_value(dense.distance) != enum_value(models.Distance.COSINE):
            errors.append(f"{TEXT_DENSE_VECTOR_NAME} must be {DENSE_VECTOR_SIZE} cosine dimensions")

        if colpali is None:
            errors.append(f"missing {PAGE_IMAGE_COLPALI_VECTOR_NAME} vector")
        elif colpali.size != COLPALI_VECTOR_SIZE or enum_value(colpali.distance) != enum_value(models.Distance.COSINE):
            errors.append(f"{PAGE_IMAGE_COLPALI_VECTOR_NAME} must be {COLPALI_VECTOR_SIZE} cosine dimensions")
        else:
            multivector_config = getattr(colpali, "multivector_config", None)
            if (
                not multivector_config
                or enum_value(multivector_config.comparator) != enum_value(models.MultiVectorComparator.MAX_SIM)
            ):
                errors.append(f"{PAGE_IMAGE_COLPALI_VECTOR_NAME} must use MAX_SIM multivector comparison")

    if not isinstance(sparse_vectors, dict) or TEXT_SPARSE_VECTOR_NAME not in sparse_vectors:
        errors.append(f"missing {TEXT_SPARSE_VECTOR_NAME} sparse vector")
    elif not uses_idf_modifier(sparse_vectors[TEXT_SPARSE_VECTOR_NAME], models):
        errors.append(f"{TEXT_SPARSE_VECTOR_NAME} must use IDF sparse-vector modifier")

    if errors:
        details = "; ".join(errors)
        raise RuntimeError(
            f"Existing Qdrant collection {config.paths.collection_name!r} has an incompatible schema: "
            f"{details}. Recreate or migrate the collection before indexing."
        )


def ensure_collection(config: AppConfig) -> None:
    from qdrant_client import QdrantClient, models

    client = QdrantClient(path=str(config.paths.db_path))
    collection = config.paths.collection_name
    if client.collection_exists(collection):
        validate_collection_schema(config)
        ensure_payload_indexes(client, collection, models)
        return

    client.create_collection(
        collection_name=collection,
        vectors_config={
            TEXT_DENSE_VECTOR_NAME: models.VectorParams(size=DENSE_VECTOR_SIZE, distance=models.Distance.COSINE),
            PAGE_IMAGE_COLPALI_VECTOR_NAME: models.VectorParams(
                size=COLPALI_VECTOR_SIZE,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                ),
            ),
        },
        sparse_vectors_config={TEXT_SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    ensure_payload_indexes(client, collection, models)


def _delete_existing_points_for_sources(
    client: Any,
    collection_name: str,
    models: Any,
    *,
    source_names: set[str],
    visual: bool,
) -> None:
    visual_condition = models.FieldCondition(
        key="is_visual_page",
        match=models.MatchValue(value=True),
    )
    for source_name in sorted(source_names):
        source_condition = models.FieldCondition(
            key="source",
            match=models.MatchValue(value=source_name),
        )
        if visual:
            point_filter = models.Filter(must=[source_condition, visual_condition])
        else:
            point_filter = models.Filter(must=[source_condition], must_not=[visual_condition])
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=point_filter),
            wait=True,
        )


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def upsert_text_vectors(
    config: AppConfig,
    chunks_path: str | Path | None = None,
    *,
    batch_size: int = TEXT_INDEX_BATCH_SIZE,
) -> None:
    from fastembed import SparseTextEmbedding, TextEmbedding
    from qdrant_client import QdrantClient, models

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    ensure_collection(config)
    chunks = load_chunks(chunks_path or config.paths.chunks_json)
    documents = [chunk["chunk_content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    dense_model = TextEmbedding(config.models.dense_model)
    sparse_model = SparseTextEmbedding(config.models.sparse_model)
    client = QdrantClient(path=str(config.paths.db_path))
    source_names = {str(meta["source"]) for meta in metadatas if meta.get("source")}
    _delete_existing_points_for_sources(
        client,
        config.paths.collection_name,
        models,
        source_names=source_names,
        visual=False,
    )
    upserted_chunks = 0
    for start in range(0, len(documents), batch_size):
        batch_documents = documents[start : start + batch_size]
        batch_metadatas = metadatas[start : start + batch_size]
        dense_embeddings = list(dense_model.embed(batch_documents))
        sparse_embeddings = list(sparse_model.embed(batch_documents))

        points = []
        for doc, meta, dense_vec, sparse_vec in zip(
            batch_documents,
            batch_metadatas,
            dense_embeddings,
            sparse_embeddings,
        ):
            payload = dict(meta)
            payload["chunk_content"] = doc
            page_idx = int(payload.get("page_idx", payload.get("page_start", 0)))
            chunk_id = payload.get("chunk_id", payload.get("chunk_idx"))
            points.append(
                models.PointStruct(
                    id=point_id(payload["source"], page_idx, chunk_id=chunk_id),
                    payload=payload,
                    vector={
                        TEXT_DENSE_VECTOR_NAME: dense_vec.tolist(),
                        TEXT_SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                )
            )

        client.upsert(collection_name=config.paths.collection_name, points=points)
        upserted_chunks += len(points)

    print(f"Upserted {upserted_chunks} text points into {config.paths.collection_name}")


def _page_payloads_from_chunks(
    chunks: list[dict[str, Any]],
    *,
    source_name: str,
) -> dict[int, dict[str, Any]]:
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        meta = dict(chunk.get("metadata", {}))
        if meta.get("source") != source_name:
            continue
        pages = meta.get("page_indices")
        if not isinstance(pages, list) or not pages:
            pages = [meta.get("page_idx", meta.get("page_start", 0))]
        for page in pages:
            try:
                by_page[int(page)].append(chunk)
            except (TypeError, ValueError):
                continue

    payloads = {}
    for page_idx, page_chunks in by_page.items():
        first_meta = dict(page_chunks[0].get("metadata", {}))
        section_path = first_meta.get("section_path", [])
        chunk_ids = []
        for chunk in page_chunks:
            meta = dict(chunk.get("metadata", {}))
            if meta.get("chunk_id"):
                chunk_ids.append(str(meta["chunk_id"]))
        payload: dict[str, Any] = {
            "source": source_name,
            "page_idx": page_idx,
            "page_start": page_idx,
            "page_end": page_idx,
            "page_indices": [page_idx],
            "is_visual_page": True,
            "chunk_ids_on_page": chunk_ids,
        }
        if section_path:
            payload["section_path"] = section_path
            payload["section_title"] = first_meta.get("section_title") or section_path[-1]
            if first_meta.get("section_level") is not None:
                payload["section_level"] = first_meta["section_level"]
            if first_meta.get("section_source"):
                payload["section_source"] = first_meta["section_source"]
        payloads[page_idx] = payload
    return payloads


def _visual_page_payload(
    page_payloads: dict[int, dict[str, Any]],
    *,
    source_name: str,
    page_idx: int,
) -> dict[str, Any]:
    payload = dict(page_payloads.get(page_idx, {}))
    if not payload:
        payload = {
            "source": source_name,
            "page_idx": page_idx,
            "page_start": page_idx,
            "page_end": page_idx,
            "page_indices": [page_idx],
            "is_visual_page": True,
            "chunk_ids_on_page": [],
    }
    payload.setdefault("source", source_name)
    payload.setdefault("page_idx", page_idx)
    payload.setdefault("page_start", page_idx)
    payload.setdefault("page_end", page_idx)
    payload.setdefault("page_indices", [page_idx])
    payload.setdefault("is_visual_page", True)
    payload.setdefault("chunk_ids_on_page", [])
    return payload


def _page_batches(page_count: int, batch_size: int) -> list[tuple[int, int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        (start_idx, start_idx + 1, min(start_idx + batch_size, page_count))
        for start_idx in range(0, page_count, batch_size)
    ]


def upsert_colpali_vectors(
    config: AppConfig,
    *,
    pdf_path: str | Path | None = None,
    source_name: str | None = None,
    batch_size: int = VISUAL_INDEX_BATCH_SIZE,
    dpi: int = VISUAL_INDEX_DPI,
) -> None:
    import torch
    from colpali_engine.models import ColPali, ColPaliProcessor
    from pdf2image import convert_from_path, pdfinfo_from_path
    from qdrant_client import QdrantClient, models
    from tqdm import tqdm

    ensure_collection(config)
    client = QdrantClient(path=str(config.paths.db_path))
    device = get_torch_device(feature="ColPali visual indexing")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    resolved_source_name = source_name or config.paths.source_name
    colpali_model_location = resolve_model_location(
        config.models.colpali_model,
        explicit_path=config.models.colpali_model_path,
        local_root=config.models.colpali_local_model_root,
    )

    print(f"ColPali model: {colpali_model_location}")
    processor = ColPaliProcessor.from_pretrained(colpali_model_location)
    model = ColPali.from_pretrained(
        colpali_model_location,
        torch_dtype=dtype,
        device_map=device,
    ).eval()

    resolved_pdf_path = Path(pdf_path or config.paths.source_pdf)
    page_count = int(pdfinfo_from_path(str(resolved_pdf_path))["Pages"])
    try:
        page_payloads = _page_payloads_from_chunks(
            load_chunks(config.paths.chunks_json),
            source_name=resolved_source_name,
        )
    except (FileNotFoundError, json.JSONDecodeError):
        page_payloads = {}
    _delete_existing_points_for_sources(
        client,
        config.paths.collection_name,
        models,
        source_names={resolved_source_name},
        visual=True,
    )

    upserted_pages = 0
    printed_embedding_shape = False
    for start_idx, first_page, last_page in tqdm(
        _page_batches(page_count, batch_size),
        desc="Processing pages",
    ):
        batch_images = convert_from_path(
            str(resolved_pdf_path),
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
        )
        batch_page_indices = list(range(start_idx, start_idx + len(batch_images)))

        with torch.no_grad():
            batch_inputs = processor.process_images(batch_images).to(device)
            batch_inputs = {
                key: value.to(dtype) if value.is_floating_point() else value
                for key, value in batch_inputs.items()
            }
            embeddings = model(**batch_inputs)

        if not printed_embedding_shape and len(embeddings):
            first_embedding = embeddings[0]
            patch_count = len(first_embedding)
            vector_size = len(first_embedding[0]) if patch_count else 0
            print(f"{PAGE_IMAGE_COLPALI_VECTOR_NAME}: {patch_count} patches x {vector_size} dims")
            printed_embedding_shape = True

        points = []
        for page_idx, embedding in zip(batch_page_indices, embeddings):
            points.append(
                models.PointStruct(
                    id=visual_point_id(resolved_source_name, page_idx),
                    payload=_visual_page_payload(
                        page_payloads,
                        source_name=resolved_source_name,
                        page_idx=page_idx,
                    ),
                    vector={PAGE_IMAGE_COLPALI_VECTOR_NAME: embedding.cpu().float().tolist()},
                )
            )

        client.upsert(collection_name=config.paths.collection_name, points=points)
        upserted_pages += len(points)

    print(f"Upserted {upserted_pages} ColPali visual page points into {config.paths.collection_name}")


def inspect_collection(config: AppConfig, limit: int = 10) -> None:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(config.paths.db_path))
    info = client.get_collection(config.paths.collection_name)
    print(f"Status: {info.status}")
    print(f"Points: {info.points_count}")
    for name, params in info.config.params.vectors.items():
        print(f"{name}: {params.size} dims")

    records, _ = client.scroll(
        collection_name=config.paths.collection_name,
        limit=limit,
        with_vectors=True,
        with_payload=True,
    )
    if not records:
        print("No points found.")
        return

    valid = next((record for record in records if PAGE_IMAGE_COLPALI_VECTOR_NAME in record.vector), records[0])
    print(f"Sample point: {valid.id}")
    print(f"Source: {valid.payload.get('source')} page_idx={valid.payload.get('page_idx')}")
    for name in [TEXT_DENSE_VECTOR_NAME, TEXT_SPARSE_VECTOR_NAME, PAGE_IMAGE_COLPALI_VECTOR_NAME]:
        if name not in valid.vector:
            print(f"{name}: missing")
            continue
        vector = valid.vector[name]
        if name == PAGE_IMAGE_COLPALI_VECTOR_NAME:
            print(f"{name}: {len(vector)} patches x {len(vector[0])} dims")
        elif name == TEXT_SPARSE_VECTOR_NAME:
            count = len(vector.get("indices", [])) if isinstance(vector, dict) else len(vector.indices)
            print(f"{name}: {count} sparse features")
        else:
            print(f"{name}: {len(vector)} dims")


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Build or inspect the Qdrant manual index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    text_parser = subparsers.add_parser("text", help="Upsert dense and sparse text vectors.")
    text_parser.add_argument("--chunks", default=str(config.paths.chunks_json))
    text_parser.add_argument("--batch-size", type=int, default=config.indexing.text_batch_size)

    visual_parser = subparsers.add_parser("visual", help="Upsert ColPali visual vectors.")
    visual_parser.add_argument("--pdf", default=str(config.paths.source_pdf))
    visual_parser.add_argument("--source-name", help="Source PDF name stored in visual payloads.")
    visual_parser.add_argument("--batch-size", type=int, default=config.indexing.visual_batch_size)
    visual_parser.add_argument("--dpi", type=int, default=config.indexing.visual_dpi)

    subparsers.add_parser("inspect", help="Inspect collection vector completeness.")
    args = parser.parse_args(argv)

    if args.command == "text":
        upsert_text_vectors(config, args.chunks, batch_size=args.batch_size)
    elif args.command == "visual":
        pdf_path = Path(args.pdf)
        source_name = args.source_name
        if source_name is None and pdf_path != config.paths.source_pdf:
            source_name = pdf_path.name
        upsert_colpali_vectors(
            config,
            pdf_path=pdf_path,
            source_name=source_name,
            batch_size=args.batch_size,
            dpi=args.dpi,
        )
    elif args.command == "inspect":
        inspect_collection(config)


if __name__ == "__main__":
    main()
