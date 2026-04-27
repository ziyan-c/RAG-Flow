from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from .config import AppConfig
from .runtime import get_torch_device

DENSE_VECTOR_SIZE = 1024
COLPALI_VECTOR_SIZE = 128


def point_id(source_name: str, page_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_name}_{page_idx}"))


def validate_collection_schema(config: AppConfig) -> None:
    from qdrant_client import QdrantClient, models

    def enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    client = QdrantClient(path=str(config.paths.db_path))
    info = client.get_collection(config.paths.collection_name)
    vectors = info.config.params.vectors
    sparse_vectors = getattr(info.config.params, "sparse_vectors", {}) or {}
    errors = []

    if not isinstance(vectors, dict):
        errors.append("collection must use named vectors")
    else:
        dense = vectors.get("page-dense")
        colpali = vectors.get("page-colpali")
        if dense is None:
            errors.append("missing page-dense vector")
        elif dense.size != DENSE_VECTOR_SIZE or enum_value(dense.distance) != enum_value(models.Distance.COSINE):
            errors.append(f"page-dense must be {DENSE_VECTOR_SIZE} cosine dimensions")

        if colpali is None:
            errors.append("missing page-colpali vector")
        elif colpali.size != COLPALI_VECTOR_SIZE or enum_value(colpali.distance) != enum_value(models.Distance.COSINE):
            errors.append(f"page-colpali must be {COLPALI_VECTOR_SIZE} cosine dimensions")
        else:
            multivector_config = getattr(colpali, "multivector_config", None)
            if (
                not multivector_config
                or enum_value(multivector_config.comparator) != enum_value(models.MultiVectorComparator.MAX_SIM)
            ):
                errors.append("page-colpali must use MAX_SIM multivector comparison")

    if not isinstance(sparse_vectors, dict) or "page-sparse" not in sparse_vectors:
        errors.append("missing page-sparse sparse vector")

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
        return

    client.create_collection(
        collection_name=collection,
        vectors_config={
            "page-dense": models.VectorParams(size=DENSE_VECTOR_SIZE, distance=models.Distance.COSINE),
            "page-colpali": models.VectorParams(
                size=COLPALI_VECTOR_SIZE,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                ),
            ),
        },
        sparse_vectors_config={"page-sparse": models.SparseVectorParams()},
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="source",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="page_idx",
        field_schema=models.PayloadSchemaType.INTEGER,
    )


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def upsert_text_vectors(config: AppConfig, chunks_path: str | Path | None = None) -> None:
    from fastembed import SparseTextEmbedding, TextEmbedding
    from qdrant_client import QdrantClient, models

    ensure_collection(config)
    chunks = load_chunks(chunks_path or config.paths.chunks_json)
    documents = [chunk["page_content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    dense_model = TextEmbedding(config.models.dense_model)
    sparse_model = SparseTextEmbedding(config.models.sparse_model)
    dense_embeddings = list(dense_model.embed(documents))
    sparse_embeddings = list(sparse_model.embed(documents))

    points = []
    for doc, meta, dense_vec, sparse_vec in zip(documents, metadatas, dense_embeddings, sparse_embeddings):
        payload = dict(meta)
        payload["page_content"] = doc
        points.append(
            models.PointStruct(
                id=point_id(payload["source"], int(payload["page_idx"])),
                payload=payload,
                vector={
                    "page-dense": dense_vec.tolist(),
                    "page-sparse": models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
            )
        )

    client = QdrantClient(path=str(config.paths.db_path))
    client.upsert(collection_name=config.paths.collection_name, points=points)
    print(f"Upserted {len(points)} text points into {config.paths.collection_name}")


def upsert_colpali_vectors(
    config: AppConfig,
    *,
    pdf_path: str | Path | None = None,
    source_name: str | None = None,
    batch_size: int = 8,
    dpi: int = 200,
) -> None:
    import torch
    from colpali_engine.models import ColPali, ColPaliProcessor
    from pdf2image import convert_from_path
    from qdrant_client import QdrantClient, models
    from tqdm import tqdm

    ensure_collection(config)
    client = QdrantClient(path=str(config.paths.db_path))
    device = get_torch_device(feature="ColPali visual indexing")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    resolved_source_name = source_name or config.paths.source_name

    processor = ColPaliProcessor.from_pretrained(config.models.colpali_model)
    model = ColPali.from_pretrained(
        config.models.colpali_model,
        torch_dtype=dtype,
        device_map=device,
    ).eval()

    images = convert_from_path(str(pdf_path or config.paths.source_pdf), dpi=dpi)
    skipped_pages: list[int] = []
    last_valid_page_idx = 0

    for start in tqdm(range(0, len(images), batch_size), desc="Processing pages"):
        batch_images = images[start : start + batch_size]
        batch_page_indices = list(range(start, start + len(batch_images)))

        with torch.no_grad():
            batch_inputs = processor.process_images(batch_images).to(device)
            batch_inputs = {
                key: value.to(dtype) if value.is_floating_point() else value
                for key, value in batch_inputs.items()
            }
            embeddings = model(**batch_inputs)

        points_to_update = []
        for page_idx, embedding in zip(batch_page_indices, embeddings):
            points_to_update.append(
                models.PointVectors(
                    id=point_id(resolved_source_name, page_idx),
                    vector={"page-colpali": embedding.cpu().float().tolist()},
                )
            )

        try:
            client.update_vectors(collection_name=config.paths.collection_name, points=points_to_update)
            last_valid_page_idx = batch_page_indices[-1]
        except Exception:
            for current_page_idx, point in zip(batch_page_indices, points_to_update):
                try:
                    client.update_vectors(collection_name=config.paths.collection_name, points=[point])
                    last_valid_page_idx = current_page_idx
                except Exception:
                    client.upsert(
                        collection_name=config.paths.collection_name,
                        points=[
                            models.PointStruct(
                                id=point_id(resolved_source_name, current_page_idx),
                                payload={
                                    "source": resolved_source_name,
                                    "page_idx": current_page_idx,
                                    "parent_page_idx": last_valid_page_idx,
                                    "is_table_continuation": True,
                                    "page_content": (
                                        "[System Note: This page is a continuation containing only "
                                        f"tables/images. Please refer to page {last_valid_page_idx + 1} "
                                        "for headers and primary context.]"
                                    ),
                                },
                                vector={"page-colpali": point.vector["page-colpali"]},
                            )
                        ],
                    )
                    skipped_pages.append(current_page_idx)

    if skipped_pages:
        print(f"Inserted {len(skipped_pages)} visual-only continuation pages: {skipped_pages}")
    print(f"ColPali vectors are ready in {config.paths.collection_name}")


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

    valid = next((record for record in records if "page-colpali" in record.vector), records[0])
    print(f"Sample point: {valid.id}")
    print(f"Source: {valid.payload.get('source')} page_idx={valid.payload.get('page_idx')}")
    for name in ["page-dense", "page-sparse", "page-colpali"]:
        if name not in valid.vector:
            print(f"{name}: missing")
            continue
        vector = valid.vector[name]
        if name == "page-colpali":
            print(f"{name}: {len(vector)} patches x {len(vector[0])} dims")
        elif name == "page-sparse":
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

    visual_parser = subparsers.add_parser("visual", help="Upsert ColPali visual vectors.")
    visual_parser.add_argument("--pdf", default=str(config.paths.source_pdf))
    visual_parser.add_argument("--batch-size", type=int, default=8)
    visual_parser.add_argument("--dpi", type=int, default=200)

    subparsers.add_parser("inspect", help="Inspect collection vector completeness.")
    args = parser.parse_args(argv)

    if args.command == "text":
        upsert_text_vectors(config, args.chunks)
    elif args.command == "visual":
        upsert_colpali_vectors(config, pdf_path=args.pdf, batch_size=args.batch_size, dpi=args.dpi)
    elif args.command == "inspect":
        inspect_collection(config)


if __name__ == "__main__":
    main()
