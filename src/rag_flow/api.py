from __future__ import annotations

import argparse
import logging
from typing import Any

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from .config import AppConfig
from .retrieval import FinalOutput, RetrievalEngine, RetrievedImage, build_final_output

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class RetrievedImageResponse(BaseModel):
    hit_rank: int
    chunk_id: str
    source_relpath: str
    img_path: str
    image_path: str
    image_exists: bool
    page_idx: int
    page_number: int
    bbox: list[float] = Field(default_factory=list)
    image_answering_policy: str
    image_answering_confidence: str = ""
    image_answering_reason: str = ""
    image_caption: str = ""
    image_description_vlm: str = ""


class HitDetailResponse(BaseModel):
    rank: int
    page_idx: int
    page_number: int
    page_indices: list[int] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    score: float
    is_continuation: bool
    chunk_id: str = ""
    source_relpath: str = ""
    visual_page_prior: float = 0.0
    visual_alignment_score: float = 0.0
    dense_rrf_score: float = 0.0
    sparse_rrf_score: float = 0.0
    visual_rrf_score: float = 0.0
    direct_text_rrf_score: float = 0.0
    image_references: list[RetrievedImageResponse] = Field(default_factory=list)


class FinalOutputResponse(BaseModel):
    mode: str
    context: str
    content: list[dict[str, Any]] = Field(default_factory=list)
    images: list[RetrievedImageResponse] = Field(default_factory=list)


class QueryResponse(BaseModel):
    hit_page: int
    all_hits: list[HitDetailResponse]
    context: str
    images: list[RetrievedImageResponse] = Field(default_factory=list)
    final_output: FinalOutputResponse


def _image_response(image: RetrievedImage) -> RetrievedImageResponse:
    return RetrievedImageResponse(
        hit_rank=image.hit_rank,
        chunk_id=image.chunk_id,
        source_relpath=image.source_relpath,
        img_path=image.img_path,
        image_path=image.image_path,
        image_exists=image.image_exists,
        page_idx=image.page_idx,
        page_number=image.page_number,
        bbox=list(image.bbox),
        image_answering_policy=image.image_answering_policy,
        image_answering_confidence=image.image_answering_confidence,
        image_answering_reason=image.image_answering_reason,
        image_caption=image.image_caption,
        image_description_vlm=image.image_description_vlm,
    )


def _final_output_response(final_output: FinalOutput) -> FinalOutputResponse:
    return FinalOutputResponse(
        mode=final_output.mode,
        context=final_output.context,
        content=[dict(item) for item in final_output.content],
        images=[_image_response(image) for image in final_output.images],
    )


def create_app(config: AppConfig | None = None) -> FastAPI:
    resolved_config = config or AppConfig.from_env()
    engine = RetrievalEngine(resolved_config)
    app = FastAPI(title="RAG Flow Retrieval API")

    @app.on_event("startup")
    def startup_event() -> None:
        engine.load()

    @app.post("/retrieve", response_model=QueryResponse)
    def retrieve(req: QueryRequest, authorization: str | None = Header(default=None)) -> QueryResponse:
        if resolved_config.server.retriever_api_key:
            expected = f"Bearer {resolved_config.server.retriever_api_key}"
            if authorization != expected:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized",
                )
        if len(req.query) > resolved_config.server.max_query_chars:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Query must be {resolved_config.server.max_query_chars} characters or fewer.",
            )

        try:
            result = engine.retrieve(req.query)
        except Exception as exc:
            logger.exception("Retriever request failed")
            raise HTTPException(status_code=500, detail="Retriever request failed.") from exc
        final_output = result.final_output or build_final_output(
            context=result.context,
            images=result.images,
            include_images=bool(resolved_config.retrieval.final_output_images),
        )

        return QueryResponse(
            hit_page=result.hit_page,
            all_hits=[
                HitDetailResponse(
                    rank=hit.rank,
                    page_idx=hit.page_idx,
                    page_number=hit.page_number,
                    page_indices=hit.page_indices,
                    page_numbers=hit.page_numbers,
                    score=hit.score,
                    is_continuation=hit.is_continuation,
                    chunk_id=hit.chunk_id,
                    source_relpath=hit.source_relpath,
                    visual_page_prior=hit.visual_page_prior,
                    visual_alignment_score=hit.visual_alignment_score,
                    dense_rrf_score=hit.dense_rrf_score,
                    sparse_rrf_score=hit.sparse_rrf_score,
                    visual_rrf_score=hit.visual_rrf_score,
                    direct_text_rrf_score=hit.direct_text_rrf_score,
                    image_references=[_image_response(image) for image in hit.image_references],
                )
                for hit in result.all_hits
            ],
            context=result.context,
            images=[_image_response(image) for image in result.images],
            final_output=_final_output_response(final_output),
        )

    return app


app = create_app()


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Start the RAG Flow retrieval API.")
    parser.add_argument("--host", default=config.server.host)
    parser.add_argument("--port", type=int, default=config.server.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    uvicorn.run("rag_flow.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
