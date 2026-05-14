from __future__ import annotations

import argparse
import logging

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from .config import AppConfig
from .retrieval import RetrievalEngine

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class HitDetailResponse(BaseModel):
    rank: int
    page_idx: int
    page_number: int
    score: float
    is_continuation: bool
    chunk_id: str = ""
    visual_page_prior: float = 0.0
    visual_alignment_score: float = 0.0
    section_bonus: float = 0.0
    page_bonus: float = 0.0
    dense_rrf_score: float = 0.0
    sparse_rrf_score: float = 0.0
    visual_rrf_score: float = 0.0
    direct_text_rrf_score: float = 0.0
    is_visual_seed: bool = False
    seed_page_idx: int = 0
    seed_source_route: str = ""
    candidate_page_distance: int = 0


class QueryResponse(BaseModel):
    hit_page: int
    all_hits: list[HitDetailResponse]
    context: str


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

        return QueryResponse(
            hit_page=result.hit_page,
            all_hits=[
                HitDetailResponse(
                    rank=hit.rank,
                    page_idx=hit.page_idx,
                    page_number=hit.page_number,
                    score=hit.score,
                    is_continuation=hit.is_continuation,
                    chunk_id=hit.chunk_id,
                    visual_page_prior=hit.visual_page_prior,
                    visual_alignment_score=hit.visual_alignment_score,
                    section_bonus=hit.section_bonus,
                    page_bonus=hit.page_bonus,
                    dense_rrf_score=hit.dense_rrf_score,
                    sparse_rrf_score=hit.sparse_rrf_score,
                    visual_rrf_score=hit.visual_rrf_score,
                    direct_text_rrf_score=hit.direct_text_rrf_score,
                    is_visual_seed=hit.is_visual_seed,
                    seed_page_idx=hit.seed_page_idx,
                    seed_source_route=hit.seed_source_route,
                    candidate_page_distance=hit.candidate_page_distance,
                )
                for hit in result.all_hits
            ],
            context=result.context,
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
