from __future__ import annotations

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
                )
                for hit in result.all_hits
            ],
            context=result.context,
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    config = AppConfig.from_env()
    uvicorn.run("rag_flow.api:app", host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
