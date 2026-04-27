from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import AppConfig
from .retrieval import RetrievalEngine


class QueryRequest(BaseModel):
    query: str


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
    def retrieve(req: QueryRequest) -> QueryResponse:
        try:
            result = engine.retrieve(req.query)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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
