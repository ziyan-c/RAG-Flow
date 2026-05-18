from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from rag_flow.config import AppConfig
from rag_flow.preprocessing.small_icons import image_to_data_url, strip_reasoning_text
from rag_flow.retrieval import RetrievalEngine, RetrievalResult


DEFAULT_QUERY_SET = Path("thesis/08-end-to-end-answering/data/answering_qaset_50.jsonl")
DEFAULT_OUTPUT_DIR = Path("thesis/08-end-to-end-answering/data/answering-runs")
THINKING_LEAK_RE = re.compile(r"(<think>|</think>|\banalysis\s*:|\breasoning\s*:)", re.IGNORECASE)


def _load_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            return dict(usage.model_dump())
        except Exception:
            pass
    output: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
    ):
        if hasattr(usage, key):
            value = getattr(usage, key)
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            output[key] = value
    return output


def _usage_int(usage: dict[str, Any], key: str) -> int:
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _list_ints(values: Any) -> list[int]:
    if isinstance(values, list):
        out = []
        for value in values:
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                continue
        return out
    if values is None:
        return []
    try:
        return [int(values)]
    except (TypeError, ValueError):
        return []


def _gold_chunk_ids(query: dict[str, Any]) -> set[str]:
    ids = query.get("gold_chunk_ids")
    if isinstance(ids, list):
        return {str(item) for item in ids}
    primary = query.get("primary_gold_chunk_id")
    return {str(primary)} if primary else set()


def _gold_page_indices(query: dict[str, Any]) -> set[int]:
    pages = _list_ints(query.get("gold_page_indices"))
    if pages:
        return set(pages)
    page_numbers = _list_ints(query.get("gold_page_numbers"))
    return {page - 1 for page in page_numbers if page > 0}


def _hit_chunk_ids(result: RetrievalResult) -> list[str]:
    return [hit.chunk_id for hit in result.all_hits if hit.chunk_id]


def _hit_page_indices(result: RetrievalResult) -> list[int]:
    pages: list[int] = []
    for hit in result.all_hits:
        hit_pages = hit.page_indices or [hit.page_idx]
        pages.extend(hit_pages)
    return sorted(set(pages))


def _page_scores(result: RetrievalResult) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for hit in result.all_hits:
        pages = hit.page_indices or [hit.page_idx]
        pages = sorted(set(int(page) for page in pages))
        if not pages:
            continue
        share = float(hit.score) / len(pages)
        for page in pages:
            scores[page] = scores.get(page, 0.0) + share
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _estimate_text_tokens(text: str, *, chars_per_token: float) -> int:
    return max(1, math.ceil(len(text) / max(1.0, chars_per_token)))


def _estimate_image_tokens(width: int, height: int) -> int:
    return max(1, math.ceil(width * height / 784.0))


def _render_pdf_page(pdf_path: Path, *, page_idx: int, dpi: int) -> Any:
    from pdf2image import convert_from_path

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_idx + 1,
        last_page=page_idx + 1,
    )
    if not images:
        raise RuntimeError(f"Failed to render page {page_idx + 1} from {pdf_path}")
    return images[0]


def _attached_page_images(
    *,
    pdf_path: Path,
    page_scores: list[tuple[int, float]],
    image_count: int,
    dpi: int,
    save_dir: Path | None = None,
    query_id: str = "",
) -> list[dict[str, Any]]:
    attached = []
    for page_idx, score in page_scores[: max(0, image_count)]:
        image = _render_pdf_page(pdf_path, page_idx=page_idx, dpi=dpi)
        width, height = int(image.width), int(image.height)
        image_path = ""
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            image_path = str(save_dir / f"{query_id}_page-{page_idx + 1}_dpi-{dpi}.png")
            image.save(image_path)
        attached.append(
            {
                "page_idx": page_idx,
                "page_number": page_idx + 1,
                "page_score": float(score),
                "dpi": dpi,
                "width": width,
                "height": height,
                "estimated_image_tokens": _estimate_image_tokens(width, height),
                "data_url": image_to_data_url(image),
                "image_path": image_path,
            }
        )
    return attached


def _answering_messages(*, query: str, context: str, attached_images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system = (
        "You are a senior technical manual assistant. Answer only from the provided retrieved "
        "context and attached PDF page images. If the answer is not supported, say that the "
        "provided evidence is insufficient. Do not output reasoning, analysis, chain-of-thought, "
        "or hidden thinking. Return only the final answer."
    )
    user_intro = (
        "[Retrieved text context]\n"
        f"{context}\n\n"
        "[User question]\n"
        f"{query}\n\n"
        "Answer the question using the retrieved context. Cite source page numbers for factual claims. "
        "For table questions, preserve parameter names and their meanings. For UI/image questions, use "
        "attached PDF page images if present to verify visible regions, fields, buttons, and layout. "
        "Do not mention internal scoring, gold labels, or this evaluation setup."
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": user_intro}]
    if attached_images:
        content.append(
            {
                "type": "text",
                "text": (
                    "\nAttached retrieved PDF page images follow. They are selected from the retrieval "
                    "results by page score, not from gold labels."
                ),
            }
        )
    for image in attached_images:
        content.append(
            {
                "type": "text",
                "text": (
                    f"Attached PDF page {image['page_number']} "
                    f"(retrieved page_score={image['page_score']:.6f}, dpi={image['dpi']})."
                ),
            }
        )
        content.append({"type": "image_url", "image_url": {"url": image["data_url"]}})
    content.append(
        {
            "type": "text",
            "text": (
                "\nReturn the final answer only. Do not output reasoning, analysis, or chain-of-thought."
            ),
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _call_answering_llm(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    enable_thinking: bool,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "separate_reasoning": True,
        },
    )
    message = response.choices[0].message
    raw_answer = message.content or ""
    reasoning = getattr(message, "reasoning_content", None) or ""
    answer = strip_reasoning_text(raw_answer)
    usage = _usage_to_dict(getattr(response, "usage", None))
    if hasattr(response, "model_dump"):
        raw_response = response.model_dump()
    else:
        raw_response = {"id": getattr(response, "id", ""), "usage": usage}
    return answer, reasoning, usage, raw_response


def _make_run_config(
    config: AppConfig,
    *,
    context_cap: int,
    enable_visual_retrieval: bool,
    route_mode: str,
    candidate_mode: str,
    visual_weight: float,
) -> AppConfig:
    retrieval = replace(
        config.retrieval,
        max_context_tokens=context_cap,
        enable_visual=enable_visual_retrieval,
        route_mode=route_mode,
        candidate_mode=candidate_mode,
        visual_weight=visual_weight,
    )
    return replace(config, retrieval=retrieval)


def run_answering_benchmark(
    *,
    query_set: Path,
    output_dir: Path,
    run_id: str,
    context_cap: int,
    image_count: int,
    image_dpi: int,
    enable_thinking: bool,
    enable_visual_retrieval: bool,
    route_mode: str,
    candidate_mode: str,
    visual_weight: float,
    max_tokens: int,
    llm_base_url: str,
    llm_model: str,
    llm_api_key: str,
    request_timeout: float,
    limit: int | None,
    save_images: bool,
    dry_run: bool,
) -> Path:
    base_config = AppConfig.from_env()
    run_config = _make_run_config(
        base_config,
        context_cap=context_cap,
        enable_visual_retrieval=enable_visual_retrieval,
        route_mode=route_mode,
        candidate_mode=candidate_mode,
        visual_weight=visual_weight,
    )
    queries = _load_jsonl(query_set, limit=limit)
    run_dir = output_dir / _safe_slug(run_id)
    responses_dir = run_dir / "llm-responses"
    answers_dir = run_dir / "answers"
    contexts_dir = run_dir / "contexts"
    retrieval_dir = run_dir / "retrieval"
    images_dir = run_dir / "attached-page-images" if save_images else None
    for path in (responses_dir, answers_dir, contexts_dir, retrieval_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "query_set": str(query_set),
        "query_count": len(queries),
        "context_cap": context_cap,
        "image_count": image_count,
        "image_dpi": image_dpi,
        "enable_thinking": enable_thinking,
        "enable_visual_retrieval": enable_visual_retrieval,
        "route_mode": route_mode,
        "candidate_mode": candidate_mode,
        "visual_weight": visual_weight,
        "max_tokens": max_tokens,
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "source_pdf": str(run_config.paths.source_pdf),
        "retrieval": {
            "retrieval_k": run_config.retrieval.retrieval_k,
            "final_top_k": run_config.retrieval.final_top_k,
            "rrf_k": run_config.retrieval.rrf_k,
            "min_score_ratio": run_config.retrieval.min_score_ratio,
            "context_chars_per_token": run_config.retrieval.context_chars_per_token,
        },
    }
    if dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    engine = RetrievalEngine(run_config)
    engine.load()
    from openai import OpenAI

    llm_client = OpenAI(api_key=llm_api_key, base_url=llm_base_url, timeout=request_timeout)
    rows: list[dict[str, Any]] = []
    errors = 0
    for index, query in enumerate(queries, start=1):
        query_id = str(query.get("query_id") or f"query-{index:04d}")
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        result = engine.retrieve(str(query.get("query", "")))
        retrieval_latency = time.perf_counter() - retrieval_started
        page_scores = _page_scores(result)
        attached_images: list[dict[str, Any]] = []
        image_error = ""
        try:
            attached_images = _attached_page_images(
                pdf_path=run_config.paths.source_pdf,
                page_scores=page_scores,
                image_count=image_count,
                dpi=image_dpi,
                save_dir=images_dir,
                query_id=query_id,
            )
        except Exception as exc:
            image_error = str(exc)
            if image_count > 0:
                errors += 1
        messages = _answering_messages(
            query=str(query.get("query", "")),
            context=result.context,
            attached_images=attached_images,
        )
        answer = ""
        reasoning = ""
        usage: dict[str, Any] = {}
        raw_response: dict[str, Any] = {}
        llm_error = ""
        answering_started = time.perf_counter()
        try:
            answer, reasoning, usage, raw_response = _call_answering_llm(
                client=llm_client,
                model=llm_model,
                messages=messages,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            errors += 1
            llm_error = str(exc)
        answering_latency = time.perf_counter() - answering_started
        total_latency = time.perf_counter() - started

        returned_chunks = _hit_chunk_ids(result)
        returned_pages = _hit_page_indices(result)
        gold_chunks = _gold_chunk_ids(query)
        gold_pages = _gold_page_indices(query)
        attached_page_indices = [int(item["page_idx"]) for item in attached_images]
        thinking_leak = bool(THINKING_LEAK_RE.search(answer or ""))
        estimated_image_tokens = sum(int(item["estimated_image_tokens"]) for item in attached_images)
        estimated_text_tokens = _estimate_text_tokens(
            result.context,
            chars_per_token=run_config.retrieval.context_chars_per_token,
        )
        row = {
            "query_id": query_id,
            "run_id": run_id,
            "query": query.get("query", ""),
            "difficulty": query.get("difficulty", ""),
            "query_type": query.get("query_type", ""),
            "evidence_type": query.get("evidence_type", ""),
            "requires_visual": int(bool(query.get("requires_visual", False))),
            "requires_table": int(bool(query.get("requires_table", False))),
            "context_cap": context_cap,
            "image_count": image_count,
            "image_dpi": image_dpi,
            "enable_thinking": int(enable_thinking),
            "enable_visual_retrieval": int(enable_visual_retrieval),
            "route_mode": route_mode,
            "candidate_mode": candidate_mode,
            "visual_weight": visual_weight,
            "retrieval_latency_seconds": round(retrieval_latency, 4),
            "answering_latency_seconds": round(answering_latency, 4),
            "total_latency_seconds": round(total_latency, 4),
            "hit_count": len(result.all_hits),
            "returned_chunk_ids": "|".join(returned_chunks),
            "returned_page_indices": "|".join(str(page) for page in returned_pages),
            "gold_chunk_ids": "|".join(sorted(gold_chunks)),
            "gold_page_indices": "|".join(str(page) for page in sorted(gold_pages)),
            "retrieval_gold_context_hit": int(bool(gold_chunks.intersection(returned_chunks))),
            "retrieval_gold_page_hit": int(bool(gold_pages.intersection(returned_pages))),
            "attached_page_indices": "|".join(str(page) for page in attached_page_indices),
            "attached_page_numbers": "|".join(str(page + 1) for page in attached_page_indices),
            "attached_gold_page_hit": int(bool(gold_pages.intersection(attached_page_indices))),
            "attached_image_widths": "|".join(str(item["width"]) for item in attached_images),
            "attached_image_heights": "|".join(str(item["height"]) for item in attached_images),
            "estimated_text_tokens": estimated_text_tokens,
            "estimated_image_tokens": estimated_image_tokens,
            "usage_prompt_tokens": _usage_int(usage, "prompt_tokens"),
            "usage_completion_tokens": _usage_int(usage, "completion_tokens"),
            "usage_total_tokens": _usage_int(usage, "total_tokens"),
            "usage_missing": int(not bool(usage)),
            "reasoning_chars": len(reasoning or ""),
            "thinking_leak": int(thinking_leak),
            "answer_chars": len(answer or ""),
            "image_error": image_error,
            "llm_error": llm_error,
            "answer_path": str(answers_dir / f"{query_id}.md"),
            "context_path": str(contexts_dir / f"{query_id}.txt"),
            "retrieval_path": str(retrieval_dir / f"{query_id}.json"),
            "response_path": str(responses_dir / f"{query_id}.json"),
            "answer_score": "",
            "failure_type": "",
            "review_notes": "",
        }
        rows.append(row)

        (answers_dir / f"{query_id}.md").write_text(answer or "", encoding="utf-8")
        (contexts_dir / f"{query_id}.txt").write_text(result.context, encoding="utf-8")
        retrieval_payload = {
            "hit_page": result.hit_page,
            "context": result.context,
            "all_hits": [hit.__dict__ for hit in result.all_hits],
            "page_scores": [{"page_idx": page, "page_number": page + 1, "score": score} for page, score in page_scores],
            "attached_pages": [
                {key: value for key, value in item.items() if key != "data_url"}
                for item in attached_images
            ],
        }
        (retrieval_dir / f"{query_id}.json").write_text(
            json.dumps(retrieval_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        response_payload = {
            "answer": answer,
            "reasoning_content": reasoning,
            "usage": usage,
            "raw_response": raw_response,
            "messages": [
                {
                    **message,
                    "content": "[omitted multimodal payload]" if message.get("role") == "user" else message.get("content"),
                }
                for message in messages
            ],
            "errors": {"image_error": image_error, "llm_error": llm_error},
        }
        (responses_dir / f"{query_id}.json").write_text(
            json.dumps(response_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[{index}/{len(queries)}] {query_id}: total={total_latency:.2f}s "
            f"retrieve={retrieval_latency:.2f}s answer={answering_latency:.2f}s "
            f"usage={row['usage_total_tokens'] or 'missing'} pages={row['attached_page_numbers'] or '-'}"
        )

    metrics_path = run_dir / "answering_metrics.csv"
    if rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "run_id": run_id,
        "queries": len(rows),
        "errors": errors,
        "thinking_leaks": sum(int(row["thinking_leak"]) for row in rows),
        "usage_missing": sum(int(row["usage_missing"]) for row in rows),
        "avg_total_latency_seconds": round(
            sum(float(row["total_latency_seconds"]) for row in rows) / len(rows), 4
        )
        if rows
        else 0,
        "avg_usage_total_tokens": round(
            sum(int(row["usage_total_tokens"]) for row in rows) / max(1, sum(1 for row in rows if int(row["usage_total_tokens"]) > 0)),
            2,
        )
        if rows
        else 0,
        "metrics_path": str(metrics_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return run_dir


def build_parser(config: AppConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run end-to-end answering benchmarks with Qwen/SGLang.")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    run_parser = subparsers.add_parser("run", help="Run one answering benchmark configuration.")
    run_parser.add_argument("--query-set", type=Path, default=DEFAULT_QUERY_SET)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--context-cap", type=int, default=config.retrieval.max_context_tokens)
    run_parser.add_argument("--image-count", type=int, default=0)
    run_parser.add_argument("--image-dpi", type=int, default=200)
    run_parser.add_argument("--enable-thinking", action="store_true")
    run_parser.add_argument("--enable-visual-retrieval", action="store_true")
    run_parser.add_argument("--route-mode", default=config.retrieval.route_mode)
    run_parser.add_argument("--candidate-mode", default=config.retrieval.candidate_mode)
    run_parser.add_argument("--visual-weight", type=float, default=config.retrieval.visual_weight)
    run_parser.add_argument("--llm-base-url", default=config.models.llm_base_url)
    run_parser.add_argument("--llm-model", default=config.models.llm_model)
    run_parser.add_argument("--api-key", default=config.models.llm_api_key)
    run_parser.add_argument("--max-tokens", type=int, default=4000)
    run_parser.add_argument("--timeout", type=float, default=180.0)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--save-images", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = build_parser(config)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.stage == "run":
        run_answering_benchmark(
            query_set=args.query_set,
            output_dir=args.output_dir,
            run_id=args.run_id,
            context_cap=args.context_cap,
            image_count=args.image_count,
            image_dpi=args.image_dpi,
            enable_thinking=args.enable_thinking,
            enable_visual_retrieval=args.enable_visual_retrieval,
            route_mode=args.route_mode,
            candidate_mode=args.candidate_mode,
            visual_weight=args.visual_weight,
            max_tokens=args.max_tokens,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_api_key=args.api_key,
            request_timeout=args.timeout,
            limit=args.limit,
            save_images=args.save_images,
            dry_run=args.dry_run,
        )
        return

    raise SystemExit(f"Unknown answering benchmark stage: {args.stage}")


if __name__ == "__main__":
    main()
