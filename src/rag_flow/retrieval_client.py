from __future__ import annotations

import argparse
import time
from typing import Any

from .config import AppConfig


def test_query(
    api_url: str,
    query_text: str,
    *,
    print_context: bool = True,
    timeout: int = 120,
    api_key: str = "",
) -> None:
    import requests

    start = time.time()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    response = requests.post(api_url, json={"query": query_text}, headers=headers, timeout=timeout)
    response.raise_for_status()
    elapsed = time.time() - start
    result: dict[str, Any] = response.json()

    print(f"\nQuery: {query_text}")
    print(f"Retrieved in {elapsed:.2f}s")
    print(f"Top hit page: {result.get('hit_page')}")

    hits = result.get("all_hits", [])
    if hits:
        print("\nHits:")
        for hit in hits:
            tag = "visual-continuation" if hit.get("is_continuation") else "text-page"
            print(
                f"  Rank {hit.get('rank'):>2} | "
                f"Page {hit.get('page_number'):>4} | "
                f"Score {hit.get('score', 0):.4f} | {tag}"
            )
    else:
        print("\nNo hits returned.")

    if print_context:
        print("\nContext:")
        print(result.get("context", ""))


def interactive(api_url: str, *, print_context: bool, timeout: int, api_key: str) -> None:
    print("RAG Flow retrieval tester. Type exit or quit to stop.")
    while True:
        try:
            query = input("\nQuery: ").strip()
        except KeyboardInterrupt:
            print()
            break
        if query.lower() in {"exit", "quit", "q"}:
            break
        if query:
            try:
                test_query(api_url, query, print_context=print_context, timeout=timeout, api_key=api_key)
            except Exception as exc:
                if exc.__class__.__name__ == "ConnectionError":
                    print(f"Cannot connect to {api_url}. Is `rag-flow retriever` running?")
                else:
                    print(f"Request failed: {exc}")


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Test the retrieval API from the terminal.")
    parser.add_argument("query", nargs="*", help="Query text. If omitted, starts interactive mode.")
    parser.add_argument("--url", default=config.server.retriever_url, help="Retriever /retrieve URL.")
    parser.add_argument("--api-key", default=config.server.retriever_api_key, help="Bearer token for /retrieve.")
    parser.add_argument("--no-context", action="store_true", help="Only print hit metadata.")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)

    query = " ".join(args.query).strip()
    if query:
        test_query(args.url, query, print_context=not args.no_context, timeout=args.timeout, api_key=args.api_key)
    else:
        interactive(args.url, print_context=not args.no_context, timeout=args.timeout, api_key=args.api_key)


if __name__ == "__main__":
    main()
