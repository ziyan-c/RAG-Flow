from __future__ import annotations

import argparse
import sys

import requests
from openai import OpenAI

from .config import AppConfig


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Chat with the manual through the retrieval API and an OpenAI-compatible LLM.")
    parser.add_argument("--retriever-url", default=config.server.retriever_url)
    parser.add_argument("--llm-base-url", default=config.models.llm_base_url)
    parser.add_argument("--llm-model", default=config.models.llm_model)
    parser.add_argument("--api-key", default=config.models.llm_api_key)
    parser.add_argument("--max-tokens", type=int, default=config.models.llm_max_tokens)
    args = parser.parse_args(argv)

    llm_client = OpenAI(api_key=args.api_key, base_url=args.llm_base_url)
    chat_history: list[dict[str, str]] = []

    print("RAG Flow chat is ready. Type q, quit, or exit to stop.")
    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in {"q", "quit", "exit"}:
            print("Goodbye.")
            break
        if not user_query:
            continue

        try:
            response = requests.post(args.retriever_url, json={"query": user_query}, timeout=120)
            response.raise_for_status()
            data = response.json()
            context = data["context"]
            print(f"\n(Top reference page: {data['hit_page']})\n")

            current_turn_prompt = (
                f"[Latest Retrieved Data]\n{context}\n\n"
                f"[User's Latest Question]\n{user_query}\n\n"
                "Instruction: Answer based on the retrieved data and chat history. "
                "If the answer is not present in the provided data, say so clearly. "
                "Always cite source page numbers for factual claims. "
                "Answer in the same human language as the user's question."
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a senior technical expert assistant. "
                        "You answer questions only from the provided reference materials."
                    ),
                },
                *chat_history,
                {"role": "user", "content": current_turn_prompt},
            ]

            stream = llm_client.chat.completions.create(
                model=args.llm_model,
                messages=messages,
                max_tokens=args.max_tokens,
                stream=True,
                top_p=0.85,
                temperature=1.0,
            )

            assistant_reply = ""
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    sys.stdout.write(f"\033[90m{reasoning}\033[0m")
                    sys.stdout.flush()
                if delta.content:
                    sys.stdout.write(delta.content)
                    sys.stdout.flush()
                    assistant_reply += delta.content
            print("\n")

            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": assistant_reply.strip()})
            if len(chat_history) > 12:
                chat_history = chat_history[-12:]

        except requests.exceptions.ConnectionError as exc:
            print(f"Connection failed. Check the retriever and LLM services. Details: {exc}")
        except Exception as exc:
            print(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()
