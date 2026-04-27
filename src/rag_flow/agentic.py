from __future__ import annotations

import json
import os
from typing import Any

import requests


def get_current_weather(location: str) -> str:
    if "Beijing" in location or "北京" in location:
        return json.dumps({"temperature": "15C", "condition": "sunny", "wind": "light"})
    return json.dumps({"temperature": "22C", "condition": "cloudy"})


def run_agent(user_query: str, *, api_url: str, api_key: str, model: str) -> None:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "City name."}},
                    "required": ["location"],
                },
            },
        }
    ]
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_query}]

    for _ in range(3):
        payload = {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto"}
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        messages.append(message)

        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                if function_name == "get_weather":
                    result = get_current_weather(args["location"])
                else:
                    result = json.dumps({"error": "Tool not found"})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": result,
                    }
                )
            continue

        print(message.get("content", ""))
        break


def main() -> None:
    api_url = os.environ.get("RAG_FLOW_REMOTE_LLM_URL", "")
    api_key = os.environ.get("RAG_FLOW_REMOTE_LLM_API_KEY", "")
    model = os.environ.get("RAG_FLOW_REMOTE_LLM_MODEL", "qwen3.5-35b-instruct")
    if not api_url or not api_key:
        raise SystemExit("Set RAG_FLOW_REMOTE_LLM_URL and RAG_FLOW_REMOTE_LLM_API_KEY first.")
    run_agent("北京今天天气怎么样？出门需要带伞吗？", api_url=api_url, api_key=api_key, model=model)


if __name__ == "__main__":
    main()
