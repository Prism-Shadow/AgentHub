---
name: agenthub-python
description: Guidance for using the AgentHub Python SDK (`agenthub-python`). Use when developing agents that call different LLM APIs, need a unified interface for LLM providers, mention AgentHub, request `agenthub-python`, or already import it.
---

# AgentHub Python

AgentHub is a unified SDK for calling LLMs across providers with shared data models, tool calling, tracing, and playground support.

## Installation

```bash
uv add agenthub-python
# or
pip install agenthub-python
```

For model IDs, API keys, and base URLs, see [Model selection](reference/models.md).

## Basic Usage

This example asks GPT to call a weather tool, runs the tool, then sends the result back.

```python
import asyncio
from agenthub import AutoLLMClient


def get_weather(location: str) -> str:
    return f"Temperature in {location}: 22 C"


# Map tool names to their implementations so calls can be dispatched by name.
TOOLS = {"get_weather": get_weather}


async def main():
    weather_function = {
        "name": "get_weather",
        "description": "Gets the current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["location"]
        }
    }

    client = AutoLLMClient(model="gpt-5.5")
    config = {"tools": [weather_function]}

    tool_call = None
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [{"type": "text", "text": "What's the weather in London?"}]
        },
        config=config
    ):
        for item in event["content_items"]:
            if item["type"] == "tool_call":  # collected as the stream arrives; no second pass
                tool_call = item

    if tool_call:
        # Dispatch by tool name instead of hardcoding the function.
        result = TOOLS[tool_call["name"]](**tool_call["arguments"])

        async for event in client.streaming_response_stateful(
            message={
                "role": "user",
                "content_items": [
                    {
                        "type": "tool_result",
                        "text": result,
                        "tool_call_id": tool_call["tool_call_id"]
                    }
                ]
            },
            config=config
        ):
            print(event)
            # Example printed event:
            # {'role': 'assistant', 'event_type': 'delta',
            #  'content_items': [{'type': 'text', 'text': 'The weather in London is 22 C.'}],
            #  'usage_metadata': None, 'finish_reason': None, 'created_at': 1694502400000}


asyncio.run(main())
```

## Notes

Agent loop rules:

- Send every tool result with the exact `tool_call_id` from its originating `tool_call`. Do not invent, normalize, or reuse IDs across unrelated tool calls.
- Preserve `thinking` and `inline_thinking` items. Do not strip `phase` or `signature` fields.
- For embedding models, each `UniMessage` in the `messages` array produces **one embedding vector**. Within a single message, all items in `content_items` are aggregated into a single embedding. Set `embedding_config.dimensions` in the config to control vector size.

## Reference

- [Model selection](reference/models.md) — model IDs, API keys, base URLs, and OpenAI-compatible routing.
- [Data models](reference/data-models.md) — `UniConfig`, `UniMessage`, `UniEvent`, and the tool-call streaming protocol.
- [APIs](reference/api.md) — client initialization and method signatures.
- [Tracer & Playground](reference/integrations.md) — local tracing UI and the manual chat playground.
