# APIs & Usage

`AutoLLMClient` exposes five basic APIs. Prefer the stateful stream for agent loops.

## Initialization

Initialize `AutoLLMClient` in one of three common ways:

```python
# Initialize with model name
client = AutoLLMClient(model="gpt-5.5")

# Optionally specify API key (if not using environment variables)
client = AutoLLMClient(
    model="gpt-5.5",
    api_key="your-openai-api-key",
    base_url="https://api.openai.com/v1",
)

# Use OpenAI Chat Completions-compatible routing explicitly
client = AutoLLMClient(model="custom-model", client_type="openai")
```

## Method signatures

```python
async def streaming_response(messages: list[UniMessage], config: UniConfig) -> AsyncIterator[UniEvent]:
    """Stream one stateless response from a full message list."""

async def streaming_response_stateful(message: UniMessage, config: UniConfig) -> AsyncIterator[UniEvent]:
    """Stream one stateful response and update client history."""

def get_history() -> list[UniMessage]:
    """Return a copy of stateful history."""

def set_history(history: list[UniMessage]) -> None:
    """Replace stateful history with a copy."""

def clear_history() -> None:
    """Clear stateful history."""
```

## Basic Usage

This example asks GPT to call a weather tool, runs the tool, then sends the result back.

```python
import asyncio
from agenthub import AutoLLMClient


def get_weather(location: str) -> str:
    return f"Temperature in {location}: 22 C"


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

    events = []
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [{"type": "text", "text": "What's the weather in London?"}]
        },
        config=config
    ):
        events.append(event)

    tool_call = None
    for event in events:
        for item in event["content_items"]:
            if item["type"] == "tool_call":
                tool_call = item
                break

        if tool_call:
            break

    if tool_call:
        result = get_weather(**tool_call["arguments"])

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


asyncio.run(main())
```

## Notes

Agent loop rules:

- Send every tool result with the exact `tool_call_id` from its originating `tool_call`. Do not invent, normalize, or reuse IDs across unrelated tool calls.
- Preserve `thinking` and `inline_thinking` items. Do not strip `phase` or `signature` fields.
- For embedding models, each `UniMessage` in the `messages` array produces **one embedding vector**. Within a single message, all items in `content_items` are aggregated into a single embedding. Set `embedding_config.dimensions` in the config to control vector size.
