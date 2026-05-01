---
name: agenthub-python
description: Use AgentHub's Python SDK (`agenthub-python`) to build or modify Python agents, chat runtimes, tests, provider routing, tool-calling loops, tracing, or playground workflows with AutoLLMClient, UniConfig, UniMessage, UniEvent, and AgentHub's universal content model.
---

# AgentHub Python

## Overview

AgentHub provides a universal SDK for calling supported LLM providers through one client, one message shape, and one streaming event shape. Prefer `AutoLLMClient` for application code unless a task explicitly needs a provider-specific client.

## Installation

```bash
uv add agenthub-python
# or
pip install agenthub-python
```

Import the client and shared enums from `agenthub`:

```python
from agenthub import AutoLLMClient, PromptCaching, ThinkingLevel
```

## Supported Models and Environment Variables

`AutoLLMClient` routes by `client_type`, then `CLIENT_TYPE`, then `model.lower()`. Use `client_type` when the model name is an alias, an OpenRouter or SiliconFlow slug, a local vLLM name, or another compatible gateway name.

Supported routing substrings: `gpt-5.4`, `gpt-5.5`, `claude` with `4-6`, `gemini-3-`, `gemini-3.1-`, `glm-5`, `kimi-k2.5`, and `qwen3`.

Set the provider API key before creating the client: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`, or `QWEN3_API_KEY`. Use `ZAI_API_KEY` and `ZAI_BASE_URL` for GLM-5. Optional base URL variables otherwise use the same provider prefix, such as `OPENAI_BASE_URL`. Use `CLIENT_TYPE` as a global routing override and `AGENTHUB_CACHE_DIR` for trace storage.

## Basic Usage

Follow this flow when an agent calls AgentHub:

1. Create an `AutoLLMClient` for the target model after setting the provider API key.
2. Represent the next user turn as a `UniMessage` with `role` and `content_items`; keep provider-native message payloads out of AgentHub.
3. Put request options such as `temperature`, tools, system prompts, and tracing in a `UniConfig` dictionary.
4. Prefer `(async) streaming_response_stateful(message, config)` for chat and agent loops because it stores conversation history inside the client. Use stateless `streaming_response(messages, config)` only when the caller owns the full history.
5. Consume the async stream as `UniEvent` objects. Read each event's `content_items`, `usage_metadata`, and `finish_reason` instead of assuming provider-specific response shapes.

```python
import asyncio
import os

from agenthub import AutoLLMClient

os.environ["OPENAI_API_KEY"] = "your-openai-api-key"


async def main() -> None:
    client = AutoLLMClient(model="gpt-5.5")
    message = {  # UniMessage
        "role": "user",
        "content_items": [{"type": "text", "text": "Say 'Hello, World!'"}],
    }
    config = {"temperature": 1.0}  # UniConfig

    async for event in client.streaming_response_stateful(
        message=message,
        config=config,
    ):
        # event is a UniEvent.
        print(event)


asyncio.run(main())
```

## Data Model

Always send and store AgentHub's universal data model. Do not pass provider-native OpenAI, Anthropic, Gemini, GLM, Kimi, or Qwen message payloads into `AutoLLMClient`.

### UniConfig

`UniConfig` is the provider-independent request configuration passed as the `config` argument to `streaming_response` and `streaming_response_stateful`. Use it for generation options, tools, system prompts, tracing, caching, image output, and TTS settings.

Fields:

```python
config = {
    "max_tokens": 500,
    "temperature": 1.0,
    "tools": [tool_definition],
    "thinking_summary": True,
    "thinking_level": ThinkingLevel.HIGH,
    "tool_choice": "auto",  # "auto", "required", "none", or ["tool_name"]
    "system_prompt": "You are a helpful assistant.",
    "prompt_caching": PromptCaching.ENABLE,
    "image_config": {"aspect_ratio": "4:3", "image_size": "1K"},
    "tts_config": [{"voice": "Kore"}],
    "trace_id": "agent1/conversation_001",
}
```

Use snake_case field names in all AgentHub configs and content items.

### UniMessage

`UniMessage` is the durable conversation record used for API input and stateful history. Pass the next user turn to `streaming_response_stateful`, pass a full list of messages to `streaming_response`, and store assistant responses from `get_history()` in this shape.

`content_items` are typed message parts inside `UniMessage`:

- `text`: natural-language text. Assistant text may include `phase`; signed text may include `signature`.
- `image_url`: image input by URL.
- `inline_data`: binary input or output with `data` bytes and `mime_type`, mainly for image and audio data.
- `thinking`: model reasoning text, optionally signed.
- `inline_thinking`: binary thinking data with `data`, `mime_type`, and optional `signature`; use for image or audio data in the thinking process.
- `tool_call`: complete tool request with `name`, `arguments`, and `tool_call_id`.
- `tool_result`: tool response with `text`, optional `images`, and `tool_call_id`.

Durable history record:

```python
message = {
    "role": "user",
    "content_items": [
        {"type": "text", "text": "Weather in London?"},
        {"type": "tool_result", "text": "15 C", "tool_call_id": "call_123"},
    ],
    "usage_metadata": None,
    "finish_reason": None,
    "created_at": 1694502400000,
}
```

`role` is `user` or `assistant`. User messages normally contain user text or tool results. Assistant messages contain generated text, thinking, tool calls, media output, usage metadata, finish reason, and timestamp.

### UniEvent

`UniEvent` is the streamed response event returned by `streaming_response` and `streaming_response_stateful`. Consume it in the async iterator while the response is being generated; the stateful client folds completed events into assistant `UniMessage` history.

`content_items` in `UniEvent` use the same item types as `UniMessage`, plus `partial_tool_call` for streamed tool-call fragments.

Streamed return shape:

```python
event = {
    "role": "assistant",
    "event_type": "delta",
    "content_items": [{"type": "text", "text": "Hello"}],
    "usage_metadata": None,
    "finish_reason": None,
    "created_at": 1694502400000,
}
```

`event_type` can be `start`, `delta`, `stop`, or `unused`. Intermediate events often have `usage_metadata=None` and `finish_reason=None`. The final event must carry `usage_metadata` and `finish_reason`; AgentHub raises an error if a stream ends without them.

`usage_metadata` appears on final events and stored assistant messages. It normalizes token accounting across providers:

- `cached_tokens`: prompt tokens served from cache.
- `prompt_tokens`: non-cached input tokens.
- `thoughts_tokens`: reasoning or thinking output tokens.
- `response_tokens`: visible response output tokens.
- `input_tokens = cached_tokens + prompt_tokens`.
- `output_tokens = thoughts_tokens + response_tokens`.
- `total_tokens = input_tokens + output_tokens`.

For GPT-5.5, AgentHub maps OpenAI usage as `prompt_tokens = input_tokens - cached_tokens` and `response_tokens = output_tokens - reasoning_tokens`.

## APIs

`AutoLLMClient` is the main class for interacting with the AgentHub SDK. It provides the following methods:

- `(async) streaming_response(messages, config)`: Streams the response of LLMs in a stateless manner.
- `(async) streaming_response_stateful(message, config)`: Streams the response of LLMs in a stateful manner.
- `clear_history()`: Clears the history of the stateful LLM client.
- `get_history()`: Returns the history of the stateful LLM client.
- `set_history(history)`: Replaces the history of the stateful LLM client with a copy of the provided list.

## Usage Example

Tool calling is a two-turn stateful flow: the first `streaming_response_stateful` call may emit a complete `tool_call` with `arguments`; execute the local function, then send a second `UniMessage` containing a `tool_result` with the same `tool_call_id` so the model can continue from the tool output.

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

- Send tool results with the exact `tool_call_id` from the emitted `tool_call`; wait for a complete `tool_call` before executing anything.
- Preserve signed `thinking`, `inline_thinking`, and signed text items in history.

## Tracer Usage

Set `trace_id` in `config` to save trace files under `AGENTHUB_CACHE_DIR` or `cache`:

```python
config = {"trace_id": "agent1/conversation_001"}
```

Browse traces from Python:

```python
from agenthub.integration.tracer import Tracer

Tracer("cache").start_web_server(host="127.0.0.1", port=25750)
```

Or start the tracer from the CLI:

```bash
python -m agenthub.integration.tracer --cache_dir ./cache --host 127.0.0.1 --port 25750
```

## Playground Usage

Use the playground for manual model checks:

```python
from agenthub.integration.playground import start_playground_server

start_playground_server(host="127.0.0.1", port=25751)
```
