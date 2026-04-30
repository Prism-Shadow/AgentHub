---
name: agenthub-python
description: Use AgentHub's Python SDK for AutoLLMClient agents, streaming, tools, tracing, multimodal Gemini, and provider routing.
---

# AgentHub Python SDK Usage

Use this skill when writing Python agents, CLIs, chat apps, or tests that call the `agenthub-python` package.

## Installation

```bash
uv add agenthub-python
# or
pip install agenthub-python
```

## AutoLLMClient Overview

`AutoLLMClient` is a stateful client that automatically routes requests to the right provider for the selected model. It maintains conversation history and provides a unified interface for different LLM providers.

### Initialization

Create a client by specifying the model name:

```python
from agenthub import AutoLLMClient

client = AutoLLMClient(model="gpt-5.5")
client = AutoLLMClient(model="gpt-5.5", api_key="your-openai-api-key")
client = AutoLLMClient(model="gpt-5.5", api_key="...", base_url="...", client_type="gpt-5.5")
```

The client selects the provider from `client_type`, then `CLIENT_TYPE`, then `model.lower()`.

Supported routing substrings:

- `gemini-3-` or `gemini-3.1-`
- `claude` plus `4-6`
- `gpt-5.4` or `gpt-5.5`
- `glm-5`
- `kimi-k2.5`
- `qwen3`

Use `client_type` for aliases, OpenRouter slugs, SiliconFlow slugs, local vLLM names, or compatible gateways.

### Environment Variables

| Provider | API key | Base URL | Notes |
| --- | --- | --- | --- |
| GPT 5.4 / 5.5 | `OPENAI_API_KEY` | `OPENAI_BASE_URL` | Base URL optional. |
| Claude 4.6 | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` | Use `bedrock://REGION` for Bedrock. |
| Gemini 3 / 3.1 | `GEMINI_API_KEY` | `GEMINI_BASE_URL` | Service-account JSON is accepted when the key starts with `{`. |
| GLM-5 | `ZAI_API_KEY` | `ZAI_BASE_URL` | Defaults to `https://api.z.ai/api/paas/v4/`. |
| Kimi K2.5 | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` | Defaults to `https://api.moonshot.cn/v1`. |
| Qwen3 | `QWEN3_API_KEY` | `QWEN3_BASE_URL` | Defaults to `http://127.0.0.1:8000/v1/`. |
| Routing override | `CLIENT_TYPE` | n/a | Use a recognizable provider family string. |
| Trace cache | `AGENTHUB_CACHE_DIR` | n/a | Defaults to `cache`. |

For GLM-5, use `ZAI_API_KEY` / `ZAI_BASE_URL`; avoid older `GLM_API_KEY` examples.

## Core Methods

### streaming_response

Stateless method that requires passing the full message history on each call:

```python
async for event in client.streaming_response(
    messages=[
        {"role": "user", "content_items": [{"type": "text", "text": "Hello!"}]},
    ],
    config={"temperature": 1.0},
):
    print(event)
```

### streaming_response_stateful

Stateful method that maintains conversation history internally:

```python
import asyncio
from agenthub import AutoLLMClient

async def main():
    client = AutoLLMClient(model="gpt-5.5")
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [{"type": "text", "text": "Hello!"}],
        },
        config={"temperature": 1.0},
    ):
        print(event)

asyncio.run(main())
```

### get_history

Retrieve the conversation history:

```python
history = client.get_history()
```

### clear_history

Clear the conversation history:

```python
client.clear_history()
assert len(client.get_history()) == 0
```

### set_history

Replace the conversation history with a copy of the provided list:

```python
saved_history = client.get_history()
client.set_history(saved_history)
```

## Tool Calling

When using tools, handle `tool_call_id` correctly and use `arguments`, not `argument`:

```python
weather_function = {
    "name": "get_weather",
    "description": "Gets the current weather for a given location.",
    "parameters": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}

events = []
config = {"tools": [weather_function], "tool_choice": "auto"}

async for event in client.streaming_response_stateful(message=user_message, config=config):
    events.append(event)

tool_call = next(
    (item for event in events for item in event["content_items"] if item["type"] == "tool_call"),
    None,
)

if tool_call:
    result = get_weather(**tool_call["arguments"])
    tool_result = {
        "role": "user",
        "content_items": [
            {
                "type": "tool_result",
                "text": result,
                "tool_call_id": tool_call["tool_call_id"],
            }
        ],
    }
    async for event in client.streaming_response_stateful(message=tool_result, config=config):
        print(event)
```

Ignore `partial_tool_call` for execution until AgentHub materializes a complete `tool_call`.

## Message Format

### UniMessage Structure

```python
{
    "role": "user" | "assistant",
    "content_items": [
        {"type": "text", "text": "Hello"},
        {"type": "image_url", "image_url": "https://..."},
        {"type": "inline_data", "mime_type": "image/png", "data": b"..."},
        {"type": "thinking", "thinking": "...", "signature": "optional"},
        {"type": "inline_thinking", "mime_type": "image/png", "data": b"...", "signature": "optional"},
        {"type": "tool_call", "name": "get_weather", "arguments": {"location": "London"}, "tool_call_id": "call_abc123"},
        {"type": "tool_result", "text": "London is 22 C today.", "tool_call_id": "call_abc123"},
    ],
}
```

Keep signed `thinking` and `inline_thinking` items in history when models emit them. They may be required for interleaved thinking across tool calls.

### UniEvent Structure

```python
{
    "role": "assistant",
    "event_type": "delta",
    "content_items": [{"type": "text", "text": "Hello"}],
    "usage_metadata": None,
    "finish_reason": None,
    "created_at": 1694502400000,
}
```

`usage_metadata` contains `cached_tokens`, `prompt_tokens`, `thoughts_tokens`, and `response_tokens`.

### Tool Response with tool_call_id

When responding to a tool call, include the exact `tool_call_id`:

```python
{
    "role": "user",
    "content_items": [
        {
            "type": "tool_result",
            "text": "London is 22 C today.",
            "tool_call_id": "call_abc123",
        }
    ],
}
```

## Configuration Options

```python
from agenthub import PromptCaching, ThinkingLevel

config = {
    "max_tokens": 500,
    "temperature": 1.0,
    "tools": [tool_definition],
    "thinking_summary": True,
    "thinking_level": ThinkingLevel.HIGH,
    "tool_choice": "auto",  # "auto", "required", "none", or ["tool_name"]
    "system_prompt": "You are a helpful assistant",
    "prompt_caching": PromptCaching.ENABLE,
    "image_config": {"aspect_ratio": "4:3", "image_size": "1K"},
    "tts_config": [{"voice": "Kore"}],
    "trace_id": "agent1/conversation_001",
}
```

## Multimodal Usage

Image input:

```python
message = {
    "role": "user",
    "content_items": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": "https://example.com/image.png"},
    ],
}
```

Gemini image generation requires an image-capable Gemini model, such as `gemini-3.1-flash-image-preview`; `image_config` configures image output for models with that capability. Generated images return image `inline_data`:

```python
config = {"image_config": {"aspect_ratio": "4:3", "image_size": "1K"}}
```

Gemini TTS requires a TTS Gemini model, such as `gemini-3.1-flash-tts-preview`; `tts_config` configures audio output for TTS models. TTS returns audio `inline_data` and should use text input only:

```python
config = {"tts_config": [{"voice": "Kore"}]}
```

## Conversation Tracing

AgentHub provides a built-in `Tracer` to save and browse conversation history. When you specify `trace_id`, conversations are saved to JSON and TXT formats.

### Basic Usage

```python
config = {"trace_id": "agent1/conversation_001"}

async for event in client.streaming_response_stateful(
    message={"role": "user", "content_items": [{"type": "text", "text": "Hello"}]},
    config=config,
):
    pass
```

The default cache directory is `cache`; change it with `AGENTHUB_CACHE_DIR`.

### Browsing Traces with Web Interface

```python
from agenthub.integration.tracer import Tracer

Tracer("path/to/cache").start_web_server(host="127.0.0.1", port=25750)
```

Or use the CLI:

```bash
python -m agenthub.integration.tracer --cache_dir ./cache --host 127.0.0.1 --port 25750
```

### Test with Playground

```python
from agenthub.integration.playground import start_playground_server

start_playground_server()
```

Or use the CLI:

```bash
python -m agenthub.integration.playground --host 127.0.0.1 --port 25751
```

## Migration Checklist

1. Replace provider-native clients with `AutoLLMClient`.
2. Convert messages to `content_items`.
3. Move provider-specific options into `config`.
4. Preserve `tool_call_id`, signed thinking, usage metadata, and timestamps in durable history.
5. Use `client_type` plus `base_url` for aliases and compatible gateways.
6. Add `trace_id` around important agent runs.
