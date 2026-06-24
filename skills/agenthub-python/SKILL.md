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

## Model Selection

Use exact model IDs. If a model ID is not listed, ask the user to confirm the exact ID before using it.

| Family | Provider | Model IDs | API Key | Base URL |
| --- | --- | --- | --- | --- |
| Gemini 3 | Official / Vertex AI | `gemini-3.1-pro-preview`, `gemini-3.5-flash`, `gemini-3.1-flash-lite` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Gemini 3 Image | Official / Vertex AI | `gemini-3.1-flash-image-preview`, `gemini-3-pro-image-preview` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Gemini 3 TTS | Official / Vertex AI | `gemini-3.1-flash-tts-preview` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Gemini Embedding | Official / Vertex AI | `gemini-embedding-2` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Claude 4.6 | Official / ModelVerse | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 4.6 | Bedrock | `global.anthropic.claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 4.7 | Official / ModelVerse | `claude-opus-4-7` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 4.7 | Bedrock | `global.anthropic.claude-opus-4-7` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 4.8 | Official / ModelVerse | `claude-opus-4-8` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 4.8 | Bedrock | `global.anthropic.claude-opus-4-8` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 5 | Official / ModelVerse | `claude-fable-5` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 5 | Bedrock | `global.anthropic.claude-fable-5` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| GPT 5.4 | Official / ModelVerse | `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| GPT 5.5 | Official / ModelVerse | `gpt-5.5` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| OpenAI Embedding | Official | `text-embedding-3-small`, `text-embedding-3-large` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| Kimi-K2.6 | Official | `kimi-k2.6` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| Kimi-K2.6 | OpenRouter | `moonshotai/kimi-k2.6` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| Kimi-K2.6 | SiliconFlow | `Pro/moonshotai/Kimi-K2.6` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| DeepSeek V4 | Official | `deepseek-v4-pro`, `deepseek-v4-flash` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| DeepSeek V4 | OpenRouter | `deepseek/deepseek-v4-pro`, `deepseek/deepseek-v4-flash` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| DeepSeek V4 | SiliconFlow | `deepseek-ai/DeepSeek-V4-Pro`, `deepseek-ai/DeepSeek-V4-Flash` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| GLM-5.1 | Official | `glm-5.1` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| GLM-5.1 | OpenRouter | `z-ai/glm-5.1` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| GLM-5.1 | SiliconFlow | `Pro/zai-org/GLM-5.1` | `ZAI_API_KEY` | `ZAI_BASE_URL` |

Common gateway base URLs:

- OpenRouter: `https://openrouter.ai/api/v1`
- SiliconFlow: `https://api.siliconflow.cn/v1`
- ModelVerse: `https://api.modelverse.cn/v1` (`https://api.modelverse.cn/` for Claude)
- vLLM: `http://127.0.0.1:8000/v1/`

For models accessed through OpenAI-compatible APIs (e.g., Qwen series models via SiliconFlow or OpenRouter), pass `client_type="openai"` (`client_type="openai-embedding"` for embedding endpoints), and set `OPENAI_API_KEY` and `OPENAI_BASE_URL`:

```python
client = AutoLLMClient(model="Qwen/Qwen3-Embedding-0.6B", client_type="openai-embedding")
```

## Data Models

AgentHub uses `UniConfig`, `UniMessage`, and `UniEvent` to represent request options, conversation history, and streamed outputs across providers.

### UniConfig

`UniConfig` is the request config for `streaming_response` and `streaming_response_stateful`. All fields are optional.

```python
config = {
    "max_tokens": 1024,
    "temperature": 1.0,
    "tools": [{
        "name": "get_weather",
        "description": "Get weather.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name."}},
            "required": ["location"],
        },
    }],
    "tool_choice": "auto",
    "thinking_summary": True,
    "thinking_level": "high",
    "system_prompt": "You are helpful.",
    "prompt_caching": "enable",
    "image_config": {"aspect_ratio": "4:3", "image_size": "1K"},
    "tts_config": [{"voice": "Kore"}],
    "embedding_config": {"dimensions": 768},
    "trace_id": "agent1/conversation_001",
}
```

Fields:

- `max_tokens` (`int`): Output token limit.
- `temperature` (`float`): Sampling temperature; support varies by model.
- `tools` (`list[ToolSchema]`): Tools with `name`, `description`, and optional JSON Schema `parameters`.
- `thinking_summary` (`bool`): Request a thinking summary when supported.
- `thinking_level` (`ThinkingLevel`): `none`, `low`, `medium`, `high`, or `xhigh`.
- `tool_choice` (`ToolChoice`): `auto`, `required`, `none`, or a list of tool names; support varies by model.
- `system_prompt` (`str`): System instruction text.
- `prompt_caching` (`PromptCaching`): `enable`, `disable`, or `enhance`.
- `image_config` (`ImageConfig`): `aspect_ratio` (`1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`, `16:9`, `21:9`) and `image_size` (`1K`, `2K`).
- `tts_config` (`list[SpeakerConfig]`): Voice config; each item has `voice` and optional `speaker`.
- `embedding_config` (`EmbeddingConfig`): Embedding config, currently `dimensions`.
- `trace_id` (`str`): Stable ID for tracer output.

### UniMessage

`UniMessage` is the durable message shape used in history.

```python
message = {
    "role": "user",
    "content_items": [
        {"type": "text", "text": "Hello", "phase": None, "signature": "sig"},
        {"type": "image_url", "image_url": "https://example.com/image.jpg"},
        {"type": "inline_data", "data": b"...", "mime_type": "image/png", "signature": "sig"},
        {"type": "thinking", "thinking": "Reasoning", "signature": "sig"},
        {"type": "inline_thinking", "data": b"...", "mime_type": "image/png", "signature": "sig"},
        {"type": "tool_call", "name": "get_weather", "arguments": {"location": "Paris"}, "tool_call_id": "call_1", "signature": "sig"},
        {"type": "tool_result", "text": "22 C", "tool_call_id": "call_1"},
        {"type": "embedding", "embedding": [0.1, 0.2]},
    ],
}
```

Fields:

- `role` (`Role`): `user` or `assistant`.
- `content_items` (`list[ContentItem]`): Message payload.
- `usage_metadata` (`UsageMetadata | None`): Optional token counts on completed assistant messages.
- `finish_reason` (`FinishReason | None`): `stop`, `length`, `tool_call`, `unknown`, or `None`.
- `created_at` (`int`): Unix milliseconds.

Content items:

- `text`: Text chunk; `phase` marks sub-stage; `signature` verifies signed content.
- `image_url`: Image URL or data URI.
- `inline_data`: Inline media bytes with MIME type; may carry `signature`.
- `thinking`: Text reasoning content; may carry `signature`.
- `inline_thinking`: Binary reasoning artifact; may carry `signature`.
- `tool_call`: Complete model tool request with name, args, ID, and optional `signature`.
- `tool_result`: Tool output text for a `tool_call_id`; may include image URLs.
- `embedding`: Numeric embedding vector.

Preserve `phase` and `signature`; never drop either field.

### UniEvent

`UniEvent` is the streamed output shape. Read token counts from `usage_metadata` here.

```python
event = {
    "role": "assistant",
    "event_type": "delta",
    "content_items": [
        {"type": "partial_tool_call", "name": "get_weather", "arguments": "{\"location\":\"Par", "tool_call_id": "call_1"}
    ],
    "usage_metadata": {"cached_tokens": 0, "prompt_tokens": 10, "thoughts_tokens": None, "response_tokens": 1},
    "finish_reason": None,
    "created_at": 1694502400000,
}
```

Fields:

- `role` (`Role`): `user` or `assistant`.
- `event_type` (`EventType`): `start`, `delta`, `stop`, or `unused`.
- `content_items` (`list[PartialContentItem]`): Stream payload; includes `ContentItem` plus `partial_tool_call`.
- `usage_metadata` (`UsageMetadata | None`): Token counts: `cached_tokens`, `prompt_tokens`, `thoughts_tokens`, `response_tokens`.
  Token math: `input = cached_tokens + prompt_tokens`; `output = thoughts_tokens + response_tokens`; treat `None` as `0`.
- `finish_reason` (`FinishReason | None`): `stop`, `length`, `tool_call`, `unknown`, or `None`.
- `created_at` (`int`): Unix milliseconds.

Event-only content item:

- `partial_tool_call`: Streaming tool-call fragment with `name`, partial JSON `arguments`, and `tool_call_id`.

### Tool-Call Streaming Protocol

Every provider streams a tool call as the same ordered sequence of events, so consumers handle all models the same way. A single tool call is announced, then its arguments stream in, then it is delivered complete:

1. **Announce (name + id first).** The first event for a tool call carries a `partial_tool_call` whose `name` and `tool_call_id` are non-empty and whose `arguments` is `""`. The tool's identity always arrives **before** any argument bytes.
2. **Argument deltas.** Subsequent `delta` events carry `partial_tool_call` items whose `arguments` holds the next fragment of the arguments JSON **string** (`name` and `tool_call_id` are empty `""` on these fragments). Concatenate fragments in order to reconstruct the JSON.
3. **Complete call (last).** When the arguments are complete, one final event carries a complete `tool_call` item: `name`, `tool_call_id`, and `arguments` parsed into a **dict** (no longer a string).

For **consecutive or parallel tool calls**, each new tool call restarts at step 1 with its own `name` and `tool_call_id` — one call's arguments never bleed into the next. So a two-tool turn streams as: announce A → arg deltas A → complete A → announce B → arg deltas B → complete B.

```python
# Drive a tool call from a stream: read complete calls from `tool_call` items.
tool_calls = []
async for event in client.streaming_response_stateful(message=msg, config=config):
    for item in event["content_items"]:
        if item["type"] == "partial_tool_call":
            # steps 1-2: live progress only; arguments is a partial JSON string. Do not execute yet.
            ...
        elif item["type"] == "tool_call":
            # step 3: complete; arguments is a parsed dict. Safe to execute.
            tool_calls.append(item)
```

Execute tools only from the complete `tool_call` items (step 3); use `partial_tool_call` items (steps 1-2) for live progress display. Send each tool result back with the exact `tool_call_id` from its `tool_call`.

## APIs

`AutoLLMClient` exposes five basic APIs. Prefer the stateful stream for agent loops.

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

## Tracer

Tracer saves trace files and serves a local UI for inspecting conversations.

Set `trace_id` to save trace files:

```python
from agenthub import AutoLLMClient

client = AutoLLMClient(model="gpt-5.5")

config = {"trace_id": "agent1/conversation_001"}

async for event in client.streaming_response_stateful(
    message={"role": "user", "content_items": [{"type": "text", "text": "Hello"}]},
    config=config
):
    pass
```

Default cache dir: `cache`, or `AGENTHUB_CACHE_DIR`. For `trace_id="agent1/conversation_001"`, AgentHub writes:

- `cache/agent1/conversation_001.json`: Structured trace data with the full history and config.
- `cache/agent1/conversation_001.txt`: Human-readable conversation transcript.

Browse traces:

```python
from agenthub.integration.tracer import Tracer

Tracer().start_web_server(host="127.0.0.1", port=25750)
```

Or CLI:

```bash
python -m agenthub.integration.tracer --cache_dir ./cache --host 127.0.0.1 --port 25750
```

Open Tracer at `http://127.0.0.1:25750`.

## Playground

Playground starts a local chat UI for manual model checks.

Start Playground for manual chat:

```python
from agenthub.integration.playground import start_playground_server

start_playground_server(host="127.0.0.1", port=25751)
```

Open Playground at `http://127.0.0.1:25751`.

## Notes

Agent loop rules:

- Send every tool result with the exact `tool_call_id` from its originating `tool_call`. Do not invent, normalize, or reuse IDs across unrelated tool calls.
- Preserve `thinking` and `inline_thinking` items. Do not strip `phase` or `signature` fields.
- For embedding models, each `UniMessage` in the `messages` array produces **one embedding vector**. Within a single message, all items in `content_items` are aggregated into a single embedding. Set `embedding_config.dimensions` in the config to control vector size.
