---
name: agenthub-python
description: Guidance for using the AgentHub Python SDK (`agenthub-python`). Use this skill when developing agents that need to invoke different LLM APIs, or when a unified interface for different LLM providers is required. Also use this skill if the user mentions AgentHub, requests the use of the `agenthub-python` library, or when `agenthub-python` is already imported in the project.
---

# AgentHub Python

This skill provides guidance for using AgentHub's Python SDK (`agenthub-python`). AgentHub is a unified and precise LLM API hub for autonomous agents, providing a consistent interface across LLMs, reliable multi-step tool-call handling, and lightweight tracing for debugging and auditing executions.

## Installation

```bash
uv add agenthub-python
# or
pip install agenthub-python
```

## How to specify models

Choose a model ID from the table, set the API key and base URL using the environment variables that the routed Python client reads automatically, then create the client with `AutoLLMClient(model=model_id)`. Alternatively, pass them explicitly without relying on environment variables: `AutoLLMClient(model=model_id, api_key=your_api_key, base_url=your_base_url)`.

| Model name | Vendor | Model IDs | API Key | Base URL |
| --- | --- | --- | --- | --- |
| Gemini 3 / Gemini 3.1 LLM | Official / Google Vertex AI | `gemini-3-flash-preview`, `gemini-3.1-flash-lite`, `gemini-3.1-flash-lite-preview`, `gemini-3.1-pro-preview` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Gemini 3.1 image generation | Official / Google Vertex AI | `gemini-3.1-flash-image-preview`, `gemini-3-pro-image-preview` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Gemini 3.1 TTS | Official / Google Vertex AI | `gemini-3.1-flash-tts-preview` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Claude 4.6 | Official / ModelVerse | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Claude 4.6 | Amazon Bedrock | `global.anthropic.claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| GPT-5.4 / GPT-5.5 | Official / ModelVerse | `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.5` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| GLM-5 | Official | `glm-5` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| GLM-5 | OpenRouter | `z-ai/glm-5` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| GLM-5 | SiliconFlow | `Pro/zai-org/GLM-5` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| Kimi-K2.5 | Official | `kimi-k2.5` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| Kimi-K2.5 | OpenRouter | `moonshotai/kimi-k2.5` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| Kimi-K2.5 | SiliconFlow | `Pro/moonshotai/Kimi-K2.5` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| Qwen3 | OpenRouter | `qwen/qwen3-8b`, `qwen/qwen3-30b-a3b-thinking-2507` | `QWEN3_API_KEY` | `QWEN3_BASE_URL` |
| Qwen3 | SiliconFlow | `Qwen/Qwen3-8B` | `QWEN3_API_KEY` | `QWEN3_BASE_URL` |

## Basic Usage

Use `AutoLLMClient` and its methods for all LLM interaction and conversation tasks.

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
        # UniEvent
        print(event)


asyncio.run(main())
```

## APIs

`AutoLLMClient` is the main class for interacting with the AgentHub SDK. Prefer `streaming_response_stateful` for agent loops.

After declaring `client = AutoLLMClient(...)`, you can use the following `AutoLLMClient` methods:

```python
async def streaming_response(
    messages: list[UniMessage],
    config: UniConfig,
) -> AsyncIterator[UniEvent]:
    """Stream one response without using client history.

    Call as `async for event in client.streaming_response(messages, config)`.
    Provide the complete ordered conversation in `messages`; the client yields
    universal stream events and leaves stateful history unchanged.

    Args:
        messages: Complete ordered conversation to send for this request.
        config: Request options such as tools, temperature, or trace_id.

    Yields:
        UniEvent: Stream event in AgentHub's universal event format.
    """
```


```python
async def streaming_response_stateful(
    message: UniMessage,
    config: UniConfig,
) -> AsyncIterator[UniEvent]:
    """Stream one response while the client manages conversation history.

    Call as `async for event in client.streaming_response_stateful(message, config)`.
    Pass only the next message; the client combines it with stored history and
    records the completed turn after a successful response.

    Args:
        message: New user or assistant message for the next turn.
        config: Request options such as tools, temperature, or trace_id.

    Yields:
        UniEvent: Stream event in AgentHub's universal event format.
    """
```


```python
def clear_history() -> None:
    """Reset the stored stateful conversation history.

    Call as `client.clear_history()` before starting an unrelated conversation
    with the same model client.

    Returns:
        None.
    """
```


```python
def get_history() -> list[UniMessage]:
    """Return the current stateful conversation history.

    Call as `history = client.get_history()` to inspect or persist messages
    accumulated by `streaming_response_stateful`.

    Returns:
        list[UniMessage]: Copy of the client's stored conversation list.
    """
```


```python
def set_history(history: list[UniMessage]) -> None:
    """Replace the stored stateful conversation history.

    Call as `client.set_history(history)` when restoring a conversation or
    transferring history between clients. Pass a complete ordered message list.

    Args:
        history: Complete ordered conversation to store on the client.

    Returns:
        None.
    """
```

## Data Model: UniConfig, UniMessage and UniEvent

Use AgentHub's universal data model instead of provider-native response payloads.

### UniConfig

`UniConfig` is the request configuration passed as the `config` argument to `streaming_response` and `streaming_response_stateful`.

Example:

```python
config = {
  "max_tokens": 1024,
  "temperature": 1.0,
  "tools": [
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
          "type": "object",
          "properties": {
              "location": {
                  "type": "string",
                  "description": "The city and state, e.g. San Francisco, CA"
              }
          },
          "required": ["location"]
      }
    }
  ],
  "thinking_summary": True,
  "thinking_level": "high",
  "tool_choice": "auto",
  "system_prompt": "You are a helpful assistant.",
  "prompt_caching": "enable",
  "image_config": {"aspect_ratio": "4:3", "image_size": "1K"},
  "tts_config": [{"voice": "Kore"}],
  "trace_id": "agent1/conversation_001",
}
```

All fields of `UniConfig` are optional.

Fields:

- `max_tokens` (`int`): Output-token limit. Caps generated output length when the provider supports it.
- `temperature` (`float`): Sampling temperature. Controls randomness when the provider supports it.
- `tools` (`list[ToolSchema]`): List of tools available to the model. Each tool requires `name: str` and `description: str`, and may include `parameters: dict[str, Any]` as a JSON Schema Python dict.
- `thinking_summary` (`bool`): Indicates whether the model should return its thinking process.
- `thinking_level` (`ThinkingLevel`): Reasoning effort level, one of `"none"`, `"low"`, `"medium"`, or `"high"`.
- `tool_choice` (`Literal["auto", "required", "none"] | list[str]`): Tool-calling configuration, one of `"auto"`, `"required"`, `"none"`, or a list of allowed tool names such as `["tool_a"]`. Only meaningful when `tools` is provided.
- `system_prompt` (`str`): System instruction text.
- `prompt_caching` (`PromptCaching`): Prompt cache mode, one of `"enable"`, `"disable"`, or `"enhance"`.
- `image_config` (`ImageConfig`): Image-generation configuration with optional `aspect_ratio: AspectRatio` and `image_size: ImageSize`. `AspectRatio` is one of `"1:1"`, `"2:3"`, `"3:2"`, `"3:4"`, `"4:3"`, `"9:16"`, `"16:9"`, or `"21:9"`; `ImageSize` is `"1K"` or `"2K"`.
- `tts_config` (`list[SpeakerConfig]`): Speech-generation configuration. Each item requires `voice: str` and may include `speaker: str`; use one item for single-speaker TTS and two items with `speaker` names for multi-speaker TTS.
- `trace_id` (`str`): Stable trace identifier. Saves conversation history under this ID for the tracer.

### UniMessage

`UniMessage` is the durable conversation message shape, passed as `message` to `streaming_response_stateful`, as an element of `messages` to `streaming_response`, and returned by `get_history`.

Example:

```python
message = {
  "role": "user",
  "content_items": [
    {"type": "text", "text": "How are you doing?"},
    {"type": "image_url", "image_url": "https://example.com/image.jpg"},
    {"type": "inline_data", "mime_type": "image/jpeg", "data": b"<bytes>"},
    {"type": "thinking", "thinking": "I am thinking.", "signature": "0x123456"},
    {"type": "inline_thinking", "mime_type": "image/jpeg", "data": b"<bytes>"},
    {"type": "tool_call", "name": "math", "arguments": {"expression": "2 + 3"}, "tool_call_id": "123"},
    {"type": "tool_result", "text": "2 + 3 = 5", "images": [], "tool_call_id": "123"}
  ]
}
```

Fields:

- `role` (`Role`): Message author, either `"user"` or `"assistant"`.
- `content_items` (`list[ContentItem]`): Durable message payload stored in history and trace records.
- `usage_metadata` (`UsageMetadata | None`): Optional token usage for completed assistant messages.
- `finish_reason` (`FinishReason | None`): Stop reason for a completed assistant message, one of `"stop"`, `"length"`, `"tool_call"`, `"unknown"`, or `None`.
- `created_at` (`int`): Message creation timestamp in milliseconds since epoch.

Durable `content_items` types:

- `text`: Plain text content. Required properties: `type: Literal["text"]`, `text: str`.
- `image_url`: External image URL or data URI. Required properties: `type: Literal["image_url"]`, `image_url: str`.
- `inline_data`: Inline binary image or audio content. Required properties: `type: Literal["inline_data"]`, `data: bytes`, `mime_type: str`.
- `thinking`: Text reasoning content returned by the model. Required properties: `type: Literal["thinking"]`, `thinking: str`.
- `inline_thinking`: Binary reasoning artifact, such as generated-image thinking data. Required properties: `type: Literal["inline_thinking"]`, `data: bytes`, `mime_type: str`.
- `tool_call`: Complete tool invocation. Required properties: `type: Literal["tool_call"]`, `name: str`, `arguments: dict[str, Any]`.
- `tool_result`: Tool execution result to send back to the model. Required properties: `type: Literal["tool_result"]`, `text: str`, `tool_call_id: str`. Optional property: `images: list[str]`.

The `text`, `inline_data`, `thinking`, `inline_thinking`, `tool_call`, and `partial_tool_call` items may carry an optional `signature` field. Do not strip or modify `signature` fields.

### UniEvent

`UniEvent` is the streamed output shape, yielded from `streaming_response` and `streaming_response_stateful`.

Example:

```python
event = {
  "role": "assistant",
  "event_type": "start",
  "content_items": [
    {"type": "partial_tool_call", "name": "math", "arguments": "", "tool_call_id": "123"}
  ],
  "usage_metadata": {
    "cached_tokens": None,
    "prompt_tokens": 10,
    "thoughts_tokens": None,
    "response_tokens": 1
  },
  "finish_reason": None,
  "created_at": 1694502400000
}
```

Fields:

- `role` (`Role`): Event author, either `"user"` or `"assistant"`.
- `event_type` (`EventType`): Stream lifecycle marker, including `start`, `stop`, and `unused`. Indicates where the event sits in the stream lifecycle.
- `content_items` (`list[PartialContentItem]`): Stream payload. Same as `UniMessage.content_items`, plus event-only `partial_tool_call`.
- `usage_metadata` (`UsageMetadata | None`): Stream token accounting, or `None`.
- `finish_reason` (`FinishReason | None`): Stream stop reason, one of `"stop"`, `"length"`, `"tool_call"`, `"unknown"`, or `None`.
- `created_at` (`int`): Event creation timestamp in milliseconds since epoch.

Event-only `content_items` type:

- `partial_tool_call`: Streamed tool-call fragment. Required properties: `type: Literal["partial_tool_call"]`, `name: str`, `arguments: str`, `tool_call_id: str`. `arguments` carries partial JSON string content, and `tool_call_id` links the later complete `tool_call`.

## Token Usage Calculation

AgentHub provides token usage information through the `usage_metadata` field in `UniMessage` and `UniEvent`.

`UsageMetadata` contains four fields:

- `cached_tokens` (`int | None`): Cached input tokens.
- `prompt_tokens` (`int | None`): Non-cached input tokens.
- `thoughts_tokens` (`int | None`): Chain-of-thought output tokens.
- `response_tokens` (`int | None`): Non-chain-of-thought output tokens.

Calculate total token usage as:

- `input_tokens = (cached_tokens or 0) + (prompt_tokens or 0)`
- `output_tokens = (thoughts_tokens or 0) + (response_tokens or 0)`
- `total_tokens = input_tokens + output_tokens`

## Usage Example

Tool calling example:

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

## Tracer Usage

Use Tracer to save AgentHub conversation history and inspect it in a local web UI.

Set `trace_id` in `config` to save trace files:

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

After starting the server, open `http://127.0.0.1:25750` in a browser to inspect traces.

## Playground Usage

Use Playground when a manual chat web UI is needed or when chatting with LLMs manually.

```python
from agenthub.integration.playground import start_playground_server

start_playground_server(host="127.0.0.1", port=25751)
```

After starting the server, open `http://127.0.0.1:25751` in a browser to chat.

## Notes

Agent loop rules:

- Send every tool result with the exact `tool_call_id` from its originating `tool_call`. Do not invent, normalize, or reuse IDs across unrelated tool calls.
- Set a stable `trace_id` in `config` before the first call.
- Format tool outputs as AgentHub `tool_result` items with `type`, `text`, and `tool_call_id`. Include `images` only when the target model supports image tool results.
- Continue calling `streaming_response_stateful` until `finish_reason` is `"stop"`. When `finish_reason` is `"tool_call"`, send tool results and call again.
- Use `streaming_response_stateful` to keep conversation history automatically. Use `streaming_response(messages=...)` only when managing history explicitly.
- Do not manually append streamed events to `client.get_history()`. The stateful API manages history automatically. Use `get_history`, `set_history`, and `clear_history` only to inspect, replace, or reset state.
- Preserve `thinking` and `inline_thinking` items. Do not strip `signature` fields from any content item.
