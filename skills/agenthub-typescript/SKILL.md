---
name: agenthub-typescript
description: Guidance for using the AgentHub TypeScript SDK (`@prismshadow/agenthub`). Use this skill when developing agents that need to invoke different LLM APIs, or when a unified interface for different LLM providers is required. Also use this skill if the user mentions AgentHub, requests the use of the `@prismshadow/agenthub` package, or when `@prismshadow/agenthub` is already imported in the project.
---

# AgentHub TypeScript

This skill provides guidance for using AgentHub's TypeScript SDK (`@prismshadow/agenthub`).

## Installation

```bash
npm install @prismshadow/agenthub
```


## How to specify models

Choose a model ID from the table, set the API key and base URL for that vendor in environment variables, then create the client with `new AutoLLMClient({ model })`.

| Model name | Vendor | Model IDs | API Key | Base URL |
| --- | --- | --- | --- | --- |
| Gemini 3 / Gemini 3.1 | Official/Google Vertex AI | `gemini-3-flash-preview`, `gemini-3.1-flash-image-preview`, `gemini-3.1-flash-tts-preview` | `GEMINI_API_KEY` | `GEMINI_BASE_URL` |
| Claude 4.6 | Official/Amazon Bedrock/UModelVerse | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| GPT-5.4 / GPT-5.5 |  Official/UModelVerse | `gpt-5.4`, `gpt-5.5` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| GLM-5 | Official/OpenRouter/SiliconFlow | `glm-5` | `ZAI_API_KEY` | `ZAI_BASE_URL` |
| Kimi-K2.5 | Official/OpenRouter/SiliconFlow | `kimi-k2.5` | `MOONSHOT_API_KEY` | `MOONSHOT_BASE_URL` |
| Qwen3 | OpenRouter/SiliconFlow/vLLM | `Qwen3-8B` | `QWEN3_API_KEY` | `QWEN3_BASE_URL` |

## Basic Usage

Use `AutoLLMClient` and its methods for all LLM interaction and conversation tasks.

```typescript
import { AutoLLMClient } from "@prismshadow/agenthub";

process.env.OPENAI_API_KEY = "your-openai-api-key";

async function main(): Promise<void> {
  const client = new AutoLLMClient({ model: "gpt-5.5" });
  const message = { // UniMessage
    role: "user",
    content_items: [{ type: "text", text: "Say 'Hello, World!'" }],
  };
  const config = { temperature: 1.0 }; // UniConfig

  for await (const event of client.streamingResponseStateful({
    message,
    config,
  })) {
    // UniEvent
    console.log(event);
  }
}

void main();
```

## APIs

`AutoLLMClient` is the main class for interacting with the AgentHub SDK. Prefer `streamingResponseStateful` for agent loops.

`AutoLLMClient` provides the following methods:

- `(async) streamingResponse({ messages, config })`: Streams the response of LLMs in a stateless manner.
- `(async) streamingResponseStateful({ message, config })`: Streams the response of LLMs in a stateful manner.
- `clearHistory()`: Clears the history of the stateful LLM client.
- `getHistory()`: Returns the history of the stateful LLM client.
- `setHistory(history)`: Replaces the history of the stateful LLM client with a copy of the provided list.

## Data Model: UniConfig, UniMessage and UniEvent

Use AgentHub's universal data model instead of provider-native response payloads.

### UniConfig

`UniConfig` is the request configuration passed as the `config` argument to `streamingResponse` and `streamingResponseStateful`.

Example:

```typescript
const config = {
  max_tokens: 1024,
  temperature: 1.0,
  tools: [
    {
      name: "get_current_weather",
      description: "Get the current weather in a given location",
      parameters: {
        type: "object",
        properties: {
          location: {
            type: "string",
            description: "The city and state, e.g. San Francisco, CA",
          },
        },
        required: ["location"],
      },
    },
  ],
  thinking_summary: true,
  thinking_level: ThinkingLevel.HIGH,
  tool_choice: "auto",
  system_prompt: "You are a helpful assistant.",
  prompt_caching: PromptCaching.ENABLE,
  image_config: { aspect_ratio: "4:3", image_size: "1K" },
  tts_config: [{ voice: "Kore" }],
  trace_id: null,
};
```

All fields are optional.

Fields:

- `max_tokens`: Output-token limit.
- `temperature`: Sampling temperature.
- `tools`: List of tools available to the model. Each tool has required `name: string` and `description: string`, and optionally `parameters: Record<string, any>` JSON Schema.
- `thinking_summary`: Indicates whether the model should return its thinking process.
- `thinking_level`: Reasoning effort level, one of `ThinkingLevel.NONE`, `ThinkingLevel.LOW`, `ThinkingLevel.MEDIUM`, `ThinkingLevel.HIGH`.
- `tool_choice`: Tool-calling configuration, one of `"auto"`, `"required"`, `"none"`, or a list of allowed tool names (e.g., `["tool_a", "tool_b"]`). Only meaningful when `tools` is provided.
- `system_prompt`: System instruction.
- `prompt_caching`: Prompt cache mode, one of `PromptCaching.ENABLE`, `PromptCaching.DISABLE`, `PromptCaching.ENHANCE`.
- `image_config`: Image-generation configuration, with optional `aspect_ratio: AspectRatio` (one of `"1:1"`, `"2:3"`, `"3:2"`, `"3:4"`, `"4:3"`, `"9:16"`, `"16:9"`, `"21:9"`) and optional `image_size: ImageSize` (one of `"1K"`, `"2K"`).
- `tts_config`: Speech-generation configuration; each item requires `voice: string`. Include `speaker: string` for multi-speaker speech.
- `trace_id`: Trace identifier. Saves conversation history under this ID.

### UniMessage

`UniMessage` is the durable conversation message shape, passed as `message` to `streamingResponseStateful`, as an element of `messages` to `streamingResponse`, and returned by `getHistory`.

Example:

```typescript
const message = {
  role: "user" as const,
  content_items: [
    { type: "text", text: "How are you doing?" },
    { type: "image_url", image_url: "https://example.com/image.jpg" },
    { type: "inline_data", mime_type: "image/jpeg", data: Buffer.from("...") },
    { type: "thinking", thinking: "I am thinking.", signature: "0x123456" },
    { type: "inline_thinking", mime_type: "image/jpeg", data: Buffer.from("...") },
    { type: "tool_call", name: "math", arguments: { expression: "2 + 3" }, tool_call_id: "123" },
    { type: "tool_result", text: "2 + 3 = 5", images: [], tool_call_id: "123" },
  ],
};
```

Fields:

- `role`: Either `"user"` or `"assistant"`.
- `content_items`: Durable message payload stored in history and trace records. Each item is distinguished by its `type` field. Valid durable types include:
  - `text`: Plain text content with optional `phase` (message phase label).
  - `image_url`: External image referenced by URL.
  - `inline_data`: Inline binary data for images or audio. `data` stores a `Buffer`; `mime_type` describes the media type.
  - `thinking`: Text reasoning content produced by the model.
  - `inline_thinking`: Binary reasoning artifact produced during image generation. `data` stores a `Buffer`; `mime_type` describes the media type.
  - `tool_call`: A complete tool invocation with `name`, `arguments` (JSON object as `Record<string, any>`), and `tool_call_id`.
  - `tool_result`: Tool execution result with `text`, optional `images` (list of image URLs), and `tool_call_id`.

  The `text`, `inline_data`, `thinking`, `inline_thinking`, `tool_call`, and `partial_tool_call` items may carry an optional `signature` field (`string` in TypeScript, base64-encoded). Do not strip or modify `signature` fields.

- `usage_metadata`: Token usage statistics.
- `finish_reason`: Stop reason for a completed assistant message, one of `"stop"`, `"length"`, `"tool_call"`, or `"unknown"`.
- `created_at`: Timestamp in milliseconds for the message.


### UniEvent

`UniEvent` is the streamed output shape, yielded from `streamingResponse` and `streamingResponseStateful`.

Example:

```typescript
const event = {
  role: "assistant" as const,
  event_type: "delta" as const,
  content_items: [
    { type: "partial_tool_call", name: "math", arguments: "", tool_call_id: "123" },
  ],
  usage_metadata: {
    cached_tokens: null,
    prompt_tokens: 10,
    thoughts_tokens: null,
    response_tokens: 1,
  },
  finish_reason: null,
  created_at: 1694502400000,
};
```

Fields:

- `role`: Either `"user"` or `"assistant"`.
- `event_type`: Stream lifecycle marker, one of `"start"`, `"delta"`, `"stop"`, `"unused"`.
- `content_items`: Same as `UniMessage.content_items`, plus the event-only `partial_tool_call` type.
  - `partial_tool_call`: Streamed tool-call fragment. `name` selects the tool, `arguments` carries partial JSON string content, and `tool_call_id` links the later complete `tool_call`.
- `usage_metadata`: Stream token accounting.
- `finish_reason`: Stream stop reason.
- `created_at`: Event timestamp in milliseconds.

## Token Usage Calculation

AgentHub provides token usage information through the `usage_metadata` field in `UniMessage` and `UniEvent`.

`UsageMetadata` contains four fields:

- `cached_tokens`: Cached input tokens.
- `prompt_tokens`: Non-cached input tokens.
- `thoughts_tokens`: Chain-of-thought output tokens.
- `response_tokens`: Non-chain-of-thought output tokens.

Calculate total token usage as:

- `input_tokens = (cached_tokens ?? 0) + (prompt_tokens ?? 0)`
- `output_tokens = (thoughts_tokens ?? 0) + (response_tokens ?? 0)`
- `total_tokens = input_tokens + output_tokens`

## Usage Example

Tool calling example:

```typescript
import { AutoLLMClient } from "@prismshadow/agenthub";

function getWeather(location: string): string {
  return `Temperature in ${location}: 22 C`;
}

async function main(): Promise<void> {
  const weatherTool = {
    name: "get_weather",
    description: "Gets the current weather for a given location.",
    parameters: {
      type: "object" as const,
      properties: {
        location: {
          type: "string" as const,
          description: "The city name",
        },
      },
      required: ["location"],
    },
  };

  const client = new AutoLLMClient({ model: "gpt-5.5" });
  const config = { tools: [weatherTool] };

  const events = [];
  for await (const event of client.streamingResponseStateful({
    message: {
      role: "user",
      content_items: [{ type: "text", text: "What's the weather in London?" }],
    },
    config,
  })) {
    events.push(event);
  }

  let toolCall: { name: string; arguments: Record<string, any>; tool_call_id: string } | null = null;
  for (const event of events) {
    for (const item of event.content_items) {
      if (item.type === "tool_call") {
        toolCall = item;
        break;
      }
    }
    if (toolCall) break;
  }

  if (toolCall) {
    const result = getWeather(toolCall.arguments.location as string);

    for await (const event of client.streamingResponseStateful({
      message: {
        role: "user",
        content_items: [
          {
            type: "tool_result",
            text: result,
            tool_call_id: toolCall.tool_call_id,
          },
        ],
      },
      config,
    })) {
      console.log(event);
    }
  }
}

void main();
```

## Tracer Usage

Use Tracer when a web UI for inspecting agent conversation history is needed. Tracer is a local trace viewer for browsing saved AgentHub conversation histories, messages, metadata, and model outputs.

Set `trace_id` in `config` to save trace files:

```typescript
const config = { trace_id: "agent1/conversation_001" };
```

Browse traces from TypeScript:

```typescript
import { Tracer } from "@prismshadow/agenthub/integration/tracer";

const tracer = new Tracer("./cache");
const model = "gpt-5.5";
const history = client.getHistory();
const config = {};

tracer.saveHistory(model, history, "agent1/conversation_001", config);
tracer.startWebServer("127.0.0.1", 25750);
```

## Playground Usage

Use Playground when a manual chat web UI is needed or when chatting with LLMs manually. Playground is a local web UI for trying AgentHub request configuration.

Use the playground for manual model checks:

```typescript
import { startPlaygroundServer } from "@prismshadow/agenthub/integration/playground";

startPlaygroundServer("127.0.0.1", 25751);
```

After starting the server, open `http://127.0.0.1:25751` in a browser to chat.

## Notes

Agent loop rules:

- Send every tool result with the exact `tool_call_id` from its originating `tool_call`. Do not invent, normalize, or reuse IDs across unrelated tool calls.
- Set a stable `trace_id` in `config` before the first call.
- Format tool outputs as AgentHub `tool_result` items with `type`, `text`, and `tool_call_id`. Include `images` only when the target model supports image tool results.
- Continue calling `streamingResponseStateful` until `finish_reason` is `"stop"`. When `finish_reason` is `"tool_call"`, send tool results and call again.
- Use `streamingResponseStateful` to keep conversation history automatically. Use `streamingResponse({ messages })` only when managing history explicitly.
- Do not manually append streamed events to `client.getHistory()`. The stateful API manages history automatically. Use `getHistory`, `setHistory`, and `clearHistory` only to inspect, replace, or reset state.
- Preserve `thinking` and `inline_thinking` items. Do not strip `signature` fields from any content item.
- Calculate token totals with `(value ?? 0)`. Token fields in `usage_metadata` can be `null`.
