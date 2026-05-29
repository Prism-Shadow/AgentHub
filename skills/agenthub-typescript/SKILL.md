---
name: agenthub-typescript
description: Guidance for using the AgentHub TypeScript SDK (`@prismshadow/agenthub`). Use when developing agents that call different LLM APIs, need a unified interface for LLM providers, mention AgentHub, request `@prismshadow/agenthub`, or already import it.
---

# AgentHub TypeScript

AgentHub is a unified SDK for calling LLMs across providers with shared data models, tool calling, tracing, and playground support.

## Installation

```bash
npm install @prismshadow/agenthub
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
| GPT 5.4 | Official / ModelVerse | `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| GPT 5.5 | Official / ModelVerse | `gpt-5.5` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
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

## Data Models

AgentHub uses `UniConfig`, `UniMessage`, and `UniEvent` to represent request options, conversation history, and streamed outputs across providers.

### UniConfig

`UniConfig` is the request config for `streamingResponse` and `streamingResponseStateful`. All fields are optional.

```typescript
const config = {
  max_tokens: 1024,
  temperature: 1.0,
  tools: [{
    name: "get_weather",
    description: "Get weather.",
    parameters: {
      type: "object",
      properties: { location: { type: "string", description: "City name." } },
      required: ["location"],
    },
  }],
  tool_choice: "auto",
  thinking_summary: true,
  thinking_level: ThinkingLevel.HIGH,
  system_prompt: "You are helpful.",
  prompt_caching: PromptCaching.ENABLE,
  image_config: { aspect_ratio: "4:3", image_size: "1K" },
  tts_config: [{ voice: "Kore" }],
  embedding_config: { dimensions: 768 },
  trace_id: "agent1/conversation_001",
};
```

Fields:

- `max_tokens` (`number`): Output token limit.
- `temperature` (`number`): Sampling temperature; support varies by model.
- `tools` (`ToolSchema[]`): Tools with `name`, `description`, and optional JSON Schema `parameters`.
- `thinking_summary` (`boolean`): Request a thinking summary when supported.
- `thinking_level` (`ThinkingLevel`): `NONE`, `LOW`, `MEDIUM`, `HIGH`, or `XHIGH`.
- `tool_choice` (`ToolChoice`): `auto`, `required`, `none`, or a list of tool names; support varies by model.
- `system_prompt` (`string`): System instruction text.
- `prompt_caching` (`PromptCaching`): `ENABLE`, `DISABLE`, or `ENHANCE`.
- `image_config` (`ImageConfig`): `aspect_ratio` (`1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `9:16`, `16:9`, `21:9`) and `image_size` (`1K`, `2K`).
- `tts_config` (`SpeakerConfig[]`): Voice config; each item has `voice` and optional `speaker`.
- `embedding_config` (`EmbeddingConfig`): Embedding config, currently `dimensions`.
- `trace_id` (`string`): Stable ID for tracer output.

### UniMessage

`UniMessage` is the durable message shape used in history.

```typescript
const message = {
  role: "user",
  content_items: [
    { type: "text", text: "Hello", phase: null, signature: "sig" },
    { type: "image_url", image_url: "https://example.com/image.jpg" },
    { type: "inline_data", data: Buffer.from("..."), mime_type: "image/png", signature: "sig" },
    { type: "thinking", thinking: "Reasoning", signature: "sig" },
    { type: "inline_thinking", data: Buffer.from("..."), mime_type: "image/png", signature: "sig" },
    { type: "tool_call", name: "get_weather", arguments: { location: "Paris" }, tool_call_id: "call_1", signature: "sig" },
    { type: "tool_result", text: "22 C", tool_call_id: "call_1" },
    { type: "embedding", embedding: [0.1, 0.2] },
  ],
};
```

Fields:

- `role` (`Role`): `user` or `assistant`.
- `content_items` (`ContentItem[]`): Message payload.
- `usage_metadata` (`UsageMetadata | null`): Optional token counts on completed assistant messages.
- `finish_reason` (`FinishReason | null`): `stop`, `length`, `tool_call`, `unknown`, or `null`.
- `created_at` (`number`): Unix milliseconds.

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

```typescript
const event = {
  role: "assistant",
  event_type: "delta",
  content_items: [
    { type: "partial_tool_call", name: "get_weather", arguments: "{\"location\":\"Par", tool_call_id: "call_1" },
  ],
  usage_metadata: { cached_tokens: 0, prompt_tokens: 10, thoughts_tokens: null, response_tokens: 1 },
  finish_reason: null,
  created_at: 1694502400000,
};
```

Fields:

- `role` (`Role`): `user` or `assistant`.
- `event_type` (`EventType`): `start`, `delta`, `stop`, or `unused`.
- `content_items` (`PartialContentItem[]`): Stream payload; includes `ContentItem` plus `partial_tool_call`.
- `usage_metadata` (`UsageMetadata | null`): Token counts: `cached_tokens`, `prompt_tokens`, `thoughts_tokens`, `response_tokens`.
  Token math: `input = cached_tokens + prompt_tokens`; `output = thoughts_tokens + response_tokens`; treat `null` as `0`.
- `finish_reason` (`FinishReason | null`): `stop`, `length`, `tool_call`, `unknown`, or `null`.
- `created_at` (`number`): Unix milliseconds.

Event-only content item:

- `partial_tool_call`: Streaming tool-call fragment with `name`, partial JSON `arguments`, and `tool_call_id`.

## APIs

`AutoLLMClient` exposes five basic APIs. Prefer the stateful stream for agent loops.

Initialize `AutoLLMClient` in one of three common ways:

```typescript
// Initialize with model name
const clientByModel = new AutoLLMClient({ model: "gpt-5.5" });

// Optionally specify API key (if not using environment variables)
const clientWithEndpoint = new AutoLLMClient({
  model: "gpt-5.5",
  apiKey: "your-openai-api-key",
  baseUrl: "https://api.openai.com/v1",
});

// Use OpenAI Chat Completions-compatible routing explicitly
const clientWithType = new AutoLLMClient({
  model: "custom-model",
  clientType: "openai",
});
```

```typescript
/** Stream one stateless response from a full message list. */
streamingResponse(options: { messages: UniMessage[]; config: UniConfig }): AsyncGenerator<UniEvent>;

/** Stream one stateful response and update client history. */
streamingResponseStateful(options: { message: UniMessage; config: UniConfig }): AsyncGenerator<UniEvent>;

/** Return a copy of stateful history. */
getHistory(): UniMessage[];

/** Replace stateful history with a copy. */
setHistory(history: UniMessage[]): void;

/** Clear stateful history. */
clearHistory(): void;
```

## Basic Usage

This example asks GPT to call a weather tool, runs the tool, then sends the result back.

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


## Tracer

Tracer saves trace files and serves a local UI for inspecting conversations.

Set `trace_id` to save trace files:

```typescript
import { AutoLLMClient } from "@prismshadow/agenthub";

const client = new AutoLLMClient({ model: "gpt-5.5" });

const config = { trace_id: "agent1/conversation_001" };

for await (const event of client.streamingResponseStateful({
  message: {
    role: "user",
    content_items: [{ type: "text", text: "Hello" }],
  },
  config,
})) {
  console.log(event);
}
```

Default cache dir: `cache`, or `AGENTHUB_CACHE_DIR`. For `trace_id="agent1/conversation_001"`, AgentHub writes:

- `cache/agent1/conversation_001.json`: Structured trace data with the full history and config.
- `cache/agent1/conversation_001.txt`: Human-readable conversation transcript.

Browse traces:

```typescript
import { Tracer } from "@prismshadow/agenthub/integration/tracer";

const tracer = new Tracer();
tracer.startWebServer("127.0.0.1", 25750);
```

Open Tracer at `http://127.0.0.1:25750`.

## Playground

Playground starts a local chat UI for manual model checks.

Start Playground for manual chat:

```typescript
import { startPlaygroundServer } from "@prismshadow/agenthub/integration/playground";

startPlaygroundServer("127.0.0.1", 25751);
```

Open Playground at `http://127.0.0.1:25751`.

## Notes

Keep these points in mind for agent loops:

- Send every tool result with the exact `tool_call_id` from its originating `tool_call`. Do not invent, normalize, or reuse IDs across unrelated tool calls.
- Preserve `thinking` and `inline_thinking` items. Do not strip `phase` or `signature` fields.
