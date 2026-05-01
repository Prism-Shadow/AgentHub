---
name: agenthub-typescript
description: Use AgentHub's TypeScript SDK (`@prismshadow/agenthub`) to build or modify Node.js agents, chat runtimes, tests, provider routing, tool-calling loops, tracing, or playground workflows with AutoLLMClient, UniConfig, UniMessage, UniEvent, and AgentHub's universal content model.
---

# AgentHub TypeScript

## Overview

AgentHub provides a universal SDK for calling supported LLM providers through one client, one message shape, and one streaming event shape. Prefer `AutoLLMClient` for application code unless a task explicitly needs a provider-specific client.

## Installation

```bash
npm install @prismshadow/agenthub
```

Import the client and shared enums from the package:

```typescript
import { AutoLLMClient, PromptCaching, ThinkingLevel } from "@prismshadow/agenthub";
```

## Supported Models and Environment Variables

`AutoLLMClient` routes by `clientType`, then `CLIENT_TYPE`, then `model.toLowerCase()`. Use `clientType` when the model name is an alias, an OpenRouter or SiliconFlow slug, a local vLLM name, or another compatible gateway name.

Supported routing substrings: `gpt-5.4`, `gpt-5.5`, `claude` with `4-6`, `gemini-3-`, `gemini-3.1-`, `glm-5`, `kimi-k2.5`, and `qwen3`.

Set the provider API key before creating the client: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`, or `QWEN3_API_KEY`. Use `ZAI_API_KEY` and `ZAI_BASE_URL` for GLM-5. Optional base URL variables otherwise use the same provider prefix, such as `OPENAI_BASE_URL`. Use `CLIENT_TYPE` as a global routing override and `AGENTHUB_CACHE_DIR` for trace storage.

## Basic Usage

Follow this flow when an agent calls AgentHub:

1. Create an `AutoLLMClient` for the target model after setting the provider API key.
2. Represent the next user turn as a `UniMessage` with `role` and `content_items`; keep provider-native message payloads out of AgentHub.
3. Put request options such as `temperature`, tools, system prompts, and tracing in a `UniConfig` object.
4. Prefer `(async) streamingResponseStateful({ message, config })` for chat and agent loops because it stores conversation history inside the client. Use stateless `streamingResponse({ messages, config })` only when the caller owns the full history.
5. Consume the async stream as `UniEvent` objects. Read each event's `content_items`, `usage_metadata`, and `finish_reason` instead of assuming provider-specific response shapes.

```typescript
import { AutoLLMClient } from "@prismshadow/agenthub";

process.env.OPENAI_API_KEY = "your-openai-api-key";

async function main(): Promise<void> {
  const client = new AutoLLMClient({ model: "gpt-5.5" });
  const message = { // UniMessage
    role: "user",
    content_items: [{ type: "text", text: "Say 'Hello, World!'" }],
  } as const;
  const config = { temperature: 1.0 }; // UniConfig

  for await (const event of client.streamingResponseStateful({
    message,
    config,
  })) {
    // event is a UniEvent.
    console.log(event);
  }
}

void main();
```

## Data Model

Always send and store AgentHub's universal data model. Do not pass provider-native OpenAI, Anthropic, Gemini, GLM, Kimi, or Qwen message payloads into `AutoLLMClient`.

### UniConfig

`UniConfig` is the provider-independent request configuration passed as the `config` argument to `streamingResponse` and `streamingResponseStateful`. Use it for generation options, tools, system prompts, tracing, caching, image output, and TTS settings.

Fields:

```typescript
const config = {
  max_tokens: 500,
  temperature: 1.0,
  tools: [toolDefinition],
  thinking_summary: true,
  thinking_level: ThinkingLevel.HIGH,
  tool_choice: "auto", // "auto", "required", "none", or ["tool_name"]
  system_prompt: "You are a helpful assistant.",
  prompt_caching: PromptCaching.ENABLE,
  image_config: { aspect_ratio: "4:3", image_size: "1K" },
  tts_config: [{ voice: "Kore" }],
  trace_id: "agent1/conversation_001",
};
```

Use snake_case field names in AgentHub configs and content items. Use camelCase only for TypeScript method names and constructor options such as `clientType`, `apiKey`, and `baseUrl`.

### UniMessage

`UniMessage` is the durable conversation record used for API input and stateful history. Pass the next user turn to `streamingResponseStateful`, pass a full list of messages to `streamingResponse`, and store assistant responses from `getHistory()` in this shape.

`content_items` are typed message parts inside `UniMessage`:

- `text`: natural-language text. Assistant text may include `phase`; signed text may include `signature`.
- `image_url`: image input by URL.
- `inline_data`: binary input or output with `data` as a `Buffer` and `mime_type`, mainly for image and audio data.
- `thinking`: model reasoning text, optionally signed.
- `inline_thinking`: binary thinking data with `data`, `mime_type`, and optional `signature`; use for image or audio data in the thinking process.
- `tool_call`: complete tool request with `name`, `arguments`, and `tool_call_id`.
- `tool_result`: tool response with `text`, optional `images`, and `tool_call_id`.

Durable history record:

```typescript
const message = {
  role: "user",
  content_items: [
    { type: "text", text: "Weather in London?" },
    { type: "tool_result", text: "15 C", tool_call_id: "call_123" },
  ],
  usage_metadata: null,
  finish_reason: null,
  created_at: 1694502400000,
};
```

`role` is `user` or `assistant`. User messages normally contain user text or tool results. Assistant messages contain generated text, thinking, tool calls, media output, usage metadata, finish reason, and timestamp.

### UniEvent

`UniEvent` is the streamed response event returned by `streamingResponse` and `streamingResponseStateful`. Consume it in the async iterator while the response is being generated; the stateful client folds completed events into assistant `UniMessage` history.

`content_items` in `UniEvent` use the same item types as `UniMessage`, plus `partial_tool_call` for streamed tool-call fragments.

Streamed return shape:

```typescript
const event = {
  role: "assistant",
  event_type: "delta",
  content_items: [{ type: "text", text: "Hello" }],
  usage_metadata: null,
  finish_reason: null,
  created_at: 1694502400000,
};
```

`event_type` can be `start`, `delta`, `stop`, or `unused`. Intermediate events often have `usage_metadata: null` and `finish_reason: null`. The final event must carry `usage_metadata` and `finish_reason`; AgentHub raises an error if a stream ends without them.

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

- `(async) streamingResponse({ messages, config })`: Streams the response of LLMs in a stateless manner.
- `(async) streamingResponseStateful({ message, config })`: Streams the response of LLMs in a stateful manner.
- `clearHistory()`: Clears the history of the stateful LLM client.
- `getHistory()`: Returns the history of the stateful LLM client.
- `setHistory(history)`: Replaces the history of the stateful LLM client with a copy of the provided list.

## Tracer Usage

Explanation: Use `Tracer` when a TypeScript agent needs to save a conversation and inspect it in the local web UI. The minimal flow is to create a tracer, save conversation history with `saveHistory`, then start the web server.

```typescript
import { Tracer } from "@prismshadow/agenthub/integration/tracer";

const tracer = new Tracer("./cache");

const model = "gpt-5.5";
const history = [
  { role: "user", content_items: [{ type: "text", text: "Hello!" }] },
  { role: "assistant", content_items: [{ type: "text", text: "Hi there!" }] },
];
const config = {};

tracer.saveHistory(model, history, "session/conv_001", config);
tracer.startWebServer("127.0.0.1", 25750);
console.log("Open http://127.0.0.1:25750 to inspect the trace.");
```

## Note

- Use `saveHistory(model, history, traceId, config)` for explicit snapshots, as shown above. Set `trace_id` in `config` when the client should save each streamed turn automatically under `AGENTHUB_CACHE_DIR` or `cache`.
- Preserve signed `thinking`, `inline_thinking`, and signed text items in history. TypeScript signatures are base64 strings.

## Playground Usage

Use the playground for manual model checks:

```typescript
import { startPlaygroundServer } from "@prismshadow/agenthub/integration/playground";

startPlaygroundServer("127.0.0.1", 25751);
```
