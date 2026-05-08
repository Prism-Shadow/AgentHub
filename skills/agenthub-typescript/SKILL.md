---
name: agenthub-typescript
description: Guidance for using the AgentHub TypeScript SDK (`@prismshadow/agenthub`). Use this skill when developing agents that need to invoke different LLM APIs, or when a unified interface for different LLM providers is required. Also use this skill if the user mentions AgentHub, requests the use of the `@prismshadow/agenthub` package, or when `@prismshadow/agenthub` is already imported in the project.
---

# AgentHub TypeScript

This skill provides guidance for using AgentHub's TypeScript SDK (`@prismshadow/agenthub`). AgentHub is a unified and precise LLM API hub for autonomous agents, providing a consistent interface across LLMs, reliable multi-step tool-call handling, and lightweight tracing for debugging and auditing executions.

## Installation

```bash
npm install @prismshadow/agenthub
```

## How to specify models

Choose a model ID from the table, set the API key and base URL using the environment variables that the routed TypeScript client reads automatically, then create the client with `new AutoLLMClient({ model: modelId })`. Alternatively, pass them explicitly without relying on environment variables: `new AutoLLMClient({ model: modelId, apiKey: yourApiKey, baseUrl: yourBaseUrl })`.

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

After declaring `const client = new AutoLLMClient({ ... })`, you can use the following `AutoLLMClient` methods:

```typescript
/**
 * Stream one response without using client history.
 *
 * Call as `for await (const event of client.streamingResponse({ messages, config }))`.
 * Provide the complete ordered conversation in `messages`; the client yields
 * universal stream events and leaves stateful history unchanged.
 *
 * Args:
 *   options.messages: Complete ordered conversation to send for this request.
 *   options.config: Request options such as tools, temperature, or trace_id.
 *
 * Yields:
 *   UniEvent: Stream event in AgentHub's universal event format.
 */
streamingResponse(options: {
  messages: UniMessage[];
  config: UniConfig;
}): AsyncGenerator<UniEvent>;
```


```typescript
/**
 * Stream one response while the client manages conversation history.
 *
 * Call as `for await (const event of client.streamingResponseStateful({ message, config }))`.
 * Pass only the next message; the client combines it with stored history and
 * records the completed turn after a successful response.
 *
 * Args:
 *   options.message: New user or assistant message for the next turn.
 *   options.config: Request options such as tools, temperature, or trace_id.
 *
 * Yields:
 *   UniEvent: Stream event in AgentHub's universal event format.
 */
streamingResponseStateful(options: {
  message: UniMessage;
  config: UniConfig;
}): AsyncGenerator<UniEvent>;
```


```typescript
/**
 * Reset the stored stateful conversation history.
 *
 * Call as `client.clearHistory()` before starting an unrelated conversation
 * with the same model client.
 *
 * Returns:
 *   void.
 */
clearHistory(): void;
```


```typescript
/**
 * Return the current stateful conversation history.
 *
 * Call as `const history = client.getHistory()` to inspect or persist messages
 * accumulated by `streamingResponseStateful`.
 *
 * Returns:
 *   UniMessage[]: Copy of the client's stored conversation list.
 */
getHistory(): UniMessage[];
```


```typescript
/**
 * Replace the stored stateful conversation history.
 *
 * Call as `client.setHistory(history)` when restoring a conversation or
 * transferring history between clients. Pass a complete ordered message list.
 *
 * Args:
 *   history: Complete ordered conversation to store on the client.
 *
 * Returns:
 *   void.
 */
setHistory(history: UniMessage[]): void;
```

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
  trace_id: "agent1/conversation_001",
};
```

All fields of `UniConfig` are optional.

Fields:

- `max_tokens` (`number`): Output-token limit. Caps generated output length when the provider supports it.
- `temperature` (`number`): Sampling temperature. Controls randomness when the provider supports it.
- `tools` (`ToolSchema[]`): List of tools available to the model. Each tool requires `name: string` and `description: string`, and may include `parameters: Record<string, any>` as a JSON Schema object.
- `thinking_summary` (`boolean`): Indicates whether the model should return its thinking process.
- `thinking_level` (`ThinkingLevel`): Reasoning effort level, one of `ThinkingLevel.NONE`, `ThinkingLevel.LOW`, `ThinkingLevel.MEDIUM`, or `ThinkingLevel.HIGH`.
- `tool_choice` (`ToolChoice`): Tool-calling configuration, one of `"auto"`, `"required"`, `"none"`, or a list of allowed tool names such as `["tool_a"]`. Only meaningful when `tools` is provided.
- `system_prompt` (`string`): System instruction text.
- `prompt_caching` (`PromptCaching`): Prompt cache mode, one of `PromptCaching.ENABLE`, `PromptCaching.DISABLE`, or `PromptCaching.ENHANCE`.
- `image_config` (`ImageConfig`): Image-generation configuration with optional `aspect_ratio: AspectRatio` and `image_size: ImageSize`. `AspectRatio` is one of `"1:1"`, `"2:3"`, `"3:2"`, `"3:4"`, `"4:3"`, `"9:16"`, `"16:9"`, or `"21:9"`; `ImageSize` is `"1K"` or `"2K"`.
- `tts_config` (`SpeakerConfig[]`): Speech-generation configuration. Each item requires `voice: string` and may include `speaker: string`; use one item for single-speaker TTS and two items with `speaker` names for multi-speaker TTS.
- `trace_id` (`string`): Stable trace identifier. Saves conversation history under this ID for the tracer.

### UniMessage

`UniMessage` is the durable conversation message shape, passed as `message` to `streamingResponseStateful`, as an element of `messages` to `streamingResponse`, and returned by `getHistory`.

Example:

```typescript
const message = {
  role: "user",
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

- `role` (`Role`): Message author, either `"user"` or `"assistant"`.
- `content_items` (`ContentItem[]`): Durable message payload stored in history and trace records.
- `usage_metadata` (`UsageMetadata | null`): Optional token usage for completed assistant messages.
- `finish_reason` (`FinishReason | null`): Stop reason for a completed assistant message, one of `"stop"`, `"length"`, `"tool_call"`, `"unknown"`, or `null`.
- `created_at` (`number`): Message creation timestamp in milliseconds since epoch.

Durable `content_items` types:

- `text`: Plain text content. Required properties: `type: "text"`, `text: string`. Optional property: `phase: string | null`.
- `image_url`: External image URL or data URI. Required properties: `type: "image_url"`, `image_url: string`.
- `inline_data`: Inline binary image or audio content. Required properties: `type: "inline_data"`, `data: Buffer`, `mime_type: string`.
- `thinking`: Text reasoning content returned by the model. Required properties: `type: "thinking"`, `thinking: string`.
- `inline_thinking`: Binary reasoning artifact, such as generated-image thinking data. Required properties: `type: "inline_thinking"`, `data: Buffer`, `mime_type: string`.
- `tool_call`: Complete tool invocation. Required properties: `type: "tool_call"`, `name: string`, `arguments: Record<string, any>`, `tool_call_id: string`.
- `tool_result`: Tool execution result to send back to the model. Required properties: `type: "tool_result"`, `text: string`, `tool_call_id: string`. Optional property: `images: string[]`.

The `text`, `inline_data`, `thinking`, `inline_thinking`, `tool_call`, and `partial_tool_call` items may carry an optional `signature` field. Do not strip or modify `signature` fields.

### UniEvent

`UniEvent` is the streamed output shape, yielded from `streamingResponse` and `streamingResponseStateful`.

Example:

```typescript
const event = {
  role: "assistant",
  event_type: "start",
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

- `role` (`Role`): Event author, either `"user"` or `"assistant"`.
- `event_type` (`EventType`): Stream lifecycle marker, including `start`, `stop`, and `unused`. Indicates where the event sits in the stream lifecycle.
- `content_items` (`PartialContentItem[]`): Stream payload. Same as `UniMessage.content_items`, plus event-only `partial_tool_call`.
- `usage_metadata` (`UsageMetadata | null`): Stream token accounting, or `null`.
- `finish_reason` (`FinishReason | null`): Stream stop reason, one of `"stop"`, `"length"`, `"tool_call"`, `"unknown"`, or `null`.
- `created_at` (`number`): Event creation timestamp in milliseconds since epoch.

Event-only `content_items` type:

- `partial_tool_call`: Streamed tool-call fragment. Required properties: `type: "partial_tool_call"`, `name: string`, `arguments: string`, `tool_call_id: string`. `arguments` carries partial JSON string content, and `tool_call_id` links the later complete `tool_call`.

## Token Usage Calculation

AgentHub provides token usage information through the `usage_metadata` field in `UniMessage` and `UniEvent`.

`UsageMetadata` contains four fields:

- `cached_tokens` (`number | null`): Cached input tokens.
- `prompt_tokens` (`number | null`): Non-cached input tokens.
- `thoughts_tokens` (`number | null`): Chain-of-thought output tokens.
- `response_tokens` (`number | null`): Non-chain-of-thought output tokens.

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

Tool-call responses must use the exact `tool_call_id` from the originating `tool_call`; do not invent, normalize, or reuse IDs across unrelated tool calls.

## Tracer Usage

Use Tracer to save AgentHub conversation history and inspect it in a local web UI.

Set `trace_id` in `config` to save trace files:

```typescript
import { AutoLLMClient } from "@prismshadow/agenthub";

const client = new AutoLLMClient({ model: "gpt-5.5" });

// Add trace_id to config
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

The default cache directory is `cache`; change it by setting the `AGENTHUB_CACHE_DIR` environment variable. With `trace_id="agent1/conversation_001"`, AgentHub creates:

- `cache/agent1/conversation_001.json`: Structured trace data with the full history and config.
- `cache/agent1/conversation_001.txt`: Human-readable conversation transcript.

Browse traces from TypeScript using the default cache directory:

```typescript
import { Tracer } from "@prismshadow/agenthub/integration/tracer";

const tracer = new Tracer();
tracer.startWebServer("127.0.0.1", 25750);
```

After starting the server, open `http://127.0.0.1:25750` in a browser to inspect traces.

## Playground Usage

Use Playground when a manual chat web UI is needed or when chatting with LLMs manually.

Use the playground for manual model checks:

```typescript
import { startPlaygroundServer } from "@prismshadow/agenthub/integration/playground";

startPlaygroundServer("127.0.0.1", 25751);
```

After starting the server, open `http://127.0.0.1:25751` in a browser to chat.

## Notes

Keep these points in mind for agent loops:

- Use `streamingResponseStateful` to keep conversation history automatically. Use `streamingResponse({ messages })` only when managing history explicitly.
- Do not manually append streamed events to `client.getHistory()`.
