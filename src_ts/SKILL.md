---
name: agenthub-typescript
description: Use AgentHub's TypeScript SDK for AutoLLMClient agents, streaming, tools, tracing, multimodal Gemini, and provider routing.
---

# AgentHub TypeScript SDK Usage

Use this skill when writing TypeScript or Node.js agents, CLIs, chat apps, or tests that call the `@prismshadow/agenthub` package.

## Installation

```bash
npm install @prismshadow/agenthub
```

## Usage

### Basic Client Usage

```typescript
import { AutoLLMClient } from "@prismshadow/agenthub";

process.env.OPENAI_API_KEY = "your-openai-api-key";

async function main() {
  const client = new AutoLLMClient({ model: "gpt-5.5" });

  for await (const event of client.streamingResponseStateful({
    message: {
      role: "user",
      content_items: [{ type: "text", text: "Hello!" }],
    },
    config: { temperature: 1.0 },
  })) {
    console.log(event);
  }
}

main().catch(console.error);
```

### Initialization And Routing

```typescript
const client = new AutoLLMClient({ model: "gpt-5.5" });
const clientWithOverrides = new AutoLLMClient({
  model: "gpt-5.5",
  apiKey: "your-openai-api-key",
  baseUrl: "https://example.com/v1",
  clientType: "gpt-5.5",
});
```

The client selects the provider from `clientType`, then `CLIENT_TYPE`, then `model.toLowerCase()`.

Supported routing substrings:

- `gemini-3-` or `gemini-3.1-`
- `claude` plus `4-6`
- `gpt-5.4` or `gpt-5.5`
- `glm-5`
- `kimi-k2.5`
- `qwen3`

Use `clientType` for aliases, OpenRouter slugs, SiliconFlow slugs, local vLLM names, or compatible gateways.

Environment variables:

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

For GLM-5, use `ZAI_API_KEY` / `ZAI_BASE_URL`; avoid older GLM variable names.

### Core Methods

Stateful method that maintains conversation history:

```typescript
for await (const event of client.streamingResponseStateful({ message, config })) {
  console.log(event);
}
```

Stateless method that requires the full message history:

```typescript
for await (const event of client.streamingResponse({ messages, config })) {
  console.log(event);
}
```

### History Management

```typescript
// Get current history
const history = client.getHistory();

// Clear all history
client.clearHistory();

// Replace history with a saved copy
client.setHistory(history);
```

### Message Format

Always use `content_items`; do not pass provider-native message formats.

```typescript
const message = {
  role: "user",
  content_items: [
    { type: "text", text: "Hello" },
    { type: "image_url", image_url: "https://example.com/image.png" },
    { type: "inline_data", mime_type: "image/png", data: Buffer.from("...") },
    { type: "thinking", thinking: "...", signature: "optional" },
    { type: "inline_thinking", mime_type: "image/png", data: Buffer.from("..."), signature: "optional" },
    { type: "tool_call", name: "math", arguments: { expression: "2 + 3" }, tool_call_id: "call_123" },
    { type: "tool_result", text: "2 + 3 = 5", images: [], tool_call_id: "call_123" },
  ],
};
```

Streaming output:

```typescript
{
  role: "assistant",
  event_type: "delta",
  content_items: [{ type: "text", text: "Hello" }],
  usage_metadata: null,
  finish_reason: null,
  created_at: 1694502400000,
}
```

`usage_metadata` contains `cached_tokens`, `prompt_tokens`, `thoughts_tokens`, and `response_tokens`.

Keep signed `thinking` and `inline_thinking` items in history when models emit them. TypeScript signatures are base64 strings.

### Configuration Options

```typescript
import { PromptCaching, ThinkingLevel } from "@prismshadow/agenthub";

const config = {
  max_tokens: 500,
  temperature: 1.0,
  tools: [toolDefinition],
  thinking_summary: true,
  thinking_level: ThinkingLevel.HIGH,
  tool_choice: "auto", // "auto", "required", "none", or ["tool_name"]
  system_prompt: "You are a helpful assistant",
  prompt_caching: PromptCaching.ENABLE,
  image_config: { aspect_ratio: "4:3", image_size: "1K" },
  tts_config: [{ voice: "Kore" }],
  trace_id: "agent1/conversation_001",
};
```

### Tool Calling

When using tools, handle `tool_call_id` correctly and use `arguments`, not `argument`:

```typescript
const weatherTool = {
  name: "get_weather",
  description: "Gets the current weather for a given location.",
  parameters: {
    type: "object",
    properties: {
      location: { type: "string", description: "The city name" },
    },
    required: ["location"],
  },
};

const config = { tools: [weatherTool], tool_choice: "auto" };
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

const toolCall = events
  .flatMap((event) => event.content_items)
  .find((item) => item.type === "tool_call");

if (toolCall?.type === "tool_call") {
  const result = getWeather(toolCall.arguments as { location: string });

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
```

Ignore `partial_tool_call` for execution until AgentHub materializes a complete `tool_call`.

Tool response shape:

```typescript
{
  role: "user",
  content_items: [
    {
      type: "tool_result",
      text: "London is 22 C today.",
      tool_call_id: "call_abc123",
    },
  ],
}
```

### Multimodal Usage

Image input:

```typescript
const message = {
  role: "user",
  content_items: [
    { type: "text", text: "Describe this image." },
    { type: "image_url", image_url: "https://example.com/image.png" },
  ],
};
```

Gemini image generation requires an image-capable Gemini model, such as `gemini-3.1-flash-image-preview`; `image_config` configures image output for models with that capability. Generated images return image `inline_data`:

```typescript
const config = { image_config: { aspect_ratio: "4:3", image_size: "1K" } };
```

Gemini TTS requires a TTS Gemini model, such as `gemini-3.1-flash-tts-preview`; `tts_config` configures audio output for TTS models. TTS returns audio `inline_data` and should use text input only:

```typescript
const config = { tts_config: [{ voice: "Kore" }] };
```

### Tracer Usage

Save and browse conversation history with a web interface:

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
```

For automatic trace saving, set `trace_id` in the request config. The default cache directory is `cache`; change it with `AGENTHUB_CACHE_DIR`.

### Playground Usage

Interactive web interface for chatting with LLMs:

```typescript
import { startPlaygroundServer } from "@prismshadow/agenthub/integration/playground";

startPlaygroundServer("127.0.0.1", 25751);
```

Migration checklist:

1. Replace provider-native clients with `AutoLLMClient`.
2. Convert messages to `content_items`.
3. Move provider-specific options into `config`.
4. Preserve `tool_call_id`, signed thinking, usage metadata, and timestamps in durable history.
5. Use `clientType` plus `baseUrl` for aliases and compatible gateways.
6. Add `trace_id` around important agent runs.
