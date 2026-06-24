# APIs

`AutoLLMClient` exposes five basic APIs. Prefer the stateful stream for agent loops. See [Basic Usage](../SKILL.md#basic-usage) for a full tool-use example.

## Initialization

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

## Method signatures

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

## Notes

Keep these points in mind for agent loops:

- Send every tool result with the exact `tool_call_id` from its originating `tool_call`. Do not invent, normalize, or reuse IDs across unrelated tool calls.
- Preserve `thinking` and `inline_thinking` items. Do not strip `phase` or `signature` fields.
- For embedding models, each `UniMessage` in the `messages` array produces **one embedding vector**. Within a single message, all items in `content_items` are aggregated into a single embedding. Set `embedding_config.dimensions` in the config to control vector size.
