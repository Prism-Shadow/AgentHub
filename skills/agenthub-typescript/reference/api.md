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

## Module-level helpers

```typescript
/**
 * List supported models as (model, base_url, client) triples covering official
 * endpoints plus OpenRouter and SiliconFlow; the triple maps onto the
 * AutoLLMClient constructor (model, baseUrl, clientType).
 */
function listSupportedModels(): SupportedModel[];
```

## Errors

All AgentHub errors subclass `AgentHubError`. Unsupported `UniConfig` values (e.g.
`temperature` or `tool_choice` on models that reject them) throw
`UnsupportedParameterError`, which carries `client` and `parameter` fields. Thinking
levels never throw: every client maps each `ThinkingLevel` to the closest supported level.
