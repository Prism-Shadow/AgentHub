# Adaptive thinking

Let Claude dynamically determine when and how much to use extended thinking with adaptive thinking mode.

---

Adaptive thinking is the recommended way to use [extended thinking](/docs/en/build-with-claude/extended-thinking) with Claude Opus 4.6 and Sonnet 4.6. Instead of manually setting a thinking token budget, adaptive thinking lets Claude dynamically determine when and how much to use extended thinking based on the complexity of each request.

<Tip>
Adaptive thinking can drive better performance than extended thinking with a fixed `budget_tokens` for many workloads, especially bimodal tasks and long-horizon agentic workflows. No beta header is required.

For workloads where predictable latency and token usage matter, or where you need precise control over thinking costs, extended thinking with `budget_tokens` continues to be fully supported. Consider testing both modes on your specific workloads to determine which works best.
</Tip>

## Supported models

Adaptive thinking is supported on the following models:

- Claude Opus 4.6 (`claude-opus-4-6`)
- Claude Sonnet 4.6 (`claude-sonnet-4-6`)

<Warning>
`thinking.type: "enabled"` and `budget_tokens` are **deprecated** on Opus 4.6 and Sonnet 4.6 and will be removed in a future model release. Use `thinking.type: "adaptive"` with the `effort` parameter instead. If you are already using extended thinking with `budget_tokens`, it continues to work and no immediate changes are required.

Older models (Sonnet 4.5, Opus 4.5, etc.) do not support adaptive thinking and require `thinking.type: "enabled"` with `budget_tokens`.
</Warning>

## How adaptive thinking works

In adaptive mode, thinking is optional for the model. Claude evaluates the complexity of each request and determines whether and how much to use extended thinking. At the default effort level (`high`), Claude almost always thinks. At lower effort levels, Claude may skip thinking for simpler problems.

Adaptive thinking also automatically enables [interleaved thinking](/docs/en/build-with-claude/extended-thinking#interleaved-thinking). This means Claude can think between tool calls, making it especially effective for agentic workflows.

## How to use adaptive thinking

Set `thinking.type` to `"adaptive"` in your API request:

<CodeGroup>
```bash Shell
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-opus-4-6",
    "max_tokens": 16000,
    "thinking": {
        "type": "adaptive"
    },
    "messages": [
        {
            "role": "user",
            "content": "Explain why the sum of two even numbers is always even."
        }
    ]
}'
```

```python Python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=16000,
    thinking={"type": "adaptive"},
    messages=[
        {
            "role": "user",
            "content": "Explain why the sum of two even numbers is always even.",
        }
    ],
)

for block in response.content:
    if block.type == "thinking":
        print(f"\nThinking: {block.thinking}")
    elif block.type == "text":
        print(f"\nResponse: {block.text}")
```

```typescript TypeScript hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const response = await client.messages.create({
  model: "claude-opus-4-6",
  max_tokens: 16000,
  thinking: {
    type: "adaptive"
  },
  messages: [
    {
      role: "user",
      content: "Explain why the sum of two even numbers is always even."
    }
  ]
} as unknown as Anthropic.MessageCreateParamsNonStreaming);

for (const block of response.content) {
  if (block.type === "thinking") {
    console.log(`\nThinking: ${block.thinking}`);
  } else if (block.type === "text") {
    console.log(`\nResponse: ${block.text}`);
  }
}
```

```csharp C#
using System;
using System.Threading.Tasks;
using Anthropic;
using Anthropic.Models.Messages;

class Program
{
    static async Task Main(string[] args)
    {
        AnthropicClient client = new();

        var parameters = new MessageCreateParams
        {
            Model = Model.ClaudeOpus4_6,
            MaxTokens = 16000,
            Thinking = new ThinkingConfigAdaptive(),
            Messages = [
                new() {
                    Role = Role.User,
                    Content = "Explain why the sum of two even numbers is always even."
                }
            ]
        };

        var message = await client.Messages.Create(parameters);

        foreach (var block in message.Content)
        {
            if (block.TryPickThinking(out ThinkingBlock? thinking))
            {
                Console.WriteLine($"\nThinking: {thinking.Thinking}");
            }
            else if (block.TryPickText(out TextBlock? text))
            {
                Console.WriteLine($"\nResponse: {text.Text}");
            }
        }
    }
}
```
</CodeGroup>

## Adaptive thinking with the effort parameter

You can combine adaptive thinking with the [effort parameter](/docs/en/build-with-claude/effort) to guide how much thinking Claude does. The effort level acts as soft guidance for Claude's thinking allocation:

| Effort level | Thinking behavior |
|:-------------|:------------------|
| `max` | Claude always thinks with no constraints on thinking depth. Opus 4.6 only. Requests using `max` on other models return an error. |
| `high` (default) | Claude always thinks. Provides deep reasoning on complex tasks. |
| `medium` | Claude uses moderate thinking. May skip thinking for very simple queries. |
| `low` | Claude minimizes thinking. Skips thinking for simple tasks where speed matters most. |

<CodeGroup>
```bash Shell
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-opus-4-6",
    "max_tokens": 16000,
    "thinking": {
        "type": "adaptive"
    },
    "output_config": {
        "effort": "medium"
    },
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
}'
```

```python Python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=16000,
    thinking={"type": "adaptive"},
    output_config={"effort": "medium"},
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(response.content[0].text)
```

```typescript TypeScript hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const response = await client.messages.create({
  model: "claude-opus-4-6",
  max_tokens: 16000,
  thinking: {
    type: "adaptive"
  },
  output_config: {
    effort: "medium"
  },
  messages: [
    {
      role: "user",
      content: "What is the capital of France?"
    }
  ]
} as unknown as Anthropic.MessageCreateParamsNonStreaming);

for (const block of response.content) {
  if (block.type === "text") {
    console.log(block.text);
  }
}
```

```csharp C#
using System;
using System.Threading.Tasks;
using Anthropic;
using Anthropic.Models.Messages;

class Program
{
    static async Task Main(string[] args)
    {
        AnthropicClient client = new();

        var parameters = new MessageCreateParams
        {
            Model = Model.ClaudeOpus4_6,
            MaxTokens = 16000,
            Thinking = new ThinkingConfigAdaptive(),
            OutputConfig = new OutputConfig { Effort = Effort.Medium },
            Messages = [new() { Role = Role.User, Content = "What is the capital of France?" }]
        };

        var message = await client.Messages.Create(parameters);
        Console.WriteLine(message);
    }
}
```
</CodeGroup>

## Streaming with adaptive thinking

Adaptive thinking works seamlessly with [streaming](/docs/en/build-with-claude/streaming). Thinking blocks are streamed via `thinking_delta` events just like manual thinking mode:

<CodeGroup>
```python Python
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-opus-4-6",
    max_tokens=16000,
    thinking={"type": "adaptive"},
    messages=[
        {
            "role": "user",
            "content": "What is the greatest common divisor of 1071 and 462?",
        }
    ],
) as stream:
    for event in stream:
        if event.type == "content_block_start":
            print(f"\nStarting {event.content_block.type} block...")
        elif event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                print(event.delta.thinking, end="", flush=True)
            elif event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
```

```typescript TypeScript hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const stream = await client.messages.stream({
  model: "claude-opus-4-6",
  max_tokens: 16000,
  thinking: { type: "adaptive" },
  messages: [{ role: "user", content: "What is the greatest common divisor of 1071 and 462?" }]
} as unknown as Anthropic.MessageStreamParams);

for await (const event of stream) {
  if (event.type === "content_block_start") {
    console.log(`\nStarting ${event.content_block.type} block...`);
  } else if (event.type === "content_block_delta") {
    if (event.delta.type === "thinking_delta") {
      process.stdout.write(event.delta.thinking);
    } else if (event.delta.type === "text_delta") {
      process.stdout.write(event.delta.text);
    }
  }
}
```

```csharp C#
using System;
using System.Threading.Tasks;
using Anthropic;
using Anthropic.Models.Messages;

class Program
{
    static async Task Main(string[] args)
    {
        AnthropicClient client = new();

        var parameters = new MessageCreateParams
        {
            Model = Model.ClaudeOpus4_6,
            MaxTokens = 16000,
            Thinking = new ThinkingConfigAdaptive(),
            Messages = [new() { Role = Role.User, Content = "What is the greatest common divisor of 1071 and 462?" }]
        };

        await foreach (var msg in client.Messages.CreateStreaming(parameters))
        {
            Console.Write(msg);
        }
    }
}
```
</CodeGroup>

## Adaptive vs manual vs disabled thinking

| Mode | Config | Availability | When to use |
|:-----|:-------|:-------------|:------------|
| **Adaptive** | `thinking: {type: "adaptive"}` | Opus 4.6, Sonnet 4.6 | Claude determines when and how much to use extended thinking. Use `effort` to guide. |
| **Manual** | `thinking: {type: "enabled", budget_tokens: N}` | All models. Deprecated on Opus 4.6 and Sonnet 4.6 (consider adaptive mode instead). | When you need precise control over thinking token spend. |
| **Disabled** | Omit `thinking` parameter | All models | When you don't need extended thinking and want the lowest latency. |

<Note>
Adaptive thinking is available on Opus 4.6 and Sonnet 4.6. Older models only support `type: "enabled"` with `budget_tokens`. On both Opus 4.6 and Sonnet 4.6, `type: "enabled"` with `budget_tokens` is still accepted but deprecated.

**Interleaved thinking availability by mode:**
- **Adaptive mode:** Interleaved thinking is automatically enabled on both Opus 4.6 and Sonnet 4.6.
- **Manual mode on Sonnet 4.6:** Interleaved thinking is supported via the `interleaved-thinking-2025-05-14` beta header.
- **Manual mode on Opus 4.6:** Interleaved thinking is not available. If your agentic workflow requires thinking between tool calls on Opus 4.6, use adaptive mode.
</Note>

## Important considerations

### Validation changes

When using adaptive thinking, previous assistant turns don't need to start with thinking blocks. This is more flexible than manual mode, where the API enforces that thinking-enabled turns begin with a thinking block.

### Prompt caching

Consecutive requests using `adaptive` thinking preserve [prompt cache](/docs/en/build-with-claude/prompt-caching) breakpoints. However, switching between `adaptive` and `enabled`/`disabled` thinking modes breaks cache breakpoints for messages. System prompts and tool definitions remain cached regardless of mode changes.

### Tuning thinking behavior

Adaptive thinking's triggering behavior is promptable. If Claude is thinking more or less often than you'd like, you can add guidance to your system prompt:

```text
Extended thinking adds latency and should only be used when it
will meaningfully improve answer quality — typically for problems
that require multi-step reasoning. When in doubt, respond directly.
```

<Warning>
Steering Claude to think less often may reduce quality on tasks that benefit from reasoning. Measure the impact on your specific workloads before deploying prompt-based tuning to production. Consider testing with lower [effort levels](/docs/en/build-with-claude/effort) first.
</Warning>

### Cost control

Use `max_tokens` as a hard limit on total output (thinking + response text). The `effort` parameter provides additional soft guidance on how much thinking Claude allocates. Together, these give you effective control over cost.

At `high` and `max` effort levels, Claude may think more extensively and can be more likely to exhaust the `max_tokens` budget. If you observe `stop_reason: "max_tokens"` in responses, consider increasing `max_tokens` to give the model more room, or lowering the effort level.

## Working with thinking blocks

The following concepts apply to all models that support extended thinking, regardless of whether you use adaptive or manual mode.

### Summarized thinking

With extended thinking enabled, the Messages API for Claude 4 models returns a summary of Claude's full thinking process. Summarized thinking provides the full intelligence benefits of extended thinking, while preventing misuse.

Here are some important considerations for summarized thinking:

- You're charged for the full thinking tokens generated by the original request, not the summary tokens.
- The billed output token count will **not match** the count of tokens you see in the response.
- The first few lines of thinking output are more verbose, providing detailed reasoning that's particularly helpful for prompt engineering purposes.
- As Anthropic seeks to improve the extended thinking feature, summarization behavior is subject to change.
- Summarization preserves the key ideas of Claude's thinking process with minimal added latency, enabling a streamable user experience and easy migration from Claude Sonnet 3.7 to Claude 4 and later models.
- Summarization is processed by a different model than the one you target in your requests. The thinking model does not see the summarized output.

<Note>
Claude Sonnet 3.7 continues to return full thinking output.

In rare cases where you need access to full thinking output for Claude 4 models, [contact our sales team](mailto:sales@anthropic.com).
</Note>

### Thinking encryption

Full thinking content is encrypted and returned in the `signature` field. This field is used to verify that thinking blocks were generated by Claude when passed back to the API.

<Note>
It is only strictly necessary to send back thinking blocks when using [tools with extended thinking](/docs/en/build-with-claude/extended-thinking#extended-thinking-with-tool-use). Otherwise you can omit thinking blocks from previous turns, or let the API strip them for you if you pass them back.

If sending back thinking blocks, we recommend passing everything back as you received it for consistency and to avoid potential issues.
</Note>

Here are some important considerations on thinking encryption:
- When [streaming responses](/docs/en/build-with-claude/extended-thinking#streaming-thinking), the signature is added via a `signature_delta` inside a `content_block_delta` event just before the `content_block_stop` event.
- `signature` values are significantly longer in Claude 4 models than in previous models.
- The `signature` field is an opaque field and should not be interpreted or parsed - it exists solely for verification purposes.
- `signature` values are compatible across platforms (Claude APIs, [Amazon Bedrock](/docs/en/build-with-claude/claude-on-amazon-bedrock), and [Vertex AI](/docs/en/build-with-claude/claude-on-vertex-ai)). Values generated on one platform will be compatible with another.

### Thinking redaction

Occasionally Claude's internal reasoning will be flagged by our safety systems. When this occurs, we encrypt some or all of the `thinking` block and return it to you as a `redacted_thinking` block. `redacted_thinking` blocks are decrypted when passed back to the API, allowing Claude to continue its response without losing context.

When building customer-facing applications that use extended thinking:

- Be aware that redacted thinking blocks contain encrypted content that isn't human-readable
- Consider providing a simple explanation like: "Some of Claude's internal reasoning has been automatically encrypted for safety reasons. This doesn't affect the quality of responses."
- If showing thinking blocks to users, you can filter out redacted blocks while preserving normal thinking blocks
- Be transparent that using extended thinking features may occasionally result in some reasoning being encrypted
- Implement appropriate error handling to gracefully manage redacted thinking without breaking your UI

Here's an example showing both normal and redacted thinking blocks:

```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me analyze this step by step...",
      "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3URve/op91XWHOEBLLqIOMfFG/UvLEczmEsUjavL...."
    },
    {
      "type": "redacted_thinking",
      "data": "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2Xlxh0L5L8rLVyIwxtE3rAFBa8cr3qpPkNRj2YfWXGmKDxH4mPnZ5sQ7vB9URj2pLmN3kF8/dW5hR7xJ0aP1oLs9yTcMnKVf2wRpEGjH9XZaBt4UvDcPrQ..."
    },
    {
      "type": "text",
      "text": "Based on my analysis..."
    }
  ]
}
```

<Note>
Seeing redacted thinking blocks in your output is expected behavior. The model can still use this redacted reasoning to inform its responses while maintaining safety guardrails.

If you need to test redacted thinking handling in your application, you can use this special test string as your prompt: `ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB`
</Note>

When passing `thinking` and `redacted_thinking` blocks back to the API in a multi-turn conversation, you must include the complete unmodified block back to the API for the last assistant turn. This is critical for maintaining the model's reasoning flow. We suggest always passing back all thinking blocks to the API. For more details, see the [Preserving thinking blocks](/docs/en/build-with-claude/extended-thinking#preserving-thinking-blocks) section.

### Pricing

For complete pricing information including base rates, cache writes, cache hits, and output tokens, see the [pricing page](/docs/en/about-claude/pricing).

The thinking process incurs charges for:
- Tokens used during thinking (output tokens)
- Thinking blocks from the last assistant turn included in subsequent requests (input tokens)
- Standard text output tokens

<Note>
When extended thinking is enabled, a specialized system prompt is automatically included to support this feature.
</Note>

When using summarized thinking:
- **Input tokens**: Tokens in your original request (excludes thinking tokens from previous turns)
- **Output tokens (billed)**: The original thinking tokens that Claude generated internally
- **Output tokens (visible)**: The summarized thinking tokens you see in the response
- **No charge**: Tokens used to generate the summary

<Warning>
The billed output token count will **not** match the visible token count in the response. You are billed for the full thinking process, not the summary you see.
</Warning>

### Additional topics

The extended thinking page covers several topics in more detail with mode-specific code examples:

- **[Tool use with thinking](/docs/en/build-with-claude/extended-thinking#extended-thinking-with-tool-use)**: The same rules apply for adaptive thinking: preserve thinking blocks between tool calls and be aware of `tool_choice` limitations when thinking is active.
- **[Prompt caching](/docs/en/build-with-claude/extended-thinking#extended-thinking-with-prompt-caching)**: With adaptive thinking, consecutive requests using the same thinking mode preserve cache breakpoints. Switching between `adaptive` and `enabled`/`disabled` modes breaks cache breakpoints for messages (system prompts and tool definitions remain cached).
- **[Context windows](/docs/en/build-with-claude/extended-thinking#max-tokens-and-context-window-size-with-extended-thinking)**: How thinking tokens interact with `max_tokens` and context window limits.

## Next steps

<CardGroup>
  <Card title="Extended thinking" icon="settings" href="/docs/en/build-with-claude/extended-thinking">
    Learn more about extended thinking, including manual mode, tool use, and prompt caching.
  </Card>
  <Card title="Effort parameter" icon="gauge" href="/docs/en/build-with-claude/effort">
    Control how thoroughly Claude responds with the effort parameter.
  </Card>
</CardGroup>
