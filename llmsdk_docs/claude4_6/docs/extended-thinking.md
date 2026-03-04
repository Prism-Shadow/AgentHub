# Building with extended thinking

---

Extended thinking gives Claude enhanced reasoning capabilities for complex tasks, while providing varying levels of transparency into its step-by-step thought process before it delivers its final answer.

<Note>
For Claude Opus 4.6, use [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) (`thinking: {type: "adaptive"}`) with the [effort parameter](/docs/en/build-with-claude/effort) instead of the manual thinking mode described on this page. The manual `thinking: {type: "enabled", budget_tokens: N}` configuration is deprecated on Opus 4.6 and will be removed in a future model release.
</Note>

## Supported models

Extended thinking is supported in the following models:

- Claude Opus 4.6 (`claude-opus-4-6`), [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) only; manual mode (`type: "enabled"`) is deprecated
- Claude Opus 4.5 (`claude-opus-4-5-20251101`)
- Claude Opus 4.1 (`claude-opus-4-1-20250805`)
- Claude Opus 4 (`claude-opus-4-20250514`)
- Claude Sonnet 4.6 (`claude-sonnet-4-6`), supports both manual extended thinking with [interleaved mode](#interleaved-thinking) and [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking)
- Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- Claude Sonnet 3.7 (`claude-3-7-sonnet-20250219`) ([deprecated](/docs/en/about-claude/model-deprecations))
- Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)

<Note>
API behavior differs across Claude Sonnet 3.7 and Claude 4 models, but the API shapes remain exactly the same.

For more information, see [Differences in thinking across model versions](#differences-in-thinking-across-model-versions).
</Note>

## How extended thinking works

When extended thinking is turned on, Claude creates `thinking` content blocks where it outputs its internal reasoning. Claude incorporates insights from this reasoning before crafting a final response.

The API response will include `thinking` content blocks, followed by `text` content blocks.

Here's an example of the default response format:

```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me analyze this step by step...",
      "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3URve/op91XWHOEBLLqIOMfFG/UvLEczmEsUjavL...."
    },
    {
      "type": "text",
      "text": "Based on my analysis..."
    }
  ]
}
```

For more information about the response format of extended thinking, see the [Messages API Reference](/docs/en/api/messages).

## How to use extended thinking

Here is an example of using extended thinking in the Messages API:

<CodeGroup>
```bash Shell
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-sonnet-4-6",
    "max_tokens": 16000,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 10000
    },
    "messages": [
        {
            "role": "user",
            "content": "Are there an infinite number of prime numbers such that n mod 4 == 3?"
        }
    ]
}'
```

```python Python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[
        {
            "role": "user",
            "content": "Are there an infinite number of prime numbers such that n mod 4 == 3?",
        }
    ],
)

# The response will contain summarized thinking blocks and text blocks
for block in response.content:
    if block.type == "thinking":
        print(f"\nThinking summary: {block.thinking}")
    elif block.type == "text":
        print(f"\nResponse: {block.text}")
```

```typescript TypeScript hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const response = await client.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 16000,
  thinking: {
    type: "enabled",
    budget_tokens: 10000
  },
  messages: [
    {
      role: "user",
      content: "Are there an infinite number of prime numbers such that n mod 4 == 3?"
    }
  ]
});

// The response will contain summarized thinking blocks and text blocks
for (const block of response.content) {
  if (block.type === "thinking") {
    console.log(`\nThinking summary: ${block.thinking}`);
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
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 16000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 10000),
            Messages = [
                new() {
                    Role = Role.User,
                    Content = "Are there an infinite number of prime numbers such that n mod 4 == 3?"
                }
            ]
        };

        var message = await client.Messages.Create(parameters);

        foreach (var block in message.Content)
        {
            if (block.TryPickThinking(out ThinkingBlock? thinking))
            {
                Console.WriteLine($"\nThinking summary: {thinking.Thinking}");
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

To turn on extended thinking, add a `thinking` object, with the `type` parameter set to `enabled` and the `budget_tokens` to a specified token budget for extended thinking. For Claude Opus 4.6, use `type: "adaptive"` instead. See [Adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) for details. While `type: "enabled"` with `budget_tokens` is still supported on Opus 4.6, it is deprecated and will be removed in a future release.

The `budget_tokens` parameter determines the maximum number of tokens Claude is allowed to use for its internal reasoning process. In Claude 4 and later models, this limit applies to full thinking tokens, and not to [the summarized output](#summarized-thinking). Larger budgets can improve response quality by enabling more thorough analysis for complex problems, although Claude may not use the entire budget allocated, especially at ranges above 32k.

<Warning>
`budget_tokens` is deprecated on Claude Opus 4.6 and will be removed in a future model release. Use [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) with the [effort parameter](/docs/en/build-with-claude/effort) to control thinking depth instead.
</Warning>

<Note>
Claude Opus 4.6 supports up to 128K output tokens. Earlier models support up to 64K output tokens.
</Note>

`budget_tokens` must be set to a value less than `max_tokens`. However, when using [interleaved thinking with tools](#interleaved-thinking), you can exceed this limit as the token limit becomes your entire context window (200k tokens).

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

### Streaming thinking

You can stream extended thinking responses using [server-sent events (SSE)](https://developer.mozilla.org/en-US/Web/API/Server-sent%5Fevents/Using%5Fserver-sent%5Fevents).

When streaming is enabled for extended thinking, you receive thinking content via `thinking_delta` events.

For more documentation on streaming via the Messages API, see [Streaming Messages](/docs/en/build-with-claude/streaming).

Here's how to handle streaming with thinking:

<CodeGroup>
```bash Shell
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-sonnet-4-6",
    "max_tokens": 16000,
    "stream": true,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 10000
    },
    "messages": [
        {
            "role": "user",
            "content": "What is the greatest common divisor of 1071 and 462?"
        }
    ]
}'
```

```python Python hidelines={1..4}
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[
        {
            "role": "user",
            "content": "What is the greatest common divisor of 1071 and 462?",
        }
    ],
) as stream:
    thinking_started = False
    response_started = False

    for event in stream:
        if event.type == "content_block_start":
            print(f"\nStarting {event.content_block.type} block...")
            # Reset flags for each new block
            thinking_started = False
            response_started = False
        elif event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                if not thinking_started:
                    print("Thinking: ", end="", flush=True)
                    thinking_started = True
                print(event.delta.thinking, end="", flush=True)
            elif event.delta.type == "text_delta":
                if not response_started:
                    print("Response: ", end="", flush=True)
                    response_started = True
                print(event.delta.text, end="", flush=True)
        elif event.type == "content_block_stop":
            print("\nBlock complete.")
```

```typescript TypeScript hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const stream = await client.messages.stream({
  model: "claude-sonnet-4-6",
  max_tokens: 16000,
  thinking: {
    type: "enabled",
    budget_tokens: 10000
  },
  messages: [
    {
      role: "user",
      content: "What is the greatest common divisor of 1071 and 462?"
    }
  ]
});

let thinkingStarted = false;
let responseStarted = false;

for await (const event of stream) {
  if (event.type === "content_block_start") {
    console.log(`\nStarting ${event.content_block.type} block...`);
    // Reset flags for each new block
    thinkingStarted = false;
    responseStarted = false;
  } else if (event.type === "content_block_delta") {
    if (event.delta.type === "thinking_delta") {
      if (!thinkingStarted) {
        process.stdout.write("Thinking: ");
        thinkingStarted = true;
      }
      process.stdout.write(event.delta.thinking);
    } else if (event.delta.type === "text_delta") {
      if (!responseStarted) {
        process.stdout.write("Response: ");
        responseStarted = true;
      }
      process.stdout.write(event.delta.text);
    }
  } else if (event.type === "content_block_stop") {
    console.log("\nBlock complete.");
  }
}
```

```csharp C#
using System;
using System.Threading.Tasks;
using Anthropic;
using Anthropic.Models.Messages;

public class Program
{
    public static async Task Main()
    {
        AnthropicClient client = new();

        var parameters = new MessageCreateParams
        {
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 16000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 10000),
            Messages = [new() { Role = Role.User, Content = "What is the greatest common divisor of 1071 and 462?" }]
        };

        bool thinkingStarted = false;
        bool responseStarted = false;

        await foreach (var streamEvent in client.Messages.CreateStreaming(parameters))
        {
            if (streamEvent.Type == "content_block_start")
            {
                Console.WriteLine($"\nStarting {streamEvent.ContentBlock?.Type} block...");
                thinkingStarted = false;
                responseStarted = false;
            }
            else if (streamEvent.Type == "content_block_delta")
            {
                if (streamEvent.Delta?.Type == "thinking_delta")
                {
                    if (!thinkingStarted)
                    {
                        Console.Write("Thinking: ");
                        thinkingStarted = true;
                    }
                    Console.Write(streamEvent.Delta.Thinking);
                }
                else if (streamEvent.Delta?.Type == "text_delta")
                {
                    if (!responseStarted)
                    {
                        Console.Write("Response: ");
                        responseStarted = true;
                    }
                    Console.Write(streamEvent.Delta.Text);
                }
            }
            else if (streamEvent.Type == "content_block_stop")
            {
                Console.WriteLine("\nBlock complete.");
            }
        }
    }
}
```

</CodeGroup>

<TryInConsoleButton userPrompt="What is the greatest common divisor of 1071 and 462?" thinkingBudgetTokens={16000}>
  Try in Console
</TryInConsoleButton>

Example streaming output:
```sse
event: message_start
data: {"type": "message_start", "message": {"id": "msg_01...", "type": "message", "role": "assistant", "content": [], "model": "claude-sonnet-4-6", "stop_reason": null, "stop_sequence": null}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "I need to find the GCD of 1071 and 462 using the Euclidean algorithm.\n\n1071 = 2 × 462 + 147"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "\n462 = 3 × 147 + 21\n147 = 7 × 21 + 0\n\nSo GCD(1071, 462) = 21"}}

// Additional thinking deltas...

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "The greatest common divisor of 1071 and 462 is **21**."}}

// Additional text deltas...

event: content_block_stop
data: {"type": "content_block_stop", "index": 1}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}}

event: message_stop
data: {"type": "message_stop"}
```

<Note>
When using streaming with thinking enabled, you might notice that text sometimes arrives in larger chunks alternating with smaller, token-by-token delivery. This is expected behavior, especially for thinking content.

The streaming system needs to process content in batches for optimal performance, which can result in this "chunky" delivery pattern, with possible delays between streaming events. Anthropic is continuously working to improve this experience, with future updates focused on making thinking content stream more smoothly.
</Note>

## Extended thinking with tool use

Extended thinking can be used alongside [tool use](/docs/en/agents-and-tools/tool-use/overview), allowing Claude to reason through tool selection and results processing.

When using extended thinking with tool use, be aware of the following limitations:

1. **Tool choice limitation**: Tool use with thinking only supports `tool_choice: {"type": "auto"}` (the default) or `tool_choice: {"type": "none"}`. Using `tool_choice: {"type": "any"}` or `tool_choice: {"type": "tool", "name": "..."}` will result in an error because these options force tool use, which is incompatible with extended thinking.

2. **Preserving thinking blocks**: During tool use, you must pass `thinking` blocks back to the API for the last assistant message. Include the complete unmodified block back to the API to maintain reasoning continuity.

### Toggling thinking modes in conversations

You cannot toggle thinking in the middle of an assistant turn, including during tool use loops. The entire assistant turn should operate in a single thinking mode:

- **If thinking is enabled**, the final assistant turn should start with a thinking block.
- **If thinking is disabled**, the final assistant turn should not contain any thinking blocks

From the model's perspective, **tool use loops are part of the assistant turn**. An assistant turn doesn't complete until Claude finishes its full response, which may include multiple tool calls and results.

For example, this sequence is all part of a **single assistant turn**:
```text
User: "What's the weather in Paris?"
Assistant: [thinking] + [tool_use: get_weather]
User: [tool_result: "20°C, sunny"]
Assistant: [text: "The weather in Paris is 20°C and sunny"]
```

Even though there are multiple API messages, the tool use loop is conceptually part of one continuous assistant response.

#### Graceful thinking degradation

When a mid-turn thinking conflict occurs (such as toggling thinking on or off during a tool use loop), the API automatically disables thinking for that request. To preserve model quality and remain on-distribution, the API may:

- Strip thinking blocks from the conversation when they would create an invalid turn structure
- Disable thinking for the current request when the conversation history is incompatible with thinking being enabled

This means that attempting to toggle thinking mid-turn won't cause an error, but thinking will be silently disabled for that request. To confirm whether thinking was active, check for the presence of `thinking` blocks in the response.

#### Practical guidance

**Best practice**: Plan your thinking strategy at the start of each turn rather than trying to toggle mid-turn.

**Example: Toggling thinking after completing a turn**
```text
User: "What's the weather?"
Assistant: [tool_use] (thinking disabled)
User: [tool_result]
Assistant: [text: "It's sunny"]
User: "What about tomorrow?"
Assistant: [thinking] + [text: "..."] (thinking enabled - new turn)
```

By completing the assistant turn before toggling thinking, you ensure that thinking is actually enabled for the new request.

<Note>
Toggling thinking modes also invalidates prompt caching for message history. For more details, see the [Extended thinking with prompt caching](#extended-thinking-with-prompt-caching) section.
</Note>

<section title="Example: Passing thinking blocks with tool results">

Here's a practical example showing how to preserve thinking blocks when providing tool results:

<CodeGroup>
```python Python
weather_tool = {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}

# First request - Claude responds with thinking and tool request
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    tools=[weather_tool],
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
)
```

```typescript TypeScript nocheck
const weatherTool: Anthropic.Tool = {
  name: "get_weather",
  description: "Get current weather for a location",
  input_schema: {
    type: "object",
    properties: {
      location: { type: "string" }
    },
    required: ["location"]
  }
};

// First request - Claude responds with thinking and tool request
const response = await client.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 16000,
  thinking: {
    type: "enabled",
    budget_tokens: 10000
  },
  tools: [weatherTool],
  messages: [{ role: "user", content: "What's the weather in Paris?" }]
});
```

```csharp C#
using System;
using System.Text.Json;
using System.Threading.Tasks;
using Anthropic;
using Anthropic.Models.Messages;

class Program
{
    static async Task Main(string[] args)
    {
        AnthropicClient client = new();

        var weatherTool = new ToolUnion(new Tool()
        {
            Name = "get_weather",
            Description = "Get current weather for a location",
            InputSchema = new InputSchema()
            {
                Properties = new Dictionary<string, JsonElement>
                {
                    ["location"] = JsonSerializer.SerializeToElement(new { type = "string" }),
                },
                Required = ["location"],
            },
        });

        var parameters = new MessageCreateParams
        {
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 16000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 10000),
            Tools = [weatherTool],
            Messages = [new() { Role = Role.User, Content = "What's the weather in Paris?" }]
        };

        var message = await client.Messages.Create(parameters);
        Console.WriteLine(message);
    }
}
```

</CodeGroup>

The API response will include thinking, text, and tool_use blocks:

```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "The user wants to know the current weather in Paris. I have access to a function `get_weather`...",
      "signature": "BDaL4VrbR2Oj0hO4XpJxT28J5TILnCrrUXoKiiNBZW9P+nr8XSj1zuZzAl4egiCCpQNvfyUuFFJP5CncdYZEQPPmLxYsNrcs...."
    },
    {
      "type": "text",
      "text": "I can help you get the current weather information for Paris. Let me check that for you"
    },
    {
      "type": "tool_use",
      "id": "toolu_01CswdEQBMshySk6Y9DFKrfq",
      "name": "get_weather",
      "input": {
        "location": "Paris"
      }
    }
  ]
}
```

Now let's continue the conversation and use the tool

<CodeGroup>
```python Python hidelines={1..4}
import anthropic
from typing import Any

client = anthropic.Anthropic()
weather_tool = {
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city and state"}
        },
        "required": ["location"],
    },
}
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    tools=[weather_tool],
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
)
# Extract thinking block and tool use block
thinking_block = next(
    (block for block in response.content if block.type == "thinking"), None
)
tool_use_block = next(
    (block for block in response.content if block.type == "tool_use"), None
)

# Call your actual weather API, here is where your actual API call would go
# let's pretend this is what we get back
weather_data = {"temperature": 88}

# Second request - Include thinking block and tool result
# No new thinking blocks will be generated in the response
continuation = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    tools=[weather_tool],
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"},
        # notice that the thinking_block is passed in as well as the tool_use_block
        # if this is not passed in, an error is raised
        {"role": "assistant", "content": [thinking_block, tool_use_block]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": f"Current temperature: {weather_data['temperature']}°F",
                }
            ],
        },
    ],
)
```

```typescript TypeScript nocheck
// Extract thinking block and tool use block
const thinkingBlock = response.content.find(
  (block): block is Anthropic.ThinkingBlock => block.type === "thinking"
);
const toolUseBlock = response.content.find(
  (block): block is Anthropic.ToolUseBlock => block.type === "tool_use"
);

// Call your actual weather API, here is where your actual API call would go
// let's pretend this is what we get back
const weatherData = { temperature: 88 };

if (thinkingBlock && toolUseBlock) {
  // Second request - Include thinking block and tool result
  // No new thinking blocks will be generated in the response
  const continuation = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 16000,
    thinking: {
      type: "enabled",
      budget_tokens: 10000
    },
    tools: [weatherTool],
    messages: [
      { role: "user", content: "What's the weather in Paris?" },
      // notice that the thinkingBlock is passed in as well as the toolUseBlock
      // if this is not passed in, an error is raised
      { role: "assistant", content: [thinkingBlock, toolUseBlock] },
      {
        role: "user",
        content: [
          {
            type: "tool_result" as const,
            tool_use_id: toolUseBlock.id,
            content: `Current temperature: ${weatherData.temperature}°F`
          }
        ]
      }
    ]
  });
}
```

```csharp C# nocheck
using System;
using System.Text.Json;
using System.Linq;
using System.Threading.Tasks;
using Anthropic;
using Anthropic.Models.Messages;

public class Program
{
    public static async Task Main(string[] args)
    {
        AnthropicClient client = new();

        var weatherTool = new ToolUnion(new Tool()
        {
            Name = "get_weather",
            Description = "Get current weather for a location",
            InputSchema = new InputSchema()
            {
                Properties = new Dictionary<string, JsonElement>
                {
                    ["location"] = JsonSerializer.SerializeToElement(new { type = "string", description = "City name" }),
                },
                Required = ["location"],
            },
        });

        var parameters = new MessageCreateParams
        {
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 16000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 10000),
            Tools = [weatherTool],
            Messages = [
                new() { Role = Role.User, Content = "What is the weather in Paris?" }
            ]
        };

        var response = await client.Messages.Create(parameters);

        // Extract thinking and tool_use blocks from response
        var thinkingBlock = response.Content.FirstOrDefault(b => b.TryPickThinking(out _));
        var toolUseBlock = response.Content.FirstOrDefault(b => b.TryPickToolUse(out _));

        var weatherData = new { temperature = 88 };

        // Build continuation with tool result
        var continuationParams = new MessageCreateParams
        {
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 16000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 10000),
            Tools = [weatherTool],
            Messages = [
                new() { Role = Role.User, Content = "What is the weather in Paris?" },
                new() { Role = Role.Assistant, Content = response.Content },
                new() { Role = Role.User, Content = new MessageParamContent(new List<ContentBlockParam>
                {
                    new ContentBlockParam(new ToolResultBlockParam()
                    {
                        ToolUseID = toolUseBlock?.Id ?? "",
                        Content = $"Current temperature: {weatherData.temperature}\u00b0F"
                    })
                })}
            ]
        };

        var continuation = await client.Messages.Create(continuationParams);
        Console.WriteLine(continuation);
    }
}
```

</CodeGroup>

The API response will now **only** include text

```json
{
  "content": [
    {
      "type": "text",
      "text": "Currently in Paris, the temperature is 88°F (31°C)"
    }
  ]
}
```

</section>

### Preserving thinking blocks

During tool use, you must pass `thinking` blocks back to the API, and you must include the complete unmodified block back to the API. This is critical for maintaining the model's reasoning flow and conversation integrity.

<Tip>
While you can omit `thinking` blocks from prior `assistant` role turns, always pass back all thinking blocks to the API for any multi-turn conversation. The API will:
- Automatically filter the provided thinking blocks
- Use the relevant thinking blocks necessary to preserve the model's reasoning
- Only bill for the input tokens for the blocks shown to Claude
</Tip>

<Note>
When toggling thinking modes during a conversation, remember that the entire assistant turn (including tool use loops) must operate in a single thinking mode. For more details, see [Toggling thinking modes in conversations](#toggling-thinking-modes-in-conversations).
</Note>

When Claude invokes tools, it is pausing its construction of a response to await external information. When tool results are returned, Claude will continue building that existing response. This necessitates preserving thinking blocks during tool use, for a couple of reasons:

1. **Reasoning continuity**: The thinking blocks capture Claude's step-by-step reasoning that led to tool requests. When you post tool results, including the original thinking ensures Claude can continue its reasoning from where it left off.

2. **Context maintenance**: While tool results appear as user messages in the API structure, they're part of a continuous reasoning flow. Preserving thinking blocks maintains this conceptual flow across multiple API calls. For more information on context management, see the [guide on context windows](/docs/en/build-with-claude/context-windows).

**Important**: When providing `thinking` blocks, the entire sequence of consecutive `thinking` blocks must match the outputs generated by the model during the original request; you cannot rearrange or modify the sequence of these blocks.

### Interleaved thinking

Extended thinking with tool use in Claude 4 models supports interleaved thinking, which enables Claude to think between tool calls and make more sophisticated reasoning after receiving tool results.

With interleaved thinking, Claude can:
- Reason about the results of a tool call before deciding what to do next
- Chain multiple tool calls with reasoning steps in between
- Make more nuanced decisions based on intermediate results

**Model support:**
- **Claude Opus 4.6**: Interleaved thinking is automatically enabled when using [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking). No beta header is needed. The `interleaved-thinking-2025-05-14` beta header is **deprecated** on Opus 4.6 and is safely ignored if included.
- **Claude Sonnet 4.6**: Supports the `interleaved-thinking-2025-05-14` beta header with manual extended thinking (`thinking: {type: "enabled"}`). You can also use [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking), which automatically enables interleaved thinking.
- **Other Claude 4 models** (Opus 4.5, Opus 4.1, Opus 4, Sonnet 4.5, Sonnet 4): Add [the beta header](/docs/en/api/beta-headers) `interleaved-thinking-2025-05-14` to your API request to enable interleaved thinking.

Here are some important considerations for interleaved thinking:
- With interleaved thinking, the `budget_tokens` can exceed the `max_tokens` parameter, as it represents the total budget across all thinking blocks within one assistant turn.
- Interleaved thinking is only supported for [tools used via the Messages API](/docs/en/agents-and-tools/tool-use/overview).
- Direct calls to the Claude API allow you to pass `interleaved-thinking-2025-05-14` in requests to any model, with no effect (except Opus 4.6, where it's deprecated and safely ignored).
- On 3rd-party platforms (for example, [Amazon Bedrock](/docs/en/build-with-claude/claude-on-amazon-bedrock) and [Vertex AI](/docs/en/build-with-claude/claude-on-vertex-ai)), if you pass `interleaved-thinking-2025-05-14` to any model aside from Claude Sonnet 4.6, Claude Opus 4.5, Claude Opus 4.1, Opus 4, Sonnet 4.5, or Sonnet 4, your request will fail.

<section title="Tool use without interleaved thinking">

Without interleaved thinking, Claude thinks once at the start of the assistant turn. Subsequent responses after tool results continue without new thinking blocks.

```text
User: "What's the total revenue if we sold 150 units at $50 each,
       and how does this compare to our average monthly revenue?"

Turn 1: [thinking] "I need to calculate 150 * $50, then check the database..."
        [tool_use: calculator] { "expression": "150 * 50" }
  ↓ tool result: "7500"

Turn 2: [tool_use: database_query] { "query": "SELECT AVG(revenue)..." }
        ↑ no thinking block
  ↓ tool result: "5200"

Turn 3: [text] "The total revenue is $7,500, which is 44% above your
        average monthly revenue of $5,200."
        ↑ no thinking block
```

</section>

<section title="Tool use with interleaved thinking">

With interleaved thinking enabled, Claude can think after receiving each tool result, allowing it to reason about intermediate results before continuing.

```text
User: "What's the total revenue if we sold 150 units at $50 each,
       and how does this compare to our average monthly revenue?"

Turn 1: [thinking] "I need to calculate 150 * $50 first..."
        [tool_use: calculator] { "expression": "150 * 50" }
  ↓ tool result: "7500"

Turn 2: [thinking] "Got $7,500. Now I should query the database to compare..."
        [tool_use: database_query] { "query": "SELECT AVG(revenue)..." }
        ↑ thinking after receiving calculator result
  ↓ tool result: "5200"

Turn 3: [thinking] "$7,500 vs $5,200 average - that's a 44% increase..."
        [text] "The total revenue is $7,500, which is 44% above your
        average monthly revenue of $5,200."
        ↑ thinking before final answer
```

</section>

## Extended thinking with prompt caching

[Prompt caching](/docs/en/build-with-claude/prompt-caching) with thinking has several important considerations:

<Tip>
Extended thinking tasks often take longer than 5 minutes to complete. Consider using the [1-hour cache duration](/docs/en/build-with-claude/prompt-caching#1-hour-cache-duration) to maintain cache hits across longer thinking sessions and multi-step workflows.
</Tip>

**Thinking block context removal**
- Thinking blocks from previous turns are removed from context, which can affect cache breakpoints
- When continuing conversations with tool use, thinking blocks are cached and count as input tokens when read from cache
- This creates a tradeoff: while thinking blocks don't consume context window space visually, they still count toward your input token usage when cached
- If thinking becomes disabled and you pass thinking content in the current tool use turn, the thinking content will be stripped and thinking will remain disabled for that request

**Cache invalidation patterns**
- Changes to thinking parameters (enabled/disabled or budget allocation) invalidate message cache breakpoints
- [Interleaved thinking](#interleaved-thinking) amplifies cache invalidation, as thinking blocks can occur between multiple [tool calls](#extended-thinking-with-tool-use)
- System prompts and tools remain cached despite thinking parameter changes or block removal

<Note>
While thinking blocks are removed for caching and context calculations, they must be preserved when continuing conversations with [tool use](#extended-thinking-with-tool-use), especially with [interleaved thinking](#interleaved-thinking).
</Note>

### Understanding thinking block caching behavior

When using extended thinking with tool use, thinking blocks exhibit specific caching behavior that affects token counting:

**How it works:**

1. Caching only occurs when you make a subsequent request that includes tool results
2. When the subsequent request is made, the previous conversation history (including thinking blocks) can be cached
3. These cached thinking blocks count as input tokens in your usage metrics when read from the cache
4. When a non-tool-result user block is included, all previous thinking blocks are ignored and stripped from context

**Detailed example flow:**

**Request 1:**
```text
User: "What's the weather in Paris?"
```
**Response 1:**
```text
[thinking_block_1] + [tool_use block 1]
```

**Request 2:**
```text
User: ["What's the weather in Paris?"],
Assistant: [thinking_block_1] + [tool_use block 1],
User: [tool_result_1, cache=True]
```
**Response 2:**
```text
[thinking_block_2] + [text block 2]
```
Request 2 writes a cache of the request content (not the response). The cache includes the original user message, the first thinking block, tool use block, and the tool result.

**Request 3:**
```text
User: ["What's the weather in Paris?"],
Assistant: [thinking_block_1] + [tool_use block 1],
User: [tool_result_1, cache=True],
Assistant: [thinking_block_2] + [text block 2],
User: [Text response, cache=True]
```
For Claude Opus 4.5 and later (including Claude Opus 4.6), all previous thinking blocks are kept by default. For older models, because a non-tool-result user block was included, all previous thinking blocks are ignored. This request will be processed the same as:
```text
User: ["What's the weather in Paris?"],
Assistant: [tool_use block 1],
User: [tool_result_1, cache=True],
Assistant: [text block 2],
User: [Text response, cache=True]
```

**Key points:**
- This caching behavior happens automatically, even without explicit `cache_control` markers
- This behavior is consistent whether using regular thinking or interleaved thinking

<section title="System prompt caching (preserved when thinking changes)">

<CodeGroup>
```python Python hidelines={1,4..5}
from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup

client = Anthropic()


def fetch_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text
    text = soup.get_text()

    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


# Fetch the content of the article
book_url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
book_content = fetch_article_content(book_url)
# Use just enough text for caching (first few chapters)
LARGE_TEXT = book_content[:5000]

SYSTEM_PROMPT = [
    {
        "type": "text",
        "text": "You are an AI assistant that is tasked with literary analysis. Analyze the following text carefully.",
    },
    {"type": "text", "text": LARGE_TEXT, "cache_control": {"type": "ephemeral"}},
]

MESSAGES = [{"role": "user", "content": "Analyze the tone of this passage."}]

# First request - establish cache
print("First request - establishing cache")
response1 = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=20000,
    thinking={"type": "enabled", "budget_tokens": 4000},
    system=SYSTEM_PROMPT,
    messages=MESSAGES,
)

print(f"First response usage: {response1.usage}")

MESSAGES.append({"role": "assistant", "content": response1.content})
MESSAGES.append({"role": "user", "content": "Analyze the characters in this passage."})
# Second request - same thinking parameters (cache hit expected)
print("\nSecond request - same thinking parameters (cache hit expected)")
response2 = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=20000,
    thinking={"type": "enabled", "budget_tokens": 4000},
    system=SYSTEM_PROMPT,
    messages=MESSAGES,
)

print(f"Second response usage: {response2.usage}")

# Third request - different thinking parameters (cache miss for messages)
print("\nThird request - different thinking parameters (cache miss for messages)")
response3 = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=20000,
    thinking={
        "type": "enabled",
        "budget_tokens": 8000,  # Changed thinking budget
    },
    system=SYSTEM_PROMPT,  # System prompt remains cached
    messages=MESSAGES,  # Messages cache is invalidated
)

print(f"Third response usage: {response3.usage}")
```

```typescript TypeScript nocheck hidelines={1}
import Anthropic from "@anthropic-ai/sdk";
import axios from "axios";
import * as cheerio from "cheerio";

const client = new Anthropic();

async function fetchArticleContent(url: string): Promise<string> {
  const response = await axios.get(url);
  const $ = cheerio.load(response.data);
  $("script, style").remove();
  let text = $.text();
  const lines = text.split("\n").map((line) => line.trim());
  text = lines.filter((line) => line.length > 0).join("\n");
  return text;
}

async function main(): Promise<void> {
  const bookUrl = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt";
  const bookContent = await fetchArticleContent(bookUrl);
  const LARGE_TEXT = bookContent.slice(0, 5000);

  const SYSTEM_PROMPT: Anthropic.TextBlockParam[] = [
    {
      type: "text",
      text: "You are an AI assistant that is tasked with literary analysis. Analyze the following text carefully."
    },
    {
      type: "text",
      text: LARGE_TEXT,
      cache_control: { type: "ephemeral" }
    }
  ];

  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: "Analyze the tone of this passage." }
  ];

  // First request - establish cache
  console.log("First request - establishing cache");
  const response1 = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 20000,
    thinking: { type: "enabled", budget_tokens: 4000 },
    system: SYSTEM_PROMPT,
    messages
  });

  console.log(`First response usage: ${JSON.stringify(response1.usage)}`);

  messages.push({
    role: "assistant",
    content: response1.content as Anthropic.ContentBlockParam[]
  });
  messages.push({
    role: "user",
    content: "Analyze the characters in this passage."
  });

  // Second request - same thinking parameters (cache hit expected)
  console.log("\nSecond request - same thinking parameters (cache hit expected)");
  const response2 = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 20000,
    thinking: { type: "enabled", budget_tokens: 4000 },
    system: SYSTEM_PROMPT,
    messages
  });

  console.log(`Second response usage: ${JSON.stringify(response2.usage)}`);

  // Third request - different thinking parameters (cache miss for messages)
  console.log("\nThird request - different thinking parameters (cache miss for messages)");
  const response3 = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 20000,
    thinking: { type: "enabled", budget_tokens: 8000 },
    system: SYSTEM_PROMPT,
    messages
  });

  console.log(`Third response usage: ${JSON.stringify(response3.usage)}`);
}

main();
```

```csharp C# nocheck
using System;
using System.Net.Http;
using System.Threading.Tasks;
using System.Collections.Generic;
using Anthropic;
using Anthropic.Models.Messages;

public class Program
{
    static async Task Main(string[] args)
    {
        AnthropicClient client = new();

        // Fetch book content
        using var httpClient = new HttpClient();
        var bookContent = await httpClient.GetStringAsync("https://www.gutenberg.org/cache/epub/1342/pg1342.txt");
        var largeText = bookContent.Substring(0, Math.Min(5000, bookContent.Length));

        var systemPrompt = new MessageCreateParamsSystem(new List<TextBlockParam>
        {
            new TextBlockParam()
            {
                Text = "You are an AI assistant that is tasked with literary analysis. Analyze the following text carefully."
            },
            new TextBlockParam()
            {
                Text = largeText,
                CacheControl = new CacheControlEphemeral(),
            },
        });

        var messages = new List<MessageParam>
        {
            new() { Role = Role.User, Content = "Analyze the tone of this passage." }
        };

        // First request - establish cache
        Console.WriteLine("First request - establishing cache");
        var parameters1 = new MessageCreateParams
        {
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 20000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 4000),
            System = systemPrompt,
            Messages = messages
        };

        var response1 = await client.Messages.Create(parameters1);
        Console.WriteLine($"First response usage: {response1.Usage}");

        messages.Add(new() { Role = Role.Assistant, Content = response1.Content });
        messages.Add(new() { Role = Role.User, Content = "Analyze the characters in this passage." });

        // Second request - same thinking parameters (cache hit expected)
        Console.WriteLine("\nSecond request - same thinking parameters (cache hit expected)");
        var parameters2 = new MessageCreateParams
        {
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 20000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 4000),
            System = systemPrompt,
            Messages = messages
        };

        var response2 = await client.Messages.Create(parameters2);
        Console.WriteLine($"Second response usage: {response2.Usage}");

        // Third request - different thinking parameters (cache miss for messages)
        Console.WriteLine("\nThird request - different thinking parameters (cache miss for messages)");
        var parameters3 = new MessageCreateParams
        {
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 20000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 8000),
            System = systemPrompt,
            Messages = messages
        };

        var response3 = await client.Messages.Create(parameters3);
        Console.WriteLine($"Third response usage: {response3.Usage}");
    }
}
```
</CodeGroup>

</section>
<section title="Messages caching (invalidated when thinking changes)">

<CodeGroup>
```python Python hidelines={1,4..5}
from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup

client = Anthropic()


def fetch_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text
    text = soup.get_text()

    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


# Fetch the content of the article
book_url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
book_content = fetch_article_content(book_url)
# Use just enough text for caching (first few chapters)
LARGE_TEXT = book_content[:5000]

# No system prompt - caching in messages instead
MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": LARGE_TEXT,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "Analyze the tone of this passage."},
        ],
    }
]

# First request - establish cache
print("First request - establishing cache")
response1 = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=20000,
    thinking={"type": "enabled", "budget_tokens": 4000},
    messages=MESSAGES,
)

print(f"First response usage: {response1.usage}")

MESSAGES.append({"role": "assistant", "content": response1.content})
MESSAGES.append({"role": "user", "content": "Analyze the characters in this passage."})
# Second request - same thinking parameters (cache hit expected)
print("\nSecond request - same thinking parameters (cache hit expected)")
response2 = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=20000,
    thinking={
        "type": "enabled",
        "budget_tokens": 4000,  # Same thinking budget
    },
    messages=MESSAGES,
)

print(f"Second response usage: {response2.usage}")

MESSAGES.append({"role": "assistant", "content": response2.content})
MESSAGES.append({"role": "user", "content": "Analyze the setting in this passage."})

# Third request - different thinking budget (cache miss expected)
print("\nThird request - different thinking budget (cache miss expected)")
response3 = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=20000,
    thinking={
        "type": "enabled",
        "budget_tokens": 8000,  # Different thinking budget breaks cache
    },
    messages=MESSAGES,
)

print(f"Third response usage: {response3.usage}")
```

```typescript TypeScript nocheck hidelines={1}
import Anthropic from "@anthropic-ai/sdk";
import axios from "axios";
import * as cheerio from "cheerio";

const client = new Anthropic();

async function fetchArticleContent(url: string): Promise<string> {
  const response = await axios.get(url);
  const $ = cheerio.load(response.data);

  // Remove script and style elements
  $("script, style").remove();

  // Get text
  let text = $.text();

  // Clean up text (break into lines, remove whitespace)
  const lines = text.split("\n").map((line) => line.trim());
  const chunks = lines.flatMap((line) => line.split("  ").map((phrase) => phrase.trim()));
  text = chunks.filter((chunk) => chunk).join("\n");

  return text;
}

async function main(): Promise<void> {
  const bookUrl = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt";
  const bookContent = await fetchArticleContent(bookUrl);
  const LARGE_TEXT = bookContent.substring(0, 5000);

  // No system prompt - caching in messages instead
  const messages: Anthropic.MessageParam[] = [
    {
      role: "user",
      content: [
        {
          type: "text",
          text: LARGE_TEXT,
          cache_control: { type: "ephemeral" }
        },
        {
          type: "text",
          text: "Analyze the tone of this passage."
        }
      ]
    }
  ];

  // First request - establish cache
  console.log("First request - establishing cache");
  const response1 = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 20000,
    thinking: { type: "enabled", budget_tokens: 4000 },
    messages
  });

  console.log("First response usage: ", response1.usage);

  messages.push(
    { role: "assistant", content: response1.content as Anthropic.ContentBlockParam[] },
    { role: "user", content: "Analyze the characters in this passage." }
  );

  // Second request - same thinking parameters (cache hit expected)
  console.log("\nSecond request - same thinking parameters (cache hit expected)");
  const response2 = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 20000,
    thinking: { type: "enabled", budget_tokens: 4000 },
    messages
  });

  console.log("Second response usage: ", response2.usage);

  messages.push(
    { role: "assistant", content: response2.content as Anthropic.ContentBlockParam[] },
    { role: "user", content: "Analyze the setting in this passage." }
  );

  // Third request - different thinking budget (cache miss expected)
  console.log("\nThird request - different thinking budget (cache miss expected)");
  const response3 = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 20000,
    thinking: { type: "enabled", budget_tokens: 8000 },
    messages
  });

  console.log("Third response usage: ", response3.usage);
}

main().catch(console.error);
```

```csharp C# nocheck
using System;
using System.Net.Http;
using System.Collections.Generic;
using System.Threading.Tasks;
using Anthropic;
using Anthropic.Models.Messages;

public class Program
{
    static async Task Main(string[] args)
    {
        AnthropicClient client = new();

        string bookUrl = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt";
        string bookContent = await FetchArticleContent(bookUrl);
        string largeText = bookContent.Substring(0, Math.Min(5000, bookContent.Length));

        Console.WriteLine("First request - establishing cache");
        var parameters1 = new MessageCreateParams
        {
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 20000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 4000),
            Messages =
            [
                new()
                {
                    Role = Role.User,
                    Content = new MessageParamContent(new List<ContentBlockParam>
                    {
                        new ContentBlockParam(new TextBlockParam()
                        {
                            Text = largeText,
                            CacheControl = new CacheControlEphemeral(),
                        }),
                        new ContentBlockParam(new TextBlockParam()
                        {
                            Text = "Analyze the tone of this passage."
                        }),
                    })
                }
            ]
        };

        var response1 = await client.Messages.Create(parameters1);
        Console.WriteLine($"First response usage: {response1.Usage}");

        Console.WriteLine("\nSecond request - same thinking parameters (cache hit expected)");
        var parameters2 = new MessageCreateParams
        {
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 20000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 4000),
            Messages =
            [
                new()
                {
                    Role = Role.User,
                    Content = new MessageParamContent(new List<ContentBlockParam>
                    {
                        new ContentBlockParam(new TextBlockParam()
                        {
                            Text = largeText,
                            CacheControl = new CacheControlEphemeral(),
                        }),
                        new ContentBlockParam(new TextBlockParam()
                        {
                            Text = "Analyze the tone of this passage."
                        }),
                    })
                },
                new()
                {
                    Role = Role.Assistant,
                    Content = response1.Content
                },
                new()
                {
                    Role = Role.User,
                    Content = "Analyze the characters in this passage."
                }
            ]
        };

        var response2 = await client.Messages.Create(parameters2);
        Console.WriteLine($"Second response usage: {response2.Usage}");

        Console.WriteLine("\nThird request - different thinking budget (cache miss expected)");
        var parameters3 = new MessageCreateParams
        {
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 20000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 8000),
            Messages =
            [
                new()
                {
                    Role = Role.User,
                    Content = new MessageParamContent(new List<ContentBlockParam>
                    {
                        new ContentBlockParam(new TextBlockParam()
                        {
                            Text = largeText,
                            CacheControl = new CacheControlEphemeral(),
                        }),
                        new ContentBlockParam(new TextBlockParam()
                        {
                            Text = "Analyze the tone of this passage."
                        }),
                    })
                },
                new()
                {
                    Role = Role.Assistant,
                    Content = response1.Content
                },
                new()
                {
                    Role = Role.User,
                    Content = "Analyze the characters in this passage."
                },
                new()
                {
                    Role = Role.Assistant,
                    Content = response2.Content
                },
                new()
                {
                    Role = Role.User,
                    Content = "Analyze the setting in this passage."
                }
            ]
        };

        var response3 = await client.Messages.Create(parameters3);
        Console.WriteLine($"Third response usage: {response3.Usage}");
    }

    static async Task<string> FetchArticleContent(string url)
    {
        using HttpClient httpClient = new();
        string content = await httpClient.GetStringAsync(url);
        return content;
    }
}
```

</CodeGroup>

Here is the output of the script (you may see slightly different numbers)

```text
First request - establishing cache
First response usage: { cache_creation_input_tokens: 1370, cache_read_input_tokens: 0, input_tokens: 17, output_tokens: 700 }

Second request - same thinking parameters (cache hit expected)

Second response usage: { cache_creation_input_tokens: 0, cache_read_input_tokens: 1370, input_tokens: 303, output_tokens: 874 }

Third request - different thinking budget (cache miss expected)
Third response usage: { cache_creation_input_tokens: 1370, cache_read_input_tokens: 0, input_tokens: 747, output_tokens: 619 }
```

This example demonstrates that when caching is set up in the messages array, changing the thinking parameters (budget_tokens increased from 4000 to 8000) **invalidates the cache**. The third request shows no cache hit with `cache_creation_input_tokens=1370` and `cache_read_input_tokens=0`, proving that message-based caching is invalidated when thinking parameters change.

</section>

## Max tokens and context window size with extended thinking

In older Claude models (prior to Claude Sonnet 3.7), if the sum of prompt tokens and `max_tokens` exceeded the model's context window, the system would automatically adjust `max_tokens` to fit within the context limit. This meant you could set a large `max_tokens` value and the system would silently reduce it as needed.

With Claude 3.7 and 4 models, `max_tokens` (which includes your thinking budget when thinking is enabled) is enforced as a strict limit. The system will now return a validation error if prompt tokens + `max_tokens` exceeds the context window size.

<Note>
You can read through the [guide on context windows](/docs/en/build-with-claude/context-windows) for a more thorough deep dive.
</Note>

### The context window with extended thinking

When calculating context window usage with thinking enabled, there are some considerations to be aware of:

- Thinking blocks from previous turns are stripped and not counted towards your context window
- Current turn thinking counts towards your `max_tokens` limit for that turn

The diagram below demonstrates the specialized token management when extended thinking is enabled:

![Context window diagram with extended thinking](/docs/images/context-window-thinking.svg)

The effective context window is calculated as:

```text
context window =
  (current input tokens - previous thinking tokens) +
  (thinking tokens + encrypted thinking tokens + text output tokens)
```

Use the [token counting API](/docs/en/build-with-claude/token-counting) to get accurate token counts for your specific use case, especially when working with multi-turn conversations that include thinking.

### The context window with extended thinking and tool use

When using extended thinking with tool use, thinking blocks must be explicitly preserved and returned with the tool results.

The effective context window calculation for extended thinking with tool use becomes:

```text
context window =
  (current input tokens + previous thinking tokens + tool use tokens) +
  (thinking tokens + encrypted thinking tokens + text output tokens)
```

The diagram below illustrates token management for extended thinking with tool use:

![Context window diagram with extended thinking and tool use](/docs/images/context-window-thinking-tools.svg)

### Managing tokens with extended thinking

Given the context window and `max_tokens` behavior with extended thinking Claude 3.7 and 4 models, you may need to:

- More actively monitor and manage your token usage
- Adjust `max_tokens` values as your prompt length changes
- Potentially use the [token counting endpoints](/docs/en/build-with-claude/token-counting) more frequently
- Be aware that previous thinking blocks don't accumulate in your context window

This change has been made to provide more predictable and transparent behavior, especially as maximum token limits have increased significantly.

## Thinking encryption

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

<section title="Example: Working with redacted thinking blocks">

This example demonstrates how to handle `redacted_thinking` blocks that may appear in responses when Claude's internal reasoning contains content flagged by safety systems:

<CodeGroup>
```python Python hidelines={1..4}
import anthropic

client = anthropic.Anthropic()

# Using a special prompt that triggers redacted thinking (for demonstration purposes only)
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[
        {
            "role": "user",
            "content": "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB",
        }
    ],
)

# Identify redacted thinking blocks
has_redacted_thinking = any(
    block.type == "redacted_thinking" for block in response.content
)

if has_redacted_thinking:
    print("Response contains redacted thinking blocks")
    # These blocks are still usable in subsequent requests

    # Extract all blocks (both redacted and non-redacted)
    all_thinking_blocks = [
        block
        for block in response.content
        if block.type in ["thinking", "redacted_thinking"]
    ]

    # When passing to subsequent requests, include all blocks without modification
    # This preserves the integrity of Claude's reasoning

    print(f"Found {len(all_thinking_blocks)} thinking blocks total")
    print(f"These blocks are still billable as output tokens")
```

```typescript TypeScript nocheck hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// Using a special prompt that triggers redacted thinking (for demonstration purposes only)
const response = await client.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 16000,
  thinking: {
    type: "enabled",
    budget_tokens: 10000
  },
  messages: [
    {
      role: "user",
      content:
        "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
    }
  ]
});

// Identify redacted thinking blocks
const hasRedactedThinking = response.content.some(
  (block) => block.type === "redacted_thinking"
);

if (hasRedactedThinking) {
  console.log("Response contains redacted thinking blocks");
  // These blocks are still usable in subsequent requests

  // Extract all blocks (both redacted and non-redacted)
  const allThinkingBlocks = response.content.filter(
    (block) => block.type === "thinking" || block.type === "redacted_thinking"
  );

  // When passing to subsequent requests, include all blocks without modification
  // This preserves the integrity of Claude's reasoning

  console.log(`Found ${allThinkingBlocks.length} thinking blocks total`);
  console.log("These blocks are still billable as output tokens");
}
```

```csharp C#
using System;
using System.Linq;
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
            Model = Model.ClaudeSonnet4_6,
            MaxTokens = 16000,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 10000),
            Messages = [new() {
                Role = Role.User,
                Content = "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
            }]
        };

        var response = await client.Messages.Create(parameters);

        bool hasRedactedThinking = response.Content.Any(block =>
            block.TryPickRedactedThinking(out _));

        if (hasRedactedThinking)
        {
            Console.WriteLine("Response contains redacted thinking blocks");

            int thinkingBlockCount = response.Content.Count(block =>
                block.TryPickThinking(out _) || block.TryPickRedactedThinking(out _));

            Console.WriteLine($"Found {thinkingBlockCount} thinking blocks total");
            Console.WriteLine("These blocks are still billable as output tokens");
        }
    }
}
```

</CodeGroup>

<TryInConsoleButton
  userPrompt="ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
  thinkingBudgetTokens={16000}
>
  Try in Console
</TryInConsoleButton>

</section>

## Differences in thinking across model versions

The Messages API handles thinking differently across Claude Sonnet 3.7 and Claude 4 models, primarily in redaction and summarization behavior.

See the table below for a condensed comparison:

| Feature | Claude Sonnet 3.7 | Claude 4 Models (pre-Opus 4.5) | Claude Opus 4.5 | Claude Sonnet 4.6 | Claude Opus 4.6 ([adaptive thinking](/docs/en/build-with-claude/adaptive-thinking)) |
|---------|------------------|-------------------------------|--------------------------|------------------|--------------------------|
| **Thinking Output** | Returns full thinking output | Returns summarized thinking | Returns summarized thinking | Returns summarized thinking | Returns summarized thinking |
| **Interleaved Thinking** | Not supported | Supported with `interleaved-thinking-2025-05-14` beta header | Supported with `interleaved-thinking-2025-05-14` beta header | Supported with `interleaved-thinking-2025-05-14` beta header or automatic with [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) | Automatic with adaptive thinking (beta header not supported) |
| **Thinking Block Preservation** | Not preserved across turns | Not preserved across turns | **Preserved by default** | **Preserved by default** | **Preserved by default** |

### Thinking block preservation in Claude Opus 4.5 and later

Starting with Claude Opus 4.5 (and continuing in Claude Opus 4.6), **thinking blocks from previous assistant turns are preserved in model context by default**. This differs from earlier models, which remove thinking blocks from prior turns.

**Benefits of thinking block preservation:**

- **Cache optimization**: When using tool use, preserved thinking blocks enable cache hits as they are passed back with tool results and cached incrementally across the assistant turn, resulting in token savings in multi-step workflows
- **No intelligence impact**: Preserving thinking blocks has no negative effect on model performance

**Important considerations:**

- **Context usage**: Long conversations will consume more context space since thinking blocks are retained in context
- **Automatic behavior**: This is the default behavior for Claude Opus 4.5 and later models (including Opus 4.6). No code changes or beta headers required
- **Backward compatibility**: To leverage this feature, continue passing complete, unmodified thinking blocks back to the API as you would for tool use

<Note>
For earlier models (Claude Sonnet 4.5, Opus 4.1, etc.), thinking blocks from previous turns continue to be removed from context. The existing behavior described in the [Extended thinking with prompt caching](#extended-thinking-with-prompt-caching) section applies to those models.
</Note>

## Pricing

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

## Best practices and considerations for extended thinking

### Working with thinking budgets

- **Budget optimization:** The minimum budget is 1,024 tokens. Start at the minimum and increase the thinking budget incrementally to find the optimal range for your use case. Higher token counts enable more comprehensive reasoning but with diminishing returns depending on the task. Increasing the budget can improve response quality at the tradeoff of increased latency. For critical tasks, test different settings to find the optimal balance. Note that the thinking budget is a target rather than a strict limit. Actual token usage may vary based on the task.
- **Starting points:** Start with larger thinking budgets (16k+ tokens) for complex tasks and adjust based on your needs.
- **Large budgets:** For thinking budgets above 32k, use [batch processing](/docs/en/build-with-claude/batch-processing) to avoid networking issues. Requests pushing the model to think above 32k tokens causes long running requests that might run up against system timeouts and open connection limits.
- **Token usage tracking:** Monitor thinking token usage to optimize costs and performance.

### Performance considerations

- **Response times:** Be prepared for potentially longer response times due to the additional processing required for the reasoning process. Factor in that generating thinking blocks may increase overall response time.
- **Streaming requirements:** The SDKs require streaming when `max_tokens` is greater than 21,333 to avoid HTTP timeouts on long-running requests. This is a client-side validation, not an API restriction. If you don't need to process events incrementally, use `.stream()` with `.get_final_message()` (Python) or `.finalMessage()` (TypeScript) to get the complete `Message` object without handling individual events. See [Streaming Messages](/docs/en/build-with-claude/streaming#get-the-final-message-without-handling-events) for details. When streaming, be prepared to handle both thinking and text content blocks as they arrive.

### Feature compatibility

- Thinking isn't compatible with `temperature` or `top_k` modifications as well as [forced tool use](/docs/en/agents-and-tools/tool-use/implement-tool-use#forcing-tool-use).
- When thinking is enabled, you can set `top_p` to values between 1 and 0.95.
- You cannot pre-fill responses when thinking is enabled.
- Changes to the thinking budget invalidate cached prompt prefixes that include messages. However, cached system prompts and tool definitions will continue to work when thinking parameters change.

### Usage guidelines

- **Task selection:** Use extended thinking for particularly complex tasks that benefit from step-by-step reasoning like math, coding, and analysis.
- **Context handling:** You do not need to remove previous thinking blocks yourself. The Claude API automatically ignores thinking blocks from previous turns and they are not included when calculating context usage.
- **Prompt engineering:** Review our [extended thinking prompting tips](/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#leverage-thinking-and-interleaved-thinking-capabilities) if you want to maximize Claude's thinking capabilities.

## Next steps

<CardGroup>
  <Card title="Try the extended thinking cookbook" icon="book" href="https://platform.claude.com/cookbook/extended-thinking-extended-thinking">
    Explore practical examples of thinking in the cookbook.
  </Card>
  <Card title="Extended thinking prompting tips" icon="code" href="/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#leverage-thinking-and-interleaved-thinking-capabilities">
    Learn prompt engineering best practices for extended thinking.
  </Card>
</CardGroup>
