# Streaming Messages

---

When creating a Message, you can set `"stream": true` to incrementally stream the response using [server-sent events](https://developer.mozilla.org/en-US/Web/API/Server-sent%5Fevents/Using%5Fserver-sent%5Fevents) (SSE).

## Streaming with SDKs

The [Python](https://github.com/anthropics/anthropic-sdk-python) and [TypeScript](https://github.com/anthropics/anthropic-sdk-typescript) SDKs offer multiple ways of streaming. The Python SDK allows both sync and async streams. See the documentation in each SDK for details.

<CodeGroup>
    ```python Python
    import anthropic

    client = anthropic.Anthropic()

    with client.messages.stream(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        model="claude-opus-4-6",
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    ```

    ```typescript TypeScript hidelines={1..4}
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    await client.messages
      .stream({
        messages: [{ role: "user", content: "Hello" }],
        model: "claude-opus-4-6",
        max_tokens: 1024
      })
      .on("text", (text) => {
        console.log(text);
      });
    ```

    ```csharp C#
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
                MaxTokens = 1024,
                Messages = [new() { Role = Role.User, Content = "Hello" }]
            };

            await foreach (var msg in client.Messages.CreateStreaming(parameters))
            {
                Console.Write(msg);
            }
        }
    }
    ```
</CodeGroup>

## Get the final message without handling events

If you don't need to process text as it arrives, the SDKs provide a way to use streaming under the hood while returning the complete `Message` object, identical to what `.create()` returns. This is especially useful for requests with large `max_tokens` values, where the SDKs require streaming to avoid HTTP timeouts.

<CodeGroup>
    ```python Python hidelines={1..4,-1}
    import anthropic

    client = anthropic.Anthropic()

    with client.messages.stream(
        max_tokens=128000,
        messages=[{"role": "user", "content": "Write a detailed analysis..."}],
        model="claude-opus-4-6",
    ) as stream:
        message = stream.get_final_message()

    print(message.content[0].text)
    ```

    ```typescript TypeScript hidelines={1..4}
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    const stream = client.messages.stream({
      max_tokens: 128000,
      messages: [{ role: "user", content: "Write a detailed analysis..." }],
      model: "claude-opus-4-6"
    });

    const message = await stream.finalMessage();
    const textBlock = message.content.find((block) => block.type === "text");
    if (textBlock && textBlock.type === "text") {
      console.log(textBlock.text);
    }
    ```


    ```csharp C# nocheck
    using System;
    using System.Threading.Tasks;
    using Anthropic;
    using Anthropic.Models.Messages;

    class Program
    {
        static async Task Main()
        {
            AnthropicClient client = new();

            var parameters = new MessageCreateParams
            {
                Model = Model.ClaudeOpus4_6,
                MaxTokens = 128000,
                Messages = [new() { Role = Role.User, Content = "Write a detailed analysis..." }]
            };

            var fullText = "";
            await foreach (var msg in client.Messages.CreateStreaming(parameters))
            {
                fullText += msg;
            }

            Console.WriteLine(fullText);
        }
    }
    ```
</CodeGroup>

The `.stream()` call keeps the HTTP connection alive with server-sent events, then `.get_final_message()` (Python) or `.finalMessage()` (TypeScript) accumulates all events and returns the complete `Message` object. No event handling code is needed.

## Event types

Each server-sent event includes a named event type and associated JSON data. Each event uses an SSE event name (e.g. `event: message_stop`), and includes the matching event `type` in its data.

Each stream uses the following event flow:

1. `message_start`: contains a `Message` object with empty `content`.
2. A series of content blocks, each of which have a `content_block_start`, one or more `content_block_delta` events, and a `content_block_stop` event. Each content block has an `index` that corresponds to its index in the final Message `content` array.
3. One or more `message_delta` events, indicating top-level changes to the final `Message` object.
4. A final `message_stop` event.

  <Warning>
  The token counts shown in the `usage` field of the `message_delta` event are *cumulative*.
  </Warning>

### Ping events

Event streams may also include any number of `ping` events.

### Error events

The API may occasionally send [errors](/docs/en/api/errors) in the event stream. For example, during periods of high usage, you may receive an `overloaded_error`, which would normally correspond to an HTTP 529 in a non-streaming context:

```sse Example error
event: error
data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
```

### Other events

In accordance with the [versioning policy](/docs/en/api/versioning), new event types may be added, and your code should handle unknown event types gracefully.

## Content block delta types

Each `content_block_delta` event contains a `delta` of a type that updates the `content` block at a given `index`.

### Text delta

A `text` content block delta looks like:
```sse Text delta
event: content_block_delta
data: {"type": "content_block_delta","index": 0,"delta": {"type": "text_delta", "text": "ello frien"}}
```

### Input JSON delta

The deltas for `tool_use` content blocks correspond to updates for the `input` field of the block. To support maximum granularity, the deltas are _partial JSON strings_, whereas the final `tool_use.input` is always an _object_.

You can accumulate the string deltas and parse the JSON once you receive a `content_block_stop` event, by using a library like [Pydantic](https://docs.pydantic.dev/latest/concepts/json/#partial-json-parsing) to do partial JSON parsing, or by using the [SDKs](/docs/en/api/client-sdks), which provide helpers to access parsed incremental values.

A `tool_use` content block delta looks like:
```sse Input JSON delta
event: content_block_delta
data: {"type": "content_block_delta","index": 1,"delta": {"type": "input_json_delta","partial_json": "{\"location\": \"San Fra"}}}
```
Note: Current models only support emitting one complete key and value property from `input` at a time. As such, when using tools, there may be delays between streaming events while the model is working. Once an `input` key and value are accumulated, they are emitted as multiple `content_block_delta` events with chunked partial json so that the format can automatically support finer granularity in future models.

### Thinking delta

When using [extended thinking](/docs/en/build-with-claude/extended-thinking#streaming-thinking) with streaming enabled, you'll receive thinking content via `thinking_delta` events. These deltas correspond to the `thinking` field of the `thinking` content blocks.

For thinking content, a special `signature_delta` event is sent just before the `content_block_stop` event. This signature is used to verify the integrity of the thinking block.

A typical thinking delta looks like:
```sse Thinking delta
event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "I need to find the GCD of 1071 and 462 using the Euclidean algorithm.\n\n1071 = 2 × 462 + 147"}}
```

The signature delta looks like:
```sse Signature delta
event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."}}
```

## Full HTTP Stream response

Use the [client SDKs](/docs/en/api/client-sdks) when using streaming mode. However, if you are building a direct API integration, you need to handle these events yourself.

A stream response is comprised of:
1. A `message_start` event
2. Potentially multiple content blocks, each of which contains:
    - A `content_block_start` event
    - Potentially multiple `content_block_delta` events
    - A `content_block_stop` event
3. A `message_delta` event
4. A `message_stop` event

There may be `ping` events dispersed throughout the response as well. See [Event types](#event-types) for more details on the format.

### Basic streaming request

<CodeGroup>
```bash Shell
curl https://api.anthropic.com/v1/messages \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --data \
'{
  "model": "claude-opus-4-6",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 256,
  "stream": true
}'
```

```python Python hidelines={1..4}
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-opus-4-6",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=256,
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

```typescript TypeScript hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const stream = client.messages.stream({
  model: "claude-opus-4-6",
  messages: [{ role: "user", content: "Hello" }],
  max_tokens: 256
});

for await (const event of stream) {
  if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
    process.stdout.write(event.delta.text);
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
            MaxTokens = 256,
            Messages = [new() { Role = Role.User, Content = "Hello" }]
        };

        await foreach (var msg in client.Messages.CreateStreaming(parameters))
        {
            Console.Write(msg);
        }
    }
}
```
</CodeGroup>

```sse Response
event: message_start
data: {"type": "message_start", "message": {"id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-4-6", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 25, "output_tokens": 1}}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence":null}, "usage": {"output_tokens": 15}}

event: message_stop
data: {"type": "message_stop"}

```

### Streaming request with tool use

<Tip>
Tool use supports [fine-grained streaming](/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming) for parameter values. Enable it per tool with `eager_input_streaming`.
</Tip>

This request asks Claude to use a tool to report the weather.

<CodeGroup>
```bash Shell
  curl https://api.anthropic.com/v1/messages \
    -H "content-type: application/json" \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -d '{
      "model": "claude-opus-4-6",
      "max_tokens": 1024,
      "tools": [
        {
          "name": "get_weather",
          "description": "Get the current weather in a given location",
          "input_schema": {
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
      "tool_choice": {"type": "any"},
      "messages": [
        {
          "role": "user",
          "content": "What is the weather like in San Francisco?"
        }
      ],
      "stream": true
    }'
```

```python Python hidelines={1..4}
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    }
]

with client.messages.stream(
    model="claude-opus-4-6",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "any"},
    messages=[
        {"role": "user", "content": "What is the weather like in San Francisco?"}
    ],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

```typescript TypeScript hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const tools: Anthropic.Tool[] = [
  {
    name: "get_weather",
    description: "Get the current weather in a given location",
    input_schema: {
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "The city and state, e.g. San Francisco, CA"
        }
      },
      required: ["location"]
    }
  }
];

const stream = client.messages.stream({
  model: "claude-opus-4-6",
  max_tokens: 1024,
  tools: tools,
  tool_choice: { type: "any" },
  messages: [
    {
      role: "user",
      content: "What is the weather like in San Francisco?"
    }
  ]
});

for await (const event of stream) {
  if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
    process.stdout.write(event.delta.text);
  }
}
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

        var parameters = new MessageCreateParams
        {
            Model = Model.ClaudeOpus4_6,
            MaxTokens = 1024,
            Tools = [
                new ToolUnion(new Tool()
                {
                    Name = "get_weather",
                    Description = "Get the current weather in a given location",
                    InputSchema = new InputSchema()
                    {
                        Properties = new Dictionary<string, JsonElement>
                        {
                            ["location"] = JsonSerializer.SerializeToElement(new { type = "string", description = "The city and state, e.g. San Francisco, CA" }),
                        },
                        Required = ["location"],
                    },
                }),
            ],
            ToolChoice = new ToolChoiceAny(),
            Messages = [
                new() { Role = Role.User, Content = "What is the weather like in San Francisco?" }
            ]
        };

        await foreach (var msg in client.Messages.CreateStreaming(parameters))
        {
            Console.Write(msg);
        }
    }
}
```
</CodeGroup>

```sse Response
event: message_start
data: {"type":"message_start","message":{"id":"msg_014p7gG3wDgGV9EUtLvnow3U","type":"message","role":"assistant","model":"claude-opus-4-6","stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2},"content":[],"stop_reason":null}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Okay"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":","}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" let"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"'s"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" check"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" the"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" weather"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" for"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" San"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Francisco"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":","}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" CA"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":":"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \"San"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" Francisc"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"o,"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" CA\""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":", "}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"unit\": \"fah"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"renheit\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}

event: message_stop
data: {"type":"message_stop"}
```

### Streaming request with extended thinking

This request enables extended thinking with streaming to see Claude's step-by-step reasoning.

<CodeGroup>
```bash Shell
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-opus-4-6",
    "max_tokens": 20000,
    "stream": true,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 16000
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
    model="claude-opus-4-6",
    max_tokens=20000,
    thinking={"type": "enabled", "budget_tokens": 16000},
    messages=[
        {
            "role": "user",
            "content": "What is the greatest common divisor of 1071 and 462?",
        }
    ],
) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                print(event.delta.thinking, end="", flush=True)
            elif event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
```

```typescript TypeScript hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const stream = client.messages.stream({
  model: "claude-opus-4-6",
  max_tokens: 20000,
  thinking: { type: "enabled", budget_tokens: 16000 },
  messages: [
    {
      role: "user",
      content: "What is the greatest common divisor of 1071 and 462?"
    }
  ]
});

for await (const event of stream) {
  if (event.type === "content_block_delta") {
    if (event.delta.type === "thinking_delta") {
      process.stdout.write(event.delta.thinking);
    } else if (event.delta.type === "text_delta") {
      process.stdout.write(event.delta.text);
    }
  }
}
```

```csharp C#
using Anthropic;
using Anthropic.Models.Messages;

AnthropicClient client = new();

var parameters = new MessageCreateParams
{
    Model = Model.ClaudeOpus4_6,
    MaxTokens = 20000,
    Thinking = new ThinkingConfigEnabled(budgetTokens: 16000),
    Messages = [new() { Role = Role.User, Content = "What is the greatest common divisor of 1071 and 462?" }]
};

await foreach (var msg in client.Messages.CreateStreaming(parameters))
{
    Console.Write(msg);
}
```
</CodeGroup>

```sse Response
event: message_start
data: {"type": "message_start", "message": {"id": "msg_01...", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-4-6", "stop_reason": null, "stop_sequence": null}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "I need to find the GCD of 1071 and 462 using the Euclidean algorithm.\n\n1071 = 2 × 462 + 147"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "\n462 = 3 × 147 + 21"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "\n147 = 7 × 21 + 0"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "\nThe remainder is 0, so GCD(1071, 462) = 21."}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "The greatest common divisor of 1071 and 462 is **21**."}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 1}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}}

event: message_stop
data: {"type": "message_stop"}
```

### Streaming request with web search tool use

This request asks Claude to search the web for current weather information.

<CodeGroup>
```bash Shell
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-opus-4-6",
    "max_tokens": 1024,
    "stream": true,
    "tools": [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5
        }
    ],
    "messages": [
        {
            "role": "user",
            "content": "What is the weather like in New York City today?"
        }
    ]
}'
```

```python Python hidelines={1..4}
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-opus-4-6",
    max_tokens=1024,
    tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
    messages=[
        {"role": "user", "content": "What is the weather like in New York City today?"}
    ],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

```typescript TypeScript hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const stream = client.messages.stream({
  model: "claude-opus-4-6",
  max_tokens: 1024,
  tools: [{ type: "web_search_20250305", name: "web_search", max_uses: 5 }],
  messages: [{ role: "user", content: "What is the weather like in New York City today?" }]
});

for await (const event of stream) {
  if (event.type === "content_block_delta" && event.delta.type === "text_delta") {
    process.stdout.write(event.delta.text);
  }
}
```

```csharp C#
using Anthropic;
using Anthropic.Models.Messages;

AnthropicClient client = new();

var parameters = new MessageCreateParams
{
    Model = Model.ClaudeOpus4_6,
    MaxTokens = 1024,
    Tools = [new ToolUnion(new WebSearchTool20250305() { MaxUses = 5 })],
    Messages = [new() { Role = Role.User, Content = "What is the weather like in New York City today?" }]
};

await foreach (var msg in client.Messages.CreateStreaming(parameters))
{
    Console.Write(msg);
}
```

```go Go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/anthropics/anthropic-sdk-go"
)

func main() {
	client := anthropic.NewClient()

	stream := client.Messages.NewStreaming(context.TODO(), anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeOpus4_6,
		MaxTokens: 1024,
		Tools: []anthropic.ToolParam{
			anthropic.NewWebSearchTool(anthropic.WebSearchToolParam{
				Type:    anthropic.F(anthropic.ToolTypeWebSearch20250305),
				Name:    anthropic.F("web_search"),
				MaxUses: anthropic.Int(5),
			}),
		},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("What is the weather like in New York City today?")),
		},
	})

	for stream.Next() {
		event := stream.Current()
		switch eventVariant := event.AsAny().(type) {
		case anthropic.ContentBlockDeltaEvent:
			switch deltaVariant := eventVariant.Delta.AsAny().(type) {
			case anthropic.TextDelta:
				fmt.Print(deltaVariant.Text)
			}
		}
	}
	if err := stream.Err(); err != nil {
		log.Fatal(err)
	}
}
```

```java Java
import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.Model;
import com.anthropic.models.messages.WebSearchTool20250305;

public class WebSearchStreaming {
    public static void main(String[] args) {
        AnthropicClient client = AnthropicOkHttpClient.fromEnv();

        MessageCreateParams params = MessageCreateParams.builder()
            .model(Model.CLAUDE_OPUS_4_6)
            .maxTokens(1024L)
            .addTool(WebSearchTool20250305.builder()
                .maxUses(5L)
                .build())
            .addUserMessage("What is the weather like in New York City today?")
            .build();

        try (var streamResponse = client.messages().createStreaming(params)) {
            streamResponse.stream().forEach(event -> {
                event.contentBlockDelta().ifPresent(deltaEvent ->
                    deltaEvent.delta().text().ifPresent(td ->
                        System.out.print(td.text())
                    )
                );
            });
        }
    }
}
```

```ruby Ruby
require "anthropic"

client = Anthropic::Client.new

stream = client.messages.stream(
  model: :"claude-opus-4-6",
  max_tokens: 1024,
  tools: [
    {
      type: "web_search_20250305",
      name: "web_search",
      max_uses: 5
    }
  ],
  messages: [
    {
      role: "user",
      content: "What is the weather like in New York City today?"
    }
  ]
)

stream.text.each { |text| print(text) }
```
</CodeGroup>

```sse Response
event: message_start
data: {"type":"message_start","message":{"id":"msg_01G...","type":"message","role":"assistant","model":"claude-opus-4-6","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":2679,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":3}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I'll check"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" the current weather in New York City for you"}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srvtoolu_014hJH82Qum7Td6UV8gDXThB","name":"web_search","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" \"weather"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" NY"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"C to"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"day\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1 }

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_014hJH82Qum7Td6UV8gDXThB","content":[{"type":"web_search_result","title":"Weather in New York City in May 2025 (New York) - detailed Weather Forecast for a month","url":"https://world-weather.info/forecast/usa/new_york/may-2025/","encrypted_content":"Ev0DCioIAxgCIiQ3NmU4ZmI4OC1k...","page_age":null},...]}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: content_block_start
data: {"type":"content_block_start","index":3,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"Here's the current weather information for New York"}}

event: content_block_delta
data: {"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":" City:\n\n# Weather"}}

event: content_block_delta
data: {"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":" in New York City"}}

event: content_block_delta
data: {"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"\n\n"}}

...

event: content_block_stop
data: {"type":"content_block_stop","index":17}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":10682,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":510,"server_tool_use":{"web_search_requests":1}}}

event: message_stop
data: {"type":"message_stop"}
```

## Error recovery

### Claude 4.5 and earlier

For Claude 4.5 models and earlier, you can recover a streaming request that was interrupted due to network issues, timeouts, or other errors by resuming from where the stream was interrupted. This approach saves you from re-processing the entire response.

The basic recovery strategy involves:

1. **Capture the partial response**: Save all content that was successfully received before the error occurred
2. **Construct a continuation request**: Create a new API request that includes the partial assistant response as the beginning of a new assistant message
3. **Resume streaming**: Continue receiving the rest of the response from where it was interrupted

### Claude 4.6

For Claude 4.6 models, you should add a user message that instructs the model to continue from where it left off. For example:

```text Sample prompt
Your previous response was interrupted and ended with [previous_response]. Continue from where you left off.
```

### Error recovery best practices

1. **Use SDK features**: Leverage the SDK's built-in message accumulation and error handling capabilities
2. **Handle content types**: Be aware that messages can contain multiple content blocks (`text`, `tool_use`, `thinking`). Tool use and extended thinking blocks cannot be partially recovered. You can resume streaming from the most recent text block.
