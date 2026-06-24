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

For model IDs, API keys, and base URLs, see [Model selection](reference/models.md).

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

## Reference

- [Model selection](reference/models.md) — model IDs, API keys, base URLs, and OpenAI-compatible routing.
- [Data models](reference/data-models.md) — `UniConfig`, `UniMessage`, `UniEvent`, and the tool-call streaming protocol.
- [APIs](reference/api.md) — client initialization, method signatures, and agent-loop rules.
- [Tracer & Playground](reference/integrations.md) — local tracing UI and the manual chat playground.
