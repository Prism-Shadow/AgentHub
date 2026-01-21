// Copyright 2025 Prism Shadow. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { AutoLLMClient } from "../src/autoClient";
import { ThinkingLevel, UniMessage, UniConfig, UniEvent } from "../src/types";

const IMAGE =
  "https://cdn.britannica.com/80/120980-050-D1DA5C61/Poet-narcissus.jpg";

const AVAILABLE_VISION_MODELS: string[] = [];

if (process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY) {
  AVAILABLE_VISION_MODELS.push("gemini-3-flash-preview");
}

const AVAILABLE_MODELS = AVAILABLE_VISION_MODELS;

async function createClient(model: string): Promise<AutoLLMClient> {
  return new AutoLLMClient(model);
}

async function checkEventIntegrity(event: UniEvent): Promise<void> {
  expect(event).toHaveProperty("role");
  expect(event).toHaveProperty("event_type");
  expect(event).toHaveProperty("usage_metadata");
  expect(event).toHaveProperty("finish_reason");

  for (const item of event.content_items) {
    if (item.type === "text") {
      expect(item).toHaveProperty("text");
    } else if (item.type === "thinking") {
      expect(item).toHaveProperty("thinking");
    } else if (
      item.type === "tool_call" ||
      item.type === "partial_tool_call"
    ) {
      expect(item).toHaveProperty("name");
      expect(item).toHaveProperty("arguments");
      expect(item).toHaveProperty("tool_call_id");
    }
  }

  if (event.usage_metadata) {
    expect(event.usage_metadata).toHaveProperty("prompt_tokens");
    expect(event.usage_metadata).toHaveProperty("thoughts_tokens");
    expect(event.usage_metadata).toHaveProperty("response_tokens");
    expect(event.usage_metadata).toHaveProperty("cached_tokens");
  }
}

describe.each(AVAILABLE_MODELS)("Client tests for %s", (model) => {
  test(
    "should stream basic response",
    async () => {
      const client = await createClient(model);
      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [{ type: "text", text: "What is 2+3?" }],
        },
      ];
      const config: UniConfig = {};

      let text = "";
      for await (const event of client.streamingResponse(
        messages,
        config
      )) {
        await checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(text).toContain("5");
    },
    30000
  );

  test(
    "should stream response with all parameters",
    async () => {
      const client = await createClient(model);
      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [{ type: "text", text: "What is 2+3?" }],
        },
      ];
      const config: UniConfig = {
        max_tokens: 8192,
        temperature: 0.7,
        thinking_summary: true,
        thinking_level: ThinkingLevel.LOW,
      };

      let text = "";
      for await (const event of client.streamingResponse(
        messages,
        config
      )) {
        await checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(text).toContain("5");
    },
    30000
  );

  test(
    "should handle stateful streaming",
    async () => {
      const client = await createClient(model);
      const config: UniConfig = {};

      const message1: UniMessage = {
        role: "user",
        content_items: [{ type: "text", text: "My name is Alice" }],
      };
      for await (const event of client.streamingResponseStateful(
        message1,
        config
      )) {
        await checkEventIntegrity(event);
      }

      expect(client.getHistory().length).toBe(2);

      const message2: UniMessage = {
        role: "user",
        content_items: [{ type: "text", text: "What is my name?" }],
      };
      let text = "";
      for await (const event of client.streamingResponseStateful(
        message2,
        config
      )) {
        await checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(text.toLowerCase()).toContain("alice");
      expect(client.getHistory().length).toBe(4);
    },
    60000
  );

  test(
    "should clear history",
    async () => {
      const client = await createClient(model);
      const message: UniMessage = {
        role: "user",
        content_items: [{ type: "text", text: "Hello" }],
      };
      const config: UniConfig = {};

      for await (const _ of client.streamingResponseStateful(
        message,
        config
      )) {
        // consume the stream
      }

      expect(client.getHistory().length).toBeGreaterThan(0);

      client.clearHistory();
      expect(client.getHistory().length).toBe(0);
    },
    30000
  );

  test(
    "should concatenate events to message",
    async () => {
      const client = await createClient(model);
      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [{ type: "text", text: "Say hello" }],
        },
      ];
      const config: UniConfig = {};

      const events: UniEvent[] = [];
      let text = "";
      for await (const event of client.streamingResponse(
        messages,
        config
      )) {
        events.push(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      const message = client.concatUniEventsToUniMessage(events);
      expect(message.role).toBe("assistant");
      for (const item of message.content_items) {
        if (item.type === "text") {
          expect(item.text).toBe(text);
        }
      }
    },
    30000
  );

  test(
    "should handle tool use",
    async () => {
      const client = await createClient(model);

      const weatherTool = {
        name: "get_weather",
        description: "Get the current weather in a given location",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city name, e.g. San Francisco",
            },
          },
          required: ["location"],
        },
      };

      const config: UniConfig = { tools: [weatherTool] };
      let toolCallId: string | undefined;
      const partialToolCallData: {
        name?: string;
        arguments?: string;
        tool_call_id?: string;
      } = {};
      let toolName: string | undefined;
      let toolArguments: Record<string, unknown> | undefined;

      const message1: UniMessage = {
        role: "user",
        content_items: [
          { type: "text", text: "What is the weather in San Francisco?" },
        ],
      };
      for await (const event of client.streamingResponseStateful(
        message1,
        config
      )) {
        await checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "partial_tool_call") {
            if (!partialToolCallData.name) {
              partialToolCallData.name = item.name;
              partialToolCallData.arguments = item.arguments;
              partialToolCallData.tool_call_id = item.tool_call_id;
            } else {
              partialToolCallData.arguments += item.arguments;
            }
          } else if (item.type === "tool_call") {
            toolName = item.name;
            toolArguments = item.arguments;
            toolCallId = item.tool_call_id;
          }
        }
      }

      expect(toolName).toBe(weatherTool.name);
      expect(toolArguments).toHaveProperty("location");
      expect(toolCallId).toBeDefined();
      expect(partialToolCallData.name).toBe(toolName);
      expect(partialToolCallData.tool_call_id).toBe(toolCallId);
      if (partialToolCallData.arguments && toolArguments) {
        expect(JSON.parse(partialToolCallData.arguments)).toEqual(
          toolArguments
        );
      }

      const message2: UniMessage = {
        role: "user",
        content_items: [
          {
            type: "tool_result",
            result: "It's 20 degrees in San Francisco.",
            tool_call_id: toolCallId || "",
          },
        ],
      };
      let text = "";
      for await (const event of client.streamingResponseStateful(
        message2,
        config
      )) {
        await checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(text).toContain("20");
    },
    60000
  );
});

});

describe.each(AVAILABLE_VISION_MODELS)(
  "Vision model tests for %s",
  (model) => {
    test(
      "should handle image understanding",
      async () => {
        const client = await createClient(model);
        const config: UniConfig = {};
        const messages: UniMessage[] = [
          {
            role: "user",
            content_items: [
              { type: "text", text: "What's in this image?" },
              { type: "image_url", image_url: IMAGE },
            ],
          },
        ];

        let text = "";
        for await (const event of client.streamingResponse(
          messages,
          config
        )) {
          await checkEventIntegrity(event);
          for (const item of event.content_items) {
            if (item.type === "text") {
              text += item.text;
            }
          }
        }

        expect(
          text.toLowerCase().includes("flower") ||
            text.toLowerCase().includes("narcissus")
        ).toBe(true);
      },
      30000
    );
  }
);

describe.each(AVAILABLE_MODELS)("System prompt tests for %s", (model) => {
  test(
    "should handle system prompt",
    async () => {
      const client = await createClient(model);
      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [{ type: "text", text: "Hello" }],
        },
      ];
      const config: UniConfig = {
        system_prompt:
          "You are a kitten that must end with the word 'meow'.",
      };

      let text = "";
      for await (const event of client.streamingResponse(
        messages,
        config
      )) {
        await checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(text.toLowerCase()).toContain("meow");
    },
    30000
  );
});

test("should reject unknown model", () => {
  expect(() => new AutoLLMClient("unknown-model")).toThrow(
    "not supported"
  );
});
