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

const AVAILABLE_TEXT_MODELS: string[] = [];
const AVAILABLE_VISION_MODELS: string[] = [];
const OPENROUTER_MODELS: string[] = [];
const SILICONFLOW_MODELS: string[] = [];

if (process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY) {
  AVAILABLE_VISION_MODELS.push("gemini-3-flash-preview");
}

if (process.env.ANTHROPIC_API_KEY) {
  AVAILABLE_VISION_MODELS.push("claude-sonnet-4-5-20250929");
}

if (process.env.OPENAI_API_KEY) {
  AVAILABLE_VISION_MODELS.push("gpt-5.2");
}

if (process.env.GLM_API_KEY) {
  AVAILABLE_TEXT_MODELS.push("glm-4.7");
}

if (process.env.OPENROUTER_API_KEY) {
  OPENROUTER_MODELS.push("z-ai/glm-4.7");
  OPENROUTER_MODELS.push("qwen/qwen3-30b-a3b-thinking-2507");
}

if (process.env.SILICONFLOW_API_KEY) {
  SILICONFLOW_MODELS.push("Pro/zai-org/GLM-4.7");
  SILICONFLOW_MODELS.push("Qwen/Qwen3-8B");
}

const AVAILABLE_MODELS = [
  ...AVAILABLE_VISION_MODELS,
  ...AVAILABLE_TEXT_MODELS,
  ...OPENROUTER_MODELS,
  ...SILICONFLOW_MODELS,
];

function createClient(model: string): AutoLLMClient {
  let apiKey: string | undefined;
  let baseUrl: string | undefined;

  if (OPENROUTER_MODELS.includes(model)) {
    apiKey = process.env.OPENROUTER_API_KEY;
    baseUrl = "https://openrouter.ai/api/v1";
  } else if (SILICONFLOW_MODELS.includes(model)) {
    apiKey = process.env.SILICONFLOW_API_KEY;
    baseUrl = "https://api.siliconflow.cn/v1";
  } else {
    apiKey = undefined;
    baseUrl = undefined;
  }

  return new AutoLLMClient({ model, apiKey, baseUrl });
}

function checkEventIntegrity(event: UniEvent): void {
  expect(event).toHaveProperty("role");
  expect(event).toHaveProperty("event_type");
  expect(event).toHaveProperty("usage_metadata");
  expect(event).toHaveProperty("finish_reason");

  expect(["user", "assistant"]).toContain(event.role);
  expect(["start", "delta", "stop"]).toContain(event.event_type);
  expect(["stop", "length", "unknown", null]).toContain(event.finish_reason);

  for (const item of event.content_items) {
    if (item.type === "text") {
      expect(item).toHaveProperty("text");
    } else if (item.type === "thinking") {
      expect(item).toHaveProperty("thinking");
    } else if (item.type === "tool_call" || item.type === "partial_tool_call") {
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

if (AVAILABLE_MODELS.length > 0) {
  describe.each(AVAILABLE_MODELS)("Client tests for %s", (model) => {
    test("should stream basic response", async () => {
      const client = createClient(model);
      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [{ type: "text", text: "What is 2+3?" }],
        },
      ];
      const config: UniConfig = {};

      let text = "";
      for await (const event of client.streamingResponse({
        messages,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(text).toContain("5");
    }, 60000);

    test("should stream response with all parameters", async () => {
      const client = createClient(model);
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

      if (model.includes("gpt-5.2")) {
        await expect(async () => {
          for await (const _ of client.streamingResponse({
            messages,
            config,
          })) {
            // This should throw before we get here
          }
        }).rejects.toThrow("GPT-5.2 does not support setting temperature.");
      } else {
        let text = "";
        for await (const event of client.streamingResponse({
          messages,
          config,
        })) {
          checkEventIntegrity(event);
          for (const item of event.content_items) {
            if (item.type === "text") {
              text += item.text;
            }
          }
        }

        expect(text).toContain("5");
      }
    }, 60000);

    test("should handle stateful streaming", async () => {
      const client = createClient(model);
      const config: UniConfig = {};

      const message1: UniMessage = {
        role: "user",
        content_items: [{ type: "text", text: "My name is Alice" }],
      };
      for await (const event of client.streamingResponseStateful({
        message: message1,
        config,
      })) {
        checkEventIntegrity(event);
      }

      expect(client.getHistory().length).toBe(2);

      const message2: UniMessage = {
        role: "user",
        content_items: [{ type: "text", text: "What is my name?" }],
      };
      let text = "";
      for await (const event of client.streamingResponseStateful({
        message: message2,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(text.toLowerCase()).toContain("alice");
      expect(client.getHistory().length).toBe(4);
    }, 60000);

    test("should clear history", async () => {
      const client = createClient(model);
      const message: UniMessage = {
        role: "user",
        content_items: [{ type: "text", text: "Hello" }],
      };
      const config: UniConfig = {};

      for await (const _ of client.streamingResponseStateful({
        message,
        config,
      })) {
        // consume the stream
      }

      expect(client.getHistory().length).toBeGreaterThan(0);

      client.clearHistory();
      expect(client.getHistory().length).toBe(0);
    }, 60000);

    test("should concatenate events to message", async () => {
      const client = createClient(model);
      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [{ type: "text", text: "Say hello" }],
        },
      ];
      const config: UniConfig = {};

      const events: UniEvent[] = [];
      let text = "";
      for await (const event of client.streamingResponse({
        messages,
        config,
      })) {
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
    }, 60000);

    test("should handle tool use", async () => {
      const client = createClient(model);

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
      for await (const event of client.streamingResponseStateful({
        message: message1,
        config,
      })) {
        checkEventIntegrity(event);
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
          toolArguments,
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
      for await (const event of client.streamingResponseStateful({
        message: message2,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(text).toContain("20");
    }, 60000);
    test("should handle tool result with image", async () => {
      const client = createClient(model);

      const imageTool = {
        name: "get_image",
        description: "Get an image URL",
        parameters: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "The image to retrieve",
            },
          },
          required: ["query"],
        },
      };

      const config: UniConfig = { tools: [imageTool] };
      let toolCallId: string | undefined;
      let toolName: string | undefined;

      const message1: UniMessage = {
        role: "user",
        content_items: [
          { type: "text", text: "Get me a narcissus flower image" },
        ],
      };
      for await (const event of client.streamingResponseStateful({
        message: message1,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "tool_call") {
            toolName = item.name;
            toolCallId = item.tool_call_id;
          }
        }
      }

      expect(toolName).toBe(imageTool.name);
      expect(toolCallId).toBeDefined();

      const message2: UniMessage = {
        role: "user",
        content_items: [
          {
            type: "tool_result",
            result: [
              { type: "text", text: "Here is a narcissus flower image:" },
              { type: "image_url", image_url: IMAGE },
            ],
            tool_call_id: toolCallId || "",
          },
        ],
      };
      let text = "";
      for await (const event of client.streamingResponseStateful({
        message: message2,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(
        text.toLowerCase().includes("flower") ||
          text.toLowerCase().includes("narcissus"),
      ).toBe(true);
    }, 60000);
    test("should handle system prompt", async () => {
      const client = createClient(model);
      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [{ type: "text", text: "Hello" }],
        },
      ];
      const config: UniConfig = {
        system_prompt: "You are a kitten that must end with the word 'meow'.",
      };

      let text = "";
      for await (const event of client.streamingResponse({
        messages,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(text.toLowerCase()).toContain("meow");
    }, 60000);
  });
}

if (AVAILABLE_VISION_MODELS.length > 0) {
  describe.each(AVAILABLE_VISION_MODELS)("Vision test for %s", (model) => {
    test("should handle image understanding", async () => {
      const client = createClient(model);
      const config: UniConfig = {};
      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [
            {
              type: "text",
              text: "What's in this image? Describe it briefly.",
            },
            { type: "image_url", image_url: IMAGE },
          ],
        },
      ];

      let text = "";
      for await (const event of client.streamingResponse({
        messages,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(
        text.toLowerCase().includes("flower") ||
          text.toLowerCase().includes("narcissus"),
      ).toBe(true);
    }, 60000);

    test("should handle base64 encoded image understanding", async () => {
      const client = createClient(model);
      const config: UniConfig = {};

      // Read test image and encode to base64
      const imagePath = IMAGE;
      const mimeType = "image/jpeg";
      const response = await fetch(imagePath);
      const imageBuffer = await response.arrayBuffer();
      const base64Image = Buffer.from(imageBuffer).toString("base64");

      // Create data URI
      const dataUri = `data:${mimeType};base64,${base64Image}`;

      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [
            {
              type: "text",
              text: "What's in this image? Describe it briefly.",
            },
            { type: "image_url", image_url: dataUri },
          ],
        },
      ];

      let text = "";
      for await (const event of client.streamingResponse({
        messages,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "text") {
            text += item.text;
          }
        }
      }

      expect(
        text.toLowerCase().includes("flower") ||
          text.toLowerCase().includes("narcissus"),
      ).toBe(true);
    }, 60000);
  });
}

test("should reject unknown model", () => {
  expect(() => new AutoLLMClient({ model: "unknown-model" })).toThrow(
    "not supported",
  );
});
