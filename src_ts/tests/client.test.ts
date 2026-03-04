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

import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { AutoLLMClient } from "../src/autoClient";
import { ThinkingLevel, UniMessage, UniConfig, UniEvent } from "../src/types";

const IMAGE =
  "https://cdn.britannica.com/80/120980-050-D1DA5C61/Poet-narcissus.jpg";

interface Model {
  name: string;
  supportVision: boolean;
  supportTemperature: boolean;
  provider: "official" | "siliconflow" | "openrouter" | "bedrock" | "vertex";
}

const AVAILABLE_MODELS: Model[] = [];

if (process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "gemini-3-flash-preview",
    supportVision: true,
    supportTemperature: true,
    provider: "official",
  });
}

if (process.env.ANTHROPIC_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "claude-sonnet-4-6",
    supportVision: true,
    supportTemperature: true,
    provider: "official",
  });
}

if (process.env.OPENAI_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "gpt-5.2",
    supportVision: true,
    supportTemperature: false,
    provider: "official",
  });
}

if (process.env.ZAI_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "glm-5",
    supportVision: false,
    supportTemperature: true,
    provider: "official",
  });
}

if (process.env.MOONSHOT_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "kimi-k2.5",
    supportVision: true,
    supportTemperature: false,
    provider: "official",
  });
}

if (process.env.OPENROUTER_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "z-ai/glm-5",
    supportVision: false,
    supportTemperature: true,
    provider: "openrouter",
  });
  AVAILABLE_MODELS.push({
    name: "qwen/qwen3-30b-a3b-thinking-2507",
    supportVision: false,
    supportTemperature: true,
    provider: "openrouter",
  });
  AVAILABLE_MODELS.push({
    name: "moonshotai/kimi-k2.5",
    supportVision: true,
    supportTemperature: false,
    provider: "openrouter",
  });
}

if (process.env.SILICONFLOW_API_KEY) {
  // AVAILABLE_MODELS.push({ name: "Pro/zai-org/GLM-5", supportVision: false, supportTemperature: true, provider: "siliconflow" });
  AVAILABLE_MODELS.push({
    name: "Qwen/Qwen3-8B",
    supportVision: false,
    supportTemperature: true,
    provider: "siliconflow",
  });
  AVAILABLE_MODELS.push({
    name: "Pro/moonshotai/Kimi-K2.5",
    supportVision: true,
    supportTemperature: false,
    provider: "siliconflow",
  });
}

if (process.env.BEDROCK_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "global.anthropic.claude-sonnet-4-6",
    supportVision: true,
    supportTemperature: true,
    provider: "bedrock",
  });
}

if (process.env.VERTEX_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "gemini-3-flash-preview",
    supportVision: true,
    supportTemperature: true,
    provider: "vertex",
  });
}

function createClient(model: Model): AutoLLMClient {
  let apiKey: string | undefined;
  let baseUrl: string | undefined;

  if (model.provider === "vertex") {
    const vertexKey = process.env.VERTEX_API_KEY;
    if (vertexKey) {
      const tmpFile = path.join(os.tmpdir(), `vertex_key_${Date.now()}.json`);
      fs.writeFileSync(tmpFile, vertexKey);
      try {
        apiKey = tmpFile;
        return new AutoLLMClient({ model: model.name, apiKey, baseUrl });
      } finally {
        fs.rmSync(tmpFile, { force: true });
      }
    }
    return new AutoLLMClient({ model: model.name, apiKey, baseUrl });
  } else if (model.provider === "openrouter") {
    apiKey = process.env.OPENROUTER_API_KEY;
    baseUrl = "https://openrouter.ai/api/v1";
  } else if (model.provider === "siliconflow") {
    apiKey = process.env.SILICONFLOW_API_KEY;
    baseUrl = "https://api.siliconflow.cn/v1";
  } else if (model.provider === "bedrock") {
    apiKey = process.env.BEDROCK_API_KEY;
    baseUrl = "bedrock://us-east-1";
  } else {
    apiKey = undefined;
    baseUrl = undefined;
  }

  return new AutoLLMClient({ model: model.name, apiKey, baseUrl });
}

function checkEventIntegrity(event: UniEvent): void {
  expect(event).toHaveProperty("role");
  expect(event).toHaveProperty("event_type");
  expect(event).toHaveProperty("usage_metadata");
  expect(event).toHaveProperty("finish_reason");

  expect(["user", "assistant"]).toContain(event.role);
  expect(["start", "delta", "stop"]).toContain(event.event_type);
  expect(["stop", "length", "tool_call", "unknown", null]).toContain(
    event.finish_reason,
  );

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
    expect(event.usage_metadata).toHaveProperty("cached_tokens");
    expect(event.usage_metadata).toHaveProperty("prompt_tokens");
    expect(event.usage_metadata).toHaveProperty("thoughts_tokens");
    expect(event.usage_metadata).toHaveProperty("response_tokens");

    if (event.usage_metadata.cached_tokens !== null) {
      expect(event.usage_metadata.cached_tokens).toBeGreaterThanOrEqual(0);
    }
    if (event.usage_metadata.prompt_tokens !== null) {
      expect(event.usage_metadata.prompt_tokens).toBeGreaterThanOrEqual(0);
    }
    if (event.usage_metadata.thoughts_tokens !== null) {
      expect(event.usage_metadata.thoughts_tokens).toBeGreaterThanOrEqual(0);
    }
    if (event.usage_metadata.response_tokens !== null) {
      expect(event.usage_metadata.response_tokens).toBeGreaterThanOrEqual(0);
    }
  }
}

if (AVAILABLE_MODELS.length > 0) {
  describe.each(AVAILABLE_MODELS.map((m) => [m.name, m]))(
    "Client tests for %s",
    (_name, model: Model) => {
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

        if (!model.supportTemperature) {
          await expect(async () => {
            for await (const _ of client.streamingResponse({
              messages,
              config,
            })) {
              // This should throw before we get here
            }
          }).rejects.toThrow("not support");
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
            content_items: [
              {
                type: "text",
                text: "Say 'The quick brown fox jumps over the lazy dog.'",
              },
            ],
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
              text: "It's 20 degrees in San Francisco.",
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

      test("should handle image understanding", async () => {
        if (!model.supportVision) {
          return;
        }

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
        if (!model.supportVision) {
          return;
        }

        const client = createClient(model);
        const config: UniConfig = {};

        const mimeType = "image/jpeg";
        const response = await fetch(IMAGE);
        const imageBuffer = await response.arrayBuffer();
        const base64Image = Buffer.from(imageBuffer).toString("base64");

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

      test("should handle tool result with image", async () => {
        if (!model.supportVision) {
          return;
        }

        const client = createClient(model);

        const imageTool = {
          name: "get_image",
          description: "Get an image URL",
          parameters: {
            type: "object",
            properties: {
              seed: {
                type: "integer",
                description: "The random seed to retrieve the image.",
              },
            },
            required: ["seed"],
          },
        };

        const config: UniConfig = { tools: [imageTool] };
        let toolCallId: string | undefined;
        let toolName: string | undefined;

        const message1: UniMessage = {
          role: "user",
          content_items: [
            {
              type: "text",
              text: "Get me a random image and describe it briefly.",
            },
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
              text: "Here is the result image:",
              images: [IMAGE],
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
    },
  );
}

test("should reject unknown model", () => {
  expect(() => new AutoLLMClient({ model: "unknown-model" })).toThrow(
    "not supported",
  );
});

test("should validate last event has usage_metadata and finish_reason", () => {
  const { LLMClient } = require("../src/baseClient");

  const validEvent = {
    role: "assistant",
    event_type: "stop",
    content_items: [],
    usage_metadata: {
      cached_tokens: 0,
      prompt_tokens: 10,
      thoughts_tokens: null,
      response_tokens: 5,
    },
    finish_reason: "stop",
  };

  // should not throw
  expect(() => LLMClient._validateLastEvent(validEvent)).not.toThrow();

  // null event
  expect(() => LLMClient._validateLastEvent(null)).toThrow("no events");

  // missing usage_metadata
  expect(() =>
    LLMClient._validateLastEvent({ ...validEvent, usage_metadata: null }),
  ).toThrow("usage_metadata");

  // missing finish_reason
  expect(() =>
    LLMClient._validateLastEvent({ ...validEvent, finish_reason: null }),
  ).toThrow("finish_reason");
});
