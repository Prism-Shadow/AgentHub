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
import { listSupportedModels } from "../src/registry";
import {
  PromptCaching,
  ThinkingLevel,
  UniMessage,
  UniConfig,
  UniEvent,
} from "../src/types";
import { expect, describe, test } from "@jest/globals";

const IMAGE =
  "https://cdn.britannica.com/80/120980-050-D1DA5C61/Poet-narcissus.jpg";
const IMAGE_KEYWORDS = ["flower", "narcissus", "daffodil", "bloom"];

interface Model {
  name: string;
  supportTextGeneration: boolean;
  supportTemperature: boolean;
  supportImageUnderstanding: boolean;
  supportImageGeneration: boolean;
  supportAudioGeneration: boolean;
  supportEmbedding: boolean;
  clientType?: string;
  provider:
    | "official"
    | "bedrock"
    | "vertex"
    | "siliconflow"
    | "openrouter"
    | "modelverse";
}

const AVAILABLE_MODELS: Model[] = [];

if (process.env.GEMINI_API_KEY) {

  AVAILABLE_MODELS.push({
    name: "gemini-3.6-flash",
    supportTextGeneration: true,
    supportTemperature: false,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "official",
  });


  AVAILABLE_MODELS.push({
    name: "gemini-3.1-flash-image",
    supportTextGeneration: false,
    supportTemperature: false,
    supportImageUnderstanding: false,
    supportImageGeneration: true,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "official",
  });

  AVAILABLE_MODELS.push({
    name: "gemini-3.1-flash-tts-preview",
    supportTextGeneration: false,
    supportTemperature: false,
    supportImageUnderstanding: false,
    supportImageGeneration: false,
    supportAudioGeneration: true,
    supportEmbedding: false,
    provider: "official",
  });

  AVAILABLE_MODELS.push({
    name: "gemini-embedding-2",
    supportTextGeneration: false,
    supportTemperature: false,
    supportImageUnderstanding: false,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: true,
    provider: "official",
  });
}

if (process.env.ANTHROPIC_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "claude-sonnet-4-6",
    supportTextGeneration: true,
    supportTemperature: true,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "official",
  });
}

if (process.env.OPENAI_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "gpt-5.5",
    supportTextGeneration: true,
    supportTemperature: false,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "official",
  });

  AVAILABLE_MODELS.push({
    name: "text-embedding-3-large",
    supportTextGeneration: false,
    supportTemperature: false,
    supportImageUnderstanding: false,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: true,
    clientType: "openai-embedding",
    provider: "official",
  });
}

if (process.env.ZAI_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "glm-5.2",
    supportTextGeneration: true,
    supportTemperature: true,
    supportImageUnderstanding: false,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "official",
  });
}

if (process.env.MOONSHOT_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "kimi-k3",
    supportTextGeneration: true,
    supportTemperature: false,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "official",
  });
}

if (process.env.MINIMAX_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "MiniMax-M3",
    supportTextGeneration: true,
    supportTemperature: true,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    clientType: "minimax-m3",
    provider: "official",
  });
}

if (process.env.DEEPSEEK_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "deepseek-v4-flash",
    supportTextGeneration: true,
    supportTemperature: false,
    supportImageUnderstanding: false,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "official",
  });
}

if (process.env.BEDROCK_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "global.anthropic.claude-sonnet-4-6",
    supportTextGeneration: true,
    supportTemperature: true,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "bedrock",
  });
}

if (process.env.VERTEX_API_KEY) {
  AVAILABLE_MODELS.push({
    name: "gemini-3.6-flash",
    supportTextGeneration: true,
    supportTemperature: false,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "vertex",
  });

  AVAILABLE_MODELS.push({
    name: "gemini-3.1-flash-image",
    supportTextGeneration: false,
    supportTemperature: false,
    supportImageUnderstanding: false,
    supportImageGeneration: true,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "vertex",
  });

  AVAILABLE_MODELS.push({
    name: "gemini-3.1-flash-tts-preview",
    supportTextGeneration: false,
    supportTemperature: false,
    supportImageUnderstanding: false,
    supportImageGeneration: false,
    supportAudioGeneration: true,
    supportEmbedding: false,
    provider: "vertex",
  });
}

const RUN_SLOW_TEST = process.env.RUN_SLOW_TEST === "1";

if (process.env.OPENROUTER_API_KEY && RUN_SLOW_TEST) {
  AVAILABLE_MODELS.push({
    name: "z-ai/glm-5.2",
    supportTextGeneration: true,
    supportTemperature: true,
    supportImageUnderstanding: false,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "openrouter",
  });
  AVAILABLE_MODELS.push({
    name: "qwen/qwen3.6-35b-a3b",
    supportTextGeneration: true,
    supportTemperature: true,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    clientType: "openai",
    provider: "openrouter",
  });
  AVAILABLE_MODELS.push({
    name: "qwen/qwen3-embedding-4b",
    supportTextGeneration: false,
    supportTemperature: false,
    supportImageUnderstanding: false,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: true,
    clientType: "openai-embedding",
    provider: "openrouter",
  });
  AVAILABLE_MODELS.push({
    name: "moonshotai/kimi-k3",
    supportTextGeneration: true,
    supportTemperature: false,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "openrouter",
  });
}

if (process.env.SILICONFLOW_API_KEY && RUN_SLOW_TEST) {
  AVAILABLE_MODELS.push({
    name: "zai-org/GLM-5.2",
    supportTextGeneration: true,
    supportTemperature: true,
    supportImageUnderstanding: false,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "siliconflow",
  });
  AVAILABLE_MODELS.push({
    name: "Qwen/Qwen3.6-35B-A3B",
    supportTextGeneration: true,
    supportTemperature: true,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    clientType: "openai",
    provider: "siliconflow",
  });
  AVAILABLE_MODELS.push({
    name: "Pro/moonshotai/Kimi-K2.6",
    supportTextGeneration: true,
    supportTemperature: false,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "siliconflow",
  });
  AVAILABLE_MODELS.push({
    name: "Qwen/Qwen3-Embedding-8B",
    supportTextGeneration: false,
    supportTemperature: false,
    supportImageUnderstanding: false,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: true,
    clientType: "openai-embedding",
    provider: "siliconflow",
  });
}

if (process.env.MODELVERSE_API_KEY && RUN_SLOW_TEST) {
  AVAILABLE_MODELS.push({
    name: "claude-sonnet-4-6",
    supportTextGeneration: true,
    supportTemperature: true,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "modelverse",
  });
  AVAILABLE_MODELS.push({
    name: "gpt-5.5",
    supportTextGeneration: true,
    supportTemperature: false,
    supportImageUnderstanding: true,
    supportImageGeneration: false,
    supportAudioGeneration: false,
    supportEmbedding: false,
    provider: "modelverse",
  });
}

function createClient(model: Model): AutoLLMClient {
  let apiKey: string | undefined;
  let baseUrl: string | undefined;

  if (model.provider === "bedrock") {
    apiKey = process.env.BEDROCK_API_KEY;
    baseUrl = "bedrock://us-east-1";
  } else if (model.provider === "vertex") {
    apiKey = process.env.VERTEX_API_KEY;
    baseUrl = undefined;
  } else if (model.provider === "openrouter") {
    apiKey = process.env.OPENROUTER_API_KEY;
    baseUrl = "https://openrouter.ai/api/v1";
  } else if (model.provider === "siliconflow") {
    apiKey = process.env.SILICONFLOW_API_KEY;
    baseUrl = "https://api.siliconflow.cn/v1";
  } else if (model.provider === "modelverse") {
    apiKey = process.env.MODELVERSE_API_KEY;
    if (model.name.startsWith("claude-")) {
      baseUrl = "https://api.modelverse.cn/";
    } else {
      baseUrl = "https://api.modelverse.cn/v1";
    }
  } else {
    apiKey = undefined;
    baseUrl = undefined;
  }

  return new AutoLLMClient({
    model: model.name,
    apiKey,
    baseUrl,
    clientType: model.clientType,
  });
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
  expect(event).toHaveProperty("created_at");
  expect(event.created_at).toBeGreaterThan(0);

  for (const item of event.content_items) {
    if (item.type === "text") {
      expect(typeof item.text).toBe("string");
    } else if (item.type === "image_url") {
      expect(typeof item.image_url).toBe("string");
    } else if (item.type === "inline_data") {
      expect(Buffer.isBuffer(item.data)).toBe(true);
      expect(item.data.length).toBeGreaterThan(0);
      expect(typeof item.mime_type).toBe("string");
    } else if (item.type === "thinking") {
      expect(typeof item.thinking).toBe("string");
    } else if (item.type === "inline_thinking") {
      expect(Buffer.isBuffer(item.data)).toBe(true);
      expect(item.data.length).toBeGreaterThan(0);
      expect(typeof item.mime_type).toBe("string");
    } else if (item.type === "tool_call") {
      expect(typeof item.name).toBe("string");
      expect(typeof item.arguments).toBe("object");
      expect(typeof item.tool_call_id).toBe("string");
    } else if (item.type === "partial_tool_call") {
      expect(typeof item.name).toBe("string");
      expect(typeof item.arguments).toBe("string");
      expect(typeof item.tool_call_id).toBe("string");
    } else if (item.type === "tool_result") {
      expect(typeof item.text).toBe("string");
      expect(typeof item.tool_call_id).toBe("string");
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
  describe.each(
    AVAILABLE_MODELS.map((m): [string, Model] => [
      `${m.name}:${m.provider}`,
      m,
    ]),
  )("Client tests for %s", (_name, model: Model) => {
    test("should stream basic response", async () => {
      if (!model.supportTextGeneration) {
        return;
      }
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
      if (!model.supportTextGeneration) {
        return;
      }
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
      if (!model.supportTextGeneration) {
        return;
      }
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

    test("should set history", () => {
      const client = createClient(model);
      const newHistory: UniMessage[] = [
        { role: "user", content_items: [{ type: "text", text: "Hi" }] },
        {
          role: "assistant",
          content_items: [{ type: "text", text: "Hello!" }],
        },
      ];

      client.setHistory(newHistory);
      expect(client.getHistory()).toEqual(newHistory);

      // Mutating the original array must not affect the stored history
      newHistory.splice(0);
      expect(client.getHistory().length).toBe(2);
    });

    test("should clear history", async () => {
      const client = createClient(model);
      const newHistory: UniMessage[] = [
        { role: "user", content_items: [{ type: "text", text: "Hi" }] },
        {
          role: "assistant",
          content_items: [{ type: "text", text: "Hello!" }],
        },
      ];

      client.setHistory(newHistory);
      expect(client.getHistory()).toEqual(newHistory);

      client.clearHistory();
      expect(client.getHistory().length).toBe(0);
    });

    test("should concatenate events to message", async () => {
      if (!model.supportTextGeneration) {
        return;
      }

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
      const allText = message.content_items
        .filter((item) => item.type === "text")
        .map((item) => (item as { type: "text"; text: string }).text)
        .join("");
      expect(allText).toBe(text);
    }, 60000);

    test("should handle tool use", async () => {
      if (!model.supportTextGeneration) {
        return;
      }
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
      const toolCalls: {
        name: string;
        arguments: Record<string, unknown>;
        tool_call_id: string;
      }[] = [];
      // A model may open several tool calls for this prompt. Clients whose provider interleaves
      // the argument deltas tag each delta with a correlation id in fidelity; the rest stream one
      // call at a time, so a completed tool_call marks the boundary between them.
      const partials: { name: string; arguments: string }[] = [];
      const partialByCall = new Map<unknown, number>();
      let openIndex: number | undefined;

      const toolPrompt = "What is the weather in San Francisco and in New York?";
      const message1: UniMessage = {
        role: "user",
        content_items: [{ type: "text", text: toolPrompt }],
      };
      for await (const event of client.streamingResponseStateful({
        message: message1,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "partial_tool_call") {
            const itemId = item.fidelity?.item_id;
            let index: number;
            if (itemId !== undefined && itemId !== null) {
              index = partialByCall.get(itemId) ?? partials.length;
              partialByCall.set(itemId, index);
            } else {
              openIndex ??= partials.length;
              index = openIndex;
            }
            while (partials.length <= index) {
              partials.push({ name: "", arguments: "" });
            }
            partials[index].name ||= item.name;
            partials[index].arguments += item.arguments;
          } else if (item.type === "tool_call") {
            toolCalls.push(item);
            // A completed call ends the uncorrelated stream, so the next delta opens the next.
            openIndex = undefined;
          }
        }
      }

      // Check that at least one function call was made, and that each one is fully reconstructable
      // from its own delta stream. Calls are matched by completion order, because a tool_call_id
      // is not unique for every provider: Gemini reuses the function name for its parallel calls.
      expect(toolCalls.length).toBeGreaterThan(0);
      expect(partials.length).toBe(toolCalls.length);
      toolCalls.forEach((toolCall, index) => {
        expect(toolCall.name).toBe(weatherTool.name);
        expect(toolCall.arguments).toHaveProperty("location");
        expect(toolCall.tool_call_id).toBeDefined();
        expect(partials[index].name).toBe(toolCall.name);
        expect(JSON.parse(partials[index].arguments)).toEqual(
          toolCall.arguments,
        );
      });

      // Every open call needs a result, otherwise the replayed history is incomplete.
      const message2: UniMessage = {
        role: "user",
        content_items: toolCalls.map((toolCall) => ({
          type: "tool_result",
          text: "It's 20 degrees.",
          tool_call_id: toolCall.tool_call_id,
        })),
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
      if (!model.supportTextGeneration) {
        return;
      }
      const client = createClient(model);
      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [{ type: "text", text: "Hello" }],
        },
      ];
      const config: UniConfig = {
        system_prompt:
          "You are a kitten. Every reply MUST contain the exact word 'meow' — " +
          "never a variant like 'mreow' or a *purrs* action instead.",
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
      if (!model.supportImageUnderstanding) {
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
        IMAGE_KEYWORDS.some((keyword) => text.toLowerCase().includes(keyword)),
      ).toBe(true);
    }, 60000);

    test("should handle base64 encoded image understanding", async () => {
      if (!model.supportImageUnderstanding) {
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
        IMAGE_KEYWORDS.some((keyword) => text.toLowerCase().includes(keyword)),
      ).toBe(true);
    }, 60000);

    test("should handle tool result with image", async () => {
      if (!model.supportImageUnderstanding) {
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
      const toolCalls: { name: string; tool_call_id: string }[] = [];

      // Prescriptive on purpose: this test covers a tool *result* carrying an image, so
      // reaching that state is setup. Natural tool selection is covered by the tool-use test.
      const toolPrompt =
        "Call get_image exactly once with seed 42. Make that function call your only action " +
        "this turn, then describe the returned image briefly.";
      const message1: UniMessage = {
        role: "user",
        content_items: [{ type: "text", text: toolPrompt }],
      };
      for await (const event of client.streamingResponseStateful({
        message: message1,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "tool_call") {
            toolCalls.push(item);
          }
        }
      }

      expect(toolCalls.length).toBeGreaterThan(0);
      for (const toolCall of toolCalls) {
        expect(toolCall.name).toBe(imageTool.name);
        expect(toolCall.tool_call_id).toBeDefined();
      }

      // Every open call needs a result, otherwise the replayed history is incomplete.
      const message2: UniMessage = {
        role: "user",
        content_items: toolCalls.map((toolCall) => ({
          type: "tool_result",
          text: "Here is the result image:",
          images: [IMAGE],
          tool_call_id: toolCall.tool_call_id,
        })),
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
        IMAGE_KEYWORDS.some((keyword) => text.toLowerCase().includes(keyword)),
      ).toBe(true);
    }, 60000);

    test("should handle image generation", async () => {
      if (!model.supportImageGeneration) {
        return;
      }
      const client = createClient(model);
      const config: UniConfig = {
        image_config: { aspect_ratio: "1:1", image_size: "1K" },
      };
      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [
            {
              type: "text",
              text: "Generate a cozy watercolor illustration of two white flowers with raindrops.",
            },
          ],
        },
      ];

      const inlineItems: { data: Buffer; mime_type: string }[] = [];
      for await (const event of client.streamingResponse({
        messages,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "inline_data") {
            inlineItems.push(item);
          }
        }
      }

      expect(inlineItems.length).toBeGreaterThan(0);
      expect(
        inlineItems.some((item) => item.mime_type.startsWith("image/")),
      ).toBe(true);
      expect(inlineItems.every((item) => item.data.length > 0)).toBe(true);
    }, 180000);

    test("should handle tts generation", async () => {
      if (!model.supportAudioGeneration) {
        return;
      }

      const client = createClient(model);
      const config: UniConfig = {
        tts_config: [{ voice: "Kore" }],
      };
      const messages: UniMessage[] = [
        {
          role: "user",
          content_items: [
            {
              type: "text",
              text: "Say cheerfully: Have a wonderful day!",
            },
          ],
        },
      ];

      const inlineItems: { data: Buffer; mime_type: string }[] = [];
      for await (const event of client.streamingResponse({
        messages,
        config,
      })) {
        checkEventIntegrity(event);
        for (const item of event.content_items) {
          if (item.type === "inline_data") {
            inlineItems.push(item);
          }
        }
      }

      expect(inlineItems.length).toBeGreaterThan(0);
      expect(
        inlineItems.some(
          (item) =>
            item.mime_type.startsWith("audio/") ||
            item.mime_type === "application/octet-stream",
        ),
      ).toBe(true);
      expect(inlineItems.every((item) => item.data.length > 0)).toBe(true);
    }, 180000);

    test("should stream text embeddings", async () => {
      if (!model.supportEmbedding) {
        return;
      }

      const client = createClient(model);
      const messages = [
        {
          role: "user" as const,
          content_items: [{ type: "text" as const, text: "Hello world" }],
        },
        {
          role: "user" as const,
          content_items: [
            { type: "text" as const, text: "Goodbye " },
            { type: "text" as const, text: "world" },
          ],
        },
      ];

      const events: UniEvent[] = [];
      for await (const event of client.streamingResponse({
        messages,
        config: { embedding_config: { dimensions: 768 } },
      })) {
        checkEventIntegrity(event);
        events.push(event);
      }

      const embeddingItems = events.flatMap((event) =>
        event.content_items.filter((item) => item.type === "embedding"),
      );
      expect(embeddingItems.length).toBe(2);
      for (const item of embeddingItems) {
        expect(item.embedding.length).toBe(768);
        expect(item.embedding.every((v) => typeof v === "number")).toBe(true);
      }
    }, 60000);
  });
}

test("should reject unknown model", () => {
  expect(() => new AutoLLMClient({ model: "unknown-model" })).toThrow(
    "not supported",
  );
});

test("should list supported model entries", () => {
  const entries = listSupportedModels();
  const kimi = entries.find((entry) => entry.model === "kimi-k3");
  expect(kimi).toBeDefined();
  expect(kimi?.base_url).toBe("https://api.moonshot.cn/v1");
  expect(kimi?.client).toBe("kimi-k3");
  expect(kimi?.context_window).toBe(1048576);
  expect(kimi?.input_modalities).toEqual(["Text", "Image"]);
  expect(kimi?.output_modalities).toEqual(["Text"]);
  // stored in USD (official CNY prices pre-converted at 7 CNY/USD)
  expect(kimi?.pricing).toEqual({
    currency: "USD",
    prompt_tokens: 2.857143,
    thoughts_tokens: 14.285714,
    response_tokens: 14.285714,
    cached_tokens: 0.285714,
  });

  const kimiCny = listSupportedModels("CNY").find(
    (entry) => entry.model === "kimi-k3",
  );
  expect(kimiCny?.pricing?.currency).toBe("CNY");
  expect(kimiCny?.pricing?.prompt_tokens).toBeCloseTo(20.0, 3);
  expect(kimiCny?.pricing?.thoughts_tokens).toBeCloseTo(100.0, 3);
  expect(kimiCny?.pricing?.response_tokens).toBeCloseTo(100.0, 3);
  expect(kimiCny?.pricing?.cached_tokens).toBeCloseTo(2.0, 3);

  const glm52 = entries.find((entry) => entry.model === "z-ai/glm-5.2");
  expect(glm52?.base_url).toBe("https://openrouter.ai/api/v1");
  expect(glm52?.client).toBe("glm-5.2");

  const minimaxEntries = entries.filter(
    (entry) => entry.base_url === "https://api.minimax.io/v1",
  );
  expect(minimaxEntries).toEqual([
    {
      model: "MiniMax-M3",
      base_url: "https://api.minimax.io/v1",
      client: "minimax-m3",
      input_modalities: ["Text", "Image"],
      output_modalities: ["Text"],
      context_window: 1000000,
    },
  ]);

  const minimaxM3 = new AutoLLMClient({
    model: "MiniMax-M3",
    apiKey: "test-key",
  });
  // Automatic routing matches the exact model id only ...
  expect(
    () =>
      new AutoLLMClient({ model: "MiniMax-M3-preview", apiKey: "test-key" }),
  ).toThrow("not supported");
  // ... but an explicit clientType reaches gateway-hosted or aliased deployments, as for every
  // other client.
  expect(
    () =>
      new AutoLLMClient({
        model: "MiniMax-M3-preview",
        apiKey: "test-key",
        clientType: "minimax-m3",
      }),
  ).not.toThrow();
  // A MiniMax credential is required; the wrapped SDK must never fall back to OPENAI_API_KEY.
  const savedMinimaxKey = process.env.MINIMAX_API_KEY;
  const savedOpenaiKey = process.env.OPENAI_API_KEY;
  delete process.env.MINIMAX_API_KEY;
  process.env.OPENAI_API_KEY = "sk-openai-must-not-reach-the-minimax-host";
  try {
    expect(
      () => new AutoLLMClient({ model: "MiniMax-M3", clientType: "minimax-m3" }),
    ).toThrow("MINIMAX_API_KEY is required");
  } finally {
    if (savedMinimaxKey === undefined) {
      delete process.env.MINIMAX_API_KEY;
    } else {
      process.env.MINIMAX_API_KEY = savedMinimaxKey;
    }
    if (savedOpenaiKey === undefined) {
      delete process.env.OPENAI_API_KEY;
    } else {
      process.env.OPENAI_API_KEY = savedOpenaiKey;
    }
  }

  const m3ThinkingEfforts = [
    [ThinkingLevel.NONE, "none"],
    [ThinkingLevel.LOW, "low"],
    [ThinkingLevel.MEDIUM, "medium"],
    [ThinkingLevel.HIGH, "high"],
    [ThinkingLevel.XHIGH, "high"],
  ] as const;
  for (const [thinkingLevel, effort] of m3ThinkingEfforts) {
    expect(
      minimaxM3.transformUniConfigToModelConfig({
        thinking_level: thinkingLevel,
      }).reasoning,
    ).toEqual({ effort });
  }
  expect(
    minimaxM3.transformUniConfigToModelConfig({
      prompt_caching: PromptCaching.ENABLE,
    }),
  ).not.toHaveProperty("prompt_caching");
  for (const promptCaching of [
    PromptCaching.DISABLE,
    PromptCaching.ENHANCE,
  ]) {
    expect(() =>
      minimaxM3.transformUniConfigToModelConfig({
        prompt_caching: promptCaching,
      }),
    ).toThrow("does not support");
  }
  expect(() =>
    minimaxM3.transformUniConfigToModelConfig({ tool_choice: "required" }),
  ).toThrow("does not support");
  expect(() =>
    minimaxM3.transformUniConfigToModelConfig({ tool_choice: ["lookup"] }),
  ).toThrow("does not support");

  // Assistant text order is preserved around top-level items; the reasoning item is rebuilt from
  // the thinking text alone, and the tool call re-serializes its parsed arguments.
  expect(
    minimaxM3.transformUniMessageToModelInput([
      {
        role: "assistant",
        content_items: [
          { type: "text", text: "before" },
          { type: "thinking", thinking: "reasoning" },
          {
            type: "tool_call",
            name: "lookup",
            arguments: { 城市: "上海" },
            tool_call_id: "call-1",
          },
          { type: "text", text: "after" },
        ],
      },
    ]),
  ).toEqual([
    { role: "assistant", content: [{ type: "output_text", text: "before" }] },
    {
      type: "reasoning",
      content: [{ type: "reasoning_text", text: "reasoning" }],
    },
    {
      type: "function_call",
      call_id: "call-1",
      name: "lookup",
      arguments: '{"城市":"上海"}',
    },
    { role: "assistant", content: [{ type: "output_text", text: "after" }] },
  ]);


  for (const entry of entries) {
    expect(entry.input_modalities.length).toBeGreaterThan(0);
    expect(entry.output_modalities.length).toBeGreaterThan(0);
    const client = new AutoLLMClient({
      model: entry.model,
      apiKey: "test-key",
      baseUrl: entry.base_url,
      clientType: entry.client,
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect((client as any)._client).toBeDefined();
  }
});

test.each([
  ["openai-compatible", "OpenaiClient"],
  ["openai-embedding-compatible", "OpenaiEmbeddingClient"],
])("should route %s clientType to %s", (clientType, clientName) => {
  const client = new AutoLLMClient({
    model: "unknown-model",
    apiKey: "test-key",
    clientType,
  });

  expect((client as any)._client.constructor.name).toBe(clientName);
});

test("should reject non-text content items for OpenAI embeddings", () => {
  const client = new AutoLLMClient({
    model: "unknown-model",
    apiKey: "test-key",
    clientType: "openai-embedding",
  });
  const messages: UniMessage[] = [
    {
      role: "user",
      content_items: [{ type: "image_url", image_url: IMAGE }],
    },
  ];

  expect(() => client.transformUniMessageToModelInput(messages)).toThrow(
    "only support text",
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
