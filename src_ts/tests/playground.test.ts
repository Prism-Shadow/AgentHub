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

import request from "supertest";
import { describe, test, expect, beforeEach, jest } from "@jest/globals";
import { AutoLLMClient } from "../src/autoClient";
import { createChatApp } from "../src/integration/playground";
import { UniConfig, UniEvent, UniMessage } from "../src/types";

let mockLastStreamingOptions: {
  message: UniMessage;
  config: UniConfig;
} | null = null;

jest.mock("../src/autoClient", () => ({
  AutoLLMClient: jest.fn().mockImplementation(() => ({
    streamingResponseStateful: async function* (options: {
      message: UniMessage;
      config: UniConfig;
    }): AsyncGenerator<UniEvent> {
      mockLastStreamingOptions = options;
      yield {
        role: "assistant",
        event_type: "stop",
        content_items: [],
        usage_metadata: null,
        finish_reason: "stop",
        created_at: 0,
      };
    },
    clearHistory: jest.fn(),
  })),
}));

describe("Playground", () => {
  beforeEach(() => {
    mockLastStreamingOptions = null;
    jest.clearAllMocks();
  });

  test("should render client connection inputs", async () => {
    const app = createChatApp();

    const response = await request(app).get("/");

    expect(response.status).toBe(200);
    expect(response.text).toContain(
      '<h1 class="text-xl font-semibold">AgentHub</h1>',
    );
    expect(response.text).toContain('id="modelCombobox"');
    expect(response.text).toContain('id="thinkingLevelCombobox"');
    expect(response.text).toContain('id="thinkingSummaryCombobox"');
    expect(response.text).toContain('id="toolChoiceCombobox"');
    expect(response.text).toContain(
      'data-combobox-option data-value="gpt-5.5"',
    );
    expect(response.text).toContain("toggleCombobox('modelCombobox')");
    expect(response.text).toContain(
      "selectComboboxOption('modelCombobox', this)",
    );
    expect(response.text).toContain("customModelInput");
    expect(response.text).toContain("handleModelSelectChange()");
    expect(response.text).not.toContain("modelDropdown");
    expect(response.text).not.toContain("toggleModelMenu()");
    expect(response.text).not.toContain("<select");
    expect(response.text).not.toContain("<datalist");
    expect(response.text).toContain("apiKeyInput");
    expect(response.text).toContain("apiKeyVisibilityToggle");
    expect(response.text).toContain("toggleApiKeyVisibility()");
    expect(response.text).toContain(
      'id="apiKeyVisibilityShowIcon" class="hidden"',
    );
    expect(response.text).toContain('id="apiKeyVisibilityHideIcon" xmlns=');
    expect(response.text).toContain("baseUrlInput");
    expect(response.text).toContain("renderEmbedding");
    expect(response.text).toContain("item.embedding.slice(0, 5)");
    expect(response.text).toContain('href="/tracer/"');
    expect(response.text).toContain('target="_blank"');
    expect(response.text).toContain("Open Tracer");
    expect(
      response.text.indexOf('<h1 class="text-xl font-semibold">'),
    ).toBeLessThan(response.text.indexOf(">GitHub<"));
    expect(response.text.indexOf(">GitHub<")).toBeLessThan(
      response.text.indexOf(">Open Tracer<"),
    );
    expect(response.text).not.toContain("temperatureInput");
    expect(response.text).not.toContain("maxTokensInput");
  });

  test("should mount tracer on the same app", async () => {
    const app = createChatApp();

    const response = await request(app).get("/tracer/");

    expect(response.status).toBe(200);
    expect(response.text).toContain("Tracer");
    expect(response.text).toContain('href="/tracer/"');
  });

  test("should use client connection options outside request config", async () => {
    const app = createChatApp();
    const message: UniMessage = {
      role: "user",
      content_items: [{ type: "text", text: "Hello" }],
    };

    const response = await request(app)
      .post("/api/chat")
      .send({
        session_id: "connection-options",
        message,
        config: {
          model: "gpt-5.5",
          api_key: "test-key",
          base_url: "https://example.test/v1",
          thinking_level: "low",
        },
      });

    expect(response.status).toBe(200);
    expect(response.text).toContain("data:");
    expect(AutoLLMClient).toHaveBeenCalledWith({
      model: "gpt-5.5",
      apiKey: "test-key",
      baseUrl: "https://example.test/v1",
    });
    expect(mockLastStreamingOptions?.config).toEqual({
      thinking_level: "low",
    });
  });

  test("should accept image payloads larger than the default JSON limit", async () => {
    const app = createChatApp();
    const largeImage = `data:image/png;base64,${"a".repeat(150_000)}`;
    const message: UniMessage = {
      role: "user",
      content_items: [{ type: "image_url", image_url: largeImage }],
    };

    const response = await request(app)
      .post("/api/chat")
      .send({
        session_id: "large-image",
        message,
        config: {
          model: "gpt-5.5",
        },
      });

    expect(response.status).toBe(200);
    expect(response.text).toContain("data:");
    expect(mockLastStreamingOptions?.message.content_items[0]).toEqual({
      type: "image_url",
      image_url: largeImage,
    });
  });
});
