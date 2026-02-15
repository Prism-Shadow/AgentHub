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
import * as path from "path";
import { Tracer } from "../src/integration/tracer";
import { UniMessage } from "../src/types";

describe("Tracer", () => {
  let tempCacheDir: string;

  beforeEach(() => {
    tempCacheDir = fs.mkdtempSync(path.join(process.cwd(), "test-cache-"));
  });

  afterEach(() => {
    if (fs.existsSync(tempCacheDir)) {
      fs.rmSync(tempCacheDir, { recursive: true, force: true });
    }
  });

  test("should initialize with cache directory", () => {
    const tracer = new Tracer(tempCacheDir);
    expect(fs.existsSync(tempCacheDir)).toBe(true);
  });

  test("should save conversation history to files", () => {
    const tracer = new Tracer(tempCacheDir);

    const model = "fake-model";
    const history: UniMessage[] = [
      {
        role: "user",
        content_items: [{ type: "text", text: "Hello" }],
      },
      {
        role: "assistant",
        content_items: [{ type: "text", text: "Hi there!" }],
      },
    ];

    const config = { temperature: 0.7 };
    const configWithModel = { ...config, model };
    const fileId = "test/conversation";

    tracer.saveHistory(model, history, fileId, config);

    const jsonPath = path.join(tempCacheDir, fileId + ".json");
    const txtPath = path.join(tempCacheDir, fileId + ".txt");

    expect(fs.existsSync(jsonPath)).toBe(true);
    expect(fs.existsSync(txtPath)).toBe(true);

    const txtContent = fs.readFileSync(txtPath, "utf-8");
    expect(txtContent).toContain("USER:");
    expect(txtContent).toContain("ASSISTANT:");
    expect(txtContent).toContain("Hello");
    expect(txtContent).toContain("Hi there!");
    expect(txtContent).toContain("temperature");

    const jsonContent = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
    expect(jsonContent.history).toHaveLength(2);
    expect(jsonContent.config).toEqual(configWithModel);
    expect(jsonContent.timestamp).toBeDefined();
  });

  test("should create necessary directories when saving history", () => {
    const tracer = new Tracer(tempCacheDir);

    const model = "fake-model";
    const history: UniMessage[] = [
      {
        role: "user",
        content_items: [{ type: "text", text: "Test" }],
      },
    ];

    const fileId = "agent1/subfolder/conversation";
    const config = {};

    tracer.saveHistory(model, history, fileId, config);

    const jsonPath = path.join(tempCacheDir, fileId + ".json");
    expect(fs.existsSync(jsonPath)).toBe(true);
    expect(fs.existsSync(path.dirname(jsonPath))).toBe(true);
  });

  test("should overwrite existing files", () => {
    const tracer = new Tracer(tempCacheDir);

    const model = "fake-model";
    const history1: UniMessage[] = [
      {
        role: "user",
        content_items: [{ type: "text", text: "First message" }],
      },
    ];

    const history2: UniMessage[] = [
      {
        role: "user",
        content_items: [{ type: "text", text: "Second message" }],
      },
    ];

    const fileId = "test/conversation";
    const config = {};

    tracer.saveHistory(model, history1, fileId, config);
    tracer.saveHistory(model, history2, fileId, config);

    const jsonPath = path.join(tempCacheDir, fileId + ".json");
    const jsonContent = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));

    expect(jsonContent.history).toHaveLength(1);
    expect(jsonContent.history[0].content_items[0].text).toBe("Second message");
  });

  test("should handle relative paths", () => {
    const tracer = new Tracer(tempCacheDir);

    const model = "fake-model";
    const history: UniMessage[] = [
      {
        role: "user",
        content_items: [{ type: "text", text: "Test" }],
      },
    ];

    const relativePath = "conversations/conv1";
    const config = {};

    tracer.saveHistory(model, history, relativePath, config);

    const jsonPath = path.join(tempCacheDir, relativePath + ".json");
    expect(fs.existsSync(jsonPath)).toBe(true);
  });

  test("should create web app", () => {
    const tracer = new Tracer(tempCacheDir);
    const app = tracer.createWebApp();
    expect(app).toBeDefined();
  });

  test("should format history with usage metadata", () => {
    const tracer = new Tracer(tempCacheDir);

    const model = "fake-model";
    const history: UniMessage[] = [
      {
        role: "user",
        content_items: [{ type: "text", text: "Hello" }],
      },
      {
        role: "assistant",
        content_items: [{ type: "text", text: "Hi!" }],
        usage_metadata: {
          prompt_tokens: 10,
          thoughts_tokens: 5,
          response_tokens: 15,
          cached_tokens: 2,
        },
        finish_reason: "stop",
      },
    ];

    const config = {};
    const fileId = "test/metadata_test";

    tracer.saveHistory(model, history, fileId, config);

    const txtPath = path.join(tempCacheDir, fileId + ".txt");
    const txtContent = fs.readFileSync(txtPath, "utf-8");

    expect(txtContent).toContain("Prompt Tokens: 10");
    expect(txtContent).toContain("Thoughts Tokens: 5");
    expect(txtContent).toContain("Response Tokens: 15");
    expect(txtContent).toContain("Cached Tokens: 2");
    expect(txtContent).toContain("Finish Reason: stop");
  });

  test("should pre-render system and tools in config", () => {
    const tracer = new Tracer(tempCacheDir);

    const model = "fake-model";
    const history: UniMessage[] = [
      {
        role: "user",
        content_items: [{ type: "text", text: "Hello" }],
      },
    ];

    const config = {
      system_prompt: "You are a helpful assistant.",
      tools: [
        {
          name: "get_weather",
          description: "Get the weather for a location",
          parameters: {
            type: "object",
            properties: {
              location: { type: "string", description: "City name" },
            },
          },
        },
      ],
      temperature: 0.7,
    };
    const fileId = "test/config_render_test";

    tracer.saveHistory(model, history, fileId, config);

    const txtPath = path.join(tempCacheDir, fileId + ".txt");
    const txtContent = fs.readFileSync(txtPath, "utf-8");

    // Check that system_prompt is rendered with proper formatting
    expect(txtContent).toContain("system_prompt:");
    expect(txtContent).toContain("You are a helpful assistant");

    // Check that tools are rendered as JSON
    expect(txtContent).toContain("tools:");
    expect(txtContent).toContain("get_weather");
    expect(txtContent).toContain("parameters");
  });
});
