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

import { expect, describe, test } from "@jest/globals";
import { LLMClient } from "../src/baseClient";
import { UniConfig, UniEvent, UniMessage } from "../src/types";

class PartialAbortClient extends LLMClient {
  constructor() {
    super();
    this._model = "partial-abort-test";
  }

  transformUniConfigToModelConfig(config: UniConfig): UniConfig {
    return config;
  }

  transformUniMessageToModelInput(messages: UniMessage[]): UniMessage[] {
    return messages;
  }

  transformModelOutputToUniEvent(modelOutput: UniEvent): UniEvent {
    return modelOutput;
  }

  async *_streamingResponseInternal(options: {
    messages: UniMessage[];
    config: UniConfig;
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent> {
    yield {
      role: "assistant",
      event_type: "delta",
      content_items: [{ type: "text", text: "hello" }],
      usage_metadata: null,
      finish_reason: null,
    };

    if (options.signal?.aborted) {
      throw new Error("aborted");
    }

    yield {
      role: "assistant",
      event_type: "stop",
      content_items: [],
      usage_metadata: {
        cached_tokens: null,
        prompt_tokens: 1,
        thoughts_tokens: null,
        response_tokens: 1,
      },
      finish_reason: "stop",
    };
  }
}

class IncompleteStreamingClient extends LLMClient {
  constructor() {
    super();
    this._model = "incomplete-streaming-test";
  }

  transformUniConfigToModelConfig(config: UniConfig): UniConfig {
    return config;
  }

  transformUniMessageToModelInput(messages: UniMessage[]): UniMessage[] {
    return messages;
  }

  transformModelOutputToUniEvent(modelOutput: UniEvent): UniEvent {
    return modelOutput;
  }

  async *_streamingResponseInternal(): AsyncGenerator<UniEvent> {
    yield {
      role: "assistant",
      event_type: "delta",
      content_items: [{ type: "text", text: "hello" }],
      usage_metadata: null,
      finish_reason: null,
    };
  }
}

describe("LLMClient stateful abort handling", () => {
  test("should commit partial history when signal aborts after output", async () => {
    const client = new PartialAbortClient();
    const controller = new AbortController();
    const message: UniMessage = {
      role: "user",
      content_items: [{ type: "text", text: "hello" }],
    };
    const stream = client.streamingResponseStateful({
      message,
      config: {},
      signal: controller.signal,
    });

    const first = await stream.next();
    expect(first.done).toBe(false);
    expect(first.value?.content_items).toEqual([
      { type: "text", text: "hello" },
    ]);

    controller.abort("stop");

    await expect(stream.next()).rejects.toThrow("aborted");

    const history = client.getHistory();
    expect(history).toHaveLength(2);
    expect(history[0].role).toBe("user");
    expect(history[0].created_at).toBeGreaterThan(0);
    expect(history[1].role).toBe("assistant");
    expect(history[1].content_items).toEqual([{ type: "text", text: "hello" }]);
    expect(history[1].usage_metadata).toBeNull();
    expect(history[1].finish_reason).toBeNull();
  });

  test("should discard partial history when non-abort error occurs", async () => {
    const client = new IncompleteStreamingClient();
    const message: UniMessage = {
      role: "user",
      content_items: [{ type: "text", text: "hello" }],
    };
    const stream = client.streamingResponseStateful({
      message,
      config: {},
    });

    const first = await stream.next();
    expect(first.done).toBe(false);
    expect(first.value?.content_items).toEqual([
      { type: "text", text: "hello" },
    ]);

    await expect(stream.next()).rejects.toThrow("usage_metadata");

    expect(client.getHistory()).toEqual([]);
  });
});
