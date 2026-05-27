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

import { AutoLLMClient, PromptCaching, ThinkingLevel } from "../src";
import type {
  ContentItem,
  FinishReason,
  ToolSchema,
  UniConfig,
  UniEvent,
  UniMessage,
  UsageMetadata,
} from "../src";
import { expect, test } from "@jest/globals";

test("root entrypoint exports public types and accepts AbortSignal options", () => {
  const signal = new AbortController().signal;
  const usageMetadata: UsageMetadata = {
    cached_tokens: null,
    prompt_tokens: 1,
    thoughts_tokens: null,
    response_tokens: null,
  };
  const finishReason: FinishReason = "stop";
  const contentItem: ContentItem = { type: "text", text: "hello" };
  const message: UniMessage = {
    role: "user",
    content_items: [contentItem],
  };
  const tool: ToolSchema = {
    name: "lookup",
    description: "Lookup a value",
  };
  const config: UniConfig = {
    tools: [tool],
    prompt_caching: PromptCaching.ENABLE,
    thinking_level: ThinkingLevel.LOW,
  };
  const event: UniEvent = {
    role: "assistant",
    event_type: "stop",
    content_items: [],
    usage_metadata: usageMetadata,
    finish_reason: finishReason,
  };

  const client = new AutoLLMClient({
    model: "qwen3.6",
    apiKey: "test-key",
  });

  client.transformUniMessageToModelInput([message], signal);

  expect(
    client.streamingResponse({ messages: [message], config, signal })[
      Symbol.asyncIterator
    ],
  ).toBeDefined();
  expect(
    client.streamingResponseStateful({ message, config, signal })[
      Symbol.asyncIterator
    ],
  ).toBeDefined();
  expect(event.finish_reason).toBe("stop");
});
