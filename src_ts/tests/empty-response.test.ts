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
import {
  AgentHubError,
  AutoLLMClient,
  EmptyResponseError,
  TextContentItem,
  ToolCallArgumentParseError,
  UniConfig,
  UniEvent,
  UniMessage,
} from "../src";

type FakeOpenAICompatibleClient = {
  baseURL: string;
  chat: {
    completions: {
      create: () => Promise<AsyncIterable<unknown>>;
    };
  };
};

type ReasoningStreamClient = {
  streamingResponse(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncIterable<UniEvent>;
};

interface ReasoningStreamCase {
  expectedClient: string;
  model: string;
  clientType: string;
}

const REASONING_STREAM_CASES: ReasoningStreamCase[] = [
  {
    expectedClient: "OpenaiClient",
    model: "gpt-5.5",
    clientType: "openai",
  },
  {
    expectedClient: "GLM5_1Client",
    model: "glm-5.1",
    clientType: "glm-5.1",
  },
  {
    expectedClient: "KimiK2_6Client",
    model: "kimi-k2.6",
    clientType: "kimi-k2.6",
  },
  {
    expectedClient: "DeepSeekV4Client",
    model: "deepseek-v4",
    clientType: "deepseek-v4",
  },
];

const messages: UniMessage[] = [
  {
    role: "user",
    content_items: [{ type: "text", text: "Create a memo." }],
  },
];

function streamFromChunks(chunks: unknown[]): AsyncIterable<unknown> {
  return {
    async *[Symbol.asyncIterator]() {
      for (const chunk of chunks) {
        yield chunk;
      }
    },
  };
}

function installFakeOpenAICompatibleStream(
  client: ReasoningStreamClient,
  chunks: unknown[],
): void {
  const fakeClient: FakeOpenAICompatibleClient = {
    baseURL: "https://api.test.invalid/v1",
    chat: {
      completions: {
        create: async () => streamFromChunks(chunks),
      },
    },
  };
  const routedClient = (
    client as unknown as { _client: { _client: FakeOpenAICompatibleClient } }
  )._client;
  routedClient._client = fakeClient;
}

function createAutoClient(testCase: ReasoningStreamCase): AutoLLMClient {
  return new AutoLLMClient({
    model: testCase.model,
    apiKey: "test-key",
    clientType: testCase.clientType,
  });
}

function deltaChunk(delta: {
  content?: string;
  reasoning_content?: string;
}): unknown {
  return {
    choices: [{ delta, finish_reason: null }],
    usage: null,
  };
}

function stopChunk(finishReason: string): unknown {
  return {
    choices: [{ delta: {}, finish_reason: finishReason }],
    usage: {
      prompt_tokens: 1,
      completion_tokens: 1,
      completion_tokens_details: { reasoning_tokens: 1 },
      prompt_cache_hit_tokens: 0,
      prompt_cache_miss_tokens: 1,
    },
  };
}

async function collectEvents(
  stream: AsyncIterable<UniEvent>,
): Promise<UniEvent[]> {
  const events: UniEvent[] = [];
  for await (const event of stream) {
    events.push(event);
  }
  return events;
}

async function captureStreamError(
  stream: AsyncIterable<UniEvent>,
): Promise<unknown> {
  let capturedError: unknown;
  try {
    await collectEvents(stream);
  } catch (error) {
    capturedError = error;
  }
  return capturedError;
}

describe.each(REASONING_STREAM_CASES)(
  "Reasoning output validation for $clientType",
  (testCase) => {
    test("rejects thinking-only responses", async () => {
      const client = createAutoClient(testCase);
      installFakeOpenAICompatibleStream(client, [
        deltaChunk({ reasoning_content: "Let me think about the memo." }),
        stopChunk("stop"),
      ]);

      const capturedError = await captureStreamError(
        client.streamingResponse({ messages, config: {} }),
      );

      expect(capturedError).toBeInstanceOf(EmptyResponseError);
      const emptyError = capturedError as EmptyResponseError;
      expect(emptyError.client).toBe(testCase.expectedClient);
      expect(emptyError.finishReason).toBe("stop");
      expect(emptyError.message).toContain("no content other than thinking");
    });

    test("rejects responses without any content", async () => {
      const client = createAutoClient(testCase);
      installFakeOpenAICompatibleStream(client, [stopChunk("length")]);

      const capturedError = await captureStreamError(
        client.streamingResponse({ messages, config: {} }),
      );

      expect(capturedError).toBeInstanceOf(EmptyResponseError);
      expect((capturedError as EmptyResponseError).finishReason).toBe("length");
    });

    test("accepts responses with text content", async () => {
      const client = createAutoClient(testCase);
      installFakeOpenAICompatibleStream(client, [
        deltaChunk({ reasoning_content: "Let me think about the memo." }),
        deltaChunk({ content: "Here is the memo." }),
        stopChunk("stop"),
      ]);

      const events = await collectEvents(
        client.streamingResponse({ messages, config: {} }),
      );
      const texts = events.flatMap((event) =>
        event.content_items
          .filter((item): item is TextContentItem => item.type === "text")
          .map((item) => item.text),
      );
      expect(texts).toEqual(["Here is the memo."]);
    });
  },
);

test("AgentHub errors share the AgentHubError base class", () => {
  const emptyError = new EmptyResponseError({
    client: "OpenaiClient",
    finishReason: "stop",
  });
  expect(emptyError).toBeInstanceOf(AgentHubError);
  const parseError = new ToolCallArgumentParseError({
    client: "OpenaiClient",
    toolName: "exec_command",
    toolCallId: "call_ok",
    rawArguments: "[]",
    reason: "Expected a JSON object.",
  });
  expect(parseError).toBeInstanceOf(AgentHubError);
});
