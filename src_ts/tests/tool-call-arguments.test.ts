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
  AutoLLMClient,
  ToolCallArgumentParseError,
  ToolCallContentItem,
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

type OpenAICompatibleToolStreamClient = {
  streamingResponse(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncIterable<UniEvent>;
};

interface OpenAICompatibleToolStreamCase {
  expectedClient: string;
  model: string;
  clientType: string;
}

const OPENAI_COMPATIBLE_TOOL_STREAM_CASES: OpenAICompatibleToolStreamCase[] = [
  {
    expectedClient: "OpenaiChatClient",
    model: "gpt-5.5",
    clientType: "openai",
  },
  {
    expectedClient: "GLM5_3Client",
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
  client: OpenAICompatibleToolStreamClient,
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

function createAutoClient(
  testCase: OpenAICompatibleToolStreamCase,
): AutoLLMClient {
  return new AutoLLMClient({
    model: testCase.model,
    apiKey: "test-key",
    clientType: testCase.clientType,
  });
}

function toolDeltaChunk(
  toolCallId: string,
  name: string,
  args: string,
): unknown {
  return {
    choices: [
      {
        delta: {
          tool_calls: [
            {
              id: toolCallId,
              function: { name, arguments: args },
            },
          ],
        },
        finish_reason: null,
      },
    ],
    usage: null,
  };
}

function toolStopChunk(): unknown {
  return {
    choices: [{ delta: {}, finish_reason: "tool_calls" }],
    usage: {
      completion_tokens: 1,
      completion_tokens_details: { reasoning_tokens: 0 },
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

describe.each(OPENAI_COMPATIBLE_TOOL_STREAM_CASES)(
  "OpenAI-compatible tool call streaming for $clientType",
  (testCase) => {
    test("combines valid streamed tool call arguments", async () => {
      const client = createAutoClient(testCase);
      installFakeOpenAICompatibleStream(client, [
        toolDeltaChunk("call_ok", "exec_command", '{"cmd":'),
        toolDeltaChunk("", "", '"echo ok"}'),
        toolStopChunk(),
      ]);

      const events = await collectEvents(
        client.streamingResponse({ messages, config: {} }),
      );
      const toolCalls = events.flatMap((event) =>
        event.content_items.filter(
          (item): item is ToolCallContentItem => item.type === "tool_call",
        ),
      );

      expect(toolCalls).toHaveLength(1);
      expect(toolCalls[0]).toEqual({
        type: "tool_call",
        name: "exec_command",
        arguments: { cmd: "echo ok" },
        tool_call_id: "call_ok",
      });
    });

    test("reports malformed streamed tool call arguments with context", async () => {
      const client = createAutoClient(testCase);
      installFakeOpenAICompatibleStream(client, [
        toolDeltaChunk(
          "call_bad",
          "exec_command",
          '{"cmd":"python create_docx.py',
        ),
        toolStopChunk(),
      ]);

      const capturedError = await captureStreamError(
        client.streamingResponse({ messages, config: {} }),
      );

      expect(capturedError).toBeInstanceOf(ToolCallArgumentParseError);
      const parseError = capturedError as ToolCallArgumentParseError;
      expect(parseError.client).toBe(testCase.expectedClient);
      expect(parseError.toolName).toBe("exec_command");
      expect(parseError.toolCallId).toBe("call_bad");
      expect(parseError.rawArgumentsLength).toBeGreaterThan(0);
      expect(parseError.rawArgumentsPreview).toContain("create_docx.py");
      expect(parseError.message).toMatch(/Unterminated string/u);
    });

    test("reports non-object streamed tool call arguments with context", async () => {
      const client = createAutoClient(testCase);
      installFakeOpenAICompatibleStream(client, [
        toolDeltaChunk("call_array", "exec_command", "[]"),
        toolStopChunk(),
      ]);

      const capturedError = await captureStreamError(
        client.streamingResponse({ messages, config: {} }),
      );

      expect(capturedError).toBeInstanceOf(ToolCallArgumentParseError);
      const parseError = capturedError as ToolCallArgumentParseError;
      expect(parseError.client).toBe(testCase.expectedClient);
      expect(parseError.toolName).toBe("exec_command");
      expect(parseError.toolCallId).toBe("call_array");
      expect(parseError.rawArgumentsLength).toBe(2);
      expect(parseError.rawArgumentsPreview).toBe("[]");
      expect(parseError.message).toContain("Expected a JSON object.");
    });
  },
);
