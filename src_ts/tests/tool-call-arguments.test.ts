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
import { DeepSeekV4Client } from "../src/deepseek_v4";
import {
  parseToolCallArguments,
  ToolCallArgumentParseError,
} from "../src/errors";
import { GLM5_1Client } from "../src/glm5_1";
import { KimiK2_6Client } from "../src/kimi_k2_6";
import { OpenaiClient } from "../src/openai";
import {
  ToolCallContentItem,
  UniConfig,
  UniEvent,
  UniMessage,
} from "../src/types";

type FakeOpenAICompatibleClient = {
  baseURL: string;
  chat: {
    completions: {
      create: () => Promise<AsyncIterable<unknown>>;
    };
  };
};

type OpenAICompatibleToolStreamClient = {
  _streamingResponseInternal(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncIterable<UniEvent>;
};

interface OpenAICompatibleToolStreamCase {
  name: string;
  clientName: string;
  createClient: () => OpenAICompatibleToolStreamClient;
}

const OPENAI_COMPATIBLE_TOOL_STREAM_CASES: OpenAICompatibleToolStreamCase[] = [
  {
    name: "openai",
    clientName: "openai",
    createClient: () =>
      new OpenaiClient({ model: "gpt-5.5", apiKey: "test-key" }),
  },
  {
    name: "glm5_1",
    clientName: "glm5_1",
    createClient: () =>
      new GLM5_1Client({ model: "glm-5.1", apiKey: "test-key" }),
  },
  {
    name: "kimi_k2_6",
    clientName: "kimi_k2_6",
    createClient: () =>
      new KimiK2_6Client({ model: "kimi-k2.6", apiKey: "test-key" }),
  },
  {
    name: "deepseek_v4",
    clientName: "deepseek_v4",
    createClient: () =>
      new DeepSeekV4Client({ model: "deepseek-v4", apiKey: "test-key" }),
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
  (client as unknown as { _client: FakeOpenAICompatibleClient })._client =
    fakeClient;
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

test("rejects non-object tool call arguments", () => {
  let capturedError: unknown;
  try {
    parseToolCallArguments("[]", "openai", "exec_command", "call_array");
  } catch (error) {
    capturedError = error;
  }

  expect(capturedError).toBeInstanceOf(ToolCallArgumentParseError);
  const parseError = capturedError as ToolCallArgumentParseError;
  expect(parseError.client).toBe("openai");
  expect(parseError.toolName).toBe("exec_command");
  expect(parseError.toolCallId).toBe("call_array");
  expect(parseError.message).toContain("expected a JSON object");
});

describe.each(OPENAI_COMPATIBLE_TOOL_STREAM_CASES)(
  "OpenAI-compatible tool call streaming for $name",
  (testCase) => {
    test("combines valid streamed tool call arguments", async () => {
      const client = testCase.createClient();
      installFakeOpenAICompatibleStream(client, [
        toolDeltaChunk("call_ok", "exec_command", '{"cmd":'),
        toolDeltaChunk("", "", '"echo ok"}'),
        toolStopChunk(),
      ]);

      const events = await collectEvents(
        client._streamingResponseInternal({ messages, config: {} }),
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
      const client = testCase.createClient();
      installFakeOpenAICompatibleStream(client, [
        toolDeltaChunk(
          "call_bad",
          "exec_command",
          '{"cmd":"python create_docx.py',
        ),
        toolStopChunk(),
      ]);

      let capturedError: unknown;
      try {
        await collectEvents(
          client._streamingResponseInternal({ messages, config: {} }),
        );
      } catch (error) {
        capturedError = error;
      }

      expect(capturedError).toBeInstanceOf(ToolCallArgumentParseError);
      const parseError = capturedError as ToolCallArgumentParseError;
      expect(parseError.client).toBe(testCase.clientName);
      expect(parseError.toolName).toBe("exec_command");
      expect(parseError.toolCallId).toBe("call_bad");
      expect(parseError.rawArgumentsLength).toBeGreaterThan(0);
      expect(parseError.rawArgumentsPreview).toContain("create_docx.py");
      expect(parseError.message).toMatch(/Unterminated string/u);
    });
  },
);
