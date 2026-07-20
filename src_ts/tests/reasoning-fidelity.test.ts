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
import { AutoLLMClient, UniEvent, UniMessage } from "../src";

type FakeOpenAICompatibleClient = {
  baseURL: string;
  chat: {
    completions: {
      create: () => Promise<AsyncIterable<unknown>>;
    };
  };
};

interface ReasoningReplayCase {
  model: string;
  clientType: string;
}

const REASONING_REPLAY_CASES: ReasoningReplayCase[] = [
  { model: "gpt-5.5", clientType: "openai" },
  { model: "glm-5.1", clientType: "glm-5.1" },
  { model: "kimi-k2.6", clientType: "kimi-k2.6" },
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
  client: AutoLLMClient,
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

function createAutoClient(testCase: ReasoningReplayCase): AutoLLMClient {
  return new AutoLLMClient({
    model: testCase.model,
    apiKey: "test-key",
    clientType: testCase.clientType,
  });
}

function deltaChunk(delta: {
  content?: string;
  reasoning_content?: string;
  reasoning?: string;
}): unknown {
  return {
    choices: [{ delta, finish_reason: null }],
    usage: null,
  };
}

function stopChunk(finishReason: string = "stop"): unknown {
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

function userMessage(): UniMessage {
  return {
    role: "user",
    content_items: [{ type: "text", text: "Create a memo." }],
  };
}

async function transformHistory(
  client: AutoLLMClient,
  history: UniMessage[],
): Promise<Record<string, unknown>[]> {
  return (await client.transformUniMessageToModelInput(history)) as Record<
    string,
    unknown
  >[];
}

async function runTurnAndReplay(client: AutoLLMClient): Promise<{
  historyMessage: UniMessage;
  replayedMessage: Record<string, unknown>;
}> {
  const events: UniEvent[] = [];
  for await (const event of client.streamingResponseStateful({
    message: userMessage(),
    config: {},
  })) {
    events.push(event);
  }

  const history = client.getHistory();
  const modelInput = await transformHistory(client, history);
  const historyMessage = history[history.length - 1];
  const replayedMessage = modelInput[modelInput.length - 1];
  if (!historyMessage || !replayedMessage) {
    throw new Error("history or model input is empty");
  }

  return { historyMessage, replayedMessage };
}

function thinkingItems(message: UniMessage): unknown[] {
  return message.content_items.filter((item) => item.type === "thinking");
}

describe.each(REASONING_REPLAY_CASES)(
  "Reasoning field fidelity for $clientType",
  (testCase) => {
    test("replay preserves the reasoning_content field", async () => {
      const client = createAutoClient(testCase);
      installFakeOpenAICompatibleStream(client, [
        deltaChunk({ reasoning_content: "Let me think" }),
        deltaChunk({ reasoning_content: " about the memo." }),
        deltaChunk({ content: "Here is the memo." }),
        stopChunk(),
      ]);

      const { historyMessage, replayedMessage } =
        await runTurnAndReplay(client);
      expect(thinkingItems(historyMessage)).toEqual([
        {
          type: "thinking",
          thinking: "Let me think about the memo.",
          signature: "reasoning_content",
        },
      ]);
      expect(replayedMessage.reasoning_content).toBe(
        "Let me think about the memo.",
      );
      expect(replayedMessage).not.toHaveProperty("reasoning");
    });

    test("replay preserves the reasoning field", async () => {
      const client = createAutoClient(testCase);
      installFakeOpenAICompatibleStream(client, [
        deltaChunk({ reasoning: "Let me think" }),
        deltaChunk({ reasoning: " about the memo." }),
        deltaChunk({ content: "Here is the memo." }),
        stopChunk(),
      ]);

      const { historyMessage, replayedMessage } =
        await runTurnAndReplay(client);
      expect(thinkingItems(historyMessage)).toEqual([
        {
          type: "thinking",
          thinking: "Let me think about the memo.",
          signature: "reasoning",
        },
      ]);
      expect(replayedMessage.reasoning).toBe("Let me think about the memo.");
      expect(replayedMessage).not.toHaveProperty("reasoning_content");
    });

    test("replay keeps both fields when the origin is ambiguous", async () => {
      const client = createAutoClient(testCase);
      installFakeOpenAICompatibleStream(client, [
        deltaChunk({
          reasoning_content: "Let me think.",
          reasoning: "Let me think.",
        }),
        deltaChunk({ content: "Here is the memo." }),
        stopChunk(),
      ]);

      const { historyMessage, replayedMessage } =
        await runTurnAndReplay(client);
      expect(thinkingItems(historyMessage)).toEqual([
        { type: "thinking", thinking: "Let me think." },
      ]);
      expect(replayedMessage.reasoning_content).toBe("Let me think.");
      expect(replayedMessage.reasoning).toBe("Let me think.");
    });

    test("replay of unsigned thinking sends both fields", async () => {
      const client = createAutoClient(testCase);
      const history: UniMessage[] = [
        userMessage(),
        {
          role: "assistant",
          content_items: [
            { type: "thinking", thinking: "Let me think." },
            { type: "text", text: "Here is the memo." },
          ],
        },
      ];

      const modelInput = await transformHistory(client, history);
      const replayedMessage = modelInput[modelInput.length - 1];
      if (!replayedMessage) {
        throw new Error("model input is empty");
      }

      expect(replayedMessage.reasoning_content).toBe("Let me think.");
      expect(replayedMessage.reasoning).toBe("Let me think.");
    });
  },
);
