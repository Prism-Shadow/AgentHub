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
  TextContentItem,
  UniConfig,
  UniEvent,
  UniMessage,
} from "../src";

type StreamClient = {
  streamingResponse(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncIterable<UniEvent>;
};

interface StreamCase {
  expectedClient: string;
  model: string;
  clientType: string;
}

// Every client that parses the OpenAI Responses SSE shape.
const RESPONSES_STREAM_CASES: StreamCase[] = [
  {
    expectedClient: "GPT5_6Client",
    model: "gpt-5.6",
    clientType: "gpt-5.6",
  },
  {
    expectedClient: "OpenaiResponsesClient",
    model: "gpt-5.6",
    clientType: "openai-responses",
  },
  {
    expectedClient: "MiniMaxM3Client",
    model: "minimax-m3",
    clientType: "minimax-m3",
  },
];

// Every client that parses the OpenAI Chat Completions chunk shape.
const CHAT_STREAM_CASES: StreamCase[] = [
  {
    expectedClient: "OpenaiChatClient",
    model: "gpt-5.6",
    clientType: "openai-chat",
  },
  {
    expectedClient: "DeepSeekV4Client",
    model: "deepseek-v4",
    clientType: "deepseek-v4",
  },
  {
    expectedClient: "GLM5_3Client",
    model: "glm-5.3",
    clientType: "glm-5.3",
  },
  {
    expectedClient: "KimiK3Client",
    model: "kimi-k3",
    clientType: "kimi-k3",
  },
];

// Every client that parses the Anthropic Messages event shape.
const MESSAGES_STREAM_CASES: StreamCase[] = [
  {
    expectedClient: "Claude5Client",
    model: "claude-sonnet-5",
    clientType: "claude-sonnet-5",
  },
  {
    expectedClient: "AntMessagesClient",
    model: "claude-sonnet-5",
    clientType: "ant-messages",
  },
];

// Every client that parses the Gemini generateContent chunk shape.
const GEMINI_STREAM_CASES: StreamCase[] = [
  {
    expectedClient: "Gemini3_7Client",
    model: "gemini-3.7-flash",
    clientType: "gemini-3.7",
  },
];

const messages: UniMessage[] = [
  {
    role: "user",
    content_items: [{ type: "text", text: "Create a memo." }],
  },
];

function streamFromEvents(events: unknown[]): AsyncIterable<unknown> {
  return {
    async *[Symbol.asyncIterator]() {
      for (const event of events) {
        yield event;
      }
    },
  };
}

function installFakeStream(client: StreamClient, fakeClient: unknown): void {
  const routedClient = (
    client as unknown as { _client: { _client: unknown } }
  )._client;
  routedClient._client = fakeClient;
}

function installFakeResponsesStream(
  client: StreamClient,
  events: unknown[],
): void {
  installFakeStream(client, {
    responses: { create: async () => streamFromEvents(events) },
  });
}

function installFakeChatStream(client: StreamClient, events: unknown[]): void {
  installFakeStream(client, {
    baseURL: "https://api.test.invalid/v1",
    chat: {
      completions: { create: async () => streamFromEvents(events) },
    },
  });
}

function installFakeMessagesStream(
  client: StreamClient,
  events: unknown[],
): void {
  installFakeStream(client, {
    baseURL: "https://api.test.invalid",
    beta: { messages: { create: async () => streamFromEvents(events) } },
  });
}

function installFakeGeminiStream(
  client: StreamClient,
  events: unknown[],
): void {
  installFakeStream(client, {
    models: { generateContentStream: async () => streamFromEvents(events) },
  });
}

function routedClientName(client: StreamClient): string {
  return (client as unknown as { _client: object })._client.constructor.name;
}

function createAutoClient(testCase: StreamCase): AutoLLMClient {
  return new AutoLLMClient({
    model: testCase.model,
    apiKey: "test-key",
    clientType: testCase.clientType,
  });
}

// Heartbeats come from gateways in front of the provider (one-api-style proxies), never
// from the official APIs, so the event shapes below are synthesized from the report in
// https://github.com/Prism-Shadow/penguin-harness/issues/286.
function responsesKeepaliveEvent(sequenceNumber: number): unknown {
  return { type: "keepalive", sequence_number: sequenceNumber };
}

function responsesTextDeltaEvent(text: string): unknown {
  return { type: "response.output_text.delta", delta: text };
}

function responsesCompletedEvent(): unknown {
  return {
    type: "response.completed",
    response: {
      status: "completed",
      usage: {
        input_tokens: 2,
        output_tokens: 3,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens_details: { reasoning_tokens: 1 },
      },
    },
  };
}

function chatKeepaliveChunk(sequenceNumber: number): unknown {
  // A heartbeat is not a Chat Completions chunk, so the fields the client reads are
  // simply absent: choices arrives as undefined rather than an empty list.
  return { type: "keepalive", sequence_number: sequenceNumber };
}

function chatTextChunk(text: string): unknown {
  return {
    choices: [{ delta: { content: text }, finish_reason: null }],
  };
}

function chatStopChunk(): unknown {
  return {
    choices: [{ delta: {}, finish_reason: "stop" }],
    usage: {
      prompt_tokens: 2,
      completion_tokens: 3,
      completion_tokens_details: { reasoning_tokens: 1 },
      prompt_cache_hit_tokens: 0,
      prompt_cache_miss_tokens: 2,
    },
  };
}

function messagesPingEvent(): unknown {
  return { type: "ping" };
}

function messagesStartEvent(): unknown {
  return {
    type: "message_start",
    message: {
      usage: {
        input_tokens: 2,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
    },
  };
}

function messagesTextDeltaEvent(text: string): unknown {
  return {
    type: "content_block_delta",
    delta: { type: "text_delta", text: text },
  };
}

function messagesStopEvent(): unknown {
  return {
    type: "message_delta",
    delta: { stop_reason: "end_turn" },
    usage: {
      input_tokens: 2,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      output_tokens: 3,
    },
  };
}

function geminiKeepaliveChunk(): unknown {
  // The SDK maps only the fields it knows onto the response, so a heartbeat reaches the
  // client as a chunk carrying neither candidates nor usage.
  return {};
}

function geminiTextChunk(text: string): unknown {
  return {
    candidates: [{ content: { parts: [{ text: text }] }, finishReason: null }],
  };
}

function geminiStopChunk(): unknown {
  return {
    // FinishReason is a string enum, so the raw value keys the client's mapping
    candidates: [{ content: { parts: [] }, finishReason: "STOP" }],
    usageMetadata: {
      promptTokenCount: 2,
      cachedContentTokenCount: 0,
      thoughtsTokenCount: 1,
      candidatesTokenCount: 3,
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

function collectedTexts(events: UniEvent[]): string[] {
  return events.flatMap((event) =>
    event.content_items
      .filter((item): item is TextContentItem => item.type === "text")
      .map((item) => item.text),
  );
}

describe.each(RESPONSES_STREAM_CASES)(
  "Keepalive handling for $clientType",
  (testCase) => {
    test("skips gateway keepalive heartbeats between stream events", async () => {
      const client = createAutoClient(testCase);
      expect(routedClientName(client)).toBe(testCase.expectedClient);
      installFakeResponsesStream(client, [
        responsesKeepaliveEvent(1),
        responsesTextDeltaEvent("Here is"),
        responsesKeepaliveEvent(2),
        responsesTextDeltaEvent(" the memo."),
        responsesCompletedEvent(),
        responsesKeepaliveEvent(3),
      ]);

      const events = await collectEvents(
        client.streamingResponse({ messages, config: {} }),
      );
      expect(collectedTexts(events)).toEqual(["Here is", " the memo."]);
      expect(events[events.length - 1].finish_reason).toBe("stop");
    });

    test("still rejects genuinely unknown events", async () => {
      const client = createAutoClient(testCase);
      installFakeResponsesStream(client, [
        { type: "response.mystery_event" },
        responsesCompletedEvent(),
      ]);

      await expect(
        collectEvents(client.streamingResponse({ messages, config: {} })),
      ).rejects.toThrow("Unknown output");
    });
  },
);

describe.each(CHAT_STREAM_CASES)(
  "Keepalive handling for $clientType",
  (testCase) => {
    test("skips gateway keepalive heartbeats between stream chunks", async () => {
      const client = createAutoClient(testCase);
      expect(routedClientName(client)).toBe(testCase.expectedClient);
      installFakeChatStream(client, [
        chatKeepaliveChunk(1),
        chatTextChunk("Here is"),
        chatKeepaliveChunk(2),
        chatTextChunk(" the memo."),
        chatStopChunk(),
        chatKeepaliveChunk(3),
      ]);

      const events = await collectEvents(
        client.streamingResponse({ messages, config: {} }),
      );
      expect(collectedTexts(events)).toEqual(["Here is", " the memo."]);
      expect(events[events.length - 1].finish_reason).toBe("stop");
    });
  },
);

describe.each(MESSAGES_STREAM_CASES)(
  "Keepalive handling for $clientType",
  (testCase) => {
    test("skips gateway ping heartbeats between stream events", async () => {
      const client = createAutoClient(testCase);
      expect(routedClientName(client)).toBe(testCase.expectedClient);
      installFakeMessagesStream(client, [
        messagesPingEvent(),
        messagesStartEvent(),
        messagesTextDeltaEvent("Here is"),
        messagesPingEvent(),
        messagesTextDeltaEvent(" the memo."),
        messagesStopEvent(),
        messagesPingEvent(),
      ]);

      const events = await collectEvents(
        client.streamingResponse({ messages, config: {} }),
      );
      expect(collectedTexts(events)).toEqual(["Here is", " the memo."]);
      expect(events[events.length - 1].finish_reason).toBe("stop");
    });

    test("still rejects genuinely unknown events", async () => {
      const client = createAutoClient(testCase);
      installFakeMessagesStream(client, [
        messagesStartEvent(),
        { type: "mystery_event" },
        messagesStopEvent(),
      ]);

      await expect(
        collectEvents(client.streamingResponse({ messages, config: {} })),
      ).rejects.toThrow("Unknown output");
    });
  },
);

describe.each(GEMINI_STREAM_CASES)(
  "Keepalive handling for $clientType",
  (testCase) => {
    test("skips gateway keepalive heartbeats between stream chunks", async () => {
      const client = createAutoClient(testCase);
      expect(routedClientName(client)).toBe(testCase.expectedClient);
      installFakeGeminiStream(client, [
        geminiKeepaliveChunk(),
        geminiTextChunk("Here is"),
        geminiKeepaliveChunk(),
        geminiTextChunk(" the memo."),
        geminiStopChunk(),
        geminiKeepaliveChunk(),
      ]);

      const events = await collectEvents(
        client.streamingResponse({ messages, config: {} }),
      );
      expect(collectedTexts(events)).toEqual(["Here is", " the memo."]);
      // a heartbeat must not surface as an empty event of its own
      expect(events).toHaveLength(3);
      expect(events[events.length - 1].finish_reason).toBe("stop");
    });
  },
);
