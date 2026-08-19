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

type FakeResponsesClient = {
  responses: {
    create: () => Promise<AsyncIterable<unknown>>;
  };
};

type ResponsesStreamClient = {
  streamingResponse(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncIterable<UniEvent>;
};

interface ResponsesStreamCase {
  expectedClient: string;
  model: string;
  clientType: string;
}

// Every client that parses the OpenAI Responses SSE shape.
const RESPONSES_STREAM_CASES: ResponsesStreamCase[] = [
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

function installFakeResponsesStream(
  client: ResponsesStreamClient,
  events: unknown[],
): void {
  const fakeClient: FakeResponsesClient = {
    responses: {
      create: async () => streamFromEvents(events),
    },
  };
  const routedClient = (
    client as unknown as { _client: { _client: FakeResponsesClient } }
  )._client;
  routedClient._client = fakeClient;
}

function routedClientName(client: ResponsesStreamClient): string {
  return (client as unknown as { _client: object })._client.constructor.name;
}

function createAutoClient(testCase: ResponsesStreamCase): AutoLLMClient {
  return new AutoLLMClient({
    model: testCase.model,
    apiKey: "test-key",
    clientType: testCase.clientType,
  });
}

function keepaliveEvent(sequenceNumber: number): unknown {
  // Heartbeats come from gateways in front of Responses-compatible servers (one-api-style
  // proxies), never from the official API, so the event shape is synthesized from the
  // report in https://github.com/Prism-Shadow/penguin-harness/issues/286.
  return { type: "keepalive", sequence_number: sequenceNumber };
}

function textDeltaEvent(text: string): unknown {
  return { type: "response.output_text.delta", delta: text };
}

function completedEvent(): unknown {
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

async function collectEvents(
  stream: AsyncIterable<UniEvent>,
): Promise<UniEvent[]> {
  const events: UniEvent[] = [];
  for await (const event of stream) {
    events.push(event);
  }
  return events;
}

describe.each(RESPONSES_STREAM_CASES)(
  "Keepalive handling for $clientType",
  (testCase) => {
    test("skips gateway keepalive heartbeats between stream events", async () => {
      const client = createAutoClient(testCase);
      expect(routedClientName(client)).toBe(testCase.expectedClient);
      installFakeResponsesStream(client, [
        keepaliveEvent(1),
        textDeltaEvent("Here is"),
        keepaliveEvent(2),
        textDeltaEvent(" the memo."),
        keepaliveEvent(3),
        completedEvent(),
      ]);

      const events = await collectEvents(
        client.streamingResponse({ messages, config: {} }),
      );
      const texts = events.flatMap((event) =>
        event.content_items
          .filter((item): item is TextContentItem => item.type === "text")
          .map((item) => item.text),
      );
      expect(texts).toEqual(["Here is", " the memo."]);
      expect(events[events.length - 1].finish_reason).toBe("stop");
    });

    test("still rejects genuinely unknown events", async () => {
      const client = createAutoClient(testCase);
      installFakeResponsesStream(client, [
        { type: "response.mystery_event" },
        completedEvent(),
      ]);

      await expect(
        collectEvents(client.streamingResponse({ messages, config: {} })),
      ).rejects.toThrow("Unknown output");
    });
  },
);
