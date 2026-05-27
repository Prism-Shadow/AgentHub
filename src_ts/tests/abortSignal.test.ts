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

import { expect, afterEach, describe, jest, test } from "@jest/globals";
import { AutoLLMClient } from "../src/autoClient";
import { LLMClient } from "../src/baseClient";
import { Claude4_6Client } from "../src/claude4_6";
import { Claude4_7Client } from "../src/claude4_7";
import { DeepSeekV4Client } from "../src/deepseek_v4";
import { Gemini3Client } from "../src/gemini3";
import { GLM5_1Client } from "../src/glm5_1";
import { GPT5_5Client } from "../src/gpt5_5";
import { KimiK2_6Client } from "../src/kimi_k2_6";
import { Qwen3_6Client } from "../src/qwen3_6";
import type { UniConfig, UniEvent, UniMessage } from "../src/types";

const TEXT_MESSAGE: UniMessage = {
  role: "user",
  content_items: [{ type: "text", text: "hello" }],
};

const IMAGE_MESSAGE: UniMessage = {
  role: "user",
  content_items: [
    {
      type: "image_url",
      image_url: "https://example.test/image.png",
    },
  ],
};

const STOP_EVENT: UniEvent = {
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

const originalFetch = global.fetch;

type AbortableStreamingClient = {
  _streamingResponseInternal(options: {
    messages: UniMessage[];
    config: UniConfig;
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent>;
};

type AbortableTransformClient = {
  transformUniMessageToModelInput(
    messages: UniMessage[],
    signal?: AbortSignal,
  ): unknown;
};

class RecordingClient extends LLMClient {
  transformSignal: AbortSignal | undefined;
  internalSignal: AbortSignal | undefined;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformUniConfigToModelConfig(config: UniConfig): any {
    return config;
  }

  transformUniMessageToModelInput(
    messages: UniMessage[],
    signal?: AbortSignal,
  ): UniMessage[] {
    this.transformSignal = signal;
    return messages;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformModelOutputToUniEvent(_modelOutput: any): UniEvent {
    return { ...STOP_EVENT };
  }

  async *_streamingResponseInternal(options: {
    messages: UniMessage[];
    config: UniConfig;
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent> {
    this.internalSignal = options.signal;
    yield { ...STOP_EVENT };
  }
}

async function collectEvents(
  iterable: AsyncIterable<UniEvent>,
): Promise<UniEvent[]> {
  const events: UniEvent[] = [];
  for await (const event of iterable) {
    events.push(event);
  }
  return events;
}

async function* emptyStream(): AsyncGenerator<never> {
  return;
}

function setPrivateClient(client: object, replacement: unknown): void {
  (client as unknown as { _client: unknown })._client = replacement;
}

function createFetchMock(): ReturnType<typeof jest.fn> {
  const fetchMock = jest.fn(async () => {
    return new Response(Buffer.from("image"), {
      status: 200,
      headers: { "content-type": "image/png" },
    });
  });
  global.fetch = fetchMock as unknown as typeof fetch;
  return fetchMock;
}

function getMockCallArgument(
  mockFn: ReturnType<typeof jest.fn>,
  callIndex: number,
  argumentIndex: number,
): unknown {
  return (mockFn.mock.calls as unknown[][])[callIndex][argumentIndex];
}

afterEach(() => {
  global.fetch = originalFetch;
  jest.restoreAllMocks();
});

test("LLMClient passes AbortSignal to stateless streaming internals", async () => {
  const client = new RecordingClient();
  const signal = new AbortController().signal;

  await collectEvents(
    client.streamingResponse({
      messages: [TEXT_MESSAGE],
      config: {},
      signal,
    }),
  );

  expect(client.internalSignal).toBe(signal);
});

test("LLMClient passes AbortSignal to stateful streaming internals", async () => {
  const client = new RecordingClient();
  const signal = new AbortController().signal;

  await collectEvents(
    client.streamingResponseStateful({
      message: TEXT_MESSAGE,
      config: {},
      signal,
    }),
  );

  expect(client.internalSignal).toBe(signal);
});

test("AutoLLMClient forwards AbortSignal to delegated client", async () => {
  const client = new AutoLLMClient({
    model: "qwen3.6",
    apiKey: "test-key",
  });
  const innerClient = new RecordingClient();
  const signal = new AbortController().signal;
  setPrivateClient(client, innerClient);

  client.transformUniMessageToModelInput([TEXT_MESSAGE], signal);
  await collectEvents(
    client.streamingResponse({
      messages: [TEXT_MESSAGE],
      config: {},
      signal,
    }),
  );
  await collectEvents(
    client.streamingResponseStateful({
      message: TEXT_MESSAGE,
      config: {},
      signal,
    }),
  );

  expect(innerClient.transformSignal).toBe(signal);
  expect(innerClient.internalSignal).toBe(signal);
});

describe("provider request cancellation", () => {
  const openAiCompatibleCases: Array<[string, () => AbortableStreamingClient]> =
    [
      [
        "DeepSeekV4Client",
        () =>
          new DeepSeekV4Client({ model: "deepseek-v4-chat", apiKey: "test" }),
      ],
      [
        "GLM5_1Client",
        () => new GLM5_1Client({ model: "glm-5.1", apiKey: "test" }),
      ],
      [
        "KimiK2_6Client",
        () => new KimiK2_6Client({ model: "kimi-k2.6", apiKey: "test" }),
      ],
      [
        "Qwen3_6Client",
        () => new Qwen3_6Client({ model: "qwen3.6", apiKey: "test" }),
      ],
    ];

  test.each(openAiCompatibleCases)(
    "%s passes AbortSignal to chat completions",
    async (_name, createClient) => {
      const client = createClient();
      const signal = new AbortController().signal;
      const create = jest.fn(async () => emptyStream());
      setPrivateClient(client, {
        baseURL: "https://example.test/v1",
        chat: { completions: { create } },
      });

      await collectEvents(
        client._streamingResponseInternal({
          messages: [TEXT_MESSAGE],
          config: {},
          signal,
        }),
      );

      expect(create).toHaveBeenCalledTimes(1);
      expect(getMockCallArgument(create, 0, 1)).toEqual({ signal });
    },
  );

  test("GPT5_5Client passes AbortSignal to responses", async () => {
    const client: AbortableStreamingClient = new GPT5_5Client({
      model: "gpt-5.5",
      apiKey: "test",
    });
    const signal = new AbortController().signal;
    const create = jest.fn(async () => emptyStream());
    setPrivateClient(client, {
      responses: { create },
    });

    await collectEvents(
      client._streamingResponseInternal({
        messages: [TEXT_MESSAGE],
        config: {},
        signal,
      }),
    );

    expect(create).toHaveBeenCalledTimes(1);
    expect(getMockCallArgument(create, 0, 1)).toEqual({ signal });
  });

  test.each([
    [
      "Claude4_6Client",
      () => new Claude4_6Client({ model: "claude-sonnet-4-6", apiKey: "test" }),
    ],
    [
      "Claude4_7Client",
      () => new Claude4_7Client({ model: "claude-sonnet-4-7", apiKey: "test" }),
    ],
  ])("%s passes AbortSignal to messages", async (_name, createClient) => {
    const client: AbortableStreamingClient = createClient();
    const signal = new AbortController().signal;
    const create = jest.fn(async () => emptyStream());
    setPrivateClient(client, {
      beta: { messages: { create } },
    });

    await collectEvents(
      client._streamingResponseInternal({
        messages: [TEXT_MESSAGE],
        config: {},
        signal,
      }),
    );

    expect(create).toHaveBeenCalledTimes(1);
    expect(getMockCallArgument(create, 0, 1)).toEqual({ signal });
  });

  test("Gemini3Client passes AbortSignal to generateContentStream", async () => {
    const client: AbortableStreamingClient = new Gemini3Client({
      model: "gemini-3-flash-preview",
      apiKey: "test",
    });
    const signal = new AbortController().signal;
    const generateContentStream = jest.fn(async () => emptyStream());
    setPrivateClient(client, {
      models: { generateContentStream },
    });

    await collectEvents(
      client._streamingResponseInternal({
        messages: [TEXT_MESSAGE],
        config: {},
        signal,
      }),
    );

    expect(generateContentStream).toHaveBeenCalledTimes(1);
    expect(getMockCallArgument(generateContentStream, 0, 0)).toEqual(
      expect.objectContaining({
        config: expect.objectContaining({ abortSignal: signal }),
      }),
    );
  });

  test("Gemini3Client passes AbortSignal to embedContent", async () => {
    const client: AbortableStreamingClient = new Gemini3Client({
      model: "gemini-embedding-001",
      apiKey: "test",
    });
    const signal = new AbortController().signal;
    const embedContent = jest.fn(async () => ({
      embeddings: [],
      metadata: { billableCharacterCount: 1 },
    }));
    setPrivateClient(client, {
      models: { embedContent },
    });

    await collectEvents(
      client._streamingResponseInternal({
        messages: [TEXT_MESSAGE],
        config: {},
        signal,
      }),
    );

    expect(embedContent).toHaveBeenCalledTimes(1);
    expect(getMockCallArgument(embedContent, 0, 0)).toEqual(
      expect.objectContaining({
        config: expect.objectContaining({ abortSignal: signal }),
      }),
    );
  });
});

describe("provider image fetch cancellation", () => {
  const transformCases: Array<[string, () => AbortableTransformClient]> = [
    [
      "Claude4_6Client Bedrock",
      () =>
        new Claude4_6Client({
          model: "global.anthropic.claude-sonnet-4-6",
          apiKey: "access,secret",
          baseUrl: "bedrock://us-east-1",
        }),
    ],
    [
      "Claude4_7Client Bedrock",
      () =>
        new Claude4_7Client({
          model: "global.anthropic.claude-sonnet-4-7",
          apiKey: "access,secret",
          baseUrl: "bedrock://us-east-1",
        }),
    ],
    [
      "Gemini3Client",
      () =>
        new Gemini3Client({ model: "gemini-3-flash-preview", apiKey: "test" }),
    ],
    [
      "KimiK2_6Client",
      () => new KimiK2_6Client({ model: "kimi-k2.6", apiKey: "test" }),
    ],
    [
      "Qwen3_6Client",
      () => new Qwen3_6Client({ model: "qwen3.6", apiKey: "test" }),
    ],
  ];

  test.each(transformCases)(
    "%s passes AbortSignal to image fetches",
    async (_name, createClient) => {
      const fetchMock = createFetchMock();
      const signal = new AbortController().signal;
      const client = createClient();

      await client.transformUniMessageToModelInput([IMAGE_MESSAGE], signal);

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(getMockCallArgument(fetchMock, 0, 1)).toEqual({ signal });
    },
  );
});
