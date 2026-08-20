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

import { AutoLLMClient, UnsupportedOperationError } from "../src";

interface ListCase {
  expectedClient: string;
  model: string;
  clientType: string;
}

// Every client whose SDK exposes a models.list() endpoint returning ids.
const SDK_LIST_CASES: ListCase[] = [
  { expectedClient: "GPT5_6Client", model: "gpt-5.6", clientType: "gpt-5.6" },
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
  { expectedClient: "GLM5_3Client", model: "glm-5.3", clientType: "glm-5.3" },
  { expectedClient: "KimiK3Client", model: "kimi-k3", clientType: "kimi-k3" },
  {
    expectedClient: "OpenaiEmbeddingClient",
    model: "qwen3-embedding",
    clientType: "openai-embedding",
  },
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

// What the endpoint serves is whatever it says it serves: a gateway fronting several
// vendors answers with all of them, in its own order.
const servedIds = ["gpt-5.6", "claude-sonnet-5", "deepseek-v4"];

function asyncIterable(items: unknown[]): AsyncIterable<unknown> {
  return {
    async *[Symbol.asyncIterator]() {
      for (const item of items) {
        yield item;
      }
    },
  };
}

function installFakeModels(client: AutoLLMClient, fakeClient: unknown): void {
  const routedClient = (client as unknown as { _client: { _client: unknown } })
    ._client;
  routedClient._client = fakeClient;
}

function routedClientName(client: AutoLLMClient): string {
  return (client as unknown as { _client: object })._client.constructor.name;
}

describe.each(SDK_LIST_CASES)("listModels for $clientType", (testCase) => {
  test("returns the ids the endpoint serves", async () => {
    const client = new AutoLLMClient({
      model: testCase.model,
      apiKey: "test-key",
      clientType: testCase.clientType,
    });
    expect(routedClientName(client)).toBe(testCase.expectedClient);
    installFakeModels(client, {
      models: { list: () => asyncIterable(servedIds.map((id) => ({ id }))) },
    });

    await expect(client.listModels()).resolves.toEqual(servedIds);
  });
});

describe("listModels", () => {
  test("the Gemini client strips the path from model names", async () => {
    const client = new AutoLLMClient({
      model: "gemini-3.7-flash",
      apiKey: "test-key",
      clientType: "gemini-3.7",
    });
    expect(routedClientName(client)).toBe("Gemini3_7Client");
    installFakeModels(client, {
      models: {
        list: async () =>
          asyncIterable(
            [
              "models/gemini-3.7-flash",
              "publishers/google/models/gemini-3.7-pro",
            ].map((name) => ({ name })),
          ),
      },
    });

    await expect(client.listModels()).resolves.toEqual([
      "gemini-3.7-flash",
      "gemini-3.7-pro",
    ]);
  });

  test("the Claude client reports that Bedrock cannot list models", async () => {
    const client = new AutoLLMClient({
      model: "claude-sonnet-5",
      apiKey: "access-key,secret-key",
      baseUrl: "bedrock://us-east-1",
    });

    await expect(client.listModels()).rejects.toThrow(
      UnsupportedOperationError,
    );
  });
});
