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


import http from "http";
import { AddressInfo } from "net";
import {
  describe,
  test,
  expect,
  beforeAll,
  beforeEach,
  afterAll,
} from "@jest/globals";

import { AutoLLMClient } from "../src";

interface HeaderCase {
  clientType: string;
  model: string;
  baseUrlSuffix: string;
}

// One case per SDK: the OpenAI and Anthropic clients take defaultHeaders directly, while the
// Gemini SDK carries them inside httpOptions.
const HEADER_CASES: HeaderCase[] = [
  { clientType: "openai-chat", model: "gpt-5.6", baseUrlSuffix: "/v1" },
  { clientType: "ant-messages", model: "claude-sonnet-5", baseUrlSuffix: "" },
  { clientType: "gemini-3.7", model: "gemini-3.7-flash", baseUrlSuffix: "" },
];

const extraHeaders = {
  "X-App": "cli",
  "HTTP-Referer": "https://example.test",
};

let server: http.Server;
let origin: string;
// node lowercases incoming header names, which is what makes this comparison safe: the SDKs
// disagree on the casing they send
let receivedHeaders: Record<string, string> = {};

beforeAll(async () => {
  server = http.createServer((req, res) => {
    receivedHeaders = Object.fromEntries(
      Object.entries(req.headers).map(([name, value]) => [
        name.toLowerCase(),
        String(value),
      ]),
    );
    const payload = req.url?.includes("v1beta")
      ? { models: [{ name: "models/m1" }, { name: "models/m2" }] }
      : { object: "list", data: [{ id: "m1" }, { id: "m2" }], has_more: false };
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(payload));
  });
  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", resolve);
  });
  origin = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterAll(async () => {
  await new Promise<void>((resolve) => {
    server.close(() => resolve());
  });
});

beforeEach(() => {
  receivedHeaders = {};
});

describe.each(HEADER_CASES)("defaultHeaders for $clientType", (testCase) => {
  test("reach the endpoint", async () => {
    const client = new AutoLLMClient({
      model: testCase.model,
      apiKey: "test-key",
      baseUrl: origin + testCase.baseUrlSuffix,
      clientType: testCase.clientType,
      defaultHeaders: extraHeaders,
    });

    await expect(client.listModels()).resolves.toEqual(["m1", "m2"]);
    expect(receivedHeaders["x-app"]).toBe("cli");
    expect(receivedHeaders["http-referer"]).toBe("https://example.test");
  });
});

describe("defaultHeaders", () => {
  test("are absent from a request that declares none", async () => {
    const client = new AutoLLMClient({
      model: "gpt-5.6",
      apiKey: "test-key",
      baseUrl: origin + "/v1",
      clientType: "openai-chat",
    });

    await expect(client.listModels()).resolves.toEqual(["m1", "m2"]);
    expect(receivedHeaders["x-app"]).toBeUndefined();
  });
});
