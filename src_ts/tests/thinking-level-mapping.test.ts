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
import { ThinkingLevel as GeminiThinkingLevel } from "@google/genai";
import { AutoLLMClient, ThinkingLevel } from "../src";

// Not every Gemini model accepts every thinking level (verified live 2026-07-24;
// see llmsdk_docs/gemini3/docs/thinking.md): pro models reject "minimal"
// (gemini-3-pro also "medium"), image models accept only "minimal" and "high",
// and the 2.5 series rejects the thinking_level parameter outright. Unsupported
// levels must clamp to the closest supported one — or be dropped entirely for
// models that take none — never error.
const GEMINI3_THINKING_LEVEL_CASES: Array<
  [string, ThinkingLevel, GeminiThinkingLevel | undefined]
> = [
  ["gemini-3.1-pro-preview", ThinkingLevel.NONE, GeminiThinkingLevel.LOW],
  ["gemini-3.1-pro-preview", ThinkingLevel.LOW, GeminiThinkingLevel.LOW],
  ["gemini-3.1-pro-preview", ThinkingLevel.MEDIUM, GeminiThinkingLevel.MEDIUM],
  ["gemini-3.1-pro-preview", ThinkingLevel.HIGH, GeminiThinkingLevel.HIGH],
  ["gemini-3.1-pro-preview", ThinkingLevel.XHIGH, GeminiThinkingLevel.HIGH],
  ["gemini-3-pro-preview", ThinkingLevel.NONE, GeminiThinkingLevel.LOW],
  ["gemini-3-pro-preview", ThinkingLevel.MEDIUM, GeminiThinkingLevel.HIGH],
  ["gemini-3.1-flash-image", ThinkingLevel.NONE, GeminiThinkingLevel.MINIMAL],
  ["gemini-3.1-flash-image", ThinkingLevel.LOW, GeminiThinkingLevel.MINIMAL],
  ["gemini-3.1-flash-image", ThinkingLevel.MEDIUM, GeminiThinkingLevel.HIGH],
  // "-image" wins over "gemini-3-pro" (LOW would stay LOW under the pro set).
  ["gemini-3-pro-image", ThinkingLevel.LOW, GeminiThinkingLevel.MINIMAL],
  ["gemini-3-flash-preview", ThinkingLevel.NONE, GeminiThinkingLevel.MINIMAL],
  ["gemini-3.5-flash", ThinkingLevel.MEDIUM, GeminiThinkingLevel.MEDIUM],
  // The 2.5 series rejects thinking_level for every value: drop the parameter.
  ["gemini-2.5-pro", ThinkingLevel.NONE, undefined],
  ["gemini-2.5-flash", ThinkingLevel.HIGH, undefined],
  ["gemini-2.5-flash-lite", ThinkingLevel.LOW, undefined],
  // A future pro generation falls into the generic "-pro" branch.
  ["gemini-4-pro", ThinkingLevel.NONE, GeminiThinkingLevel.LOW],
  // An unrecognized model inherits the full four-level default.
  ["gemini-9-flash", ThinkingLevel.NONE, GeminiThinkingLevel.MINIMAL],
];

// clientType pins routing so pre-3 and hypothetical model names reach
// Gemini3Client the same way an explicit override would in user code.
function createGemini3AutoClient(model: string): AutoLLMClient {
  return new AutoLLMClient({
    model,
    apiKey: "test-key",
    clientType: "gemini-3",
  });
}

describe("gemini3 thinking level clamping", () => {
  test.each(GEMINI3_THINKING_LEVEL_CASES)(
    "%s clamps %s to %s",
    (model, level, expected) => {
      const client = createGemini3AutoClient(model);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      expect((client as any)._client._convertThinkingLevel(level)).toBe(
        expected,
      );
    },
  );

  test("thinking config carries the clamped level", () => {
    const client = createGemini3AutoClient("gemini-3.1-pro-preview");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const config = (client as any)._client.transformUniConfigToModelConfig({
      thinking_level: ThinkingLevel.NONE,
    });
    expect(config.thinkingConfig.thinkingLevel).toBe(GeminiThinkingLevel.LOW);
  });

  test("thinking config omits the level for pre-3 models", () => {
    const client = createGemini3AutoClient("gemini-2.5-flash");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const config = (client as any)._client.transformUniConfigToModelConfig({
      thinking_level: ThinkingLevel.HIGH,
    });
    expect(config.thinkingConfig.thinkingLevel).toBeUndefined();
  });
});
