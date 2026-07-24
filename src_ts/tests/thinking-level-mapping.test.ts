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
// (gemini-3-pro also "medium"), image models accept only "minimal" and "high".
// Unsupported levels must clamp to the closest supported one, never error.
const GEMINI3_THINKING_LEVEL_CASES: Array<
  [string, ThinkingLevel, GeminiThinkingLevel]
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
  ["gemini-3-flash-preview", ThinkingLevel.NONE, GeminiThinkingLevel.MINIMAL],
  ["gemini-3.5-flash", ThinkingLevel.MEDIUM, GeminiThinkingLevel.MEDIUM],
];

describe("gemini3 thinking level clamping", () => {
  test.each(GEMINI3_THINKING_LEVEL_CASES)(
    "%s clamps %s to %s",
    (model, level, expected) => {
      const client = new AutoLLMClient({ model, apiKey: "test-key" });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      expect((client as any)._client._convertThinkingLevel(level)).toBe(
        expected,
      );
    },
  );
});
