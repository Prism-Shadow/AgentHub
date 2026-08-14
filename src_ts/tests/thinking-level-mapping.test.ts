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
// the unified Gemini3_7Client the same way an explicit override would in user code.
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

// The 3.7 generation drops "minimal" (verified live 2026-08-13; see
// llmsdk_docs/gemini3_7/docs/thinking.md); the 3.6-generation models routed to
// the same client keep the full four-level set.
const GEMINI3_7_THINKING_LEVEL_CASES: Array<
  [string, ThinkingLevel, GeminiThinkingLevel]
> = [
  ["gemini-3.7-flash", ThinkingLevel.NONE, GeminiThinkingLevel.LOW],
  ["gemini-3.7-flash", ThinkingLevel.LOW, GeminiThinkingLevel.LOW],
  ["gemini-3.7-flash", ThinkingLevel.MEDIUM, GeminiThinkingLevel.MEDIUM],
  ["gemini-3.7-flash", ThinkingLevel.HIGH, GeminiThinkingLevel.HIGH],
  ["gemini-3.7-flash", ThinkingLevel.XHIGH, GeminiThinkingLevel.HIGH],
  ["gemini-3.6-flash", ThinkingLevel.NONE, GeminiThinkingLevel.MINIMAL],
  ["gemini-3.5-flash-lite", ThinkingLevel.NONE, GeminiThinkingLevel.MINIMAL],
];

describe("gemini3_7 thinking level clamping", () => {
  test.each(GEMINI3_7_THINKING_LEVEL_CASES)(
    "%s clamps %s to %s",
    (model, level, expected) => {
      // These are real model ids, so automatic routing reaches Gemini3_7Client directly.
      const client = new AutoLLMClient({ model, apiKey: "test-key" });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      expect((client as any)._client.constructor.name).toBe("Gemini3_7Client");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      expect((client as any)._client._convertThinkingLevel(level)).toBe(
        expected,
      );
    },
  );
});

// GLM-5.3 cannot disable thinking and accepts only low/high/max reasoning_effort
// (llmsdk_docs/glm5_3/docs/thinking.md, docs-only: the API is not yet live); GLM-5.2
// keeps the full pass-through vocabulary and pre-5.2 models take no effort parameter.
const GLM_THINKING_LEVEL_CASES: Array<
  [string, ThinkingLevel, string, string | undefined]
> = [
  ["glm-5.3", ThinkingLevel.NONE, "enabled", "low"],
  ["glm-5.3", ThinkingLevel.LOW, "enabled", "low"],
  ["glm-5.3", ThinkingLevel.MEDIUM, "enabled", "high"],
  ["glm-5.3", ThinkingLevel.HIGH, "enabled", "high"],
  ["glm-5.3", ThinkingLevel.XHIGH, "enabled", "max"],
  ["glm-5.2", ThinkingLevel.NONE, "disabled", undefined],
  ["glm-5.2", ThinkingLevel.MEDIUM, "enabled", "medium"],
  ["glm-5.2", ThinkingLevel.XHIGH, "enabled", "xhigh"],
  ["glm-5.1", ThinkingLevel.HIGH, "enabled", undefined],
  // Provider-hosted ids keep their own casing (SiliconFlow), so generation
  // detection must be case-insensitive.
  ["zai-org/GLM-5.2", ThinkingLevel.XHIGH, "enabled", "xhigh"],
  ["Pro/zai-org/GLM-5.1", ThinkingLevel.HIGH, "enabled", undefined],
];

describe("glm thinking level mapping", () => {
  test.each(GLM_THINKING_LEVEL_CASES)(
    "%s maps %s to thinking=%s effort=%s",
    (model, level, thinkingType, effort) => {
      const client = new AutoLLMClient({ model, apiKey: "test-key" });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      expect((client as any)._client.constructor.name).toBe("GLM5_3Client");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const config = (client as any)._client.transformUniConfigToModelConfig({
        thinking_level: level,
      });
      expect(config.extra_body.thinking.type).toBe(thinkingType);
      expect(config.reasoning_effort).toBe(effort);
    },
  );
});
