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

export type Modality = "Text" | "Image" | "Video" | "Audio" | "Embed";
export type Currency = "USD" | "CNY";

/**
 * List prices per million tokens, in the currency requested from
 * listSupportedModels. `cached_input` is the vendor's cache-hit price;
 * `cache_write` is only present for vendors that bill cache writes separately
 * (e.g. Anthropic at 1.25x input).
 */
export interface ModelPricing {
  currency: Currency;
  input: number;
  output: number;
  cached_input?: number;
  cache_write?: number;
}

/**
 * One supported model entry.
 *
 * (model, base_url, client) maps directly onto the AutoLLMClient constructor:
 * `new AutoLLMClient({ model, baseUrl: base_url, clientType: client })`.
 * Modalities describe what is usable through that client; `context_window` and
 * `pricing` are omitted where the platform publishes no authoritative value.
 */
export interface SupportedModel {
  model: string;
  base_url: string;
  client: string;
  input_modalities: Modality[];
  output_modalities: Modality[];
  context_window?: number;
  pricing?: ModelPricing;
}

const GOOGLE = "https://generativelanguage.googleapis.com";
const ANTHROPIC = "https://api.anthropic.com";
const OPENAI = "https://api.openai.com/v1";
const ZAI = "https://api.z.ai/api/paas/v4/";
const MOONSHOT = "https://api.moonshot.cn/v1";
const DEEPSEEK = "https://api.deepseek.com";
const OPENROUTER = "https://openrouter.ai/api/v1";
const SILICONFLOW = "https://api.siliconflow.cn/v1";

// Display convention shared with the AgentHub apps: official CNY prices convert
// at 7 CNY/USD so a currency switch shows exactly the vendor's published numbers.
const CNY_PER_USD = 7.0;

function usd(
  input: number,
  output: number,
  cachedInput?: number,
  cacheWrite?: number,
): ModelPricing {
  return {
    currency: "USD",
    input,
    output,
    ...(cachedInput !== undefined ? { cached_input: cachedInput } : {}),
    ...(cacheWrite !== undefined ? { cache_write: cacheWrite } : {}),
  };
}

function cny(
  input: number,
  output: number,
  cachedInput?: number,
  cacheWrite?: number,
): ModelPricing {
  return { ...usd(input, output, cachedInput, cacheWrite), currency: "CNY" };
}

const TEXT: Modality[] = ["Text"];
const TEXT_IMAGE: Modality[] = ["Text", "Image"];
const GEMINI_INPUTS: Modality[] = ["Text", "Image", "Video", "Audio"];

// Entries store pricing in the vendor's official pricing currency; platform data
// (context windows, OpenRouter USD prices) verified against the live /models APIs
// on 2026-07-22, SiliconFlow CNY prices from the vendors' official price lists.
const SUPPORTED_MODELS: SupportedModel[] = [
  // official vendor endpoints
  {
    model: "gemini-3.6-flash",
    base_url: GOOGLE,
    client: "gemini-3.6",
    input_modalities: GEMINI_INPUTS,
    output_modalities: TEXT,
    context_window: 1048576,
    pricing: usd(1.5, 7.5),
  },
  {
    model: "gemini-3.5-flash-lite",
    base_url: GOOGLE,
    client: "gemini-3.6",
    input_modalities: GEMINI_INPUTS,
    output_modalities: TEXT,
    context_window: 1048576,
    pricing: usd(0.3, 2.5),
  },
  {
    model: "gemini-3.5-flash",
    base_url: GOOGLE,
    client: "gemini-3",
    input_modalities: GEMINI_INPUTS,
    output_modalities: TEXT,
    context_window: 1048576,
    pricing: usd(1.5, 9.0, 0.15),
  },
  {
    model: "gemini-3.1-flash-image-preview",
    base_url: GOOGLE,
    client: "gemini-3",
    input_modalities: TEXT_IMAGE,
    output_modalities: ["Image"],
  },
  {
    model: "gemini-3.1-flash-tts-preview",
    base_url: GOOGLE,
    client: "gemini-3",
    input_modalities: TEXT,
    output_modalities: ["Audio"],
  },
  {
    model: "gemini-embedding-2",
    base_url: GOOGLE,
    client: "gemini-3",
    input_modalities: TEXT,
    output_modalities: ["Embed"],
  },
  {
    model: "claude-fable-5",
    base_url: ANTHROPIC,
    client: "claude-5",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: usd(10.0, 50.0, 1.0, 12.5),
  },
  {
    model: "claude-sonnet-5",
    base_url: ANTHROPIC,
    client: "claude-5",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: usd(2.0, 10.0, 0.2, 2.5),
  },
  {
    model: "claude-opus-4-8",
    base_url: ANTHROPIC,
    client: "claude-5",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: usd(5.0, 25.0, 0.5, 6.25),
  },
  {
    model: "claude-sonnet-4-6",
    base_url: ANTHROPIC,
    client: "claude-4-6",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: usd(3.0, 15.0, 0.3, 3.75),
  },
  {
    model: "gpt-5.5",
    base_url: OPENAI,
    client: "gpt-5.5",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1050000,
    pricing: usd(5.0, 30.0, 0.5),
  },
  {
    model: "text-embedding-3-large",
    base_url: OPENAI,
    client: "openai-embedding",
    input_modalities: TEXT,
    output_modalities: ["Embed"],
    pricing: usd(0.13, 0.0),
  },
  {
    model: "glm-5.1",
    base_url: ZAI,
    client: "glm-5.1",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 200000,
    pricing: usd(1.4, 4.4, 0.26),
  },
  {
    model: "kimi-k3",
    base_url: MOONSHOT,
    client: "kimi-k3",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1048576,
    pricing: cny(20.0, 100.0, 2.0),
  },
  {
    model: "kimi-k2.6",
    base_url: MOONSHOT,
    client: "kimi-k2.6",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 262144,
    pricing: cny(6.5, 27.0, 1.1),
  },
  {
    model: "deepseek-v4-flash",
    base_url: DEEPSEEK,
    client: "deepseek-v4",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: cny(1.0, 2.0, 0.02),
  },
  {
    model: "deepseek-v4-pro",
    base_url: DEEPSEEK,
    client: "deepseek-v4",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: cny(3.0, 6.0, 0.025),
  },
  // OpenRouter (USD prices, context windows and modality flags from the live /models API)
  {
    model: "anthropic/claude-fable-5",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: usd(10.0, 50.0, 1.0, 12.5),
  },
  {
    model: "anthropic/claude-opus-4.8",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: usd(5.0, 25.0, 0.5, 6.25),
  },
  {
    model: "anthropic/claude-opus-4.7",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: usd(5.0, 25.0, 0.5, 6.25),
  },
  {
    model: "anthropic/claude-sonnet-5",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: usd(2.0, 10.0, 0.2, 2.5),
  },
  {
    model: "deepseek/deepseek-v4-flash",
    base_url: OPENROUTER,
    client: "deepseek-v4",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 1048576,
    pricing: usd(0.098, 0.196, 0.0196),
  },
  {
    model: "deepseek/deepseek-v4-pro",
    base_url: OPENROUTER,
    client: "deepseek-v4",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 1048576,
    pricing: usd(0.435, 0.87, 0.003625),
  },
  {
    model: "google/gemini-3.5-flash",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1048576,
    pricing: usd(1.5, 9.0, 0.15),
  },
  {
    model: "minimax/minimax-m3",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1048576,
    pricing: usd(0.3, 1.2, 0.06),
  },
  {
    model: "moonshotai/kimi-k3",
    base_url: OPENROUTER,
    client: "kimi-k3",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1048576,
    pricing: usd(3.0, 15.0, 0.3),
  },
  {
    model: "moonshotai/kimi-k2.6",
    base_url: OPENROUTER,
    client: "kimi-k2.6",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 262144,
    pricing: usd(0.684, 3.42, 0.144),
  },
  {
    model: "nvidia/nemotron-3-ultra-550b-a55b:free",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: usd(0.0, 0.0),
  },
  {
    model: "openai/gpt-5.6-sol",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1050000,
    pricing: usd(5.0, 30.0, 0.5, 6.25),
  },
  {
    model: "openai/gpt-5.6-terra",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1050000,
    pricing: usd(2.5, 15.0, 0.25, 3.125),
  },
  {
    model: "openai/gpt-5.5",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1050000,
    pricing: usd(5.0, 30.0, 0.5),
  },
  {
    model: "qwen/qwen3.6-35b-a3b",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 262144,
    pricing: usd(0.14, 1.0),
  },
  {
    model: "stepfun/step-3.7-flash",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 262144,
    pricing: usd(0.2, 1.15, 0.04),
  },
  {
    model: "tencent/hy3",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 262144,
    pricing: usd(0.14, 0.58, 0.035),
  },
  {
    model: "x-ai/grok-4.5",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 500000,
    pricing: usd(2.0, 6.0, 0.3),
  },
  {
    model: "xiaomi/mimo-v2.5",
    base_url: OPENROUTER,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 1050000,
    pricing: usd(0.14, 0.28, 0.0028),
  },
  {
    model: "z-ai/glm-5.2",
    base_url: OPENROUTER,
    client: "glm-5.1",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 1048576,
    pricing: usd(0.8204, 2.5784, 0.15236),
  },
  {
    model: "z-ai/glm-5.1",
    base_url: OPENROUTER,
    client: "glm-5.1",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 204800,
    pricing: usd(0.966, 3.036, 0.1794),
  },
  // SiliconFlow (official CNY price list; the platform publishes no pricing API)
  {
    model: "deepseek-ai/DeepSeek-V4-Flash",
    base_url: SILICONFLOW,
    client: "deepseek-v4",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: cny(1.0, 2.0, 0.02),
  },
  {
    model: "deepseek-ai/DeepSeek-V4-Pro",
    base_url: SILICONFLOW,
    client: "deepseek-v4",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: cny(12.0, 24.0, 0.1),
  },
  {
    model: "meituan-longcat/LongCat-2.0",
    base_url: SILICONFLOW,
    client: "openai",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: cny(5.0, 20.0, 0.1),
  },
  {
    model: "moonshotai/Kimi-K2.7-Code",
    base_url: SILICONFLOW,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 262144,
    pricing: cny(6.5, 27.0, 1.3),
  },
  {
    model: "zai-org/GLM-5.2",
    base_url: SILICONFLOW,
    client: "glm-5.1",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 1000000,
    pricing: cny(8.0, 28.0, 2.0),
  },
  {
    model: "Pro/zai-org/GLM-5.1",
    base_url: SILICONFLOW,
    client: "glm-5.1",
    input_modalities: TEXT,
    output_modalities: TEXT,
    context_window: 200000,
  },
  {
    model: "Pro/moonshotai/Kimi-K2.6",
    base_url: SILICONFLOW,
    client: "kimi-k2.6",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 262144,
  },
  {
    model: "Qwen/Qwen3.6-35B-A3B",
    base_url: SILICONFLOW,
    client: "openai",
    input_modalities: TEXT_IMAGE,
    output_modalities: TEXT,
    context_window: 262144,
  },
  {
    model: "Qwen/Qwen3-Embedding-8B",
    base_url: SILICONFLOW,
    client: "openai-embedding",
    input_modalities: TEXT,
    output_modalities: ["Embed"],
  },
];

function convertPricing(pricing: ModelPricing, currency: Currency): ModelPricing {
  if (pricing.currency === currency) {
    return { ...pricing };
  }

  const rate = currency === "CNY" ? CNY_PER_USD : 1 / CNY_PER_USD;
  const round = (v: number): number => Math.round(v * rate * 1e6) / 1e6;
  return {
    currency,
    input: round(pricing.input),
    output: round(pricing.output),
    ...(pricing.cached_input !== undefined
      ? { cached_input: round(pricing.cached_input) }
      : {}),
    ...(pricing.cache_write !== undefined
      ? { cache_write: round(pricing.cache_write) }
      : {}),
  };
}

/**
 * List supported models with base URL, client, modalities, context window, and
 * pricing.
 *
 * Covers the official vendor endpoints plus the OpenRouter and SiliconFlow
 * platforms; `client` is the `clientType` token that routes the model to its
 * protocol client. Prices are returned per million tokens in `currency` ("USD"
 * or "CNY", converted at 7 CNY/USD from the vendor's official price list).
 */
export function listSupportedModels(currency: Currency = "USD"): SupportedModel[] {
  return SUPPORTED_MODELS.map((entry) => ({
    ...entry,
    input_modalities: [...entry.input_modalities],
    output_modalities: [...entry.output_modalities],
    ...(entry.pricing ? { pricing: convertPricing(entry.pricing, currency) } : {}),
  }));
}
