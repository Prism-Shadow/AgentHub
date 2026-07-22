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

/**
 * One supported model as a (model, base_url, client) triple.
 *
 * The triple maps directly onto the AutoLLMClient constructor:
 * `new AutoLLMClient({ model, baseUrl: base_url, clientType: client })`.
 */
export interface SupportedModel {
  model: string;
  base_url: string;
  client: string;
}

const GOOGLE = "https://generativelanguage.googleapis.com";
const ANTHROPIC = "https://api.anthropic.com";
const OPENAI = "https://api.openai.com/v1";
const ZAI = "https://api.z.ai/api/paas/v4/";
const MOONSHOT = "https://api.moonshot.cn/v1";
const DEEPSEEK = "https://api.deepseek.com";
const OPENROUTER = "https://openrouter.ai/api/v1";
const SILICONFLOW = "https://api.siliconflow.cn/v1";

const SUPPORTED_MODELS: SupportedModel[] = [
  // official vendor endpoints
  { model: "gemini-3.6-flash", base_url: GOOGLE, client: "gemini-3.6" },
  { model: "gemini-3.5-flash-lite", base_url: GOOGLE, client: "gemini-3.6" },
  { model: "gemini-3.5-flash", base_url: GOOGLE, client: "gemini-3" },
  {
    model: "gemini-3.1-flash-image-preview",
    base_url: GOOGLE,
    client: "gemini-3",
  },
  {
    model: "gemini-3.1-flash-tts-preview",
    base_url: GOOGLE,
    client: "gemini-3",
  },
  { model: "gemini-embedding-2", base_url: GOOGLE, client: "gemini-3" },
  { model: "claude-fable-5", base_url: ANTHROPIC, client: "claude-5" },
  { model: "claude-sonnet-5", base_url: ANTHROPIC, client: "claude-5" },
  { model: "claude-opus-4-8", base_url: ANTHROPIC, client: "claude-5" },
  { model: "claude-sonnet-4-6", base_url: ANTHROPIC, client: "claude-4-6" },
  { model: "gpt-5.5", base_url: OPENAI, client: "gpt-5.5" },
  {
    model: "text-embedding-3-large",
    base_url: OPENAI,
    client: "openai-embedding",
  },
  { model: "glm-5.1", base_url: ZAI, client: "glm-5.1" },
  { model: "kimi-k3", base_url: MOONSHOT, client: "kimi-k3" },
  { model: "kimi-k2.6", base_url: MOONSHOT, client: "kimi-k2.6" },
  { model: "deepseek-v4-flash", base_url: DEEPSEEK, client: "deepseek-v4" },
  // OpenRouter
  { model: "z-ai/glm-5.2", base_url: OPENROUTER, client: "glm-5.1" },
  { model: "z-ai/glm-5.1", base_url: OPENROUTER, client: "glm-5.1" },
  { model: "moonshotai/kimi-k3", base_url: OPENROUTER, client: "kimi-k3" },
  { model: "moonshotai/kimi-k2.6", base_url: OPENROUTER, client: "kimi-k2.6" },
  { model: "qwen/qwen3.6-35b-a3b", base_url: OPENROUTER, client: "openai" },
  {
    model: "qwen/qwen3-embedding-4b",
    base_url: OPENROUTER,
    client: "openai-embedding",
  },
  // SiliconFlow
  { model: "zai-org/GLM-5.2", base_url: SILICONFLOW, client: "glm-5.1" },
  { model: "Pro/zai-org/GLM-5.1", base_url: SILICONFLOW, client: "glm-5.1" },
  {
    model: "Pro/moonshotai/Kimi-K2.6",
    base_url: SILICONFLOW,
    client: "kimi-k2.6",
  },
  { model: "Qwen/Qwen3.6-35B-A3B", base_url: SILICONFLOW, client: "openai" },
  {
    model: "Qwen/Qwen3-Embedding-8B",
    base_url: SILICONFLOW,
    client: "openai-embedding",
  },
];

/**
 * List supported models as (model, base_url, client) triples.
 *
 * Covers the official vendor endpoints plus the OpenRouter and SiliconFlow
 * platforms; `client` is the `clientType` token that routes the model to its
 * protocol client.
 */
export function listSupportedModels(): SupportedModel[] {
  return SUPPORTED_MODELS.map((entry) => ({ ...entry }));
}
