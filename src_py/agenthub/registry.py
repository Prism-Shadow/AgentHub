# Copyright 2025 Prism Shadow. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal, NotRequired, TypedDict


Modality = Literal["Text", "Image", "Video", "Audio", "Embed"]
Currency = Literal["USD", "CNY"]


class ModelPricing(TypedDict):
    """List prices per million tokens, in the currency requested from list_supported_models.

    ``cached_input`` is the vendor's cache-hit price; ``cache_write`` is only present for
    vendors that bill cache writes separately (e.g. Anthropic at 1.25x input).
    """

    currency: Currency
    input: float
    output: float
    cached_input: NotRequired[float]
    cache_write: NotRequired[float]


class SupportedModel(TypedDict):
    """One supported model entry.

    (model, base_url, client) maps directly onto the AutoLLMClient constructor:
    ``AutoLLMClient(model=entry["model"], base_url=entry["base_url"], client_type=entry["client"])``.
    Modalities describe what is usable through that client; ``context_window`` and
    ``pricing`` are omitted where the platform publishes no authoritative value.
    """

    model: str
    base_url: str
    client: str
    input_modalities: list[Modality]
    output_modalities: list[Modality]
    context_window: NotRequired[int]
    pricing: NotRequired[ModelPricing]


_GOOGLE = "https://generativelanguage.googleapis.com"
_ANTHROPIC = "https://api.anthropic.com"
_OPENAI = "https://api.openai.com/v1"
_ZAI = "https://api.z.ai/api/paas/v4/"
_MOONSHOT = "https://api.moonshot.cn/v1"
_DEEPSEEK = "https://api.deepseek.com"
_OPENROUTER = "https://openrouter.ai/api/v1"
_SILICONFLOW = "https://api.siliconflow.cn/v1"

# Display convention shared with the AgentHub apps: official CNY prices convert at 7 CNY/USD
# so a currency switch shows exactly the vendor's published numbers.
_CNY_PER_USD = 7.0


def _usd(
    input: float, output: float, cached_input: float | None = None, cache_write: float | None = None
) -> ModelPricing:
    pricing: ModelPricing = {"currency": "USD", "input": input, "output": output}
    if cached_input is not None:
        pricing["cached_input"] = cached_input
    if cache_write is not None:
        pricing["cache_write"] = cache_write
    return pricing


def _cny(
    input: float, output: float, cached_input: float | None = None, cache_write: float | None = None
) -> ModelPricing:
    pricing = _usd(input, output, cached_input, cache_write)
    pricing["currency"] = "CNY"
    return pricing


_TEXT: list[Modality] = ["Text"]
_TEXT_IMAGE: list[Modality] = ["Text", "Image"]
_GEMINI_INPUTS: list[Modality] = ["Text", "Image", "Video", "Audio"]

# Entries store pricing in the vendor's official pricing currency; platform data (context
# windows, OpenRouter USD prices) verified against the live /models APIs on 2026-07-22,
# SiliconFlow CNY prices from the vendors' official price lists.
_SUPPORTED_MODELS: list[SupportedModel] = [
    # official vendor endpoints
    {
        "model": "gemini-3.6-flash",
        "base_url": _GOOGLE,
        "client": "gemini-3.6",
        "input_modalities": _GEMINI_INPUTS,
        "output_modalities": _TEXT,
        "context_window": 1048576,
        "pricing": _usd(1.5, 7.5),
    },
    {
        "model": "gemini-3.5-flash-lite",
        "base_url": _GOOGLE,
        "client": "gemini-3.6",
        "input_modalities": _GEMINI_INPUTS,
        "output_modalities": _TEXT,
        "context_window": 1048576,
        "pricing": _usd(0.3, 2.5),
    },
    {
        "model": "gemini-3.5-flash",
        "base_url": _GOOGLE,
        "client": "gemini-3",
        "input_modalities": _GEMINI_INPUTS,
        "output_modalities": _TEXT,
        "context_window": 1048576,
        "pricing": _usd(1.5, 9.0, cached_input=0.15),
    },
    {
        "model": "gemini-3.1-flash-image-preview",
        "base_url": _GOOGLE,
        "client": "gemini-3",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": ["Image"],
    },
    {
        "model": "gemini-3.1-flash-tts-preview",
        "base_url": _GOOGLE,
        "client": "gemini-3",
        "input_modalities": _TEXT,
        "output_modalities": ["Audio"],
    },
    {
        "model": "gemini-embedding-2",
        "base_url": _GOOGLE,
        "client": "gemini-3",
        "input_modalities": _TEXT,
        "output_modalities": ["Embed"],
    },
    {
        "model": "claude-fable-5",
        "base_url": _ANTHROPIC,
        "client": "claude-5",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _usd(10.0, 50.0, cached_input=1.0, cache_write=12.5),
    },
    {
        "model": "claude-sonnet-5",
        "base_url": _ANTHROPIC,
        "client": "claude-5",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _usd(2.0, 10.0, cached_input=0.2, cache_write=2.5),
    },
    {
        "model": "claude-opus-4-8",
        "base_url": _ANTHROPIC,
        "client": "claude-5",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _usd(5.0, 25.0, cached_input=0.5, cache_write=6.25),
    },
    {
        "model": "claude-sonnet-4-6",
        "base_url": _ANTHROPIC,
        "client": "claude-4-6",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _usd(3.0, 15.0, cached_input=0.3, cache_write=3.75),
    },
    {
        "model": "gpt-5.5",
        "base_url": _OPENAI,
        "client": "gpt-5.5",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1050000,
        "pricing": _usd(5.0, 30.0, cached_input=0.5),
    },
    {
        "model": "text-embedding-3-large",
        "base_url": _OPENAI,
        "client": "openai-embedding",
        "input_modalities": _TEXT,
        "output_modalities": ["Embed"],
        "pricing": _usd(0.13, 0.0),
    },
    {
        "model": "glm-5.1",
        "base_url": _ZAI,
        "client": "glm-5.1",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 200000,
        "pricing": _usd(1.4, 4.4, cached_input=0.26),
    },
    {
        "model": "kimi-k3",
        "base_url": _MOONSHOT,
        "client": "kimi-k3",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1048576,
        "pricing": _cny(20.0, 100.0, cached_input=2.0),
    },
    {
        "model": "kimi-k2.6",
        "base_url": _MOONSHOT,
        "client": "kimi-k2.6",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 262144,
        "pricing": _cny(6.5, 27.0, cached_input=1.1),
    },
    {
        "model": "deepseek-v4-flash",
        "base_url": _DEEPSEEK,
        "client": "deepseek-v4",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _cny(1.0, 2.0, cached_input=0.02),
    },
    {
        "model": "deepseek-v4-pro",
        "base_url": _DEEPSEEK,
        "client": "deepseek-v4",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _cny(3.0, 6.0, cached_input=0.025),
    },
    # OpenRouter (USD prices, context windows and modality flags from the live /models API)
    {
        "model": "anthropic/claude-fable-5",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _usd(10.0, 50.0, cached_input=1.0, cache_write=12.5),
    },
    {
        "model": "anthropic/claude-opus-4.8",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _usd(5.0, 25.0, cached_input=0.5, cache_write=6.25),
    },
    {
        "model": "anthropic/claude-opus-4.7",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _usd(5.0, 25.0, cached_input=0.5, cache_write=6.25),
    },
    {
        "model": "anthropic/claude-sonnet-5",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _usd(2.0, 10.0, cached_input=0.2, cache_write=2.5),
    },
    {
        "model": "deepseek/deepseek-v4-flash",
        "base_url": _OPENROUTER,
        "client": "deepseek-v4",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 1048576,
        "pricing": _usd(0.098, 0.196, cached_input=0.0196),
    },
    {
        "model": "deepseek/deepseek-v4-pro",
        "base_url": _OPENROUTER,
        "client": "deepseek-v4",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 1048576,
        "pricing": _usd(0.435, 0.87, cached_input=0.003625),
    },
    {
        "model": "google/gemini-3.5-flash",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1048576,
        "pricing": _usd(1.5, 9.0, cached_input=0.15),
    },
    {
        "model": "minimax/minimax-m3",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1048576,
        "pricing": _usd(0.3, 1.2, cached_input=0.06),
    },
    {
        "model": "moonshotai/kimi-k3",
        "base_url": _OPENROUTER,
        "client": "kimi-k3",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1048576,
        "pricing": _usd(3.0, 15.0, cached_input=0.3),
    },
    {
        "model": "moonshotai/kimi-k2.6",
        "base_url": _OPENROUTER,
        "client": "kimi-k2.6",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 262144,
        "pricing": _usd(0.684, 3.42, cached_input=0.144),
    },
    {
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _usd(0.0, 0.0),
    },
    {
        "model": "openai/gpt-5.6-sol",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1050000,
        "pricing": _usd(5.0, 30.0, cached_input=0.5, cache_write=6.25),
    },
    {
        "model": "openai/gpt-5.6-terra",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1050000,
        "pricing": _usd(2.5, 15.0, cached_input=0.25, cache_write=3.125),
    },
    {
        "model": "openai/gpt-5.5",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1050000,
        "pricing": _usd(5.0, 30.0, cached_input=0.5),
    },
    {
        "model": "qwen/qwen3.6-35b-a3b",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 262144,
        "pricing": _usd(0.14, 1.0),
    },
    {
        "model": "stepfun/step-3.7-flash",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 262144,
        "pricing": _usd(0.2, 1.15, cached_input=0.04),
    },
    {
        "model": "tencent/hy3",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 262144,
        "pricing": _usd(0.14, 0.58, cached_input=0.035),
    },
    {
        "model": "x-ai/grok-4.5",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 500000,
        "pricing": _usd(2.0, 6.0, cached_input=0.3),
    },
    {
        "model": "xiaomi/mimo-v2.5",
        "base_url": _OPENROUTER,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 1050000,
        "pricing": _usd(0.14, 0.28, cached_input=0.0028),
    },
    {
        "model": "z-ai/glm-5.2",
        "base_url": _OPENROUTER,
        "client": "glm-5.1",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 1048576,
        "pricing": _usd(0.8204, 2.5784, cached_input=0.15236),
    },
    {
        "model": "z-ai/glm-5.1",
        "base_url": _OPENROUTER,
        "client": "glm-5.1",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 204800,
        "pricing": _usd(0.966, 3.036, cached_input=0.1794),
    },
    # SiliconFlow (official CNY price list; the platform publishes no pricing API)
    {
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "base_url": _SILICONFLOW,
        "client": "deepseek-v4",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _cny(1.0, 2.0, cached_input=0.02),
    },
    {
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "base_url": _SILICONFLOW,
        "client": "deepseek-v4",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _cny(12.0, 24.0, cached_input=0.1),
    },
    {
        "model": "meituan-longcat/LongCat-2.0",
        "base_url": _SILICONFLOW,
        "client": "openai",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _cny(5.0, 20.0, cached_input=0.1),
    },
    {
        "model": "moonshotai/Kimi-K2.7-Code",
        "base_url": _SILICONFLOW,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 262144,
        "pricing": _cny(6.5, 27.0, cached_input=1.3),
    },
    {
        "model": "zai-org/GLM-5.2",
        "base_url": _SILICONFLOW,
        "client": "glm-5.1",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 1000000,
        "pricing": _cny(8.0, 28.0, cached_input=2.0),
    },
    {
        "model": "Pro/zai-org/GLM-5.1",
        "base_url": _SILICONFLOW,
        "client": "glm-5.1",
        "input_modalities": _TEXT,
        "output_modalities": _TEXT,
        "context_window": 200000,
    },
    {
        "model": "Pro/moonshotai/Kimi-K2.6",
        "base_url": _SILICONFLOW,
        "client": "kimi-k2.6",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 262144,
    },
    {
        "model": "Qwen/Qwen3.6-35B-A3B",
        "base_url": _SILICONFLOW,
        "client": "openai",
        "input_modalities": _TEXT_IMAGE,
        "output_modalities": _TEXT,
        "context_window": 262144,
    },
    {
        "model": "Qwen/Qwen3-Embedding-8B",
        "base_url": _SILICONFLOW,
        "client": "openai-embedding",
        "input_modalities": _TEXT,
        "output_modalities": ["Embed"],
    },
]


def _convert_pricing(pricing: ModelPricing, currency: Currency) -> ModelPricing:
    if pricing["currency"] == currency:
        return dict(pricing)

    rate = _CNY_PER_USD if currency == "CNY" else 1 / _CNY_PER_USD
    converted: ModelPricing = {"currency": currency, "input": 0.0, "output": 0.0}
    for key in ("input", "output", "cached_input", "cache_write"):
        if key in pricing:
            converted[key] = round(pricing[key] * rate, 6)
    return converted


def list_supported_models(currency: Currency = "USD") -> list[SupportedModel]:
    """List supported models with base URL, client, modalities, context window, and pricing.

    Covers the official vendor endpoints plus the OpenRouter and SiliconFlow platforms;
    ``client`` is the ``client_type`` token that routes the model to its protocol client.
    Prices are returned per million tokens in ``currency`` ("USD" or "CNY", converted at
    7 CNY/USD from the vendor's official price list).
    """
    entries: list[SupportedModel] = []
    for entry in _SUPPORTED_MODELS:
        copied = dict(entry)
        copied["input_modalities"] = list(entry["input_modalities"])
        copied["output_modalities"] = list(entry["output_modalities"])
        if "pricing" in entry:
            copied["pricing"] = _convert_pricing(entry["pricing"], currency)
        entries.append(copied)
    return entries
