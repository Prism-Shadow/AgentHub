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

from typing import TypedDict


class SupportedModel(TypedDict):
    """One supported model as a (model, base_url, client) triple.

    The triple maps directly onto the AutoLLMClient constructor:
    ``AutoLLMClient(model=entry["model"], base_url=entry["base_url"], client_type=entry["client"])``.
    """

    model: str
    base_url: str
    client: str


def _entry(model: str, base_url: str, client: str) -> SupportedModel:
    return {"model": model, "base_url": base_url, "client": client}


_GOOGLE = "https://generativelanguage.googleapis.com"
_ANTHROPIC = "https://api.anthropic.com"
_OPENAI = "https://api.openai.com/v1"
_ZAI = "https://api.z.ai/api/paas/v4/"
_MOONSHOT = "https://api.moonshot.cn/v1"
_DEEPSEEK = "https://api.deepseek.com"
_OPENROUTER = "https://openrouter.ai/api/v1"
_SILICONFLOW = "https://api.siliconflow.cn/v1"

_SUPPORTED_MODELS: list[SupportedModel] = [
    # official vendor endpoints
    _entry("gemini-3.6-flash", _GOOGLE, "gemini-3.6"),
    _entry("gemini-3.5-flash-lite", _GOOGLE, "gemini-3.6"),
    _entry("gemini-3.5-flash", _GOOGLE, "gemini-3"),
    _entry("gemini-3.1-flash-image-preview", _GOOGLE, "gemini-3"),
    _entry("gemini-3.1-flash-tts-preview", _GOOGLE, "gemini-3"),
    _entry("gemini-embedding-2", _GOOGLE, "gemini-3"),
    _entry("claude-fable-5", _ANTHROPIC, "claude-5"),
    _entry("claude-sonnet-5", _ANTHROPIC, "claude-5"),
    _entry("claude-opus-4-8", _ANTHROPIC, "claude-5"),
    _entry("claude-sonnet-4-6", _ANTHROPIC, "claude-4-6"),
    _entry("gpt-5.5", _OPENAI, "gpt-5.5"),
    _entry("text-embedding-3-large", _OPENAI, "openai-embedding"),
    _entry("glm-5.1", _ZAI, "glm-5.1"),
    _entry("kimi-k3", _MOONSHOT, "kimi-k3"),
    _entry("kimi-k2.6", _MOONSHOT, "kimi-k2.6"),
    _entry("deepseek-v4-flash", _DEEPSEEK, "deepseek-v4"),
    # OpenRouter
    _entry("z-ai/glm-5.2", _OPENROUTER, "glm-5.1"),
    _entry("z-ai/glm-5.1", _OPENROUTER, "glm-5.1"),
    _entry("moonshotai/kimi-k3", _OPENROUTER, "kimi-k3"),
    _entry("moonshotai/kimi-k2.6", _OPENROUTER, "kimi-k2.6"),
    _entry("qwen/qwen3.6-35b-a3b", _OPENROUTER, "openai"),
    _entry("qwen/qwen3-embedding-4b", _OPENROUTER, "openai-embedding"),
    # SiliconFlow
    _entry("zai-org/GLM-5.2", _SILICONFLOW, "glm-5.1"),
    _entry("Pro/zai-org/GLM-5.1", _SILICONFLOW, "glm-5.1"),
    _entry("Pro/moonshotai/Kimi-K2.6", _SILICONFLOW, "kimi-k2.6"),
    _entry("Qwen/Qwen3.6-35B-A3B", _SILICONFLOW, "openai"),
    _entry("Qwen/Qwen3-Embedding-8B", _SILICONFLOW, "openai-embedding"),
]


def list_supported_models() -> list[SupportedModel]:
    """List supported models as (model, base_url, client) triples.

    Covers the official vendor endpoints plus the OpenRouter and SiliconFlow platforms;
    ``client`` is the ``client_type`` token that routes the model to its protocol client.
    """
    return [dict(entry) for entry in _SUPPORTED_MODELS]
