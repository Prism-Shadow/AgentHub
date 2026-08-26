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

import pytest
from google.genai import types

from agenthub import AutoLLMClient, ThinkingLevel


# Not every Gemini model accepts every thinking level (verified live 2026-07-24;
# see llmsdk_docs/gemini3/docs/thinking.md): pro models reject "minimal"
# (gemini-3-pro also "medium"), image models accept only "minimal" and "high",
# and the 2.5 series rejects the thinking_level parameter outright. Unsupported
# levels must clamp to the closest supported one — or be dropped entirely for
# models that take none — never error.
GEMINI3_THINKING_LEVEL_CASES = [
    ("gemini-3.1-pro-preview", ThinkingLevel.NONE, types.ThinkingLevel.LOW),
    ("gemini-3.1-pro-preview", ThinkingLevel.LOW, types.ThinkingLevel.LOW),
    ("gemini-3.1-pro-preview", ThinkingLevel.MEDIUM, types.ThinkingLevel.MEDIUM),
    ("gemini-3.1-pro-preview", ThinkingLevel.HIGH, types.ThinkingLevel.HIGH),
    ("gemini-3.1-pro-preview", ThinkingLevel.XHIGH, types.ThinkingLevel.HIGH),
    ("gemini-3.1-pro-preview", ThinkingLevel.MAX, types.ThinkingLevel.HIGH),
    ("gemini-3-pro-preview", ThinkingLevel.NONE, types.ThinkingLevel.LOW),
    ("gemini-3-pro-preview", ThinkingLevel.MEDIUM, types.ThinkingLevel.HIGH),
    ("gemini-3.1-flash-image", ThinkingLevel.NONE, types.ThinkingLevel.MINIMAL),
    ("gemini-3.1-flash-image", ThinkingLevel.LOW, types.ThinkingLevel.MINIMAL),
    ("gemini-3.1-flash-image", ThinkingLevel.MEDIUM, types.ThinkingLevel.HIGH),
    # "-image" wins over "gemini-3-pro" (LOW would stay LOW under the pro set).
    ("gemini-3-pro-image", ThinkingLevel.LOW, types.ThinkingLevel.MINIMAL),
    ("gemini-3-flash-preview", ThinkingLevel.NONE, types.ThinkingLevel.MINIMAL),
    ("gemini-3.5-flash", ThinkingLevel.MEDIUM, types.ThinkingLevel.MEDIUM),
    # The 2.5 series rejects thinking_level for every value: drop the parameter.
    ("gemini-2.5-pro", ThinkingLevel.NONE, None),
    ("gemini-2.5-flash", ThinkingLevel.HIGH, None),
    ("gemini-2.5-flash-lite", ThinkingLevel.LOW, None),
    # A future pro generation falls into the generic "-pro" branch.
    ("gemini-4-pro", ThinkingLevel.NONE, types.ThinkingLevel.LOW),
    # An unrecognized model inherits the full four-level default.
    ("gemini-9-flash", ThinkingLevel.NONE, types.ThinkingLevel.MINIMAL),
]


def _create_gemini3_auto_client(model: str) -> AutoLLMClient:
    # client_type pins routing so pre-3 and hypothetical model names reach
    # the unified Gemini3_7Client the same way an explicit override would in user code.
    return AutoLLMClient(model=model, api_key="test-key", client_type="gemini-3")


@pytest.mark.parametrize(("model", "level", "expected"), GEMINI3_THINKING_LEVEL_CASES)
def test_gemini3_thinking_level_clamps_to_model_support(
    model: str, level: ThinkingLevel, expected: types.ThinkingLevel | None
):
    client = _create_gemini3_auto_client(model)
    assert client._client._convert_thinking_level(level) == expected  # noqa: SLF001


def test_gemini3_thinking_config_carries_clamped_level():
    client = _create_gemini3_auto_client("gemini-3.1-pro-preview")
    config = client._client.transform_uni_config_to_model_config(  # noqa: SLF001
        {"thinking_level": ThinkingLevel.NONE}
    )
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_gemini3_thinking_config_omits_level_for_pre_3_models():
    client = _create_gemini3_auto_client("gemini-2.5-flash")
    config = client._client.transform_uni_config_to_model_config(  # noqa: SLF001
        {"thinking_level": ThinkingLevel.HIGH}
    )
    assert config.thinking_config.thinking_level is None


# The 3.7 generation drops "minimal" (verified live 2026-08-13; see
# llmsdk_docs/gemini3_7/docs/thinking.md); the 3.6-generation models routed to
# the same client keep the full four-level set.
GEMINI3_7_THINKING_LEVEL_CASES = [
    ("gemini-3.7-flash", ThinkingLevel.NONE, types.ThinkingLevel.LOW),
    ("gemini-3.7-flash", ThinkingLevel.LOW, types.ThinkingLevel.LOW),
    ("gemini-3.7-flash", ThinkingLevel.MEDIUM, types.ThinkingLevel.MEDIUM),
    ("gemini-3.7-flash", ThinkingLevel.HIGH, types.ThinkingLevel.HIGH),
    ("gemini-3.7-flash", ThinkingLevel.XHIGH, types.ThinkingLevel.HIGH),
    # Gemini has no level above "high", so MAX clamps there too.
    ("gemini-3.7-flash", ThinkingLevel.MAX, types.ThinkingLevel.HIGH),
    ("gemini-3.6-flash", ThinkingLevel.NONE, types.ThinkingLevel.MINIMAL),
    ("gemini-3.5-flash-lite", ThinkingLevel.NONE, types.ThinkingLevel.MINIMAL),
]


@pytest.mark.parametrize(("model", "level", "expected"), GEMINI3_7_THINKING_LEVEL_CASES)
def test_gemini3_7_thinking_level_clamps_to_model_support(
    model: str, level: ThinkingLevel, expected: types.ThinkingLevel
):
    # These are real model ids, so automatic routing reaches Gemini3_7Client directly.
    client = AutoLLMClient(model=model, api_key="test-key")
    assert client._client.__class__.__name__ == "Gemini3_7Client"
    assert client._client._convert_thinking_level(level) == expected  # noqa: SLF001


# GLM-5.3 cannot disable thinking and accepts only low/high/max reasoning_effort
# (llmsdk_docs/glm5_3/docs/thinking.md); GLM-5.2
# keeps the full pass-through vocabulary and pre-5.2 models take no effort parameter.
GLM_THINKING_LEVEL_CASES = [
    ("glm-5.3", ThinkingLevel.NONE, "enabled", "low"),
    ("glm-5.3", ThinkingLevel.LOW, "enabled", "low"),
    ("glm-5.3", ThinkingLevel.MEDIUM, "enabled", "high"),
    ("glm-5.3", ThinkingLevel.HIGH, "enabled", "high"),
    ("glm-5.3", ThinkingLevel.XHIGH, "enabled", "max"),
    ("glm-5.3", ThinkingLevel.MAX, "enabled", "max"),
    ("glm-5.3-flash", ThinkingLevel.MAX, "enabled", "max"),
    ("glm-5.2", ThinkingLevel.NONE, "disabled", None),
    ("glm-5.2", ThinkingLevel.MEDIUM, "enabled", "medium"),
    ("glm-5.2", ThinkingLevel.XHIGH, "enabled", "xhigh"),
    ("glm-5.2", ThinkingLevel.MAX, "enabled", "max"),
    ("glm-5.1", ThinkingLevel.HIGH, "enabled", None),
    # Provider-hosted ids keep their own casing (SiliconFlow), so generation
    # detection must be case-insensitive.
    ("zai-org/GLM-5.2", ThinkingLevel.XHIGH, "enabled", "xhigh"),
    ("Pro/zai-org/GLM-5.1", ThinkingLevel.HIGH, "enabled", None),
]


@pytest.mark.parametrize(("model", "level", "thinking_type", "effort"), GLM_THINKING_LEVEL_CASES)
def test_glm_thinking_level_maps_per_generation(
    model: str, level: ThinkingLevel, thinking_type: str, effort: str | None
):
    client = AutoLLMClient(model=model, api_key="test-key")
    assert client._client.__class__.__name__ == "GLM5_3Client"
    config = client._client.transform_uni_config_to_model_config({"thinking_level": level})  # noqa: SLF001
    assert config["extra_body"]["thinking"]["type"] == thinking_type
    assert config.get("reasoning_effort") == effort


# What each remaining client puts on the wire for a level, per its vendor's effort
# vocabulary: OpenAI takes the full set, Claude tops out at max (xhigh only from 4.7),
# DeepSeek and Kimi accept low/high/max, DeepSeek turns thinking off with none, and
# MiniMax has no level above high.
THINKING_EFFORT_CASES = [
    ("gpt-5.6", None, ThinkingLevel.XHIGH, "xhigh"),
    ("gpt-5.6", None, ThinkingLevel.MAX, "max"),
    ("gpt-5.6", "openai-responses", ThinkingLevel.MAX, "max"),
    ("claude-sonnet-5", None, ThinkingLevel.XHIGH, "xhigh"),
    ("claude-sonnet-5", None, ThinkingLevel.MAX, "max"),
    # 4.6 has no xhigh but does take max.
    ("claude-sonnet-4-6", None, ThinkingLevel.XHIGH, "high"),
    ("claude-sonnet-4-6", None, ThinkingLevel.MAX, "max"),
    ("claude-sonnet-5", "ant-messages", ThinkingLevel.MAX, "max"),
    ("deepseek-v4", None, ThinkingLevel.NONE, "none"),
    ("deepseek-v4", None, ThinkingLevel.LOW, "low"),
    ("deepseek-v4", None, ThinkingLevel.MEDIUM, "high"),
    ("deepseek-v4", None, ThinkingLevel.HIGH, "high"),
    # DeepSeek maps xhigh onto high server-side, so the client sends high.
    ("deepseek-v4", None, ThinkingLevel.XHIGH, "high"),
    ("deepseek-v4", None, ThinkingLevel.MAX, "max"),
    ("kimi-k3", None, ThinkingLevel.LOW, "low"),
    ("kimi-k3", None, ThinkingLevel.MEDIUM, "high"),
    ("kimi-k3", None, ThinkingLevel.XHIGH, "max"),
    ("kimi-k3", None, ThinkingLevel.MAX, "max"),
    ("MiniMax-M3", "minimax-m3", ThinkingLevel.XHIGH, "high"),
    ("MiniMax-M3", "minimax-m3", ThinkingLevel.MAX, "high"),
]


def _wire_effort(config: dict) -> str | None:
    """Read the effort out of whichever config key the client used."""
    if "reasoning" in config:
        return config["reasoning"].get("effort")
    if "output_config" in config:
        return config["output_config"].get("effort")
    return config.get("reasoning_effort")


@pytest.mark.parametrize(("model", "client_type", "level", "expected"), THINKING_EFFORT_CASES)
def test_thinking_level_maps_to_vendor_effort(
    model: str, client_type: str | None, level: ThinkingLevel, expected: str | None
):
    client = AutoLLMClient(model=model, api_key="test-key", client_type=client_type)
    config = client._client.transform_uni_config_to_model_config({"thinking_level": level})  # noqa: SLF001
    assert _wire_effort(config) == expected
