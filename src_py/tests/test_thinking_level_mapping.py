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
