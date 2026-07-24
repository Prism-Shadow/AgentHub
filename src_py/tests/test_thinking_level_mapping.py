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
# (gemini-3-pro also "medium"), image models accept only "minimal" and "high".
# Unsupported levels must clamp to the closest supported one, never error.
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
    ("gemini-3-flash-preview", ThinkingLevel.NONE, types.ThinkingLevel.MINIMAL),
    ("gemini-3.5-flash", ThinkingLevel.MEDIUM, types.ThinkingLevel.MEDIUM),
]


@pytest.mark.parametrize(("model", "level", "expected"), GEMINI3_THINKING_LEVEL_CASES)
def test_gemini3_thinking_level_clamps_to_model_support(
    model: str, level: ThinkingLevel, expected: types.ThinkingLevel
):
    client = AutoLLMClient(model=model, api_key="test-key")
    assert client._client._convert_thinking_level(level) == expected  # noqa: SLF001
