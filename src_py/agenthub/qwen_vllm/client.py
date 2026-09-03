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

from typing import Any

from ..openai_chat import OpenaiChatClient
from ..types import ThinkingLevel, UniConfig


class QwenVllmClient(OpenaiChatClient):
    """Qwen models served through vLLM's OpenAI-compatible Chat Completions API."""

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """Map AgentHub's level to the boolean switch consumed by Qwen chat templates."""
        qwen_config = super().transform_uni_config_to_model_config(config)

        if config.get("thinking_level") is not None:
            qwen_config["chat_template_kwargs"] = {
                "enable_thinking": config["thinking_level"] != ThinkingLevel.NONE,
            }

        return qwen_config
