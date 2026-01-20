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

from typing import Any, AsyncIterator

from .base_client import LLMClient
from .types import UniConfig, UniEvent, UniMessage


class AutoLLMClient(LLMClient):
    """
    Auto-routing LLM client that dispatches to appropriate model-specific client.

    This client is stateful - it knows the model name at initialization and maintains
    conversation history for that specific model.
    """

    def __init__(self, model: str, api_key: str | None = None):
        """
        Initialize AutoLLMClient with a specific model.

        Args:
            model: Model identifier (determines which client to use)
            api_key: Optional API key
        """
        self._client = self._create_client_for_model(model, api_key)

    def _create_client_for_model(self, model: str, api_key: str | None = None) -> LLMClient:
        """Create the appropriate client for the given model."""
        if "gemini-3" in model.lower():  # e.g., gemini-3-flash-preview
            from .gemini3 import Gemini3Client

            return Gemini3Client(model=model, api_key=api_key)
        elif "claude" in model.lower() and "4-5" in model.lower():  # e.g., claude-sonnet-4-5
            from .claude4_5 import Claude4_5Client

            return Claude4_5Client(model=model, api_key=api_key)
        elif "glm-4.7" in model.lower():  # e.g., glm-4.7
            from .glm4_7 import GLM4_7Client

            return GLM4_7Client(model=model, api_key=api_key)
        elif "gpt-5.2" in model.lower():  # e.g., gpt-5.2
            raise NotImplementedError("GPT models not yet implemented.")
        else:
            raise ValueError(f"{model} is not supported.")

    def transform_uni_config_to_model_config(self, config: UniConfig) -> Any:
        """Delegate to underlying client's transform_uni_config_to_model_config."""
        return self._client.transform_uni_config_to_model_config(config)

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> Any:
        """Delegate to underlying client's transform_uni_message_to_model_input."""
        return self._client.transform_uni_message_to_model_input(messages)

    def transform_model_output_to_uni_event(self, model_output: Any) -> UniEvent:
        """Delegate to underlying client's transform_model_output_to_uni_event."""
        return self._client.transform_model_output_to_uni_event(model_output)

    async def streaming_response(
        self,
        messages: list[UniMessage],
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        """Route to underlying client's streaming_response."""
        async for event in self._client.streaming_response(
            messages=messages,
            config=config,
        ):
            yield event

    async def streaming_response_stateful(
        self,
        message: UniMessage,
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        """Route to underlying client's streaming_response_stateful."""
        async for event in self._client.streaming_response_stateful(
            message=message,
            config=config,
        ):
            yield event

    def clear_history(self) -> None:
        """Clear history in the underlying client."""
        self._client.clear_history()

    def get_history(self) -> list[UniMessage]:
        """Get history from the underlying client."""
        return self._client.get_history()
