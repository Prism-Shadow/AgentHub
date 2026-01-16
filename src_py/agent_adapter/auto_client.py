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

from typing import Any, AsyncIterator, List, Optional

from .base_client import LLMClient
from .gemini_client import GeminiClient
from .types import MessageDict, ThinkingLevel, ToolChoice


class AutoLLMClient(LLMClient):
    """
    Auto-routing LLM client that dispatches to appropriate model-specific client.

    This client is stateful - it knows the model name at initialization and maintains
    conversation history for that specific model.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize AutoLLMClient with a specific model.

        Args:
            model: Model identifier (determines which client to use)
            api_key: Optional API key
        """
        self._model = model
        self._api_key = api_key
        self._client = self._create_client_for_model(model)

    def _create_client_for_model(self, model: str) -> LLMClient:
        """Create the appropriate client for the given model."""
        if "gemini" in model.lower():
            return GeminiClient(self._api_key)
        elif "gpt" in model.lower() or "o1" in model.lower():
            raise NotImplementedError("GPT models not yet implemented")
        elif "claude" in model.lower():
            raise NotImplementedError("Claude models not yet implemented")
        else:
            raise ValueError(f"Unknown model type: {model}")

    def convert_config_to_model_config(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Any]] = None,
        thinking_level: Optional[ThinkingLevel] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Any:
        """Delegate to underlying client's convert_config_to_model_config."""
        return self._client.convert_config_to_model_config(
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            thinking_level=thinking_level,
            tool_choice=tool_choice,
        )

    def convert_messages_to_model_input(self, messages: List[MessageDict]) -> Any:
        """Delegate to underlying client's convert_messages_to_model_input."""
        return self._client.convert_messages_to_model_input(messages)

    def convert_model_output_to_message(self, model_output: Any) -> MessageDict:
        """Delegate to underlying client's convert_model_output_to_message."""
        return self._client.convert_model_output_to_message(model_output)

    async def stream_generate(
        self,
        messages: List[MessageDict],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Any]] = None,
        thinking_level: Optional[ThinkingLevel] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> AsyncIterator[Any]:
        """Route to underlying client's stream_generate."""
        async for chunk in self._client.stream_generate(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            thinking_level=thinking_level,
            tool_choice=tool_choice,
        ):
            yield chunk

    async def stream_generate_stateful(
        self,
        message: MessageDict,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Any]] = None,
        thinking_level: Optional[ThinkingLevel] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> AsyncIterator[Any]:
        """Route to underlying client's stream_generate_stateful."""
        async for chunk in self._client.stream_generate_stateful(
            message=message,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            thinking_level=thinking_level,
            tool_choice=tool_choice,
        ):
            yield chunk

    def clear_history(self) -> None:
        """Clear history in the underlying client."""
        if hasattr(self._client, "clear_history"):
            self._client.clear_history()

    def get_history(self) -> List[MessageDict]:
        """Get history from the underlying client."""
        if hasattr(self._client, "get_history"):
            return self._client.get_history()
        return []
