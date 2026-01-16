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

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional

from .types import MessageDict, ThinkingLevel, ToolChoice


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.

    All model-specific clients must inherit from this class and implement
    five abstract methods for complete SDK abstraction:
    1. convert_config_to_model_config - Convert unified config to model-specific config
    2. convert_messages_to_model_input - Convert standard messages to model input
    3. convert_model_output_to_message - Convert model output to standard message
    4. stream_generate - Stateless streaming generation
    5. stream_generate_stateful - Stateful streaming generation
    """

    @abstractmethod
    def convert_config_to_model_config(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Any]] = None,
        thinking_level: Optional[ThinkingLevel] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Any:
        """
        Convert unified configuration to model-specific configuration.

        Args:
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            tools: List of tools/functions available to the model
            thinking_level: Level of reasoning depth (none, low, medium, high)
            tool_choice: Tool usage preference (none/auto/required or list of tool names)

        Returns:
            Model-specific configuration object
        """
        pass

    @abstractmethod
    def convert_messages_to_model_input(self, messages: List[MessageDict]) -> Any:
        """
        Convert standard message format to model-specific input format.

        Args:
            messages: List of message dictionaries in standard format

        Returns:
            Model-specific input format (e.g., Gemini's Content list, OpenAI's messages array)
        """
        pass

    @abstractmethod
    def convert_model_output_to_message(self, model_output: Any) -> MessageDict:
        """
        Convert model output to standard message format.

        Args:
            model_output: Model-specific output object

        Returns:
            Standard message dictionary with role and content
        """
        pass

    @abstractmethod
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
        """
        Generate content in streaming mode (stateless).

        This method should use convert_config_to_model_config and
        convert_messages_to_model_input to prepare the request.

        Args:
            messages: List of message dictionaries containing conversation history
            model: Model identifier to use for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            tools: List of tools/functions available to the model
            thinking_level: Level of reasoning depth (none, low, medium, high)
            tool_choice: Tool usage preference (none/auto/required or list of tool names)

        Yields:
            Message chunks from the streaming response.
        """
        pass

    @abstractmethod
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
        """
        Generate content in streaming mode (stateful).

        This method should use convert_config_to_model_config,
        convert_messages_to_model_input, and convert_model_output_to_message
        to manage the conversation flow.

        Args:
            message: Latest message dictionary to add to conversation
            model: Model identifier to use for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            tools: List of tools/functions available to the model
            thinking_level: Level of reasoning depth (none, low, medium, high)
            tool_choice: Tool usage preference (none/auto/required or list of tool names)

        Yields:
            Message chunks from the streaming response.
        """
        pass
