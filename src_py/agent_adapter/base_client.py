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
    both abstract methods: stream_generate and stream_generate_stateful.
    """

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
