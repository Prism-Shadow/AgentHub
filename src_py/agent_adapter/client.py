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

from .types import MessageDict, ThinkingLevel, ToolChoice


class LLMClient:
    """
    Unified LLM client that provides a consistent interface for different model SDKs.

    This client provides two main methods:
    - stream_generate: Stateless async method that requires full message history
    - stream_generate_stateful: Stateful async method that only requires the latest message
    """

    def __init__(self):
        """Initialize the LLM client."""
        self._message_history: List[MessageDict] = []

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

        This method requires the full message history to be passed in each call.

        Args:
            messages: List of message dictionaries containing conversation history
            model: Model identifier to use for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            tools: List of tools/functions available to the model
            thinking_level: Level of reasoning depth (none, low, medium, high)
            tool_choice: Tool usage preference (none/auto/required or list of tool names)

        Yields:
            Message chunks from the streaming response. Each chunk is a dictionary
            with the response data. The actual structure depends on the SDK implementation.
        """
        # TODO: Implement actual SDK integration
        # For now, return a simple mock response as a dict
        # Future implementations should return SDK-specific response objects
        yield {"type": "text", "content": "Mock response chunk"}

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

        This method maintains conversation history internally. Only the latest
        message needs to be provided.

        Args:
            message: Latest message dictionary to add to conversation
            model: Model identifier to use for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            tools: List of tools/functions available to the model
            thinking_level: Level of reasoning depth (none, low, medium, high)
            tool_choice: Tool usage preference (none/auto/required or list of tool names)

        Yields:
            Message chunks from the streaming response. Each chunk is a dictionary
            with the response data. The actual structure depends on the SDK implementation.
        """
        # Add message to history
        self._message_history.append(message)

        # Use the stateless method with full history
        async for chunk in self.stream_generate(
            messages=self._message_history,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            thinking_level=thinking_level,
            tool_choice=tool_choice,
        ):
            yield chunk

    def clear_history(self) -> None:
        """Clear the message history for stateful generation."""
        self._message_history.clear()

    def get_history(self) -> List[MessageDict]:
        """Get the current message history."""
        return self._message_history.copy()
