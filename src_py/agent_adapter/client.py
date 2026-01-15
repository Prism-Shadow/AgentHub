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

import os
from typing import Any, AsyncIterator, List, Optional

from google import genai
from google.genai import types

from .types import MessageDict, ThinkingLevel, ToolChoice


class LLMClient:
    """
    Unified LLM client that provides a consistent interface for different model SDKs.

    This client provides two main methods:
    - stream_generate: Stateless async method that requires full message history
    - stream_generate_stateful: Stateful async method that only requires the latest message
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            api_key: Optional API key. If not provided, will use environment variable.
        """
        self.history: List[MessageDict] = []
        self._gemini_client = None
        self._api_key = api_key

    def _get_gemini_client(self):
        """Get or create Gemini client."""
        if self._gemini_client is None:
            api_key = self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self._gemini_client = genai.Client(api_key=api_key) if api_key else genai.Client()
        return self._gemini_client

    def _convert_thinking_level(self, thinking_level: Optional[ThinkingLevel]) -> Optional[types.ThinkingLevel]:
        """Convert our ThinkingLevel enum to Gemini's ThinkingLevel."""
        if thinking_level is None:
            return None
        mapping = {
            ThinkingLevel.NONE: types.ThinkingLevel.MINIMAL,
            ThinkingLevel.LOW: types.ThinkingLevel.LOW,
            ThinkingLevel.MEDIUM: types.ThinkingLevel.MEDIUM,
            ThinkingLevel.HIGH: types.ThinkingLevel.HIGH,
        }
        return mapping.get(thinking_level)

    def _convert_tool_choice(self, tool_choice: Optional[ToolChoice]) -> Optional[types.FunctionCallingConfig]:
        """Convert our ToolChoice to Gemini's tool config."""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, list):
            # For list of tool names, use ANY mode with allowed function names
            return types.FunctionCallingConfig(mode="ANY", allowed_function_names=tool_choice)
        elif tool_choice == "none":
            return types.FunctionCallingConfig(mode="NONE")
        elif tool_choice == "auto":
            return types.FunctionCallingConfig(mode="AUTO")
        elif tool_choice == "required":
            return types.FunctionCallingConfig(mode="ANY")
        return None

    def _convert_messages_to_contents(self, messages: List[MessageDict]) -> List[types.Content]:
        """Convert our message format to Gemini's Content format."""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Map our roles to Gemini's roles
            if role == "assistant":
                gemini_role = "model"
            elif role == "tool":
                gemini_role = "tool"
            else:
                gemini_role = "user"

            content_data = msg.get("content", "")

            # Handle different content formats
            if isinstance(content_data, str):
                parts = [types.Part.from_text(text=content_data)]
            elif isinstance(content_data, list):
                parts = []
                for item in content_data:
                    if isinstance(item, dict):
                        item_type = item.get("type", "text")
                        item_value = item.get("value", "")
                        if item_type == "text":
                            parts.append(types.Part.from_text(text=item_value))
                        elif item_type == "image_url":
                            # For image URLs, use from_uri
                            parts.append(types.Part.from_uri(file_uri=item_value, mime_type="image/jpeg"))
                    else:
                        parts.append(types.Part.from_text(text=str(item)))
            else:
                parts = [types.Part.from_text(text=str(content_data))]

            contents.append(types.Content(role=gemini_role, parts=parts))
        return contents

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
        # Dispatch based on model name
        if "gemini" in model.lower():
            async for chunk in self._stream_generate_gemini(
                messages, model, max_tokens, temperature, tools, thinking_level, tool_choice
            ):
                yield chunk
        elif "gpt" in model.lower() or "o1" in model.lower():
            # TODO: Implement GPT/OpenAI integration
            raise NotImplementedError("GPT models not yet implemented")
        elif "claude" in model.lower():
            # TODO: Implement Claude integration
            raise NotImplementedError("Claude models not yet implemented")
        else:
            raise ValueError(f"Unknown model type: {model}")

    async def _stream_generate_gemini(
        self,
        messages: List[MessageDict],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Any]] = None,
        thinking_level: Optional[ThinkingLevel] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> AsyncIterator[Any]:
        """Stream generate using Gemini SDK."""
        client = self._get_gemini_client()

        # Build config
        config_params = {}
        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_params["temperature"] = temperature

        # Handle thinking level
        if thinking_level is not None:
            gemini_thinking_level = self._convert_thinking_level(thinking_level)
            if gemini_thinking_level:
                config_params["thinking_config"] = types.ThinkingConfig(thinking_level=gemini_thinking_level)

        # Handle tools and tool choice
        if tools is not None:
            config_params["tools"] = tools
            if tool_choice is not None:
                tool_config = self._convert_tool_choice(tool_choice)
                if tool_config:
                    config_params["tool_config"] = types.ToolConfig(function_calling_config=tool_config)

        config = types.GenerateContentConfig(**config_params) if config_params else None

        # Convert messages to Gemini format
        contents = self._convert_messages_to_contents(messages)

        # Stream generate
        response_stream = await client.aio.models.generate_content_stream(
            model=model, contents=contents, config=config
        )

        async for chunk in response_stream:
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
        # Add user message to history
        self.history.append(message)

        # Collect the complete response
        response_text = ""
        async for chunk in self.stream_generate(
            messages=self.history,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            thinking_level=thinking_level,
            tool_choice=tool_choice,
        ):
            # Yield the chunk
            yield chunk

            # Accumulate response text
            if hasattr(chunk, "text") and chunk.text:
                response_text += chunk.text

        # Add model response to history
        if response_text:
            self.history.append({"role": "assistant", "content": response_text})

    def clear_history(self) -> None:
        """Clear the message history for stateful generation."""
        self.history.clear()

    def get_history(self) -> List[MessageDict]:
        """Get the current message history."""
        return self.history.copy()
