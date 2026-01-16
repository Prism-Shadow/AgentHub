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

from .base_client import LLMClient
from .types import MessageDict, ThinkingLevel, ToolChoice


class GeminiClient(LLMClient):
    """Gemini-specific LLM client implementation."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client."""
        self._history: List[MessageDict] = []
        self._gemini_client = None
        self._api_key = api_key

    def _get_gemini_client(self):
        """Get or create Gemini client."""
        if self._gemini_client is None:
            api_key = self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self._gemini_client = genai.Client(api_key=api_key) if api_key else genai.Client()
        return self._gemini_client

    def _detect_mime_type(self, url: str) -> str:
        """Detect MIME type from URL extension."""
        url_lower = url.lower()
        if url_lower.endswith(".png"):
            return "image/png"
        elif url_lower.endswith(".jpg") or url_lower.endswith(".jpeg"):
            return "image/jpeg"
        elif url_lower.endswith(".gif"):
            return "image/gif"
        elif url_lower.endswith(".webp"):
            return "image/webp"
        else:
            return "image/jpeg"

    def convert_config_to_model_config(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Any]] = None,
        thinking_level: Optional[ThinkingLevel] = None,
        tool_choice: Optional[ToolChoice] = None,
    ) -> Optional[types.GenerateContentConfig]:
        """
        Convert unified configuration to Gemini-specific configuration.

        Args:
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)
            tools: List of tools/functions available to the model
            thinking_level: Level of reasoning depth (none, low, medium, high)
            tool_choice: Tool usage preference (none/auto/required or list of tool names)

        Returns:
            Gemini GenerateContentConfig object or None if no config needed
        """
        config_params = {}

        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens
        if temperature is not None:
            config_params["temperature"] = temperature

        # Convert thinking level
        if thinking_level is not None:
            gemini_thinking_level = self._convert_thinking_level(thinking_level)
            if gemini_thinking_level:
                config_params["thinking_config"] = types.ThinkingConfig(thinking_level=gemini_thinking_level)

        # Convert tools and tool choice
        if tools is not None:
            config_params["tools"] = tools
            if tool_choice is not None:
                tool_config = self._convert_tool_choice(tool_choice)
                if tool_config:
                    config_params["tool_config"] = types.ToolConfig(function_calling_config=tool_config)

        return types.GenerateContentConfig(**config_params) if config_params else None

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
            return types.FunctionCallingConfig(mode="ANY", allowed_function_names=tool_choice)
        elif tool_choice == "none":
            return types.FunctionCallingConfig(mode="NONE")
        elif tool_choice == "auto":
            return types.FunctionCallingConfig(mode="AUTO")
        elif tool_choice == "required":
            return types.FunctionCallingConfig(mode="ANY")
        return None

    def convert_messages_to_model_input(self, messages: List[MessageDict]) -> List[types.Content]:
        """
        Convert standard message format to Gemini's Content format.

        Args:
            messages: List of message dictionaries in standard format

        Returns:
            List of Gemini Content objects
        """
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "assistant":
                gemini_role = "model"
            elif role == "tool":
                gemini_role = "tool"
            else:
                gemini_role = "user"

            content_data = msg.get("content", "")

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
                            mime_type = self._detect_mime_type(item_value)
                            parts.append(types.Part.from_uri(file_uri=item_value, mime_type=mime_type))
                        elif item_type == "thought_signature":
                            # Pass thought signature back to API
                            parts.append(types.Part(thought_signature=item_value))
                    else:
                        parts.append(types.Part.from_text(text=str(item)))
            else:
                parts = [types.Part.from_text(text=str(content_data))]

            contents.append(types.Content(role=gemini_role, parts=parts))
        return contents

    def convert_model_output_to_message(self, model_output: Any) -> MessageDict:
        """
        Convert Gemini model output to standard message format.

        Args:
            model_output: Gemini response chunk

        Returns:
            Standard message dictionary with role and content
        """
        response_text = ""
        thought_signature = None

        # Extract text and thought signature from chunk
        try:
            if hasattr(model_output, "candidates") and model_output.candidates:
                for candidate in model_output.candidates:
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                response_text += part.text
                            if hasattr(part, "thought_signature") and part.thought_signature:
                                thought_signature = part.thought_signature
            elif hasattr(model_output, "text") and model_output.text:
                response_text += model_output.text
        except Exception:
            pass

        # Build standard message
        content_items = []
        if response_text:
            content_items.append({"type": "text", "value": response_text})
        if thought_signature:
            content_items.append({"type": "thought_signature", "value": thought_signature})

        return {
            "role": "assistant",
            "content": content_items if len(content_items) > 1 else (response_text if response_text else ""),
        }

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
        """Stream generate using Gemini SDK with unified conversion methods."""
        client = self._get_gemini_client()

        # Use unified config conversion
        config = self.convert_config_to_model_config(
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            thinking_level=thinking_level,
            tool_choice=tool_choice,
        )

        # Use unified message conversion
        contents = self.convert_messages_to_model_input(messages)

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
        """Stream generate with automatic history management using unified conversion methods."""
        # Add user message to history
        self._history.append(message)

        # Accumulate complete output for conversion
        complete_output = None

        async for chunk in self.stream_generate(
            messages=self._history,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            thinking_level=thinking_level,
            tool_choice=tool_choice,
        ):
            yield chunk
            # Keep last chunk for conversion
            complete_output = chunk

        # Convert model output to standard message and add to history
        if complete_output is not None:
            standard_message = self.convert_model_output_to_message(complete_output)
            if standard_message.get("content"):
                self._history.append(standard_message)

    def clear_history(self) -> None:
        """Clear the message history."""
        self._history.clear()

    def get_history(self) -> List[MessageDict]:
        """Get the current message history."""
        return self._history.copy()
