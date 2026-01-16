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
from typing import AsyncIterator, List, Optional

from google import genai
from google.genai import types

from .base_client import LLMClient
from .types import ThinkingLevel, ToolChoice, UniConfig, UniEvent, UniMessage


class GeminiClient(LLMClient):
    """Gemini-specific LLM client implementation."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        """Initialize Gemini client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._history: List[UniMessage] = []

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

    def transform_uni_config_to_model_config(self, config: UniConfig) -> Optional[types.GenerateContentConfig]:
        """
        Transform universal configuration to Gemini-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            Gemini GenerateContentConfig object or None if no config needed
        """
        config_params = {}

        if config.get("max_tokens") is not None:
            config_params["max_output_tokens"] = config["max_tokens"]
        if config.get("temperature") is not None:
            config_params["temperature"] = config["temperature"]

        # Convert thinking level
        thinking_level = config.get("thinking_level")
        if thinking_level is not None:
            gemini_thinking_level = self._convert_thinking_level(thinking_level)
            if gemini_thinking_level:
                config_params["thinking_config"] = types.ThinkingConfig(thinking_level=gemini_thinking_level)

        # Convert tools and tool choice
        tools = config.get("tools")
        if tools is not None:
            config_params["tools"] = tools
            tool_choice = config.get("tool_choice")
            if tool_choice is not None:
                tool_config = self._convert_tool_choice(tool_choice)
                if tool_config:
                    config_params["tool_config"] = types.ToolConfig(function_calling_config=tool_config)

        return types.GenerateContentConfig(**config_params) if config_params else None

    def _convert_thinking_level(self, thinking_level: ThinkingLevel) -> Optional[types.ThinkingLevel]:
        """Convert ThinkingLevel enum to Gemini's ThinkingLevel."""
        mapping = {
            ThinkingLevel.NONE: types.ThinkingLevel.MINIMAL,
            ThinkingLevel.LOW: types.ThinkingLevel.LOW,
            ThinkingLevel.MEDIUM: types.ThinkingLevel.MEDIUM,
            ThinkingLevel.HIGH: types.ThinkingLevel.HIGH,
        }
        return mapping.get(thinking_level)

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> Optional[types.FunctionCallingConfig]:
        """Convert ToolChoice to Gemini's tool config."""
        if isinstance(tool_choice, list):
            return types.FunctionCallingConfig(mode="ANY", allowed_function_names=tool_choice)
        elif tool_choice == "none":
            return types.FunctionCallingConfig(mode="NONE")
        elif tool_choice == "auto":
            return types.FunctionCallingConfig(mode="AUTO")
        elif tool_choice == "required":
            return types.FunctionCallingConfig(mode="ANY")
        return None

    def transform_uni_message_to_model_input(self, messages: List[UniMessage]) -> List[types.Content]:
        """
        Transform universal message format to Gemini's Content format.

        Args:
            messages: List of universal message dictionaries

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

            content_items = msg.get("content_items", [])
            parts = []

            for item in content_items:
                item_type = item.get("type", "text")
                if item_type == "text":
                    text_value = item.get("text", "")
                    parts.append(types.Part.from_text(text=text_value))
                elif item_type == "image_url":
                    url_value = item.get("image_url", "")
                    mime_type = self._detect_mime_type(url_value)
                    parts.append(types.Part.from_uri(file_uri=url_value, mime_type=mime_type))
                elif item_type == "reasoning_signature":
                    signature_value = item.get("signature", "")
                    parts.append(types.Part(thought_signature=signature_value))

            contents.append(types.Content(role=gemini_role, parts=parts))
        return contents

    def transform_model_output_to_uni_event(self, model_output: types.GenerateContentResponse) -> UniEvent:
        """
        Transform Gemini model output to universal event format.

        Args:
            model_output: Gemini response chunk

        Returns:
            Universal event dictionary
        """
        content_items = []

        try:
            if hasattr(model_output, "candidates") and model_output.candidates:
                candidate = model_output.candidates[0]
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        content_items.append({"type": "text", "text": part.text})
                    if hasattr(part, "thought_signature") and part.thought_signature:
                        content_items.append({"type": "reasoning_signature", "signature": part.thought_signature})
            elif hasattr(model_output, "text") and model_output.text:
                content_items.append({"type": "text", "text": model_output.text})
        except Exception:
            pass

        return {
            "role": "assistant",
            "content_items": content_items,
        }

    async def streaming_response(
        self,
        messages: List[UniMessage],
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        """Stream generate using Gemini SDK with unified conversion methods."""
        # Use unified config conversion
        gemini_config = self.transform_uni_config_to_model_config(config)

        # Use unified message conversion
        contents = self.transform_uni_message_to_model_input(messages)

        # Stream generate
        response_stream = await self._client.aio.models.generate_content_stream(
            model=self._model, contents=contents, config=gemini_config
        )

        async for chunk in response_stream:
            event = self.transform_model_output_to_uni_event(chunk)
            yield event

    async def streaming_response_stateful(
        self,
        message: UniMessage,
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        """Stream generate with automatic history management using unified conversion methods."""
        # Add user message to history
        self._history.append(message)

        # Collect all events for history
        events = []

        async for event in self.streaming_response(
            messages=self._history,
            config=config,
        ):
            events.append(event)
            yield event

        # Convert events to message and add to history
        if events:
            assistant_message = self.transform_uni_event_to_uni_message(events)
            self._history.append(assistant_message)

    def clear_history(self) -> None:
        """Clear the message history."""
        self._history.clear()

    def get_history(self) -> List[UniMessage]:
        """Get the current message history."""
        return self._history.copy()
