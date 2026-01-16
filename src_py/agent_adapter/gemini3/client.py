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

import json
import os
from typing import AsyncIterator

from google import genai
from google.genai import types

from ..base_client import LLMClient
from ..types import (
    ContentItem,
    FinishReason,
    ThinkingLevel,
    ToolChoice,
    UniConfig,
    UniEvent,
    UniMessage,
    UsageMetadata,
)


class Gemini3Client(LLMClient):
    """Gemini 3-specific LLM client implementation."""

    def __init__(self, model: str, api_key: str | None = None):
        """Initialize Gemini 3 client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self._history: list[UniMessage] = []

    def _detect_mime_type(self, url: str) -> str | None:
        """Detect MIME type from URL extension."""
        import mimetypes

        mime_type, _ = mimetypes.guess_type(url)
        return mime_type

    def transform_uni_config_to_model_config(self, config: UniConfig) -> types.GenerateContentConfig | None:
        """
        Transform universal configuration to Gemini-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            Gemini GenerateContentConfig object or None if no config needed
        """
        config_params = {}
        if config.get("system_prompt") is not None:
            config_params["system_instruction"] = config["system_prompt"]

        if config.get("max_tokens") is not None:
            config_params["max_output_tokens"] = config["max_tokens"]

        if config.get("temperature") is not None:
            config_params["temperature"] = config["temperature"]

        # Convert thinking level
        thinking_summary = config.get("thinking_summary")
        thinking_level = config.get("thinking_level")
        if thinking_summary is not None or thinking_level is not None:
            gemini_thinking_summary = thinking_summary or False
            gemini_thinking_level = self._convert_thinking_level(thinking_level)
            config_params["thinking_config"] = types.ThinkingConfig(
                include_thoughts=gemini_thinking_summary, thinking_level=gemini_thinking_level
            )

        # Convert tools and tool choice
        tools = config.get("tools")
        if tools is not None:
            config_params["tools"] = [types.Tool(function_declarations=tools)]
            tool_choice = config.get("tool_choice")
            if tool_choice is not None:
                tool_config = self._convert_tool_choice(tool_choice)
                config_params["tool_config"] = types.ToolConfig(function_calling_config=tool_config)

        return types.GenerateContentConfig(**config_params) if config_params else None

    def _convert_thinking_level(self, thinking_level: ThinkingLevel) -> types.ThinkingLevel | None:
        """Convert ThinkingLevel enum to Gemini's ThinkingLevel."""
        mapping = {
            ThinkingLevel.NONE: types.ThinkingLevel.MINIMAL,
            ThinkingLevel.LOW: types.ThinkingLevel.LOW,
            ThinkingLevel.MEDIUM: types.ThinkingLevel.MEDIUM,
            ThinkingLevel.HIGH: types.ThinkingLevel.HIGH,
        }
        return mapping.get(thinking_level)

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> types.FunctionCallingConfig:
        """Convert ToolChoice to Gemini's tool config."""
        if isinstance(tool_choice, list):
            return types.FunctionCallingConfig(mode="ANY", allowed_function_names=tool_choice)
        elif tool_choice == "none":
            return types.FunctionCallingConfig(mode="NONE")
        elif tool_choice == "auto":
            return types.FunctionCallingConfig(mode="AUTO")
        elif tool_choice == "required":
            return types.FunctionCallingConfig(mode="ANY")

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[types.Content]:
        """
        Transform universal message format to Gemini's Content format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of Gemini Content objects
        """
        mapping = {
            "user": "user",
            "assistant": "model",
            "tool": "user",
        }
        contents = []
        for msg in messages:
            role = msg["role"]
            parts = []
            for item in msg["content_items"]:
                if role == "tool" and item["type"] == "text":
                    function_name = item["tool_call_id"]
                    function_response = {"output": item["text"]}
                    parts.append(types.Part.from_function_response(name=function_name, response=function_response))
                elif item["type"] == "image_url":
                    url_value = item["image_url"]
                    mime_type = self._detect_mime_type(url_value)
                    parts.append(types.Part.from_uri(file_uri=url_value, mime_type=mime_type))
                elif item["type"] == "text":
                    parts.append(types.Part(text=item["text"], thought_signature=item.get("signature")))
                elif item["type"] == "reasoning":
                    parts.append(
                        types.Part(text=item["reasoning"], thought=True, thought_signature=item.get("signature"))
                    )
                elif item["type"] == "function_call":
                    function_call = types.FunctionCall(name=item["name"], args=json.loads(item["argument"]))
                    parts.append(types.Part(function_call=function_call, thought_signature=item.get("signature")))
                else:
                    raise ValueError(f"Unknown item: {item}")

            contents.append(types.Content(role=mapping[msg["role"]], parts=parts))

        return contents

    def transform_model_output_to_uni_event(self, model_output: types.GenerateContentResponse) -> UniEvent:
        """
        Transform Gemini model output to universal event format.

        Args:
            model_output: Gemini response chunk

        Returns:
            Universal event dictionary
        """
        mapping = {
            types.FinishReason.STOP: "stop",
            types.FinishReason.MAX_TOKENS: "length",
        }
        content_items: list[ContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        candidate = model_output.candidates[0]
        for part in candidate.content.parts:
            if part.function_call is not None:
                content_items.append(
                    {
                        "type": "function_call",
                        "name": part.function_call.name,
                        "argument": json.dumps(part.function_call.args, ensure_ascii=False),
                        "tool_call_id": part.function_call.name,
                        "signature": part.thought_signature,
                    }
                )
            elif part.text is not None and part.thought:
                content_items.append(
                    {"type": "reasoning", "reasoning": part.text, "signature": part.thought_signature}
                )
            elif part.text is not None:
                content_items.append({"type": "text", "text": part.text, "signature": part.thought_signature})
            else:
                raise ValueError(f"Unknown output: {part}")

        if model_output.usage_metadata:
            usage_metadata = {
                "prompt_tokens": model_output.usage_metadata.prompt_token_count,
                "thoughts_tokens": model_output.usage_metadata.thoughts_token_count,
                "response_tokens": model_output.usage_metadata.candidates_token_count,
            }

        if candidate.finish_reason:
            finish_reason = mapping.get(candidate.finish_reason, "unknown")

        return {
            "role": "assistant",
            "content_items": content_items,
            "usage_metadata": usage_metadata,
            "finish_reason": finish_reason,
        }

    async def streaming_response(
        self,
        messages: list[UniMessage],
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
            yield self.transform_model_output_to_uni_event(chunk)

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
        async for event in self.streaming_response(messages=self._history, config=config):
            events.append(event)
            yield event

        # Convert events to message and add to history
        if events:
            assistant_message = self.concat_uni_events_to_uni_message(events)
            self._history.append(assistant_message)

    def clear_history(self) -> None:
        """Clear the message history."""
        self._history.clear()

    def get_history(self) -> list[UniMessage]:
        """Get the current message history."""
        return self._history.copy()
