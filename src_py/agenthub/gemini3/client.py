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

import base64
import json
import mimetypes
import os
import re
from typing import AsyncIterator

import requests
from google import genai
from google.genai import types

from ..base_client import LLMClient
from ..types import (
    EventType,
    FinishReason,
    PartialContentItem,
    PromptCaching,
    ThinkingLevel,
    ToolChoice,
    UniConfig,
    UniEvent,
    UniMessage,
    UsageMetadata,
)


class Gemini3Client(LLMClient):
    """Gemini 3-specific LLM client implementation."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        """Initialize Gemini 3 client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        base_url = base_url or os.getenv("GOOGLE_GEMINI_BASE_URL")
        self._client = (
            genai.Client(api_key=api_key, http_options={"base_url": base_url}) if api_key else genai.Client()
        )
        self._history: list[UniMessage] = []

    def _convert_thinking_level(self, thinking_level: ThinkingLevel | None) -> types.ThinkingLevel | None:
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

        thinking_summary = config.get("thinking_summary")
        thinking_level = config.get("thinking_level")
        if thinking_summary is not None or thinking_level is not None:
            config_params["thinking_config"] = types.ThinkingConfig(
                include_thoughts=thinking_summary, thinking_level=self._convert_thinking_level(thinking_level)
            )

        if config.get("tools") is not None:
            config_params["tools"] = [types.Tool(function_declarations=config["tools"])]
            tool_choice = config.get("tool_choice")
            if tool_choice is not None:
                tool_config = self._convert_tool_choice(tool_choice)
                config_params["tool_config"] = types.ToolConfig(function_calling_config=tool_config)

        if config.get("prompt_caching") is not None and config["prompt_caching"] != PromptCaching.ENABLE:
            raise ValueError("prompt_caching must be ENABLE for Gemini 3.")

        return types.GenerateContentConfig(**config_params) if config_params else None

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[types.Content]:
        """
        Transform universal message format to Gemini's Content format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of Gemini Content objects
        """
        mapping = {"user": "user", "assistant": "model"}
        contents = []
        for msg in messages:
            parts = []
            for item in msg["content_items"]:
                if item["type"] == "text":
                    parts.append(types.Part(text=item["text"], thought_signature=item.get("signature")))
                elif item["type"] == "image_url":
                    image_url = item["image_url"]
                    if image_url.startswith("data:"):
                        match = re.match(r"data:([^;]+);base64,(.+)", image_url)
                        if match:
                            mime_type = match.group(1)
                            base64_string = match.group(2)
                            base64_bytes = base64.b64decode(base64_string)
                            parts.append(types.Part.from_bytes(data=base64_bytes, mime_type=mime_type))
                        else:
                            raise ValueError(f"Invalid base64 image: {image_url}")
                    else:
                        mime_type, _ = mimetypes.guess_type(image_url)
                        parts.append(types.Part.from_uri(file_uri=image_url, mime_type=mime_type))
                elif item["type"] == "thinking":
                    parts.append(
                        types.Part(text=item["thinking"], thought=True, thought_signature=item.get("signature"))
                    )
                elif item["type"] == "tool_call":
                    function_call = types.FunctionCall(name=item["name"], args=item["arguments"])
                    parts.append(types.Part(function_call=function_call, thought_signature=item.get("signature")))
                elif item["type"] == "tool_result":
                    if "tool_call_id" not in item:
                        raise ValueError("tool_call_id is required for tool result.")

                    result_response = {"result": item["text"]}
                    multimodal_parts = []

                    if "image_url" in item:
                        image_url = item["image_url"]
                        if image_url.startswith("data:"):
                            match = re.match(r"data:([^;]+);base64,(.+)", image_url)
                            if match:
                                mime_type = match.group(1)
                                base64_string = match.group(2)
                                image_bytes = base64.b64decode(base64_string)
                            else:
                                raise ValueError(f"Invalid base64 image: {image_url}")
                        else:
                            response = requests.get(image_url)
                            response.raise_for_status()
                            image_bytes = response.content
                            mime_type, _ = mimetypes.guess_type(image_url)
                            if not mime_type:
                                mime_type = "image/jpeg"
                        multimodal_parts.append(
                            types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    mime_type=mime_type,
                                    data=image_bytes,
                                )
                            )
                        )

                    parts.append(
                        types.Part.from_function_response(
                            name=item["tool_call_id"], response=result_response, parts=multimodal_parts
                        )
                    )
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
        event_type: EventType = "delta"
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        candidate = model_output.candidates[0]
        for part in candidate.content.parts:
            if part.function_call is not None:
                content_items.append(
                    {
                        "type": "tool_call",
                        "name": part.function_call.name,
                        "arguments": part.function_call.args,
                        "tool_call_id": part.function_call.name,
                        "signature": part.thought_signature,
                    }
                )
            elif part.text is not None and part.thought:
                content_items.append({"type": "thinking", "thinking": part.text, "signature": part.thought_signature})
            elif part.text is not None:
                content_items.append({"type": "text", "text": part.text, "signature": part.thought_signature})
            else:
                raise ValueError(f"Unknown output: {part}")

        if candidate.finish_reason:
            event_type = "stop"
            stop_reason_mapping = {
                types.FinishReason.STOP: "stop",
                types.FinishReason.MAX_TOKENS: "length",
            }
            finish_reason = stop_reason_mapping.get(candidate.finish_reason, "unknown")

        if model_output.usage_metadata:
            event_type = event_type or "delta"  # deal with separate usage data
            usage_metadata = {
                "prompt_tokens": model_output.usage_metadata.prompt_token_count,
                "thoughts_tokens": model_output.usage_metadata.thoughts_token_count,
                "response_tokens": model_output.usage_metadata.candidates_token_count,
                "cached_tokens": model_output.usage_metadata.cached_content_token_count,
            }

        return {
            "role": "assistant",
            "event_type": event_type,
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
            event = self.transform_model_output_to_uni_event(chunk)
            for item in event["content_items"]:
                if item["type"] == "tool_call":
                    # gemini 3 does not support partial tool call, mock a partial tool call event
                    yield {
                        "role": "assistant",
                        "event_type": "delta",
                        "content_items": [
                            {
                                "type": "partial_tool_call",
                                "name": item["name"],
                                "arguments": json.dumps(item["arguments"], ensure_ascii=False),
                                "tool_call_id": item["tool_call_id"],
                                "signature": item.get("signature"),
                            }
                        ],
                        "usage_metadata": None,
                        "finish_reason": None,
                    }

            yield event
