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
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from ..base_client import LLMClient
from ..types import (
    FinishReason,
    PartialContentItem,
    PartialUniEvent,
    ThinkingLevel,
    ToolChoice,
    UniConfig,
    UniEvent,
    UniMessage,
    UsageMetadata,
)


class GLM4_7Client(LLMClient):
    """GLM-4.7-specific LLM client implementation using OpenAI-compatible API."""

    def __init__(self, model: str, api_key: str | None = None):
        """Initialize GLM-4.7 client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("GLM_API_KEY")
        base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._history: list[UniMessage] = []

    def _convert_thinking_level_to_config(self, thinking_level: ThinkingLevel) -> dict:
        """Convert ThinkingLevel enum to GLM's thinking configuration."""
        mapping = {
            ThinkingLevel.NONE: {"type": "disabled"},
            ThinkingLevel.LOW: {"type": "enabled"},
            ThinkingLevel.MEDIUM: {"type": "enabled"},
            ThinkingLevel.HIGH: {"type": "enabled"},
        }
        return mapping.get(thinking_level)

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> str | dict[str, Any]:
        """Convert ToolChoice to OpenAI's tool_choice format."""
        if isinstance(tool_choice, list):
            raise ValueError("GLM only supports 'auto' for tool_choice.")
        elif tool_choice == "auto":
            return "auto"
        else:
            raise ValueError("GLM only supports 'auto' for tool_choice.")

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """
        Transform universal configuration to GLM-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            GLM configuration dictionary
        """
        glm_config = {"model": self._model, "stream": True}

        # Add max_tokens
        if config.get("max_tokens") is not None:
            glm_config["max_tokens"] = config["max_tokens"]

        # Add temperature
        if config.get("temperature") is not None:
            glm_config["temperature"] = config["temperature"]

        # Prepare extra_body for additional parameters
        extra_body = {}

        # Convert thinking configuration
        if config.get("thinking_level") is not None:
            thinking_config = self._convert_thinking_level_to_config(config["thinking_level"])
            extra_body["thinking"] = thinking_config

        # Add extra_body if not empty
        if extra_body:
            glm_config["extra_body"] = extra_body

        # Convert tools to OpenAI's tool schema
        if config.get("tools") is not None:
            glm_tools = []
            for tool in config["tools"]:
                glm_tool = {"type": "function", "function": tool}
                glm_tools.append(glm_tool)
            glm_config["tools"] = glm_tools

        # Convert tool_choice
        if config.get("tool_choice") is not None:
            glm_config["tool_choice"] = self._convert_tool_choice(config["tool_choice"])

        return glm_config

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[dict[str, Any]]:
        """
        Transform universal message format to OpenAI's message format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of OpenAI message dictionaries
        """
        openai_messages = []

        for msg in messages:
            content_parts = []
            tool_calls = []

            for item in msg["content_items"]:
                if item["type"] == "text":
                    content_parts.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    content_parts.append({"type": "image_url", "image_url": {"url": item["image_url"]}})
                elif item["type"] == "thinking":
                    # GLM stores thinking in reasoning_content
                    pass  # Handled separately in message construction
                elif item["type"] == "tool_call":
                    tool_calls.append(
                        {
                            "id": item["tool_call_id"],
                            "type": "function",
                            "function": {"name": item["name"], "arguments": json.dumps(item["argument"])},
                        }
                    )
                elif item["type"] == "tool_result":
                    # Tool results are sent as separate messages
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": item["tool_call_id"],
                            "content": item["result"],
                        }
                    )
                    continue
                else:
                    raise ValueError(f"Unknown item type: {item['type']}")

            # Build the message
            message = {"role": msg["role"]}

            # Set content
            if content_parts:
                message["content"] = content_parts
            else:
                message["content"] = ""

            # Add tool calls if present
            if tool_calls:
                message["tool_calls"] = tool_calls

            # Add reasoning content for thinking
            for item in msg["content_items"]:
                if item["type"] == "thinking":
                    message["reasoning_content"] = item["thinking"]
                    break

            openai_messages.append(message)

        return openai_messages

    def transform_model_output_to_uni_event(self, model_output: Any) -> PartialUniEvent:
        """
        Transform GLM model output to universal event format.

        Args:
            model_output: OpenAI streaming chunk

        Returns:
            Universal event dictionary
        """
        event_type = None
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        if not model_output.choices:
            event_type = "stop"
            return {
                "role": "assistant",
                "event": event_type,
                "content_items": content_items,
                "usage_metadata": usage_metadata,
                "finish_reason": finish_reason,
            }

        choice = model_output.choices[0]
        delta = choice.delta

        # Check for reasoning content (thinking)
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            event_type = "delta"
            content_items.append({"type": "thinking", "thinking": delta.reasoning_content})

        # Check for text content
        if hasattr(delta, "content") and delta.content:
            event_type = "delta"
            content_items.append({"type": "text", "text": delta.content})

        # Check for tool calls
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            event_type = "delta"
            for tool_call in delta.tool_calls:
                if tool_call.function:
                    content_items.append(
                        {
                            "type": "tool_call",
                            "name": tool_call.function.name,
                            "argument": json.loads(tool_call.function.arguments),
                            "tool_call_id": tool_call.id,
                        }
                    )

        # Check for finish reason
        if choice.finish_reason:
            event_type = "stop"
            finish_reason_mapping = {
                "stop": "stop",
                "length": "length",
                "tool_calls": "stop",
                "content_filter": "stop",
            }
            finish_reason = finish_reason_mapping.get(choice.finish_reason, "unknown")

        # Check for usage metadata
        if hasattr(model_output, "usage") and model_output.usage:
            completion_tokens_details = getattr(model_output.usage, "completion_tokens_details", None)
            reasoning_tokens = None
            if completion_tokens_details:
                reasoning_tokens = completion_tokens_details.reasoning_tokens

            usage_metadata = {
                "prompt_tokens": model_output.usage.prompt_tokens,
                "thoughts_tokens": reasoning_tokens,
                "response_tokens": model_output.usage.completion_tokens,
            }

        event_type = "delta"

        return {
            "role": "assistant",
            "event": event_type,
            "content_items": content_items,
            "usage_metadata": usage_metadata,
            "finish_reason": finish_reason,
        }

    async def streaming_response(
        self,
        messages: list[UniMessage],
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        """Stream generate using GLM SDK with unified conversion methods."""
        # Use unified config conversion
        glm_config = self.transform_uni_config_to_model_config(config)

        # Use unified message conversion
        glm_messages = self.transform_uni_message_to_model_input(messages)

        # Extract system prompt if present
        if config.get("system_prompt"):
            glm_messages.insert(0, {"role": "system", "content": config["system_prompt"]})

        # Stream generate
        stream = await self._client.chat.completions.create(**glm_config, messages=glm_messages)

        async for chunk in stream:
            event = self.transform_model_output_to_uni_event(chunk)

            if event["event"] == "delta":
                event.pop("event")
                yield event
