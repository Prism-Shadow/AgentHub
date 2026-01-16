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

from anthropic import AsyncAnthropic
from anthropic.types import (
    Message,
    MessageStreamEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    TextBlock,
    ToolUseBlock,
)

from .base_client import LLMClient
from .types import ContentItem, FinishReason, ThinkingLevel, ToolChoice, UniConfig, UniEvent, UniMessage, UsageMetadata


class ClaudeClient(LLMClient):
    """Claude 4.5-specific LLM client implementation."""

    def __init__(self, model: str, api_key: str | None = None):
        """Initialize Claude client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()
        self._history: list[UniMessage] = []

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """
        Transform universal configuration to Claude-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            Claude configuration dictionary
        """
        config_params: dict[str, Any] = {}

        # System prompt
        if config.get("system_prompt") is not None:
            config_params["system"] = config["system_prompt"]

        # Max tokens
        if config.get("max_tokens") is not None:
            config_params["max_tokens"] = config["max_tokens"]

        # Temperature
        if config.get("temperature") is not None:
            config_params["temperature"] = config["temperature"]

        # Extended thinking configuration
        thinking_level = config.get("thinking_level")
        if thinking_level is not None:
            budget_tokens = self._get_thinking_budget(thinking_level)
            config_params["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}

        # Tools configuration
        tools = config.get("tools")
        if tools is not None:
            config_params["tools"] = tools
            tool_choice = config.get("tool_choice")
            if tool_choice is not None:
                config_params["tool_choice"] = self._convert_tool_choice(tool_choice)

        return config_params

    def _get_thinking_budget(self, thinking_level: ThinkingLevel) -> int:
        """Convert ThinkingLevel enum to Claude's thinking budget."""
        mapping = {
            ThinkingLevel.NONE: 1024,  # Minimum budget
            ThinkingLevel.LOW: 4000,
            ThinkingLevel.MEDIUM: 10000,
            ThinkingLevel.HIGH: 16000,
        }
        return mapping.get(thinking_level, 1024)

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> dict[str, Any]:
        """Convert ToolChoice to Claude's tool choice format."""
        if isinstance(tool_choice, list):
            # Claude doesn't support specific tool selection via list
            # Use "any" to force tool use
            return {"type": "any"}
        elif tool_choice == "none":
            return {"type": "none"}
        elif tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "required":
            return {"type": "any"}
        return {"type": "auto"}

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[dict[str, Any]]:
        """
        Transform universal message format to Claude's message format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of Claude message dictionaries
        """
        claude_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "tool":
                # Tool messages become user messages with tool_result content
                content = []
                for item in msg["content_items"]:
                    if item["type"] == "text":
                        content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": item.get("tool_call_id", ""),
                                "content": item["text"],
                            }
                        )
                claude_messages.append({"role": "user", "content": content})
            else:
                # Assistant and user messages
                content = []
                for item in msg["content_items"]:
                    if item["type"] == "text":
                        content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "reasoning":
                        # Reasoning blocks are thinking blocks in Claude
                        content.append(
                            {
                                "type": "thinking",
                                "thinking": item["reasoning"],
                                **({"signature": item["signature"]} if "signature" in item else {}),
                            }
                        )
                    elif item["type"] == "image_url":
                        # Claude uses base64 image format
                        content.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": item["image_url"]},
                            }
                        )
                    elif item["type"] == "function_call":
                        content.append(
                            {
                                "type": "tool_use",
                                "id": item.get("tool_call_id", ""),
                                "name": item["name"],
                                "input": json.loads(item["argument"]),
                            }
                        )
                claude_messages.append({"role": role, "content": content})

        return claude_messages

    def transform_model_output_to_uni_event(self, model_output: MessageStreamEvent) -> UniEvent:
        """
        Transform Claude streaming event to universal event format.

        Args:
            model_output: Claude streaming event

        Returns:
            Universal event dictionary
        """
        content_items: list[ContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        # Handle different event types
        if isinstance(model_output, RawContentBlockStartEvent):
            # New content block started
            block = model_output.content_block
            if hasattr(block, "type"):
                if block.type == "text":
                    content_items.append({"type": "text", "text": ""})
                elif block.type == "thinking":
                    content_items.append({"type": "reasoning", "reasoning": ""})
                elif block.type == "tool_use":
                    tool_block = block
                    content_items.append(
                        {
                            "type": "function_call",
                            "name": tool_block.name,
                            "argument": "",
                            "tool_call_id": tool_block.id,
                        }
                    )

        elif isinstance(model_output, RawContentBlockDeltaEvent):
            # Content block delta
            delta = model_output.delta
            if delta.type == "text_delta":
                content_items.append({"type": "text", "text": delta.text})
            elif delta.type == "thinking_delta":
                content_items.append({"type": "reasoning", "reasoning": delta.thinking})
            elif delta.type == "input_json_delta":
                # Tool input delta - accumulate as argument
                content_items.append(
                    {
                        "type": "function_call",
                        "name": "",
                        "argument": delta.partial_json,
                        "tool_call_id": "",
                    }
                )

        return {
            "role": "assistant",
            "content_items": content_items,
            "usage_metadata": usage_metadata,
            "finish_reason": finish_reason,
        }

    def _transform_message_to_uni_message(self, message: Message) -> UniMessage:
        """Transform a complete Claude Message to universal message format."""
        content_items: list[ContentItem] = []

        for block in message.content:
            if isinstance(block, TextBlock):
                content_items.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                content_items.append(
                    {
                        "type": "function_call",
                        "name": block.name,
                        "argument": json.dumps(block.input),
                        "tool_call_id": block.id,
                    }
                )
            elif hasattr(block, "type") and block.type == "thinking":
                content_items.append(
                    {
                        "type": "reasoning",
                        "reasoning": block.thinking,
                        "signature": block.signature if hasattr(block, "signature") else "",
                    }
                )

        usage_metadata: UsageMetadata = {
            "prompt_tokens": message.usage.input_tokens,
            "thoughts_tokens": None,
            "response_tokens": message.usage.output_tokens,
        }

        # Map stop_reason to finish_reason
        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish_reason: FinishReason = finish_reason_map.get(message.stop_reason, "unknown")

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
        """Stream generate using Claude SDK with unified conversion methods."""
        # Use unified config conversion
        claude_config = self.transform_uni_config_to_model_config(config)

        # Use unified message conversion
        claude_messages = self.transform_uni_message_to_model_input(messages)

        # Stream generate
        async with self._client.messages.stream(
            model=self._model, messages=claude_messages, **claude_config
        ) as stream:
            async for event in stream:
                yield self.transform_model_output_to_uni_event(event)

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
