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
from anthropic.types import MessageParam

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


class Claude4_5Client(LLMClient):
    """Claude 4.5-specific LLM client implementation."""

    def __init__(self, model: str, api_key: str | None = None):
        """Initialize Claude 4.5 client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(api_key=api_key)
        self._history: list[UniMessage] = []

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """
        Transform universal configuration to Claude-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            Claude configuration dictionary
        """
        claude_config: dict[str, Any] = {"model": self._model}

        # Add max_tokens (required for Claude)
        if config.get("max_tokens") is not None:
            claude_config["max_tokens"] = config["max_tokens"]
        else:
            claude_config["max_tokens"] = 4096  # Default value

        # Add temperature
        if config.get("temperature") is not None:
            claude_config["temperature"] = config["temperature"]

        # Add system prompt
        if config.get("system_prompt") is not None:
            claude_config["system"] = config["system_prompt"]

        # Convert thinking configuration
        # Claude uses "thinking" parameter with type "enabled" and "budget_tokens"
        thinking_summary = config.get("thinking_summary")
        thinking_level = config.get("thinking_level")
        if thinking_summary is not None or thinking_level is not None:
            # Calculate budget tokens based on thinking level
            budget_tokens = self._convert_thinking_level_to_budget(thinking_level)
            if budget_tokens > 0:
                claude_config["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}

        # Convert tools
        tools = config.get("tools")
        if tools is not None:
            claude_config["tools"] = tools

        # Convert tool_choice
        tool_choice = config.get("tool_choice")
        if tool_choice is not None:
            claude_config["tool_choice"] = self._convert_tool_choice(tool_choice)

        return claude_config

    def _convert_thinking_level_to_budget(self, thinking_level: ThinkingLevel | None) -> int:
        """Convert ThinkingLevel enum to Claude's budget_tokens."""
        if thinking_level is None:
            return 0

        mapping = {
            ThinkingLevel.NONE: 0,
            ThinkingLevel.LOW: 2000,
            ThinkingLevel.MEDIUM: 5000,
            ThinkingLevel.HIGH: 10000,
        }
        return mapping.get(thinking_level, 0)

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> dict[str, Any]:
        """Convert ToolChoice to Claude's tool_choice format."""
        if isinstance(tool_choice, list):
            # Claude doesn't support specific tool lists in the same way
            # Use "any" to force tool use
            return {"type": "any"}
        elif tool_choice == "none":
            return {"type": "none"}
        elif tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "required":
            return {"type": "any"}
        return {"type": "auto"}

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[MessageParam]:
        """
        Transform universal message format to Claude's MessageParam format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of Claude MessageParam objects
        """
        claude_messages: list[MessageParam] = []

        for msg in messages:
            role = msg["role"]
            content_blocks = []

            for item in msg["content_items"]:
                if item["type"] == "text":
                    # Check if this is a tool result
                    if role == "tool" and "tool_call_id" in item:
                        content_blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": item["tool_call_id"],
                                "content": item["text"],
                            }
                        )
                    else:
                        content_blocks.append({"type": "text", "text": item["text"]})

                elif item["type"] == "reasoning":
                    # Claude's thinking blocks
                    thinking_block: dict[str, Any] = {
                        "type": "thinking",
                        "thinking": item["reasoning"],
                    }
                    if "signature" in item and item["signature"]:
                        thinking_block["signature"] = item["signature"]
                    content_blocks.append(thinking_block)

                elif item["type"] == "image_url":
                    # Claude expects base64 encoded images
                    # For now, we assume the URL is accessible and should be fetched/encoded externally
                    # This is a known limitation - images should be base64 encoded before passing to Claude
                    content_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": item["image_url"],
                            },
                        }
                    )

                elif item["type"] == "function_call":
                    # Claude's tool_use blocks
                    try:
                        input_dict = json.loads(item["argument"])
                    except json.JSONDecodeError:
                        # If parsing fails, use empty dict
                        input_dict = {}

                    tool_use_block: dict[str, Any] = {
                        "type": "tool_use",
                        "id": item["tool_call_id"],
                        "name": item["name"],
                        "input": input_dict,
                    }
                    content_blocks.append(tool_use_block)

            # Map role
            if role == "tool":
                # Tool results are sent with user role
                claude_role = "user"
            elif role == "assistant":
                claude_role = "assistant"
            else:
                claude_role = "user"

            claude_messages.append({"role": claude_role, "content": content_blocks})

        return claude_messages

    def transform_model_output_to_uni_event(self, model_output: Any) -> UniEvent:
        """
        Transform Claude model output to universal event format.

        Args:
            model_output: Claude streaming event

        Returns:
            Universal event dictionary
        """
        content_items: list[ContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        event_type = model_output.type

        if event_type == "content_block_start":
            # New content block starting
            block = model_output.content_block
            if block.type == "thinking":
                content_items.append({"type": "reasoning", "reasoning": "", "signature": None})
            elif block.type == "text":
                content_items.append({"type": "text", "text": "", "signature": None})
            elif block.type == "tool_use":
                # Tool use block - store metadata for later delta events
                content_items.append(
                    {
                        "type": "function_call",
                        "name": block.name,
                        "argument": "",
                        "tool_call_id": block.id,
                        "signature": None,
                    }
                )

        elif event_type == "content_block_delta":
            # Content being streamed
            delta = model_output.delta
            if delta.type == "thinking_delta":
                content_items.append({"type": "reasoning", "reasoning": delta.thinking, "signature": None})
            elif delta.type == "text_delta":
                content_items.append({"type": "text", "text": delta.text, "signature": None})
            elif delta.type == "input_json_delta":
                # Tool input being streamed - we only stream the partial JSON
                # The name and id were already set in content_block_start
                # Here we just append the partial JSON for incremental parsing
                content_items.append({"type": "text", "text": delta.partial_json, "signature": None})
            elif delta.type == "signature_delta":
                # Signature for thinking block
                # This is attached as a separate event that should be merged
                content_items.append({"type": "text", "text": "", "signature": delta.signature})

        elif event_type == "content_block_stop":
            # Content block finished - nothing to add
            pass

        elif event_type == "message_start":
            # Message starting - extract usage if available
            message = model_output.message
            if hasattr(message, "usage") and message.usage:
                usage_metadata = {
                    "prompt_tokens": message.usage.input_tokens,
                    "thoughts_tokens": 0,  # Claude doesn't separate this
                    "response_tokens": message.usage.output_tokens,
                }

        elif event_type == "message_delta":
            # Message delta - may contain finish reason and usage
            delta = model_output.delta
            if hasattr(delta, "stop_reason") and delta.stop_reason:
                # Map Claude's finish reasons to our format
                stop_reason_mapping = {
                    "end_turn": "stop",
                    "max_tokens": "length",
                    "stop_sequence": "stop",
                    "tool_use": "stop",
                }
                finish_reason = stop_reason_mapping.get(delta.stop_reason, "unknown")

            # Update usage if available
            if hasattr(model_output, "usage") and model_output.usage:
                usage_metadata = {
                    "prompt_tokens": 0,  # Not in delta
                    "thoughts_tokens": 0,
                    "response_tokens": model_output.usage.output_tokens,
                }

        elif event_type == "message_stop":
            # Message finished - nothing to add
            pass

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
        async with self._client.messages.stream(**claude_config, messages=claude_messages) as stream:
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
