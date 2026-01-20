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
from anthropic.types import MessageParam, MessageStreamEvent

from ..base_client import LLMClient
from ..types import (
    FinishReason,
    PartialContentItem,
    PartialUniEvent,
    PromptCaching,
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

    def _convert_thinking_level_to_budget(self, thinking_level: ThinkingLevel) -> dict:
        """Convert ThinkingLevel enum to Claude's budget_tokens."""

        mapping = {
            ThinkingLevel.NONE: {"type": "disabled"},
            ThinkingLevel.LOW: {"type": "enabled", "budget_tokens": 1024},
            ThinkingLevel.MEDIUM: {"type": "enabled", "budget_tokens": 4096},
            ThinkingLevel.HIGH: {"type": "enabled", "budget_tokens": 16384},
        }
        return mapping.get(thinking_level)

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> dict[str, Any]:
        """Convert ToolChoice to Claude's tool_choice format."""
        if isinstance(tool_choice, list):
            if len(tool_choice) > 1:
                raise ValueError("Claude supports only one tool choice.")

            return {"type": "any", "name": tool_choice[0]}
        elif tool_choice == "none":
            return {"type": "none"}
        elif tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "required":
            return {"type": "any"}

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """
        Transform universal configuration to Claude-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            Claude configuration dictionary
        """
        claude_config = {"model": self._model}

        if config.get("system_prompt") is not None:
            claude_config["system"] = config["system_prompt"]

        if config.get("max_tokens") is not None:
            claude_config["max_tokens"] = config["max_tokens"]
        else:
            claude_config["max_tokens"] = 32768  # Claude requires max_tokens to be specified

        if config.get("temperature") is not None:
            claude_config["temperature"] = config["temperature"]

        # Convert thinking configuration
        # NOTE: Claude always provides thinking summary
        if config.get("thinking_level") is not None:
            claude_config["temperature"] = 1.0  # `temperature` may only be set to 1 when thinking is enabled
            claude_config["thinking"] = self._convert_thinking_level_to_budget(config["thinking_level"])

        # Convert tools to Claude's tool schema
        if config.get("tools") is not None:
            claude_tools = []
            for tool in config["tools"]:
                claude_tool = {}
                for key, value in tool.items():
                    claude_tool[key.replace("parameters", "input_schema")] = value

                claude_tools.append(claude_tool)

            claude_config["tools"] = claude_tools

        # Convert tool_choice
        if config.get("tool_choice") is not None:
            claude_config["tool_choice"] = self._convert_tool_choice(config["tool_choice"])

        return claude_config

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
            content_blocks = []
            for item in msg["content_items"]:
                if item["type"] == "text":
                    content_blocks.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    # TODO: support base64 encoded images
                    content_blocks.append({"type": "image", "source": {"type": "url", "url": item["image_url"]}})
                elif item["type"] == "thinking":
                    content_blocks.append(
                        {"type": "thinking", "thinking": item["thinking"], "signature": item["signature"]}
                    )
                elif item["type"] == "tool_call":
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": item["tool_call_id"],
                            "name": item["name"],
                            "input": item["argument"],
                        }
                    )
                elif item["type"] == "tool_result":
                    if "tool_call_id" not in item:
                        raise ValueError("tool_call_id is required for tool result.")

                    content_blocks.append(
                        {"type": "tool_result", "content": item["result"], "tool_use_id": item["tool_call_id"]}
                    )
                else:
                    raise ValueError(f"Unknown item: {item}")

            claude_messages.append({"role": msg["role"], "content": content_blocks})

        return claude_messages

    def transform_model_output_to_uni_event(self, model_output: MessageStreamEvent) -> PartialUniEvent:
        """
        Transform Claude model output to universal event format.

        NOTE: Claude always has only one content item per event.

        Args:
            model_output: Claude streaming event

        Returns:
            Universal event dictionary
        """
        event_type = None
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        claude_event_type = model_output.type
        if claude_event_type == "content_block_start":
            event_type = "start"
            block = model_output.content_block
            if block.type == "tool_use":
                content_items.append(
                    {"type": "partial_tool_call", "name": block.name, "argument": "", "tool_call_id": block.id}
                )

        elif claude_event_type == "content_block_delta":
            event_type = "delta"
            delta = model_output.delta
            if delta.type == "thinking_delta":
                content_items.append({"type": "thinking", "thinking": delta.thinking})
            elif delta.type == "text_delta":
                content_items.append({"type": "text", "text": delta.text})
            elif delta.type == "input_json_delta":
                content_items.append({"type": "partial_tool_call", "argument": delta.partial_json})
            elif delta.type == "signature_delta":
                content_items.append({"type": "thinking", "thinking": "", "signature": delta.signature})

        elif claude_event_type == "content_block_stop":
            event_type = "stop"

        elif claude_event_type == "message_start":
            event_type = "start"
            message = model_output.message
            if getattr(message, "usage", None):
                # cached_tokens is cache_read_input_tokens from Claude API
                cached_tokens = message.usage.cache_read_input_tokens if hasattr(message.usage, "cache_read_input_tokens") else None
                usage_metadata = {
                    "prompt_tokens": message.usage.input_tokens,
                    "thoughts_tokens": None,
                    "response_tokens": None,
                    "cached_tokens": cached_tokens,
                }

        elif claude_event_type == "message_delta":
            event_type = "stop"
            delta = model_output.delta
            if getattr(delta, "stop_reason", None):
                stop_reason_mapping = {
                    "end_turn": "stop",
                    "max_tokens": "length",
                    "stop_sequence": "stop",
                    "tool_use": "stop",
                }
                finish_reason = stop_reason_mapping.get(delta.stop_reason, "unknown")

            if getattr(model_output, "usage", None):
                usage_metadata = {
                    "prompt_tokens": None,
                    "thoughts_tokens": None,
                    "response_tokens": model_output.usage.output_tokens,
                    "cached_tokens": None,
                }

        elif claude_event_type == "message_stop":
            event_type = "stop"

        elif claude_event_type in ["text", "thinking", "signature", "input_json"]:
            event_type = "unused"

        else:
            raise ValueError(f"Unknown output: {model_output}")

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
        """Stream generate using Claude SDK with unified conversion methods."""
        # Use unified config conversion
        claude_config = self.transform_uni_config_to_model_config(config)

        # Use unified message conversion
        claude_messages = self.transform_uni_message_to_model_input(messages)

        # Add cache_control to last user message's last item if enabled
        prompt_cache = config.get("prompt_cache", PromptCaching.ENABLE)
        if prompt_cache != PromptCaching.DISABLE and claude_messages:
            # Find last user message
            for i in range(len(claude_messages) - 1, -1, -1):
                if claude_messages[i]["role"] == "user":
                    content = claude_messages[i]["content"]
                    if content and isinstance(content, list):
                        # Add cache_control to last content item
                        last_item = content[-1]
                        if last_item.get("type") == "text":
                            cache_control = {"type": "ephemeral"}
                            if prompt_cache == PromptCaching.ENHANCE:
                                cache_control["ttl"] = "1h"
                            last_item["cache_control"] = cache_control
                    break

        # Stream generate
        partial_tool_call = {}
        partial_usage = {}
        async with self._client.messages.stream(**claude_config, messages=claude_messages) as stream:
            async for event in stream:
                event = self.transform_model_output_to_uni_event(event)
                if event["event"] == "start":
                    if event["content_items"] and event["content_items"][0]["type"] == "partial_tool_call":
                        partial_tool_call["name"] = event["content_items"][0]["name"]
                        partial_tool_call["argument"] = ""
                        partial_tool_call["tool_call_id"] = event["content_items"][0]["tool_call_id"]

                    if event["usage_metadata"] is not None:
                        partial_usage["prompt_tokens"] = event["usage_metadata"]["prompt_tokens"]
                        partial_usage["cached_tokens"] = event["usage_metadata"]["cached_tokens"]

                elif event["event"] == "delta":
                    if event["content_items"][0]["type"] == "partial_tool_call":
                        partial_tool_call["argument"] += event["content_items"][0]["argument"]
                    else:
                        event.pop("event")
                        yield event

                elif event["event"] == "stop":
                    if "name" in partial_tool_call and "argument" in partial_tool_call:
                        yield {
                            "role": "assistant",
                            "content_items": [
                                {
                                    "type": "tool_call",
                                    "name": partial_tool_call["name"],
                                    "argument": json.loads(partial_tool_call["argument"]),
                                    "tool_call_id": partial_tool_call["tool_call_id"],
                                }
                            ],
                        }
                        partial_tool_call = {}

                    if "prompt_tokens" in partial_usage and event["usage_metadata"] is not None:
                        yield {
                            "role": "assistant",
                            "content_items": [],
                            "usage_metadata": {
                                "prompt_tokens": partial_usage["prompt_tokens"],
                                "thoughts_tokens": None,
                                "response_tokens": event["usage_metadata"]["response_tokens"],
                                "cached_tokens": partial_usage.get("cached_tokens", None),
                            },
                            "finish_reason": event["finish_reason"],
                        }
                        partial_usage = {}
