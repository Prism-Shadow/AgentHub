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
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from ..base_client import LLMClient
from ..errors import UnsupportedParameterError, parse_tool_call_arguments
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


class DeepSeekV4Client(LLMClient):
    """DeepSeek V4-specific LLM client implementation using OpenAI-compatible Chat Completions."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        """Initialize DeepSeek client with model, API key, and base URL."""
        self._model = model
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._history: list[UniMessage] = []

    def _convert_thinking_level_to_config(self, thinking_level: ThinkingLevel) -> dict[str, str]:
        """Convert ThinkingLevel enum to DeepSeek's thinking configuration."""
        mapping = {
            ThinkingLevel.NONE: {"type": "disabled"},
            ThinkingLevel.LOW: {"type": "enabled"},
            ThinkingLevel.MEDIUM: {"type": "enabled"},
            ThinkingLevel.HIGH: {"type": "enabled"},
            ThinkingLevel.XHIGH: {"type": "enabled"},
        }
        return mapping[thinking_level]

    def _convert_reasoning_effort(self, thinking_level: ThinkingLevel) -> str | None:
        """Convert ThinkingLevel enum to DeepSeek's reasoning_effort."""
        mapping = {
            ThinkingLevel.NONE: None,
            ThinkingLevel.LOW: "high",
            ThinkingLevel.MEDIUM: "high",
            ThinkingLevel.HIGH: "high",
            ThinkingLevel.XHIGH: "max",
        }
        return mapping[thinking_level]

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> str:
        """Convert ToolChoice to DeepSeek's OpenAI-compatible tool_choice format."""
        if tool_choice in ["auto", "none"]:
            return tool_choice
        raise UnsupportedParameterError(
            self.__class__.__name__, "tool_choice", "DeepSeek V4 only supports 'auto' and 'none' for tool_choice."
        )

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """
        Transform universal configuration to DeepSeek-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            DeepSeek configuration dictionary
        """
        deepseek_config = {"model": self._model, "stream": True, "stream_options": {"include_usage": True}}

        if config.get("max_tokens") is not None:
            deepseek_config["max_tokens"] = config["max_tokens"]

        if config.get("temperature") is not None and config["temperature"] != 1.0:
            raise UnsupportedParameterError(
                self.__class__.__name__, "temperature", "DeepSeek V4 does not support setting temperature."
            )

        thinking_level = config.get("thinking_level")
        if thinking_level is not None:
            deepseek_config["extra_body"] = {"thinking": self._convert_thinking_level_to_config(thinking_level)}
            reasoning_effort = self._convert_reasoning_effort(thinking_level)
            if reasoning_effort is not None:
                deepseek_config["reasoning_effort"] = reasoning_effort

        if config.get("tools") is not None:
            deepseek_config["tools"] = [{"type": "function", "function": tool} for tool in config["tools"]]

        if config.get("tool_choice") is not None:
            deepseek_config["tool_choice"] = self._convert_tool_choice(config["tool_choice"])

        if config.get("prompt_caching") is not None and config["prompt_caching"] != PromptCaching.ENABLE:
            raise UnsupportedParameterError(
                self.__class__.__name__, "prompt_caching", "prompt_caching must be ENABLE for DeepSeek."
            )

        return deepseek_config

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[ChatCompletionMessageParam]:
        """
        Transform universal message format to DeepSeek's OpenAI-compatible message format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of OpenAI-compatible message dictionaries
        """
        deepseek_messages = []

        for msg in messages:
            content_parts = []  # may be empty for tool results
            tool_calls = []  # may be empty for no tool calls
            thinking = ""
            for item in msg["content_items"]:
                if item["type"] == "text":
                    content_parts.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    raise ValueError("DeepSeek does not support image url inputs.")
                elif item["type"] == "thinking":
                    thinking += item["thinking"]
                elif item["type"] == "tool_call":
                    tool_calls.append(
                        {
                            "id": item["tool_call_id"],
                            "type": "function",
                            "function": {
                                "name": item["name"],
                                "arguments": json.dumps(item["arguments"], ensure_ascii=False),
                            },
                        }
                    )
                elif item["type"] == "tool_result":
                    if "tool_call_id" not in item:
                        raise ValueError("tool_call_id is required for tool result.")

                    content = [{"type": "text", "text": item["text"]}]

                    if "images" in item and item["images"]:
                        raise ValueError("DeepSeek does not support images in tool results.")

                    # Tool results are sent as separate messages
                    deepseek_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": item["tool_call_id"],
                            "content": content,
                        }
                    )
                else:
                    raise ValueError(f"Unknown item type: {item['type']}")

            message = {"role": msg["role"]}
            if content_parts:
                message["content"] = content_parts

            if tool_calls:
                message["tool_calls"] = tool_calls

            if thinking:
                message["reasoning_content"] = thinking

            # message may be empty for tool results
            if len(message.keys()) > 1:
                deepseek_messages.append(message)

        return deepseek_messages

    def transform_model_output_to_uni_event(self, model_output: ChatCompletionChunk) -> UniEvent:
        """
        Transform DeepSeek streaming chunk to universal event format.

        Args:
            model_output: OpenAI-compatible streaming chunk

        Returns:
            Universal event dictionary
        """
        event_type: EventType | None = None
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        if len(model_output.choices) > 0:
            choice = model_output.choices[0]
            delta = choice.delta

            if getattr(delta, "reasoning_content", None):
                event_type = "delta"
                # record the wire field so a replay through another OpenAI-compatible
                # client reproduces the exact field DeepSeek produced
                content_items.append(
                    {
                        "type": "thinking",
                        "thinking": getattr(delta, "reasoning_content"),
                        "fidelity": {"reasoning_field": "reasoning_content"},
                    }
                )

            if delta.content:
                event_type = "delta"
                content_items.append({"type": "text", "text": delta.content})

            if delta.tool_calls:
                event_type = "delta"
                for tool_call in delta.tool_calls:
                    content_item: PartialContentItem = {
                        "type": "partial_tool_call",
                        "name": tool_call.function.name or "",
                        "arguments": tool_call.function.arguments or "",
                        "tool_call_id": tool_call.id or "",
                    }
                    content_items.append(content_item)

            if choice.finish_reason:
                event_type = event_type or "stop"
                finish_reason_mapping = {
                    "stop": "stop",
                    "length": "length",
                    "tool_calls": "tool_call",
                    "content_filter": "stop",
                }
                finish_reason = finish_reason_mapping.get(choice.finish_reason, "unknown")

        if model_output.usage:
            event_type = event_type or "stop"
            completion_token_details = model_output.usage.completion_tokens_details
            reasoning_tokens = (
                getattr(completion_token_details, "reasoning_tokens", None) if completion_token_details else None
            )
            response_tokens = model_output.usage.completion_tokens - (reasoning_tokens or 0)

            # usage.prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens
            usage_metadata = {
                "cached_tokens": getattr(model_output.usage, "prompt_cache_hit_tokens", 0),
                "prompt_tokens": getattr(model_output.usage, "prompt_cache_miss_tokens", 0),
                "thoughts_tokens": reasoning_tokens,
                "response_tokens": response_tokens,
            }

        return {
            "role": "assistant",
            "event_type": event_type,
            "content_items": content_items,
            "usage_metadata": usage_metadata,
            "finish_reason": finish_reason,
        }

    async def _streaming_response_internal(
        self,
        messages: list[UniMessage],
        config: UniConfig,
    ) -> AsyncIterator[UniEvent]:
        """Stream generate using DeepSeek's OpenAI-compatible Chat Completions API."""
        deepseek_config = self.transform_uni_config_to_model_config(config)
        deepseek_messages = self.transform_uni_message_to_model_input(messages)

        if config.get("system_prompt"):
            deepseek_messages.insert(0, {"role": "system", "content": config["system_prompt"]})

        stream = await self._client.chat.completions.create(**deepseek_config, messages=deepseek_messages)

        partial_tool_call = {}
        partial_usage = {}
        async for chunk in stream:
            event = self.transform_model_output_to_uni_event(chunk)
            partial_usage["finish_reason"] = event["finish_reason"] or partial_usage.get("finish_reason")
            partial_usage["usage_metadata"] = event["usage_metadata"] or partial_usage.get("usage_metadata")

            if event["event_type"] == "delta":
                for item in event["content_items"]:
                    if item["type"] == "partial_tool_call":
                        if not partial_tool_call:
                            partial_tool_call = {
                                "name": item["name"],
                                "arguments": item["arguments"],
                                "tool_call_id": item["tool_call_id"],
                            }
                        elif item["name"]:
                            yield {
                                "role": "assistant",
                                "event_type": "delta",
                                "content_items": [
                                    {
                                        "type": "tool_call",
                                        "name": partial_tool_call["name"],
                                        "arguments": parse_tool_call_arguments(
                                            partial_tool_call["arguments"],
                                            self.__class__.__name__,
                                            partial_tool_call["name"],
                                            partial_tool_call["tool_call_id"],
                                        ),
                                        "tool_call_id": partial_tool_call["tool_call_id"],
                                    }
                                ],
                                "usage_metadata": None,
                                "finish_reason": None,
                            }
                            partial_tool_call = {
                                "name": item["name"],
                                "arguments": item["arguments"],
                                "tool_call_id": item["tool_call_id"],
                            }
                        else:
                            partial_tool_call["arguments"] += item["arguments"]
                            partial_tool_call["tool_call_id"] = (
                                item["tool_call_id"] or partial_tool_call["tool_call_id"]
                            )

                yield event
            elif event["event_type"] == "stop":
                if partial_tool_call:
                    yield {
                        "role": "assistant",
                        "event_type": "delta",
                        "content_items": [
                            {
                                "type": "tool_call",
                                "name": partial_tool_call["name"],
                                "arguments": parse_tool_call_arguments(
                                    partial_tool_call["arguments"],
                                    self.__class__.__name__,
                                    partial_tool_call["name"],
                                    partial_tool_call["tool_call_id"],
                                ),
                                "tool_call_id": partial_tool_call["tool_call_id"],
                            }
                        ],
                        "usage_metadata": None,
                        "finish_reason": None,
                    }
                    partial_tool_call = {}

                if partial_usage.get("finish_reason") and partial_usage.get("usage_metadata"):
                    yield {
                        "role": "assistant",
                        "event_type": "stop",
                        "content_items": event["content_items"],
                        "usage_metadata": partial_usage["usage_metadata"],
                        "finish_reason": partial_usage["finish_reason"],
                    }
                    partial_usage = {}
