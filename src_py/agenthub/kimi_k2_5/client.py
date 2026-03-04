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
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

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
from ..utils import fix_openrouter_usage_metadata


class KimiK2_5Client(LLMClient):
    """Kimi K2.5-specific LLM client implementation using OpenAI-compatible API."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        """Initialize Kimi K2.5 client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        base_url = base_url or os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._history: list[UniMessage] = []

    async def _convert_image_url_to_base64(self, url: str) -> str:
        """Convert image URL to base64-encoded string.

        Args:
            url: Image URL to convert

        Returns:
            Base64-encoded image string
        """
        if url.startswith("data:"):
            return url

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            image_bytes = response.content
            mime_type = mimetypes.guess_type(url)[0] or "image/jpeg"
            base64_string = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:{mime_type};base64,{base64_string}"

    def _convert_thinking_level_to_config(self, thinking_level: ThinkingLevel) -> dict[str, str]:
        """Convert ThinkingLevel enum to Kimi's thinking configuration."""
        mapping = {
            ThinkingLevel.NONE: {"type": "disabled"},
            ThinkingLevel.LOW: {"type": "enabled"},
            ThinkingLevel.MEDIUM: {"type": "enabled"},
            ThinkingLevel.HIGH: {"type": "enabled"},
        }
        return mapping.get(thinking_level)

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> str:
        """Convert ToolChoice to OpenAI's tool_choice format."""
        if tool_choice == "auto":
            return "auto"
        elif tool_choice == "none":
            return "none"
        else:
            raise ValueError("Kimi only supports 'auto' and 'none' for tool_choice.")

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """
        Transform universal configuration to Kimi-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            Kimi configuration dictionary
        """
        kimi_config = {"model": self._model, "stream": True, "stream_options": {"include_usage": True}}

        if config.get("max_tokens") is not None:
            kimi_config["max_tokens"] = config["max_tokens"]

        if config.get("temperature") is not None and config["temperature"] != 1.0:
            raise ValueError("Kimi K2.5 does not support setting temperature.")

        if config.get("thinking_level") is not None:
            thinking_config = self._convert_thinking_level_to_config(config["thinking_level"])
            kimi_config.setdefault("extra_body", {})["thinking"] = thinking_config

        if config.get("tools") is not None:
            kimi_config["tools"] = [{"type": "function", "function": tool} for tool in config["tools"]]

        if config.get("tool_choice") is not None:
            kimi_config["tool_choice"] = self._convert_tool_choice(config["tool_choice"])

        if config.get("prompt_caching") is not None and config["prompt_caching"] != PromptCaching.ENABLE:
            raise ValueError("prompt_caching must be ENABLE for Kimi K2.5.")

        if config.get("trace_id") is not None:  # use trace_id as the prompt cache key
            kimi_config["prompt_cache_key"] = config["trace_id"]

        return kimi_config

    async def transform_uni_message_to_model_input(
        self, messages: list[UniMessage]
    ) -> list[ChatCompletionMessageParam]:
        """
        Transform universal message format to OpenAI's message format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of OpenAI message dictionaries
        """
        openai_messages = []

        for msg in messages:
            content_parts = []  # may be empty for tool results
            tool_calls = []  # may be empty for no tool calls
            thinking = ""
            for item in msg["content_items"]:
                if item["type"] == "text":
                    content_parts.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    base64_image = await self._convert_image_url_to_base64(item["image_url"])
                    content_parts.append({"type": "image_url", "image_url": {"url": base64_image}})
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
                        for image_url in item["images"]:
                            base64_image = await self._convert_image_url_to_base64(image_url)
                            if "siliconflow.cn" in str(self._client.base_url):
                                # siliconflow does not support image_url in tool result
                                content_parts.append({"type": "image_url", "image_url": {"url": base64_image}})
                            else:
                                content.append({"type": "image_url", "image_url": {"url": base64_image}})

                    # Tool results are sent as separate messages
                    openai_messages.append(
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
                message["reasoning_content"] = thinking  # vLLM & siliconflow compatibility
                message["reasoning"] = thinking  # openrouter compatibility

            # message may be empty for tool results
            if len(message.keys()) > 1:
                openai_messages.append(message)

        return openai_messages

    def transform_model_output_to_uni_event(self, model_output: ChatCompletionChunk) -> UniEvent:
        """
        Transform Kimi model output to universal event format.

        Args:
            model_output: OpenAI streaming chunk

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

            if delta.content:
                event_type = "delta"
                content_items.append({"type": "text", "text": delta.content})

            # vLLM & siliconflow compatibility
            if getattr(delta, "reasoning_content", None):
                event_type = "delta"
                content_items.append({"type": "thinking", "thinking": getattr(delta, "reasoning_content")})

            # openrouter compatibility
            elif getattr(delta, "reasoning", None):
                event_type = "delta"
                content_items.append({"type": "thinking", "thinking": getattr(delta, "reasoning")})

            if delta.tool_calls:
                event_type = "delta"
                for tool_call in delta.tool_calls:
                    content_items.append(
                        {
                            "type": "partial_tool_call",
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                            "tool_call_id": tool_call.id,
                        }
                    )

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
            event_type = event_type or "stop"  # deal with separate usage data

            if model_output.usage.prompt_tokens_details:
                cached_tokens = model_output.usage.prompt_tokens_details.cached_tokens
            else:
                cached_tokens = None

            if model_output.usage.completion_tokens_details:
                reasoning_tokens = model_output.usage.completion_tokens_details.reasoning_tokens
            else:
                reasoning_tokens = None

            if cached_tokens is not None:
                prompt_tokens = model_output.usage.prompt_tokens - cached_tokens
            else:
                prompt_tokens = model_output.usage.prompt_tokens

            if reasoning_tokens is not None:
                response_tokens = model_output.usage.completion_tokens - reasoning_tokens
            else:
                response_tokens = model_output.usage.completion_tokens

            usage_metadata = {
                "cached_tokens": cached_tokens,
                "prompt_tokens": prompt_tokens,
                "thoughts_tokens": reasoning_tokens,
                "response_tokens": response_tokens,
            }
            usage_metadata = fix_openrouter_usage_metadata(usage_metadata, str(self._client.base_url))

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
        """Stream generate using Kimi SDK with unified conversion methods."""
        kimi_config = self.transform_uni_config_to_model_config(config)
        kimi_messages = await self.transform_uni_message_to_model_input(messages)

        # Extract system prompt if present
        if config.get("system_prompt"):
            kimi_messages.insert(0, {"role": "system", "content": config["system_prompt"]})

        # Stream generate
        stream = await self._client.chat.completions.create(**kimi_config, messages=kimi_messages)

        partial_tool_call = {}
        partial_usage = {}
        async for chunk in stream:
            event = self.transform_model_output_to_uni_event(chunk)
            # the finish reason and usage metadata should be accumulated
            partial_usage["finish_reason"] = event["finish_reason"] or partial_usage.get("finish_reason")
            partial_usage["usage_metadata"] = event["usage_metadata"] or partial_usage.get("usage_metadata")
            if event["event_type"] == "delta":
                for item in event["content_items"]:
                    if item["type"] == "partial_tool_call":
                        if not partial_tool_call:
                            # start new partial tool call
                            partial_tool_call = {
                                "name": item["name"],
                                "arguments": item["arguments"],
                                "tool_call_id": item["tool_call_id"],
                            }
                        elif item["name"]:
                            # finish previous partial tool call
                            yield {
                                "role": "assistant",
                                "event_type": "delta",
                                "content_items": [
                                    {
                                        "type": "tool_call",
                                        "name": partial_tool_call["name"],
                                        "arguments": json.loads(partial_tool_call["arguments"] or "{}"),
                                        "tool_call_id": partial_tool_call["tool_call_id"],
                                    }
                                ],
                                "usage_metadata": None,
                                "finish_reason": None,
                            }
                            # start new partial tool call
                            partial_tool_call = {
                                "name": item["name"],
                                "arguments": item["arguments"],
                                "tool_call_id": item["tool_call_id"],
                            }
                        else:
                            # update partial tool call
                            partial_tool_call["arguments"] += item["arguments"]

                yield event
            elif event["event_type"] == "stop":
                if partial_tool_call:
                    # finish partial tool call
                    yield {
                        "role": "assistant",
                        "event_type": "delta",
                        "content_items": [
                            {
                                "type": "tool_call",
                                "name": partial_tool_call["name"],
                                "arguments": json.loads(partial_tool_call["arguments"] or "{}"),
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
                        "content_items": [],
                        "usage_metadata": partial_usage["usage_metadata"],
                        "finish_reason": partial_usage["finish_reason"],
                    }
                    partial_usage = {}
