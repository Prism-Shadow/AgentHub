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
from ..types import (
    EventType,
    FinishReason,
    PartialContentItem,
    PromptCaching,
    ToolChoice,
    UniConfig,
    UniEvent,
    UniMessage,
    UsageMetadata,
)


class Qwen3Client(LLMClient):
    """Qwen3-specific LLM client implementation using OpenAI-compatible API."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        """Initialize Qwen3 client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("QWEN3_API_KEY")
        base_url = base_url or os.getenv("QWEN3_BASE_URL", "http://127.0.0.1:8000/v1/")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._history: list[UniMessage] = []

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> str:
        """Convert ToolChoice to OpenAI's tool_choice format."""
        if tool_choice == "auto":
            return "auto"
        else:
            raise ValueError("Qwen3 only supports 'auto' for tool_choice.")

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """
        Transform universal configuration to Qwen3-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            Qwen3 configuration dictionary
        """
        qwen3_config = {"model": self._model, "stream": True}

        if config.get("max_tokens") is not None:
            qwen3_config["max_tokens"] = config["max_tokens"]

        if config.get("temperature") is not None:
            qwen3_config["temperature"] = config["temperature"]

        if config.get("tools") is not None:
            qwen3_config["tools"] = [{"type": "function", "function": tool} for tool in config["tools"]]

        if config.get("tool_choice") is not None:
            qwen3_config["tool_choice"] = self._convert_tool_choice(config["tool_choice"])

        if config.get("prompt_caching") is not None and config["prompt_caching"] != PromptCaching.ENABLE:
            raise ValueError("prompt_caching must be ENABLE for Qwen3.")

        return qwen3_config

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[ChatCompletionMessageParam]:
        """
        Transform universal message format to Qwen3-specific message format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of Qwen3 message dictionaries
        """
        qwen3_messages = []

        for msg in messages:
            content_parts = []  # may be empty for tool results
            tool_calls = []  # may be empty for no tool calls
            thinking = ""
            for item in msg["content_items"]:
                if item["type"] == "text":
                    content_parts.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    raise ValueError("Qwen3 does not support image_url.")
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

                    result = item["result"]
                    if isinstance(result, str):
                        qwen3_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": item["tool_call_id"],
                                "content": result,
                            }
                        )
                    else:
                        content_parts = []
                        for result_item in result:
                            if result_item["type"] == "text":
                                content_parts.append({"type": "text", "text": result_item["text"]})
                            elif result_item["type"] == "image_url":
                                raise ValueError("Qwen3 does not support image_url in tool results.")
                        qwen3_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": item["tool_call_id"],
                                "content": content_parts,
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
                qwen3_messages.append(message)

        return qwen3_messages

    def transform_model_output_to_uni_event(self, model_output: ChatCompletionChunk) -> UniEvent:
        """
        Transform Qwen3 model output to universal event format.

        Args:
            model_output: OpenAI streaming chunk

        Returns:
            Universal event dictionary
        """
        event_type: EventType | None = None
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        choice = model_output.choices[0]
        delta = choice.delta

        if delta.content:
            # manually check for content since tool parser of vLLM is not stable
            if delta.content == "<tool_call>":
                event_type = "start"
            elif delta.content == "</tool_call>":
                event_type = "stop"
            else:
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
            for tool_call in delta.tool_calls:
                event_type = "delta"
                content_items.append(
                    {
                        "type": "partial_tool_call",
                        "name": tool_call.function.name or "",
                        "arguments": tool_call.function.arguments or "",
                        "tool_call_id": tool_call.function.name or "",
                    }
                )

        if choice.finish_reason:
            event_type = "stop"
            finish_reason_mapping = {
                "stop": "stop",
                "length": "length",
                "tool_calls": "stop",
                "content_filter": "stop",
            }
            finish_reason = finish_reason_mapping.get(choice.finish_reason, "unknown")

        if model_output.usage:
            event_type = event_type or "delta"  # deal with separate usage data
            if model_output.usage.completion_tokens_details:
                reasoning_tokens = model_output.usage.completion_tokens_details.reasoning_tokens
            else:
                reasoning_tokens = None

            if model_output.usage.prompt_tokens_details:
                cached_tokens = model_output.usage.prompt_tokens_details.cached_tokens
            else:
                cached_tokens = None

            usage_metadata = {
                "prompt_tokens": model_output.usage.prompt_tokens,
                "thoughts_tokens": reasoning_tokens,
                "response_tokens": model_output.usage.completion_tokens,
                "cached_tokens": cached_tokens,
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
        """Stream generate using Qwen3 SDK with unified conversion methods."""
        # Use unified config conversion
        qwen3_config = self.transform_uni_config_to_model_config(config)

        # Use unified message conversion
        qwen3_messages = self.transform_uni_message_to_model_input(messages)

        # Extract system prompt if present
        if config.get("system_prompt"):
            qwen3_messages.insert(0, {"role": "system", "content": config["system_prompt"]})

        # Stream generate
        stream = await self._client.chat.completions.create(**qwen3_config, messages=qwen3_messages)

        partial_tool_call = {}
        async for chunk in stream:
            event = self.transform_model_output_to_uni_event(chunk)
            if event["event_type"] == "start":
                # initialize partial_tool_call for <tool_call>
                partial_tool_call = {"data": ""}
            elif event["event_type"] == "delta":
                if "data" in partial_tool_call:
                    # update partial_tool_call for <tool_call>
                    partial_tool_call["data"] += event["content_items"][0]["text"]
                    continue

                for item in event["content_items"]:
                    if item["type"] == "partial_tool_call":
                        if not partial_tool_call:
                            # initialize partial_tool_call for tool call object
                            partial_tool_call = {"name": item["name"], "arguments": item["arguments"]}
                        else:
                            # update partial_tool_call for tool call object
                            partial_tool_call["arguments"] += item["arguments"]

                yield event
            elif event["event_type"] == "stop":
                if "data" in partial_tool_call:
                    # finish partial_tool_call for <tool_call>
                    tool_call = json.loads(partial_tool_call["data"].strip())
                    yield {
                        "role": "assistant",
                        "event_type": "delta",
                        "content_items": [
                            {
                                "type": "partial_tool_call",
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["arguments"], ensure_ascii=False),
                                "tool_call_id": tool_call["name"],
                            }
                        ],
                        "usage_metadata": None,
                        "finish_reason": None,
                    }
                    yield {
                        "role": "assistant",
                        "event_type": "delta",
                        "content_items": [
                            {
                                "type": "tool_call",
                                "name": tool_call["name"],
                                "arguments": tool_call["arguments"],
                                "tool_call_id": tool_call["name"],
                            }
                        ],
                        "usage_metadata": None,
                        "finish_reason": None,
                    }
                    partial_tool_call = {}

                if "name" in partial_tool_call and "arguments" in partial_tool_call:
                    # finish partial_tool_call for tool call object
                    yield {
                        "role": "assistant",
                        "event_type": "delta",
                        "content_items": [
                            {
                                "type": "tool_call",
                                "name": partial_tool_call["name"],
                                "arguments": json.loads(partial_tool_call["arguments"]),
                                "tool_call_id": partial_tool_call["name"],
                            }
                        ],
                        "usage_metadata": None,
                        "finish_reason": None,
                    }
                    partial_tool_call = {}

                if event["finish_reason"] or event["usage_metadata"]:
                    yield event
