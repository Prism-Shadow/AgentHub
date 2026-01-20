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
from openai.types.chat import ChatCompletionChunk

from ..base_client import LLMClient
from ..types import (
    FinishReason,
    PartialContentItem,
    PartialUniEvent,
    ToolChoice,
    UniConfig,
    UniEvent,
    UniMessage,
    UsageMetadata,
)


class GPT5_2Client(LLMClient):
    """GPT-5.2-specific LLM client implementation."""

    def __init__(self, model: str, api_key: str | None = None):
        """Initialize GPT-5.2 client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)
        self._history: list[UniMessage] = []

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> str | dict[str, Any]:
        """Convert ToolChoice to OpenAI's tool_choice format."""
        if isinstance(tool_choice, list):
            if len(tool_choice) > 1:
                raise ValueError("OpenAI supports only one tool choice at a time.")
            return {"type": "function", "function": {"name": tool_choice[0]}}
        elif tool_choice == "none":
            return "none"
        elif tool_choice == "auto":
            return "auto"
        elif tool_choice == "required":
            return "required"

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """
        Transform universal configuration to OpenAI-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            OpenAI configuration dictionary
        """
        openai_config = {"model": self._model}

        if config.get("max_tokens") is not None:
            openai_config["max_tokens"] = config["max_tokens"]

        if config.get("temperature") is not None:
            openai_config["temperature"] = config["temperature"]

        if config.get("tools") is not None:
            openai_tools = []
            for tool in config["tools"]:
                openai_tool = {"type": "function", "function": tool}
                openai_tools.append(openai_tool)
            openai_config["tools"] = openai_tools

        if config.get("tool_choice") is not None:
            openai_config["tool_choice"] = self._convert_tool_choice(config["tool_choice"])

        return openai_config

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
            content_blocks = []
            tool_calls = []

            for item in msg["content_items"]:
                if item["type"] == "text":
                    content_blocks.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    content_blocks.append({"type": "image_url", "image_url": {"url": item["image_url"]}})
                elif item["type"] == "tool_call":
                    tool_calls.append(
                        {
                            "id": item["tool_call_id"],
                            "type": "function",
                            "function": {"name": item["name"], "arguments": json.dumps(item["argument"])},
                        }
                    )
                elif item["type"] == "tool_result":
                    if "tool_call_id" not in item:
                        raise ValueError("tool_call_id is required for tool result.")
                    openai_messages.append(
                        {"role": "tool", "content": item["result"], "tool_call_id": item["tool_call_id"]}
                    )
                else:
                    raise ValueError(f"Unknown item type: {item['type']}")

            if content_blocks or msg["role"] == "user":
                message = {"role": msg["role"], "content": content_blocks if content_blocks else ""}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                openai_messages.append(message)
            elif tool_calls:
                openai_messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        return openai_messages

    def transform_model_output_to_uni_event(self, model_output: ChatCompletionChunk) -> PartialUniEvent:
        """
        Transform OpenAI model output to universal event format.

        Args:
            model_output: OpenAI streaming chunk

        Returns:
            Universal event dictionary
        """
        event_type = "delta"
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        if model_output.choices:
            choice = model_output.choices[0]
            delta = choice.delta

            if delta.content:
                content_items.append({"type": "text", "text": delta.content})

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    if tool_call.function:
                        content_items.append(
                            {
                                "type": "partial_tool_call",
                                "name": tool_call.function.name or "",
                                "argument": tool_call.function.arguments or "",
                                "tool_call_id": tool_call.id or "",
                            }
                        )

            if choice.finish_reason:
                finish_reason_mapping = {
                    "stop": "stop",
                    "length": "length",
                    "tool_calls": "stop",
                    "content_filter": "stop",
                }
                finish_reason = finish_reason_mapping.get(choice.finish_reason, "unknown")

        if model_output.usage:
            usage_metadata = {
                "prompt_tokens": model_output.usage.prompt_tokens,
                "thoughts_tokens": None,
                "response_tokens": model_output.usage.completion_tokens,
            }

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
        """Stream generate using OpenAI SDK with unified conversion methods."""
        openai_config = self.transform_uni_config_to_model_config(config)
        openai_messages = self.transform_uni_message_to_model_input(messages)

        if config.get("system_prompt"):
            openai_messages.insert(0, {"role": "system", "content": config["system_prompt"]})

        partial_tool_calls = {}
        stream = await self._client.chat.completions.create(**openai_config, messages=openai_messages, stream=True)

        async for chunk in stream:
            event = self.transform_model_output_to_uni_event(chunk)

            if event["event"] == "delta":
                if event["content_items"]:
                    for item in event["content_items"]:
                        if item["type"] == "partial_tool_call":
                            tool_call_id = item["tool_call_id"]
                            if tool_call_id not in partial_tool_calls:
                                partial_tool_calls[tool_call_id] = {
                                    "name": item["name"],
                                    "argument": "",
                                    "tool_call_id": tool_call_id,
                                }
                            if item["name"]:
                                partial_tool_calls[tool_call_id]["name"] = item["name"]
                            partial_tool_calls[tool_call_id]["argument"] += item["argument"]
                        else:
                            yield {"role": "assistant", "content_items": [item]}

            if event["finish_reason"] or event["usage_metadata"]:
                if partial_tool_calls:
                    for tool_call in partial_tool_calls.values():
                        yield {
                            "role": "assistant",
                            "content_items": [
                                {
                                    "type": "tool_call",
                                    "name": tool_call["name"],
                                    "argument": json.loads(tool_call["argument"]) if tool_call["argument"] else {},
                                    "tool_call_id": tool_call["tool_call_id"],
                                }
                            ],
                        }
                    partial_tool_calls = {}

                if event["usage_metadata"] or event["finish_reason"]:
                    final_event = {"role": "assistant", "content_items": []}
                    if event["usage_metadata"]:
                        final_event["usage_metadata"] = event["usage_metadata"]
                    if event["finish_reason"]:
                        final_event["finish_reason"] = event["finish_reason"]
                    yield final_event
