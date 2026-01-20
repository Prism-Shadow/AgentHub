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
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
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
        else:
            raise ValueError(f"Unknown tool_choice value: {tool_choice}")

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """
        Transform universal configuration to OpenAI Responses API configuration.

        Args:
            config: Universal configuration dict

        Returns:
            OpenAI Responses API configuration dictionary
        """
        openai_config = {"model": self._model}

        if config.get("max_tokens") is not None:
            openai_config["max_tokens"] = config["max_tokens"]

        if config.get("temperature") is not None:
            openai_config["temperature"] = config["temperature"]

        if config.get("system_prompt") is not None:
            openai_config["instructions"] = config["system_prompt"]

        if config.get("tools") is not None:
            openai_tools = []
            for tool in config["tools"]:
                openai_tool = {"type": "function", **tool}
                openai_tools.append(openai_tool)
            openai_config["tools"] = openai_tools

        if config.get("tool_choice") is not None:
            openai_config["tool_choice"] = self._convert_tool_choice(config["tool_choice"])

        return openai_config

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[dict[str, Any]]:
        """
        Transform universal message format to OpenAI Responses API input format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of input items for OpenAI Responses API
        """
        input_list = []

        for msg in messages:
            content_items = []

            for item in msg["content_items"]:
                if item["type"] == "text":
                    content_items.append({"type": "input_text", "text": item["text"]})
                elif item["type"] == "image_url":
                    content_items.append({"type": "input_image", "image_url": item["image_url"]})
                elif item["type"] == "tool_call":
                    input_list.append(
                        {
                            "type": "function_call",
                            "call_id": item["tool_call_id"],
                            "name": item["name"],
                            "arguments": json.dumps(item["argument"]),
                        }
                    )
                elif item["type"] == "tool_result":
                    if "tool_call_id" not in item:
                        raise ValueError("tool_call_id is required for tool result.")
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": item["tool_call_id"],
                            "output": item["result"],
                        }
                    )
                else:
                    raise ValueError(f"Unknown item type: {item['type']}")

            if content_items:
                if len(content_items) == 1 and content_items[0]["type"] == "input_text":
                    input_list.append({"role": msg["role"], "content": content_items[0]["text"]})
                else:
                    input_list.append({"role": msg["role"], "content": content_items})

        return input_list

    def transform_model_output_to_uni_event(self, model_output: Any) -> PartialUniEvent:
        """
        Transform OpenAI Responses API streaming event to universal event format.

        Args:
            model_output: OpenAI Responses API streaming event

        Returns:
            Universal event dictionary
        """
        event_type = "delta"
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        if hasattr(model_output, "type"):
            event_obj_type = model_output.type

            if event_obj_type == "response.output_text.delta":
                if hasattr(model_output, "delta"):
                    content_items.append({"type": "text", "text": model_output.delta})

            elif event_obj_type == "response.function_call_arguments.delta":
                if hasattr(model_output, "delta") and hasattr(model_output, "call_id"):
                    content_items.append(
                        {
                            "type": "partial_tool_call",
                            "name": getattr(model_output, "name", ""),
                            "argument": model_output.delta,
                            "tool_call_id": model_output.call_id,
                        }
                    )

            elif event_obj_type == "response.function_call_arguments.done":
                if hasattr(model_output, "name") and hasattr(model_output, "call_id"):
                    content_items.append(
                        {
                            "type": "partial_tool_call",
                            "name": model_output.name,
                            "argument": "",
                            "tool_call_id": model_output.call_id,
                        }
                    )

            elif event_obj_type == "response.done":
                if hasattr(model_output, "response"):
                    response_obj = model_output.response
                    if hasattr(response_obj, "status"):
                        status_mapping = {
                            "completed": "stop",
                            "incomplete": "length",
                            "failed": "stop",
                        }
                        finish_reason = status_mapping.get(response_obj.status, "unknown")

                    if hasattr(response_obj, "usage"):
                        usage = response_obj.usage
                        usage_metadata = {
                            "prompt_tokens": getattr(usage, "input_tokens", None),
                            "thoughts_tokens": None,
                            "response_tokens": getattr(usage, "output_tokens", None),
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
        """Stream generate using OpenAI Responses API with unified conversion methods."""
        openai_config = self.transform_uni_config_to_model_config(config)
        input_list = self.transform_uni_message_to_model_input(messages)

        openai_config["input"] = input_list

        partial_tool_calls = {}
        stream = await self._client.responses.create(**openai_config, stream=True)

        async for event in stream:
            transformed_event = self.transform_model_output_to_uni_event(event)

            if transformed_event["event"] == "delta":
                if transformed_event["content_items"]:
                    for item in transformed_event["content_items"]:
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

            if transformed_event["finish_reason"] or transformed_event["usage_metadata"]:
                if partial_tool_calls:
                    for tool_call in partial_tool_calls.values():
                        try:
                            argument = json.loads(tool_call["argument"]) if tool_call["argument"] else {}
                        except json.JSONDecodeError:
                            argument = {}
                        yield {
                            "role": "assistant",
                            "content_items": [
                                {
                                    "type": "tool_call",
                                    "name": tool_call["name"],
                                    "argument": argument,
                                    "tool_call_id": tool_call["tool_call_id"],
                                }
                            ],
                        }
                    partial_tool_calls = {}

                if transformed_event["usage_metadata"] or transformed_event["finish_reason"]:
                    final_event = {"role": "assistant", "content_items": []}
                    if transformed_event["usage_metadata"]:
                        final_event["usage_metadata"] = transformed_event["usage_metadata"]
                    if transformed_event["finish_reason"]:
                        final_event["finish_reason"] = transformed_event["finish_reason"]
                    yield final_event
