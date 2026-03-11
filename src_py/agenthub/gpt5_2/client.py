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
from openai.types.responses import ResponseInputParam, ResponseStreamEvent

from ..base_client import LLMClient
from ..types import (
    EventType,
    FinishReason,
    PartialContentItem,
    ThinkingLevel,
    ToolChoice,
    UniConfig,
    UniEvent,
    UniMessage,
    UsageMetadata,
)


class GPT5_4Client(LLMClient):
    """GPT-5.4-specific LLM client implementation."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        """Initialize GPT-5.4 client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._history: list[UniMessage] = []

    def _convert_thinking_level_to_effort(self, thinking_level: ThinkingLevel) -> str:
        """Convert ThinkingLevel enum to OpenAI's reasoning effort."""
        mapping = {
            ThinkingLevel.NONE: "none",
            ThinkingLevel.LOW: "low",
            ThinkingLevel.MEDIUM: "medium",
            ThinkingLevel.HIGH: "high",
        }
        return mapping.get(thinking_level)

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> str | dict[str, Any]:
        """Convert ToolChoice to OpenAI's tool_choice format with allowed tools support."""
        if isinstance(tool_choice, list):
            return {"mode": "required", "tools": [{"type": "function", "name": name} for name in tool_choice]}
        elif tool_choice == "none":
            return "none"
        elif tool_choice == "auto":
            return "auto"
        elif tool_choice == "required":
            return "required"

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """
        Transform universal configuration to OpenAI Responses API configuration.

        Args:
            config: Universal configuration dict

        Returns:
            OpenAI Responses API configuration dictionary
        """
        # Set store to False to avoid validation error
        # https://community.openai.com/t/one-potential-cause-of-item-rs-xx-of-type-reasoning-was-provided-without-its-required-following-item-error-stateless-using-agents-sdk/1370540
        openai_config = {"model": self._model, "store": False, "include": ["reasoning.encrypted_content"]}

        if config.get("system_prompt") is not None:
            openai_config["instructions"] = config["system_prompt"]

        if config.get("max_tokens") is not None:
            openai_config["max_output_tokens"] = config["max_tokens"]

        if config.get("temperature") is not None and config["temperature"] != 1.0:
            raise ValueError("GPT-5.4 does not support setting temperature.")

        if config.get("thinking_level") is not None:
            openai_config["reasoning"] = {"effort": self._convert_thinking_level_to_effort(config["thinking_level"])}
            if config.get("thinking_summary"):
                openai_config["reasoning"]["summary"] = "concise"

        if config.get("tools") is not None:
            openai_config["tools"] = [{"type": "function", **tool} for tool in config["tools"]]

        if config.get("tool_choice") is not None:
            openai_config["tool_choice"] = self._convert_tool_choice(config["tool_choice"])

        return openai_config

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> ResponseInputParam:
        """
        Transform universal message format to OpenAI Responses API input format.

        Args:
            messages: List of universal message dictionaries

        Returns:
            List of input items for OpenAI Responses API
        """
        input_list: list[ResponseInputParam] = []

        for msg in messages:
            current_content: list = []
            current_phase: str | None = None

            for item in msg["content_items"]:
                if item["type"] == "text":
                    item_phase = item.get("phase") if msg["role"] == "assistant" else None
                    converted = (
                        {"type": "input_text", "text": item["text"]}
                        if msg["role"] == "user"
                        else {"type": "output_text", "text": item["text"]}
                    )
                    if not current_content or item_phase == current_phase:
                        current_content.append(converted)
                        current_phase = item_phase
                    else:
                        # Phase changed — flush the current group
                        entry: dict = {"role": msg["role"], "content": current_content}
                        if current_phase is not None:
                            entry["phase"] = current_phase
                        input_list.append(entry)
                        current_content = [converted]
                        current_phase = item_phase
                elif item["type"] == "image_url":
                    current_content.append({"type": "input_image", "image_url": item["image_url"]})
                elif item["type"] == "thinking":
                    # Flush pending content before inserting a reasoning item
                    if current_content:
                        entry = {"role": msg["role"], "content": current_content}
                        if current_phase is not None:
                            entry["phase"] = current_phase
                        input_list.append(entry)
                        current_content = []
                        current_phase = None
                    signature = json.loads(item["signature"])
                    input_list.append(
                        {
                            "type": "reasoning",
                            "id": signature["id"],
                            "summary": [{"type": "summary_text", "text": item["thinking"]}]
                            if item["thinking"]
                            else [],
                            "encrypted_content": signature["encrypted_content"],
                        }
                    )
                elif item["type"] == "tool_call":
                    # Flush pending content before inserting a function call
                    if current_content:
                        entry = {"role": msg["role"], "content": current_content}
                        if current_phase is not None:
                            entry["phase"] = current_phase
                        input_list.append(entry)
                        current_content = []
                        current_phase = None
                    input_list.append(
                        {
                            "type": "function_call",
                            "call_id": item["tool_call_id"],
                            "name": item["name"],
                            "arguments": json.dumps(item["arguments"], ensure_ascii=False),
                        }
                    )
                elif item["type"] == "tool_result":
                    if "tool_call_id" not in item:
                        raise ValueError("tool_call_id is required for tool result.")

                    # NOTE: tool results are input items
                    tool_result = [{"type": "input_text", "text": item["text"]}]
                    if "images" in item:
                        for image_url in item["images"]:
                            tool_result.append({"type": "input_image", "image_url": image_url})

                    input_list.append(
                        {"type": "function_call_output", "call_id": item["tool_call_id"], "output": tool_result}
                    )
                else:
                    raise ValueError(f"Unknown item: {item}")

            # Flush any remaining content
            if current_content:
                entry = {"role": msg["role"], "content": current_content}
                if current_phase is not None:
                    entry["phase"] = current_phase
                input_list.append(entry)

        return input_list

    def transform_model_output_to_uni_event(self, model_output: ResponseStreamEvent) -> UniEvent:
        """
        Transform OpenAI Responses API streaming event to universal event format.

        Args:
            model_output: OpenAI Responses API streaming event

        Returns:
            Universal event dictionary
        """
        event_type: EventType | None = None
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        openai_event_type = model_output.type
        if openai_event_type == "response.output_text.delta":
            event_type = "delta"
            content_items.append({"type": "text", "text": model_output.delta})

        elif openai_event_type == "response.reasoning_summary_text.delta":
            event_type = "delta"
            content_items.append({"type": "thinking", "thinking": model_output.delta})

        elif openai_event_type == "response.output_item.added":
            if model_output.item.type == "function_call":
                event_type = "start"
                content_items.append(
                    {
                        "type": "partial_tool_call",
                        "name": model_output.item.name,
                        "arguments": "",
                        "tool_call_id": model_output.item.call_id,
                    }
                )
            elif model_output.item.type == "reasoning":
                event_type = "delta"
                signature = {
                    "id": model_output.item.id,
                    "encrypted_content": model_output.item.encrypted_content,
                }
                content_items.append({"type": "thinking", "thinking": "", "signature": json.dumps(signature)})
            else:
                event_type = "unused"

        elif openai_event_type == "response.output_item.done":
            # not sure about the signature of openai, need to check
            if model_output.item.type == "reasoning":
                event_type = "delta"
                signature = {
                    "id": model_output.item.id,
                    "encrypted_content": model_output.item.encrypted_content,
                }
                content_items.append({"type": "thinking", "thinking": "", "signature": json.dumps(signature)})
            else:
                event_type = "unused"

        elif openai_event_type == "response.function_call_arguments.delta":
            event_type = "delta"
            content_items.append(
                {"type": "partial_tool_call", "name": "", "arguments": model_output.delta, "tool_call_id": ""}
            )

        elif openai_event_type == "response.function_call_arguments.done":
            event_type = "stop"

        elif openai_event_type == "response.completed":
            event_type = "stop"
            finish_reason_mapping = {
                "completed": "stop",
                "incomplete": "length",
            }
            finish_reason = finish_reason_mapping.get(model_output.response.status, "unknown")

            input_tokens = model_output.response.usage.input_tokens
            output_tokens = model_output.response.usage.output_tokens

            cached_tokens = model_output.response.usage.input_tokens_details.cached_tokens
            reasoning_tokens = model_output.response.usage.output_tokens_details.reasoning_tokens

            usage_metadata = {
                "cached_tokens": cached_tokens,
                "prompt_tokens": input_tokens - cached_tokens,
                "thoughts_tokens": reasoning_tokens,
                "response_tokens": output_tokens - reasoning_tokens,
            }

        elif openai_event_type in [
            "response.created",
            "response.in_progress",
            "response.output_text.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.reasoning_summary_text.done",
            "response.content_part.added",
            "response.content_part.done",
        ]:
            event_type = "unused"

        else:
            raise ValueError(f"Unknown output: {model_output}")

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
        """Stream generate using OpenAI Responses API with unified conversion methods."""
        # Use unified config conversion
        openai_config = self.transform_uni_config_to_model_config(config)

        # Use unified message conversion
        input_list = self.transform_uni_message_to_model_input(messages)

        # Stream generate
        partial_tool_call = {}
        current_phase: str | None = None
        stream = await self._client.responses.create(**openai_config, input=input_list, stream=True)

        async for raw_event in stream:
            # Track phase from output_item.added events for output_text items
            if raw_event.type == "response.output_item.added" and raw_event.item.type == "output_text":
                current_phase = getattr(raw_event.item, "phase", None)

            event = self.transform_model_output_to_uni_event(raw_event)

            # Inject current phase into text delta content items
            if event["event_type"] == "delta" and current_phase is not None:
                for item in event["content_items"]:
                    if item["type"] == "text":
                        item["phase"] = current_phase

            if event["event_type"] == "start":
                for item in event["content_items"]:
                    if item["type"] == "partial_tool_call":
                        # initialize partial_tool_call
                        partial_tool_call = {
                            "name": item["name"],
                            "arguments": "",
                            "tool_call_id": item["tool_call_id"],
                        }
                        yield event
            elif event["event_type"] == "delta":
                for item in event["content_items"]:
                    if item["type"] == "partial_tool_call":
                        # update partial_tool_call
                        partial_tool_call["arguments"] += item["arguments"]

                yield event
            elif event["event_type"] == "stop":
                if "name" in partial_tool_call and "arguments" in partial_tool_call:
                    # finish partial_tool_call
                    yield {
                        "role": "assistant",
                        "event_type": "delta",
                        "content_items": [
                            {
                                "type": "tool_call",
                                "name": partial_tool_call["name"],
                                "arguments": json.loads(partial_tool_call["arguments"]),
                                "tool_call_id": partial_tool_call["tool_call_id"],
                            }
                        ],
                        "usage_metadata": None,
                        "finish_reason": None,
                    }
                    partial_tool_call = {}

                if event["finish_reason"] or event["usage_metadata"]:
                    yield event
