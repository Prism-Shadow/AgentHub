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
from ..utils import fix_openrouter_usage_metadata


class GLM5_3Client(LLMClient):
    """Unified client for the GLM series, named for the newest generation it serves (5.3).

    The wire format is shared across GLM-5.1 through 5.3; only the thinking parameter
    contract differs per generation, handled model-by-model below.
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        """Initialize GLM client with model and API key."""
        self._model = model
        api_key = api_key or os.getenv("ZAI_API_KEY")
        base_url = base_url or os.getenv("ZAI_BASE_URL") or "https://api.z.ai/api/paas/v4/"
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._history: list[UniMessage] = []

    def _convert_thinking_level_to_config(self, thinking_level: ThinkingLevel) -> dict[str, str | bool]:
        """Convert ThinkingLevel enum to GLM's thinking configuration.

        GLM-5.3 uses forced thinking and errors on {"type": "disabled"}, so NONE stays
        enabled there and degrades through the lightest reasoning effort instead
        (llmsdk_docs/glm5_3/docs/thinking.md).
        """
        # Provider-hosted ids keep their own casing (e.g. SiliconFlow's zai-org/GLM-5.2),
        # so generation detection is case-insensitive.
        if thinking_level == ThinkingLevel.NONE and "glm-5.3" not in self._model.lower():
            return {"type": "disabled"}
        return {"type": "enabled", "clear_thinking": False}

    def _convert_thinking_level_to_reasoning_effort(self, thinking_level: ThinkingLevel) -> str | None:
        """Convert ThinkingLevel enum to the reasoning_effort the model accepts.

        GLM-5.3 accepts only low/high/max and errors on anything else, so the client
        clamps to the closest value; NONE rides on low because 5.3 cannot disable
        thinking. GLM-5.2 accepts the full vocabulary and maps it server-side
        (low/medium to high, xhigh to max); NONE disables thinking there instead.
        Models before 5.2 take no reasoning_effort parameter at all.
        """
        model = self._model.lower()  # provider-hosted ids keep their own casing
        if "glm-5.3" in model:
            mapping = {
                ThinkingLevel.NONE: "low",
                ThinkingLevel.LOW: "low",
                ThinkingLevel.MEDIUM: "high",
                ThinkingLevel.HIGH: "high",
                ThinkingLevel.XHIGH: "max",
                ThinkingLevel.MAX: "max",
            }
            return mapping.get(thinking_level)
        if "glm-5.2" in model:
            mapping = {
                ThinkingLevel.NONE: None,
                ThinkingLevel.LOW: "low",
                ThinkingLevel.MEDIUM: "medium",
                ThinkingLevel.HIGH: "high",
                ThinkingLevel.XHIGH: "xhigh",
                ThinkingLevel.MAX: "max",
            }
            return mapping.get(thinking_level)
        return None

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> str:
        """Convert ToolChoice to OpenAI's tool_choice format."""
        if tool_choice == "auto":
            return "auto"
        else:
            raise UnsupportedParameterError(
                self.__class__.__name__, "tool_choice", "GLM only supports 'auto' for tool_choice."
            )

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """
        Transform universal configuration to GLM-specific configuration.

        Args:
            config: Universal configuration dict

        Returns:
            GLM configuration dictionary
        """
        glm_config = {"model": self._model, "stream": True, "extra_body": {"tool_stream": True}}

        if config.get("max_tokens") is not None:
            glm_config["max_tokens"] = config["max_tokens"]

        if config.get("temperature") is not None:
            glm_config["temperature"] = config["temperature"]

        # NOTE: glm-5 always provides thinking summary
        if config.get("thinking_level") is not None:
            thinking_config = self._convert_thinking_level_to_config(config["thinking_level"])
            # thinking is only effective when using the official API endpoint
            glm_config.setdefault("extra_body", {})["thinking"] = thinking_config
            reasoning_effort = self._convert_thinking_level_to_reasoning_effort(config["thinking_level"])
            if reasoning_effort is not None:
                glm_config["reasoning_effort"] = reasoning_effort

        if config.get("tools") is not None:
            glm_config["tools"] = [{"type": "function", "function": tool} for tool in config["tools"]]

        if config.get("tool_choice") is not None:
            glm_config["tool_choice"] = self._convert_tool_choice(config["tool_choice"])

        if config.get("fast_mode"):
            raise UnsupportedParameterError(self.__class__.__name__, "fast_mode", "GLM does not support fast mode.")

        if config.get("prompt_caching") is not None and config["prompt_caching"] != PromptCaching.ENABLE:
            raise UnsupportedParameterError(
                self.__class__.__name__, "prompt_caching", "prompt_caching must be ENABLE for GLM."
            )

        return glm_config

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[ChatCompletionMessageParam]:
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
            thinking_fields: set[str | None] = set()
            for item in msg["content_items"]:
                if item["type"] == "text":
                    content_parts.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    raise ValueError("GLM-5 does not support image inputs.")
                elif item["type"] == "thinking":
                    thinking += item["thinking"]
                    thinking_fields.add((item.get("fidelity") or {}).get("reasoning_field"))
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

                    if "images" in item and item["images"]:
                        raise ValueError("GLM-5 does not support images in tool results.")

                    # Tool results are sent as separate messages
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": item["tool_call_id"],
                            "content": item["text"],
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
                # send thinking back through the exact field the upstream produced (recorded
                # in the item fidelity); servers may reject the spelling they did not emit
                if thinking_fields == {"reasoning_content"}:
                    message["reasoning_content"] = thinking
                elif thinking_fields == {"reasoning"}:
                    message["reasoning"] = thinking
                else:
                    message["reasoning_content"] = thinking  # vLLM & siliconflow compatibility
                    message["reasoning"] = thinking  # openrouter compatibility

            # message may be empty for tool results
            if len(message.keys()) > 1:
                openai_messages.append(message)

        return openai_messages

    def transform_model_output_to_uni_event(self, model_output: ChatCompletionChunk) -> UniEvent:
        """
        Transform GLM model output to universal event format.

        Args:
            model_output: OpenAI streaming chunk

        Returns:
            Universal event dictionary
        """
        event_type: EventType | None = None
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None

        # gateways inject content-free heartbeat chunks on long generations, whose choices
        # the SDK leaves as None rather than an empty list
        if model_output.choices:
            choice = model_output.choices[0]
            delta = choice.delta

            if delta.content:
                event_type = "delta"
                content_items.append({"type": "text", "text": delta.content})

            # the thinking field name differs by server: vLLM & siliconflow use reasoning_content
            # while openrouter uses reasoning; record the wire field that carried each delta
            # so a replay can reproduce exactly the field the upstream produced
            reasoning_content = getattr(delta, "reasoning_content", None)
            reasoning = getattr(delta, "reasoning", None)
            if reasoning_content and reasoning:
                event_type = "delta"
                # ambiguous origin: record no fidelity so a replay sends both fields back
                content_items.append({"type": "thinking", "thinking": reasoning_content})
            elif reasoning_content:
                event_type = "delta"
                content_items.append(
                    {
                        "type": "thinking",
                        "thinking": reasoning_content,
                        "fidelity": {"reasoning_field": "reasoning_content"},
                    }
                )
            elif reasoning:
                event_type = "delta"
                content_items.append(
                    {"type": "thinking", "thinking": reasoning, "fidelity": {"reasoning_field": "reasoning"}}
                )

            if delta.tool_calls:
                event_type = "delta"
                for tool_call in delta.tool_calls:
                    content_items.append(
                        {
                            "type": "partial_tool_call",
                            "name": tool_call.function.name or "",
                            "arguments": tool_call.function.arguments or "",
                            "tool_call_id": tool_call.id or "",
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
                        "content_items": [],
                        "usage_metadata": partial_usage["usage_metadata"],
                        "finish_reason": partial_usage["finish_reason"],
                    }
                    partial_usage = {}
