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


_DEFAULT_BASE_URL = "https://api.minimax.io/v1"
_MINIMAX_M2_7_MODELS = frozenset({"minimax-m2.7", "minimax-m2.7-highspeed"})
_WIRE_ITEM_FIDELITY_KEY = "wire_item"


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an SDK model or a raw capture dictionary."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_json_value(value: Any) -> Any:
    """Convert SDK models and nested containers to JSON-style values."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json", exclude_unset=True)
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return value


def _require_json_object(value: Any, context: str) -> dict[str, Any]:
    """Normalize and validate a JSON object used for wire-fidelity replay."""
    normalized = _normalize_json_value(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{context} must be a JSON object.")
    try:
        json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain only JSON-style values.") from exc
    return normalized


def _wire_item_from_fidelity(fidelity: dict[str, Any], context: str) -> dict[str, Any] | None:
    """Extract a complete provider output item from a fidelity payload."""
    wire_item = fidelity.get(_WIRE_ITEM_FIDELITY_KEY)
    if wire_item is None:
        return None
    return _require_json_object(wire_item, f"{context} wire item")


def _wire_content_text(wire_item: dict[str, Any], content_type: str) -> str | None:
    """Concatenate text from matching content parts in a provider output item."""
    content = wire_item.get("content")
    if not isinstance(content, list):
        return None

    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != content_type:
            continue
        text = part.get("text")
        if not isinstance(text, str):
            return None
        text_parts.append(text)
    return "".join(text_parts) if text_parts else None


def _json_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON-style values without conflating booleans and numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def _usage_from_response(response: Any) -> UsageMetadata | None:
    usage = _field(response, "usage")
    if usage is None:
        return None

    input_details = _field(usage, "input_tokens_details", {})
    output_details = _field(usage, "output_tokens_details", {})
    cached_tokens = _field(input_details, "cached_tokens", 0) or 0
    reasoning_tokens = _field(output_details, "reasoning_tokens", 0) or 0
    input_tokens = _field(usage, "input_tokens", 0) or 0
    output_tokens = _field(usage, "output_tokens", 0) or 0
    return {
        "cached_tokens": cached_tokens,
        "prompt_tokens": input_tokens - cached_tokens,
        "thoughts_tokens": reasoning_tokens,
        "response_tokens": output_tokens - reasoning_tokens,
    }


def _format_response_error(event_type: str, error: Any, response_id: Any = None) -> str:
    normalized_error = _normalize_json_value(error)
    details: list[str] = []
    if isinstance(normalized_error, dict):
        for field_name in ("code", "message", "param"):
            if field_name in normalized_error and normalized_error[field_name] is not None:
                details.append(f"{field_name}={normalized_error[field_name]!r}")
    elif normalized_error is not None:
        details.append(f"details={normalized_error!r}")
    if not details:
        details.append("no error details were provided")

    response_context = f" for response {response_id!r}" if response_id is not None else ""
    return f"MiniMax Responses API {event_type}{response_context}: {', '.join(details)}"


class MiniMaxM3Client(LLMClient):
    """MiniMax M-series client using MiniMax's Responses API."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        """Initialize a MiniMax Responses client with a Subscription Key or API key."""
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key or os.getenv("MINIMAX_API_KEY"),
            base_url=base_url or os.getenv("MINIMAX_BASE_URL") or _DEFAULT_BASE_URL,
        )
        self._history: list[UniMessage] = []

    def _convert_thinking_level_to_effort(self, thinking_level: ThinkingLevel) -> str:
        """Map AgentHub thinking levels to the MiniMax reasoning effort vocabulary."""
        mapping = {
            ThinkingLevel.NONE: "low" if self._model.lower() in _MINIMAX_M2_7_MODELS else "none",
            ThinkingLevel.LOW: "low",
            ThinkingLevel.MEDIUM: "medium",
            ThinkingLevel.HIGH: "high",
            ThinkingLevel.XHIGH: "high",
        }
        return mapping[thinking_level]

    def _convert_tool_choice(self, tool_choice: ToolChoice) -> str:
        """Validate MiniMax's supported automatic tool-selection modes."""
        if tool_choice in ("auto", "none"):
            return tool_choice
        raise UnsupportedParameterError(
            self.__class__.__name__,
            "tool_choice",
            "MiniMax Responses API does not support required or named tool selection.",
        )

    def transform_uni_config_to_model_config(self, config: UniConfig) -> dict[str, Any]:
        """Transform universal configuration to MiniMax's Responses API payload."""
        minimax_config: dict[str, Any] = {"model": self._model, "store": False}

        if config.get("system_prompt") is not None:
            minimax_config["instructions"] = config["system_prompt"]
        if config.get("max_tokens") is not None:
            minimax_config["max_output_tokens"] = config["max_tokens"]
        if config.get("temperature") is not None:
            temperature = config["temperature"]
            if not 0 <= temperature <= 1:
                raise UnsupportedParameterError(
                    self.__class__.__name__,
                    "temperature",
                    "MiniMax Responses API does not support temperatures outside the range 0 to 1.",
                )
            minimax_config["temperature"] = temperature
        if config.get("thinking_level") is not None:
            minimax_config["reasoning"] = {"effort": self._convert_thinking_level_to_effort(config["thinking_level"])}
        if config.get("tools") is not None:
            minimax_config["tools"] = [{"type": "function", **tool} for tool in config["tools"]]
        if config.get("tool_choice") is not None:
            minimax_config["tool_choice"] = self._convert_tool_choice(config["tool_choice"])
        if config.get("prompt_caching") == PromptCaching.DISABLE:
            raise UnsupportedParameterError(
                self.__class__.__name__,
                "prompt_caching",
                "MiniMax Responses API does not support disabling its automatic prompt cache.",
            )
        if config.get("prompt_caching") == PromptCaching.ENHANCE:
            raise UnsupportedParameterError(
                self.__class__.__name__,
                "prompt_caching",
                "MiniMax Responses API does not support enhancing its automatic prompt cache.",
            )

        return minimax_config

    def transform_uni_message_to_model_input(self, messages: list[UniMessage]) -> list[dict[str, Any]]:
        """Transform universal messages to MiniMax Responses input items."""
        input_list: list[dict[str, Any]] = []

        for message in messages:
            content_items: list[dict[str, Any]] = []
            for item in message["content_items"]:
                if item["type"] == "text":
                    fidelity = item.get("fidelity") or {}
                    wire_item = (
                        _wire_item_from_fidelity(fidelity, "MiniMax message fidelity")
                        if isinstance(fidelity, dict)
                        else None
                    )
                    if wire_item is None:
                        content_items.append(
                            {
                                "type": "input_text" if message["role"] == "user" else "output_text",
                                "text": item["text"],
                            }
                        )
                    else:
                        if message["role"] != "assistant" or wire_item.get("type") != "message":
                            raise ValueError("MiniMax message wire fidelity requires an assistant message item.")
                        if content_items:
                            input_list.append({"role": message["role"], "content": content_items})
                            content_items = []

                        wire_text = _wire_content_text(wire_item, "output_text")
                        if wire_text == item["text"] and wire_item.get("role") == message["role"]:
                            input_list.append(wire_item)
                        else:
                            input_list.append(
                                {
                                    **wire_item,
                                    "type": "message",
                                    "role": message["role"],
                                    "content": [{"type": "output_text", "text": item["text"]}],
                                }
                            )
                    continue
                if item["type"] == "image_url":
                    content_items.append({"type": "input_image", "image_url": item["image_url"]})
                    continue

                if content_items:
                    input_list.append({"role": message["role"], "content": content_items})
                    content_items = []

                if item["type"] == "thinking":
                    fidelity = _require_json_object(item.get("fidelity") or {}, "MiniMax reasoning fidelity")
                    wire_item = _wire_item_from_fidelity(fidelity, "MiniMax reasoning fidelity")
                    if wire_item is None:
                        wire_item = {
                            key: value
                            for key, value in fidelity.items()
                            if key not in {_WIRE_ITEM_FIDELITY_KEY, "phase"}
                        }
                    if (
                        not isinstance(wire_item.get("id"), str)
                        or not isinstance(wire_item.get("summary"), list)
                        or not isinstance(wire_item.get("content"), list)
                    ):
                        raise ValueError("MiniMax reasoning replay requires valid id, summary, and content fidelity.")

                    wire_text = _wire_content_text(wire_item, "reasoning_text")
                    reasoning_content = (
                        wire_item["content"]
                        if wire_text == item["thinking"]
                        else [{"type": "reasoning_text", "text": item["thinking"]}]
                    )
                    input_list.append({**wire_item, "type": "reasoning", "content": reasoning_content})
                elif item["type"] == "tool_call":
                    fidelity = _require_json_object(item.get("fidelity") or {}, "MiniMax function-call fidelity")
                    wire_item = _wire_item_from_fidelity(fidelity, "MiniMax function-call fidelity")
                    if wire_item is None:
                        wire_item = {
                            key: value
                            for key, value in fidelity.items()
                            if key not in {_WIRE_ITEM_FIDELITY_KEY, "phase"}
                        }

                    arguments = json.dumps(item["arguments"], ensure_ascii=False, separators=(",", ":"))
                    raw_arguments = wire_item.get("arguments")
                    if isinstance(raw_arguments, str):
                        parsed_arguments = parse_tool_call_arguments(
                            raw_arguments,
                            self.__class__.__name__,
                            item["name"],
                            item["tool_call_id"],
                        )
                        if _json_values_equal(parsed_arguments, item["arguments"]):
                            arguments = raw_arguments

                    input_list.append(
                        {
                            **wire_item,
                            "type": "function_call",
                            "call_id": item["tool_call_id"],
                            "name": item["name"],
                            "arguments": arguments,
                        }
                    )
                elif item["type"] == "tool_result":
                    output: str | list[dict[str, Any]] = item["text"]
                    if item.get("images"):
                        output = [{"type": "input_text", "text": item["text"]}]
                        output.extend({"type": "input_image", "image_url": image_url} for image_url in item["images"])
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": item["tool_call_id"],
                            "output": output,
                        }
                    )
                else:
                    raise ValueError(f"Unknown item: {item}")

            if content_items:
                input_list.append({"role": message["role"], "content": content_items})

        return input_list

    def transform_model_output_to_uni_event(self, model_output: Any) -> UniEvent:
        """Transform a MiniMax streaming event to AgentHub's universal event format."""
        event_type: EventType = "unused"
        content_items: list[PartialContentItem] = []
        usage_metadata: UsageMetadata | None = None
        finish_reason: FinishReason | None = None
        minimax_event_type = _field(model_output, "type")

        if minimax_event_type == "response.output_text.delta":
            event_type = "delta"
            content_items.append(
                {
                    "type": "text",
                    "text": _field(model_output, "delta", ""),
                    "fidelity": {"phase": _normalize_json_value(_field(model_output, "item_id"))},
                }
            )
        elif minimax_event_type == "response.reasoning_text.delta":
            event_type = "delta"
            content_items.append({"type": "thinking", "thinking": _field(model_output, "delta", "")})
        elif minimax_event_type == "response.output_item.added":
            item = _field(model_output, "item")
            if _field(item, "type") == "function_call":
                event_type = "start"
                content_items.append(
                    {
                        "type": "partial_tool_call",
                        "name": _field(item, "name", ""),
                        "arguments": "",
                        "tool_call_id": _field(item, "call_id", ""),
                        "fidelity": {
                            "item_id": _normalize_json_value(_field(item, "id")),
                            "output_index": _normalize_json_value(_field(model_output, "output_index")),
                        },
                    }
                )
        elif minimax_event_type == "response.function_call_arguments.delta":
            event_type = "delta"
            content_items.append(
                {
                    "type": "partial_tool_call",
                    "name": "",
                    "arguments": _field(model_output, "delta", ""),
                    "tool_call_id": "",
                    "fidelity": {
                        "item_id": _normalize_json_value(_field(model_output, "item_id")),
                        "output_index": _normalize_json_value(_field(model_output, "output_index")),
                    },
                }
            )
        elif minimax_event_type == "response.output_item.done":
            item = _normalize_json_value(_field(model_output, "item"))
            if not isinstance(item, dict):
                raise ValueError(f"MiniMax output item must be a JSON object, got: {item!r}")

            if item.get("type") == "reasoning":
                event_type = "delta"
                content_items.append(
                    {
                        "type": "thinking",
                        "thinking": "",
                        "fidelity": {_WIRE_ITEM_FIDELITY_KEY: item},
                    }
                )
            elif item.get("type") == "function_call":
                event_type = "delta"
                tool_name = item.get("name") or ""
                tool_call_id = item.get("call_id") or ""
                content_items.append(
                    {
                        "type": "tool_call",
                        "name": tool_name,
                        "arguments": parse_tool_call_arguments(
                            item.get("arguments"), self.__class__.__name__, tool_name, tool_call_id
                        ),
                        "tool_call_id": tool_call_id,
                        "fidelity": {_WIRE_ITEM_FIDELITY_KEY: item},
                    }
                )
            elif item.get("type") == "message":
                event_type = "delta"
                phase = item.get("id") or f"output-{_field(model_output, 'output_index', 0)}"
                content_items.append(
                    {
                        "type": "text",
                        "text": "",
                        "fidelity": {"phase": phase, _WIRE_ITEM_FIDELITY_KEY: item},
                    }
                )
        elif minimax_event_type in {"response.completed", "response.incomplete"}:
            event_type = "stop"
            response = _field(model_output, "response")
            usage_metadata = _usage_from_response(response)
            if minimax_event_type == "response.completed":
                output = _field(response, "output", []) or []
                finish_reason = (
                    "tool_call" if any(_field(item, "type") == "function_call" for item in output) else "stop"
                )
            else:
                incomplete_reason = _field(_field(response, "incomplete_details"), "reason")
                finish_reason = {"max_output_tokens": "length", "content_filter": "stop"}.get(
                    incomplete_reason, "unknown"
                )
        elif minimax_event_type == "response.failed":
            response = _field(model_output, "response")
            raise RuntimeError(
                _format_response_error(minimax_event_type, _field(response, "error"), _field(response, "id"))
            )
        elif minimax_event_type in {"error", "response.error"}:
            response = _field(model_output, "response")
            error = _field(model_output, "error")
            if error is None and response is not None:
                error = _field(response, "error")
            raise RuntimeError(
                _format_response_error(
                    minimax_event_type,
                    error if error is not None else model_output,
                    _field(response, "id") if response is not None else None,
                )
            )
        elif minimax_event_type not in {
            "response.created",
            "response.in_progress",
            "response.output_text.done",
            "response.reasoning_text.done",
            "response.function_call_arguments.done",
            "response.content_part.added",
            "response.content_part.done",
        }:
            raise ValueError(f"Unknown output: {model_output}")

        return {
            "role": "assistant",
            "event_type": event_type,
            "content_items": content_items,
            "usage_metadata": usage_metadata,
            "finish_reason": finish_reason,
        }

    async def _streaming_response_internal(
        self, messages: list[UniMessage], config: UniConfig
    ) -> AsyncIterator[UniEvent]:
        """Stream MiniMax Responses events."""
        minimax_config = self.transform_uni_config_to_model_config(config)
        input_list = self.transform_uni_message_to_model_input(messages)
        stream = await self._client.responses.create(**minimax_config, input=input_list, stream=True)

        async for model_event in stream:
            event = self.transform_model_output_to_uni_event(model_event)
            if event["event_type"] != "unused":
                yield event
