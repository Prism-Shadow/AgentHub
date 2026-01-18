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

from enum import StrEnum
from typing import Any, Literal, NotRequired, TypedDict


class ThinkingLevel(StrEnum):
    """Thinking level for model reasoning."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Tool choice can be a literal string or a list of tool names
ToolChoice = Literal["auto", "required", "none"] | list[str]
Role = Literal["user", "assistant"]
Event = Literal["start", "delta", "stop", "unused"]
FinishReason = Literal["stop", "length", "unknown"]


class TextContentItem(TypedDict):
    type: Literal["text"]
    text: str
    signature: NotRequired[str | bytes]
    tool_call_id: NotRequired[str]


class ImageContentItem(TypedDict):
    type: Literal["image_url"]
    image_url: str


class ThinkingContentItem(TypedDict):
    type: Literal["thinking"]
    thinking: str
    signature: NotRequired[str | bytes]


class ToolCallContentItem(TypedDict):
    type: Literal["tool_call"]
    name: str
    argument: dict[str, Any]
    tool_call_id: str
    signature: NotRequired[str | bytes]


class PartialToolCallContentItem(TypedDict):
    type: Literal["partial_tool_call"]
    name: str
    argument: str
    tool_call_id: str
    signature: NotRequired[str | bytes]


class ToolResultContentItem(TypedDict):
    type: Literal["tool_result"]
    result: str
    tool_call_id: str


ContentItem = TextContentItem | ImageContentItem | ThinkingContentItem | ToolCallContentItem | ToolResultContentItem

PartialContentItem = ContentItem | PartialToolCallContentItem


class UsageMetadata(TypedDict):
    """Usage metadata for model response."""

    prompt_tokens: int | None
    thoughts_tokens: int | None
    response_tokens: int | None


class UniMessage(TypedDict):
    """Universal message format for LLM communication."""

    role: Role
    content_items: list[ContentItem]
    usage_metadata: NotRequired[UsageMetadata]
    finish_reason: NotRequired[FinishReason]


class UniEvent(TypedDict):
    """Universal event format for streaming responses."""

    role: Role
    content_items: list[ContentItem]
    usage_metadata: NotRequired[UsageMetadata]
    finish_reason: NotRequired[FinishReason]


class PartialUniEvent(TypedDict):
    """Partial universal event format for streaming responses."""

    role: Role
    event: Event
    content_items: list[PartialContentItem]
    usage_metadata: NotRequired[UsageMetadata]
    finish_reason: NotRequired[FinishReason]


class ToolSchema(TypedDict):
    """Available tool schema."""

    name: str
    description: str
    parameters: NotRequired[str]


class UniConfig(TypedDict):
    """Universal configuration format for LLM requests."""

    max_tokens: NotRequired[int]
    temperature: NotRequired[float]
    tools: NotRequired[list[ToolSchema]]
    thinking_summary: NotRequired[bool]
    thinking_level: NotRequired[ThinkingLevel]
    tool_choice: NotRequired[ToolChoice]
    system_prompt: NotRequired[str]
    trace_id: NotRequired[str]
