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


class PromptCaching(StrEnum):
    """Prompt cache configuration for Claude models."""

    ENABLE = "enable"
    DISABLE = "disable"
    ENHANCE = "enhance"


# Tool choice can be a literal string or a list of tool names
ToolChoice = Literal["auto", "required", "none"] | list[str]
Role = Literal["user", "assistant"]
EventType = Literal["start", "delta", "stop", "unused"]
FinishReason = Literal["stop", "length", "tool_call", "unknown"]
AspectRatio = Literal["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"]
ImageSize = Literal["1K", "2K"]


class TextContentItem(TypedDict):
    type: Literal["text"]
    text: str
    phase: NotRequired[str | None]
    signature: NotRequired[str | bytes]


class ImageContentItem(TypedDict):
    type: Literal["image_url"]
    image_url: str


class InlineDataContentItem(TypedDict):
    type: Literal["inline_data"]
    data: bytes
    mime_type: str
    signature: NotRequired[str | bytes]


class ThinkingContentItem(TypedDict):
    type: Literal["thinking"]
    thinking: str
    signature: NotRequired[str | bytes]


class InlineThinkingContentItem(TypedDict):
    type: Literal["inline_thinking"]
    data: bytes
    mime_type: str
    signature: NotRequired[str | bytes]


class ToolCallContentItem(TypedDict):
    type: Literal["tool_call"]
    name: str
    arguments: dict[str, Any]
    tool_call_id: str
    signature: NotRequired[str | bytes]


class PartialToolCallContentItem(TypedDict):
    type: Literal["partial_tool_call"]
    name: str
    arguments: str
    tool_call_id: str
    signature: NotRequired[str | bytes]


class ToolResultContentItem(TypedDict):
    type: Literal["tool_result"]
    text: str
    images: NotRequired[list[str]]
    tool_call_id: str


ContentItem = (
    TextContentItem
    | ImageContentItem
    | InlineDataContentItem
    | ThinkingContentItem
    | InlineThinkingContentItem
    | ToolCallContentItem
    | ToolResultContentItem
)

PartialContentItem = ContentItem | PartialToolCallContentItem


class UsageMetadata(TypedDict):
    """Usage metadata for model response."""

    cached_tokens: int | None
    prompt_tokens: int | None
    thoughts_tokens: int | None
    response_tokens: int | None


class UniMessage(TypedDict):
    """Universal message format for LLM communication."""

    role: Role
    content_items: list[ContentItem]
    usage_metadata: NotRequired[UsageMetadata | None]
    finish_reason: NotRequired[FinishReason | None]
    created_at: NotRequired[int]


class UniEvent(TypedDict):
    """Universal event format for streaming responses."""

    role: Role
    event_type: EventType
    content_items: list[PartialContentItem]
    usage_metadata: UsageMetadata | None
    finish_reason: FinishReason | None
    created_at: int


class ToolSchema(TypedDict):
    """Available tool schema."""

    name: str
    description: str
    parameters: NotRequired[dict[str, Any]]


class ImageConfig(TypedDict):
    """Image generation configuration for models that support image output."""

    aspect_ratio: NotRequired[AspectRatio]
    image_size: NotRequired[ImageSize]


class UniConfig(TypedDict):
    """Universal configuration format for LLM requests."""

    max_tokens: NotRequired[int]
    temperature: NotRequired[float]
    tools: NotRequired[list[ToolSchema]]
    thinking_summary: NotRequired[bool]
    thinking_level: NotRequired[ThinkingLevel]
    tool_choice: NotRequired[ToolChoice]
    system_prompt: NotRequired[str]
    prompt_caching: NotRequired[PromptCaching]
    image_config: NotRequired[ImageConfig]
    trace_id: NotRequired[str]
