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
from typing import Literal, NotRequired, TypedDict


class ThinkingLevel(StrEnum):
    """Thinking level for model reasoning."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Tool choice can be a literal string or a list of tool names
ToolChoice = Literal["auto", "required", "none"] | list[str]
Role = Literal["user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "unknown"]


class TextContentItem(TypedDict):
    type: Literal["text"]
    text: str
    signature: str
    tool_call_id: str


class ReasoningContentItem(TypedDict):
    type: Literal["reasoning"]
    reasoning: str
    signature: str


class ImageContentItem(TypedDict):
    type: Literal["image_url"]
    image_url: str


class FunctionCallContentItem(TypedDict):
    type: Literal["function_call"]
    name: str
    argument: str
    signature: str
    tool_call_id: str


class UnknownContentItem(TypedDict):
    type: Literal["unknown"]
    data: str


ContentItem = TextContentItem | ReasoningContentItem | ImageContentItem | FunctionCallContentItem | UnknownContentItem


class UsageMetadata(TypedDict):
    prompt_tokens: int | None
    thoughts_tokens: int | None
    response_tokens: int | None


class UniMessage(TypedDict):
    """Universal message format for LLM communication."""

    role: Role
    content_items: list[ContentItem]  # List of content items
    usage_metadata: NotRequired[UsageMetadata]
    finish_reason: NotRequired[FinishReason]


class UniEvent(TypedDict):
    """Universal event format for streaming responses."""

    role: Role
    content_items: list[ContentItem]  # List of content items
    usage_metadata: NotRequired[UsageMetadata]
    finish_reason: NotRequired[FinishReason]


class UniConfig(TypedDict):
    """Universal configuration format for LLM requests."""

    max_tokens: int | None
    temperature: float | None
    tools: list[str] | None
    thinking_summary: bool | None
    thinking_level: ThinkingLevel | None
    tool_choice: ToolChoice | None
    system_prompt: str | None
