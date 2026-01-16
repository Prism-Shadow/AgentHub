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

from enum import Enum
from typing import Any, List, Literal, NotRequired, TypedDict, Union


class ThinkingLevel(str, Enum):
    """Thinking level for model reasoning."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Tool choice can be a literal string or a list of tool names
ToolChoice = Union[Literal["none", "auto", "required"], List[str]]


class ContentItem(TypedDict):
    """A content item within a message."""

    type: str  # "text", "image_url", or "thought_signature"
    value: str


class UniMessage(TypedDict):
    """Universal message format for LLM communication."""

    role: str  # "user", "assistant", "tool", or "system"
    content: Union[str, List[ContentItem]]  # Text or list of content items
    tool_call_id: NotRequired[str]  # Optional tool call ID for tool responses


class UniConfig(TypedDict, total=False):
    """Universal configuration format for LLM requests."""

    max_tokens: int
    temperature: float
    tools: List[Any]
    thinking_level: ThinkingLevel
    tool_choice: ToolChoice


class UniEvent(TypedDict):
    """Universal event format for streaming responses."""

    type: str  # "text", "thought_signature", "tool_call", etc.
    content: str  # The actual content of the event
    metadata: NotRequired[dict]  # Optional metadata


# Legacy aliases for backward compatibility
MessageDict = UniMessage
