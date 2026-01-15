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
from typing import Any, Dict, List, Literal, Optional, Union


class ThinkingLevel(str, Enum):
    """Thinking level for model reasoning."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Tool choice can be a literal string or a list of tool names
ToolChoice = Union[Literal["none", "auto", "required"], List[str]]


class ContentPart:
    """Represents a part of message content."""

    def __init__(self, type: str, value: Any):
        self.type = type
        self.value = value


class Message:
    """Represents a message in the conversation."""

    def __init__(
        self,
        role: str,
        content: Union[str, List[Dict[str, Any]]],
        tool_call_id: Optional[str] = None,
    ):
        self.role = role
        self.content = content
        self.tool_call_id = tool_call_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result


# Type alias for message dict format
MessageDict = Dict[str, Any]
