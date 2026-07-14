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
from typing import Any


def _preview_tool_call_arguments(raw: str) -> str:
    max_length = 160
    if len(raw) <= max_length:
        return raw

    edge_length = 72
    return f"{raw[:edge_length]}...[truncated]...{raw[-edge_length:]}"


class ToolCallArgumentParseError(ValueError):
    def __init__(self, client: str, tool_name: str, tool_call_id: str, raw_arguments: str, reason: str) -> None:
        self.client = client
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.raw_arguments_length = len(raw_arguments)
        self.raw_arguments_preview = _preview_tool_call_arguments(raw_arguments)
        super().__init__(
            f'Invalid streamed tool call arguments from {client} for tool "{tool_name}" '
            f'(tool_call_id="{tool_call_id}", length={self.raw_arguments_length}, '
            f"preview={self.raw_arguments_preview!r}): {reason}"
        )


class EmptyAssistantResponseError(Exception):
    """Raised when DeepSeek reaches the output token limit while still thinking."""

    def __init__(self) -> None:
        super().__init__("DeepSeek reached the output token limit before producing content or tool calls.")


def parse_tool_call_arguments(
    raw_arguments: str | None,
    client: str,
    tool_name: str,
    tool_call_id: str,
) -> dict[str, Any]:
    raw = raw_arguments or "{}"
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("expected a JSON object")
    except (TypeError, ValueError) as exc:
        raise ToolCallArgumentParseError(client, tool_name, tool_call_id, raw, str(exc)) from exc
