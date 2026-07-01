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

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from agenthub.deepseek_v4 import DeepSeekV4Client
from agenthub.errors import ToolCallArgumentParseError
from agenthub.glm5_1 import GLM5_1Client
from agenthub.kimi_k2_6 import KimiK2_6Client
from agenthub.openai import OpenaiClient


@dataclass
class OpenAICompatibleToolStreamCase:
    name: str
    client_name: str
    create_client: Callable[[], Any]


OPENAI_COMPATIBLE_TOOL_STREAM_CASES = [
    OpenAICompatibleToolStreamCase(
        name="openai",
        client_name="openai",
        create_client=lambda: OpenaiClient("gpt-5.5", api_key="test-key"),
    ),
    OpenAICompatibleToolStreamCase(
        name="glm5_1",
        client_name="glm5_1",
        create_client=lambda: GLM5_1Client("glm-5.1", api_key="test-key"),
    ),
    OpenAICompatibleToolStreamCase(
        name="kimi_k2_6",
        client_name="kimi_k2_6",
        create_client=lambda: KimiK2_6Client("kimi-k2.6", api_key="test-key"),
    ),
    OpenAICompatibleToolStreamCase(
        name="deepseek_v4",
        client_name="deepseek_v4",
        create_client=lambda: DeepSeekV4Client("deepseek-v4", api_key="test-key"),
    ),
]


async def _stream_from_chunks(chunks: list[object]) -> AsyncIterator[object]:
    for chunk in chunks:
        yield chunk


class _FakeOpenAICompatibleCompletions:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = chunks

    async def create(self, **_kwargs: object) -> AsyncIterator[object]:
        return _stream_from_chunks(self._chunks)


class _FakeOpenAICompatibleClient:
    def __init__(self, chunks: list[object]) -> None:
        self.base_url = "https://api.test.invalid/v1"
        self.chat = SimpleNamespace(completions=_FakeOpenAICompatibleCompletions(chunks))


def _tool_delta_chunk(tool_call_id: str, name: str, arguments: str) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=tool_call_id,
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                ),
                finish_reason=None,
            )
        ],
        usage=None,
    )


def _tool_stop_chunk() -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason="tool_calls")],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            prompt_tokens_details=None,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            prompt_cache_hit_tokens=0,
            prompt_cache_miss_tokens=1,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    OPENAI_COMPATIBLE_TOOL_STREAM_CASES,
    ids=[case.name for case in OPENAI_COMPATIBLE_TOOL_STREAM_CASES],
)
async def test_openai_compatible_clients_combine_streamed_tool_call_arguments(
    case: OpenAICompatibleToolStreamCase,
):
    client = case.create_client()
    client._client = _FakeOpenAICompatibleClient(  # noqa: SLF001 - deterministic transport for offline regression
        [
            _tool_delta_chunk("call_ok", "exec_command", '{"cmd":'),
            _tool_delta_chunk("", "", '"echo ok"}'),
            _tool_stop_chunk(),
        ]
    )

    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Create a memo."}]}]
    events = [event async for event in client._streaming_response_internal(messages, {})]
    tool_calls = [item for event in events for item in event["content_items"] if item["type"] == "tool_call"]

    assert tool_calls == [
        {
            "type": "tool_call",
            "name": "exec_command",
            "arguments": {"cmd": "echo ok"},
            "tool_call_id": "call_ok",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    OPENAI_COMPATIBLE_TOOL_STREAM_CASES,
    ids=[case.name for case in OPENAI_COMPATIBLE_TOOL_STREAM_CASES],
)
async def test_openai_compatible_clients_report_malformed_streamed_tool_call_arguments(
    case: OpenAICompatibleToolStreamCase,
):
    client = case.create_client()
    client._client = _FakeOpenAICompatibleClient(  # noqa: SLF001 - deterministic transport for offline regression
        [
            _tool_delta_chunk("call_bad", "exec_command", '{"cmd":"python create_docx.py'),
            _tool_stop_chunk(),
        ]
    )

    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Create a memo."}]}]
    with pytest.raises(ToolCallArgumentParseError) as exc_info:
        async for _event in client._streaming_response_internal(messages, {}):
            pass

    parse_error = exc_info.value
    assert parse_error.client == case.client_name
    assert parse_error.tool_name == "exec_command"
    assert parse_error.tool_call_id == "call_bad"
    assert parse_error.raw_arguments_length > 0
    assert "create_docx.py" in parse_error.raw_arguments_preview
    message = str(parse_error)
    assert "exec_command" in message
    assert "call_bad" in message
    assert "length=" in message
    assert "Unterminated string" in message
