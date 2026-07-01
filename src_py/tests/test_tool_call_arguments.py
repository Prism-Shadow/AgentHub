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

from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agenthub import AutoLLMClient, ToolCallArgumentParseError


@dataclass
class OpenAICompatibleToolStreamCase:
    name: str
    client_name: str
    model: str
    client_type: str


OPENAI_COMPATIBLE_TOOL_STREAM_CASES = [
    OpenAICompatibleToolStreamCase(
        name="openai",
        client_name="openai",
        model="gpt-5.5",
        client_type="openai",
    ),
    OpenAICompatibleToolStreamCase(
        name="glm5_1",
        client_name="glm5_1",
        model="glm-5.1",
        client_type="glm-5.1",
    ),
    OpenAICompatibleToolStreamCase(
        name="kimi_k2_6",
        client_name="kimi_k2_6",
        model="kimi-k2.6",
        client_type="kimi-k2.6",
    ),
    OpenAICompatibleToolStreamCase(
        name="deepseek_v4",
        client_name="deepseek_v4",
        model="deepseek-v4",
        client_type="deepseek-v4",
    ),
]


def _create_auto_client(case: OpenAICompatibleToolStreamCase) -> AutoLLMClient:
    return AutoLLMClient(model=case.model, api_key="test-key", client_type=case.client_type)


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


def _install_fake_openai_compatible_stream(client: AutoLLMClient, chunks: list[object]) -> None:
    client._client._client = _FakeOpenAICompatibleClient(chunks)  # noqa: SLF001


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


async def _capture_tool_argument_error(stream: AsyncIterator[object]) -> ToolCallArgumentParseError:
    with pytest.raises(ToolCallArgumentParseError) as exc_info:
        async for _event in stream:
            pass
    return exc_info.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    OPENAI_COMPATIBLE_TOOL_STREAM_CASES,
    ids=[case.name for case in OPENAI_COMPATIBLE_TOOL_STREAM_CASES],
)
async def test_openai_compatible_clients_combine_streamed_tool_call_arguments(
    case: OpenAICompatibleToolStreamCase,
):
    client = _create_auto_client(case)
    _install_fake_openai_compatible_stream(
        client,
        [
            _tool_delta_chunk("call_ok", "exec_command", '{"cmd":'),
            _tool_delta_chunk("", "", '"echo ok"}'),
            _tool_stop_chunk(),
        ],
    )

    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Create a memo."}]}]
    events = [event async for event in client.streaming_response(messages, {})]
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
    client = _create_auto_client(case)
    _install_fake_openai_compatible_stream(
        client,
        [
            _tool_delta_chunk("call_bad", "exec_command", '{"cmd":"python create_docx.py'),
            _tool_stop_chunk(),
        ],
    )

    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Create a memo."}]}]
    parse_error = await _capture_tool_argument_error(client.streaming_response(messages, {}))
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    OPENAI_COMPATIBLE_TOOL_STREAM_CASES,
    ids=[case.name for case in OPENAI_COMPATIBLE_TOOL_STREAM_CASES],
)
async def test_openai_compatible_clients_report_non_object_streamed_tool_call_arguments(
    case: OpenAICompatibleToolStreamCase,
):
    client = _create_auto_client(case)
    _install_fake_openai_compatible_stream(
        client,
        [
            _tool_delta_chunk("call_array", "exec_command", "[]"),
            _tool_stop_chunk(),
        ],
    )

    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Create a memo."}]}]
    parse_error = await _capture_tool_argument_error(client.streaming_response(messages, {}))
    assert parse_error.client == case.client_name
    assert parse_error.tool_name == "exec_command"
    assert parse_error.tool_call_id == "call_array"
    assert parse_error.raw_arguments_length == 2
    assert parse_error.raw_arguments_preview == "[]"
    assert "expected a JSON object" in str(parse_error)
