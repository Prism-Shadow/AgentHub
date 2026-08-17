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

from agenthub import AgentHubError, AutoLLMClient, EmptyResponseError, ToolCallArgumentParseError


@dataclass
class ReasoningStreamCase:
    expected_client: str
    model: str
    client_type: str


REASONING_STREAM_CASES = [
    ReasoningStreamCase(
        expected_client="OpenaiChatClient",
        model="gpt-5.5",
        client_type="openai",
    ),
    ReasoningStreamCase(
        expected_client="GLM5_3Client",
        model="glm-5.1",
        client_type="glm-5.1",
    ),
    ReasoningStreamCase(
        expected_client="KimiK2_6Client",
        model="kimi-k2.6",
        client_type="kimi-k2.6",
    ),
    ReasoningStreamCase(
        expected_client="DeepSeekV4Client",
        model="deepseek-v4",
        client_type="deepseek-v4",
    ),
]


def _create_auto_client(case: ReasoningStreamCase) -> AutoLLMClient:
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


def _delta_chunk(text: str | None = None, reasoning_content: str | None = None) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None, reasoning_content=reasoning_content),
                finish_reason=None,
            )
        ],
        usage=None,
    )


def _stop_chunk(finish_reason: str = "stop") -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None), finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=1,
            prompt_tokens_details=None,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
            prompt_cache_hit_tokens=0,
            prompt_cache_miss_tokens=1,
        ),
    )


MESSAGES = [{"role": "user", "content_items": [{"type": "text", "text": "Create a memo."}]}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    REASONING_STREAM_CASES,
    ids=[case.client_type for case in REASONING_STREAM_CASES],
)
async def test_reasoning_clients_reject_thinking_only_response(case: ReasoningStreamCase):
    client = _create_auto_client(case)
    _install_fake_openai_compatible_stream(
        client,
        [
            _delta_chunk(reasoning_content="Let me think about the memo."),
            _stop_chunk(finish_reason="stop"),
        ],
    )

    with pytest.raises(EmptyResponseError) as exc_info:
        async for _event in client.streaming_response(MESSAGES, {}):
            pass

    empty_error = exc_info.value
    assert empty_error.client == case.expected_client
    assert empty_error.finish_reason == "stop"
    assert "no content other than thinking" in str(empty_error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    REASONING_STREAM_CASES,
    ids=[case.client_type for case in REASONING_STREAM_CASES],
)
async def test_reasoning_clients_reject_response_without_any_content(case: ReasoningStreamCase):
    client = _create_auto_client(case)
    _install_fake_openai_compatible_stream(client, [_stop_chunk(finish_reason="length")])

    with pytest.raises(EmptyResponseError) as exc_info:
        async for _event in client.streaming_response(MESSAGES, {}):
            pass

    assert exc_info.value.finish_reason == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    REASONING_STREAM_CASES,
    ids=[case.client_type for case in REASONING_STREAM_CASES],
)
async def test_reasoning_clients_accept_response_with_text_content(case: ReasoningStreamCase):
    client = _create_auto_client(case)
    _install_fake_openai_compatible_stream(
        client,
        [
            _delta_chunk(reasoning_content="Let me think about the memo."),
            _delta_chunk(text="Here is the memo."),
            _stop_chunk(finish_reason="stop"),
        ],
    )

    events = [event async for event in client.streaming_response(MESSAGES, {})]
    texts = [item["text"] for event in events for item in event["content_items"] if item["type"] == "text"]
    assert texts == ["Here is the memo."]


def test_agenthub_error_hierarchy():
    assert issubclass(AgentHubError, ValueError)
    assert issubclass(EmptyResponseError, AgentHubError)
    assert issubclass(ToolCallArgumentParseError, AgentHubError)
