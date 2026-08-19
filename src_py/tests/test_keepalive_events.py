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

from agenthub import AutoLLMClient


@dataclass
class ResponsesStreamCase:
    expected_client: str
    model: str
    client_type: str


# Every client that parses the OpenAI Responses SSE shape.
RESPONSES_STREAM_CASES = [
    ResponsesStreamCase(expected_client="GPT5_6Client", model="gpt-5.6", client_type="gpt-5.6"),
    ResponsesStreamCase(expected_client="OpenaiResponsesClient", model="gpt-5.6", client_type="openai-responses"),
    ResponsesStreamCase(expected_client="MiniMaxM3Client", model="minimax-m3", client_type="minimax-m3"),
]


def _create_auto_client(case: ResponsesStreamCase) -> AutoLLMClient:
    return AutoLLMClient(model=case.model, api_key="test-key", client_type=case.client_type)


async def _stream_from_events(events: list[object]) -> AsyncIterator[object]:
    for event in events:
        yield event


class _FakeResponses:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    async def create(self, **_kwargs: object) -> AsyncIterator[object]:
        return _stream_from_events(self._events)


def _install_fake_responses_stream(client: AutoLLMClient, events: list[object]) -> None:
    client._client._client = SimpleNamespace(responses=_FakeResponses(events))  # noqa: SLF001


def _keepalive_event(sequence_number: int) -> object:
    # Heartbeats come from gateways in front of Responses-compatible servers (one-api-style
    # proxies), never from the official API, so the event shape is synthesized from the
    # report in https://github.com/Prism-Shadow/penguin-harness/issues/286.
    return SimpleNamespace(type="keepalive", sequence_number=sequence_number)


def _text_delta_event(text: str) -> object:
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def _completed_event() -> object:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            status="completed",
            usage=SimpleNamespace(
                input_tokens=2,
                output_tokens=3,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens_details=SimpleNamespace(reasoning_tokens=1),
            ),
        ),
    )


MESSAGES = [{"role": "user", "content_items": [{"type": "text", "text": "Create a memo."}]}]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", RESPONSES_STREAM_CASES, ids=[case.client_type for case in RESPONSES_STREAM_CASES])
async def test_responses_clients_skip_keepalive_heartbeats(case: ResponsesStreamCase):
    client = _create_auto_client(case)
    assert type(client._client).__name__ == case.expected_client  # noqa: SLF001
    _install_fake_responses_stream(
        client,
        [
            _keepalive_event(1),
            _text_delta_event("Here is"),
            _keepalive_event(2),
            _text_delta_event(" the memo."),
            _keepalive_event(3),
            _completed_event(),
        ],
    )

    events = [event async for event in client.streaming_response(MESSAGES, {})]
    texts = [item["text"] for event in events for item in event["content_items"] if item["type"] == "text"]
    assert texts == ["Here is", " the memo."]
    assert events[-1]["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", RESPONSES_STREAM_CASES, ids=[case.client_type for case in RESPONSES_STREAM_CASES])
async def test_responses_clients_still_reject_unknown_events(case: ResponsesStreamCase):
    client = _create_auto_client(case)
    _install_fake_responses_stream(
        client,
        [SimpleNamespace(type="response.mystery_event"), _completed_event()],
    )

    with pytest.raises(ValueError, match="Unknown output"):
        async for _event in client.streaming_response(MESSAGES, {}):
            pass
