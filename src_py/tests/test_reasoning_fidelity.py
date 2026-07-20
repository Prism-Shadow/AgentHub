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

import inspect
from collections.abc import AsyncIterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from agenthub import AutoLLMClient


@dataclass
class ReasoningReplayCase:
    model: str
    client_type: str


REASONING_REPLAY_CASES = [
    ReasoningReplayCase(model="gpt-5.5", client_type="openai"),
    ReasoningReplayCase(model="glm-5.1", client_type="glm-5.1"),
    ReasoningReplayCase(model="kimi-k2.6", client_type="kimi-k2.6"),
]


def _create_auto_client(case: ReasoningReplayCase) -> AutoLLMClient:
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


def _delta_chunk(text: str | None = None, **reasoning_fields: str) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None, **reasoning_fields),
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


def _user_message() -> dict[str, Any]:
    return {"role": "user", "content_items": [{"type": "text", "text": "Create a memo."}]}


async def _transform_history(client: AutoLLMClient, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_input = client.transform_uni_message_to_model_input(history)
    if inspect.isawaitable(model_input):
        model_input = await model_input

    return model_input


async def _run_turn_and_replay(client: AutoLLMClient) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one fake streamed turn, then rebuild the request payload from the stored history."""
    async for _event in client.streaming_response_stateful(_user_message(), {}):
        pass

    history = client.get_history()
    model_input = await _transform_history(client, history)
    return history[-1], model_input[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", REASONING_REPLAY_CASES, ids=[case.client_type for case in REASONING_REPLAY_CASES])
async def test_replay_preserves_reasoning_content_field(case: ReasoningReplayCase):
    client = _create_auto_client(case)
    _install_fake_openai_compatible_stream(
        client,
        [
            _delta_chunk(reasoning_content="Let me think"),
            _delta_chunk(reasoning_content=" about the memo."),
            _delta_chunk(text="Here is the memo."),
            _stop_chunk(),
        ],
    )

    history_message, replayed_message = await _run_turn_and_replay(client)
    thinking_items = [item for item in history_message["content_items"] if item["type"] == "thinking"]
    assert thinking_items == [
        {
            "type": "thinking",
            "thinking": "Let me think about the memo.",
            "fidelity": {"reasoning_field": "reasoning_content"},
        }
    ]
    assert replayed_message["reasoning_content"] == "Let me think about the memo."
    assert "reasoning" not in replayed_message


@pytest.mark.asyncio
@pytest.mark.parametrize("case", REASONING_REPLAY_CASES, ids=[case.client_type for case in REASONING_REPLAY_CASES])
async def test_replay_preserves_reasoning_field(case: ReasoningReplayCase):
    client = _create_auto_client(case)
    _install_fake_openai_compatible_stream(
        client,
        [
            _delta_chunk(reasoning="Let me think"),
            _delta_chunk(reasoning=" about the memo."),
            _delta_chunk(text="Here is the memo."),
            _stop_chunk(),
        ],
    )

    history_message, replayed_message = await _run_turn_and_replay(client)
    thinking_items = [item for item in history_message["content_items"] if item["type"] == "thinking"]
    assert thinking_items == [
        {
            "type": "thinking",
            "thinking": "Let me think about the memo.",
            "fidelity": {"reasoning_field": "reasoning"},
        }
    ]
    assert replayed_message["reasoning"] == "Let me think about the memo."
    assert "reasoning_content" not in replayed_message


@pytest.mark.asyncio
@pytest.mark.parametrize("case", REASONING_REPLAY_CASES, ids=[case.client_type for case in REASONING_REPLAY_CASES])
async def test_replay_keeps_both_fields_when_origin_is_ambiguous(case: ReasoningReplayCase):
    client = _create_auto_client(case)
    _install_fake_openai_compatible_stream(
        client,
        [
            _delta_chunk(reasoning_content="Let me think.", reasoning="Let me think."),
            _delta_chunk(text="Here is the memo."),
            _stop_chunk(),
        ],
    )

    history_message, replayed_message = await _run_turn_and_replay(client)
    thinking_items = [item for item in history_message["content_items"] if item["type"] == "thinking"]
    assert thinking_items == [{"type": "thinking", "thinking": "Let me think."}]
    assert replayed_message["reasoning_content"] == "Let me think."
    assert replayed_message["reasoning"] == "Let me think."


@pytest.mark.asyncio
@pytest.mark.parametrize("case", REASONING_REPLAY_CASES, ids=[case.client_type for case in REASONING_REPLAY_CASES])
async def test_replay_of_thinking_without_fidelity_sends_both_fields(case: ReasoningReplayCase):
    client = _create_auto_client(case)
    history = [
        _user_message(),
        {
            "role": "assistant",
            "content_items": [
                {"type": "thinking", "thinking": "Let me think."},
                {"type": "text", "text": "Here is the memo."},
            ],
        },
    ]

    model_input = await _transform_history(client, history)
    replayed_message = model_input[-1]
    assert replayed_message["reasoning_content"] == "Let me think."
    assert replayed_message["reasoning"] == "Let me think."


def _text_delta_event(text: str, phase: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"type": "text", "text": text}
    if phase is not None:
        item["fidelity"] = {"phase": phase}

    return {
        "role": "assistant",
        "event_type": "delta",
        "content_items": [item],
        "usage_metadata": None,
        "finish_reason": None,
    }


def test_concat_splits_text_items_only_on_phase_change():
    client = _create_auto_client(REASONING_REPLAY_CASES[0])
    message = client.concat_uni_events_to_uni_message(
        [
            _text_delta_event("", phase="commentary"),
            _text_delta_event("I'll inspect the logs."),
            _text_delta_event("", phase="final_answer"),
            _text_delta_event("Root cause:"),
            _text_delta_event(" cache invalidation race."),
            _text_delta_event("", phase="final_answer"),
            _text_delta_event(" Remediation follows."),
        ]
    )

    assert message["content_items"] == [
        {"type": "text", "text": "I'll inspect the logs.", "fidelity": {"phase": "commentary"}},
        {
            "type": "text",
            "text": "Root cause: cache invalidation race. Remediation follows.",
            "fidelity": {"phase": "final_answer"},
        },
    ]
