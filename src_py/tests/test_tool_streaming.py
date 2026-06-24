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

"""Deterministic, offline regression tests for the tool-call streaming protocol.

These tests do not hit any provider API. They replace each client's underlying
SDK stream with a synthetic sequence of provider chunks and assert that the
unified streaming loop emits a complete ``tool_call`` whose ``arguments`` is a
dict, even for a tool call that carries no arguments (the empty-args edge case
that previously crashed the stream via ``json.loads("")``).
"""

from types import SimpleNamespace

import pytest

from agenthub.claude5.client import Claude5Client
from agenthub.gemini3.client import Gemini3Client
from agenthub.gpt5_5.client import GPT5_5Client


def _make_stream(events):
    """Return an awaitable that yields ``events`` as an async iterator (an SDK stream stub)."""

    async def _create(*args, **kwargs):
        async def _gen():
            for event in events:
                yield event

        return _gen()

    return _create


async def _collect(client, config):
    message = {"role": "user", "content_items": [{"type": "text", "text": "What time is it?"}]}
    return [event async for event in client.streaming_response(messages=[message], config=config)]


def _tool_calls(events):
    return [item for event in events for item in event["content_items"] if item["type"] == "tool_call"]


TOOL_CONFIG = {
    "tools": [{"name": "get_time", "description": "Get the current time.", "parameters": {"type": "object"}}]
}


@pytest.mark.asyncio
async def test_claude_empty_tool_arguments_do_not_crash():
    """A Claude tool_use block with no input_json_delta must flush args == {} (not raise)."""
    client = Claude5Client(model="claude-fable-5", api_key="test-key")
    client._client = SimpleNamespace(
        beta=SimpleNamespace(
            messages=SimpleNamespace(
                create=_make_stream(
                    [
                        SimpleNamespace(
                            type="message_start",
                            message=SimpleNamespace(
                                usage=SimpleNamespace(
                                    input_tokens=10,
                                    cache_read_input_tokens=0,
                                    cache_creation_input_tokens=0,
                                )
                            ),
                        ),
                        SimpleNamespace(
                            type="content_block_start",
                            content_block=SimpleNamespace(type="tool_use", name="get_time", id="toolu_1"),
                        ),
                        # NOTE: no input_json_delta is emitted -> accumulated arguments stays ""
                        SimpleNamespace(type="content_block_stop"),
                        SimpleNamespace(
                            type="message_delta",
                            delta=SimpleNamespace(stop_reason="tool_use"),
                            usage=SimpleNamespace(output_tokens=5),
                        ),
                        SimpleNamespace(type="message_stop"),
                    ]
                )
            )
        )
    )

    events = await _collect(client, TOOL_CONFIG)
    tool_calls = _tool_calls(events)
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_time"
    assert tool_calls[0]["tool_call_id"] == "toolu_1"
    assert tool_calls[0]["arguments"] == {}  # empty string args coerced to {} rather than raising


@pytest.mark.asyncio
async def test_gpt5_5_empty_tool_arguments_do_not_crash():
    """A GPT-5.5 function_call with no argument deltas must flush args == {} (not raise)."""
    client = GPT5_5Client(model="gpt-5.5", api_key="test-key")
    client._client = SimpleNamespace(
        responses=SimpleNamespace(
            create=_make_stream(
                [
                    SimpleNamespace(
                        type="response.output_item.added",
                        item=SimpleNamespace(type="function_call", name="get_time", call_id="call_1"),
                    ),
                    # NOTE: no response.function_call_arguments.delta -> accumulated arguments stays ""
                    SimpleNamespace(type="response.function_call_arguments.done"),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(
                            status="completed",
                            usage=SimpleNamespace(
                                input_tokens=10,
                                output_tokens=5,
                                input_tokens_details=SimpleNamespace(cached_tokens=0),
                                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
                            ),
                        ),
                    ),
                ]
            )
        )
    )

    events = await _collect(client, TOOL_CONFIG)
    tool_calls = _tool_calls(events)
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_time"
    assert tool_calls[0]["tool_call_id"] == "call_1"
    assert tool_calls[0]["arguments"] == {}


@pytest.mark.asyncio
async def test_gemini_zero_arg_tool_call_yields_dict_arguments():
    """A Gemini functionCall with args=None must surface arguments == {} (a dict), not None/"null"."""
    from google.genai import types

    client = Gemini3Client(model="gemini-3.5-flash", api_key="test-key")
    function_call = SimpleNamespace(name="get_time", args=None)
    part = SimpleNamespace(
        function_call=function_call, thought=None, thought_signature=None, text=None, inline_data=None
    )
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason=types.FinishReason.STOP)
    chunk = SimpleNamespace(
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            cached_content_token_count=0,
            thoughts_token_count=0,
            candidates_token_count=5,
        ),
    )
    client._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=_make_stream([chunk])))
    )

    events = await _collect(client, TOOL_CONFIG)
    tool_calls = _tool_calls(events)
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_time"
    assert tool_calls[0]["arguments"] == {}
    assert isinstance(tool_calls[0]["arguments"], dict)

    # The synthesized announce fragment must serialize to "{}" rather than the literal "null".
    partials = [item for event in events for item in event["content_items"] if item["type"] == "partial_tool_call"]
    assert partials and all(item["arguments"] in ("", "{}") for item in partials)


@pytest.mark.asyncio
async def test_gemini_content_less_chunk_does_not_crash():
    """A Gemini terminal/safety chunk whose candidate.content is None must not raise mid-stream."""
    from google.genai import types

    client = Gemini3Client(model="gemini-3.5-flash", api_key="test-key")
    chunk = SimpleNamespace(
        candidates=[SimpleNamespace(content=None, finish_reason=types.FinishReason.STOP)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            cached_content_token_count=0,
            thoughts_token_count=0,
            candidates_token_count=5,
        ),
    )
    client._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=_make_stream([chunk])))
    )

    events = await _collect(client, {})
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["usage_metadata"] is not None
