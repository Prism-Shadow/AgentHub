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

import os

import pytest

from agent_adapter import AutoLLMClient, GeminiClient


# Skip tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
    reason="No Gemini API key available",
)


@pytest.mark.asyncio
async def test_streaming_response_basic():
    """Test basic stateless stream generation."""
    client = AutoLLMClient(model="gemini-3-flash-preview")
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Say hello"}]}]
    config = {}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0


@pytest.mark.asyncio
async def test_streaming_response_with_all_parameters():
    """Test stream generation with all optional parameters."""
    client = AutoLLMClient(model="gemini-3-flash-preview")
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is 2+2?"}]}]
    config = {"max_tokens": 100, "temperature": 0.7}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0


@pytest.mark.asyncio
async def test_streaming_response_stateful():
    """Test stateful stream generation."""
    client = AutoLLMClient(model="gemini-3-flash-preview")
    config = {}

    # First message
    events1 = []
    async for event in client.streaming_response_stateful(
        message={"role": "user", "content_items": [{"type": "text", "text": "My name is Alice"}]}, config=config
    ):
        events1.append(event)

    assert len(events1) > 0
    assert len(client.get_history()) == 2  # User message + assistant response

    # Second message
    events2 = []
    async for event in client.streaming_response_stateful(
        message={"role": "user", "content_items": [{"type": "text", "text": "What is my name?"}]}, config=config
    ):
        events2.append(event)

    assert len(events2) > 0
    assert len(client.get_history()) == 4  # 2 previous + 2 new


@pytest.mark.asyncio
async def test_clear_history():
    """Test clearing conversation history."""
    client = AutoLLMClient(model="gemini-3-flash-preview")
    config = {}

    async for event in client.streaming_response_stateful(
        message={"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}, config=config
    ):
        pass

    assert len(client.get_history()) > 0

    client.clear_history()
    assert len(client.get_history()) == 0


@pytest.mark.asyncio
async def test_unknown_model():
    """Test that unknown models raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model type"):
        AutoLLMClient(model="unknown-model")


@pytest.mark.asyncio
async def test_gpt_not_implemented():
    """Test that GPT models raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="GPT models not yet implemented"):
        AutoLLMClient(model="gpt-4")


@pytest.mark.asyncio
async def test_claude_not_implemented():
    """Test that Claude models raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Claude models not yet implemented"):
        AutoLLMClient(model="claude-3-opus")
