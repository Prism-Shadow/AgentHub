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

import pytest

from agent_adapter.client import LLMClient
from agent_adapter.types import ThinkingLevel


@pytest.mark.asyncio
async def test_stream_generate_basic():
    """Test basic stateless stream generation."""
    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]

    chunks = []
    async for chunk in client.stream_generate(messages=messages, model="gemini-3-flash-preview"):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert chunks[0]["type"] == "text"


@pytest.mark.asyncio
async def test_stream_generate_with_all_parameters():
    """Test stream generation with all optional parameters."""
    client = LLMClient()
    messages = [{"role": "user", "content": "What's the weather?"}]

    chunks = []
    async for chunk in client.stream_generate(
        messages=messages,
        model="gemini-3-flash-preview",
        max_tokens=100,
        temperature=0.7,
        tools=[{"name": "get_weather", "description": "Get weather data"}],
        thinking_level=ThinkingLevel.HIGH,
        tool_choice="auto",
    ):
        chunks.append(chunk)

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_stream_generate_stateful():
    """Test stateful stream generation."""
    client = LLMClient()

    # First message
    chunks1 = []
    async for chunk in client.stream_generate_stateful(
        message={"role": "user", "content": "Hello"}, model="gemini-3-flash-preview"
    ):
        chunks1.append(chunk)

    assert len(chunks1) > 0
    assert len(client.get_history()) == 1

    # Second message
    chunks2 = []
    async for chunk in client.stream_generate_stateful(
        message={"role": "user", "content": "How are you?"}, model="gemini-3-flash-preview"
    ):
        chunks2.append(chunk)

    assert len(chunks2) > 0
    assert len(client.get_history()) == 2


@pytest.mark.asyncio
async def test_clear_history():
    """Test clearing message history."""
    client = LLMClient()

    # Add a message
    async for _ in client.stream_generate_stateful(
        message={"role": "user", "content": "Hello"}, model="gemini-3-flash-preview"
    ):
        pass

    assert len(client.get_history()) == 1

    # Clear history
    client.clear_history()
    assert len(client.get_history()) == 0


@pytest.mark.asyncio
async def test_message_with_content_list():
    """Test message with content as a list of objects."""
    client = LLMClient()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "value": "What's in this image?"},
                {"type": "image_url", "value": "https://example.com/image.jpg"},
            ],
        }
    ]

    chunks = []
    async for chunk in client.stream_generate(messages=messages, model="gemini-3-flash-preview"):
        chunks.append(chunk)

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_message_with_tool_call_id():
    """Test message with tool_call_id."""
    client = LLMClient()
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "Let me check that for you."},
        {"role": "tool", "content": "Temperature: 72°F", "tool_call_id": "call_123"},
    ]

    chunks = []
    async for chunk in client.stream_generate(messages=messages, model="gemini-3-flash-preview"):
        chunks.append(chunk)

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_tool_choice_as_list():
    """Test tool_choice as a list of tool names."""
    client = LLMClient()
    messages = [{"role": "user", "content": "Get the weather"}]

    chunks = []
    async for chunk in client.stream_generate(
        messages=messages, model="gemini-3-flash-preview", tool_choice=["get_weather", "get_forecast"]
    ):
        chunks.append(chunk)

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_thinking_levels():
    """Test different thinking levels."""
    client = LLMClient()
    messages = [{"role": "user", "content": "Solve this problem"}]

    for level in [ThinkingLevel.NONE, ThinkingLevel.LOW, ThinkingLevel.MEDIUM, ThinkingLevel.HIGH]:
        chunks = []
        async for chunk in client.stream_generate(
            messages=messages, model="gemini-3-flash-preview", thinking_level=level
        ):
            chunks.append(chunk)
        assert len(chunks) > 0
