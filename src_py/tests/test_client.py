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

from agent_adapter import LLMClient


# Skip tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
    reason="No Gemini API key available",
)


@pytest.mark.asyncio
async def test_stream_generate_basic():
    """Test basic stateless stream generation."""
    client = LLMClient()
    messages = [{"role": "user", "content": "Say hello"}]

    chunks = []
    async for chunk in client.stream_generate(messages=messages, model="gemini-2.0-flash-exp"):
        chunks.append(chunk)

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_stream_generate_with_all_parameters():
    """Test stream generation with all optional parameters."""
    client = LLMClient()
    messages = [{"role": "user", "content": "What is 2+2?"}]

    chunks = []
    async for chunk in client.stream_generate(
        messages=messages,
        model="gemini-2.0-flash-exp",
        max_tokens=100,
        temperature=0.7,
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
        message={"role": "user", "content": "My name is Alice"}, model="gemini-2.0-flash-exp"
    ):
        chunks1.append(chunk)

    assert len(chunks1) > 0
    # History should contain user message and assistant response
    assert len(client.get_history()) >= 1

    # Second message
    chunks2 = []
    async for chunk in client.stream_generate_stateful(
        message={"role": "user", "content": "What is my name?"}, model="gemini-2.0-flash-exp"
    ):
        chunks2.append(chunk)

    assert len(chunks2) > 0
    # History should now have more messages
    assert len(client.get_history()) >= 2


@pytest.mark.asyncio
async def test_clear_history():
    """Test clearing message history."""
    client = LLMClient()

    # Add a message
    async for _ in client.stream_generate_stateful(
        message={"role": "user", "content": "Hello"}, model="gemini-2.0-flash-exp"
    ):
        pass

    assert len(client.get_history()) >= 1

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
                {"type": "text", "value": "Describe this"},
            ],
        }
    ]

    chunks = []
    async for chunk in client.stream_generate(messages=messages, model="gemini-2.0-flash-exp"):
        chunks.append(chunk)

    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_unknown_model():
    """Test that unknown models raise ValueError."""
    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="Unknown model type"):
        async for _ in client.stream_generate(messages=messages, model="unknown-model-xyz"):
            pass


@pytest.mark.asyncio
async def test_gpt_not_implemented():
    """Test that GPT models raise NotImplementedError."""
    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(NotImplementedError, match="GPT models not yet implemented"):
        async for _ in client.stream_generate(messages=messages, model="gpt-4"):
            pass


@pytest.mark.asyncio
async def test_claude_not_implemented():
    """Test that Claude models raise NotImplementedError."""
    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(NotImplementedError, match="Claude models not yet implemented"):
        async for _ in client.stream_generate(messages=messages, model="claude-3"):
            pass
