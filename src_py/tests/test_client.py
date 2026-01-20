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
from contextlib import nullcontext

import pytest

from agenthub import AutoLLMClient, ThinkingLevel


IMAGE = "https://cdn.britannica.com/80/120980-050-D1DA5C61/Poet-narcissus.jpg"

AVAILABLE_MODELS = []
AVAILABLE_VISION_MODELS = []

if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    AVAILABLE_MODELS.append("gemini-3-flash-preview")
    AVAILABLE_VISION_MODELS.append("gemini-3-flash-preview")

if os.getenv("ANTHROPIC_API_KEY"):
    AVAILABLE_MODELS.append("claude-sonnet-4-5-20250929")
    AVAILABLE_VISION_MODELS.append("claude-sonnet-4-5-20250929")

if os.getenv("OPENAI_API_KEY"):
    AVAILABLE_MODELS.append("gpt-5.2")
    AVAILABLE_VISION_MODELS.append("gpt-5.2")

if os.getenv("GLM_API_KEY"):
    AVAILABLE_MODELS.append("glm-4.7")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_streaming_response_basic(model):
    """Test basic stateless stream generation."""
    client = AutoLLMClient(model=model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is 2+3?"}]}]
    config = {}

    text = ""
    async for event in client.streaming_response(messages=messages, config=config):
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert "5" in text  # 2 + 3 = 5


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_streaming_response_with_all_parameters(model):
    """Test stream generation with all optional parameters."""
    client = AutoLLMClient(model=model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is 2+3?"}]}]
    config = {"max_tokens": 8192, "temperature": 0.7, "thinking_summary": True, "thinking_level": ThinkingLevel.LOW}

    if model == "gpt-5.2":
        context = pytest.raises(ValueError, match="not support")
    else:
        context = nullcontext()

    with context:
        text = ""
        async for event in client.streaming_response(messages=messages, config=config):
            for item in event["content_items"]:
                if item["type"] == "text":
                    text += item["text"]

        assert "5" in text  # 2 + 3 = 5


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_streaming_response_stateful(model):
    """Test stateful stream generation."""
    client = AutoLLMClient(model=model)
    config = {}

    message1 = {"role": "user", "content_items": [{"type": "text", "text": "My name is Alice"}]}
    async for _ in client.streaming_response_stateful(message=message1, config=config):
        pass

    assert len(client.get_history()) == 2  # user message + assistant response

    message2 = {"role": "user", "content_items": [{"type": "text", "text": "What is my name?"}]}
    text = ""
    async for event in client.streaming_response_stateful(message=message2, config=config):
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert "alice" in text.lower()
    assert len(client.get_history()) == 4  # 2 previous + 2 new


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_clear_history(model):
    """Test clearing conversation history."""
    client = AutoLLMClient(model=model)
    message = {"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}
    config = {}

    async for _ in client.streaming_response_stateful(message=message, config=config):
        pass

    assert len(client.get_history()) > 0

    client.clear_history()
    assert len(client.get_history()) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_concat_uni_events_to_uni_message(model):
    """Test concatenation of events into a single message."""
    client = AutoLLMClient(model=model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Say hello"}]}]
    config = {}

    events = []
    text = ""
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    # Concatenate events to get the full message
    message = client.concat_uni_events_to_uni_message(events)
    assert message["role"] == "assistant"
    for item in message["content_items"]:
        if item["type"] == "text":
            assert item["text"] == text


@pytest.mark.asyncio
async def test_unknown_model():
    """Test that unknown models raise ValueError."""
    with pytest.raises(ValueError, match="not support"):
        AutoLLMClient(model="unknown-model")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_tool_use(model):
    """Test tool use capability."""
    client = AutoLLMClient(model=model)

    # Define a simple weather tool
    weather_tool = {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco",
                },
            },
            "required": ["location"],
        },
    }

    config = {"tools": [weather_tool]}
    tool_call_id = None
    message1 = {"role": "user", "content_items": [{"type": "text", "text": "What is the weather in San Francisco?"}]}
    async for event in client.streaming_response_stateful(message=message1, config=config):
        for item in event["content_items"]:
            if item["type"] == "tool_call":
                assert item["name"] == weather_tool["name"]
                tool_call_id = item.get("tool_call_id")

    # Check if a function call was made
    assert tool_call_id is not None

    message2 = {
        "role": "user",
        "content_items": [
            {"type": "tool_result", "result": "It's 20 degrees in San Francisco.", "tool_call_id": tool_call_id}
        ],
    }
    text = ""
    async for event in client.streaming_response_stateful(message=message2, config=config):
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert "20" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_image_understanding(model):
    """Test image understanding with a URL."""
    if model not in AVAILABLE_VISION_MODELS:
        pytest.skip(f"Image understanding is not supported by {model}.")

    client = AutoLLMClient(model=model)
    config = {}
    messages = [
        {
            "role": "user",
            "content_items": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": IMAGE},
            ],
        }
    ]
    text = ""
    async for event in client.streaming_response(messages=messages, config=config):
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert ("flower" in text.lower()) or ("narcissus" in text.lower())


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_system_prompt(model):
    """Test system prompt capability."""
    client = AutoLLMClient(model=model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}]
    config = {"system_prompt": "You are a kitten that must end with the word 'meow'."}

    text = ""
    async for event in client.streaming_response(messages=messages, config=config):
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert "meow" in text.lower()
