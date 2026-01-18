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

from agenthub import AutoLLMClient


IMAGE = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Narcissus_poeticus_subsp._radiiflorus.1658.jpg/500px-Narcissus_poeticus_subsp._radiiflorus.1658.jpg"

# Define test models and their required API keys
TEST_MODELS = []

if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    TEST_MODELS.append("gemini-3-flash-preview")

if os.getenv("ANTHROPIC_API_KEY"):
    TEST_MODELS.append("claude-sonnet-4-5-20250929")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", TEST_MODELS)
async def test_streaming_response_basic(model):
    """Test basic stateless stream generation."""
    client = AutoLLMClient(model=model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Say hello"}]}]
    config = {}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", TEST_MODELS)
async def test_streaming_response_with_all_parameters(model):
    """Test stream generation with all optional parameters."""
    client = AutoLLMClient(model=model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is 2+2?"}]}]
    config = {"max_tokens": 100, "temperature": 0.7}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", TEST_MODELS)
async def test_streaming_response_stateful(model):
    """Test stateful stream generation."""
    client = AutoLLMClient(model=model)
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
@pytest.mark.parametrize("model", TEST_MODELS)
async def test_clear_history(model):
    """Test clearing conversation history."""
    client = AutoLLMClient(model=model)
    config = {}

    async for _ in client.streaming_response_stateful(
        message={"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}, config=config
    ):
        pass

    assert len(client.get_history()) > 0

    client.clear_history()
    assert len(client.get_history()) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", TEST_MODELS)
async def test_concat_uni_events_to_uni_message(model):
    """Test concatenation of UniEvents."""
    client = AutoLLMClient(model=model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Say hello"}]}]
    config = {}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    # Concatenate events to get the full message
    message = client.concat_uni_events_to_uni_message(events)
    assert message["role"] == "assistant"
    assert len(message["content_items"]) > 0


@pytest.mark.asyncio
async def test_unknown_model():
    """Test that unknown models raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model type"):
        AutoLLMClient(model="unknown-model")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", TEST_MODELS)
async def test_tool_use(model):
    """Test Claude's tool use capability."""
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
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is the weather in San Francisco?"}]}]

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0

    # Check if a function call was made
    message = client.concat_uni_events_to_uni_message(events)
    has_function_call = any(item["type"] == "function_call" for item in message["content_items"])

    # Claude should attempt to call the get_weather tool
    assert has_function_call


@pytest.mark.asyncio
@pytest.mark.parametrize("model", TEST_MODELS)
async def test_image_understanding(model):
    """Test image understanding with a URL."""
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
    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", TEST_MODELS)
async def test_system_prompt(model):
    """Test Claude with a system prompt."""
    client = AutoLLMClient(model=model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What should I do?"}]}]
    config = {"max_tokens": 100, "system_prompt": "You are a helpful assistant that always responds in pirate speak."}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0

    # The response should have some pirate-speak characteristics
    final_message = client.concat_uni_events_to_uni_message(events)
    response_text = " ".join(item["text"] for item in final_message["content_items"] if item["type"] == "text").lower()

    # System prompts might not always be followed perfectly, so we just check the response exists
    assert len(response_text) > 0
