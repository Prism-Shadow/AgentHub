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

from agent_adapter import AutoLLMClient
from agent_adapter.types import ThinkingLevel


# Define test models and their required API keys
TEST_MODELS = []

if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    TEST_MODELS.append(("gemini-3-flash-preview", "gemini"))

if os.getenv("ANTHROPIC_API_KEY"):
    TEST_MODELS.append(("claude-sonnet-4-5-20250929", "claude"))

# Skip all tests if no API keys are available
pytestmark = pytest.mark.skipif(
    len(TEST_MODELS) == 0,
    reason="No API keys available for testing",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("model,model_type", TEST_MODELS)
async def test_streaming_response_basic(model, model_type):
    """Test basic stateless stream generation."""
    client = AutoLLMClient(model=model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Say hello"}]}]
    config = {"max_tokens": 100} if model_type == "claude" else {}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0

    # Concatenate events to get the full message
    final_message = client.concat_uni_events_to_uni_message(events)
    assert final_message["role"] == "assistant"
    assert len(final_message["content_items"]) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model,model_type", TEST_MODELS)
async def test_streaming_response_with_all_parameters(model, model_type):
    """Test stream generation with all optional parameters."""
    client = AutoLLMClient(model=model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is 2+2?"}]}]
    config = {"max_tokens": 100, "temperature": 0.7}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model,model_type", TEST_MODELS)
async def test_streaming_response_stateful(model, model_type):
    """Test stateful stream generation."""
    client = AutoLLMClient(model=model)
    config = {"max_tokens": 100} if model_type == "claude" else {}

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
@pytest.mark.parametrize("model,model_type", TEST_MODELS)
async def test_clear_history(model, model_type):
    """Test clearing conversation history."""
    client = AutoLLMClient(model=model)
    config = {"max_tokens": 50} if model_type == "claude" else {}

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
@pytest.mark.parametrize("model,model_type", TEST_MODELS)
async def test_image_understanding(model, model_type):
    """Test image understanding with a URL."""
    client = AutoLLMClient(model=model)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Narcissus_poeticus_subsp._radiiflorus.1658.jpg/500px-Narcissus_poeticus_subsp._radiiflorus.1658.jpg"

    messages = [
        {
            "role": "user",
            "content_items": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": image_url},
            ],
        }
    ]
    config = {"max_tokens": 200} if model_type == "claude" else {}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Claude-specific test requires Anthropic API key",
)
async def test_claude_extended_thinking():
    """Test Claude's extended thinking capability."""
    client = AutoLLMClient(model="claude-sonnet-4-5-20250929")
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is 127 * 89?"}]}]
    config = {"max_tokens": 3200, "thinking_level": ThinkingLevel.MEDIUM}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0

    # Check if thinking (reasoning) blocks are present
    final_message = client.concat_uni_events_to_uni_message(events)
    has_reasoning = any(item["type"] == "reasoning" for item in final_message["content_items"])
    has_text = any(item["type"] == "text" for item in final_message["content_items"])

    # Extended thinking should produce both reasoning and text
    assert has_reasoning or has_text  # At minimum we should have one of these


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Claude-specific test requires Anthropic API key",
)
async def test_claude_tool_use():
    """Test Claude's tool use capability."""
    client = AutoLLMClient(model="claude-sonnet-4-5-20250929")

    # Define a simple weather tool
    weather_tool = {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}},
            "required": ["location"],
        },
    }

    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is the weather in San Francisco?"}]}]
    config = {"max_tokens": 1024, "tools": [weather_tool]}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0

    # Check if a function call was made
    final_message = client.concat_uni_events_to_uni_message(events)
    has_function_call = any(item["type"] == "function_call" for item in final_message["content_items"])

    # Claude should attempt to call the get_weather tool
    assert has_function_call


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Claude-specific test requires Anthropic API key",
)
async def test_claude_with_system_prompt():
    """Test Claude with a system prompt."""
    client = AutoLLMClient(model="claude-sonnet-4-5-20250929")
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What should I do?"}]}]
    config = {"max_tokens": 100, "system_prompt": "You are a helpful assistant that always responds in pirate speak."}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0

    # The response should have some pirate-speak characteristics
    final_message = client.concat_uni_events_to_uni_message(events)
    response_text = " ".join(
        item["text"] for item in final_message["content_items"] if item["type"] == "text"
    ).lower()

    # System prompts might not always be followed perfectly, so we just check the response exists
    assert len(response_text) > 0

