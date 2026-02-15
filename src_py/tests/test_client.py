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

import base64
import json
import mimetypes
import os
from contextlib import nullcontext

import httpx
import pytest

from agenthub import AutoLLMClient, ThinkingLevel


IMAGE = "https://cdn.britannica.com/80/120980-050-D1DA5C61/Poet-narcissus.jpg"

AVAILABLE_TEXT_MODELS = []
AVAILABLE_VISION_MODELS = []
OPENROUTER_MODELS = []
SILICONFLOW_MODELS = []

if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    AVAILABLE_VISION_MODELS.append("gemini-3-flash-preview")

if os.getenv("ANTHROPIC_API_KEY"):
    AVAILABLE_VISION_MODELS.append("claude-sonnet-4-5-20250929")

if os.getenv("OPENAI_API_KEY"):
    AVAILABLE_VISION_MODELS.append("gpt-5.2")

if os.getenv("GLM_API_KEY"):
    AVAILABLE_TEXT_MODELS.append(pytest.param("glm-5", marks=pytest.mark.xfail(reason="API rate limit")))

if os.getenv("OPENROUTER_API_KEY"):
    OPENROUTER_MODELS.append("z-ai/glm-5")
    OPENROUTER_MODELS.append("qwen/qwen3-30b-a3b-thinking-2507")

if os.getenv("SILICONFLOW_API_KEY"):
    SILICONFLOW_MODELS.append("Pro/zai-org/GLM-5")
    SILICONFLOW_MODELS.append("Qwen/Qwen3-8B")

AVAILABLE_MODELS = AVAILABLE_VISION_MODELS + AVAILABLE_TEXT_MODELS + OPENROUTER_MODELS + SILICONFLOW_MODELS


async def _create_client(model: str) -> AutoLLMClient:
    """Create a client for the given model."""
    if model in OPENROUTER_MODELS:
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
    elif model in SILICONFLOW_MODELS:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        base_url = "https://api.siliconflow.cn/v1"
    else:
        api_key, base_url = None, None

    return AutoLLMClient(model=model, api_key=api_key, base_url=base_url)


async def _check_event_integrity(event: dict) -> None:
    """Check event integrity."""
    assert "role" in event
    assert "event_type" in event
    assert "usage_metadata" in event
    assert "finish_reason" in event
    assert event["role"] in ["user", "assistant"]
    assert event["event_type"] in ["start", "delta", "stop"]
    assert event["finish_reason"] in ["stop", "length", "unknown", None]
    for item in event["content_items"]:
        if item["type"] == "text":
            assert "text" in item
        elif item["type"] == "thinking":
            assert "thinking" in item
        elif item["type"] == "tool_call" or item["type"] == "partial_tool_call":
            assert "name" in item
            assert "arguments" in item
            assert "tool_call_id" in item

    if event["usage_metadata"]:
        assert "prompt_tokens" in event["usage_metadata"]
        assert "thoughts_tokens" in event["usage_metadata"]
        assert "response_tokens" in event["usage_metadata"]
        assert "cached_tokens" in event["usage_metadata"]
        
        if event["usage_metadata"]["prompt_tokens"] is not None:
            assert event["usage_metadata"]["prompt_tokens"] >= 0
        if event["usage_metadata"]["thoughts_tokens"] is not None:
            assert event["usage_metadata"]["thoughts_tokens"] >= 0
        if event["usage_metadata"]["response_tokens"] is not None:
            assert event["usage_metadata"]["response_tokens"] >= 0
        if event["usage_metadata"]["cached_tokens"] is not None:
            assert event["usage_metadata"]["cached_tokens"] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_streaming_response_basic(model):
    """Test basic stateless stream generation."""
    client = await _create_client(model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is 2+3?"}]}]
    config = {}

    text = ""
    async for event in client.streaming_response(messages=messages, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert "5" in text  # 2 + 3 = 5


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_streaming_response_with_all_parameters(model):
    """Test stream generation with all optional parameters."""
    client = await _create_client(model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is 2+3?"}]}]
    config = {"max_tokens": 8192, "temperature": 0.7, "thinking_summary": True, "thinking_level": ThinkingLevel.LOW}

    if model == "gpt-5.2":
        context = pytest.raises(ValueError, match="not support")
    else:
        context = nullcontext()

    with context:
        text = ""
        async for event in client.streaming_response(messages=messages, config=config):
            await _check_event_integrity(event)
            for item in event["content_items"]:
                if item["type"] == "text":
                    text += item["text"]

        assert "5" in text  # 2 + 3 = 5


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_streaming_response_stateful(model):
    """Test stateful stream generation."""
    client = await _create_client(model)
    config = {}

    message1 = {"role": "user", "content_items": [{"type": "text", "text": "My name is Alice"}]}
    async for event in client.streaming_response_stateful(message=message1, config=config):
        await _check_event_integrity(event)

    assert len(client.get_history()) == 2  # user message + assistant response

    message2 = {"role": "user", "content_items": [{"type": "text", "text": "What is my name?"}]}
    text = ""
    async for event in client.streaming_response_stateful(message=message2, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert "alice" in text.lower()
    assert len(client.get_history()) == 4  # 2 previous + 2 new


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_clear_history(model):
    """Test clearing conversation history."""
    client = await _create_client(model)
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
    client = await _create_client(model)
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
    client = await _create_client(model)

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
    partial_tool_call_data = {}

    message1 = {"role": "user", "content_items": [{"type": "text", "text": "What is the weather in San Francisco?"}]}
    async for event in client.streaming_response_stateful(message=message1, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "partial_tool_call":
                if not partial_tool_call_data:
                    partial_tool_call_data = {
                        "name": item["name"],
                        "arguments": item["arguments"],
                        "tool_call_id": item["tool_call_id"],
                    }
                else:
                    partial_tool_call_data["arguments"] += item["arguments"]
            elif item["type"] == "tool_call":
                tool_name = item["name"]
                tool_arguments = item["arguments"]
                tool_call_id = item["tool_call_id"]

    # Check if a function call was made
    assert tool_name == weather_tool["name"]
    assert "location" in tool_arguments
    assert tool_call_id is not None
    assert partial_tool_call_data["name"] == tool_name
    assert partial_tool_call_data["tool_call_id"] == tool_call_id
    assert json.loads(partial_tool_call_data["arguments"]) == tool_arguments

    message2 = {
        "role": "user",
        "content_items": [
            {"type": "tool_result", "text": "It's 20 degrees in San Francisco.", "tool_call_id": tool_call_id}
        ],
    }
    text = ""
    async for event in client.streaming_response_stateful(message=message2, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert "20" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_system_prompt(model):
    """Test system prompt capability."""
    client = await _create_client(model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}]
    config = {"system_prompt": "You are a kitten that must end with the word 'meow'."}

    text = ""
    async for event in client.streaming_response(messages=messages, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert "meow" in text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_image_understanding(model):
    """Test image understanding with a URL."""
    if model not in AVAILABLE_VISION_MODELS:
        pytest.skip(f"Image understanding is not supported by {model}.")

    client = await _create_client(model)
    config = {}
    messages = [
        {
            "role": "user",
            "content_items": [
                {"type": "text", "text": "What's in this image? Describe it briefly."},
                {"type": "image_url", "image_url": IMAGE},
            ],
        }
    ]
    text = ""
    async for event in client.streaming_response(messages=messages, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert ("flower" in text.lower()) or ("narcissus" in text.lower())


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_image_understanding_base64(model):
    """Test image understanding with base64 encoded image."""
    if model not in AVAILABLE_VISION_MODELS:
        pytest.skip(f"Image understanding is not supported by {model}.")

    client = await _create_client(model)
    config = {}

    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(IMAGE)
        image_bytes = response.content
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(IMAGE)

    # Create data URI
    data_uri = f"data:{mime_type};base64,{base64_image}"
    messages = [
        {
            "role": "user",
            "content_items": [
                {"type": "text", "text": "What's in this image? Describe it briefly."},
                {"type": "image_url", "image_url": data_uri},
            ],
        }
    ]
    text = ""
    async for event in client.streaming_response(messages=messages, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert ("flower" in text.lower()) or ("narcissus" in text.lower())


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_tool_result_with_image(model):
    """Test tool result with image_url."""
    if model not in AVAILABLE_VISION_MODELS:
        pytest.skip(f"Image in tool result is not supported by {model}.")

    client = await _create_client(model)

    # Define a tool that returns an image
    image_tool = {
        "name": "get_image",
        "description": "Get an image URL",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The image to retrieve",
                },
            },
            "required": ["query"],
        },
    }

    config = {"tools": [image_tool]}
    tool_call_id = None

    message1 = {
        "role": "user",
        "content_items": [{"type": "text", "text": "Get me a narcissus flower image and describe it briefly."}],
    }
    async for event in client.streaming_response_stateful(message=message1, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "tool_call":
                tool_name = item["name"]
                tool_call_id = item["tool_call_id"]

    assert tool_name == image_tool["name"]
    assert tool_call_id is not None

    message2 = {
        "role": "user",
        "content_items": [
            {
                "type": "tool_result",
                "text": "Here is a narcissus flower image:",
                "images": [IMAGE],
                "tool_call_id": tool_call_id,
            }
        ],
    }
    text = ""
    async for event in client.streaming_response_stateful(message=message2, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert ("flower" in text.lower()) or ("narcissus" in text.lower())


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_tool_use(os.getenv("MODEL", "gpt-5.2")))
