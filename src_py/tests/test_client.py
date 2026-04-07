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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import pytest

from agenthub import AutoLLMClient, ThinkingLevel


IMAGE = "https://cdn.britannica.com/80/120980-050-D1DA5C61/Poet-narcissus.jpg"
CAT_IMAGE_PATH = Path(__file__).resolve().parent.parent / "examples" / "cat.jpg"


@dataclass
class Model:
    name: str
    support_vision: bool = True
    support_temperature: bool = True
    provider: Literal["official", "siliconflow", "openrouter", "bedrock", "vertex"] = "official"

    def __repr__(self) -> str:
        return f"{self.name}:{self.provider}"


AVAILABLE_MODELS: list[Model] = []
GEN_IMAGE_MODELS: list[Model] = []

if os.getenv("GEMINI_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="gemini-3-flash-preview"))
    GEN_IMAGE_MODELS.append(Model(name="gemini-3.1-flash-image-preview"))

if os.getenv("ANTHROPIC_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="claude-sonnet-4-6"))

if os.getenv("OPENAI_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="gpt-5.4", support_temperature=False))

if os.getenv("ZAI_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="glm-5", support_vision=False))

if os.getenv("MOONSHOT_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="kimi-k2.5", support_temperature=False))

RUN_SLOW_TEST = os.getenv("RUN_SLOW_TEST", "0") == "1"

if os.getenv("OPENROUTER_API_KEY") and RUN_SLOW_TEST:
    AVAILABLE_MODELS.append(Model(name="z-ai/glm-5", provider="openrouter", support_vision=False))
    AVAILABLE_MODELS.append(
        Model(name="qwen/qwen3-30b-a3b-thinking-2507", provider="openrouter", support_vision=False)
    )
    AVAILABLE_MODELS.append(Model(name="moonshotai/kimi-k2.5", provider="openrouter", support_temperature=False))

if os.getenv("SILICONFLOW_API_KEY") and RUN_SLOW_TEST:
    AVAILABLE_MODELS.append(Model(name="Pro/zai-org/GLM-5", provider="siliconflow", support_vision=False))
    AVAILABLE_MODELS.append(Model(name="Qwen/Qwen3-8B", provider="siliconflow", support_vision=False))
    AVAILABLE_MODELS.append(Model(name="Pro/moonshotai/Kimi-K2.5", provider="siliconflow", support_temperature=False))

if os.getenv("BEDROCK_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="global.anthropic.claude-sonnet-4-6", provider="bedrock"))

if os.getenv("VERTEX_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="gemini-3-flash-preview", provider="vertex"))
    GEN_IMAGE_MODELS.append(Model(name="gemini-3.1-flash-image-preview", provider="vertex"))


async def _create_client(model: Model) -> AutoLLMClient:
    """Create a client for the given model."""
    if model.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
    elif model.provider == "siliconflow":
        api_key = os.getenv("SILICONFLOW_API_KEY")
        base_url = "https://api.siliconflow.cn/v1"
    elif model.provider == "bedrock":
        api_key = os.getenv("BEDROCK_API_KEY")
        base_url = "bedrock://us-east-1"
    elif model.provider == "vertex":
        api_key = os.getenv("VERTEX_API_KEY")
        base_url = None
    else:
        api_key, base_url = None, None

    return AutoLLMClient(model=model.name, api_key=api_key, base_url=base_url)


async def _check_event_integrity(event: dict) -> None:
    """Check event integrity."""
    assert "role" in event
    assert "event_type" in event
    assert "usage_metadata" in event
    assert "finish_reason" in event
    assert event["role"] in ["user", "assistant"]
    assert event["event_type"] in ["start", "delta", "stop"]
    assert event["finish_reason"] in ["stop", "length", "tool_call", "unknown", None]
    assert isinstance(event["created_at"], int) and event["created_at"] > 0
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
        assert "cached_tokens" in event["usage_metadata"]
        assert "prompt_tokens" in event["usage_metadata"]
        assert "thoughts_tokens" in event["usage_metadata"]
        assert "response_tokens" in event["usage_metadata"]

        if event["usage_metadata"]["cached_tokens"] is not None:
            assert event["usage_metadata"]["cached_tokens"] >= 0
        if event["usage_metadata"]["prompt_tokens"] is not None:
            assert event["usage_metadata"]["prompt_tokens"] >= 0
        if event["usage_metadata"]["thoughts_tokens"] is not None:
            assert event["usage_metadata"]["thoughts_tokens"] >= 0
        if event["usage_metadata"]["response_tokens"] is not None:
            assert event["usage_metadata"]["response_tokens"] >= 0


def _file_to_data_uri(path: Path) -> str:
    """Convert a local image file to a data URI."""
    mime_type, _ = mimetypes.guess_type(path.name)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type or 'image/jpeg'};base64,{encoded}"


def _collect_inline_images(events: list[dict]) -> list[dict]:
    """Collect inline image items from streamed events."""
    images = []
    for event in events:
        for item in event["content_items"]:
            if item["type"] == "inline_image":
                images.append(item)
    return images


def _inline_image_item_as_image(item: dict):
    """Convert an inline_image content item back to a Gemini Part image."""
    gemini_types = pytest.importorskip("google.genai.types")
    part = gemini_types.Part.from_bytes(data=item["data"], mime_type=item["mime_type"])
    return part.as_image()


def _skip_if_image_model_unavailable(exc: Exception, model: Model) -> None:
    """Skip image tests when the preview image model or endpoint is unavailable."""
    current: Exception | None = exc
    while current is not None:
        if isinstance(current, (httpx.ConnectError, httpx.TimeoutException)):
            pytest.skip(f"Image endpoint unavailable for {model}: {exc}")
        current = current.__cause__ if isinstance(current.__cause__, Exception) else None

    message = str(exc).lower()
    skip_markers = [
        "404 not found",
        "not found",
        "model not found",
        "not supported",
        "unsupported",
    ]
    if any(marker in message for marker in skip_markers):
        pytest.skip(f"Image model unavailable for {model}: {exc}")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_streaming_response_basic(model: Model):
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
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_streaming_response_with_all_parameters(model: Model):
    """Test stream generation with all optional parameters."""
    client = await _create_client(model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is 2+3?"}]}]
    config = {"max_tokens": 8192, "temperature": 0.7, "thinking_summary": True, "thinking_level": ThinkingLevel.LOW}

    if not model.support_temperature:
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
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_streaming_response_stateful(model: Model):
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
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_clear_history(model: Model):
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
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_set_history(model: Model):
    """Test setting conversation history."""
    client = await _create_client(model)
    new_history: list = [
        {"role": "user", "content_items": [{"type": "text", "text": "Hi"}]},
        {"role": "assistant", "content_items": [{"type": "text", "text": "Hello!"}]},
    ]

    client.set_history(new_history)
    assert client.get_history() == new_history

    # Mutating the original list must not affect the stored history
    new_history.clear()
    assert len(client.get_history()) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_concat_uni_events_to_uni_message(model: Model):
    """Test concatenation of events into a single message."""
    client = await _create_client(model)
    messages = [
        {
            "role": "user",
            "content_items": [{"type": "text", "text": "Say 'The quick brown fox jumps over the lazy dog.'"}],
        }
    ]
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
    all_text = "".join(item["text"] for item in message["content_items"] if item["type"] == "text")
    assert all_text == text


@pytest.mark.asyncio
async def test_unknown_model():
    """Test that unknown models raise ValueError."""
    with pytest.raises(ValueError, match="not support"):
        AutoLLMClient(model="unknown-model")


@pytest.mark.asyncio
async def test_validate_last_event_raises_on_missing_usage_metadata():
    """Test that _validate_last_event raises ValueError when usage_metadata is None."""
    valid_event = {
        "role": "assistant",
        "event_type": "stop",
        "content_items": [],
        "usage_metadata": {"cached_tokens": 0, "prompt_tokens": 10, "thoughts_tokens": None, "response_tokens": 5},
        "finish_reason": "stop",
    }
    AutoLLMClient._validate_last_event(valid_event)  # should not raise

    with pytest.raises(ValueError, match="no events"):
        AutoLLMClient._validate_last_event(None)

    with pytest.raises(ValueError, match="usage_metadata"):
        AutoLLMClient._validate_last_event({**valid_event, "usage_metadata": None})

    with pytest.raises(ValueError, match="finish_reason"):
        AutoLLMClient._validate_last_event({**valid_event, "finish_reason": None})


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_tool_use(model: Model):
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
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_system_prompt(model: Model):
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
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_image_understanding(model: Model):
    """Test image understanding with a URL."""
    if not model.support_vision:
        pytest.skip(f"Image understanding is not supported by {model.name}.")

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
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_image_understanding_base64(model: Model):
    """Test image understanding with base64 encoded image."""
    if not model.support_vision:
        pytest.skip(f"Image understanding is not supported by {model.name}.")

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
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_tool_result_with_image(model: Model):
    """Test tool result with image_url."""
    if not model.support_vision:
        pytest.skip(f"Image in tool result is not supported by {model.name}.")

    client = await _create_client(model)

    # Define a tool that returns an image
    image_tool = {
        "name": "get_image",
        "description": "Get an image URL",
        "parameters": {
            "type": "object",
            "properties": {
                "seed": {
                    "type": "integer",
                    "description": "The random seed to retrieve the image.",
                },
            },
            "required": ["seed"],
        },
    }

    config = {"tools": [image_tool]}
    tool_call_id = None

    message1 = {
        "role": "user",
        "content_items": [{"type": "text", "text": "Get me a random image and describe it briefly."}],
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
                "text": "Here is the result image:",
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


@pytest.mark.asyncio
@pytest.mark.parametrize("model", GEN_IMAGE_MODELS, ids=[str(model) for model in GEN_IMAGE_MODELS])
async def test_image_generation(model: Model):
    """Test streamed text-to-image generation."""
    client = await _create_client(model)
    config = {
        "response_modalities": ["TEXT", "IMAGE"],
        "image_config": {"aspect_ratio": "16:9", "image_size": "1K"},
    }
    messages = [
        {
            "role": "user",
            "content_items": [
                {
                    "type": "text",
                    "text": "Generate a watercolor postcard of a moonlit city skyline with a small boat on a river.",
                }
            ],
        }
    ]

    events = []
    try:
        async for event in client.streaming_response(messages=messages, config=config):
            await _check_event_integrity(event)
            events.append(event)
    except Exception as exc:
        _skip_if_image_model_unavailable(exc, model)
        raise

    inline_images = _collect_inline_images(events)
    assert inline_images, f"No inline images returned by {model.name}"
    assert any(item["mime_type"].startswith("image/") for item in inline_images)
    assert all(isinstance(item["data"], bytes) and item["data"] for item in inline_images)
    assert all(_inline_image_item_as_image(item) is not None for item in inline_images)
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["usage_metadata"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("model", GEN_IMAGE_MODELS, ids=[str(model) for model in GEN_IMAGE_MODELS])
async def test_image_editing(model: Model):
    """Test streamed image editing with a local reference image."""
    client = await _create_client(model)
    config = {
        "response_modalities": ["TEXT", "IMAGE"],
        "image_config": {"aspect_ratio": "1:1", "image_size": "1K"},
    }
    messages = [
        {
            "role": "user",
            "content_items": [
                {
                    "type": "text",
                    "text": "Edit this cat into a cozy watercolor illustration with a blue scarf and rainy window.",
                },
                {"type": "image_url", "image_url": _file_to_data_uri(CAT_IMAGE_PATH)},
            ],
        }
    ]

    events = []
    try:
        async for event in client.streaming_response(messages=messages, config=config):
            await _check_event_integrity(event)
            events.append(event)
    except Exception as exc:
        _skip_if_image_model_unavailable(exc, model)
        raise

    inline_images = _collect_inline_images(events)
    assert inline_images, f"No edited inline images returned by {model.name}"
    assert any(item["mime_type"].startswith("image/") for item in inline_images)
    assert all(isinstance(item["data"], bytes) and item["data"] for item in inline_images)
    assert all(_inline_image_item_as_image(item) is not None for item in inline_images)
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["usage_metadata"] is not None


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_tool_use(Model(name=os.getenv("MODEL", "gpt-5.4"))))
