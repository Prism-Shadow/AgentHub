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
from typing import Any, Literal

import httpx
import pytest

from agenthub import AutoLLMClient, PromptCaching, ThinkingLevel, UnsupportedParameterError, list_supported_models


IMAGE = "https://cdn.britannica.com/80/120980-050-D1DA5C61/Poet-narcissus.jpg"
IMAGE_KEYWORDS = ("flower", "narcissus", "daffodil", "bloom")


@dataclass
class Model:
    name: str
    support_text: bool = True
    support_temperature: bool = True
    support_image_understanding: bool = True
    support_image_generation: bool = False
    support_tts: bool = False
    support_embedding: bool = False
    provider: Literal["official", "bedrock", "vertex", "siliconflow", "openrouter", "modelverse"] = "official"
    client_type: str | None = None

    def __repr__(self) -> str:
        return f"{self.name}:{self.provider}"


AVAILABLE_MODELS: list[Model] = []

if os.getenv("GEMINI_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="gemini-3.6-flash", support_temperature=False))
    AVAILABLE_MODELS.append(
        Model(
            name="gemini-3.1-flash-image",
            support_text=False,
            support_temperature=False,
            support_image_understanding=False,
            support_image_generation=True,
        )
    )
    AVAILABLE_MODELS.append(
        Model(
            name="gemini-3.1-flash-tts-preview",
            support_text=False,
            support_temperature=False,
            support_image_understanding=False,
            support_tts=True,
        )
    )
    AVAILABLE_MODELS.append(
        Model(
            name="gemini-embedding-2",
            support_text=False,
            support_temperature=False,
            support_image_understanding=False,
            support_embedding=True,
        )
    )

if os.getenv("ANTHROPIC_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="claude-sonnet-4-6"))

if os.getenv("OPENAI_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="gpt-5.5", support_temperature=False))
    AVAILABLE_MODELS.append(
        Model(
            name="text-embedding-3-large",
            support_text=False,
            support_temperature=False,
            support_image_understanding=False,
            support_embedding=True,
            client_type="openai-embedding",
        )
    )

if os.getenv("ZAI_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="glm-5.2", support_image_understanding=False))

if os.getenv("MOONSHOT_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="kimi-k3", support_temperature=False))

if os.getenv("MINIMAX_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="MiniMax-M3", client_type="minimax-m3"))

if os.getenv("DEEPSEEK_API_KEY"):
    AVAILABLE_MODELS.append(
        Model(name="deepseek-v4-flash", support_temperature=False, support_image_understanding=False)
    )

if os.getenv("BEDROCK_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="global.anthropic.claude-sonnet-4-6", provider="bedrock"))

if os.getenv("VERTEX_API_KEY"):
    AVAILABLE_MODELS.append(Model(name="gemini-3.6-flash", provider="vertex", support_temperature=False))
    AVAILABLE_MODELS.append(
        Model(
            name="gemini-3.1-flash-image",
            provider="vertex",
            support_text=False,
            support_temperature=False,
            support_image_understanding=False,
            support_image_generation=True,
        )
    )
    AVAILABLE_MODELS.append(
        Model(
            name="gemini-3.1-flash-tts-preview",
            provider="vertex",
            support_text=False,
            support_temperature=False,
            support_image_understanding=False,
            support_tts=True,
        )
    )

RUN_SLOW_TEST = os.getenv("RUN_SLOW_TEST", "0") == "1"

if os.getenv("OPENROUTER_API_KEY") and RUN_SLOW_TEST:
    AVAILABLE_MODELS.append(Model(name="z-ai/glm-5.2", provider="openrouter", support_image_understanding=False))
    AVAILABLE_MODELS.append(Model(name="qwen/qwen3.6-35b-a3b", provider="openrouter", client_type="openai"))
    AVAILABLE_MODELS.append(
        Model(
            name="qwen/qwen3-embedding-4b",
            support_text=False,
            support_temperature=False,
            support_image_understanding=False,
            support_embedding=True,
            provider="openrouter",
            client_type="openai-embedding",
        )
    )
    AVAILABLE_MODELS.append(Model(name="moonshotai/kimi-k3", provider="openrouter", support_temperature=False))

if os.getenv("SILICONFLOW_API_KEY") and RUN_SLOW_TEST:
    AVAILABLE_MODELS.append(Model(name="zai-org/GLM-5.2", provider="siliconflow", support_image_understanding=False))
    AVAILABLE_MODELS.append(Model(name="Qwen/Qwen3.6-35B-A3B", provider="siliconflow", client_type="openai"))
    AVAILABLE_MODELS.append(Model(name="Pro/moonshotai/Kimi-K2.6", provider="siliconflow", support_temperature=False))
    AVAILABLE_MODELS.append(
        Model(
            name="Qwen/Qwen3-Embedding-8B",
            support_text=False,
            support_temperature=False,
            support_image_understanding=False,
            support_embedding=True,
            provider="siliconflow",
            client_type="openai-embedding",
        )
    )

if os.getenv("MODELVERSE_API_KEY") and RUN_SLOW_TEST:
    AVAILABLE_MODELS.append(Model(name="claude-sonnet-4-6", provider="modelverse"))
    AVAILABLE_MODELS.append(Model(name="gpt-5.5", provider="modelverse", support_temperature=False))


async def _create_client(model: Model) -> AutoLLMClient:
    """Create a client for the given model."""
    if model.provider == "bedrock":
        api_key = os.getenv("BEDROCK_API_KEY")
        base_url = "bedrock://us-east-1"
    elif model.provider == "vertex":
        api_key = os.getenv("VERTEX_API_KEY")
        base_url = None
    elif model.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
    elif model.provider == "siliconflow":
        api_key = os.getenv("SILICONFLOW_API_KEY")
        base_url = "https://api.siliconflow.cn/v1"
    elif model.provider == "modelverse":
        api_key = os.getenv("MODELVERSE_API_KEY")
        if model.name.startswith("claude-"):
            base_url = "https://api.modelverse.cn/"
        else:
            base_url = "https://api.modelverse.cn/v1"
    else:
        api_key, base_url = None, None

    return AutoLLMClient(model=model.name, api_key=api_key, base_url=base_url, client_type=model.client_type)


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
            assert isinstance(item["text"], str)
        elif item["type"] == "image_url":
            assert isinstance(item["image_url"], str)
        elif item["type"] == "inline_data":
            assert isinstance(item["data"], bytes)
            assert isinstance(item["mime_type"], str)
        elif item["type"] == "thinking":
            assert isinstance(item["thinking"], str)
        elif item["type"] == "inline_thinking":
            assert isinstance(item["data"], bytes)
            assert isinstance(item["mime_type"], str)
        elif item["type"] == "tool_call":
            assert isinstance(item["name"], str)
            assert isinstance(item["arguments"], dict)
            assert isinstance(item["tool_call_id"], str)
        elif item["type"] == "partial_tool_call":
            assert isinstance(item["name"], str)
            assert isinstance(item["arguments"], str)
            assert isinstance(item["tool_call_id"], str)
        elif item["type"] == "tool_result":
            assert isinstance(item["text"], str)
            assert isinstance(item["tool_call_id"], str)

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


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_streaming_response_basic(model: Model):
    """Test basic stateless stream generation."""
    if not model.support_text:
        pytest.skip(f"Text generation is not supported by {model.name}.")

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
    if not model.support_text:
        pytest.skip(f"Text generation is not supported by {model.name}.")

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
    if not model.support_text:
        pytest.skip(f"Text generation is not supported by {model.name}.")

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
async def test_clear_history(model: Model):
    """Test clearing conversation history."""
    client = await _create_client(model)
    new_history: list = [
        {"role": "user", "content_items": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content_items": [{"type": "text", "text": "Hello!"}]},
    ]
    client.set_history(new_history)
    assert len(client.get_history()) > 0

    client.clear_history()
    assert len(client.get_history()) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_concat_uni_events_to_uni_message(model: Model):
    """Test concatenation of events into a single message."""
    if model.support_embedding:
        pytest.skip(f"Embedding model {model.name} do not need concatenation.")
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
async def test_list_supported_models(monkeypatch: pytest.MonkeyPatch):
    """Test that the registry lists model entries accepted by AutoLLMClient."""
    entries = list_supported_models()

    kimi = next(entry for entry in entries if entry["model"] == "kimi-k3")
    assert kimi["base_url"] == "https://api.moonshot.cn/v1"
    assert kimi["client"] == "kimi-k3"
    assert kimi["context_window"] == 1048576
    assert kimi["input_modalities"] == ["Text", "Image"]
    assert kimi["output_modalities"] == ["Text"]
    # stored in USD (official CNY prices pre-converted at 7 CNY/USD)
    assert kimi["pricing"] == {
        "currency": "USD",
        "prompt_tokens": 2.857143,
        "thoughts_tokens": 14.285714,
        "response_tokens": 14.285714,
        "cached_tokens": 0.285714,
    }

    kimi_cny = next(entry for entry in list_supported_models(currency="CNY") if entry["model"] == "kimi-k3")
    assert kimi_cny["pricing"]["currency"] == "CNY"
    assert kimi_cny["pricing"]["prompt_tokens"] == pytest.approx(20.0, abs=1e-4)
    assert kimi_cny["pricing"]["thoughts_tokens"] == pytest.approx(100.0, abs=1e-4)
    assert kimi_cny["pricing"]["response_tokens"] == pytest.approx(100.0, abs=1e-4)
    assert kimi_cny["pricing"]["cached_tokens"] == pytest.approx(2.0, abs=1e-4)

    glm_5_2 = next(entry for entry in entries if entry["model"] == "z-ai/glm-5.2")
    assert glm_5_2["base_url"] == "https://openrouter.ai/api/v1"
    assert glm_5_2["client"] == "glm-5.2"

    for entry in entries:
        assert {"model", "base_url", "client", "input_modalities", "output_modalities"} <= set(entry)
        client = AutoLLMClient(
            model=entry["model"], api_key="test-key", base_url=entry["base_url"], client_type=entry["client"]
        )
        assert client._client is not None

    minimax_entries = [entry for entry in entries if entry["base_url"] == "https://api.minimax.io/v1"]
    assert minimax_entries == [
        {
            "model": "MiniMax-M3",
            "base_url": "https://api.minimax.io/v1",
            "client": "minimax-m3",
            "input_modalities": ["Text", "Image"],
            "output_modalities": ["Text"],
            "context_window": 1000000,
        }
    ]

    minimax_m3 = AutoLLMClient(model="MiniMax-M3", api_key="test-key")
    assert minimax_m3._client.__class__.__name__ == "MiniMaxM3Client"
    # Automatic routing matches the exact model id only ...
    with pytest.raises(ValueError, match="not support"):
        AutoLLMClient(model="MiniMax-M3-preview", api_key="test-key")
    # ... but an explicit client_type reaches gateway-hosted or aliased deployments, as for every
    # other client.
    aliased = AutoLLMClient(model="MiniMax-M3-preview", api_key="test-key", client_type="minimax-m3")
    assert aliased._client.__class__.__name__ == "MiniMaxM3Client"
    # A MiniMax credential is required; the wrapped SDK must never fall back to OPENAI_API_KEY.
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-reach-the-minimax-host")
    with pytest.raises(ValueError, match="MINIMAX_API_KEY is required"):
        AutoLLMClient(model="MiniMax-M3", client_type="minimax-m3")

    expected_m3_efforts = {
        ThinkingLevel.NONE: "none",
        ThinkingLevel.LOW: "low",
        ThinkingLevel.MEDIUM: "medium",
        ThinkingLevel.HIGH: "high",
        ThinkingLevel.XHIGH: "high",
    }
    for thinking_level, effort in expected_m3_efforts.items():
        config = minimax_m3.transform_uni_config_to_model_config({"thinking_level": thinking_level})
        assert config["reasoning"] == {"effort": effort}
    assert "prompt_caching" not in minimax_m3.transform_uni_config_to_model_config(
        {"prompt_caching": PromptCaching.ENABLE}
    )
    for prompt_caching in (PromptCaching.DISABLE, PromptCaching.ENHANCE):
        with pytest.raises(UnsupportedParameterError, match="does not support"):
            minimax_m3.transform_uni_config_to_model_config({"prompt_caching": prompt_caching})
    for tool_choice in ("required", ["lookup"]):
        with pytest.raises(UnsupportedParameterError, match="does not support"):
            minimax_m3.transform_uni_config_to_model_config({"tool_choice": tool_choice})

    # Assistant text order is preserved around top-level items; the reasoning item is rebuilt from
    # the thinking text alone, and the tool call re-serializes its parsed arguments.
    ordered_input = minimax_m3.transform_uni_message_to_model_input(
        [
            {
                "role": "assistant",
                "content_items": [
                    {"type": "text", "text": "before"},
                    {"type": "thinking", "thinking": "reasoning"},
                    {
                        "type": "tool_call",
                        "name": "lookup",
                        "arguments": {"城市": "上海"},
                        "tool_call_id": "call-1",
                    },
                    {"type": "text", "text": "after"},
                ],
            }
        ]
    )
    assert ordered_input == [
        {"role": "assistant", "content": [{"type": "output_text", "text": "before"}]},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "reasoning"}]},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"城市": "上海"}',
        },
        {"role": "assistant", "content": [{"type": "output_text", "text": "after"}]},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("client_type", "client_name"),
    [
        ("openai-compatible", "OpenaiClient"),
        ("openai-embedding-compatible", "OpenaiEmbeddingClient"),
    ],
)
async def test_openai_client_type_routes(client_type: str, client_name: str):
    """Test client_type override for OpenAI-compatible models."""
    client = AutoLLMClient(model="unknown-model", api_key="test-key", client_type=client_type)

    assert client._client.__class__.__name__ == client_name


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
    if not model.support_text:
        pytest.skip(f"Text generation is not supported by {model.name}.")

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
    tool_calls: list[dict[str, Any]] = []
    # A model may open several tool calls for this prompt. Clients whose provider interleaves the
    # argument deltas tag each delta with a correlation id in fidelity; the rest stream one call
    # at a time, so a completed tool_call marks the boundary between them.
    partials: list[dict[str, str]] = []
    partial_by_call: dict[Any, int] = {}
    open_index: int | None = None

    tool_prompt = "What is the weather in San Francisco and in New York?"
    message1 = {"role": "user", "content_items": [{"type": "text", "text": tool_prompt}]}
    async for event in client.streaming_response_stateful(message=message1, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "partial_tool_call":
                item_id = (item.get("fidelity") or {}).get("item_id")
                if item_id is not None:
                    index = partial_by_call.setdefault(item_id, len(partials))
                else:
                    if open_index is None:
                        open_index = len(partials)
                    index = open_index
                while len(partials) <= index:
                    partials.append({"name": "", "arguments": ""})
                partials[index]["name"] = partials[index]["name"] or item["name"]
                partials[index]["arguments"] += item["arguments"]
            elif item["type"] == "tool_call":
                tool_calls.append(item)
                # A completed call ends the uncorrelated stream, so the next delta opens the next.
                open_index = None

    # Check that at least one function call was made, and that each one is fully reconstructable
    # from its own delta stream. Calls are matched by completion order, because a tool_call_id is
    # not unique for every provider: Gemini reuses the function name for all of its parallel calls.
    assert tool_calls
    assert len(partials) == len(tool_calls)
    for tool_call, partial in zip(tool_calls, partials, strict=True):
        assert tool_call["name"] == weather_tool["name"]
        assert "location" in tool_call["arguments"]
        assert tool_call["tool_call_id"] is not None
        assert partial["name"] == tool_call["name"]
        assert json.loads(partial["arguments"]) == tool_call["arguments"]

    # Every open call needs a result, otherwise the replayed history is incomplete.
    message2 = {
        "role": "user",
        "content_items": [
            {
                "type": "tool_result",
                "text": "It's 20 degrees.",
                "tool_call_id": tool_call["tool_call_id"],
            }
            for tool_call in tool_calls
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
    if not model.support_text:
        pytest.skip(f"Text generation is not supported by {model.name}.")

    client = await _create_client(model)
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}]
    config = {
        "system_prompt": "You are a kitten. Every reply MUST contain the exact word 'meow' — "
        "never a variant like 'mreow' or a *purrs* action instead."
    }

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
    if not model.support_image_understanding:
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

    assert any(keyword in text.lower() for keyword in IMAGE_KEYWORDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_image_understanding_base64(model: Model):
    """Test image understanding with base64 encoded image."""
    if not model.support_image_understanding:
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

    assert any(keyword in text.lower() for keyword in IMAGE_KEYWORDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_tool_result_with_image(model: Model):
    """Test tool result with image_url."""
    if not model.support_image_understanding:
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
    tool_calls: list[dict[str, Any]] = []

    # Prescriptive on purpose: this test covers a tool *result* carrying an image, so reaching
    # that state is setup. Natural tool selection is covered by test_tool_use.
    tool_prompt = (
        "Call get_image exactly once with seed 42. Make that function call your only action "
        "this turn, then describe the returned image briefly."
    )
    message1 = {"role": "user", "content_items": [{"type": "text", "text": tool_prompt}]}
    async for event in client.streaming_response_stateful(message=message1, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "tool_call":
                tool_calls.append(item)

    assert tool_calls
    for tool_call in tool_calls:
        assert tool_call["name"] == image_tool["name"]
        assert tool_call["tool_call_id"] is not None

    # Every open call needs a result, otherwise the replayed history is incomplete.
    message2 = {
        "role": "user",
        "content_items": [
            {
                "type": "tool_result",
                "text": "Here is the result image:",
                "images": [IMAGE],
                "tool_call_id": tool_call["tool_call_id"],
            }
            for tool_call in tool_calls
        ],
    }
    text = ""
    async for event in client.streaming_response_stateful(message=message2, config=config):
        await _check_event_integrity(event)
        for item in event["content_items"]:
            if item["type"] == "text":
                text += item["text"]

    assert any(keyword in text.lower() for keyword in IMAGE_KEYWORDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_image_generation(model: Model):
    """Test streamed image generation output."""
    if not model.support_image_generation:
        pytest.skip(f"Image generation is not supported by {model.name}.")

    client = await _create_client(model)
    events = []
    async for event in client.streaming_response(
        messages=[
            {
                "role": "user",
                "content_items": [
                    {
                        "type": "text",
                        "text": "Generate a cozy watercolor illustration of two white flowers with raindrops.",
                    }
                ],
            }
        ],
        config={"image_config": {"aspect_ratio": "1:1", "image_size": "1K"}},
    ):
        await _check_event_integrity(event)
        events.append(event)

    inline_items = [item for event in events for item in event["content_items"] if item["type"] == "inline_data"]
    assert inline_items, f"No inline data returned for generated image by {model.name}"
    assert any("image/" in item["mime_type"] for item in inline_items)
    assert all(isinstance(item["data"], bytes) and item["data"] for item in inline_items)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_tts_generation_single_speaker(model: Model):
    """Test single-speaker TTS output."""
    if not model.support_tts:
        pytest.skip(f"TTS is not supported by {model.name}.")

    client = await _create_client(model)
    events = []
    async for event in client.streaming_response(
        messages=[
            {
                "role": "user",
                "content_items": [
                    {
                        "type": "text",
                        "text": "Say cheerfully: Have a wonderful day!",
                    }
                ],
            }
        ],
        config={"tts_config": [{"voice": "Kore"}]},
    ):
        await _check_event_integrity(event)
        events.append(event)

    inline_items = [item for event in events for item in event["content_items"] if item["type"] == "inline_data"]
    assert inline_items, f"No inline data returned for TTS output by {model.name}"
    assert any("audio/" in item["mime_type"] for item in inline_items)
    assert all(isinstance(item["data"], bytes) and item["data"] for item in inline_items)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS, ids=[str(model) for model in AVAILABLE_MODELS])
async def test_embedding(model: Model):
    """Test streamed text embedding with dimensions configuration."""
    if not model.support_embedding:
        pytest.skip(f"Embedding is not supported by {model.name}.")

    client = await _create_client(model)
    messages = [
        {"role": "user", "content_items": [{"type": "text", "text": "Hello world"}]},
        {"role": "user", "content_items": [{"type": "text", "text": "Goodbye "}, {"type": "text", "text": "world"}]},
    ]

    events = []
    async for event in client.streaming_response(
        messages=messages,
        config={"embedding_config": {"dimensions": 768}},
    ):
        await _check_event_integrity(event)
        events.append(event)

    embedding_items = [item for event in events for item in event["content_items"] if item["type"] == "embedding"]
    assert len(embedding_items) == 2
    for item in embedding_items:
        assert len(item["embedding"]) == 768
        assert all(isinstance(v, float) for v in item["embedding"])


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_tool_use(Model(name=os.getenv("MODEL", "gpt-5.5"))))
