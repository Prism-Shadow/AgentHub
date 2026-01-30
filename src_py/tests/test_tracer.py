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
import shutil
import tempfile
from pathlib import Path

import pytest
from flask import Flask

from agenthub import AutoLLMClient
from agenthub.integration.tracer import Tracer


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_tracer_init(temp_cache_dir):
    """Test Tracer initialization."""
    tracer = Tracer(cache_dir=temp_cache_dir)
    assert tracer.cache_dir == Path(temp_cache_dir)
    assert tracer.cache_dir.exists()


def test_save_history(temp_cache_dir):
    """Test saving conversation history to a file."""
    tracer = Tracer(cache_dir=temp_cache_dir)

    # Create sample history
    model = "fake-model"
    history = [
        {"role": "user", "content_items": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content_items": [{"type": "text", "text": "Hi there!"}]},
    ]

    # Save history
    file_id = "test/conversation"
    config = {"temperature": 0.7}
    tracer.save_history(model, history, file_id, config)

    # Verify files exist (both JSON and TXT)
    json_path = Path(temp_cache_dir) / (file_id + ".json")
    txt_path = Path(temp_cache_dir) / (file_id + ".txt")
    assert json_path.exists()
    assert txt_path.exists()

    # Verify TXT content
    content = txt_path.read_text()
    assert "USER:" in content
    assert "ASSISTANT:" in content
    assert "Hello" in content
    assert "Hi there!" in content
    assert "temperature" in content

    # Verify JSON content
    import json

    with open(json_path) as f:
        data = json.load(f)
    assert "history" in data
    assert "config" in data
    assert len(data["history"]) == 2


def test_save_history_creates_directories(temp_cache_dir):
    """Test that saving history creates necessary directories."""
    tracer = Tracer(cache_dir=temp_cache_dir)

    model = "fake-model"
    history = [{"role": "user", "content_items": [{"type": "text", "text": "Test"}]}]

    file_id = "agent1/subfolder/conversation"
    config = {}
    tracer.save_history(model, history, file_id, config)

    json_path = Path(temp_cache_dir) / (file_id + ".json")
    assert json_path.exists()
    assert json_path.parent.exists()


def test_save_history_overwrites_existing(temp_cache_dir):
    """Test that saving history overwrites existing files."""
    tracer = Tracer(cache_dir=temp_cache_dir)

    model = "fake-model"
    history1 = [{"role": "user", "content_items": [{"type": "text", "text": "First message"}]}]

    history2 = [
        {"role": "user", "content_items": [{"type": "text", "text": "First message"}]},
        {"role": "assistant", "content_items": [{"type": "text", "text": "Response"}]},
        {"role": "user", "content_items": [{"type": "text", "text": "Second message"}]},
    ]

    file_id = "test/conversation"
    config = {}

    # Save first history
    tracer.save_history(model, history1, file_id, config)
    txt_path = Path(temp_cache_dir) / (file_id + ".txt")
    content1 = txt_path.read_text()
    assert "First message" in content1
    assert "Second message" not in content1

    # Save second history (should overwrite)
    tracer.save_history(model, history2, file_id, config)
    content2 = txt_path.read_text()
    assert "First message" in content2
    assert "Second message" in content2
    assert "Response" in content2


def test_format_history_with_different_content_types(temp_cache_dir):
    """Test formatting history with different content item types."""
    tracer = Tracer(cache_dir=temp_cache_dir)

    model = "fake-model"
    history = [
        {"role": "user", "content_items": [{"type": "text", "text": "What's in this image?"}]},
        {
            "role": "assistant",
            "content_items": [
                {"type": "thinking", "thinking": "Let me analyze..."},
                {"type": "text", "text": "This is a flower."},
            ],
        },
        {
            "role": "user",
            "content_items": [{"type": "tool_result", "text": "Temperature is 20C", "tool_call_id": "call_123"}],
        },
    ]

    relative_path = "test/multi_content"
    config = {"temperature": 0.8}
    tracer.save_history(model, history, relative_path, config)

    file_path = Path(temp_cache_dir) / (relative_path + ".txt")
    content = file_path.read_text()

    assert "What's in this image?" in content
    assert "Thinking:" in content
    assert "Let me analyze..." in content
    assert "This is a flower." in content
    assert "Tool Result" in content
    assert "Temperature is 20C" in content


def test_web_app_creation(temp_cache_dir):
    """Test web application creation."""
    tracer = Tracer(cache_dir=temp_cache_dir)
    app = tracer.create_web_app()
    assert app is not None
    assert isinstance(app, Flask)


def test_web_app_browse_empty_directory(temp_cache_dir):
    """Test browsing an empty cache directory."""
    tracer = Tracer(cache_dir=temp_cache_dir)
    app = tracer.create_web_app()

    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200
        assert b"Tracer" in response.data


def test_web_app_browse_with_files(temp_cache_dir):
    """Test browsing cache directory with files."""
    tracer = Tracer(cache_dir=temp_cache_dir)

    # Create some test files
    model = "fake-model"
    history = [{"role": "user", "content_items": [{"type": "text", "text": "Test"}]}]
    config = {}
    tracer.save_history(model, history, "agent1/conv1", config)
    tracer.save_history(model, history, "agent1/conv2", config)
    tracer.save_history(model, history, "agent2/conv1", config)

    app = tracer.create_web_app()

    with app.test_client() as client:
        # Browse root
        response = client.get("/")
        assert response.status_code == 200
        assert b"agent1" in response.data
        assert b"agent2" in response.data

        # Browse agent1 directory
        response = client.get("/agent1")
        assert response.status_code == 200
        # Should see both .json and .txt files
        assert b"conv1" in response.data
        assert b"conv2" in response.data

        # View a JSON file
        response = client.get("/agent1/conv1.json")
        assert response.status_code == 200
        assert b"Test" in response.data


def test_web_app_security_check(temp_cache_dir):
    """Test that web app prevents access outside cache directory."""
    tracer = Tracer(cache_dir=temp_cache_dir)
    app = tracer.create_web_app()

    with app.test_client() as client:
        # Try to access parent directory
        response = client.get("/../")
        assert response.status_code == 403


def test_web_app_nonexistent_path(temp_cache_dir):
    """Test accessing a nonexistent path."""
    tracer = Tracer(cache_dir=temp_cache_dir)
    app = tracer.create_web_app()

    with app.test_client() as client:
        response = client.get("/nonexistent")
        assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
async def test_monitoring_integration(temp_cache_dir):
    """Test monitoring integration with AutoLLMClient."""

    os.environ["AGENTHUB_CACHE_DIR"] = temp_cache_dir
    client = AutoLLMClient(model="gpt-5.2")
    config = {"trace_id": "integration_test/conversation.txt"}

    message = {"role": "user", "content_items": [{"type": "text", "text": "Say hello"}]}
    async for _ in client.streaming_response_stateful(message=message, config=config):
        pass

    # Verify file was created
    file_path = Path(temp_cache_dir) / "integration_test/conversation.txt"
    assert file_path.exists()

    # Verify content
    content = file_path.read_text()
    assert "Say hello" in content
    assert "USER:" in content
    assert "ASSISTANT:" in content


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
async def test_monitoring_updates_on_multiple_messages(temp_cache_dir):
    """Test that monitoring file is updated with each new message."""

    os.environ["AGENTHUB_CACHE_DIR"] = temp_cache_dir
    client = AutoLLMClient(model="gpt-5.2")
    config = {"trace_id": "multi_message_test/conversation.txt"}

    # First message
    message1 = {"role": "user", "content_items": [{"type": "text", "text": "First question"}]}
    async for _ in client.streaming_response_stateful(message=message1, config=config):
        pass

    file_path = Path(temp_cache_dir) / "multi_message_test/conversation.txt"
    content1 = file_path.read_text()
    assert "First question" in content1

    # Second message
    message2 = {"role": "user", "content_items": [{"type": "text", "text": "Second question"}]}
    async for _ in client.streaming_response_stateful(message=message2, config=config):
        pass

    content2 = file_path.read_text()
    assert "First question" in content2
    assert "Second question" in content2


def test_format_config_with_system_and_tools(temp_cache_dir):
    """Test formatting config with system prompt and tools."""
    tracer = Tracer(cache_dir=temp_cache_dir)

    model = "fake-model"
    history = [{"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}]

    config = {
        "system_prompt": "You are a helpful assistant.",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "City name"}},
                },
            }
        ],
        "temperature": 0.7,
    }
    file_id = "test/config_render"
    tracer.save_history(model, history, file_id, config)

    # Check TXT file
    txt_path = Path(temp_cache_dir) / (file_id + ".txt")
    txt_content = txt_path.read_text()

    # Check that system_prompt is rendered properly
    assert "system_prompt:" in txt_content
    assert "You are a helpful assistant" in txt_content

    # Check that tools are rendered as JSON
    assert "tools:" in txt_content
    assert "get_weather" in txt_content
    assert "parameters" in txt_content


def test_prepare_history_for_display_with_tool_call(temp_cache_dir):
    """Test _prepare_history_for_display correctly processes tool_call arguments with non-ASCII characters."""
    tracer = Tracer(cache_dir=temp_cache_dir)

    history = [
        {
            "role": "assistant",
            "content_items": [
                {
                    "type": "tool_call",
                    "name": "get_weather",
                    "arguments": {"location": "北京", "unit": "celsius", "details": {"温度": "20°C"}},
                    "tool_call_id": "call_123",
                }
            ],
        }
    ]

    processed_history = tracer._prepare_history_for_display(history)

    # Check that arguments_json was created
    tool_call_item = processed_history[0]["content_items"][0]
    assert "arguments_json" in tool_call_item

    # Check that Chinese characters are preserved without Unicode escaping
    arguments_json = tool_call_item["arguments_json"]
    assert "北京" in arguments_json
    assert "温度" in arguments_json
    assert "\\u" not in arguments_json  # No Unicode escaping

    # Verify it's valid JSON
    import json

    parsed = json.loads(arguments_json)
    assert parsed["location"] == "北京"
    assert parsed["details"]["温度"] == "20°C"


def test_prepare_history_for_display_handles_serialization_errors(temp_cache_dir):
    """Test _prepare_history_for_display handles non-serializable arguments gracefully."""
    tracer = Tracer(cache_dir=temp_cache_dir)

    # Create a non-serializable object (like a function)
    class NonSerializable:
        pass

    history = [
        {
            "role": "assistant",
            "content_items": [
                {
                    "type": "tool_call",
                    "name": "test_function",
                    "arguments": {"obj": NonSerializable()},
                    "tool_call_id": "call_456",
                }
            ],
        }
    ]

    # Should not raise an error, should fallback to str representation
    processed_history = tracer._prepare_history_for_display(history)

    tool_call_item = processed_history[0]["content_items"][0]
    assert "arguments_json" in tool_call_item
    # Should have fallen back to string representation
    assert isinstance(tool_call_item["arguments_json"], str)

