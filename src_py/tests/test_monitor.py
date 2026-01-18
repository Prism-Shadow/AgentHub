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

from agenthub import AutoLLMClient
from agenthub.monitor import ConversationMonitor


AVAILABLE_MODELS = []

if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    AVAILABLE_MODELS.append("gemini-3-flash-preview")

if os.getenv("ANTHROPIC_API_KEY"):
    AVAILABLE_MODELS.append("claude-sonnet-4-5-20250929")


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_conversation_monitor_init(temp_cache_dir):
    """Test ConversationMonitor initialization."""
    monitor = ConversationMonitor(cache_dir=temp_cache_dir)
    assert monitor.cache_dir == Path(temp_cache_dir)
    assert monitor.cache_dir.exists()


def test_save_history(temp_cache_dir):
    """Test saving conversation history to a file."""
    monitor = ConversationMonitor(cache_dir=temp_cache_dir)

    # Create sample history
    history = [
        {"role": "user", "content_items": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content_items": [{"type": "text", "text": "Hi there!"}]},
    ]

    # Save history
    relative_path = "test/conversation.txt"
    monitor.save_history(history, relative_path)

    # Verify file exists
    file_path = Path(temp_cache_dir) / relative_path
    assert file_path.exists()

    # Verify content
    content = file_path.read_text()
    assert "USER:" in content
    assert "ASSISTANT:" in content
    assert "Hello" in content
    assert "Hi there!" in content


def test_save_history_creates_directories(temp_cache_dir):
    """Test that saving history creates necessary directories."""
    monitor = ConversationMonitor(cache_dir=temp_cache_dir)

    history = [{"role": "user", "content_items": [{"type": "text", "text": "Test"}]}]

    relative_path = "agent1/subfolder/conversation.txt"
    monitor.save_history(history, relative_path)

    file_path = Path(temp_cache_dir) / relative_path
    assert file_path.exists()
    assert file_path.parent.exists()


def test_save_history_overwrites_existing(temp_cache_dir):
    """Test that saving history overwrites existing files."""
    monitor = ConversationMonitor(cache_dir=temp_cache_dir)

    history1 = [{"role": "user", "content_items": [{"type": "text", "text": "First message"}]}]

    history2 = [
        {"role": "user", "content_items": [{"type": "text", "text": "First message"}]},
        {"role": "assistant", "content_items": [{"type": "text", "text": "Response"}]},
        {"role": "user", "content_items": [{"type": "text", "text": "Second message"}]},
    ]

    relative_path = "test/conversation.txt"

    # Save first history
    monitor.save_history(history1, relative_path)
    file_path = Path(temp_cache_dir) / relative_path
    content1 = file_path.read_text()
    assert "First message" in content1
    assert "Second message" not in content1

    # Save second history (should overwrite)
    monitor.save_history(history2, relative_path)
    content2 = file_path.read_text()
    assert "First message" in content2
    assert "Second message" in content2
    assert "Response" in content2


def test_format_history_with_different_content_types(temp_cache_dir):
    """Test formatting history with different content item types."""
    monitor = ConversationMonitor(cache_dir=temp_cache_dir)

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
            "content_items": [{"type": "tool_result", "result": "Temperature is 20C", "tool_call_id": "call_123"}],
        },
    ]

    relative_path = "test/multi_content.txt"
    monitor.save_history(history, relative_path)

    file_path = Path(temp_cache_dir) / relative_path
    content = file_path.read_text()

    assert "What's in this image?" in content
    assert "Thinking:" in content
    assert "Let me analyze..." in content
    assert "This is a flower." in content
    assert "Tool Result" in content
    assert "Temperature is 20C" in content


def test_web_app_creation(temp_cache_dir):
    """Test web application creation."""
    monitor = ConversationMonitor(cache_dir=temp_cache_dir)
    app = monitor.create_web_app()
    assert app is not None
    assert app.name == "flask"


def test_web_app_browse_empty_directory(temp_cache_dir):
    """Test browsing an empty cache directory."""
    monitor = ConversationMonitor(cache_dir=temp_cache_dir)
    app = monitor.create_web_app()

    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200
        assert b"Conversation Monitor" in response.data


def test_web_app_browse_with_files(temp_cache_dir):
    """Test browsing cache directory with files."""
    monitor = ConversationMonitor(cache_dir=temp_cache_dir)

    # Create some test files
    history = [{"role": "user", "content_items": [{"type": "text", "text": "Test"}]}]
    monitor.save_history(history, "agent1/conv1.txt")
    monitor.save_history(history, "agent1/conv2.txt")
    monitor.save_history(history, "agent2/conv1.txt")

    app = monitor.create_web_app()

    with app.test_client() as client:
        # Browse root
        response = client.get("/")
        assert response.status_code == 200
        assert b"agent1" in response.data
        assert b"agent2" in response.data

        # Browse agent1 directory
        response = client.get("/agent1")
        assert response.status_code == 200
        assert b"conv1.txt" in response.data
        assert b"conv2.txt" in response.data

        # View a file
        response = client.get("/agent1/conv1.txt")
        assert response.status_code == 200
        assert b"Test" in response.data
        assert b"USER:" in response.data


def test_web_app_security_check(temp_cache_dir):
    """Test that web app prevents access outside cache directory."""
    monitor = ConversationMonitor(cache_dir=temp_cache_dir)
    app = monitor.create_web_app()

    with app.test_client() as client:
        # Try to access parent directory
        response = client.get("/../")
        assert response.status_code == 403


def test_web_app_nonexistent_path(temp_cache_dir):
    """Test accessing a nonexistent path."""
    monitor = ConversationMonitor(cache_dir=temp_cache_dir)
    app = monitor.create_web_app()

    with app.test_client() as client:
        response = client.get("/nonexistent")
        assert response.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_monitoring_integration(model, temp_cache_dir):
    """Test monitoring integration with AutoLLMClient."""
    # Use a custom cache directory
    from agenthub import monitor

    original_monitor = monitor._global_monitor
    monitor._global_monitor = ConversationMonitor(cache_dir=temp_cache_dir)

    try:
        client = AutoLLMClient(model=model)
        config = {"monitor_path": "integration_test/conversation.txt"}

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

    finally:
        # Restore original monitor
        monitor._global_monitor = original_monitor


@pytest.mark.asyncio
@pytest.mark.parametrize("model", AVAILABLE_MODELS)
async def test_monitoring_updates_on_multiple_messages(model, temp_cache_dir):
    """Test that monitoring file is updated with each new message."""
    from agenthub import monitor

    original_monitor = monitor._global_monitor
    monitor._global_monitor = ConversationMonitor(cache_dir=temp_cache_dir)

    try:
        client = AutoLLMClient(model=model)
        config = {"monitor_path": "multi_message_test/conversation.txt"}

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

    finally:
        monitor._global_monitor = original_monitor
