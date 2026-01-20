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

"""Unit tests for prompt cache configuration (no API calls)."""


from agenthub import PromptCaching
from agenthub.claude4_5 import Claude4_5Client


def test_prompt_cache_enable_config():
    """Test that enable mode adds cache_control to last user message item."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {
        "system_prompt": "You are a helpful assistant.",
        "prompt_cache": PromptCaching.ENABLE,
    }
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}]

    claude_messages = client.transform_uni_message_to_model_input(messages, config)

    assert len(claude_messages) == 1
    assert len(claude_messages[0]["content"]) == 1
    assert "cache_control" in claude_messages[0]["content"][0]
    assert claude_messages[0]["content"][0]["cache_control"]["type"] == "ephemeral"
    assert "ttl" not in claude_messages[0]["content"][0]["cache_control"]


def test_prompt_cache_disable_config():
    """Test that disable mode does not add cache_control."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {
        "system_prompt": "You are a helpful assistant.",
        "prompt_cache": PromptCaching.DISABLE,
    }
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}]

    claude_messages = client.transform_uni_message_to_model_input(messages, config)

    assert len(claude_messages) == 1
    assert len(claude_messages[0]["content"]) == 1
    assert "cache_control" not in claude_messages[0]["content"][0]


def test_prompt_cache_enhance_config():
    """Test that enhance mode adds cache_control with 1h TTL."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {
        "system_prompt": "You are a helpful assistant.",
        "prompt_cache": PromptCaching.ENHANCE,
    }
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}]

    claude_messages = client.transform_uni_message_to_model_input(messages, config)

    assert len(claude_messages) == 1
    assert len(claude_messages[0]["content"]) == 1
    assert "cache_control" in claude_messages[0]["content"][0]
    assert claude_messages[0]["content"][0]["cache_control"]["type"] == "ephemeral"
    assert claude_messages[0]["content"][0]["cache_control"]["ttl"] == "1h"


def test_prompt_cache_default():
    """Test that default behavior enables caching."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {
        "system_prompt": "You are a helpful assistant.",
    }
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}]

    claude_messages = client.transform_uni_message_to_model_input(messages, config)

    assert len(claude_messages) == 1
    assert len(claude_messages[0]["content"]) == 1
    assert "cache_control" in claude_messages[0]["content"][0]


def test_cache_only_on_last_user_message():
    """Test that cache_control is only on the last user message."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {
        "prompt_cache": PromptCaching.ENABLE,
    }
    messages = [
        {"role": "user", "content_items": [{"type": "text", "text": "First message"}]},
        {"role": "assistant", "content_items": [{"type": "text", "text": "Response"}]},
        {"role": "user", "content_items": [{"type": "text", "text": "Second message"}]},
    ]

    claude_messages = client.transform_uni_message_to_model_input(messages, config)

    # First user message should not have cache_control
    assert "cache_control" not in claude_messages[0]["content"][0]
    # Last user message should have cache_control
    assert "cache_control" in claude_messages[2]["content"][0]


def test_cache_on_last_item_of_last_user_message():
    """Test that cache_control is on the last item of last user message."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {
        "prompt_cache": PromptCaching.ENABLE,
    }
    messages = [
        {
            "role": "user",
            "content_items": [
                {"type": "text", "text": "First item"},
                {"type": "text", "text": "Second item"},
            ],
        }
    ]

    claude_messages = client.transform_uni_message_to_model_input(messages, config)

    # First item should not have cache_control
    assert "cache_control" not in claude_messages[0]["content"][0]
    # Last item should have cache_control
    assert "cache_control" in claude_messages[0]["content"][1]
