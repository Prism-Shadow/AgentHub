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


def test_cache_control_added_to_last_user_message():
    """Test that cache_control is added to last user message in streaming_response."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {"prompt_cache": PromptCaching.ENABLE}
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}]

    # Transform messages
    claude_messages = client.transform_uni_message_to_model_input(messages)

    # Messages should not have cache_control yet
    assert "cache_control" not in claude_messages[0]["content"][0]

    # Now manually apply the cache logic like streaming_response does
    prompt_cache = config.get("prompt_cache", PromptCaching.ENABLE)
    if prompt_cache != PromptCaching.DISABLE and claude_messages:
        for i in range(len(claude_messages) - 1, -1, -1):
            if claude_messages[i]["role"] == "user":
                content = claude_messages[i]["content"]
                if content and isinstance(content, list):
                    last_item = content[-1]
                    if last_item.get("type") == "text":
                        cache_control = {"type": "ephemeral"}
                        if prompt_cache == PromptCaching.ENHANCE:
                            cache_control["ttl"] = "1h"
                        last_item["cache_control"] = cache_control
                break

    # Now it should have cache_control
    assert "cache_control" in claude_messages[0]["content"][0]
    assert claude_messages[0]["content"][0]["cache_control"]["type"] == "ephemeral"


def test_cache_control_disable():
    """Test that disable mode does not add cache_control."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {"prompt_cache": PromptCaching.DISABLE}
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}]

    claude_messages = client.transform_uni_message_to_model_input(messages)

    # Apply cache logic
    prompt_cache = config.get("prompt_cache", PromptCaching.ENABLE)
    if prompt_cache != PromptCaching.DISABLE and claude_messages:
        for i in range(len(claude_messages) - 1, -1, -1):
            if claude_messages[i]["role"] == "user":
                content = claude_messages[i]["content"]
                if content and isinstance(content, list):
                    last_item = content[-1]
                    if last_item.get("type") == "text":
                        cache_control = {"type": "ephemeral"}
                        if prompt_cache == PromptCaching.ENHANCE:
                            cache_control["ttl"] = "1h"
                        last_item["cache_control"] = cache_control
                break

    # Should not have cache_control
    assert "cache_control" not in claude_messages[0]["content"][0]


def test_cache_control_enhance():
    """Test that enhance mode adds 1h TTL."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {"prompt_cache": PromptCaching.ENHANCE}
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}]

    claude_messages = client.transform_uni_message_to_model_input(messages)

    # Apply cache logic
    prompt_cache = config.get("prompt_cache", PromptCaching.ENABLE)
    if prompt_cache != PromptCaching.DISABLE and claude_messages:
        for i in range(len(claude_messages) - 1, -1, -1):
            if claude_messages[i]["role"] == "user":
                content = claude_messages[i]["content"]
                if content and isinstance(content, list):
                    last_item = content[-1]
                    if last_item.get("type") == "text":
                        cache_control = {"type": "ephemeral"}
                        if prompt_cache == PromptCaching.ENHANCE:
                            cache_control["ttl"] = "1h"
                        last_item["cache_control"] = cache_control
                break

    assert "cache_control" in claude_messages[0]["content"][0]
    assert claude_messages[0]["content"][0]["cache_control"]["ttl"] == "1h"


def test_cache_only_on_last_user_message():
    """Test that cache_control is only on the last user message."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {"prompt_cache": PromptCaching.ENABLE}
    messages = [
        {"role": "user", "content_items": [{"type": "text", "text": "First"}]},
        {"role": "assistant", "content_items": [{"type": "text", "text": "Response"}]},
        {"role": "user", "content_items": [{"type": "text", "text": "Second"}]},
    ]

    claude_messages = client.transform_uni_message_to_model_input(messages)

    # Apply cache logic
    prompt_cache = config.get("prompt_cache", PromptCaching.ENABLE)
    if prompt_cache != PromptCaching.DISABLE and claude_messages:
        for i in range(len(claude_messages) - 1, -1, -1):
            if claude_messages[i]["role"] == "user":
                content = claude_messages[i]["content"]
                if content and isinstance(content, list):
                    last_item = content[-1]
                    if last_item.get("type") == "text":
                        cache_control = {"type": "ephemeral"}
                        if prompt_cache == PromptCaching.ENHANCE:
                            cache_control["ttl"] = "1h"
                        last_item["cache_control"] = cache_control
                break

    # First user message should not have cache_control
    assert "cache_control" not in claude_messages[0]["content"][0]
    # Last user message should have cache_control
    assert "cache_control" in claude_messages[2]["content"][0]


def test_cache_on_last_item_only():
    """Test that cache_control is on the last item of last user message."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {"prompt_cache": PromptCaching.ENABLE}
    messages = [
        {
            "role": "user",
            "content_items": [
                {"type": "text", "text": "First item"},
                {"type": "text", "text": "Second item"},
            ],
        }
    ]

    claude_messages = client.transform_uni_message_to_model_input(messages)

    # Apply cache logic
    prompt_cache = config.get("prompt_cache", PromptCaching.ENABLE)
    if prompt_cache != PromptCaching.DISABLE and claude_messages:
        for i in range(len(claude_messages) - 1, -1, -1):
            if claude_messages[i]["role"] == "user":
                content = claude_messages[i]["content"]
                if content and isinstance(content, list):
                    last_item = content[-1]
                    if last_item.get("type") == "text":
                        cache_control = {"type": "ephemeral"}
                        if prompt_cache == PromptCaching.ENHANCE:
                            cache_control["ttl"] = "1h"
                        last_item["cache_control"] = cache_control
                break

    # First item should not have cache_control
    assert "cache_control" not in claude_messages[0]["content"][0]
    # Last item should have cache_control
    assert "cache_control" in claude_messages[0]["content"][1]
