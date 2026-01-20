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


from agenthub import PromptCache
from agenthub.claude4_5 import Claude4_5Client


def test_prompt_cache_enable_config():
    """Test that enable mode adds cache_control without explicit TTL (defaults to 5m)."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {
        "system_prompt": "You are a helpful assistant.",
        "prompt_cache": PromptCache.ENABLE,
    }

    claude_config = client.transform_uni_config_to_model_config(config)

    assert "system" in claude_config
    assert isinstance(claude_config["system"], list)
    assert len(claude_config["system"]) == 1
    assert claude_config["system"][0]["type"] == "text"
    assert claude_config["system"][0]["text"] == "You are a helpful assistant."
    assert "cache_control" in claude_config["system"][0]
    assert claude_config["system"][0]["cache_control"]["type"] == "ephemeral"
    assert "ttl" not in claude_config["system"][0]["cache_control"]


def test_prompt_cache_disable_config():
    """Test that disable mode does not add cache_control."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {
        "system_prompt": "You are a helpful assistant.",
        "prompt_cache": PromptCache.DISABLE,
    }

    claude_config = client.transform_uni_config_to_model_config(config)

    assert "system" in claude_config
    assert isinstance(claude_config["system"], str)
    assert claude_config["system"] == "You are a helpful assistant."


def test_prompt_cache_enhance_config():
    """Test that enhance mode adds cache_control with 1h TTL."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {
        "system_prompt": "You are a helpful assistant.",
        "prompt_cache": PromptCache.ENHANCE,
    }

    claude_config = client.transform_uni_config_to_model_config(config)

    assert "system" in claude_config
    assert isinstance(claude_config["system"], list)
    assert len(claude_config["system"]) == 1
    assert claude_config["system"][0]["type"] == "text"
    assert claude_config["system"][0]["text"] == "You are a helpful assistant."
    assert "cache_control" in claude_config["system"][0]
    assert claude_config["system"][0]["cache_control"]["type"] == "ephemeral"
    assert claude_config["system"][0]["cache_control"]["ttl"] == "1h"


def test_prompt_cache_default():
    """Test that default behavior enables caching."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {
        "system_prompt": "You are a helpful assistant.",
    }

    claude_config = client.transform_uni_config_to_model_config(config)

    assert "system" in claude_config
    assert isinstance(claude_config["system"], list)
    assert "cache_control" in claude_config["system"][0]


def test_no_system_prompt_no_cache():
    """Test that no system prompt means no cache_control."""
    client = Claude4_5Client(model="claude-sonnet-4-5-20250929", api_key="dummy")
    config = {
        "prompt_cache": PromptCache.ENABLE,
    }

    claude_config = client.transform_uni_config_to_model_config(config)

    assert "system" not in claude_config
