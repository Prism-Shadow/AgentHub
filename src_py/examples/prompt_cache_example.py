#!/usr/bin/env python3
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

"""Example demonstrating the prompt_cache feature with Claude models."""

import asyncio
import os

from agenthub import AutoLLMClient, PromptCache


async def main():
    """Demonstrate different prompt cache modes."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Skipping example.")
        return

    client = AutoLLMClient(model="claude-sonnet-4-5-20250929")

    # Example 1: Enable prompt caching (default behavior)
    print("\n=== Example 1: Enable prompt caching ===")
    messages = [{"role": "user", "content_items": [{"type": "text", "text": "What is 2+3?"}]}]
    config = {
        "system_prompt": "You are a helpful mathematical assistant.",
        "prompt_cache": PromptCache.ENABLE,  # This is the default
    }

    async for event in client.streaming_response(messages=messages, config=config):
        if event.get("usage_metadata"):
            usage = event["usage_metadata"]
            print("Usage metadata:")
            print(f"  Prompt tokens: {usage.get('prompt_tokens')}")
            print(f"  Response tokens: {usage.get('response_tokens')}")
            print(f"  Cache creation tokens: {usage.get('cache_creation_tokens')}")
            print(f"  Cache read tokens: {usage.get('cache_read_tokens')}")

    # Example 2: Disable prompt caching
    print("\n=== Example 2: Disable prompt caching ===")
    config_disable = {
        "system_prompt": "You are a helpful mathematical assistant.",
        "prompt_cache": PromptCache.DISABLE,
    }

    async for event in client.streaming_response(messages=messages, config=config_disable):
        if event.get("usage_metadata"):
            usage = event["usage_metadata"]
            print("Usage metadata:")
            print(f"  Prompt tokens: {usage.get('prompt_tokens')}")
            print(f"  Response tokens: {usage.get('response_tokens')}")
            print(f"  Cache creation tokens: {usage.get('cache_creation_tokens')}")
            print(f"  Cache read tokens: {usage.get('cache_read_tokens')}")

    # Example 3: Enhanced caching with 1-hour TTL
    print("\n=== Example 3: Enhanced caching (1-hour TTL) ===")
    config_enhance = {
        "system_prompt": "You are a helpful mathematical assistant.",
        "prompt_cache": PromptCache.ENHANCE,
    }

    async for event in client.streaming_response(messages=messages, config=config_enhance):
        if event.get("usage_metadata"):
            usage = event["usage_metadata"]
            print("Usage metadata:")
            print(f"  Prompt tokens: {usage.get('prompt_tokens')}")
            print(f"  Response tokens: {usage.get('response_tokens')}")
            print(f"  Cache creation tokens: {usage.get('cache_creation_tokens')}")
            print(f"  Cache read tokens: {usage.get('cache_read_tokens')}")


if __name__ == "__main__":
    asyncio.run(main())
