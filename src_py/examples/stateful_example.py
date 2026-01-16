#!/usr/bin/env python
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

"""
Example demonstrating stateful stream generation.

This example shows how to use the AutoLLMClient with automatic conversation history management.
"""

import asyncio

from agent_adapter import AutoLLMClient


async def main():
    """Example of stateful stream generation."""
    print("=" * 60)
    print("Stateful Example")
    print("=" * 60)

    client = AutoLLMClient(model="gemini-3-flash-preview")
    config = {}

    # First message
    print("User: My name is Alice")
    print("Assistant: ", end="", flush=True)

    async for event in client.streaming_response_stateful(
        message={"role": "user", "content_items": [{"type": "text", "text": "My name is Alice"}]}, config=config
    ):
        if event["content_items"]:
            for item in event["content_items"]:
                if item.get("type") == "text":
                    print(item.get("text", ""), end="", flush=True)

    print("\n")

    # Second message - history is maintained
    print("User: What's my name?")
    print("Assistant: ", end="", flush=True)

    async for event in client.streaming_response_stateful(
        message={"role": "user", "content_items": [{"type": "text", "text": "What's my name?"}]}, config=config
    ):
        if event["content_items"]:
            for item in event["content_items"]:
                if item.get("type") == "text":
                    print(item.get("text", ""), end="", flush=True)

    print("\n")

    # Show history
    print("Conversation history:")
    for i, msg in enumerate(client.get_history(), 1):
        print(f"  {i}. {msg}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
