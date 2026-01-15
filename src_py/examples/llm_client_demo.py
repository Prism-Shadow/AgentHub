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
Simple example demonstrating the LLMClient usage.

This example shows both stateless and stateful streaming generation.
"""

import asyncio

from agent_adapter import LLMClient, ThinkingLevel


async def stateless_example():
    """Example of stateless stream generation."""
    print("=" * 60)
    print("Stateless Example")
    print("=" * 60)

    client = LLMClient()

    messages = [{"role": "user", "content": "Hello! What's 2+2?"}]

    print("User: Hello! What's 2+2?")
    print("Assistant: ", end="", flush=True)

    async for chunk in client.stream_generate(messages=messages, model="gemini-3-flash-preview", temperature=0.7):
        print(chunk, end="", flush=True)

    print("\n")


async def stateful_example():
    """Example of stateful stream generation."""
    print("=" * 60)
    print("Stateful Example")
    print("=" * 60)

    client = LLMClient()

    # First message
    print("User: My name is Alice")
    print("Assistant: ", end="", flush=True)

    async for chunk in client.stream_generate_stateful(
        message={"role": "user", "content": "My name is Alice"}, model="gemini-3-flash-preview"
    ):
        print(chunk, end="", flush=True)

    print("\n")

    # Second message - history is maintained
    print("User: What's my name?")
    print("Assistant: ", end="", flush=True)

    async for chunk in client.stream_generate_stateful(
        message={"role": "user", "content": "What's my name?"}, model="gemini-3-flash-preview"
    ):
        print(chunk, end="", flush=True)

    print("\n")

    # Show history
    print("Conversation history:")
    for i, msg in enumerate(client.get_history(), 1):
        print(f"  {i}. {msg}")

    print()


async def thinking_example():
    """Example with different thinking levels."""
    print("=" * 60)
    print("Thinking Level Example")
    print("=" * 60)

    client = LLMClient()

    messages = [{"role": "user", "content": "Explain quantum computing in simple terms"}]

    print("User: Explain quantum computing in simple terms")
    print("Assistant (HIGH thinking): ", end="", flush=True)

    async for chunk in client.stream_generate(
        messages=messages, model="gemini-3-flash-preview", thinking_level=ThinkingLevel.HIGH
    ):
        print(chunk, end="", flush=True)

    print("\n")


async def multimodal_example():
    """Example with multimodal content."""
    print("=" * 60)
    print("Multimodal Example")
    print("=" * 60)

    client = LLMClient()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "value": "What's in this image?"},
                {"type": "image_url", "value": "https://example.com/sample.jpg"},
            ],
        }
    ]

    print("User: [Sends image] What's in this image?")
    print("Assistant: ", end="", flush=True)

    async for chunk in client.stream_generate(messages=messages, model="gemini-3-flash-preview"):
        print(chunk, end="", flush=True)

    print("\n")


async def main():
    """Run all examples."""
    await stateless_example()
    await stateful_example()
    await thinking_example()
    await multimodal_example()


if __name__ == "__main__":
    asyncio.run(main())
