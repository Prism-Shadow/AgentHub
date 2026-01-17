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
Example demonstrating extended thinking capability.

This example shows how to use the AutoLLMClient with extended thinking for complex reasoning tasks.
Both Gemini and Claude support extended thinking, though with different configurations.
"""

import asyncio
import os

from agent_adapter import AutoLLMClient
from agent_adapter.types import ThinkingLevel


async def main():
    """Example of extended thinking with AutoLLMClient."""
    print("=" * 60)
    print("Extended Thinking Example")
    print("=" * 60)

    # Get model from environment variable, default to gemini
    model = os.getenv("MODEL", "gemini-3-flash-preview")
    print(f"Using model: {model}")

    client = AutoLLMClient(model=model)

    # Configure extended thinking
    config = {
        "thinking_level": ThinkingLevel.HIGH,
        "thinking_summary": True,
    }

    # Claude requires max_tokens
    if "claude" in model.lower():
        config["max_tokens"] = 3200

    # A complex math problem that benefits from step-by-step reasoning
    query = "What is 127 * 89? Show your step-by-step reasoning."
    print(f"User: {query}")
    print("\nAssistant's response:")
    print("-" * 60)

    thinking_started = False
    response_started = False

    async for event in client.streaming_response(
        messages=[{"role": "user", "content_items": [{"type": "text", "text": query}]}],
        config=config,
    ):
        for item in event["content_items"]:
            if item["type"] == "reasoning" and item["reasoning"]:
                if not thinking_started:
                    print("\n[THINKING PROCESS]")
                    thinking_started = True
                print(item["reasoning"], end="", flush=True)
            elif item["type"] == "text" and item["text"]:
                if not response_started:
                    if thinking_started:
                        print("\n" + "-" * 60)
                    print("\n[FINAL ANSWER]")
                    response_started = True
                print(item["text"], end="", flush=True)

    print("\n" + "=" * 60)
    print("Extended thinking example complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
