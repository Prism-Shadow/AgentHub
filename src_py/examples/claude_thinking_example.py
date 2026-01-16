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

"""
Claude 4.5 extended thinking example.
Demonstrates how Claude uses internal reasoning to solve complex problems.
"""

import asyncio

from agent_adapter import AutoLLMClient
from agent_adapter.types import ThinkingLevel


async def main():
    # Create Claude client
    client = AutoLLMClient(model="claude-sonnet-4-5-20250929")

    # Prepare a complex math problem
    messages = [
        {
            "role": "user",
            "content_items": [
                {
                    "type": "text",
                    "text": "What is 127 * 89? Show your reasoning step by step.",
                }
            ],
        }
    ]

    # Configure with extended thinking
    config = {
        "max_tokens": 3200,
        "thinking_level": ThinkingLevel.HIGH,  # Enable extended thinking
    }

    # Stream response
    print("Claude's thinking process and response:")
    print("=" * 60)

    thinking_started = False
    response_started = False

    async for event in client.streaming_response(messages=messages, config=config):
        for item in event["content_items"]:
            if item["type"] == "reasoning" and item["reasoning"]:
                if not thinking_started:
                    print("\n[THINKING]")
                    print("-" * 60)
                    thinking_started = True
                print(item["reasoning"], end="", flush=True)
            elif item["type"] == "text" and item["text"]:
                if not response_started:
                    if thinking_started:
                        print("\n" + "-" * 60)
                    print("\n[RESPONSE]")
                    print("-" * 60)
                    response_started = True
                print(item["text"], end="", flush=True)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
