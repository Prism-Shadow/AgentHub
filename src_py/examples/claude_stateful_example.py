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
Example demonstrating stateful Claude client usage for multi-turn conversations.
"""

import asyncio

from agent_adapter import ClaudeClient, ThinkingLevel


async def main():
    """Main example function."""
    # Initialize Claude client with Claude Sonnet 4.5
    client = ClaudeClient(model="claude-sonnet-4-5-20250929")

    # Configure generation
    config = {
        "max_tokens": 2048,
        "temperature": 1.0,
        "thinking_level": ThinkingLevel.LOW,
        "thinking_summary": None,
        "tools": None,
        "tool_choice": None,
        "system_prompt": "You are a helpful AI assistant.",
    }

    # First turn
    print("User: Tell me a fun fact about space.\n")
    message1 = {
        "role": "user",
        "content_items": [{"type": "text", "text": "Tell me a fun fact about space."}],
    }

    print("Claude: ", end="", flush=True)
    async for event in client.streaming_response_stateful(message=message1, config=config):
        for item in event["content_items"]:
            if item["type"] == "text":
                print(item["text"], end="", flush=True)
    print("\n")

    # Second turn
    print("User: Can you elaborate on that?\n")
    message2 = {
        "role": "user",
        "content_items": [{"type": "text", "text": "Can you elaborate on that?"}],
    }

    print("Claude: ", end="", flush=True)
    async for event in client.streaming_response_stateful(message=message2, config=config):
        for item in event["content_items"]:
            if item["type"] == "text":
                print(item["text"], end="", flush=True)
    print("\n")

    # Show conversation history
    print("\n--- Conversation History ---")
    history = client.get_history()
    for i, msg in enumerate(history, 1):
        role = msg["role"].upper()
        content_preview = ""
        for item in msg["content_items"]:
            if item["type"] == "text":
                content_preview += item["text"][:50]
                break
        print(f"{i}. {role}: {content_preview}...")


if __name__ == "__main__":
    asyncio.run(main())
