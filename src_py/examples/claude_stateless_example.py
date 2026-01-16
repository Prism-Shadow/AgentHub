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
Example demonstrating stateless Claude client usage with extended thinking.
"""

import asyncio

from agent_adapter import ClaudeClient, ThinkingLevel


async def main():
    """Main example function."""
    # Initialize Claude client with Claude Sonnet 4.5
    client = ClaudeClient(model="claude-sonnet-4-5-20250929")

    # Create a user message
    user_message = {
        "role": "user",
        "content_items": [{"type": "text", "text": "Create a haiku about artificial intelligence."}],
    }

    # Configure generation with extended thinking
    config = {
        "max_tokens": 3200,
        "temperature": 1.0,
        "thinking_level": ThinkingLevel.MEDIUM,
        "thinking_summary": None,
        "tools": None,
        "tool_choice": None,
        "system_prompt": None,
    }

    # Stream the response
    print("Streaming response from Claude:\n")
    async for event in client.streaming_response(messages=[user_message], config=config):
        for item in event["content_items"]:
            if item["type"] == "reasoning":
                print(f"[Thinking] {item['reasoning']}", end="", flush=True)
            elif item["type"] == "text":
                print(f"{item['text']}", end="", flush=True)

    print("\n\nGeneration complete!")


if __name__ == "__main__":
    asyncio.run(main())
