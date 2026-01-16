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
Basic Claude 4.5 example demonstrating stateless streaming.
"""

import asyncio

from agent_adapter import AutoLLMClient


async def main():
    # Create Claude client
    client = AutoLLMClient(model="claude-sonnet-4-5-20250929")

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content_items": [{"type": "text", "text": "Write a haiku about artificial intelligence."}],
        }
    ]

    # Configure request
    config = {
        "max_tokens": 200,
        "temperature": 0.7,
    }

    # Stream response
    print("Claude's response:")
    print("-" * 50)

    async for event in client.streaming_response(messages=messages, config=config):
        for item in event["content_items"]:
            if item["type"] == "text" and item["text"]:
                print(item["text"], end="", flush=True)

    print("\n" + "-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
