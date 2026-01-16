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
Example demonstrating stateless stream generation.

This example shows how to use the AutoLLMClient without maintaining conversation history.
"""

import asyncio

from agent_adapter import AutoLLMClient


async def main():
    """Example of stateless stream generation."""
    print("=" * 60)
    print("Stateless Example")
    print("=" * 60)

    client = AutoLLMClient(model="gemini-3-flash")

    messages = [{"role": "user", "content_items": [{"type": "text", "text": "Hello! What's 2+2?"}]}]
    config = {"temperature": 0.7}

    print("User: Hello! What's 2+2?")
    print("Assistant: ", end="", flush=True)

    async for event in client.streaming_response(messages=messages, config=config):
        if event["content_items"]:
            for item in event["content_items"]:
                if item.get("type") == "text":
                    print(item.get("text", ""), end="", flush=True)

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
