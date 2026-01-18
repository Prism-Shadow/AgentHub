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
Example demonstrating conversation monitoring functionality.

This example shows how to:
1. Use monitor_path in UniConfig to save conversation history to files
2. Start a web server to browse and view saved conversations
"""

import asyncio
import os
import threading
import time

from agenthub import AutoLLMClient, get_trace


async def run_monitored_conversation():
    """Run a conversation with monitoring enabled."""
    # Get model from environment variable, default to gemini-3-flash-preview
    model = os.getenv("MODEL", "gemini-3-flash-preview")
    print(f"Using model: {model}")

    client = AutoLLMClient(model=model)

    # First conversation - agent1
    print("\n" + "=" * 60)
    print("Agent 1 Conversation")
    print("=" * 60)

    # Configure with monitor_path to save history (no file extension)
    config = {"monitor_path": "agent1/conversation_001", "temperature": 0.7}

    query1 = "My name is Alice and I like cats."
    print(f"\nUser: {query1}")
    print("Assistant:", end=" ")
    async for event in client.streaming_response_stateful(
        message={"role": "user", "content_items": [{"type": "text", "text": query1}]}, config=config
    ):
        for item in event["content_items"]:
            if item["type"] == "text":
                print(item["text"], end="", flush=True)
    print()

    query2 = "What's my name and what do I like?"
    print(f"\nUser: {query2}")
    print("Assistant:", end=" ")
    async for event in client.streaming_response_stateful(
        message={"role": "user", "content_items": [{"type": "text", "text": query2}]}, config=config
    ):
        for item in event["content_items"]:
            if item["type"] == "text":
                print(item["text"], end="", flush=True)
    print()

    print("\nConversation saved to cache/agent1/conversation_001.json and .txt")

    # Second conversation - agent2
    client2 = AutoLLMClient(model=model)
    print("\n" + "=" * 60)
    print("Agent 2 Conversation")
    print("=" * 60)

    config2 = {"monitor_path": "agent2/session_123", "temperature": 0.7}

    query3 = "What is 2+2?"
    print(f"\nUser: {query3}")
    print("Assistant:", end=" ")
    async for event in client2.streaming_response_stateful(
        message={"role": "user", "content_items": [{"type": "text", "text": query3}]}, config=config2
    ):
        for item in event["content_items"]:
            if item["type"] == "text":
                print(item["text"], end="", flush=True)
    print()

    print("\nConversation saved to cache/agent2/session_123.json and .txt")


def start_web_server():
    """Start the web server in a background thread."""
    trace = get_trace()
    trace.start_web_server(host="127.0.0.1", port=5000, debug=False)


async def main():
    """Main function."""
    print("=" * 60)
    print("Conversation Trace Example")
    print("=" * 60)

    # Run monitored conversations
    await run_monitored_conversation()

    print("\n" + "=" * 60)
    print("Starting Web Server")
    print("=" * 60)
    print("\nYou can now browse the conversations at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server.\n")

    # Start web server in a background thread
    server_thread = threading.Thread(target=start_web_server, daemon=True)
    server_thread.start()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")


if __name__ == "__main__":
    asyncio.run(main())
