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
Example demonstrating tool calling with multi-turn conversation.

This example shows how to use function calling with AutoLLMClient to query weather information.
"""

import asyncio
import json
import os

from agent_adapter import AutoLLMClient, UniEvent


def get_current_temperature(location: str) -> str:
    """Mock function to get current temperature for a location."""
    # Mock weather data
    weather_data = {
        "London": "15°C",
        "San Francisco": "18°C",
        "Tokyo": "22°C",
        "New York": "12°C",
        "Paris": "14°C",
    }
    return weather_data.get(location, "20°C")


async def main():
    """Example of tool calling with AutoLLMClient."""
    print("=" * 60)
    print("Tool Calling Example")
    print("=" * 60)

    # Get model from environment variable, default to gemini
    model = os.getenv("MODEL", "gemini-3-flash-preview")
    print(f"Using model: {model}")

    # Define the function declaration for the model
    # Different models use slightly different schemas
    if "claude" in model.lower():
        # Claude uses input_schema
        weather_function = {
            "name": "get_current_temperature",
            "description": "Gets the current temperature for a given location.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco",
                    },
                },
                "required": ["location"],
            },
        }
    else:
        # Gemini uses parameters
        weather_function = {
            "name": "get_current_temperature",
            "description": "Gets the current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco",
                    },
                },
                "required": ["location"],
            },
        }

    client = AutoLLMClient(model=model)
    config = {"tools": [weather_function]}

    # Claude requires max_tokens
    if "claude" in model.lower():
        config["max_tokens"] = 1024

    # First turn: User asks about temperature
    print("User: What's the temperature in London?")
    print("Assistant:")

    events: list[UniEvent] = []
    async for event in client.streaming_response_stateful(
        message={"role": "user", "content_items": [{"type": "text", "text": "What's the temperature in London?"}]},
        config=config,
    ):
        print(event)
        events.append(event)

    # Check if there's a function call in the last event
    function_call = None
    tool_call_id = None
    for event in events:
        for item in event["content_items"]:
            if item["type"] == "function_call":
                function_call = item
                tool_call_id = item["tool_call_id"]
                break

        if function_call:
            break

    if function_call:
        print(f"\nFunction to call: {function_call['name']}")
        print(f"Arguments: {function_call['argument']}")

        # Call the function
        args = json.loads(function_call["argument"])
        result = get_current_temperature(**args)
        print(f"Function result: {result}")

        # Second step: Send function result back to the model
        print("\nSending function result back to model...")
        print("Assistant:")

        async for event in client.streaming_response_stateful(
            message={
                "role": "tool",
                "content_items": [{"type": "text", "text": result, "tool_call_id": tool_call_id}],
            },
            config=config,
        ):
            print(event)

        # Second turn: Ask a follow-up question
        print("\nUser: How about in Paris?")
        print("Assistant:")

        events2 = []
        async for event in client.streaming_response_stateful(
            message={"role": "user", "content_items": [{"type": "text", "text": "How about in Paris?"}]},
            config=config,
        ):
            print(event)
            events2.append(event)

        # Check for another function call
        function_call2 = None
        tool_call_id2 = None
        for event in events2:
            for item in event["content_items"]:
                if item["type"] == "function_call":
                    function_call2 = item
                    tool_call_id2 = item["tool_call_id"]
                    break

            if function_call2:
                break

        if function_call2:
            print(f"\nFunction to call: {function_call2['name']}")
            print(f"Arguments: {function_call2['argument']}")

            # Call the function again
            args2 = json.loads(function_call2["argument"])
            result2 = get_current_temperature(**args2)
            print(f"Function result: {result2}")

            # Send result back
            print("\nSending function result back to model...")
            print("Assistant:")

            async for event in client.streaming_response_stateful(
                message={
                    "role": "tool",
                    "content_items": [{"type": "text", "text": result2, "tool_call_id": tool_call_id2}],
                },
                config=config,
            ):
                print(event)

    print("\n" + "=" * 60)
    print("Conversation complete!")
    print("=" * 60)

    print("History:")
    for i, msg in enumerate(client.get_history(), 1):
        print(f"[{i}] {msg}")


if __name__ == "__main__":
    asyncio.run(main())
