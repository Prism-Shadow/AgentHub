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
Claude 4.5 tool use example.
Demonstrates how Claude can use tools/functions to accomplish tasks.
"""

import asyncio

from agent_adapter import AutoLLMClient


async def main():
    # Create Claude client
    client = AutoLLMClient(model="claude-sonnet-4-5-20250929")

    # Define tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit of temperature, either "celsius" or "fahrenheit"',
                    },
                },
                "required": ["location"],
            },
        },
        {
            "name": "get_time",
            "description": "Get the current time in a given time zone",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "The IANA time zone name, e.g. America/Los_Angeles",
                    }
                },
                "required": ["timezone"],
            },
        },
    ]

    # Initial user message
    messages = [
        {
            "role": "user",
            "content_items": [{"type": "text", "text": "What's the weather like in San Francisco?"}],
        }
    ]

    # Configure with tools
    config = {
        "max_tokens": 1024,
        "tools": tools,
    }

    # First request - Claude will request to use the tool
    print("First request - Claude analyzes and requests tool use:")
    print("=" * 60)

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)
        for item in event["content_items"]:
            if item["type"] == "text" and item["text"]:
                print(item["text"], end="", flush=True)

    print("\n" + "=" * 60)

    # Get the complete message
    assistant_message = client.concat_uni_events_to_uni_message(events)

    # Check if Claude wants to use a tool
    function_calls = [item for item in assistant_message["content_items"] if item["type"] == "function_call"]

    if function_calls:
        print("\nClaude wants to call the following tool(s):")
        for fc in function_calls:
            print(f"  - {fc['name']}")
            print(f"    Arguments: {fc['argument']}")

        # Simulate tool execution (in a real app, you'd actually call your functions)
        tool_results = []
        for fc in function_calls:
            if fc["name"] == "get_weather":
                # Simulate weather API call
                tool_results.append(
                    {
                        "type": "text",
                        "text": "The weather in San Francisco is 68°F (20°C), partly cloudy.",
                        "tool_call_id": fc["tool_call_id"],
                    }
                )

        # Add assistant message to conversation
        messages.append(assistant_message)

        # Add tool results as a user message
        messages.append({"role": "tool", "content_items": tool_results})

        # Second request - Claude processes the tool results
        print("\n" + "=" * 60)
        print("Second request - Claude processes tool results:")
        print("=" * 60)

        async for event in client.streaming_response(messages=messages, config=config):
            for item in event["content_items"]:
                if item["type"] == "text" and item["text"]:
                    print(item["text"], end="", flush=True)

        print("\n" + "=" * 60)
    else:
        print("\nClaude didn't request any tools (this is unexpected for this example)")


if __name__ == "__main__":
    asyncio.run(main())
