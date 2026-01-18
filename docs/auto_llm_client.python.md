# AutoLLMClient Usage Guide

This document demonstrates how to use `AutoLLMClient` for unified LLM interactions in AgentHub.

## AutoLLMClient Overview

`AutoLLMClient` is a stateful client that automatically routes requests to the appropriate model-specific implementation. It maintains conversation history and provides a unified interface for different LLM providers.

### Initialization

Create a client by specifying the model name:

```python
from agenthub import AutoLLMClient

# Initialize with model name
client = AutoLLMClient(model="gemini-3-flash-preview")

# Optionally specify API key (if not using environment variables)
client = AutoLLMClient(model="gemini-3-flash-preview", api_key="your-api-key")
```

The client automatically selects the appropriate backend based on the model name.

## Core Methods

### streaming_response

Stateless method that requires passing the full message history on each call:

```python
import asyncio
from agenthub import AutoLLMClient

async def main():
    client = AutoLLMClient(model="gemini-3-flash-preview")

    messages = [
        {
            "role": "user",
            "content_items": [{"type": "text", "text": "Hello!"}]
        }
    ]

    async for event in client.streaming_response(
        messages=messages,
        config={}
    ):
        print(event)

asyncio.run(main())
```

### streaming_response_stateful

Stateful method that maintains conversation history internally:

```python
import asyncio
from agenthub import AutoLLMClient

async def main():
    client = AutoLLMClient(model="gemini-3-flash-preview")

    # First message
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [{"type": "text", "text": "My name is Alice"}]
        },
        config={}
    ):
        print(event)

    # Second message - history is maintained automatically
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [{"type": "text", "text": "What's my name?"}]
        },
        config={}
    ):
        print(event)

asyncio.run(main())
```

### get_history

Retrieve the conversation history:

```python
# Get all messages in the conversation
history = client.get_history()
print(f"Total messages: {len(history)}")

for msg in history:
    print(f"Role: {msg['role']}")
    print(f"Content: {msg['content_items']}")
```

### clear_history

Clear the conversation history:

```python
# Clear all conversation history
client.clear_history()

# Verify history is empty
assert len(client.get_history()) == 0
```

## Tool Calling

When using tools, you must handle `tool_call_id` correctly:

```python
import asyncio
import json
from agenthub import AutoLLMClient

def get_current_temperature(location: str) -> str:
    """Mock function to get current temperature for a location."""
    return f"Temperature in {location}: 22°C"

async def main():
    # Define tool
    weather_function = {
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco"
                }
            },
            "required": ["location"]
        }
    }

    client = AutoLLMClient(model="gemini-3-flash-preview")
    config = {"tools": [weather_function]}

    # User asks about weather
    events = []
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [{"type": "text", "text": "What's the temperature in London?"}]
        },
        config=config
    ):
        events.append(event)

    # Extract function call and tool_call_id
    tool_call = None
    for event in events:
        for item in event["content_items"]:
            if item["type"] == "tool_call":
                tool_call = item
                break

        if tool_call:
            break

    # Execute function and send result back with tool_call_id
    if tool_call:
        result = get_current_temperature(**tool_call["argument"])

        # IMPORTANT: Include tool_call_id in the tool response
        async for event in client.streaming_response_stateful(
            message={
                "role": "user",
                "content_items": [
                    {
                        "type": "tool_result",
                        "result": result,
                        "tool_call_id": tool_call["tool_call_id"]  # Required for tool responses
                    }
                ]
            },
            config=config
        ):
            print(event)

asyncio.run(main())
```

## Message Format

### UniMessage Structure

```python
{
    "role": "user" | "assistant",
    "content_items": [
        {"type": "text", "text": "Hello"},
        {"type": "image_url", "image_url": "https://..."}
    ]
}
```

### Tool Response with tool_call_id

When responding to a tool call, include the `tool_call_id` in the result content item:

```python
{
    "role": "user",
    "content_items": [
        {
            "type": "tool_result",
            "result": "Tool result data",
            "tool_call_id": "call_abc123"  # From tool_call event
        }
    ]
}
```

## Configuration Options

```python
from agenthub import ThinkingLevel

config = {
    "max_tokens": 500,
    "temperature": 1.0,
    "tools": [tool_definition],
    "tool_choice": "auto",  # "auto", "required", "none", or ["tool_name"]
    "thinking_level": ThinkingLevel.HIGH,
    "system_prompt": "You are a helpful assistant"
}
```
