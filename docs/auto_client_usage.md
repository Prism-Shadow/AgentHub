# AutoLLMClient Usage Guide

This document demonstrates how to use `AutoLLMClient` for unified LLM interactions in AgentAdapter.

## AutoLLMClient Overview

`AutoLLMClient` is a stateful client that automatically routes requests to the appropriate model-specific implementation. It maintains conversation history and provides a unified interface for different LLM providers.

### Initialization

Create a client by specifying the model name:

```python
from agent_adapter import AutoLLMClient

# Initialize with model name
client = AutoLLMClient(model="gemini-3-flash-preview")

# Initialize with custom API key
client = AutoLLMClient(model="gemini-3-flash-preview", api_key="your-api-key")
```

The client automatically selects the appropriate backend based on the model name.

## Core Methods

### streaming_response

Stateless method that requires passing the full message history on each call:

```python
import asyncio
from agent_adapter import AutoLLMClient

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
from agent_adapter import AutoLLMClient

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
from agent_adapter import AutoLLMClient

def get_weather(location: str) -> str:
    """Mock function to get weather."""
    return f"Temperature in {location}: 22°C"

async def main():
    # Define tool
    weather_function = {
        "name": "get_weather",
        "description": "Gets the current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name"
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
            "content_items": [{"type": "text", "text": "What's the weather in London?"}]
        },
        config=config
    ):
        events.append(event)
    
    # Extract function call and tool_call_id
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
    
    # Execute function and send result back with tool_call_id
    if function_call:
        args = json.loads(function_call["argument"])
        result = get_weather(**args)
        
        # IMPORTANT: Include tool_call_id in the tool response
        async for event in client.streaming_response_stateful(
            message={
                "role": "tool",
                "content_items": [
                    {
                        "type": "text",
                        "text": result,
                        "tool_call_id": tool_call_id  # Required for tool responses
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
    "role": "user" | "assistant" | "tool",
    "content_items": [
        {"type": "text", "text": "Hello"},
        {"type": "image_url", "image_url": "https://..."}
    ]
}
```

### Tool Response with tool_call_id

When responding to a function call, include the `tool_call_id` in the text content item:

```python
{
    "role": "tool",
    "content_items": [
        {
            "type": "text",
            "text": "Result data",
            "tool_call_id": "call_abc123"  # From function_call event
        }
    ]
}
```

## Configuration Options

```python
config = {
    "max_tokens": 500,
    "temperature": 0.9,
    "tools": [tool_definition],
    "tool_choice": "auto",  # "auto", "required", "none", or ["tool_name"]
    "thinking_level": ThinkingLevel.HIGH,
    "system_prompt": "You are a helpful assistant"
}
```

## Type Definitions

```python
from agent_adapter import (
    AutoLLMClient,
    UniMessage,
    UniEvent,
    UniConfig,
    ThinkingLevel,
    ToolChoice
)
```
