# LLM Client Usage Guide

This document demonstrates how to use `AutoLLMClient` for unified LLM interactions.

## Basic Usage

### Stateful Stream Generation

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
    
    # Second message - history is maintained
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [{"type": "text", "text": "What's my name?"}]
        },
        config={}
    ):
        print(event)
    
    # View conversation history
    print("History:", client.get_history())
    
    # Clear history when done
    client.clear_history()

asyncio.run(main())
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
