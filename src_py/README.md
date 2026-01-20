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
        result = get_weather(**tool_call["argument"])

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
        {"type": "tool_call", "name": "get_weather", "argument": {"location": "London"}, "tool_call_id": "call_abc123"}
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
            "result": "London is 22°C today.",
            "tool_call_id": "call_abc123"  # From tool_call event
        }
    ]
}
```

## Configuration Options

```python
from agenthub import PromptCaching, ThinkingLevel

config = {
    "max_tokens": 500,
    "temperature": 1.0,
    "tools": [tool_definition],
    "tool_choice": "auto",  # "auto", "required", "none", or ["tool_name"]
    "thinking_level": ThinkingLevel.HIGH,
    "system_prompt": "You are a helpful assistant",
    "prompt_caching": PromptCaching.ENABLE,
    "trace_id": "agent1/conversation_001"  # Optional: save conversation trace
}
```

## Conversation Tracing

AgentHub provides a built-in `Tracer` to save and browse conversation history. When you specify a `trace_id` in the config, conversations are automatically saved to both JSON and TXT formats.

### Basic Usage

```python
from agenthub import AutoLLMClient

client = AutoLLMClient(model="gemini-3-flash-preview")

# Add trace_id to config
config = {"trace_id": "agent1/conversation_001"}

async for event in client.streaming_response_stateful(
    message={"role": "user", "content_items": [{"type": "text", "text": "Hello"}]},
    config=config
):
    pass  # Conversation is automatically saved
```

The default cache directory is `cache`, you can change it by setting `AGENTHUB_CACHE_DIR` environment variable.

This creates two files in the `cache` directory:
- `cache/agent1/conversation_001.json` - Structured data with full history and config
- `cache/agent1/conversation_001.txt` - Human-readable conversation format

### Browsing Traces with Web Interface

Start a web server to browse and view saved conversations:

```python
from agenthub import Tracer

# Start web server
Tracer("path/to/cache").start_web_server(host="127.0.0.1", port=5000)
```

Then visit `http://127.0.0.1:5000` in your browser to browse saved conversations.
