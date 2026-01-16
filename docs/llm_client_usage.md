# LLMClient Usage Examples

This document demonstrates how to use the `LLMClient` for unified LLM interactions.

## Basic Usage

### Stateless Stream Generation

The stateless method requires passing the full message history on each call:

```python
import asyncio
from agent_adapter import LLMClient, ThinkingLevel

async def main():
    client = LLMClient()
    
    # Simple text generation
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    async for chunk in client.stream_generate(
        messages=messages,
        model="gemini-3-flash-preview"
    ):
        print(chunk)

asyncio.run(main())
```

### Stateful Stream Generation

The stateful method maintains conversation history internally:

```python
import asyncio
from agent_adapter import LLMClient

async def main():
    client = LLMClient()
    
    # First message
    async for chunk in client.stream_generate_stateful(
        message={"role": "user", "content": "My name is Alice"},
        model="gemini-3-flash-preview"
    ):
        print(chunk)
    
    # Second message - history is maintained
    async for chunk in client.stream_generate_stateful(
        message={"role": "user", "content": "What's my name?"},
        model="gemini-3-flash-preview"
    ):
        print(chunk)
    
    # View conversation history
    print("History:", client.get_history())
    
    # Clear history when done
    client.clear_history()

asyncio.run(main())
```

## Advanced Features

### Message Content Types

Content can be a string or a list of content objects:

```python
# Text only
message = {
    "role": "user",
    "content": "Hello"
}

# Mixed content with image
message = {
    "role": "user",
    "content": [
        {"type": "text", "value": "What's in this image?"},
        {"type": "image_url", "value": "https://example.com/image.jpg"}
    ]
}
```

### Tool Usage

```python
async def main():
    client = LLMClient()
    
    # Define tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
    ]
    
    messages = [
        {"role": "user", "content": "What's the weather in Boston?"}
    ]
    
    # Use tools with auto mode
    async for chunk in client.stream_generate(
        messages=messages,
        model="gemini-3-flash-preview",
        tools=tools,
        tool_choice="auto"
    ):
        print(chunk)

asyncio.run(main())
```

### Thinking Levels

Control the depth of reasoning:

```python
from agent_adapter import ThinkingLevel

async def main():
    client = LLMClient()
    
    messages = [
        {"role": "user", "content": "Solve this complex problem..."}
    ]
    
    # Use high thinking level for complex reasoning
    async for chunk in client.stream_generate(
        messages=messages,
        model="gemini-3-flash-preview",
        thinking_level=ThinkingLevel.HIGH
    ):
        print(chunk)

asyncio.run(main())
```

### Tool Choice Options

```python
# No tools (disabled)
tool_choice = "none"

# Automatic tool selection
tool_choice = "auto"

# Require tool usage
tool_choice = "required"

# Force specific tools
tool_choice = ["get_weather", "get_forecast"]
```

### Temperature and Max Tokens

```python
async def main():
    client = LLMClient()
    
    messages = [
        {"role": "user", "content": "Tell me a creative story"}
    ]
    
    async for chunk in client.stream_generate(
        messages=messages,
        model="gemini-3-flash-preview",
        temperature=0.9,  # Higher for more creativity
        max_tokens=500    # Limit response length
    ):
        print(chunk)

asyncio.run(main())
```

## Message Format

### Basic Message Structure

```python
{
    "role": "user",           # Required: user, assistant, tool, or system
    "content": "Hello",       # Required: string or list of content objects
    "tool_call_id": "call_1"  # Optional: for tool responses
}
```

### Content Object Types

```python
# Text content
{"type": "text", "value": "Hello world"}

# Image URL
{"type": "image_url", "value": "https://example.com/image.jpg"}
```

## Type Definitions

The package provides helpful type definitions:

```python
from agent_adapter import (
    LLMClient,
    Message,
    MessageDict,
    ContentPart,
    ThinkingLevel,
    ToolChoice
)
```

### ThinkingLevel Enum

- `ThinkingLevel.NONE`: No reasoning
- `ThinkingLevel.LOW`: Minimal reasoning
- `ThinkingLevel.MEDIUM`: Moderate reasoning
- `ThinkingLevel.HIGH`: Maximum reasoning depth

### ToolChoice Type

Can be one of:
- String literal: `"none"`, `"auto"`, `"required"`
- List of tool names: `["tool1", "tool2"]`
