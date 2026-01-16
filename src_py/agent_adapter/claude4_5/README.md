# Claude 4.5 Python Client

A Python client implementation for Anthropic's Claude 4.5 API, providing extended thinking and tool use capabilities through a unified interface.

## Features

- **Extended Thinking**: Support for Claude's extended thinking capability with configurable budget tokens
- **Tool Use**: Full support for function calling and tool use
- **Streaming Responses**: Efficient streaming of responses with proper event handling
- **Stateful Conversations**: Built-in conversation history management
- **Universal Interface**: Compatible with the AgentAdapter's unified API

## Installation

```bash
pip install -e .
```

This will install the required dependencies including the `anthropic` SDK.

## Quick Start

### Basic Usage

```python
import asyncio
from agent_adapter import AutoLLMClient

async def main():
    # Create Claude client
    client = AutoLLMClient(model="claude-sonnet-4-5-20250929")
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content_items": [{"type": "text", "text": "Hello, Claude!"}]
        }
    ]
    
    # Configure request
    config = {"max_tokens": 200}
    
    # Stream response
    async for event in client.streaming_response(messages=messages, config=config):
        for item in event["content_items"]:
            if item["type"] == "text" and item["text"]:
                print(item["text"], end="", flush=True)

asyncio.run(main())
```

### Extended Thinking

Claude 4.5 supports extended thinking, which allows the model to reason through complex problems:

```python
from agent_adapter import AutoLLMClient
from agent_adapter.types import ThinkingLevel

client = AutoLLMClient(model="claude-sonnet-4-5-20250929")

messages = [
    {
        "role": "user",
        "content_items": [
            {"type": "text", "text": "What is 127 * 89?"}
        ]
    }
]

config = {
    "max_tokens": 3200,
    "thinking_level": ThinkingLevel.HIGH  # Enable extended thinking
}

async for event in client.streaming_response(messages=messages, config=config):
    for item in event["content_items"]:
        if item["type"] == "reasoning":  # Thinking blocks
            print(f"[THINKING] {item['reasoning']}")
        elif item["type"] == "text":  # Response
            print(f"[RESPONSE] {item['text']}")
```

### Tool Use

Claude can use tools/functions to accomplish tasks:

```python
from agent_adapter import AutoLLMClient

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
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
]

messages = [
    {
        "role": "user",
        "content_items": [{"type": "text", "text": "What's the weather in San Francisco?"}]
    }
]

config = {
    "max_tokens": 1024,
    "tools": tools
}

# First request - Claude will request to use the tool
events = []
async for event in client.streaming_response(messages=messages, config=config):
    events.append(event)

# Get the complete message
assistant_message = client.concat_uni_events_to_uni_message(events)

# Check for function calls
function_calls = [
    item for item in assistant_message["content_items"] 
    if item["type"] == "function_call"
]

if function_calls:
    # Execute the tool and provide results
    tool_results = []
    for fc in function_calls:
        # Simulate tool execution
        result = execute_tool(fc["name"], fc["argument"])
        tool_results.append({
            "type": "text",
            "text": result,
            "tool_call_id": fc["tool_call_id"]
        })
    
    # Add assistant message and tool results to conversation
    messages.append(assistant_message)
    messages.append({"role": "tool", "content_items": tool_results})
    
    # Get final response
    async for event in client.streaming_response(messages=messages, config=config):
        # Process final response
        pass
```

### Stateful Conversations

The client supports stateful conversations with automatic history management:

```python
from agent_adapter import AutoLLMClient

client = AutoLLMClient(model="claude-sonnet-4-5-20250929")
config = {"max_tokens": 100}

# First message
async for event in client.streaming_response_stateful(
    message={"role": "user", "content_items": [{"type": "text", "text": "My name is Alice"}]},
    config=config
):
    pass

# Second message - Claude remembers the context
async for event in client.streaming_response_stateful(
    message={"role": "user", "content_items": [{"type": "text", "text": "What is my name?"}]},
    config=config
):
    for item in event["content_items"]:
        if item["type"] == "text" and item["text"]:
            print(item["text"], end="")

# Clear history when done
client.clear_history()
```

## Configuration Options

The `config` dictionary supports the following parameters:

- `max_tokens` (int, required): Maximum number of tokens to generate
- `temperature` (float, optional): Sampling temperature (0.0 to 1.0)
- `system_prompt` (str, optional): System instruction for the model
- `thinking_level` (ThinkingLevel, optional): Level of extended thinking
  - `ThinkingLevel.NONE`: No extended thinking
  - `ThinkingLevel.LOW`: Low budget (2000 tokens)
  - `ThinkingLevel.MEDIUM`: Medium budget (5000 tokens)
  - `ThinkingLevel.HIGH`: High budget (10000 tokens)
- `tools` (list, optional): List of tool definitions
- `tool_choice` (ToolChoice, optional): How to handle tool selection
  - `"auto"`: Let Claude decide
  - `"none"`: Don't use tools
  - `"required"`: Force tool use

## Content Item Types

The unified API uses the following content item types:

- `text`: Regular text content
  - Fields: `type`, `text`, `signature` (optional)
- `reasoning`: Claude's internal thinking process
  - Fields: `type`, `reasoning`, `signature` (optional)
- `function_call`: Tool/function call request
  - Fields: `type`, `name`, `argument`, `tool_call_id`, `signature` (optional)
- `image_url`: Image content
  - Fields: `type`, `image_url`

## Examples

See the `examples/` directory for complete working examples:

- `claude_basic_example.py`: Basic streaming usage
- `claude_thinking_example.py`: Extended thinking demonstration
- `claude_tool_use_example.py`: Tool use with multiple steps

## Testing

Run the tests with:

```bash
# Set your API key
export ANTHROPIC_API_KEY=your_api_key_here

# Run tests
pytest tests/test_claude_client.py -v
```

## API Reference

### Claude45Client

The main client class for Claude 4.5.

#### Methods

- `streaming_response(messages, config)`: Stream generate content (stateless)
- `streaming_response_stateful(message, config)`: Stream generate with history management
- `clear_history()`: Clear conversation history
- `get_history()`: Get current conversation history

### Supported Models

- `claude-sonnet-4-5-20250929`: Claude Sonnet 4.5
- `claude-sonnet-4-20250514`: Claude Sonnet 4
- `claude-opus-4-5-20251101`: Claude Opus 4.5
- `claude-opus-4-20250514`: Claude Opus 4
- `claude-haiku-4-5-20251001`: Claude Haiku 4.5

## Implementation Details

### Extended Thinking

The implementation maps the universal `thinking_level` parameter to Claude's `budget_tokens`:

- `ThinkingLevel.LOW` → 2000 tokens
- `ThinkingLevel.MEDIUM` → 5000 tokens
- `ThinkingLevel.HIGH` → 10000 tokens

Thinking blocks are returned as `reasoning` content items with an optional `signature` field for verification.

### Tool Use

Tools are defined using Claude's tool schema format. The client automatically:
- Converts function calls to the unified format
- Preserves tool call IDs for result matching
- Handles tool results in the conversation flow

### Streaming

The implementation properly handles all Claude streaming events:
- `content_block_start`: New content block
- `content_block_delta`: Incremental content
- `content_block_stop`: Content block finished
- `message_start`: Message metadata
- `message_delta`: Message updates (finish reason, usage)
- `message_stop`: Message complete

## License

Copyright 2025 Prism Shadow. Licensed under the Apache License, Version 2.0.
