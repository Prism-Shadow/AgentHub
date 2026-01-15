# Agent Adapter

A unified LLM client that provides a consistent interface for different model SDKs.

This repository contains both Python and TypeScript implementations of the Agent Adapter, designed to simplify interactions with various Large Language Models through a unified API.

## Features

- 🔄 **Unified Interface**: Single API for multiple LLM providers
- 🌊 **Streaming Support**: Async streaming generation for real-time responses
- 💬 **Stateful & Stateless**: Support for both conversation modes
- 🎯 **Type Safe**: Full type definitions for TypeScript and Python type hints
- 🛠️ **Tool Support**: Integrated function calling capabilities
- 🧠 **Thinking Levels**: Control reasoning depth (none, low, medium, high)
- 🖼️ **Multimodal**: Support for text, images, and other content types

## Python Package

### Installation

```bash
cd src_py
pip install -e .
```

### Quick Start

```python
import asyncio
from agent_adapter import LLMClient, ThinkingLevel

async def main():
    client = LLMClient()
    
    # Stateless generation
    messages = [{"role": "user", "content": "Hello!"}]
    async for chunk in client.stream_generate(
        messages=messages,
        model="gemini-3-flash-preview"
    ):
        print(chunk)

asyncio.run(main())
```

### Examples

```bash
cd src_py
python examples/llm_client_demo.py
```

### Documentation

See [docs/llm_client_usage.md](docs/llm_client_usage.md) for comprehensive usage examples and API documentation.

### Development

```bash
cd src_py

# Install dependencies
make install

# Run tests
make test

# Run linter
make lint

# Build package
make build
```

## TypeScript Package

TypeScript sources live in `src_ts/`.

```bash
cd src_ts
make
npm run start
```

## API Overview

### LLMClient Methods

#### `stream_generate` (Stateless)
Async method that requires full message history on each call.

```python
async for chunk in client.stream_generate(
    messages=[...],
    model="gemini-3-flash-preview",
    max_tokens=500,
    temperature=0.7,
    tools=[...],
    thinking_level=ThinkingLevel.HIGH,
    tool_choice="auto"
):
    print(chunk)
```

#### `stream_generate_stateful` (Stateful)
Async method that maintains conversation history internally.

```python
async for chunk in client.stream_generate_stateful(
    message={"role": "user", "content": "Hello"},
    model="gemini-3-flash-preview"
):
    print(chunk)
```

### Message Format

```python
{
    "role": "user",              # Required: user, assistant, tool, system
    "content": "Hello",          # Can be string or list of content objects
    "tool_call_id": "call_123"   # Optional: for tool responses
}
```

### Content Types

```python
# Simple text
{"type": "text", "value": "Hello world"}

# Image URL
{"type": "image_url", "value": "https://example.com/image.jpg"}
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
