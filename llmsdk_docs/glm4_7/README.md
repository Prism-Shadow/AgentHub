# GLM-4.7 SDK Documentation

This directory contains comprehensive documentation and examples for using Z.AI's GLM-4.7 API.

## Quick Start

- **Python Users**: See [quickstart.python.md](./quickstart.python.md)

## Documentation

The `docs/` folder contains detailed guides on various GLM-4.7 features:

- [cache.md](./docs/cache.md) - Caching mechanisms for improved performance
- [chat-completion.md](./docs/chat-completion.md) - Chat completion API details
- [function-calling.md](./docs/function-calling.md) - Function calling and tool use
- [streaming.md](./docs/streaming.md) - Streaming responses
- [thinking-mode.md](./docs/thinking-mode.md) - Thinking and reasoning capabilities

## Key Features

GLM-4.7 provides several advanced capabilities:

- **OpenAI-Compatible API**: Use familiar OpenAI SDK patterns with Z.AI models
- **Thinking Mode**: Advanced reasoning for complex problems in mathematics, science, and logic
- **Function Calling**: Tool use and function calling support
- **Streaming**: Real-time response streaming
- **Multimodal Support**: Text, images, audio, video, and file inputs
- **Preserved Thinking**: Maintain reasoning continuity across conversation turns

## Getting Started

GLM-4.7 uses the OpenAI SDK with a custom base URL. Install the OpenAI Python SDK:

```bash
pip install --upgrade 'openai>=1.0'
```

Then configure the client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-Z.AI-api-key",
    base_url="https://api.z.ai/api/paas/v4/"
)

response = client.chat.completions.create(
    model="glm-4.7",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

For more detailed examples and usage patterns, please refer to the [quickstart.python.md](./quickstart.python.md) guide.
