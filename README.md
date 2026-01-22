![AgentHub](.github/images/header.png)

# AgentHub SDK - Unified and Precise LLM SDK

[![GitHub Repo stars](https://img.shields.io/github/stars/Prism-Shadow/AgentHub?style=social)](https://github.com/Prism-Shadow/AgentHub/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/Prism-Shadow/AgentHub)](https://github.com/Prism-Shadow/AgentHub/commits/main)
[![GitHub contributors](https://img.shields.io/github/contributors/Prism-Shadow/AgentHub?color=orange)](https://github.com/Prism-Shadow/AgentHub/graphs/contributors)
[![Python tests](https://github.com/Prism-Shadow/AgentHub/actions/workflows/pytest.yml/badge.svg)](https://github.com/Prism-Shadow/AgentHub/actions/workflows/pytest.yml)
[![Javascript tests](https://github.com/Prism-Shadow/AgentHub/actions/workflows/jest.yml/badge.svg)](https://github.com/Prism-Shadow/AgentHub/actions/workflows/jest.yml)

[![PyPI](https://img.shields.io/pypi/v/agenthub-python)](https://pypi.org/project/agenthub-python/)
[![NPM](https://img.shields.io/npm/v/@prismshadow/agenthub)](https://www.npmjs.com/package/@prismshadow/agenthub)

[![Twitter](https://img.shields.io/twitter/follow/prismshadow_ai)](https://twitter.com/prismshadow_ai)

AgentHub SDK is the only SDK you need to connect to state-of-the-art LLMs (GPT-5/Claude 4.5/Gemini 3).

https://github.com/user-attachments/assets/c49a21a1-5bf9-4768-a76d-f73c9a03ca87

## Why AgentHub?

- 🔗 **Unified**: A consistent and intuitive interface for interacting with multiple LLMs.

- 🎯 **Precise**: Automatically handles interleaved thinking during multi-step tool calls, preventing performance degradation.

- 🧭 **Traceable**: Provides lightweight logging tool to help developers debug and audit LLM executions.

## Supported Models

|   Model Name   | Vendor                          | Reasoning          | Tool Use           | Image Understanding           |
| -------------- | ------------------------------- | ------------------ | ------------------ | ----------------------------- |
| Gemini 3       | Official                        | :white_check_mark: | :white_check_mark: | :white_check_mark:            |
| Claude 4.5     | Official                        | :white_check_mark: | :white_check_mark: | :white_check_mark:            |
| GPT-5.2        | Official                        | :white_check_mark: | :white_check_mark: | :white_check_mark:            |
| GLM-4.7        | Official/OpenRouter/SiliconFlow | :white_check_mark: | :white_check_mark: | :negative_squared_cross_mark: |
| Qwen3          | OpenRouter/SiliconFlow/vLLM     | :white_check_mark: | :white_check_mark: | :negative_squared_cross_mark: |

## Installation

### Python package

Install from PyPI:

```bash
uv add agenthub-python
# or
pip install agenthub-python
```

Build from source:

```bash
cd src_py && make
```

See [src_py/README.md](src_py/README.md) for comprehensive usage examples and API documentation.

### TypeScript package

Install from npm:

```bash
npm install @prismshadow/agenthub
```

Build from source:

```bash
cd src_ts && make install && make build
```

## APIs

`AutoLLMClient` is the main class for interacting with the AgentHub SDK. It provides the following asynchronous methods:

- `streaming_response(message, config)`: Streams the response of the LLM model in a stateless manner.
- `streaming_response_stateful(message, config)`: Streams the response of the LLM model in a stateful manner.
- `clear_history()`: Clears the history of the stateful LLM model.
- `get_history()`: Returns the history of the stateful LLM model.

## Basic Usage

### GPT-5.2

Python Example:

```python
import asyncio
import os
from agenthub import AutoLLMClient

async def main():
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    client = AutoLLMClient(model="gpt-5.2")
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [{"type": "text", "text": "Say 'Hello, World!'"}]
        },
        config={"temperature": 1.0}
    ):
        print(event)

asyncio.run(main())
# {'role': 'assistant', 'event_type': 'delta', 'content_items': [{'type': 'text', 'text': 'Hello'}], 'usage_metadata': None, 'finish_reason': None}
# {'role': 'assistant', 'event_type': 'delta', 'content_items': [{'type': 'text', 'text': ','}], 'usage_metadata': None, 'finish_reason': None}
# {'role': 'assistant', 'event_type': 'delta', 'content_items': [{'type': 'text', 'text': ' World'}], 'usage_metadata': None, 'finish_reason': None}
# {'role': 'assistant', 'event_type': 'delta', 'content_items': [{'type': 'text', 'text': '!'}], 'usage_metadata': None, 'finish_reason': None}
# {'role': 'assistant', 'event_type': 'stop', 'content_items': [], 'usage_metadata': {'prompt_tokens': 12, 'thoughts_tokens': 0, 'response_tokens': 8, 'cached_tokens': 0}, 'finish_reason': 'stop'}
```

TypeScript Example:

```typescript
import { AutoLLMClient } from "@prismshadow/agenthub";

process.env.OPENAI_API_KEY = "your-api-key";

async function main() {
  const client = new AutoLLMClient({ model: "gpt-5.2" });
  const message = {
    role: "user",
    content_items: [{ type: "text", text: "Say 'Hello, World!'" }],
  }
  const config = {};
  for await (const event of client.streamingResponseStateful({ message, config })) {
    console.log(event);
  }
}

main().catch(console.error);
// {'role': 'assistant', 'event_type': 'delta', 'content_items': [{'type': 'text', 'text': 'Hello'}], 'usage_metadata': null, 'finish_reason': null}
// {'role': 'assistant', 'event_type': 'delta', 'content_items': [{'type': 'text', 'text': ','}], 'usage_metadata': null, 'finish_reason': null}
// {'role': 'assistant', 'event_type': 'delta', 'content_items': [{'type': 'text', 'text': ' World'}], 'usage_metadata': null, 'finish_reason': null}
// {'role': 'assistant', 'event_type': 'delta', 'content_items': [{'type': 'text', 'text': '!'}], 'usage_metadata': null, 'finish_reason': null}
// {'role': 'assistant', 'event_type': 'stop', 'content_items': [], 'usage_metadata': {'prompt_tokens': 12, 'thoughts_tokens': 0, 'response_tokens': 8, 'cached_tokens': 0}, 'finish_reason': 'stop'}
```

### Claude 4.5

<details><summary><strong>Python Example</strong></summary>

```python
import asyncio
import os
from agenthub import AutoLLMClient

async def main():
    os.environ["OPENAI_API_KEY"] = "your-api-key"
    client = AutoLLMClient(model="claude-sonnet-4-5-20250929")
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [{"type": "text", "text": "Say 'Hello, World!'"}]
        },
        config={}
    ):
        print(event)

asyncio.run(main())
```

</details>

<details><summary><strong>TypeScript Example</strong></summary>

```typescript
import { AutoLLMClient } from "@prismshadow/agenthub";

process.env.OPENAI_API_KEY = "your-api-key";

async function main() {
  const client = new AutoLLMClient({ model: "claude-sonnet-4-5-20250929" });
  const message = {
    role: "user",
    content_items: [{ type: "text", text: "Say 'Hello, World!'" }],
  }
  const config = {};
  for await (const event of client.streamingResponseStateful({ message, config })) {
    console.log(event);
  }
}

main().catch(console.error);
```

</details>

### OpenRouter GLM-4.7

<details><summary><strong>Python Example</strong></summary>

```python
import asyncio
import os
from agenthub import AutoLLMClient

async def main():
    os.environ["GLM_API_KEY"] = "your-openrouter-api-key"
    os.environ["GLM_BASE_URL"] = "https://openrouter.ai/api/v1"
    client = AutoLLMClient(model="glm-4.7")
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [{"type": "text", "text": "Say 'Hello, World!'"}]
        },
        config={}
    ):
        print(event)

asyncio.run(main())
```

</details>
<details><summary><strong>TypeScript Example</strong></summary>

```typescript
import { AutoLLMClient } from "@prismshadow/agenthub";

process.env.GLM_API_KEY = "your-openrouter-api-key";
process.env.GLM_BASE_URL = "https://openrouter.ai/api/v1";

async function main() {
  const client = new AutoLLMClient("glm-4.7");
  const message = {
    role: "user",
    content_items: [{ type: "text", text: "Say 'Hello, World!'" }],
  }
  const config = {};
  for await (const event of client.streamingResponseStateful(message, config)) {
    console.log(event);
  }
}

main().catch(console.error);
```
</details>

### SiliconFlow Qwen3-8B

<details><summary><strong>Python Example</strong></summary>

```python
import asyncio
import os
from agenthub import AutoLLMClient

async def main():
    os.environ["QWEN3_API_KEY"] = "your-siliconflow-api-key"
    os.environ["QWEN3_BASE_URL"] = "https://api.siliconflow.cn/v1"
    client = AutoLLMClient(model="Qwen/Qwen3-8B")
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [{"type": "text", "text": "Say 'Hello, World!'"}]
        },
        config={}
    ):
        print(event)

asyncio.run(main())
```

</details>
<details><summary><strong>TypeScript Example</strong></summary>

```typescript
import { AutoLLMClient } from "@prismshadow/agenthub";

process.env.QWEN3_API_KEY = "your-siliconflow-api-key";
process.env.QWEN3_BASE_URL = "https://api.siliconflow.cn/v1";

async function main() {
  const client = new AutoLLMClient("Qwen/Qwen3-8B");
  const message = {
    role: "user",
    content_items: [{ type: "text", text: "Say 'Hello, World!'" }],
  }
  const config = {};
  for await (const event of client.streamingResponseStateful(message, config)) {
    console.log(event);
  }
}

main().catch(console.error);
```
</details>

## Concepts: UniConfig, UniMessage and UniEvent

### UniConfig

UniConfig is an object that contains the configuration for the LLM model.

Example UniConfig:

```json
{
    "max_tokens": 1024,
    "temperature": 1.0,
    "tools": [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
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
    ],
    "thinking_summary": true,
    "thinking_level": "high",
    "tool_choice": "auto",
    "system_prompt": "You are a helpful assistant.",
    "prompt_caching": "enable",
    "trace_id": null
}
```

### UniMessage

UniMessage is an object that contains the input for the LLM model.

Example UniMessage:

```json
{
    "role": "user",
    "content_items": [
        {"type": "text", "text": "How are you doing?"}
    ]
}
```

### UniEvent

UniEvent is an object that contains streaming output of the LLM model.

Example UniEvent:

```json
{
    "role": "assistant",
    "event_type": "delta",
    "content_items": [
        {"type": "text", "text": "I am"}
    ],
    "usage_metadata": {
        "prompt_tokens": 10,
        "thought_tokens": null,
        "completion_tokens": 1,
        "cached_tokens": null
    },
    "finish_reason": null
}
```

## Tracing LLM Executions

![Tracer Screenshot](.github/images/tracer.png)

We provide a tracing feature to help you monitor and debug your LLM executions. You can enable tracing by setting the `trace_id` parameter to a unique identifier in the `config` object.

```python
async for event in client.streaming_response_stateful(
    message={
        "role": "user",
        "content_items": [{"type": "text", "text": "Say 'Hello, World!'"}]
    },
    config={"trace_id": "unique-trace-id"}
):
    print(event)
```

```python
from agenthub.integration.tracer import Tracer

tracer = Tracer()
tracer.start_web_server(host="127.0.0.1", port=5000, debug=False)
```

Then you can view the tracing output in the dashboard at `http://localhost:5000/`.

## LLM Playground

![Playground Screenshot](.github/images/playground.png)

We provide a LLM playground to help you test your LLM models.

```python
from agenthub.integration.playground import start_playground_server

start_playground_server(host="127.0.0.1", port=5001, debug=False)
```

You can access the playground at `http://localhost:5001/`.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
