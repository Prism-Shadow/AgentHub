# OpenAI Python SDK

Z.AI provides interfaces compatible with OpenAI API, which means you can use existing OpenAI SDK code and seamlessly switch to Z.AI's model services by simply modifying the API key and base URL. This compatibility allows you to:

* Quickly migrate existing OpenAI applications
* Use familiar development patterns and tools
* Enjoy the powerful capabilities of Z.AI models
* Maintain code consistency and maintainability

<Warning>
  In some scenarios, there are still differences between Z.AI and OpenAI interfaces, but this does not affect overall compatibility.
</Warning>

### Core Advantages

<CardGroup cols={2}>
  <Card title="Zero Learning Cost" icon="graduation-cap">
    If you are already familiar with OpenAI SDK, you can start using it immediately
  </Card>

  <Card title="Quick Migration" icon="arrows-rotate">
    Existing OpenAI applications can be quickly migrated to Z.AI platform
  </Card>

  <Card title="Ecosystem Compatibility" icon="puzzle-piece">
    Compatible with various tools and frameworks in the OpenAI ecosystem
  </Card>

  <Card title="Continuous Updates" icon="arrow-up">
    Follow OpenAI SDK updates to maintain latest feature support
  </Card>
</CardGroup>

## Environment Requirements

<CardGroup cols={2}>
  <Card title="Python Version" icon="python">
    Python 3.7.1 or higher
  </Card>

  <Card title="OpenAI SDK" icon="package">
    OpenAI SDK version 1.0.0 or higher
  </Card>
</CardGroup>

<Warning>
  Please ensure using OpenAI SDK 1.0.0 or higher, older versions may have compatibility issues.
</Warning>

## Install OpenAI Python SDK

### Install using pip

```bash  theme={null}
# Install or upgrade to latest version
pip install --upgrade 'openai>=1.0'

# Verify installation
python -c "import openai; print(openai.__version__)"
```

### Install using poetry

```bash  theme={null}
poetry add "openai>=1.0"
```

## Quick Start

### Get API Key

1. Access [Z.AI Open Platform](https://z.ai/model-api), Register or Login.
2. Create an API Key in the [API Keys](https://z.ai/manage-apikey/apikey-list) management page.
3. Copy your API Key for use.

<Tip>
  It is recommended to set the API Key as an environment variable: `export ZAI_API_KEY=your-api-key`
</Tip>

### Create Client

<Tabs>
  <Tab title="Basic Configuration">
    ```python  theme={null}
    from openai import OpenAI

    # Create Z.AI client
    client = OpenAI(
        api_key="your-Z.AI-api-key",
        base_url="https://api.z.ai/api/paas/v4/"
    )
    ```
  </Tab>

  <Tab title="Environment Variables">
    ```python  theme={null}
    from openai import OpenAI
    import os

    # Use environment variables
    client = OpenAI(
        api_key=os.getenv("ZAI_API_KEY"),
        base_url="https://api.z.ai/api/paas/v4/"
    )
    ```
  </Tab>

  <Tab title="Configuration Class">
    ```python  theme={null}
    from openai import OpenAI
    from dataclasses import dataclass

    @dataclass
    class Z.AIConfig:
        api_key: str
        base_url: str = "https://api.z.ai/api/paas/v4/"
        timeout: int = 30
        max_retries: int = 3

    config = Z.AIConfig(api_key="your-api-key")
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries
    )
    ```
  </Tab>
</Tabs>

## Quick Start Examples

### Basic Chat

```python  theme={null}
from openai import OpenAI

client = OpenAI(
    api_key="your-Z.AI-api-key",
    base_url="https://api.z.ai/api/paas/v4/"
)

completion = client.chat.completions.create(
    model="glm-4.7",
    messages=[
        {"role": "system", "content": "You are a smart and creative novelist"},
        {"role": "user", "content": "Please write a short fairy tale story as a fairy tale master"}
    ]
)

print(completion.choices[0].message.content)
```

### Streaming Response

```python  theme={null}
from openai import OpenAI

client = OpenAI(
    api_key="your-Z.AI-api-key",
    base_url="https://api.z.ai/api/paas/v4/"
)

stream = client.chat.completions.create(
    model="glm-4.7",
    messages=[
        {"role": "user", "content": "Write a poem about artificial intelligence"}
    ],
    stream=True,
    temperature=1.0
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # New line
```

### Multi-turn Conversation

```python  theme={null}
from openai import OpenAI

class ChatBot:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.z.ai/api/paas/v4/"
        )
        self.conversation = [
            {"role": "system", "content": "You are a helpful AI assistant"}
        ]

    def chat(self, user_input: str) -> str:
        # Add user message
        self.conversation.append({"role": "user", "content": user_input})

        # Call API
        response = self.client.chat.completions.create(
            model="glm-4.7",
            messages=self.conversation,
            temperature=1.0
        )

        # Get AI response
        ai_response = response.choices[0].message.content

        # Add to conversation history
        self.conversation.append({"role": "assistant", "content": ai_response})

        return ai_response

    def clear_history(self):
        """Clear conversation history, keep system prompt"""
        self.conversation = self.conversation[:1]

# Usage example
bot = ChatBot("your-api-key")
print(bot.chat("Hello, please introduce yourself"))
print(bot.chat("Can you help me write code?"))
print(bot.chat("Write a Python quicksort algorithm"))
```

## Advanced Features

### Thinking Mode

In thinking mode, GLM-4.6, GLM-4.5 and GLM-4.5-Air can solve complex reasoning problems, including mathematics, science, and logic problems.

The param `thinking.type` can be either `enabled` or `disabled`.

```python  theme={null}
import os
from openai import OpenAI

client = OpenAI(api_key='your-api-key', base_url='https://api.z.ai/api/paas/v4/')
response = client.chat.completions.create(
        model='glm-4.7',
        messages=[
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "what is the revolution of llm?"}
        ],
        stream=True,
        extra_body={
            "thinking": {
                "type": "enabled",
            },
        }
    )
for chunk in response:
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end='')
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='')
```

### Function Calling

```python  theme={null}
import json
from openai import OpenAI

client = OpenAI(
    api_key="your-Z.AI-api-key",
    base_url="https://api.z.ai/api/paas/v4/"
)

def get_weather(location: str) -> str:
    """Get weather information for specified location"""
    # This should call a real weather API
    return f"Weather in {location}: Sunny, 25°C"

# Define function description
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for specified location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location name, e.g.: Beijing, Shanghai"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Call conversation with functions
response = client.chat.completions.create(
    model="glm-4.7",
    messages=[
        {"role": "user", "content": "How's the weather in Beijing today?"}
    ],
    tools=tools,
    tool_choice="auto"
)

# Handle function calls
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_weather":
            args = json.loads(tool_call.function.arguments)
            result = get_weather(args["location"])
            print(f"Function call result: {result}")
```

## Parameter Configuration

### Common Parameters

| Parameter   | Type         | Default  | Description                      |
| ----------- | ------------ | -------- | -------------------------------- |
| model       | string       | Required | Model name to use                |
| messages    | array        | Required | List of conversation messages    |
| temperature | float        | 0.6      | Controls output randomness (0-1) |
| top\_p      | float        | 0.95     | Nucleus sampling parameter (0-1) |
| max\_tokens | integer      | -        | Maximum output tokens            |
| stream      | boolean      | false    | Whether to use streaming output  |
| stop        | string/array | -        | Stop generation tokens           |

<Note>
  Note: The temperature parameter range is (0,1), do\_sample = False (temperature = 0) is not applicable in OpenAI calls.
</Note>

## Best Practices

<CardGroup cols={2}>
  <Card title="Performance Optimization" icon="gauge-high">
    * Use connection pooling and session reuse
    * Set reasonable timeout values
    * Implement async calls for high concurrency
    * Cache frequently used responses
  </Card>

  <Card title="Cost Control" icon="dollar-sign">
    * Set reasonable max\_tokens limits
    * Use appropriate models (don't overuse powerful models)
    * Implement request deduplication
    * Monitor API usage
  </Card>

  <Card title="Security" icon="shield">
    * Use environment variables to store API keys
    * Implement input validation and filtering
    * Log and monitor API calls
    * Rotate API keys regularly
  </Card>

  <Card title="Reliability" icon="shield-check">
    * Implement retry mechanisms and error handling
    * Set reasonable timeout values
    * Monitor API status and response times
    * Prepare fallback solutions
  </Card>
</CardGroup>

## Migration Guide

### Migrating from OpenAI

If you're already using OpenAI API, migrating to Z.AI is very simple:

```python  theme={null}
# Original OpenAI code
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",  # OpenAI API Key
    # base_url uses default value
)

# Migrate to Z.AI, only need to modify two places
client = OpenAI(
    api_key="your-Z.AI-api-key",  # Replace with Z.AI API Key
    base_url="https://api.z.ai/api/paas/v4/"  # Add Z.AI base_url
)

# Other code remains unchanged
response = client.chat.completions.create(
    model="glm-4.7",  # Use Z.AI model
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Getting Help

<CardGroup cols={2}>
  <Card title="API Documentation" icon="book" href="/api-reference">
    View complete API interface documentation
  </Card>

  <Card title="OpenAI Official Documentation" icon="link" href="https://platform.openai.com/docs">
    Refer to OpenAI official documentation for more usage
  </Card>
</CardGroup>

<Note>
  Z.AI is committed to maintaining compatibility with OpenAI API. If you encounter any issues during migration, please contact our technical support team.
</Note>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.z.ai/llms.txt
