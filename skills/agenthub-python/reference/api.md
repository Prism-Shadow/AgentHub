# APIs

`AutoLLMClient` exposes five basic APIs. Prefer the stateful stream for agent loops. See [Basic Usage](../SKILL.md#basic-usage) for a full tool-use example.

## Initialization

Initialize `AutoLLMClient` in one of three common ways:

```python
# Initialize with model name
client = AutoLLMClient(model="gpt-5.5")

# Optionally specify API key (if not using environment variables)
client = AutoLLMClient(
    model="gpt-5.5",
    api_key="your-openai-api-key",
    base_url="https://api.openai.com/v1",
)

# Use OpenAI Chat Completions-compatible routing explicitly
client = AutoLLMClient(model="custom-model", client_type="openai")
```

## Method signatures

```python
async def streaming_response(messages: list[UniMessage], config: UniConfig) -> AsyncIterator[UniEvent]:
    """Stream one stateless response from a full message list."""

async def streaming_response_stateful(message: UniMessage, config: UniConfig) -> AsyncIterator[UniEvent]:
    """Stream one stateful response and update client history."""

def get_history() -> list[UniMessage]:
    """Return a copy of stateful history."""

def set_history(history: list[UniMessage]) -> None:
    """Replace stateful history with a copy."""

def clear_history() -> None:
    """Clear stateful history."""
```
