# Multi-round Chat

DeepSeek's `/chat/completions` API is stateless. The server does not store conversation history, so callers must send the needed prior messages on every request.

## Conversation Pattern

1. Send the first user message.
2. Append the assistant response to the local `messages` list.
3. Append the next user message.
4. Send the full `messages` list again.

```python
from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

# Round 1
messages = [{"role": "user", "content": "What's the highest mountain in the world?"}]
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)

messages.append(response.choices[0].message)
print(f"Messages Round 1: {messages}")

# Round 2
messages.append({"role": "user", "content": "What is the second?"})
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
)

messages.append(response.choices[0].message)
print(f"Messages Round 2: {messages}")
```

## Request Shape

The first request only needs the first user message:

```json
[
  {
    "role": "user",
    "content": "What's the highest mountain in the world?"
  }
]
```

The next request includes the original user message, the assistant answer, and the new user message:

```json
[
  {
    "role": "user",
    "content": "What's the highest mountain in the world?"
  },
  {
    "role": "assistant",
    "content": "The highest mountain in the world is Mount Everest."
  },
  {
    "role": "user",
    "content": "What is the second?"
  }
]
```

For thinking-mode conversations, see [Thinking Mode](./thinking-mode.md) for when `reasoning_content` should be preserved.
