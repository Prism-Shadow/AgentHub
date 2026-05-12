# Thinking Mode

Thinking mode lets DeepSeek generate reasoning content before the final answer. In OpenAI-compatible responses, the reasoning is returned as `reasoning_content` alongside the final `content`.

## Toggle and Effort

| Purpose | OpenAI-compatible parameter | Anthropic-compatible parameter |
| --- | --- | --- |
| Enable or disable thinking | `{"thinking": {"type": "enabled"}}` or `{"thinking": {"type": "disabled"}}` | Same thinking object |
| Control effort | `{"reasoning_effort": "high"}` or `{"reasoning_effort": "max"}` | `{"output_config": {"effort": "high"}}` or `{"output_config": {"effort": "max"}}` |

Notes:

- Thinking mode defaults to enabled.
- In thinking mode, regular requests default to `high` effort.
- Some complex agent requests may automatically use `max`.
- `low` and `medium` map to `high`; `xhigh` maps to `max`.
- Thinking mode ignores `temperature`, `top_p`, `presence_penalty`, and `frequency_penalty`.

With the OpenAI Python SDK, pass the DeepSeek-specific `thinking` setting through `extra_body`:

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)
```

## Non-streaming Response Handling

```python
from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
```

If the assistant turn did not perform a tool call, prior `reasoning_content` does not need to be included in later context. If it is included anyway, the API ignores it.

## Streaming Response Handling

```python
from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    stream=True,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

reasoning_content = ""
content = ""

for chunk in response:
    delta = chunk.choices[0].delta
    if getattr(delta, "reasoning_content", None):
        reasoning_content += delta.reasoning_content
    elif getattr(delta, "content", None):
        content += delta.content
```

## Multi-turn Context

For normal thinking-mode turns without tool calls, append the assistant message and the next user message as usual:

```python
messages.append(
    {
        "role": "assistant",
        "reasoning_content": reasoning_content,
        "content": content,
    }
)
messages.append({"role": "user", "content": "How many Rs are there in strawberry?"})
```

The API ignores the old `reasoning_content` in this no-tool-call case.

## Tool Calls in Thinking Mode

Thinking mode supports tool calls. When a thinking-mode assistant turn performs a tool call, preserve and pass back that turn's full assistant message, including `reasoning_content`, in all subsequent requests. If `reasoning_content` is dropped for a tool-call turn, the API may return a `400` error.

```python
messages.append(response.choices[0].message)

for tool_call in response.choices[0].message.tool_calls:
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": run_tool(tool_call),
        }
    )
```

Appending `response.choices[0].message` preserves the assistant `content`, `reasoning_content`, and `tool_calls` fields together.
