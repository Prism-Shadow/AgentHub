# Tool Calls

Tool calls let DeepSeek request external function execution. The model returns the function name and arguments; the application executes the function and sends the result back as a `tool` message.

## Non-thinking Mode

```python
from openai import OpenAI


def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        tools=tools,
    )
    return response.choices[0].message


client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply a location first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
    },
]

messages = [{"role": "user", "content": "How's the weather in Hangzhou, Zhejiang?"}]
message = send_messages(messages)
print(f"User>\t {messages[0]['content']}")

tool = message.tool_calls[0]
messages.append(message)
messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24 C"})

message = send_messages(messages)
print(f"Model>\t {message.content}")
```

Execution flow:

1. The user asks about the weather.
2. The model returns a `get_weather` tool call with arguments.
3. The application executes `get_weather`.
4. The application sends the result back with the matching `tool_call_id`.
5. The model returns the final natural-language answer.

The model does not execute tools itself. Tool execution is application-owned.

## Thinking Mode

DeepSeek supports tool calls in thinking mode. For tool-call turns, keep the full assistant message in the conversation history, including `reasoning_content`, `content`, and `tool_calls`. Dropping `reasoning_content` for those turns can cause a `400` response.

See [Thinking Mode](./thinking-mode.md) for the full thinking-mode context rule.

## Strict Mode

`strict` mode is a beta feature that makes the model follow the provided JSON Schema more closely when producing tool-call arguments. It works in both thinking and non-thinking mode.

To use it:

1. Use `base_url="https://api.deepseek.com/beta"`.
2. Set `strict` to `true` on every `function` in `tools`.
3. Ensure the supplied JSON Schema uses supported forms.

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "strict": true,
    "description": "Get weather of a location, the user should supply a location first.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "The city and state, e.g. San Francisco, CA"
        }
      },
      "required": ["location"],
      "additionalProperties": false
    }
  }
}
```

Supported JSON Schema types in strict mode:

- `object`
- `string`
- `number`
- `integer`
- `boolean`
- `array`
- `enum`
- `anyOf`

Object schemas must mark every property as required and set `additionalProperties` to `false`.

Supported string constraints include `pattern` and `format`. Supported formats include `email`, `hostname`, `ipv4`, `ipv6`, and `uuid`. `minLength` and `maxLength` are not supported.

Number and integer schemas support constraints such as `const`, `default`, `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, and `multipleOf`.

Array schemas do not support `minItems` or `maxItems`.

`$def` and `$ref` can be used for reusable or recursive schema definitions.
