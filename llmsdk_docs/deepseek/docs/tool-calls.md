# Tool Calls

Tool Calls allows the model to call external tools to enhance its capabilities.

---

## Non-thinking Mode

### Sample Code

Here is an example of using Tool Calls to get the current weather information of the user's location, demonstrated with complete Python code.

For the specific API format of Tool Calls, please refer to the Chat Completion documentation.

```python
from openai import OpenAI

def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        tools=tools
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
                "required": ["location"]
            },
        }
    },
]
messages = [{"role": "user", "content": "How's the weather in Hangzhou, Zhejiang?"}]
message = send_messages(messages)
print(f"User>\t {messages[0]['content']}")

tool = message.tool_calls[0]
messages.append(message)

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})
message = send_messages(messages)
print(f"Model>\t {message.content}")
```

The execution flow of this example is as follows:

1. User: Asks about the current weather in Hangzhou
2. Model: Returns the function `get_weather({location: 'Hangzhou'})`
3. User: Calls the function `get_weather({location: 'Hangzhou'})` and provides the result to the model
4. Model: Returns in natural language, "The current temperature in Hangzhou is 24°C."

Note: In the above code, the functionality of the `get_weather` function needs to be provided by the user. The model itself does not execute specific functions.

---

## Thinking Mode

From DeepSeek-V3.2, the API supports tool use in the thinking mode. For more details, please refer to [Thinking Mode](./thinking-mode.md).

---

## `strict` Mode (Beta)

In `strict` mode, the model strictly adheres to the format requirements of the Function's JSON schema when outputting a tool call, ensuring that the model's output complies with the user's definition. It is supported by both thinking and non-thinking mode.

To use `strict` mode, you need to::

1. Use `base_url="https://api.deepseek.com/beta"` to enable Beta features
2. In the `tools` parameter, all `function` need to set the `strict` property to `true`
3. The server will validate the JSON Schema of the Function provided by the user. If the schema does not conform to the specifications or contains JSON schema types that are not supported by the server, an error message will be returned

The following is an example of a tool definition in the `strict` mode:

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

---

### Support Json Schema Types In `strict` Mode

- object
- string
- number
- integer
- boolean
- array
- enum
- anyOf

---

#### object

The `object` defines a nested structure containing key-value pairs, where `properties` specifies the schema for each key (or property) within the object. All properties of every `object` must be set as `required`, and the `additionalProperties` attribute of the `object` must be set to `false`.

Example:

```json
{
    "type": "object",
    "properties": {
        "name": { "type": "string" },
        "age": { "type": "integer" }
    },
    "required": ["name", "age"],
    "additionalProperties": false
}
```

---

#### string

Supported parameters:

- `pattern`: Uses regular expressions to constrain the format of the string
- `format`: Validates the string against predefined common formats. Currently supported formats:
  - `email`: Email address
  - `hostname`: Hostname
  - `ipv4`: IPv4 address
  - `ipv6`: IPv6 address
  - `uuid`: UUID

Unsupported parameters:

- `minLength`
- `maxLength`

Example:

```json
{
    "type": "object",
    "properties": {
        "user_email": {
            "type": "string",
            "description": "The user's email address",
            "format": "email"
        },
        "zip_code": {
            "type": "string",
            "description": "Six digit postal code",
            "pattern": "^\\d{6}$"
        }
    }
}
```

---

#### number/integer

Supported parameters:

- `const`: Specifies a constant numeric value
- `default`: Defines the default value of the number
- `minimum`: Specifies the minimum value
- `maximum`: Specifies the maximum value
- `exclusiveMinimum`: Defines a value that the number must be greater than
- `exclusiveMaximum`: Defines a value that the number must be less than
- `multipleOf`: Ensures that the number is a multiple of the specified value

Example:

```json
{
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "description": "A number from 1-5, which represents your rating, the higher, the better",
            "minimum": 1,
            "maximum": 5
        }
    },
    "required": ["score"],
    "additionalProperties": false
}
```

---

#### array

Unsupported parameters:

- minItems
- maxItems

Example:

```json
{
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "description": "Five keywords of the article, sorted by importance",
            "items": {
                "type": "string",
                "description": "A concise and accurate keyword or phrase."
            }
        }
    },
    "required": ["keywords"],
    "additionalProperties": false
}
```

---

#### enum

The `enum` ensures that the output is one of the predefined options. For example, in the case of order status, it can only be one of a limited set of specified states.

Example:

```json
{
    "type": "object",
    "properties": {
        "order_status": {
            "type": "string",
            "description": "Ordering status",
            "enum": ["pending", "processing", "shipped", "cancelled"]
        }
    }
}
```

---

#### anyOf

Matches any one of the provided schemas, allowing fields to accommodate multiple valid formats. For example, a user's account could be either an email address or a phone number:

```json
{
    "type": "object",
    "properties": {
        "account": {
            "anyOf": [
                { "type": "string", "format": "email", "description": "可以是电子邮件地址" },
                { "type": "string", "pattern": "^\\d{11}$", "description": "或11位手机号码" }
            ]
        }
    }
}
```

---

#### $ref and $def

You can use `$def` to define reusable modules and then use `$ref` to reference them, reducing schema repetition and enabling modularization. Additionally, `$ref` can be used independently to define recursive structures.

```json
{
    "type": "object",
    "properties": {
        "report_date": {
            "type": "string",
            "description": "The date when the report was published"
        },
        "authors": {
            "type": "array",
            "description": "The authors of the report",
            "items": {
                "$ref": "#/$def/author"
            }
        }
    },
    "required": ["report_date", "authors"],
    "additionalProperties": false,
    "$def": {
        "author": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "author's name"
                },
                "institution": {
                    "type": "string",
                    "description": "author's institution"
                },
                "email": {
                    "type": "string",
                    "format": "email",
                    "description": "author's email"
                }
            },
            "additionalProperties": false,
            "required": ["name", "institution", "email"]
        }
    }
}
```
