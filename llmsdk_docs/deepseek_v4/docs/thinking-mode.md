# Thinking Mode

> Source: https://api-docs.deepseek.com/guides/thinking_mode (snapshot 2026-08-19)

The DeepSeek model supports the thinking mode: before outputting the final answer, the model will first output a chain-of-thought reasoning to improve the accuracy of the final response.

## Thinking Mode Toggle and Effort Control

|  | Control Parameter (OpenAI Format) | Control Parameter (Anthropic Format) |
| --- | --- | --- |
| Thinking Mode Toggle (1) | `{"thinking": {"type": "enabled/disabled"}}` | `{"thinking": {"type": "enabled/disabled"}}` |
| Thinking Effort Control (2)(3) | `{"reasoning_effort": "low/high/max"}` | `{"output_config": {"effort": "low/high/max"}}` |

(1) The thinking toggle defaults to `enabled`.

(2) In thinking mode, the default effort is `high` for regular requests; for some complex agent requests (such as Claude Code, OpenCode), effort is automatically set to `max`.

(3) In thinking mode, a requested effort maps to an actual effort as follows: `low` to `low`,
`medium` to `high`, `high` to `high`, `xhigh` to `high`, and `max` to `max`. The same mapping
applies to `deepseek-v4-flash` and `deepseek-v4-pro`.

When using the OpenAI SDK, you need to pass the `thinking` parameter within `extra_body`:

```python
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    # ...
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)
```

## Input and Output Parameters

Thinking mode does not support the `temperature`, `top_p`, `presence_penalty`, or `frequency_penalty` parameters. Please note that, for compatibility with existing software, setting these parameters will not trigger an error but will also have no effect.

In thinking mode, the chain-of-thought content is returned via the `reasoning_content` parameter, at the same level as `content`. When concatenating subsequent turns, you can selectively return `reasoning_content` to the API:

- Between two `user` messages, if the model did not perform a tool call, the intermediate `assistant`'s `reasoning_content` does not need to participate in the context concatenation. If passed to the API in subsequent turns, it will be ignored. See [Multi-round Conversation](./multi-round-chat.md) for details.
- Between two `user` messages, if the model performed a tool call, the intermediate `assistant`'s `reasoning_content` must participate in the context concatenation and must be passed back to the API in all subsequent user interaction turns. See [Tool Calls](./tool-calls.md) for details.

## Multi-turn Conversation

In each turn of the conversation, the model outputs the CoT (`reasoning_content`) and the final answer (`content`). If there is no tool call, the CoT content from previous turns will not be concatenated into the context in the next turn, as illustrated in the following diagram:

### Sample Code

The following code, using Python as an example, demonstrates how to access the CoT and the final answer, as well as how to concatenate context in multi-turn conversations.

#### NoStreaming

```python
from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

# Turn 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

# Turn 2
# The reasoning_content will be ignored by the API
messages.append(response.choices[0].message)
messages.append({"role": "user", "content": "How many Rs are there in the word 'strawberry'?"})
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)
# ...
```

#### Streaming

```python
from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

# Turn 1
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
    if chunk.choices[0].delta.reasoning_content:
        reasoning_content += chunk.choices[0].delta.reasoning_content
    else:
        content += chunk.choices[0].delta.content

# Turn 2
# The reasoning_content will be ignored by the API
messages.append({"role": "assistant", "reasoning_content": reasoning_content, "content": content})
messages.append({"role": "user", "content": "How many Rs are there in the word 'strawberry'?"})
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    stream=True,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)
# ...
```

## Tool Calls

The DeepSeek model's thinking mode supports tool calls. Before outputting the final answer, the model can perform multiple turns of reasoning and tool calls to improve the quality of the response. The calling pattern is illustrated below:

Please note that, unlike turns in thinking mode that do not involve tool calls, for turns that do perform tool calls, the `reasoning_content` must be fully passed back to the API in all subsequent requests.

If your code does not correctly pass back `reasoning_content`, the API will return a 400 error. Please refer to the sample code below for the correct approach.

### Sample Code

Below is a simple sample code for tool calls in thinking mode:

```python
import json
import os
from datetime import datetime

from openai import OpenAI

# The definition of the tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply the location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city name"},
                    "date": {"type": "string", "description": "The date in format YYYY-mm-dd"},
                },
                "required": ["location", "date"],
            },
        },
    },
]


# The mocked version of the tool calls
def get_date_mock():
    return datetime.now().strftime("%Y-%m-%d")


def get_weather_mock(location, date):
    return "Cloudy 7~13°C"


TOOL_CALL_MAP = {
    "get_date": get_date_mock,
    "get_weather": get_weather_mock,
}


def run_turn(turn, messages):
    sub_turn = 1
    while True:
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            tools=tools,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
        messages.append(response.choices[0].message)
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls
        print(f"Turn {turn}.{sub_turn}\n{reasoning_content=}\n{content=}\n{tool_calls=}")
        # If there is no tool calls, then the model should get a final answer and we need to stop the loop
        if tool_calls is None:
            break
        for tool in tool_calls:
            tool_function = TOOL_CALL_MAP[tool.function.name]
            tool_result = tool_function(**json.loads(tool.function.arguments))
            print(f"tool result for {tool.function.name}: {tool_result}\n")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool.id,
                    "content": tool_result,
                }
            )
        sub_turn += 1
    print()


client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=os.environ.get("DEEPSEEK_BASE_URL"),
)

# The user starts a question
turn = 1
messages = [
    {
        "role": "user",
        "content": "How's the weather in Hangzhou Tomorrow",
    }
]
run_turn(turn, messages)

# The user starts a new question
turn = 2
messages.append(
    {
        "role": "user",
        "content": "How's the weather in Guangzhou Tomorrow",
    }
)
run_turn(turn, messages)
```

In each sub-request of Turn 1, the `reasoning_content` generated during that turn is sent to the API, allowing the model to continue its previous reasoning. `response.choices[0].message` contains all necessary fields for the `assistant` message, including `content`, `reasoning_content`, and `tool_calls`. For simplicity, you can directly append the message to the end of the messages list using the following code:

```python
messages.append(response.choices[0].message)
```

This line of code is equivalent to:

```python
messages.append(
    {
        "role": "assistant",
        "content": response.choices[0].message.content,
        "reasoning_content": response.choices[0].message.reasoning_content,
        "tool_calls": response.choices[0].message.tool_calls,
    }
)
```

Additionally, in the Turn 2 request, we still pass the `reasoning_content` generated in Turn 1 to the API.

The sample output of this code is as follows:

```text
Turn 1.1
reasoning_content="The user is asking about the weather in Hangzhou tomorrow. I need to get tomorrow's date first, then call the weather function."
content="Let me check tomorrow's weather in Hangzhou for you. First, let me get tomorrow's date."
tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_00_kw66qNnNto11bSfJVIdlV5Oo', function=Function(arguments='{}', name='get_date'), type='function', index=0)]
tool result for get_date: 2026-04-19

Turn 1.2
reasoning_content="Today is 2026-04-19, so tomorrow is 2026-04-20. Now I'll call the weather function for Hangzhou."
content=''
tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_00_H2SCW6136vWJGq9SQlBuhVt4', function=Function(arguments='{"location": "Hangzhou", "date": "2026-04-20"}', name='get_weather'), type='function', index=0)]
tool result for get_weather: Cloudy 7~13°C

Turn 1.3
reasoning_content='The weather result is in. Let me share this with the user.'
content="Here's the weather forecast for **Hangzhou tomorrow (April 20, 2026)**:\n\n-  **Condition:** Cloudy  \n-  **Temperature:** 7°C ~ 13°C (45°F ~ 55°F)\n\nIt'll be on the cooler side, so you might want to bring a light jacket if you're heading out! Let me know if you need anything else."
tool_calls=None

Turn 2.1
reasoning_content='The user is asking about the weather in Guangzhou tomorrow. Today is 2026-04-19, so tomorrow is 2026-04-20. I can directly call the weather function.'
content=''
tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_00_8URkLt5NjmNkVKhDmMcNq9Mo', function=Function(arguments='{"location": "Guangzhou", "date": "2026-04-20"}', name='get_weather'), type='function', index=0)]
tool result for get_weather: Cloudy 7~13°C

Turn 2.2
reasoning_content='The weather result for Guangzhou is the same as Hangzhou. Let me share this with the user.'
content="Here's the weather forecast for **Guangzhou tomorrow (April 20, 2026)**:\n\n-  **Condition:** Cloudy  \n-  **Temperature:** 7°C ~ 13°C (45°F ~ 55°F)\n\nIt'll be cool and cloudy, so a light jacket would be a good idea if you're going out. Let me know if there's anything else you'd like to know!"
tool_calls=None
```
