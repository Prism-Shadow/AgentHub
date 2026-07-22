> ## Documentation Index
> Fetch the complete documentation index at: https://docs.z.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Migrate to GLM-5.2

<Tip>
  This guide explains how to migrate your calls from GLM-5.1, GLM-5, GLM-4.7, GLM-4.6, GLM-4.5 or other earlier models to Z.AI GLM-5.2, our strongest coding model to date, covering sampling parameter differences, streaming tool calls, reasoning\_effort parameter, and other key points.
</Tip>

## GLM-5.2 Features

* Support for larger context and output: Maximum context 1M, maximum output 128K.
* New support for streaming output during tool calling process (`tool_stream=true`), real-time retrieval of tool call parameters.
* Supports deep thinking (`thinking={ type: "enabled" }`): when enabled, the model automatically determines whether to think (unlike GLM-4.7, which uses forced thinking).
* New `reasoning_effort` parameter for controlling the model's reasoning effort level when chain-of-thought thinking is enabled.
* Superior code performance and advanced reasoning capabilities.

## Migration Checklist

* [ ] Update model identifier to `glm-5.2`
* [ ] Sampling parameters: `temperature` default value `1.0`, `top_p` default value `0.95`, recommend choosing only one for tuning
* [ ] Deep thinking: Enable or disable `thinking={ type: "enabled" }` as needed for complex reasoning/coding
* [ ] Control reasoning effort: Configure `reasoning_effort` to decide between `high` (enhanced reasoning) or `max` (deep reasoning, default)
* [ ] Streaming response: Enable `stream=true` and properly handle `delta.reasoning_content` and `delta.content`
* [ ] Streaming tool calls: Enable `stream=true` and `tool_stream=true` and stream-concatenate `delta.tool_calls[*].function.arguments`
* [ ] Maximum output and context: Set `max_tokens` appropriately (GLM-5.2 maximum output 128K, context 1M)
* [ ] Prompt optimization: Work with deep thinking, use clearer instructions and constraints
* [ ] Development environment verification: Conduct use case testing and regression, focus on randomness, latency, parameter completeness in tool streams

## Start Migration

### 1. Update Model Identifier

* Update `model` to `glm-5.2`.

```python theme={null}
resp = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "Briefly describe the advantages of GLM-5.2"}]
)
```

### 2. Update Sampling Parameters

* `temperature`: Controls randomness; higher values are more divergent, lower values are more stable.
* `top_p`: Controls nucleus sampling; higher values expand candidate set, lower values converge candidate set.
* `temperature` defaults to `1.0`, `top_p` defaults to `0.95`, not recommended to adjust both simultaneously.

```python theme={null}
# Plan A: Use temperature (recommended)
resp = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "Write a more creative brand introduction"}],
    temperature=1.0
)

# Plan B: Use top_p
resp = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "Generate more stable technical documentation"}],
    top_p=0.8
)
```

### 3. Deep Thinking (Optional)

* GLM-5.2 continues to support deep thinking capability, enabled by default.
* Recommended to enable for complex reasoning and coding tasks:

```python theme={null}
resp = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "Design a three-tier microservice architecture for me"}],
    thinking={"type": "enabled"}
)
```

* Control reasoning effort, default `max` for deep reasoning:

```python theme={null}
resp = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "Design a three-tier microservice architecture for me"}],
    thinking={"type": "enabled"},
    reasoning_effort="max"
)
```

### 4. Streaming Output and Tool Calls (Optional)

* GLM-5.2 supports real-time streaming construction and output during tool calling process, disabled by default (`False`), requires enabling both:
  * `stream=True`: Enable streaming output for responses
  * `tool_stream=True`: Enable streaming output for tool call parameters

```python theme={null}
response = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "How's the weather in Beijing"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather conditions for a specified location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City, eg: Beijing, Shanghai"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    stream=True,
    tool_stream=True,
)

# Initialize streaming collection variables
reasoning_content = ""
content = ""
final_tool_calls = {}
reasoning_started = False
content_started = False

# Process streaming response
for chunk in response:
    if not chunk.choices:
        continue

    delta = chunk.choices[0].delta

    # Streaming reasoning process output
    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
        if not reasoning_started and delta.reasoning_content.strip():
            print("\n🧠 Thinking Process:")
            reasoning_started = True
        reasoning_content += delta.reasoning_content
        print(delta.reasoning_content, end="", flush=True)

    # Streaming answer content output
    if hasattr(delta, 'content') and delta.content:
        if not content_started and delta.content.strip():
            print("\n\n💬 Answer Content:")
            content_started = True
        content += delta.content
        print(delta.content, end="", flush=True)

    # Streaming tool call information (parameter concatenation)
    if delta.tool_calls:
        for tool_call in delta.tool_calls:
            idx = tool_call.index
            if idx not in final_tool_calls:
                final_tool_calls[idx] = tool_call
                final_tool_calls[idx].function.arguments = tool_call.function.arguments
            else:
                final_tool_calls[idx].function.arguments += tool_call.function.arguments

# Output final tool call information
if final_tool_calls:
    print("\n📋 Function Calls Triggered:")
    for idx, tool_call in final_tool_calls.items():
        print(f"  {idx}: Function Name: {tool_call.function.name}, Parameters: {tool_call.function.arguments}")
```

See: [Tool Streaming Output Documentation](/guides/tools/stream-tool)

### 5. Testing and Regression

> First verify in development environment that post-migration calls are stable, focus on:

* Whether responses meet expectations, whether there's excessive randomness or excessive conservatism in output
* Whether tool streaming construction and output work normally
* Latency and cost in long context and deep thinking scenarios

## More Resources

<CardGroup cols={2}>
  <Card title="Concept Parameters" icon="star" href="/guides/overview/concept-param">
    Common model parameter concepts and sampling recommendations
  </Card>

  <Card title="Tool Streaming Output" icon="code" href="/guides/tools/stream-tool">
    View tool streaming output usage details
  </Card>

  <Card title="API Reference" icon="book" href="/api-reference/introduction">
    View complete API documentation
  </Card>

  <Card title="Technical Support" icon="headset" href="https://z.ai/consultation">
    Get technical support and help
  </Card>
</CardGroup>
