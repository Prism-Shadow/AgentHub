# Create Chat Completion

```
POST https://api.deepseek.com/chat/completions
```

Creates a model response for the given chat conversation.

---

## Request

- **Content-Type**: `application/json`

### Body

| Field | Type | Required | Nullable | Description |
|-------|------|----------|----------|-------------|
| `messages` | `object[]` | **Yes** | No | A list of messages comprising the conversation so far. Minimum 1 item. |
| `model` | `string` | **Yes** | No | ID of the model to use. Possible values: `deepseek-v4-flash`, `deepseek-v4-pro`. |
| `thinking` | `object` | No | Yes | Controls the switch between thinking and non-thinking mode. |
| `thinking.type` | `string` | No | No | Possible values: `enabled`, `disabled`. Default: `enabled`. If set to `enabled`, then use thinking mode. If set to `disabled`, then use non-thinking mode. |
| `thinking.reasoning_effort` | `string` | No | No | Possible values: `high`, `max`. Controls the reasoning effort of the model. The default effort is `high` for regular requests; for some complex agent requests (such as Claude Code, OpenCode), effort is automatically set to `max`. For compatibility, `low` and `medium` are mapped to `high`, and `xhigh` is mapped to `max`. |
| `max_tokens` | `integer` | No | Yes | The maximum number of tokens that can be generated in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length. |
| `response_format` | `object` | No | Yes | An object specifying the format that the model must output. Setting to `{ "type": "json_object" }` enables JSON Output, which guarantees the message the model generates is valid JSON. **Important:** When using JSON Output, you must also instruct the model to produce JSON yourself via a system or user message. Without this, the model may generate an unending stream of whitespace until the generation reaches the token limit, resulting in a long-running and seemingly "stuck" request. Also note that the message content may be partially cut off if `finish_reason="length"`, which indicates the generation exceeded `max_tokens` or the conversation exceeded the max context length. |
| `response_format.type` | `string` | No | No | Possible values: `text`, `json_object`. Default: `text`. |
| `stop` | `string` or `string[]` | No | Yes | Up to 16 sequences where the API will stop generating further tokens. |
| `stream` | `boolean` | No | Yes | If set, partial message deltas will be sent. Tokens will be sent as data-only server-sent events (SSE) as they become available, with the stream terminated by a `data: [DONE]` message. |
| `stream_options` | `object` | No | Yes | Options for streaming response. Only set this when you set `stream: true`. |
| `stream_options.include_usage` | `boolean` | No | No | If set, an additional chunk will be streamed before the `data: [DONE]` message. The `usage` field on this chunk shows the token usage statistics for the entire request, and the `choices` field will always be an empty array. All other chunks will also include a `usage` field, but with a null value. |
| `temperature` | `number` | No | Yes | What sampling temperature to use, between 0 and 2. Default: `1`. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or `top_p` but not both. |
| `top_p` | `number` | No | Yes | An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. Default: `1`. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or `temperature` but not both. |
| `tools` | `object[]` | No | Yes | A list of tools the model may call. Currently, only functions are supported as a tool. A max of 128 functions are supported. |
| `tools[].type` | `string` | **Yes** | No | The type of the tool. Currently, only `function` is supported. |
| `tools[].function` | `object` | **Yes** | No | The function definition. |
| `tools[].function.description` | `string` | No | No | A description of what the function does, used by the model to choose when and how to call the function. |
| `tools[].function.name` | `string` | **Yes** | No | The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64. |
| `tools[].function.parameters` | `object` | No | No | The parameters the functions accepts, described as a JSON Schema object. Omitting `parameters` defines a function with an empty parameter list. |
| `tools[].function.strict` | `boolean` | No | No | Default: `false`. If set to true, the API will use strict-mode for the tool calls to ensure the output always complies with the function's JSON schema. This is a Beta feature. |
| `tool_choice` | `string` or `object` | No | Yes | Controls which (if any) tool is called by the model. `none` means the model will not call any tool and instead generates a message. `auto` means the model can pick between generating a message or calling one or more tools. `required` means the model must call one or more tools. Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool. `none` is the default when no tools are present. `auto` is the default if tools are present. |
| `tool_choice.type` | `string` | **Yes** | No | Possible values: `function`. The type of the tool. |
| `tool_choice.function` | `object` | **Yes** | No | The function specification. |
| `tool_choice.function.name` | `string` | **Yes** | No | The name of the function to call. |
| `logprobs` | `boolean` | No | Yes | Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the `content` of `message`. |
| `top_logprobs` | `integer` | No | Yes | An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. `logprobs` must be set to `true` if this parameter is used. |
| `user_id` | `string` | No | Yes | A custom user_id. Allowed character set is [a-zA-Z0-9\-_], with a maximum length of 512. Do not include user privacy information in the user_id. user_id can be used to distinguish user identities on your side to help us with content safety review. user_id can be used for KVCache isolation for privacy management. |
| `frequency_penalty` | - | No | No | **Deprecated.** This parameter is no longer supported. It will not take effect if you pass it to the API. |
| `presence_penalty` | - | No | No | **Deprecated.** This parameter is no longer supported. It will not take effect if you pass it to the API. |

### `messages` Item Types

Each item in the `messages` array is one of the following message types:

#### System Message

| Field | Type | Required | Nullable | Description |
|-------|------|----------|----------|-------------|
| `content` | `string` | **Yes** | No | The contents of the system message. |
| `role` | `string` | **Yes** | No | The role of the messages author, in this case `system`. |
| `name` | `string` | No | No | An optional name for the participant. Provides the model information to differentiate between participants of the same role. |

#### User Message

| Field | Type | Required | Nullable | Description |
|-------|------|----------|----------|-------------|
| `content` | `string` | **Yes** | No | The contents of the user message. |
| `role` | `string` | **Yes** | No | The role of the messages author, in this case `user`. |
| `name` | `string` | No | No | An optional name for the participant. Provides the model information to differentiate between participants of the same role. |

#### Assistant Message

| Field | Type | Required | Nullable | Description |
|-------|------|----------|----------|-------------|
| `content` | `string` or `null` | **Yes** | Yes | The contents of the assistant message. |
| `role` | `string` | **Yes** | No | The role of the messages author, in this case `assistant`. |
| `name` | `string` | No | No | An optional name for the participant. Provides the model information to differentiate between participants of the same role. |
| `prefix` | `boolean` | No | No | (Beta) Set this to `true` to force the model to start its answer by the content of the supplied prefix in this `assistant` message. You must set `base_url="https://api.deepseek.com/beta"` to use this feature. |
| `reasoning_content` | `string` or `null` | No | Yes | (Beta) Used for the thinking mode in the Chat Prefix Completion feature as the input for the CoT in the last assistant message. When using this feature, the `prefix` parameter must be set to `true`. |

#### Tool Message

| Field | Type | Required | Nullable | Description |
|-------|------|----------|----------|-------------|
| `content` | `string` | **Yes** | No | The contents of the tool message. |
| `role` | `string` | **Yes** | No | The role of the messages author, in this case `tool`. |
| `tool_call_id` | `string` | **Yes** | No | Tool call that this message is responding to. |

### Example Request Body

```json
{
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "Hi",
      "role": "user"
    }
  ],
  "model": "deepseek-v4-pro",
  "thinking": {
    "type": "enabled"
  },
  "reasoning_effort": "high",
  "max_tokens": 4096,
  "response_format": {
    "type": "text"
  },
  "stop": null,
  "stream": false,
  "stream_options": null,
  "temperature": 1,
  "top_p": 1,
  "tools": null,
  "tool_choice": "none",
  "logprobs": false,
  "top_logprobs": null
}
```

---

## Responses

### 200 — No Streaming

OK, returns a `chat completion object`.

- **Content-Type**: `application/json`

#### Schema

| Field | Type | Required | Nullable | Description |
|-------|------|----------|----------|-------------|
| `id` | `string` | **Yes** | No | A unique identifier for the chat completion. |
| `choices` | `object[]` | **Yes** | No | A list of chat completion choices. |
| `choices[].finish_reason` | `string` | **Yes** | No | The reason the model stopped generating tokens. Possible values: `stop`, `length`, `content_filter`, `tool_calls`, `insufficient_system_resource`. This will be `stop` if the model hit a natural stop point or a provided stop sequence, `length` if the maximum number of tokens specified in the request was reached, `content_filter` if content was omitted due to a flag from our content filters, `tool_calls` if the model called a tool, or `insufficient_system_resource` if the request is interrupted due to insufficient resource of the inference system. |
| `choices[].index` | `integer` | **Yes** | No | The index of the choice in the list of choices. |
| `choices[].message` | `object` | **Yes** | No | A chat completion message generated by the model. |
| `choices[].message.content` | `string` or `null` | **Yes** | Yes | The contents of the message. |
| `choices[].message.reasoning_content` | `string` or `null` | No | Yes | For thinking mode only. The reasoning contents of the assistant message, before the final answer. |
| `choices[].message.tool_calls` | `object[]` | No | No | The tool calls generated by the model. |
| `choices[].message.tool_calls[].id` | `string` | **Yes** | No | The ID of the tool call. |
| `choices[].message.tool_calls[].type` | `string` | **Yes** | No | The type of the tool. Currently, only `function` is supported. |
| `choices[].message.tool_calls[].function` | `object` | **Yes** | No | The function that the model called. |
| `choices[].message.tool_calls[].function.name` | `string` | **Yes** | No | The name of the function to call. |
| `choices[].message.tool_calls[].function.arguments` | `string` | **Yes** | No | The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function. |
| `choices[].message.role` | `string` | **Yes** | No | The role of the author of this message. Value: `assistant`. |
| `choices[].logprobs` | `object` or `null` | **Yes** | Yes | Log probability information for the choice. |
| `choices[].logprobs.content` | `object[]` or `null` | **Yes** | Yes | A list of message content tokens with log probability information. |
| `choices[].logprobs.content[].token` | `string` | **Yes** | No | The token. |
| `choices[].logprobs.content[].logprob` | `number` | **Yes** | No | The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value `-9999.0` is used to signify that the token is very unlikely. |
| `choices[].logprobs.content[].bytes` | `integer[]` or `null` | **Yes** | Yes | A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token. |
| `choices[].logprobs.content[].top_logprobs` | `object[]` | **Yes** | No | List of the most likely tokens and their log probability, at this token position. In rare cases, there may be fewer than the number of requested `top_logprobs` returned. |
| `choices[].logprobs.content[].top_logprobs[].token` | `string` | **Yes** | No | The token. |
| `choices[].logprobs.content[].top_logprobs[].logprob` | `number` | **Yes** | No | The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value `-9999.0` is used to signify that the token is very unlikely. |
| `choices[].logprobs.content[].top_logprobs[].bytes` | `integer[]` or `null` | **Yes** | Yes | A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token. |
| `choices[].logprobs.reasoning_content` | `object[]` or `null` | No | Yes | A list of reasoning content tokens with log probability information. |
| `choices[].logprobs.reasoning_content[].token` | `string` | **Yes** | No | The token. |
| `choices[].logprobs.reasoning_content[].logprob` | `number` | **Yes** | No | The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value `-9999.0` is used to signify that the token is very unlikely. |
| `choices[].logprobs.reasoning_content[].bytes` | `integer[]` or `null` | **Yes** | Yes | A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token. |
| `choices[].logprobs.reasoning_content[].top_logprobs` | `object[]` | **Yes** | No | List of the most likely tokens and their log probability, at this token position. In rare cases, there may be fewer than the number of requested `top_logprobs` returned. |
| `choices[].logprobs.reasoning_content[].top_logprobs[].token` | `string` | **Yes** | No | The token. |
| `choices[].logprobs.reasoning_content[].top_logprobs[].logprob` | `number` | **Yes** | No | The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value `-9999.0` is used to signify that the token is very unlikely. |
| `choices[].logprobs.reasoning_content[].top_logprobs[].bytes` | `integer[]` or `null` | **Yes** | Yes | A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token. |
| `created` | `integer` | **Yes** | No | The Unix timestamp (in seconds) of when the chat completion was created. |
| `model` | `string` | **Yes** | No | The model used for the chat completion. |
| `system_fingerprint` | `string` | **Yes** | No | This fingerprint represents the backend configuration that the model runs with. |
| `object` | `string` | **Yes** | No | The object type, which is always `chat.completion`. |
| `usage` | `object` | No | No | Usage statistics for the completion request. |
| `usage.completion_tokens` | `integer` | **Yes** | No | Number of tokens in the generated completion. |
| `usage.prompt_tokens` | `integer` | **Yes** | No | Number of tokens in the prompt. It equals `prompt_cache_hit_tokens` + `prompt_cache_miss_tokens`. |
| `usage.prompt_cache_hit_tokens` | `integer` | **Yes** | No | Number of tokens in the prompt that hits the context cache. |
| `usage.prompt_cache_miss_tokens` | `integer` | **Yes** | No | Number of tokens in the prompt that misses the context cache. |
| `usage.total_tokens` | `integer` | **Yes** | No | Total number of tokens used in the request (prompt + completion). |
| `usage.completion_tokens_details` | `object` | No | No | Breakdown of tokens used in a completion. |
| `usage.completion_tokens_details.reasoning_tokens` | `integer` | No | No | Tokens generated by the model for reasoning. |

#### Example Response

```json
{
  "id": "930c60df-bf64-41c9-a88e-3ec75f81e00e",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Hello! How can I help you today?",
        "role": "assistant"
      }
    }
  ],
  "created": 1705651092,
  "model": "deepseek-v4-pro",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 10,
    "prompt_tokens": 16,
    "total_tokens": 26
  }
}
```

---

### 200 — Streaming

OK, returns a streamed sequence of `chat completion chunk` objects.

- **Content-Type**: `text/event-stream`

#### Schema

| Field | Type | Required | Nullable | Description |
|-------|------|----------|----------|-------------|
| `id` | `string` | **Yes** | No | A unique identifier for the chat completion. Each chunk has the same ID. |
| `choices` | `object[]` | **Yes** | No | A list of chat completion choices. |
| `choices[].delta` | `object` | **Yes** | No | A chat completion delta generated by streamed model responses. |
| `choices[].delta.content` | `string` or `null` | No | Yes | The contents of the chunk message. |
| `choices[].delta.reasoning_content` | `string` or `null` | No | Yes | For thinking mode only. The reasoning contents of the assistant message, before the final answer. |
| `choices[].delta.role` | `string` | No | No | The role of the author of this message. Value: `assistant`. |
| `choices[].logprobs` | `object` or `null` | No | Yes | Log probability information for the choice. |
| `choices[].logprobs.content` | `object[]` or `null` | **Yes** | Yes | A list of message content tokens with log probability information. |
| `choices[].logprobs.content[].token` | `string` | **Yes** | No | The token. |
| `choices[].logprobs.content[].logprob` | `number` | **Yes** | No | The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value `-9999.0` is used to signify that the token is very unlikely. |
| `choices[].logprobs.content[].bytes` | `integer[]` or `null` | **Yes** | Yes | A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token. |
| `choices[].logprobs.content[].top_logprobs` | `object[]` | **Yes** | No | List of the most likely tokens and their log probability, at this token position. In rare cases, there may be fewer than the number of requested `top_logprobs` returned. |
| `choices[].logprobs.content[].top_logprobs[].token` | `string` | **Yes** | No | The token. |
| `choices[].logprobs.content[].top_logprobs[].logprob` | `number` | **Yes** | No | The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value `-9999.0` is used to signify that the token is very unlikely. |
| `choices[].logprobs.content[].top_logprobs[].bytes` | `integer[]` or `null` | **Yes** | Yes | A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token. |
| `choices[].logprobs.reasoning_content` | `object[]` or `null` | No | Yes | A list of reasoning content tokens with log probability information. |
| `choices[].logprobs.reasoning_content[].token` | `string` | **Yes** | No | The token. |
| `choices[].logprobs.reasoning_content[].logprob` | `number` | **Yes** | No | The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value `-9999.0` is used to signify that the token is very unlikely. |
| `choices[].logprobs.reasoning_content[].bytes` | `integer[]` or `null` | **Yes** | Yes | A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token. |
| `choices[].logprobs.reasoning_content[].top_logprobs` | `object[]` | **Yes** | No | List of the most likely tokens and their log probability, at this token position. In rare cases, there may be fewer than the number of requested `top_logprobs` returned. |
| `choices[].logprobs.reasoning_content[].top_logprobs[].token` | `string` | **Yes** | No | The token. |
| `choices[].logprobs.reasoning_content[].top_logprobs[].logprob` | `number` | **Yes** | No | The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value `-9999.0` is used to signify that the token is very unlikely. |
| `choices[].logprobs.reasoning_content[].top_logprobs[].bytes` | `integer[]` or `null` | **Yes** | Yes | A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token. |
| `choices[].finish_reason` | `string` or `null` | **Yes** | Yes | The reason the model stopped generating tokens. Possible values: `stop`, `length`, `content_filter`, `tool_calls`, `insufficient_system_resource`. |
| `choices[].index` | `integer` | **Yes** | No | The index of the choice in the list of choices. |
| `created` | `integer` | **Yes** | No | The Unix timestamp (in seconds) of when the chat completion was created. Each chunk has the same timestamp. |
| `model` | `string` | **Yes** | No | The model to generate the completion. |
| `system_fingerprint` | `string` | **Yes** | No | This fingerprint represents the backend configuration that the model runs with. |
| `object` | `string` | **Yes** | No | The object type, which is always `chat.completion.chunk`. |

#### Example Response

```
data: {"id": "1f633d8bfc032625086f14113c411638", "choices": [{"index": 0, "delta": {"content": "", "role": "assistant"}, "finish_reason": null, "logprobs": null}], "created": 1718345013, "model": "deepseek-v4-pro", "system_fingerprint": "fp_a49d71b8a1", "object": "chat.completion.chunk", "usage": null}

data: {"choices": [{"delta": {"content": "Hello", "role": "assistant"}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1718345013, "id": "1f633d8bfc032625086f14113c411638", "model": "deepseek-v4-pro", "object": "chat.completion.chunk", "system_fingerprint": "fp_a49d71b8a1"}

data: {"choices": [{"delta": {"content": "!", "role": "assistant"}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1718345013, "id": "1f633d8bfc032625086f14113c411638", "model": "deepseek-v4-pro", "object": "chat.completion.chunk", "system_fingerprint": "fp_a49d71b8a1"}

data: {"choices": [{"delta": {"content": " How", "role": "assistant"}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1718345013, "id": "1f633d8bfc032625086f14113c411638", "model": "deepseek-v4-pro", "object": "chat.completion.chunk", "system_fingerprint": "fp_a49d71b8a1"}

data: {"choices": [{"delta": {"content": " can", "role": "assistant"}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1718345013, "id": "1f633d8bfc032625086f14113c411638", "model": "deepseek-v4-pro", "object": "chat.completion.chunk", "system_fingerprint": "fp_a49d71b8a1"}

data: {"choices": [{"delta": {"content": " I", "role": "assistant"}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1718345013, "id": "1f633d8bfc032625086f14113c411638", "model": "deepseek-v4-pro", "object": "chat.completion.chunk", "system_fingerprint": "fp_a49d71b8a1"}

data: {"choices": [{"delta": {"content": " assist", "role": "assistant"}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1718345013, "id": "1f633d8bfc032625086f14113c411638", "model": "deepseek-v4-pro", "object": "chat.completion.chunk", "system_fingerprint": "fp_a49d71b8a1"}

data: [DONE]
```
