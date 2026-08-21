# Using the Responses API

> Source: https://api-docs.deepseek.com/guides/responses_api (snapshot 2026-08-21)

DeepSeek serves the OpenAI Responses API format at base URL `https://api.deepseek.com`, so the
OpenAI SDK reaches it with only a base URL and API key change.

## Basic Usage

```python
from openai import OpenAI

client = OpenAI(api_key="<your DeepSeek API Key>", base_url="https://api.deepseek.com")
response = client.responses.create(
    model="deepseek-v4-flash",
    instructions="You are a helpful assistant.",
    input="Hi, how are you?",
)
print(response.output_text)
```

## Streaming

Set `stream: true` to receive server-sent events. Each event carries a `type` and a monotonically
increasing sequence number, and the stream ends with a terminal event rather than `data: [DONE]`.

```python
stream = client.responses.create(
    model="deepseek-v4-flash",
    instructions="You are a helpful assistant.",
    input="Hi, how are you?",
    stream=True,
)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")
```

| Event | Purpose |
| --- | --- |
| `response.created` | Initial event; status is `in_progress` |
| `response.in_progress` | Response generation ongoing |
| `response.output_item.added` / `response.output_item.done` | Output item lifecycle (reasoning, message, function_call, ...) |
| `response.content_part.added` / `response.content_part.done` | Content part lifecycle within an item |
| `response.reasoning_text.delta` / `response.reasoning_text.done` | Chain-of-thought incremental / complete |
| `response.output_text.delta` / `response.output_text.done` | Output text incremental / complete |
| `response.function_call_arguments.delta` / `response.function_call_arguments.done` | Function arguments incremental / complete |
| `response.custom_tool_call_input.delta` / `response.custom_tool_call_input.done` | Custom tool input incremental / complete |
| `response.web_search_call.in_progress` / `.searching` / `.completed` | Web search status |
| `response.completed` | Normal completion, with the full response object |
| `response.incomplete` | Truncation (for example `max_output_tokens` reached) |
| `response.failed` | Failure, with error details |

## Image Input

`deepseek-v4-flash-vision-exp` accepts images as `input_image` content parts:

```python
response = client.responses.create(
    model="deepseek-v4-flash-vision-exp",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in this image?"},
                {"type": "input_image", "image_url": "https://example.com/image.jpg", "detail": "low"},
            ],
        }
    ],
)
print(response.output_text)
```

Images may also ride in `function_call_output` and `custom_tool_call_output`, so a tool can hand
the model an image it produced.

### `input_image` fields

- `image_url`: an HTTP(S) URL (max 8192 characters) or a base64 data URL. JPEG, PNG, GIF and WebP.
- `file_id`: an image id from the Files API (`file-api-...`).
- `detail`: `low` / `high` / `original` / `auto`. `low` downsamples to 512x512; the others keep the
  original size. Ignored when `file_id` is set.

`image_url` and `file_id` are mutually exclusive: passing both, or neither, returns `400`.

### Image restrictions

- Images are allowed only in `user` / `developer` messages and in `function_call_output` /
  `custom_tool_call_output`. An image in a `system` or `assistant` message returns `400`.
- Only vision models read `input_image` parts; other models replace them with placeholder text.
- Limits: 32 MiB per inline image, 64 MiB per `file_id` image, 64 MiB total inline (up to 200 MiB
  with `file_id`), and 600 images per request.

## Top-level request parameters

| Parameter | Status | Notes |
| --- | --- | --- |
| `model` | Supported | `deepseek-v4-flash` / `deepseek-v4-pro` / `deepseek-v4-flash-vision-exp` |
| `input` | Supported | String or input item list; `input` or `instructions` is required |
| `instructions` | Supported | Inserted as the first system message |
| `stream` | Supported | — |
| `temperature` | Supported | Range [0.0, 2.0]; no effect in thinking mode |
| `top_p` | Supported | No effect in thinking mode |
| `max_output_tokens` | Supported | — |
| `top_logprobs` | Supported | Range [0, 20] |
| `tools` | Partially supported | `function` / `web_search` supported; other types ignored |
| `tool_choice` | Supported | `none` / `auto` / `required` / a specific tool |
| `reasoning` | Partially supported | `effort` supported; `summary` accepted but never generated |
| `text` | Partially supported | `format` supported; `verbosity` has no effect |
| `user` | Supported | See Rate Limit & Isolation |
| `parallel_tool_calls` | Ignored | Always enabled |
| `max_tool_calls` | Ignored | — |
| `previous_response_id` | Not supported | Stateless API |
| `conversation` | Not supported | Stateless API |
| `store` | Not supported | Always `store: false` |
| `background` | Not supported | — |
| `metadata` | Not supported | — |
| `include` | Not supported | — |
| `prompt` | Not supported | — |
| `truncation` | Not supported | Exceeding the context window returns `400` |
| `service_tier` | Not supported | — |
| `safety_identifier` | Not supported | — |
| `prompt_cache_key` / `prompt_cache_retention` | Not supported | Context caching is automatic |
| `context_management` | Not supported | — |
| `stream_options` | Not supported | — |

Unsupported parameters are silently ignored rather than rejected.

## Input item types

| Type | Status | Notes |
| --- | --- | --- |
| `message` | Supported | Roles `user` / `assistant` / `system` / `developer` (developer is treated as user); content takes a string or `input_text` / `output_text` / `input_image` parts. The vision model reads images; other models substitute a placeholder. File inputs are unsupported |
| `function_call` | Supported | Merged into the adjacent assistant message |
| `function_call_output` | Supported | `output` takes a string or a content part list; `input_image` parts are read as real images by the vision model |
| `reasoning` | Supported | Plain-text content is merged into the assistant message; `summary` and `encrypted_content` are unsupported |
| `web_search_call` | Supported | Passed back as-is; the server restores its results |
| `custom_tool_call` / `custom_tool_call_output` | Supported | Only the `apply_patch` tool; images in the output are read by the vision model |
| Other types | Ignored | — |

## Tools

| Type | Status | Notes |
| --- | --- | --- |
| `function` | Supported | — |
| `web_search` / `web_search_2025_08_26` | Supported | Executed server-side; `search_context_size` and `user_location` are ignored; auto-continuation is capped at 10 rounds |
| `custom` | Partially supported | Only `{"type": "custom", "name": "apply_patch"}`; another name returns `400` |
| `file_search` / `code_interpreter` / `computer_use` / `mcp` / other built-ins | Ignored | — |

## Response fields and token usage

The response object matches the OpenAI Responses API structure; unsupported capabilities come back
as fixed values (`store: false`, `previous_response_id: null`, `parallel_tool_calls: true`).

- `input_tokens`, with `input_tokens_details.cached_tokens` for context cache hits.
- `output_tokens`, with `output_tokens_details.reasoning_tokens` for chain-of-thought tokens.
