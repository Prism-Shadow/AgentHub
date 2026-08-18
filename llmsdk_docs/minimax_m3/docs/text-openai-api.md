> ## Documentation Index
>
> Fetch the complete documentation index at: [/docs/llms.txt](https://platform.minimax.io/docs/llms.txt)
>
> Use this file to discover all available pages before exploring further.

[Skip to main content](https://platform.minimax.io/docs/api-reference/text-openai-api#content-area)

To meet developers’ needs for the OpenAI API ecosystem, our API now supports the OpenAI API format. With simple configuration, you can integrate MiniMax capabilities into the OpenAI API ecosystem.

## [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#quick-start)  Quick Start

### [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#1-install-openai-sdk)  1\. Install OpenAI SDK

Python

Node.js

```
pip install openai
```

```
npm install openai
```

### [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#2-configure-environment-variables)  2\. Configure Environment Variables

```
export OPENAI_BASE_URL=https://api.minimax.io/v1
export OPENAI_API_KEY=${YOUR_API_KEY}
```

### [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#3-call-api)  3\. Call API

Python

```
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[\
        {"role": "system", "content": "You are a helpful assistant."},\
        {"role": "user", "content": "Hi, how are you?"},\
    ],
    # Set reasoning_split=True to separate thinking content into reasoning_details field
    extra_body={"reasoning_split": True},
)

print(f"Thinking:\n{response.choices[0].message.reasoning_details[0]['text']}\n")
print(f"Text:\n{response.choices[0].message.content}\n")
```

### [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#4-important-note)  4\. Important Note

In multi-turn function call conversations, the complete model response (i.e., the assistant message) must be append to the conversation history to maintain the continuity of the reasoning chain.

- Append the full `response_message` object (including the `tool_calls` field) to the message history
  - For native OpenAI API with `MiniMax-M3``MiniMax-M2.7``MiniMax-M2.7-highspeed``MiniMax-M2.5``MiniMax-M2.5-highspeed``MiniMax-M2.1``MiniMax-M2.1-highspeed``MiniMax-M2` models, the `content` field will contain `<think>` tag content, which must be preserved completely
  - In the Interleaved Thinking compatible format, by enabling the additional parameter (`reasoning_split=True`), the model’s thinking content is provided separately via the `reasoning_details` field, which must also be preserved completely

## [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#supported-models)  Supported Models

When using the OpenAI SDK, the following MiniMax models are supported:

| Model Name | Context Window | Description |
| --- | --- | --- |
| MiniMax-M3 | 1,000,000 | **Latest M-series language model for agentic reasoning, tool use, coding, and long-context tasks** |
| MiniMax-M2.7 | 204,800 | **Beginning the journey of recursive self-improvement** (output speed approximately 60 tps) |
| MiniMax-M2.7-highspeed | 204,800 | **M2.7 Highspeed: Same performance, faster and more agile (output speed approximately 100 tps)** |
| MiniMax-M2.5 | 204,800 | **Peak Performance. Ultimate Value. Master the Complex (output speed approximately 60 tps)** |
| MiniMax-M2.5-highspeed | 204,800 | **M2.5 highspeed: Same performance, faster and more agile (output speed approximately 100 tps)** |
| MiniMax-M2.1 | 204,800 | **Powerful Multi-Language Programming Capabilities with Comprehensively Enhanced Programming Experience (output speed approximately 60 tps)** |
| MiniMax-M2.1-highspeed | 204,800 | **Faster and More Agile (output speed approximately 100 tps)** |
| MiniMax-M2 | 204,800 | **Agentic capabilities, Advanced reasoning** |

For details on how tps (Tokens Per Second) is calculated, please refer to [FAQ > About APIs](https://platform.minimax.io/docs/faq/about-apis#q-how-is-tps-tokens-per-second-calculated-for-text-models).

For more model information, please refer to the standard MiniMax API
documentation.

## [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#multimodal-input)  Multimodal Input

OpenAI-compatible Chat Completions support text, image, and video input for `MiniMax-M3`.Use `image_url` content parts for images and `video_url` content parts for videos. The `detail` field accepts `low`, `default`, or `high` and defaults to `default`; `max_long_side_pixel` can be used to control the longest side. Images support JPEG, PNG, GIF, and WEBP. Videos support MP4, AVI, MOV, and MKV; `fps` defaults to 1 and accepts values from 0.2 to 5. URL or base64 videos can be up to 50 MB, images can be up to 10 MB, and the request body can be up to 64 MB. For larger videos, upload through the Files API and pass `mm_file://{file_id}`; Files API videos can be up to 512 MB.Image token usage depends on image size and content. Use this as a rough single-image heuristic; check response `usage` or token counting where available for exact usage:

| `detail` | Rough single-image token usage |
| --- | --- |
| `low` | Usually a few hundred tokens, up to ~600 |
| `default` | Often ~1k-3k tokens, up to ~5k |
| `high` | Often several thousand tokens, up to ~15k+ |

Python

```
response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[\
        {\
            "role": "user",\
            "content": [\
                {"type": "text", "text": "Summarize what is happening here."},\
                {\
                    "type": "image_url",\
                    "image_url": {\
                        "url": "https://example.com/image.png",\
                        "detail": "default",\
                    },\
                },\
                {\
                    "type": "video_url",\
                    "video_url": {\
                        "url": "mm_file://file_id",\
                        "detail": "default",\
                    },\
                },\
            ],\
        }\
    ],
)
```

## [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#minimax-m3-request-parameters)  MiniMax-M3 Request Parameters

`MiniMax-M3` supports these additional Chat Completions parameters through the OpenAI-compatible API:

| Parameter | Description |
| --- | --- |
| `thinking` | Controls MiniMax-M3 thinking. `type` can be `disabled` or `adaptive`; when omitted, thinking is on by default. For M2.x models, thinking cannot be disabled. |
| `stream_options.include_usage` | When streaming, set to `true` to include token usage in the stream. |
| `max_tokens` | Legacy generation length limit. |
| `max_completion_tokens` | Generation length limit; use this field for new integrations. |
| `temperature` | Sampling temperature. Range `[0, 2]`, default `1`. |
| `top_p` | Nucleus sampling. Range `[0, 1]`. Default `0.95` for `MiniMax-M3` and `0.9` for M2.x models. |
| `tools` | Function tool definitions. |
| `reasoning_split` | Output-format switch. When enabled, separates thinking content into `reasoning_content` and `reasoning_details`. |
| `service_tier` | Request admission tier. Supported values are `standard` and `priority`; if omitted, requests use `standard`. The `priority` [price](https://platform.minimax.io/docs/guides/pricing-paygo) is 1.5 times the `standard` price and ensures priority admission so the request is processed ahead of other requests, leading to faster responses and fewer failures. |

### [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#thinking-control)  Thinking Control

For `MiniMax-M3`, the `thinking` parameter controls whether the model can emit thinking content.

- If `thinking` is omitted, thinking is on by default and the response includes thinking content.
- Set `thinking: {"type": "adaptive"}` to explicitly keep thinking on. For MiniMax-M3, `adaptive` is equivalent to thinking on.
- Set `thinking: {"type": "disabled"}` to skip thinking and answer directly.
- For M2.x models, thinking cannot be disabled; `thinking: {"type": "disabled"}` is accepted but thinking remains on.

`reasoning_split` does not enable or disable thinking. It only controls how thinking content is returned: when `true`, thinking is exposed through `reasoning_content` and `reasoning_details`; when `false`, native Chat Completions responses keep thinking inside the `content` field with `<think>...</think>` tags.

Python

```
response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[{"role": "user", "content": "Hi, how are you?"}],
    extra_body={
        "thinking": {"type": "adaptive"},
    },
)
```

## [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#examples)  Examples

### [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#streaming-response)  Streaming Response

Python

```
from openai import OpenAI

client = OpenAI()

print("Starting stream response...\n")
print("=" * 60)
print("Thinking Process:")
print("=" * 60)

stream = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[\
        {"role": "system", "content": "You are a helpful assistant."},\
        {"role": "user", "content": "Hi, how are you?"},\
    ],
    # Set reasoning_split=True to separate thinking content into reasoning_details field
    extra_body={"reasoning_split": True},
    stream=True,
)

reasoning_buffer = ""
text_buffer = ""

for chunk in stream:
    if (
        hasattr(chunk.choices[0].delta, "reasoning_details")
        and chunk.choices[0].delta.reasoning_details
    ):
        for detail in chunk.choices[0].delta.reasoning_details:
            if "text" in detail:
                reasoning_text = detail["text"]
                new_reasoning = reasoning_text[len(reasoning_buffer) :]
                if new_reasoning:
                    print(new_reasoning, end="", flush=True)
                    reasoning_buffer = reasoning_text

    if chunk.choices[0].delta.content:
        content_text = chunk.choices[0].delta.content
        new_text = content_text[len(text_buffer) :] if text_buffer else content_text
        if new_text:
            print(new_text, end="", flush=True)
            text_buffer = content_text

print("\n" + "=" * 60)
print("Response Content:")
print("=" * 60)
print(f"{text_buffer}\n")
```

### [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#tool-use-&-interleaved-thinking)  Tool Use & Interleaved Thinking

Learn how to use M3 Tool Use and Interleaved Thinking capabilities with OpenAI SDK, please refer to the following documentation.

## Tool Use & Interleaved Thinking

Learn how to leverage MiniMax-M3 tool calling and interleaved thinking capabilities to enhance performance in complex tasks.

Click here

## [​](https://platform.minimax.io/docs/api-reference/text-openai-api\#important-notes)  Important Notes

1. The `temperature` parameter range is \[0, 2\], recommended value: 1.0, values outside this range will return an error
2. Some OpenAI parameters (such as `presence_penalty`, `frequency_penalty`, `logit_bias`, etc.) will be ignored
3. Image and video inputs are supported by `MiniMax-M3` through OpenAI-compatible message content parts; audio input is not currently supported
4. The `n` parameter only supports value 1
5. The deprecated `function_call` is not supported, please use the `tools` parameter

[Anthropic SDK (Recommended)](https://platform.minimax.io/docs/api-reference/text-anthropic-api) [AI SDK](https://platform.minimax.io/docs/api-reference/text-ai-sdk)

⌘I