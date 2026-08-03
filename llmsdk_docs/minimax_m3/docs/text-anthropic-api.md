> ## Documentation Index
>
> Fetch the complete documentation index at: [/docs/llms.txt](https://platform.minimax.io/docs/llms.txt)
>
> Use this file to discover all available pages before exploring further.

[Skip to main content](https://platform.minimax.io/docs/api-reference/text-anthropic-api#content-area)

To meet developers’ needs for the Anthropic API ecosystem, our API now supports the Anthropic API format. With simple configuration, you can integrate MiniMax capabilities into the Anthropic API ecosystem.

## [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#quick-start)  Quick Start

### [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#1-install-anthropic-sdk)  1\. Install Anthropic SDK

Python

Node.js

```
pip install anthropic
```

```
npm install @anthropic-ai/sdk
```

### [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#2-configure-environment-variables)  2\. Configure Environment Variables

```
export ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic
export ANTHROPIC_API_KEY=${YOUR_API_KEY}
```

### [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#3-call-api)  3\. Call API

Python

```
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="MiniMax-M3",
    max_tokens=1000,
    system="You are a helpful assistant.",
    messages=[\
        {\
            "role": "user",\
            "content": [\
                {\
                    "type": "text",\
                    "text": "Hi, how are you?"\
                }\
            ]\
        }\
    ]
)

for block in message.content:
    if block.type == "thinking":
        print(f"Thinking:\n{block.thinking}\n")
    elif block.type == "text":
        print(f"Text:\n{block.text}\n")
```

### [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#4-important-note)  4\. Important Note

In multi-turn function call conversations, the complete model response (i.e., the assistant message) must be append to the conversation history to maintain the continuity of the reasoning chain.

- Append the full `response.content` list to the message history (includes all content blocks: thinking/text/tool\_use)

## [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#supported-models)  Supported Models

When using the Anthropic SDK, the `MiniMax-M3``MiniMax-M2.7``MiniMax-M2.7-highspeed``MiniMax-M2.5``MiniMax-M2.5-highspeed``MiniMax-M2.1``MiniMax-M2.1-highspeed``MiniMax-M2` model is supported:

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

The Anthropic API compatibility interface currently only supports the
`MiniMax-M3``MiniMax-M2.7``MiniMax-M2.7-highspeed``MiniMax-M2.5``MiniMax-M2.5-highspeed``MiniMax-M2.1``MiniMax-M2.1-highspeed``MiniMax-M2` model. For other models, please use the standard MiniMax API
interface.

## [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#compatibility)  Compatibility

### [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#supported-parameters)  Supported Parameters

When using the Anthropic SDK, we support the following input parameters:

| Parameter | Support Status | Description |
| --- | --- | --- |
| `model` | Fully supported | supports `MiniMax-M3``MiniMax-M2.7``MiniMax-M2.7-highspeed``MiniMax-M2.5``MiniMax-M2.5-highspeed``MiniMax-M2.1``MiniMax-M2.1-highspeed``MiniMax-M2` model |
| `messages` | Partial support | `MiniMax-M3` supports text, image, video, tool use, tool result, and thinking blocks. The M2.7, M2.5, M2.1, and M2 series support text and tool-call content blocks only; they do not support image or video input |
| `max_tokens` | Fully supported | Maximum number of tokens to generate |
| `stream` | Fully supported | Streaming response |
| `system` | Fully supported | System prompt |
| `temperature` | Fully supported | Range \[0, 2\], controls output randomness, recommended value: 1 |
| `tool_choice` | Fully supported | Tool selection strategy |
| `tools` | Fully supported | Tool definitions |
| `top_p` | Fully supported | Nucleus sampling parameter, range \[0, 1\]. Default 0.95 for `MiniMax-M3` and 0.9 for M2.x models |
| `metadata` | Fully Supported | Metadata |
| `thinking` | Fully Supported | Thinking is off by default for MiniMax-M3 and can be enabled with `adaptive`. Thinking cannot be disabled for M2.x models. |
| `service_tier` | Fully Supported | Request admission tier. Supported values are `standard` and `priority`; if omitted, requests use `standard`. The `priority` [price](https://platform.minimax.io/docs/guides/pricing-paygo) is 1.5 times the `standard` price and ensures priority admission so the request is processed ahead of other requests, leading to faster responses and fewer failures. |
| `top_k` | Ignored | This parameter will be ignored |
| `stop_sequences` | Ignored | This parameter will be ignored |
| `mcp_servers` | Ignored | This parameter will be ignored |
| `context_management` | Ignored | This parameter will be ignored |
| `container` | Ignored | This parameter will be ignored |

### [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#thinking-control)  Thinking Control

For `MiniMax-M3`, the `thinking` parameter controls whether the model can emit `thinking` content blocks.

- If `thinking` is omitted, thinking is off by default and the response does not include `thinking` blocks.
- Set `thinking: {"type": "adaptive"}` to explicitly enable thinking. For MiniMax-M3, `adaptive` is equivalent to thinking on.
- Set `thinking: {"type": "disabled"}` to explicitly keep MiniMax-M3 thinking output off.
- For M2.x models, thinking cannot be disabled; `thinking: {"type": "disabled"}` is accepted but thinking remains on.

When a response includes `thinking` blocks, preserve them unchanged in later turns, especially in tool-use conversations.

### [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#messages-field-support)  Messages Field Support

| Field Type | Support Status | Description |
| --- | --- | --- |
| `type="text"` | Fully supported | Text messages |
| `type="image"` | M3 only | Image input via URL or base64. Supports JPEG, PNG, GIF, WEBP |
| `type="video"` | M3 only | Video input via URL, base64, or `mm_file://{file_id}`. Supports MP4, AVI, MOV, MKV |
| `type="tool_use"` | Fully supported | Tool calls |
| `type="tool_result"` | Fully supported | Tool call results |
| `type="thinking"` | Fully supported | Reasoning content. Return the block unchanged in multi-turn thinking conversations |

For `MiniMax-M3`, URL or base64 videos can be up to 50 MB, images can be up to 10 MB, and the request body can be up to 64 MB. For larger videos, upload through the Files API and pass `mm_file://{file_id}`; Files API videos can be up to 512 MB.Image token usage depends on image size and content. Use this as a rough single-image heuristic; check `POST /anthropic/v1/messages/count_tokens` or response `usage` for exact usage:

| `detail` | Rough single-image token usage |
| --- | --- |
| `low` | Usually a few hundred tokens, up to ~600 |
| `default` | Often ~1k-3k tokens, up to ~5k |
| `high` | Often several thousand tokens, up to ~15k+ |

The Anthropic-compatible API also supports `POST /anthropic/v1/messages/count_tokens` for `MiniMax-M3` token estimation. This endpoint returns input token usage without generating model output.

## [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#examples)  Examples

### [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#streaming-response)  Streaming Response

Python

```
import anthropic

client = anthropic.Anthropic()

print("Starting stream response...\n")
print("=" * 60)
print("Thinking Process:")
print("=" * 60)

stream = client.messages.create(
    model="MiniMax-M3",
    max_tokens=1000,
    system="You are a helpful assistant.",
    messages=[\
        {"role": "user", "content": [{"type": "text", "text": "Hi, how are you?"}]}\
    ],
    stream=True,
)

reasoning_buffer = ""
text_buffer = ""

for chunk in stream:
    if chunk.type == "content_block_start":
        if hasattr(chunk, "content_block") and chunk.content_block:
            if chunk.content_block.type == "text":
                print("\n" + "=" * 60)
                print("Response Content:")
                print("=" * 60)

    elif chunk.type == "content_block_delta":
        if hasattr(chunk, "delta") and chunk.delta:
            if chunk.delta.type == "thinking_delta":
                # Stream output thinking process
                new_thinking = chunk.delta.thinking
                if new_thinking:
                    print(new_thinking, end="", flush=True)
                    reasoning_buffer += new_thinking
            elif chunk.delta.type == "text_delta":
                # Stream output text content
                new_text = chunk.delta.text
                if new_text:
                    print(new_text, end="", flush=True)
                    text_buffer += new_text

print("\n")
```

## [​](https://platform.minimax.io/docs/api-reference/text-anthropic-api\#important-notes)  Important Notes

1. The Anthropic API compatibility interface currently only supports the `MiniMax-M3``MiniMax-M2.7``MiniMax-M2.7-highspeed``MiniMax-M2.5``MiniMax-M2.5-highspeed``MiniMax-M2.1``MiniMax-M2.1-highspeed``MiniMax-M2` model
2. The `temperature` parameter range is \[0, 2\], values outside this range will return an error
3. Some Anthropic parameters (such as `top_k`, `stop_sequences`, `mcp_servers`, `context_management`, `container`) will be ignored
4. `MiniMax-M3` supports image and video input through Anthropic-compatible content blocks. The M2.7, M2.5, M2.1, and M2 series support text and tool-call content blocks only

[Error Code Reference](https://platform.minimax.io/docs/api-reference/errorcode) [OpenAI SDK](https://platform.minimax.io/docs/api-reference/text-openai-api)

Ctrl+I