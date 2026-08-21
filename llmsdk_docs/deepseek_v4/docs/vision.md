# Vision (Image Input)

> Source: https://api-docs.deepseek.com/guides/vision (snapshot 2026-08-21)

`deepseek-v4-flash-vision-exp` accepts images alongside text, so the model can describe pictures,
read text out of screenshots, and analyse charts. Formats are detected from the file content
rather than the extension: JPEG, PNG, GIF and WebP.

## Chat Completions

The `content` of a user message becomes an array of blocks, and an image arrives in one of three
ways.

### Base64 data URL

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

with open("image.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="deepseek-v4-flash-vision-exp",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }
    ],
)
print(response.choices[0].message.content)
```

### External URL

```python
response = client.chat.completions.create(
    model="deepseek-v4-flash-vision-exp",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            ],
        }
    ],
)
```

The URL may be up to 8192 characters, the image up to 32 MiB, and the download must finish within
60 seconds.

### Files API reference

```python
{"type": "file", "file_id": "file-api-xxxxxxxxxxxxxxxx"}
{"type": "file", "file_data": "data:image/jpeg;base64,<BASE64_DATA>", "filename": "image.jpg"}
```

## Detail level

| Value | Behaviour |
| --- | --- |
| `low` | Downscales to 512x512; faster and cheaper when fine detail does not matter |
| `high` | Keeps the original image (compatibility alias for `original`) |
| `original` | Keeps the original image |
| `auto` | Chooses automatically (currently `original`) |

```json
{"type": "image_url", "image_url": {"url": "https://example.com/image.jpg", "detail": "low"}}
```

## Responses API

The same three input methods arrive as `input_image` content parts in user/developer messages or
in tool output:

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
```

`detail` behaves the same way, `image_url` and `file_id` are mutually exclusive, and an image in a
system or assistant message returns `400`.

## Anthropic API

The Anthropic-compatible endpoint (`https://api.deepseek.com/anthropic`) takes an `image` content
block instead:

```python
message = client.messages.create(
    model="deepseek-v4-flash-vision-exp",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": "<BASE64_DATA>"},
                },
            ],
        }
    ],
)
```

| `source.type` | Notes |
| --- | --- |
| `base64` | Requires `media_type`: image/jpeg, image/png, image/gif, image/webp |
| `url` | Up to 8192 characters |
| `file` | Requires the `anthropic-beta: files-api-2025-04-14` header |

## Token usage and billing

Images are resized before inference: anything under roughly 384x384 is scaled up, and larger
images are scaled down to roughly 800x800, which puts an upper bound of 384 tokens on one image.
Each image counts independently, and the tokens are billed with the rest of the input.

## Limits

| Limit | Value |
| --- | --- |
| Supported formats | JPEG, PNG, GIF, WebP |
| External URL length | 8192 characters |
| Request body size | 48 MiB |
| Max single image (base64 / URL) | 32 MiB |
| Max single image (Files API `file_id`) | 64 MiB |
| Max images per request | 600 |
| Max total image size (without `file_id`) | 64 MiB |
| Max total image size (including `file_id`) | 200 MiB |
| Max image dimension | 8192 px per side (4096 px with 15+ images) |

## Restrictions

- Images are supported in user messages only: an image in a system or assistant message returns
  `400`. The Responses API additionally accepts images in `function_call_output` and
  `custom_tool_call_output`.
- Only vision models accept images. On Chat Completions another model returns `400`; on the
  Responses API it substitutes placeholder text instead
  ([responses-api.md](./responses-api.md)).
- User text containing the reserved image placeholder tokens returns `400`.
