# DeepSeek Quickstart

DeepSeek provides OpenAI-compatible and Anthropic-compatible API formats. For OpenAI SDK usage, point the client at DeepSeek's base URL and pass a DeepSeek API key.

## API Parameters

| Parameter | Value |
| --- | --- |
| OpenAI `base_url` | `https://api.deepseek.com` (Chat Completions and Responses) |
| Anthropic `base_url` | `https://api.deepseek.com/anthropic` |
| API key | Create one in the DeepSeek platform |
| Current models | `deepseek-v4-flash`, `deepseek-v4-pro`, `deepseek-v4-flash-vision-exp` |
| Compatibility model aliases | `deepseek-chat`, `deepseek-reasoner` |

`deepseek-chat` and `deepseek-reasoner` are compatibility aliases scheduled for deprecation on 2026-07-24. They correspond to non-thinking and thinking mode on `deepseek-v4-flash`.

Every model carries a 1M context window and a 384K max output, and supports JSON output, tool
calls, the Responses API, the Anthropic API and chat prefix completion (Beta).
`deepseek-v4-flash-vision-exp` adds image input, and prices the same as `deepseek-v4-flash`; its
image tokens are counted by image size and billed with the rest
(https://api-docs.deepseek.com/quick_start/pricing, snapshot 2026-08-21).

## Invoke the Chat API

The `/chat/completions` endpoint follows the OpenAI Chat Completions shape. Set `stream` to `true` for streaming responses.

### curl

```bash
curl https://api.deepseek.com/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${DEEPSEEK_API_KEY}" \
  -d '{
        "model": "deepseek-v4-pro",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "stream": false
      }'
```

### Python

Install the OpenAI SDK first:

```bash
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

print(response.choices[0].message.content)
```

When using the OpenAI Python SDK, pass the DeepSeek-specific `thinking` object through `extra_body`.

### Node.js

Install the OpenAI SDK first:

```bash
npm install openai
```

```ts
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "https://api.deepseek.com",
  apiKey: process.env.DEEPSEEK_API_KEY,
});

async function main() {
  const completion = await openai.chat.completions.create({
    model: "deepseek-v4-pro",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello!" },
    ],
    thinking: { type: "enabled" },
    reasoning_effort: "high",
    stream: false,
  });

  console.log(completion.choices[0].message.content);
}

main();
```

## Related Guides

- [Responses API](./docs/responses-api.md)
- [Vision](./docs/vision.md)
- [Thinking Mode](./docs/thinking-mode.md)
- [Multi-round Chat](./docs/multi-round-chat.md)
- [JSON Mode](./docs/json-mode.md)
- [Tool Calls](./docs/tool-calls.md)
