# GLM-5.2 SDK Documentation

This directory contains documentation for Z.AI's GLM-5.2 API, snapshotted as raw markdown
from the official documentation (https://docs.z.ai/).

## Documentation

- [glm-5.2.md](./docs/glm-5.2.md) - GLM-5.2 model guide (1M context, 128K max output)
- [migrate-to-glm-new.md](./docs/migrate-to-glm-new.md) - Migration guide from GLM-5.1/5/4.x
- [concept-param.md](./docs/concept-param.md) - Core request parameters
- [thinking.md](./docs/thinking.md) - Deep thinking capability
- [thinking-mode.md](./docs/thinking-mode.md) - Thinking modes (interleaved, preserved, turn-level)
- [stream-tool.md](./docs/stream-tool.md) - Streaming tool-call parameters (`tool_stream`)
- [chat-completion.md](./docs/chat-completion.md) - Chat Completion API reference (OpenAPI schema)

## Key protocol differences vs GLM-5.1

- New top-level `reasoning_effort` parameter (GLM-5.2 only, default `max`; effective when
  thinking is enabled). The server maps compatibility values itself: `none`/`minimal` skip
  thinking, `low`/`medium` map to `high`, and `xhigh` maps to `max`.
- `thinking` stays `{"type": "enabled"|"disabled"}` (default enabled); when enabled the
  model decides for itself whether to think. `"clear_thinking": false` still turns on
  preserved thinking on the standard endpoint and requires replaying `reasoning_content`
  unmodified.
- Maximum context grows to 1M tokens and maximum output to 128K tokens.
- `tool_choice` still supports only `auto`; `tool_stream: true` still enables streaming
  tool-call arguments; streaming fields (`reasoning_content`/`content`/`tool_calls`) are
  unchanged.
