# GLM-5.3 SDK Documentation

This directory contains documentation for Z.AI's GLM-5.3 API, snapshotted as raw markdown
from the official documentation (https://docs.z.ai/).

## Documentation

- [glm-5.3.md](./docs/glm-5.3.md) - GLM-5.3 model guide (1M context, 128K max output)
- [migrate-to-glm-new.md](./docs/migrate-to-glm-new.md) - Migration guide from GLM-5.2/5.1/5/4.x
- [concept-param.md](./docs/concept-param.md) - Core request parameters
- [thinking.md](./docs/thinking.md) - Deep thinking capability
- [thinking-mode.md](./docs/thinking-mode.md) - Thinking modes (interleaved, preserved, turn-level)
- [stream-tool.md](./docs/stream-tool.md) - Streaming tool-call parameters (`tool_stream`)
- [chat-completion.md](./docs/chat-completion.md) - Chat Completion API reference (OpenAPI schema)

## Key protocol differences vs GLM-5.2

The wire format (chat completions, streaming `reasoning_content`/`content`/`tool_calls`
fields, `tool_stream`) is unchanged; the thinking parameter contract tightens:

- **Thinking can no longer be disabled**: GLM-5.3 uses forced thinking. A request with
  `thinking.type: "disabled"` returns an API error (GLM-5.2 and earlier accept it).
- **`reasoning_effort` accepts only `max` (default), `high`, and `low`** on GLM-5.3;
  every other value returns an error. GLM-5.2 instead accepts the full compatibility
  vocabulary and maps it server-side (`low`/`medium` → `high`, `xhigh` → `max`,
  `none`/`minimal` skip thinking). `low` on GLM-5.3 is a new light-reasoning mode. The
  Coding Plan endpoint maps the wider vocabulary hierarchically instead
  (`none`/`minimal`/`low` → `low`, `medium`/`high` → `high`, `xhigh`/`max` → `max`).
- Everything else matches GLM-5.2: text-only input, 1M context, 128K max output
  (`max_tokens` default 65536, maximum 131072), `tool_choice` only `auto`, function
  calling, context caching, structured output.

## Pricing

Official list price per 1M tokens (https://docs.z.ai/guides/overview/pricing): input
$1.4, cached input $0.26, output $4.4 - the same rates as GLM-5.2 and GLM-5.1.
