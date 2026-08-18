# GLM-5.3 SDK Documentation

This directory documents Z.AI/Zhipu's GLM-5.3, snapshotted from the official Chinese
documentation (https://docs.bigmodel.cn/). At snapshot time (2026-08-14) the API is
announced but **not yet live** ("API 即将上线"), the international docs.z.ai site carries
no GLM-5.3 content, and no pricing is published; the snapshot reflects the pre-launch
documentation and should be refreshed once the API launches.

## Documentation

- [glm-5.3.md](./docs/glm-5.3.md) - GLM-5.3 model guide (1M context, 128K max output)
- [thinking.md](./docs/thinking.md) - Deep-thinking parameters, including the 5.3 changes

## Key protocol differences vs GLM-5.2

The wire format (chat completions, streaming `reasoning_content`/`content`/`tool_calls`
fields, `tool_stream`) is unchanged; the thinking parameter contract tightens:

- **Thinking can no longer be disabled**: GLM-5.3 uses forced thinking. A request with
  `thinking.type: "disabled"` returns an API error (GLM-5.2 and earlier accept it).
- **`reasoning_effort` accepts only `max` (default), `high`, and `low`** on GLM-5.3;
  every other value returns an error. GLM-5.2 instead accepts the full compatibility
  vocabulary and maps it server-side (`low`/`medium` → `high`, `xhigh` → `max`,
  `none`/`minimal` skip thinking). `low` on GLM-5.3 is a new light-reasoning mode.
- Everything else matches GLM-5.2: 1M context, 128K max output, `tool_choice` only
  `auto`, function calling, context caching, structured output.
