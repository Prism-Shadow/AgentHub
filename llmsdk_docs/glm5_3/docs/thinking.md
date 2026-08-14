# Deep Thinking (GLM-5.3 update)

> Source: https://docs.bigmodel.cn/cn/guide/capabilities/thinking (snapshot 2026-08-14)

Deep thinking is supported by the GLM-5.3, GLM-5.2, GLM-5.1, GLM-5, GLM-5-Turbo,
GLM-5V-Turbo, GLM-4.7, GLM-4.6, and GLM-4.5 series.

## `thinking.type`

- `enabled` (default): **GLM-5.3 uses forced thinking** (强制思考); the other models
  decide dynamically whether to think.
- `disabled`: direct answers without thinking. **GLM-5.3 no longer supports disabling
  thinking — an API request with `thinking.type: "disabled"` returns an error.**

## `reasoning_effort`

Controls reasoning depth when thinking is enabled; supported on GLM-5.2 and newer.

- **GLM-5.3 accepts only:** `max` (default, recommended, deep reasoning), `high`
  (enhanced reasoning), `low` (light reasoning, new to GLM-5.3). Any other value
  returns an error.
- **GLM-5.2** accepts the full compatibility vocabulary and maps it server-side:
  `none`/`minimal` skip thinking, `low`/`medium` map to `high`, `xhigh` maps to `max`.
- Models before GLM-5.2 take no `reasoning_effort` parameter.

Coding Plan requests map hierarchically instead (GLM-5.3: `none`/`minimal`/`low` → `low`,
`medium`/`high` → `high`, `xhigh`/`max` → `max`); this applies to the Coding Plan
endpoint, not the standard API.
