# Version 0.4.0

Released on 2026-07-20.

- [2026-07-20] Content items now carry an opaque `fidelity` payload that absorbs the former `signature`/`phase` fields (breaking), and OpenAI-compatible clients use it to replay thinking through exactly the reasoning field the upstream produced. ([details](2026-07-20-reasoning-field-fidelity.md), [#159](https://github.com/Prism-Shadow/agenthub/pull/159))

- [2026-07-17] Add the `agenthub-dev` skill that fixes the model-support development workflow, and the `changelog/` details directory. ([details](2026-07-17-agenthub-dev-skill.md), [#158](https://github.com/Prism-Shadow/agenthub/pull/158))

- [2026-07-14] Raise `EmptyResponseError` when a model completes a response with thinking output only, since sending it back would fail with a 400 error. It and `ToolCallArgumentParseError` now inherit the new `AgentHubError` base class. ([details](2026-07-14-empty-response-error.md), [#157](https://github.com/Prism-Shadow/agenthub/pull/157))

- [2026-06-10] Support Claude 5 models. ([details](2026-06-10-claude-5.md), [#149](https://github.com/Prism-Shadow/agenthub/pull/149))
