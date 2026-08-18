# 版本 0.4.0

[English](README.md)

发布于 2026-07-20。

- [2026-07-20] 内容项现在携带一个不透明的 `fidelity` 载荷，吸收了原有的 `signature`/`phase` 字段（破坏性变更），OpenAI 兼容客户端借助它精确地通过上游实际产出的那个 reasoning 字段回放思考内容。([详情](2026-07-20-reasoning-field-fidelity.zh.md), [#159](https://github.com/Prism-Shadow/agenthub/pull/159))

- [2026-07-17] 新增固化模型支持开发流程的 `agenthub-dev` 技能，以及 `changelog/` 详情目录。([详情](2026-07-17-agenthub-dev-skill.zh.md), [#158](https://github.com/Prism-Shadow/agenthub/pull/158))

- [2026-07-14] 当模型仅以思考输出结束响应时抛出 `EmptyResponseError`，因为将其回传会触发 400 错误。它与 `ToolCallArgumentParseError` 现在都继承新的 `AgentHubError` 基类。([详情](2026-07-14-empty-response-error.zh.md), [#157](https://github.com/Prism-Shadow/agenthub/pull/157))

- [2026-06-10] 支持 Claude 5 模型。([详情](2026-06-10-claude-5.zh.md), [#149](https://github.com/Prism-Shadow/agenthub/pull/149))
