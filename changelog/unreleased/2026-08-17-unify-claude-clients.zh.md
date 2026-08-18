# 统一 Claude 系列 client

- **Date:** 2026-08-17
- **Type:** refactor
- **Scope:** `claude5`, `auto_client`, `tests`
- **PR:** [#171](https://github.com/Prism-Shadow/agenthub/pull/171)
- **Breaking:** yes — 删除了 `claude4_6` client 文件夹（`Claude4_6Client` 的深层导入失效），且 Claude 4.6 模型上的 `temperature` 从透传改为抛 `UnsupportedParameterError`。

[English](2026-08-17-unify-claude-clients.md)

## 变更内容

- `claude4_6` client 合并进 `claude5`（`Claude5Client`），后者现在服务整个 Claude 4.6+
  系列；`claude-*-4-6` client type 路由到它。
- `temperature` 对全家族抛 `UnsupportedParameterError`（API 从 4.7 一代起移除了该参数；
  统一 client 在 4.6 上也一并拒绝）。
- `ThinkingLevel.XHIGH` 在 Claude 4.6 模型上降级为 `output_config.effort: "high"`
  （4.6 没有 xhigh 档），4.7 及以后保持 `"xhigh"`。
- `thinking_summary` 对全家族映射为 `thinking.display: "summarized"`（实测验证：
  Claude 4.6 接受 `display` 字段）。
- `fast_mode` 在 Claude 4.6 模型上抛错，与 API 的模型支持范围一致。

## 兼容性

- 从 `claude5` 导入 `Claude5Client` 以替代从 `claude4_6` 导入 `Claude4_6Client`；
  `claude-4-6` client type 字符串继续可用。
- 在 Claude 4.6 模型上设置 `temperature` 的请求需去掉该参数。
