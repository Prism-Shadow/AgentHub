# 统一 Kimi 系列 client

- **Date:** 2026-08-17
- **Type:** refactor
- **Scope:** `kimi_k3`, `auto_client`, `tests`
- **PR:** [#171](https://github.com/Prism-Shadow/agenthub/pull/171)
- **Breaking:** yes — 删除了 `kimi_k2_6` client 文件夹；`KimiK2_6Client` 的深层导入失效。

[English](2026-08-17-unify-kimi-clients.md)

## 变更内容

- `kimi_k2_6` client 合并进 `kimi_k3`（`KimiK3Client`），后者现在服务整个 Kimi K2.5+
  系列；`kimi-k2.5`/`kimi-k2.6` client type 路由到它。
- 思考配置按模型代际分支：K2.x 模型保留 `extra_body.thinking` 配置
  （`ThinkingLevel.NONE` 为 `disabled`，其余为 `enabled`+`keep: all`），K3 模型保留
  `reasoning_effort`（`NONE` 降级为 `low`）。
- `tool_choice: "required"` 仍仅限 K3；K2.x 模型对它抛 `UnsupportedParameterError`，
  强制指定单个工具对全家族抛错。
- `trace_id` 仅在 K2.x 模型上作为 `prompt_cache_key` 发送（K3 的上下文缓存是自动的）。
- `temperature`/`fast_mode`/`prompt_caching` 的拒绝信息改为指向 Kimi 家族而非单一代际。

## 兼容性

- 从 `kimi_k3` 导入 `KimiK3Client` 以替代从 `kimi_k2_6` 导入 `KimiK2_6Client`；
  `kimi-k2.5` 与 `kimi-k2.6` client type 字符串继续可用。
