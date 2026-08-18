# DeepSeek 空闲时段定价刷新

- **Date:** 2026-08-18
- **Type:** fix
- **Scope:** `registry`
- **PR:** [#172](https://github.com/Prism-Shadow/agenthub/pull/172)

[English](2026-08-18-deepseek-pricing.md)

## 变更内容

- 官方 DeepSeek registry 条目按更新后的官方价目重新定价，取空闲时段档位（高峰时段——
  北京时间 9:00-12:00 与 14:00-18:00——为其两倍）：`deepseek-v4-flash` 从每百万 token
  CNY 1.0/2.0（缓存 0.02）调整为 CNY 1.5/4.5（缓存 0.05），`deepseek-v4-pro` 从
  CNY 3.0/6.0（缓存 0.025）调整为 CNY 4.5/13.5（缓存 0.15）。
- OpenRouter 与 SiliconFlow 的 DeepSeek 条目保持各自平台的价格。
