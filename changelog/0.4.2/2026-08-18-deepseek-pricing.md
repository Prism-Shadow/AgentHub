# DeepSeek off-peak pricing refresh

- **Date:** 2026-08-18
- **Type:** fix
- **Scope:** `registry`
- **PR:** [#172](https://github.com/Prism-Shadow/agenthub/pull/172)

[中文版](2026-08-18-deepseek-pricing.zh.md)

## What changed

- The official DeepSeek registry entries were re-priced from the updated official list,
  using the off-peak tier (peak hours — Beijing 9:00-12:00 and 14:00-18:00 — are double):
  `deepseek-v4-flash` moved from CNY 1.0/2.0 (cached 0.02) to CNY 1.5/4.5 (cached 0.05)
  per million tokens, and `deepseek-v4-pro` from CNY 3.0/6.0 (cached 0.025) to
  CNY 4.5/13.5 (cached 0.15).
- The OpenRouter and SiliconFlow DeepSeek entries keep their own platforms' prices.
