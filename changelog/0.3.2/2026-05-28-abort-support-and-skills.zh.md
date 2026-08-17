# 新增中止支持与 agent 技能

- **Date:** 2026-05-28
- **Type:** feature
- **Scope:** `abort_signal`, `base_client`, `integration/playground`, `skills`
- **PR:** [#121](https://github.com/Prism-Shadow/agenthub/pull/121), [#128](https://github.com/Prism-Shadow/agenthub/pull/128), [#129](https://github.com/Prism-Shadow/agenthub/pull/129), [#130](https://github.com/Prism-Shadow/agenthub/pull/130), [#133](https://github.com/Prism-Shadow/agenthub/pull/133)

[English](2026-05-28-abort-support-and-skills.md)

## 变更内容

- Python 与 TypeScript 的流式请求均可接受中止信号 ([#128](https://github.com/Prism-Shadow/agenthub/pull/128))，流式过程中复用中止等待器 ([#133](https://github.com/Prism-Shadow/agenthub/pull/133))，playground 新增了中止控件 ([#130](https://github.com/Prism-Shadow/agenthub/pull/130))。
- 在 `skills/` 下新增 `agenthub-python` 与 `agenthub-typescript` 两个 SDK 使用技能 ([#121](https://github.com/Prism-Shadow/agenthub/pull/121), [#129](https://github.com/Prism-Shadow/agenthub/pull/129))。

*本条目在变更日志按发布版本重新整理时补录；完整背景请参见该发布版本区间的 git 历史。*
