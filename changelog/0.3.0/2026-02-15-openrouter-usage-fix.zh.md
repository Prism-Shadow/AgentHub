# 修复来自 OpenRouter 提供方的 token 用量计算

- **Date:** 2026-02-15
- **Type:** fix
- **Scope:** `utils`, `glm5`, `qwen3`, `integration/tracer`, `tests`
- **PR:** [#73](https://github.com/Prism-Shadow/agenthub/pull/73)

[English](2026-02-15-openrouter-usage-fix.md)

## 变更内容

- OpenRouter 偶尔会在 completion tokens 中遗漏 reasoning tokens；Python 与 TypeScript 两侧的用量元数据计算均已对此做出补偿 ([#73](https://github.com/Prism-Shadow/agenthub/pull/73))。

*本条目在变更日志按发布版本重新整理时补录；完整背景请参见该发布版本区间的 git 历史。*
