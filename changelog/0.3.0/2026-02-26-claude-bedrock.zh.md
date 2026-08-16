# 支持 Amazon Bedrock 上的 Claude

- **Date:** 2026-02-26
- **Type:** feature
- **Scope:** `claude4_5`, `llmsdk_docs`, `tests`
- **PR:** [#79](https://github.com/Prism-Shadow/agenthub/pull/79)

[English](2026-02-26-claude-bedrock.md)

## 变更内容

- 现已通过 Amazon Bedrock 支持 Claude 模型 ([#79](https://github.com/Prism-Shadow/agenthub/pull/79))。
- Bedrock 不接受图像 URL，因此客户端会先获取图像并将其转换为 base64，然后再发送。

*本条目在变更日志按发布版本重新整理时补录；完整背景请参见该发布版本区间的 git 历史。*
