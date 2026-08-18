# 新增 agenthub-dev 技能与 changelog 详情目录

- **Date:** 2026-07-17
- **Type:** process
- **Scope:** `skills`, `llmsdk_docs`, `changelog`, `api_captures`
- **PR:** [#158](https://github.com/Prism-Shadow/agenthub/pull/158)

[English](2026-07-17-agenthub-dev-skill.md)

## 变更内容

- 新增 `.agents/skills/agenthub-dev/SKILL.md`，固化了模型支持的开发流程：将官方文档同步到 `llmsdk_docs/`，把一次带思考内容的流式工具调用往返实况抓取到被 git 忽略的 `api_captures/`，实现 Python/TypeScript 成对的协议客户端并保证消息转换是双射的，且仅以模型范围的 e2e 测试进行验证。
- 该流程将四种情形设为必须询问用户的硬性中断点：官方文档不清晰或无法获取、缺少服务商 API key、任何实时 API 请求错误，以及不显而易见的 `UniConfig` 键映射。
- 将 `api_captures/` 加入 `.gitignore`，作为原始 API 实况抓取结果的存放处；当文档与实况抓取结果不一致时，以实况抓取结果为准。
- 新增 `changelog/` 目录：`CHANGELOG.md` 中的每个条目只保留一行简述，并链接到此处的详情文件（参见 `changelog/README.md`）。
