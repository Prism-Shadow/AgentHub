# 开发 skill 现在规定了客户端实现标准

- **Date:** 2026-08-16
- **Type:** process
- **Scope:** `skills`
- **PR:** [#170](https://github.com/Prism-Shadow/agenthub/pull/170)

[English](2026-08-16-client-implementation-standard.md)

## 变更内容

`agenthub-dev` skill 的 Stage 3 新增了客户端实现标准，取代了原先"转换必须双射……完全复现原始内容，包括 `fidelity` 载荷"这条规则：

- 回放最小化：只有当探测证明缺了它请求就会失败、或模型表现会退化时，某个字段才进入 `fidelity`；通用字段已经承载的内容一律不进。服务端生成的 item id 不存；`phase`、thinking signature、上游要求的 reasoning 字段名要存。
- `tool_call_id` 是强制的 —— 存的必须是服务端的 call id，而不是 item id。
- 每个协议家族都有一个参考客户端可循：Responses 风格协议看 `gpt5_5/`，Chat Completions 看 `openai/`。
- 默认内联；辅助函数只用于大段自包含逻辑，绝不用于几行代码。
- 流式处理只读 delta：部分工具调用在 item-added 事件上开启，通过参数 delta 累积，并在终止事件处作为完整的 `tool_call` 发出。完成类事件只被忽略而不做交叉校验，服务端错误事件交给未知事件兜底。

Stage 2 新增了为此提供依据的探测：抓取之后，把 assistant 轮次反复重发、每次移除一个候选字段，并把 API 能容忍的移除项记录在抓取旁边。
