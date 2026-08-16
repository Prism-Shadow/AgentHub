# 开发 skill 现在规定了客户端实现标准

- **Date:** 2026-08-16
- **Type:** process
- **Scope:** `skills`
- **PR:** [#170](https://github.com/Prism-Shadow/agenthub/pull/170)

[English](2026-08-16-client-implementation-standard.md)

## 变更内容

- **回放要最小化，而不是穷尽。** Stage 3 中"转换必须是双射……必须完全复现原始内容，包括 `fidelity` 载荷"这条规则已删除。现在被回放的 assistant 轮次只需要被 API 接受且模型行为不受影响即可，而 `fidelity` 是一条例外通道：只有当实况探测证明缺了它请求就会失败、或模型表现会退化时，某个字段才配写进去。API 会重新生成或直接忽略的、由服务端产生的 id —— reasoning item id、output-item id —— 一律不存；而 `phase`、thinking signature、以及严格上游所要求的确切 reasoning 字段名，则要存。
- **Stage 2 新增了做出该判断所依据的探测。** 抓取完成后，把 assistant 轮次反复重发，每次移除一个候选字段，并把 API 能容忍的移除项记录在抓取旁边。API 需要回传什么，现在是测出来的，而不是从它发出的内容里推断出来的。
- **`fidelity` 不得重复通用字段已经承载的内容** —— thinking 文本、工具调用的名称或已解析的参数。wire item 改为从通用字段重建。
- **`tool_call_id` 是强制的**：必须抓取并回放服务端的 call id。当某种格式同时带有 item id 和 call id 时，把结果与调用关联起来的是 call id。
- **照着参考客户端写，不要另起炉灶。** Responses 风格协议以 `gpt5_5/` 为准，Chat Completions 以 `openai/` 为准 —— 方法顺序、控制流、命名都保持一致，使新客户端读起来就像是对参考实现的一个 diff。
- **不要辅助函数层。** 用于字段访问、JSON 归一化或校验、错误格式化、用量算术的模块级辅助函数一律去掉。直接读取 SDK 属性，用量算术就地内联，同时兼容 dict 与 SDK 对象的双路访问 shim 在任何情况下都不成立。
- **只按 delta 处理流式输出。** 部分工具调用在 item-added 事件上开启，通过参数 delta 累积，并在终止事件处作为完整的 `tool_call` 发出，与 `gpt5_5` 一致 —— 每个客户端都必须发出完整的 `tool_call`，而不能只有 partial。完成类事件（`response.output_item.done` 及其等价物）与其他被忽略的类型列在一起，绝不与 delta 的结果相互校验；服务端错误事件（`response.failed`、`response.error`）交给未知事件兜底，而不是把它们翻译成 AgentHub 错误。

## 原因

- 这里的每一条规则，都对应 [MiniMax M3 PR](2026-08-03-minimax-m3.zh.md) 里实际交付、并在评审中被移除的一个缺陷：往 `fidelity` 里存了 API 根本不需要回传的字段、围绕 dict/SDK 双路访问包了一层模块级辅助函数、对完成的 item 做 JSON 校验、翻译服务端错误事件、以及从完成事件而不是 delta 来定型工具调用。skill 是成因而非旁观者 —— 它"必须完全复现原始内容"的措辞主动把人推向"什么都存"，因此这次是重写那条规则，而不是在它旁边再加一条。
- 最小集合无法靠读抓取推理出来，因为抓取显示的是 API *发出*什么，而不是它*需要回传*什么。探测是唯一能确定的办法，而且很便宜：MiniMax 那次探测只用了四个请求，就证明 reasoning item 仅凭其文本即可回放，完全不需要 id。
- 为每个协议家族指定一个参考实现，能把"按本仓库风格写"变成可核查的事情。评审者拿新客户端与 `gpt5_5/` 做 diff；凡不是协议差异的地方，都是问题。
