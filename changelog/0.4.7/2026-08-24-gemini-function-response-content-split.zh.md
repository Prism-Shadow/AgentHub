# Gemini 的 function response 独立成 content

- **Date:** 2026-08-24
- **Type:** fix
- **Scope:** `gemini3_7`, `tests`
- **PR:** [#189](https://github.com/Prism-Shadow/agenthub/pull/189)

[English](2026-08-24-gemini-function-response-content-split.md)

## 改了什么

Gemini 客户端此前把一条通用消息恰好转换成一条 `Content`,于是同时携带工具结果**和**文本的 user 消息——agent 把被中断轮次的工具输出与用户的下一条 prompt 一起重发、或把工具输出折叠进摘要请求——会产生一条混合 `functionResponse` 与 `text` part 的 content。

Vertex AI 拒绝这种混合,而且报错文案有误导性:HTTP 400 `"Requests ending with a model turn are not supported."`,尽管请求明明以 user content 结尾。官方 Gemini API 端点接受同样的请求,所以故障只在 Vertex 上出现,并且同一会话的每次重试都会一模一样地失败。已在两个端点上用 `gemini-3.7-flash` 实测验证:混合 content 是唯一的判别因素——拆开后两端都接受,请求的其余部分本来就合法。

`transform_uni_message_to_model_input` 现在按 part 类别把一条消息拆成连续的同 role content:每一段连续的 function response 自成一条 content,前后的其他 part 各归各的,part 顺序保持不变。不含 function response 的消息——或只含 function response 的消息——仍然转换成单条 content,普通请求的字节与之前完全一致。两个语言实现做了同样的修改。

## 测试

客户端 E2E 套件新增 `should handle tool result mixed with text`(Python 为 `test_tool_result_mixed_with_text`):真实工具调用之后,工具结果与一条后续指令放在同一条通用消息里回传,回复必须证明两半都到达了模型——工具结果里的温度,和文本要求的标记词。已在可用的 provider 上实跑验证,包括 Vertex AI 上的 `gemini-3.7-flash`——正是修复前拒绝该形状的端点。

Python tracer 套件的两个 monitoring 测试不再调用真实模型:改为驱动一个把 wire 流替换成脚本化生成器的 `AutoLLMClient`,trace_id → save_history 的集成缝照常真实运行,而套件不再需要任何 API key——恢复"所有需要真实 key 的测试都住在客户端 E2E 套件、其余套件离线运行"的不变量。
