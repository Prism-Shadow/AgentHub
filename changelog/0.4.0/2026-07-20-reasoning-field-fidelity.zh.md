# 新增 `fidelity` 字段并精确回放上游产出的 reasoning 字段

- **Date:** 2026-07-20
- **Type:** fix
- **Scope:** `types`, `base_client`, `openai`, `glm5_1`, `kimi_k2_6`
- **PR:** [#159](https://github.com/Prism-Shadow/agenthub/pull/159)
- **Breaking:** yes — 客户端不再读取内容项顶层的 `signature`/`phase`；由早期版本记录的历史消息必须将它们迁移到 `fidelity` 中

[English](2026-07-20-reasoning-field-fidelity.md)

## 问题

OpenAI Chat Completions 兼容服务端对流式思考字段的拼写各不相同——vLLM 与 SiliconFlow 使用 `reasoning_content`，而 OpenRouter 使用 `reasoning`——而在回传 assistant 历史消息时，`openai`、`glm5_1` 和 `kimi_k2_6` 客户端总是在消息上**同时**设置这两个字段。严格的上游会拒绝自己未曾输出过的那种拼写（例如返回 `reasoning_content` 的服务端会拒绝包含 `reasoning` 的请求），从而破坏多轮对话。

## `fidelity` 字段

要修复这一点，需要有地方记录是哪个线路字段承载了思考内容。相比复用 `signature`，内容项现在携带一个专门的字段：

- `fidelity`（`dict[str, Any]` / `Record<string, any>`，可选）——任意 JSON 风格的对象，承载客户端为在回放时复现原始消息而记录的线路层数据。对消费方不透明：原样回传即可。

它取代并吸收了原有的内容项级 `signature` 与 `phase` 字段：

| 客户端 | 旧 | 新 |
| --- | --- | --- |
| `claude5` / `claude4_6` | thinking 项上的 `signature: <sig>`（也用于存放 redacted-thinking 数据） | `fidelity: {"signature": <sig>}` |
| `gemini3` | text / thinking / inline / tool_call 项上的 `signature: <thought_signature>`（即使值为 `None`，键也存在） | `fidelity: {"signature": <thought_signature>}`，不存在时整个字段省略 |
| `gpt5_5` | thinking 项上的 `signature: json.dumps({"id": ..., "encrypted_content": ...})`；text 项上的 `phase: <p>` | `fidelity: {"id": ..., "encrypted_content": ...}`（不再是字符串里套 JSON）；`fidelity: {"phase": <p>}` |
| `openai` / `glm5_1` / `kimi_k2_6` / `deepseek_v4` | 不记录任何内容；两种 reasoning 拼写都回传 | 每个思考增量对应一份 `fidelity: {"reasoning_field": "reasoning_content" \| "reasoning"}` |

## reasoning 字段的修复

接收时，OpenAI 兼容客户端会记录承载每个思考增量的线路字段名。发送时，消息转换精确地通过该字段回放思考内容。回退路径保留了原先最大兼容性的行为：未记录 `reasoning_field` 的思考内容（手写的历史消息、来自其他协议的保真载荷）、同一条消息内字段混用，以及某个数据块同时携带两种拼写的歧义情形（这类增量不记录保真载荷）——这些仍然会同时发送两个字段。

## 拼接规则

`concat_uni_events_to_uni_message` / `concatUniEventsToUniMessage` 现在以 `fidelity` 为判定依据：

- text：phase 变化会开启一个新的内容项（同 phase 以及无 phase 的增量会合并，遵循 GPT-5.5 的 `phase` 指南）；到来的保真载荷（例如一个思考签名）会并入当前开放内容项的 fidelity 并结束该内容项。
- thinking：到来的保真载荷会结束当前开放的内容项（Claude 的 signature 增量、GPT-5.5 的 reasoning 标记），而携带**相同**保真载荷的一串增量会拼接成一个内容项（OpenAI 兼容客户端逐增量打上的 `reasoning_field` 标记）。

## 兼容性

由早期版本记录的历史消息在内容项顶层携带 `signature` / `phase`；客户端不再读取这些字段。要回放旧的历史消息，需将每个内容项的 `signature`/`phase` 迁移到 `fidelity` 中（Claude/Gemini 为 `{"signature": ...}`，GPT-5.5 为从其 JSON 字符串解析出的 `{"id": ..., "encrypted_content": ...}`，GPT-5.5 的 text 项为 `{"phase": ...}`）。

## 测试

离线伪流式测试套件 `src_py/tests/test_reasoning_fidelity.py` 与 `src_ts/tests/reasoning-fidelity.test.ts` 覆盖了 `openai`、`glm5_1` 和 `kimi_k2_6` 客户端上的两种 reasoning 字段拼写、两个字段同时出现的歧义情形，以及无保真载荷时的回退路径。
