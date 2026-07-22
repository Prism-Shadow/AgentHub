> Fetch the complete documentation index at: https://platform.kimi.com/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# 模型参数参考

export const DocTable = ({columns = [], rows = []}) => {
  return <div className="doc-table-wrap">
      <table className="doc-table">
        {columns.length > 0 ? <colgroup>
            {columns.map((column, index) => <col key={index} style={column.width ? {
    width: column.width
  } : undefined} />)}
          </colgroup> : null}
        <thead>
          <tr>
            {columns.map((column, index) => <th key={index}>{column.title}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => <tr key={rowIndex}>
              {row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}
            </tr>)}
        </tbody>
      </table>
    </div>;
};

不同模型系列对 Chat Completions API 参数有不同的默认值和约束。完整的模型列表请参阅[模型列表](/docs/models)。

## 参数对比

<DocTable
  columns={[
{ title: "参数", width: "20%" },
{ title: "kimi-k3", width: "20%" },
{ title: "kimi-k2.7-code 系列", width: "20%" },
{ title: "kimi-k2.6", width: "20%" },
{ title: "moonshot-v1 系列", width: "20%" },
]}
  rows={[
[<code>temperature</code>, <strong>不可修改</strong>, <strong>不可修改</strong>, <strong>不可修改</strong>, "0.0"],
[<code>top_p</code>, <>0.95 <strong>不可改</strong></>, <>0.95 <strong>不可改</strong></>, <>0.95 <strong>不可改</strong></>, "1.0"],
[<code>n</code>, <>1 <strong>不可改</strong></>, <>1 <strong>不可改</strong></>, <>1 <strong>不可改</strong></>, "1（最大 5）"],
[<code>presence_penalty</code>, <>0 <strong>不可改</strong></>, <>0 <strong>不可改</strong></>, <>0 <strong>不可改</strong></>, "0（可修改）"],
[<code>frequency_penalty</code>, <>0 <strong>不可改</strong></>, <>0 <strong>不可改</strong></>, <>0 <strong>不可改</strong></>, "0（可修改）"],
[<>推理配置</>, <code>reasoning_effort</code>, <code>thinking</code>, <code>thinking</code>, "—"],
]}
/>

<Note>
  当 `temperature` 接近 0 时，`n` 只能为 1，否则将返回 `invalid_request_error`。
</Note>

## 模型参数配置差异

切换模型时，除了替换 `model` 字段，还需要注意各模型对请求参数的支持范围和默认值不同：

| 参数                                       | `kimi-k3`                                | `kimi-k2.7-code`                               | `kimi-k2.6`                                                                      | `kimi-k2.5`                                    |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------- |
| 上下文窗口                                    | 1M tokens                                | 256K tokens                                    | 256K tokens                                                                      | 256K tokens                                    |
| `thinking`                               | —                                        | 可省略；显式设置时仅接受 `{"type":"enabled","keep":"all"}` | `{"type":"enabled"}`（默认）、`{"type":"disabled"}`、`{"type":"enabled","keep":"all"}` | `{"type":"enabled"}`（默认）、`{"type":"disabled"}` |
| `reasoning_effort`                       | `"low"` / `"high"` / `"max"`（默认 `"max"`） | 不支持                                            | 不支持                                                                              | 不支持                                            |
| `tool_choice`                            | `auto` / `none` / `required`             | 不支持 `required`                                 | 不支持 `required`                                                                   | —                                              |
| `temperature`                            | 固定 `1.0`                                 | 固定 `1.0`                                       | 思考 `1.0` / 非思考 `0.6`                                                             | 思考 `1.0` / 非思考 `0.6`                           |
| `top_p`                                  | 固定 `0.95`                                | 固定 `0.95`                                      | 固定 `0.95`                                                                        | —                                              |
| `n`                                      | 固定 `1`                                   | 固定 `1`                                         | 固定 `1`                                                                           | —                                              |
| `presence_penalty` / `frequency_penalty` | 固定 `0`                                   | 固定 `0`                                         | 固定 `0`                                                                           | —                                              |

<Note>
  表中"固定"表示该参数不可修改：传入其他值会报错，建议不要显式传入。
</Note>

### `thinking`

`thinking` 是 K2.x 专属请求参数：

* `kimi-k2.6`：支持 `{"type": "enabled"}`（默认）、`{"type": "disabled"}`、`{"type": "enabled", "keep": "all"}` 三种配置。
* `kimi-k2.7-code`：思考默认开启，仅支持 `{"type": "enabled", "keep": "all"}`，传入其他配置会报错。从 `kimi-k2.6` 切换时，需要按 Preserved Thinking 的要求在 `messages` 中回传历史 `reasoning_content`。

详见[使用思考模式](/docs/guide/use-kimi-k2-thinking-model)。

### `reasoning_effort`

K3 始终进行推理思考且保留式思考（Preserved Thinking）始终开启。通过请求顶层 `reasoning_effort` 配置推理力度，支持 `"low"` / `"high"` / `"max"` 三档，默认 `"max"`。详见[思考力度](/docs/guide/use-thinking-effort)。

<Warning>
  切换档位会破坏前缀缓存命中，建议在会话开始前确定 `effort` 档位，避免中途切换。
</Warning>

### `tool_choice`

`kimi-k3` 支持 `auto` / `none` / `required` 三档；`kimi-k2.6` 与 `kimi-k2.7-code` 不支持 `required`，传入会报错。详见[工具调用约束](/docs/guide/use-tool-choice)。

### `temperature`

* `kimi-k2.6` / `kimi-k2.5`：思考模式固定 `1.0`，非思考模式固定 `0.6`，传入其他值报错；
* `kimi-k2.7-code`：固定 `1.0`，传入其他值报错。
* `kimi-k3`：固定 `1.0`，传入其他值报错。

建议调用以上模型时不要显式传入 `temperature`。

`kimi-k2.7-code-highspeed` 与 `kimi-k2.7-code` 为同一模型、参数约束完全一致，仅输出速度不同。

### 常见问题

**从 `kimi-k2.6` 切换到 `kimi-k3`，需要改代码吗？**

将 `model` 替换为 `kimi-k3`，并移除 K2.x 的 `thinking` 配置；如需显式设置推理力度，使用顶层 `reasoning_effort`。K3 的多轮对话和工具调用需要把 API 返回的完整 assistant message 原样回传到 `messages`，包括可能返回的 `reasoning_content`。

**从 `kimi-k2.7-code` 切换到 `kimi-k3`，需要改代码吗？**

替换 `model` 即可，并继续原样回传完整 assistant message；如需显式设置推理力度，使用顶层 `reasoning_effort`。

**原来代码里用的是 OpenAI 的 `reasoning_effort`，切到 `kimi-k3` 需要改吗？**

不需要。K3 支持顶层 `reasoning_effort`，可选值为 `"low"` / `"high"` / `"max"`，默认 `"max"`。

**`tool_choice: "required"` 在 `kimi-k2.6` / `kimi-k2.7-code` 上能用吗？**

不能。这两个模型不支持 `required`，传入会报错；该档位仅 `kimi-k3` 支持。

## Kimi K2.7 Code 系列 — thinking 参数

`kimi-k2.7-code` 系列包含 `kimi-k2.7-code` 及其高速版 `kimi-k2.7-code-highspeed`，二者为同一模型、参数约束完全一致（含上方表格与 `thinking` 行为），仅输出速度不同，下文统称 `kimi-k2.7-code`。

`kimi-k2.7-code` 面向代码场景，除 `thinking` 外的参数约束与 `kimi-k2.6` 完全一致。与 `kimi-k2.6` 不同的是，它 **始终开启思考、不可禁用**（传入 `{"type": "disabled"}` 会报错），且 **Preserved Thinking 始终开启**（`thinking.keep` 不传或传 `"all"` 都按 `"all"` 处理，传入其他非法值会报错）。因此调用时无需传入 `thinking` 参数，只需切换 `model` 即可，模型始终输出 `reasoning_content`。详细用法见[使用思考模式](/docs/guide/use-kimi-k2-thinking-model)。

## Kimi K2.6 — thinking 参数

Kimi K2.6 支持通过 `thinking` 参数控制是否启用深度思考。接受 `{"type": "enabled"}` 或 `{"type": "disabled"}`。

由于 OpenAI SDK 没有原生的 `thinking` 参数，需要使用 `extra_body` 传递：

<CodeGroup>
  ```python Python theme={null}
  completion = client.chat.completions.create(
      model="kimi-k2.6",
      messages=[
          {"role": "user", "content": "你好"}
      ],
      extra_body={
          "thinking": {"type": "disabled"}
      },
      max_tokens=1024*32,
  )
  ```

  ```bash cURL theme={null}
  curl https://api.moonshot.cn/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $MOONSHOT_API_KEY" \
    -d '{
      "model": "kimi-k2.6",
      "messages": [
        {"role": "user", "content": "你好"}
      ],
      "thinking": {"type": "disabled"}
    }'
  ```
</CodeGroup>
