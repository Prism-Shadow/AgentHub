> Fetch the complete documentation index at: https://platform.kimi.com/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# 旗舰模型 Kimi K3 定价

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

## 产品定价

<DocTable
  columns={[
{ title: "模型", width: "24%" },
{ title: "计费单位", width: "12%" },
{ title: "输入价格（缓存命中）", width: "16%" },
{ title: "输入价格（缓存未命中）", width: "16%" },
{ title: "输出价格", width: "14%" },
{ title: "上下文窗口", width: "18%" },
]}
  rows={[
["kimi-k3", "1M tokens", "¥2.00", "¥20.00", "¥100.00", "1,048,576 tokens"],
]}
/>

此处 1M = 1,000,000，表格中的价格代表每消耗 1M tokens 的价格。

## 模型说明

<Warning>
  联网搜索（`web_search`）正在更新升级中，近期不建议使用该功能，当前文档已经过时，请关注后续内容更新。
</Warning>

* Kimi K3 是 Kimi 的旗舰模型，面向长程编程与端到端知识工作，1M token 上下文，综合智能达到领先水平，详见 [Kimi K3 模型介绍](/docs/guide/kimi-k3-quickstart)
* 始终进行推理，支持通过请求顶层 `reasoning_effort` 配置推理力度（`low` / `high` / `max`，默认 `max`），详见[思考力度](/docs/guide/use-thinking-effort)
* 支持[自动上下文缓存](/docs/guide/use-context-caching-feature-of-kimi-api)、[工具调用（ToolCalls）](/docs/guide/use-kimi-api-to-complete-tool-calls)、[JSON Mode](/docs/guide/use-json-mode-feature-of-kimi-api)、[结构化输出（`response_format` / JSON Schema）](/docs/guide/response_format)、[Partial Mode](/docs/guide/use-partial-mode-feature-of-kimi-api)、[联网搜索](/docs/guide/use-web-search)等能力
* K3 新增 API 能力：[工具调用约束（`tool_choice`）](/docs/guide/use-tool-choice)、[动态加载工具](/docs/guide/use-dynamic-tool-loading)，组合用法见 [K3 工具调用最佳实践](/docs/guide/kimi-k3-tool-calling-best-practice)
