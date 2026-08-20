# Playground 面板按分区归类，并默认以 debug 模式启动

- **Date:** 2026-08-20
- **Type:** feature
- **Scope:** `integration`, `tests`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[English](2026-08-20-playground-panel.md)

## 变更内容

- 配置面板从两块无标题的三列网格改为两个带标题的分区。**Connection** 放 Model、API Key、Base URL
  与 Extra Headers；**Generation** 放 Thinking Level、Thinking Summary、Tool Choice、Trace ID，
  再由 System Prompt 与 Tools 各占两列。两个分区共用同一套 `1 / 2 / 4` 列网格，因此每一行都排满，
  不会拖出参差的尾巴。
- `start_playground_server` / `startPlaygroundServer` 把 `AGENTHUB_DEBUG` 默认设为 `1`。Playground
  的用途正是看清模型与 endpoint 到底发了什么，所以未知的流式输出在这里会抛出而不是被跳过；环境中
  已有的 `AGENTHUB_DEBUG` 仍然优先。
