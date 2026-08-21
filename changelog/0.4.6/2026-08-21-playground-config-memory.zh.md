# Playground 刷新后保留配置

- **Date:** 2026-08-21
- **Type:** feature
- **Scope:** `integration`, `tests`
- **PR:** [#184](https://github.com/Prism-Shadow/agenthub/pull/184)

[English](2026-08-21-playground-config-memory.md)

## 变更内容

- Playground 每次改动都会把配置面板写进 `localStorage` 的 `agenthub.playground.config`，页面加载时
  再读回来：model、client type、API key、base URL、extra headers、thinking level、thinking
  summary、tool choice、system prompt、tools 与 trace ID。文本框按输入的原样保存，没写完的 JSON
  也会原样回来。
- 下拉列表里没有的模型，无论是 **List models** 拉出来的还是手填的自定义 id，都会以自定义条目的形式
  恢复，并带回它的 client type。
- API key 与面板其余内容一起保存，因此它会留在打开 Playground 的那个浏览器配置文件里，直到该输入框
  被清空。
- 浏览器若禁用 `localStorage`，Playground 照常可用，只是每次都从默认值开始。
