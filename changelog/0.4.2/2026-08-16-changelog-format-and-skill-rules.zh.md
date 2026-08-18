# 结构化的双语更新日志条目，以及若干开发 skill 规则

- **Date:** 2026-08-16
- **Type:** process
- **Scope:** `changelog`, `skills`
- **PR:** [#170](https://github.com/Prism-Shadow/agenthub/pull/170)

[English](2026-08-16-changelog-format-and-skill-rules.md)

## 变更内容

- 每个详情文件都以一个元数据块开头 —— `Date`、`Type`、`Scope`、`PR`、`Issue`、`Breaking` —— 紧接标题之下，并保持这一固定顺序。全部 29 个条目都带上了它，每个发布版本 README 中的行也在详情链接旁附带了 PR 链接。
- 裸写的 `(#N)` 引用改成了真正的链接，因为裸写的 `#N` 在 Markdown 文件里不会渲染成链接。
- 未发布的内容移到了 `changelog/unreleased/`：该目录以状态命名，而不是版本号。
- 正文记录这次变更做了什么：`## 变更内容`、它引入的事实性细节，以及在破坏了什么时的 `## 兼容性`。推理过程 —— 权衡过的问题、否决的备选、收集的证据、接受的风险 —— 在对话和 PR 描述中汇报，而不落盘；条目也不描述代码库的当前状态。
- 每个文件都有两种语言：`<name>.md` 与 `<name>.zh.md`，逐节互相对应，覆盖 29 个详情条目、8 个发布版本 README、根目录 `CHANGELOG.md` 以及约定文档。元数据块在两种语言里都保持英文，这样一条 `grep` 就能覆盖整棵树。
- `changelog/README.md` 记录了该格式，`agenthub-dev` skill 指向它。

## 新增的开发 skill 规则

- Stage 2：当抓取到的交互中有部分内容无法序列化为 JSON 时，保存其 `str()` 形式并用一个 JSON 对象包裹，使 `.jsonl` 仍可解析，而不是丢弃该事件或手工改写它。
- Stage 4：`AVAILABLE_MODELS` 条目是新模型唯一有权做出的测试改动；对共享 e2e 测试套件更大范围的修改需要逐次获得明确批准。
- 记录与交付：一次变更背后的推理在对话中汇报并写进 PR 描述，而不是写进更新日志。
