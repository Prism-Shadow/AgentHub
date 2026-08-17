# Changelog 详情

[English](README.md)

发布历史分三层，每层都只隔一个链接：

1. [`../CHANGELOG.zh.md`](../CHANGELOG.zh.md) —— 每个发布版本一行。
2. `<version>/README.zh.md` —— 该版本里每项变更一行，链接到它的详情文件和 PR。
3. `<version>/YYYY-MM-DD-<slug>.zh.md` —— 每项变更一个详情文件：改了什么、为什么这么改、代价是什么。

尚未发布的内容统一放进同一个目录：README 标题带 `(unreleased)` 标记的那个唯一的发布版本目录，用 `grep -l '(unreleased)' changelog/*/README.md` 即可找到。一律加进该目录。不要为尚未发布的工作另开第二个目录，也不要自行臆测下一个版本号 —— 发布准备阶段如果版本号最终不同就重命名该目录、去掉 `(unreleased)` 标记，并把该发布版本的一行加进根文件。

## 详情文件格式

复制这个模板：

```markdown
# <标题：写清这次变更，而不是 PR>

- **Date:** YYYY-MM-DD
- **Type:** feature
- **Scope:** `module`, `module`
- **PR:** [#N](https://github.com/Prism-Shadow/agenthub/pull/N)
- **Issue:** [#N](https://github.com/Prism-Shadow/agenthub/issues/N)
- **Breaking:** yes — <一句话说明坏在哪>

## 变更内容

- ...

## 原因

...
```

元数据块紧贴在 H1 下面，一个字段一个要点，顺序固定如上，这样读者和 `grep` 都能在同一个位置找到同一个字段。

| 字段 | 是否必需 | 取值 |
| --- | --- | --- |
| `Date` | 必需 | 条目日期，与文件名前缀一致。 |
| `Type` | 必需 | 只能是以下之一：`feature`（新模型或面向用户的新能力）、`fix`（缺陷修复）、`refactor`（不改变能力的结构调整）、`process`（工具链、文档、skill、CI、发布流程）。 |
| `Scope` | 必需 | 1–5 个代码区域，用反引号包起来，名称与代码树中的一致（`gemini3_7`、`registry`、`auto_client`、`tests`、`skills` 等）。 |
| `PR` | 存在时必填 | 指向发布该变更的 PR 的完整链接。裸写 `#N` 在 Markdown 文件里不会渲染成链接，所以必须写完整 URL —— 正文里也一样。正文中为提供背景而提到的其他 PR（比如被修正的早前变更）属于交叉引用，不是发布该变更的 PR，留在原地即可。 |
| `Issue` | 存在时必填 | 完整链接，规则同上。多个链接用逗号分隔。 |
| `Breaking` | 仅破坏性变更时 | `yes — <一句话>`。其余情况整行省略，这样 `grep -rl 'Breaking:' changelog/` 列出的正好就是所有破坏性变更。 |

不适用的字段直接省略，不要填占位符。没有 PR 或 issue 的变更（早于该约定的条目，或未经 PR 落地的工作）就是没有那一行。

### 正文

元数据块是固定部分；正文该多长就多长。

- 正文以一个 `##` 小节开头 —— 通常是 `## 变更内容`，除非有更贴切的名字（`## 协议发现`、`## 问题所在`）。内容保持为要点，每条以读者会去搜的模型 id、类名或参数开头。小改动到此为止即可。
- 背后有发现或决策的变更再加 `## 原因`：写代码本身承载不了的东西 —— 发现的协议差异、选定的配置映射、被否掉的备选方案及其理由。`## 原因` 才是一年后仍值得读的部分。按需增加专门小节（`## 配置行为`、`## 注册表元数据`、`## 验证`）。
- 只要 `Breaking: yes`，就必须有 `## 兼容性`：坏了什么，以及如何迁移。

引用其他条目时使用相对链接，例如 `[Gemini 3.7](../0.4.2/2026-08-13-gemini-3.7.zh.md)`；相对链接能在发布时的目录重命名后继续有效，也便于机械校验。

## 中英双语对应

本目录下每个文件都有两种语言：英文 `<name>.md` 与中文 `<name>.zh.md`，逐节对应。详情条目、发布版本 README、根目录 `CHANGELOG.md` 都适用。两个文件都存在，这项变更才算完成 —— 先写英文，再在同一个 PR 里写对应的中文。

以下内容原样保留英文，这样一条 `grep` 在两种语言下都能用：

- 元数据字段名及其取值 —— `- **Type:** feature`、`Scope` 里的标识符、`Date`、以及各个链接。只有 `Breaking` 后面那句是散文，因此只翻译它。
- 代码标识符、模型 id、参数名、错误类名、文件路径。

需要翻译的：全部散文，以及小节标题。标准小节标题固定按下表翻译，这样中文读者也能同样可靠地 `grep`：

| English | 中文 |
| --- | --- |
| `## What changed` | `## 变更内容` |
| `## Why` | `## 原因` |
| `## Compatibility` | `## 兼容性` |

专门小节的标题自然翻译即可，但顺序和数量必须与英文文件一致。每个文件在元数据块正下方链接到它的对应版本：英文文件里写 `[中文版](<name>.zh.md)`，中文文件里写 `[English](<name>.md)`。

## 发布版本 README 格式

每项变更一行，最新的在最上面：

```markdown
- [YYYY-MM-DD] 一句话描述。([详情](YYYY-MM-DD-slug.zh.md), [#N](https://github.com/Prism-Shadow/agenthub/pull/N))
```

这里重复 PR 链接是刻意的：读发布摘要时不打开详情文件也能回答"这是哪个 PR 发布的"。

## 怎么查

| 问题 | 查询 |
| --- | --- |
| 某个发布版本包含什么？ | `cat changelog/<version>/README.zh.md` |
| 历史上有哪些破坏性变更？ | `grep -rl 'Breaking:' changelog/` |
| 所有涉及某个客户端的条目 | `grep -rl 'Scope:.*gemini' changelog/` |
| 只看缺陷修复 | `grep -rl 'Type:\*\* fix' changelog/` |
| 某个条目由哪个 PR 发布 | `grep 'PR:' changelog/<version>/<entry>.md` |
