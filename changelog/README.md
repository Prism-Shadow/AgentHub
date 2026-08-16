# Changelog Details

[中文版](README.zh.md)

Release history lives in three levels, each one link away from the next:

1. [`../CHANGELOG.md`](../CHANGELOG.md) — one line per release.
2. `<version>/README.md` — one line per change in that release, linking its detail file and its PR.
3. `<version>/YYYY-MM-DD-<slug>.md` — one detail file per change: what changed, why, and what it cost.

Unreleased work goes into the upcoming version's folder. At release preparation, rename the folder if the number changed and add the release line to the root file.

## Detail file format

Copy this template:

```markdown
# <Title: name the change, not the PR>

- **Date:** YYYY-MM-DD
- **Type:** feature
- **Scope:** `module`, `module`
- **PR:** [#N](https://github.com/Prism-Shadow/agenthub/pull/N)
- **Issue:** [#N](https://github.com/Prism-Shadow/agenthub/issues/N)
- **Breaking:** yes — <what breaks, in one line>

## What changed

- ...

## Why

...
```

The metadata block sits directly under the H1, one field per bullet, always in the order above, so a reader and a `grep` find each field in the same place.

| Field | Required | Value |
| --- | --- | --- |
| `Date` | always | The entry date, matching the filename prefix. |
| `Type` | always | Exactly one of `feature` (new model or user-facing capability), `fix` (defect correction), `refactor` (restructuring with no capability change), `process` (tooling, docs, skills, CI, release plumbing). |
| `Scope` | always | 1–5 code areas, backticked, named as the tree names them (`gemini3_7`, `registry`, `auto_client`, `tests`, `skills`, …). |
| `PR` | when one exists | Full link to the PR(s) that shipped this change. A bare `#N` does not render as a link in a Markdown file, so always write the URL — in the body too. A PR mentioned in the prose for context (an earlier change being corrected, say) is a cross-reference, not a shipping PR, and stays inline. |
| `Issue` | when one exists | Full link, same rule. Multiple links are comma-separated. |
| `Breaking` | only when breaking | `yes — <one line>`. Omit the field entirely otherwise, so `grep -rl 'Breaking:' changelog/` lists exactly the breaking changes. |

Omit a field that does not apply rather than writing a placeholder. A change with no PR or issue (pre-convention entries, or work landed outside a PR) simply has no such line.

### Body

The metadata block is the fixed part; the body is as long as the change deserves.

- The body opens with a `##` section — `## What changed`, unless a more specific name fits the change (`## Protocol findings`, `## The bug`). Keep it to bullets, each led by the model id, class, or parameter a reader would search for. A small change needs nothing more.
- A change with findings or decisions behind it adds `## Why`: what the code cannot carry — protocol differences found, config mappings chosen, alternatives rejected and the reason. `## Why` is the part still worth reading a year later. Add bespoke sections as the change needs them (`## Configuration behavior`, `## Registry metadata`, `## Verification`).
- `## Compatibility` is required whenever `Breaking: yes`: what breaks, and the migration.

Cross-reference other entries with relative links, e.g. `[Gemini 3.7](../0.4.2/2026-08-13-gemini-3.7.md)`; relative links survive the folder rename at release time and can be checked mechanically.

## Chinese counterpart

Every file in this tree ships in both languages: `<name>.md` in English and `<name>.zh.md` in Chinese, mirroring it section for section. That holds for detail entries, release READMEs, and the root `CHANGELOG.md`. A change is not complete until both exist — write the English file first, then the counterpart, in the same PR.

What stays in English, verbatim, so one `grep` works across both languages:

- The metadata field names and their values — `- **Type:** feature`, the `Scope` identifiers, the `Date`, and the links. Only the `Breaking` reason is prose, so only it is translated.
- Code identifiers, model ids, parameter names, error classes, and file paths.

What gets translated: all prose, and the section headings. Use these renderings for the standard headings, so a Chinese reader can `grep` them just as reliably:

| English | 中文 |
| --- | --- |
| `## What changed` | `## 变更内容` |
| `## Why` | `## 原因` |
| `## Compatibility` | `## 兼容性` |

Bespoke headings are translated naturally, keeping the same order and count as the English file. Each file links its counterpart on the line directly below the metadata block: `[中文版](<name>.zh.md)` in the English file, `[English](<name>.md)` in the Chinese one.

## Release README format

One line per change, newest first:

```markdown
- [YYYY-MM-DD] One-sentence description. ([details](YYYY-MM-DD-slug.md), [#N](https://github.com/Prism-Shadow/agenthub/pull/N))
```

The PR link is repeated here on purpose: the release summary should answer "which PR shipped this" without opening the detail file.

## Finding things

| Question | Query |
| --- | --- |
| What shipped in a release? | `cat changelog/<version>/README.md` |
| What has ever broken compatibility? | `grep -rl 'Breaking:' changelog/` |
| Every entry touching a client | `grep -rl 'Scope:.*gemini' changelog/` |
| Only bug fixes | `grep -rl 'Type:\*\* fix' changelog/` |
| Which PR shipped an entry | `grep 'PR:' changelog/<version>/<entry>.md` |
