# Changelog Details

[中文版](README.zh.md)

Release history lives in three levels, each one link away from the next:

1. [`../CHANGELOG.md`](../CHANGELOG.md) — one line per release.
2. `<version>/README.md` — one line per change in that release, linking its detail file and its PR.
3. `<version>/YYYY-MM-DD-<slug>.md` — one detail file per change: what changed, why, and what it cost.

Everything that has not shipped goes into `unreleased/`, which is named for its state rather than a number because the version is not decided until release. Never create a numbered folder for unshipped work and never invent the next version number. At release preparation, `unreleased/` is renamed to the decided version, its README title becomes that version, and the release line is added to the root file; the next change recreates `unreleased/`.

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

[中文版](<name>.zh.md)

## What changed

- ...

## Problem

...

## Decision

...

## Alternatives considered

- ...

## Verification

- ...
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

An entry takes one of two shapes.

**Short form** — the metadata block and `## What changed`, nothing else. Correct when nothing was decided: a model registration, a dependency bump, a typo. Most historical entries are this shape.

**Full form** — the decision record, for any change where someone weighed something. Sections appear in this order; skip the ones that do not apply, never reorder the ones that do.

| Section | When | What belongs in it |
| --- | --- | --- |
| `## What changed` | always | The user-visible outcome, in bullets, each led by the model id, class, or parameter a reader would search for. Not the implementation story. |
| `## Problem` | full form | What was wrong or missing, stated as its consequence: "strict upstreams reject the spelling they did not emit, breaking multi-turn conversations", not "reasoning field handling was wrong". |
| `## Decision` | full form | What the code does now and why this shape. Present tense, and kept true as the code evolves — this is the section a future change has to update or supersede. |
| `## Alternatives considered` | full form | The options genuinely weighed and the reason each lost. "None" is not an answer: if nothing was weighed, the entry is short form. |
| bespoke | as needed | Protocol tables, config mappings, registry metadata — the detail that supports the decision. Reuse an existing name (`## Configuration behavior`, `## Registry metadata`) before inventing one. |
| `## Verification` | any client or protocol change | The evidence, named: which capture, which probe and what it returned, which e2e ran, what the live API actually did. "Tested" is not evidence. |
| `## Risks` | when evidence is incomplete | What a future reader should distrust: assumptions never checked against the live API, provider behavior that may change, coverage deliberately skipped. |
| `## Compatibility` | `Breaking: yes` | What breaks, and the migration. |
| `## Deferred` | optional | Known follow-ups, recorded so they are not rediscovered later as bugs. |

`## Proposal`, `## Plan`, and `## Acceptance criteria` never appear: an entry records work that shipped, not work that was intended.

Cross-reference other entries with relative links, e.g. `[Gemini 3.7](../0.4.1/2026-07-22-kimi-k3-gemini-3-6-registry.md)`; relative links survive the folder rename at release time and can be checked mechanically.

## Chinese counterpart

Every file in this tree ships in both languages: `<name>.md` in English and `<name>.zh.md` in Chinese, mirroring it section for section. That holds for detail entries, release READMEs, and the root `CHANGELOG.md`. A change is not complete until both exist — write the English file first, then the counterpart, in the same PR.

What stays in English, verbatim, so one `grep` works across both languages:

- The metadata field names and their values — `- **Type:** feature`, the `Scope` identifiers, the `Date`, and the links. Only the `Breaking` reason is prose, so only it is translated.
- Code identifiers, model ids, parameter names, error classes, and file paths.

What gets translated: all prose, and the section headings. Use these renderings for the standard headings, so a Chinese reader can `grep` them just as reliably:

| English | 中文 |
| --- | --- |
| `## What changed` | `## 变更内容` |
| `## Problem` | `## 问题` |
| `## Decision` | `## 决策` |
| `## Alternatives considered` | `## 备选方案权衡` |
| `## Verification` | `## 验证` |
| `## Risks` | `## 风险` |
| `## Compatibility` | `## 兼容性` |
| `## Deferred` | `## 待办` |

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
