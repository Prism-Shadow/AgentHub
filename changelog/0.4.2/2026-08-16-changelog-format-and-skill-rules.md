# Structured changelog entries, plus two rules for the dev skill

- **Date:** 2026-08-16
- **Type:** process
- **Scope:** `changelog`, `skills`
- **PR:** [#170](https://github.com/Prism-Shadow/agenthub/pull/170)

[中文版](2026-08-16-changelog-format-and-skill-rules.zh.md)

## What changed

- Every changelog detail file now opens with a metadata block — `Date`, `Type`, `Scope`, `PR`, `Issue`, `Breaking` — directly under the title, in that fixed order, so a reader and a `grep` find each field in the same place. All 27 existing entries were migrated, and each release README line now carries its PR link next to the details link.
- Bare `(#N)` references inside entry prose became real links. A bare `#N` renders as plain text in a Markdown file (GitHub only autolinks it in issue and PR comments), so those references were unusable as links for both readers and agents.
- `changelog/README.md` documents the format: the template to copy, a table defining every field, body rules, and the `grep` recipes that answer the common questions ("what has ever broken compatibility?", "every entry touching a client").
- Every file in the tree now ships in both languages — `<name>.md` and `<name>.zh.md`, mirroring each other section for section: 28 detail entries, 8 release READMEs, the root `CHANGELOG.md`, and the convention doc itself. Each file links its counterpart directly below the metadata block.
- The `agenthub-dev` skill gained two rules. Stage 2: when part of a captured exchange will not serialize to JSON, save its `str()` form and analyze that — wrapped in a JSON object so the `.jsonl` stays parseable — instead of dropping the event, hand-editing it into valid JSON, or falling back to the docs. Stage 4: the `AVAILABLE_MODELS` entry is the only test change a new model is entitled to — the shared e2e suites are contracts every client must already satisfy, and any broader edit needs explicit approval, per change.

## Why

- The entries already held the right information but not in a findable shape: PR attribution lived in prose when it existed at all, and nothing above the prose said what kind of change an entry was or which modules it touched. An agent answering "when did `temperature` stop being accepted, and in which PR?" had to read entries end to end. The metadata block makes that a `grep`, and keeps the answer next to the human-readable record rather than in a separate index that would drift.
- Plain `Key: value` lines were the first choice, following the [deepseek-harness agent notes](https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/README.md) format that inspired this, but consecutive lines collapse into one run-on paragraph when rendered. A bullet list renders correctly, stays greppable by field name, and reads the same in raw form.
- Optional fields are omitted rather than filled with a placeholder, which is what makes `grep -rl 'Breaking:' changelog/` an exact query for breaking changes.
- Two of the oldest entries (0.2.0) carry no PR link: they predate the convention and were regrouped wholesale by [#162](https://github.com/Prism-Shadow/agenthub/pull/162), so no single PR can be attributed to them without guessing. `Breaking: yes` is likewise recorded only where the entry itself documents a break — the client classes renamed by several model refreshes were never exported from `__init__.py`, so those renames broke no user code.
- Translating the metadata block would have split every query in two. Field names, `Type` values, `Scope` identifiers, dates, and links stay English verbatim, so `grep -rl 'Breaking:' changelog/` still returns the complete set whichever language a file is in; only prose and headings are translated, and the standard headings have fixed renderings (`## 变更内容`, `## 原因`, `## 兼容性`) so they stay greppable too.
- Section structure stayed a recommendation rather than a retrofit: 21 of the 27 entries were a title plus a bullet list, and rewriting their prose into a fixed section scheme would have churned the historical record for no gain. They received a `## What changed` heading and nothing else.
