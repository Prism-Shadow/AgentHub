# Structured bilingual changelog entries, plus dev-skill rules

- **Date:** 2026-08-16
- **Type:** process
- **Scope:** `changelog`, `skills`
- **PR:** [#170](https://github.com/Prism-Shadow/agenthub/pull/170)

[中文版](2026-08-16-changelog-format-and-skill-rules.zh.md)

## What changed

- Every detail file opens with a metadata block — `Date`, `Type`, `Scope`, `PR`, `Issue`, `Breaking` — directly under the title, in that fixed order. All 29 entries carry it, and each release README line carries its PR link beside the details link.
- Bare `(#N)` references became real links, since a bare `#N` does not render as a link in a Markdown file.
- Unreleased work moved to `changelog/unreleased/`, a folder named for its state rather than a version number.
- The body records what a change did: `## What changed`, the factual detail it introduced, and `## Compatibility` when it breaks something. Reasoning — problems weighed, alternatives rejected, evidence gathered, risks accepted — is reported in the conversation and the PR description instead of on disk, and entries do not describe the current state of the codebase.
- Every file ships in both languages: `<name>.md` and `<name>.zh.md`, mirroring each other section for section, across 29 detail entries, 8 release READMEs, the root `CHANGELOG.md`, and the convention doc. The metadata block stays English in both so one `grep` covers the tree.
- `changelog/README.md` documents the format, and the `agenthub-dev` skill points at it.

## Dev skill rules added

- Stage 2: when part of a captured exchange will not serialize to JSON, save its `str()` form wrapped in a JSON object so the `.jsonl` stays parseable, rather than dropping the event or hand-editing it.
- Stage 4: the `AVAILABLE_MODELS` entry is the only test change a new model is entitled to; broader edits to the shared e2e suites need explicit approval, per change.
- Record and ship: the reasoning behind a change is reported in the conversation and written into the PR description, not into the changelog.
