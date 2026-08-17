# Structured bilingual changelog entries, plus two rules for the dev skill

- **Date:** 2026-08-16
- **Type:** process
- **Scope:** `changelog`, `skills`
- **PR:** [#170](https://github.com/Prism-Shadow/agenthub/pull/170)

[中文版](2026-08-16-changelog-format-and-skill-rules.zh.md)

## What changed

- Every detail file opens with a metadata block — `Date`, `Type`, `Scope`, `PR`, `Issue`, `Breaking` — directly under the title, in that fixed order. All 29 entries carry it, and each release README line carries its PR link beside the details link.
- Bare `(#N)` references became real links. A bare `#N` renders as plain text in a Markdown file (GitHub autolinks it only in issue and PR comments), so those references were unusable as links.
- Unreleased work is pinned to `changelog/unreleased/`, a folder named for its state rather than a number, because the version is not decided until release.
- The body has a designed shape: short form for changes with nothing decided, full form (`## What changed`, `## Problem`, `## Decision`, `## Alternatives considered`, `## Verification`, `## Risks`, `## Compatibility`, `## Deferred`) for the rest, in a fixed order with a fixed vocabulary.
- Every file ships in both languages — `<name>.md` and `<name>.zh.md`, mirroring each other section for section: 29 detail entries, 8 release READMEs, the root `CHANGELOG.md`, and the convention doc.
- The `agenthub-dev` skill gained two rules. Stage 2: when part of a captured exchange will not serialize to JSON, save its `str()` form wrapped in a JSON object so the `.jsonl` stays parseable, instead of dropping the event or hand-editing it. Stage 4: the `AVAILABLE_MODELS` entry is the only test change a new model is entitled to; the shared e2e suites are contracts every client must already satisfy.

## Problem

The entries already held the right information in the wrong shape. PR attribution lived in prose when it existed at all, and nothing above the prose said what kind of change an entry was or which modules it touched, so an agent answering "when did `temperature` stop being accepted, and in which PR?" had to read entries end to end. Bodies had drifted too: four different heading names — `## Protocol findings`, `## Protocol implementation`, `## Implementation`, `## Implementation notes` — for the same idea, and no place at all for the evidence behind a change or the risks it shipped with. And the tree was English-only, in a repository whose reviews are conducted in Chinese.

## Decision

Three fixed parts and one flexible one. The metadata block is fixed and machine-checkable. The body vocabulary is fixed: a documented set of section names in a fixed order, with a short form for entries that decided nothing. The bilingual pairing is fixed: a file is unfinished without its counterpart. What stays flexible is prose length and bespoke sections, which is where a change's actual detail belongs.

The metadata block stays English in both languages — field names, `Type` values, `Scope` identifiers, dates, links — so one `grep` answers a question across the whole tree regardless of language. Only prose, headings, and the `Breaking` reason are translated, and the standard headings have fixed renderings so they stay greppable in Chinese too.

## Alternatives considered

- **Plain `Key: value` lines**, as the [deepseek-harness agent notes](https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/README.md) format that inspired this uses. Rejected: consecutive lines collapse into one run-on paragraph when rendered. A bullet list renders correctly, stays greppable by field name, and reads the same raw.
- **YAML frontmatter.** Rejected: it renders inconsistently across viewers, and the fields are worth reading in the document rather than hidden above it.
- **A separate index file** mapping entries to PRs, types, and scopes. Rejected: an index drifts from what it indexes. Keeping the fields next to the prose keeps them true.
- **Translating the metadata block too.** Rejected: it would split every query in two, so `grep -rl 'Breaking:' changelog/` would return half an answer.
- **Retrofitting the full body structure onto history.** Rejected: 21 of the 27 pre-existing entries were a title plus a bullet list. Rewriting their prose into a section scheme would churn the historical record for no gain; they received a `## What changed` heading and nothing else.

## Verification

- A format checker validates the whole tree: field order, `Date` matching the filename, `Type` in the closed set, every `PR`/`Issue` value a full link, no bare `#N` anywhere, `Breaking` entries carrying `## Compatibility`, every relative and `[details]` link resolving, `changelog/unreleased/` existing with no numbered folder claiming to be unreleased, and each pair matching on section count, metadata fields, and counterpart links. It passes on 39 pairs across 29 entries and 8 releases.
- The checker earned its place twice: it caught a `Breaking: yes` entry with no `## Compatibility` section, and its own first version reported three false positives by mistaking bold body bullets for metadata, which is why it now stops at the first blank line.

## Risks

- The checker lives outside the repository, so nothing enforces the convention on a PR that ignores it. Wiring it into CI is the obvious next step and is deliberately not part of this change.
- The heading vocabulary constrains only the standard sections. Bespoke names can drift again, the way they did before, and only review will notice.
- Translation quality is not machine-checkable. Structure, metadata, and links are verified mechanically; whether the Chinese says what the English says is not.

## Deferred

- Wiring the format checker into CI, so a malformed or unpaired entry fails the build rather than waiting for a reviewer.
