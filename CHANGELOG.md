# Changelog

## 2026-03-11

### Migrated to GPT-5.4

- Replaced the GPT-5.1/5.2 implementation with a new GPT-5.4 client (`GPT5_4Client`) in both Python and TypeScript.
- The routing in `AutoLLMClient` now matches `gpt-5.4` model identifiers; support for `gpt-5.1` and `gpt-5.2` has been removed.
- Added `phase` field to `TextContentItem` to support GPT-5.4's multi-phase output (`commentary` and `final_answer`).
  - Phase is tracked from `response.output_item.added` events and injected into streaming text delta events.
  - Text content items from different phases are stored separately (not concatenated) so model context is preserved correctly.
  - When converting history back to API input, each phase group is emitted as a separate message entry with `phase` at the same level as `role`, as required by the GPT-5.4 API.
