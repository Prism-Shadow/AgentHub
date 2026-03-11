# Changelog

## 2026-03-11

### Migrated to GPT-5.4

- Replaced GPT-5.1 and GPT-5.2 client implementations with a new GPT-5.4 client (`GPT5_4Client`) in both Python and TypeScript.
- Updated `auto_client.py` and `autoClient.ts` to route `gpt-5.4` model identifiers to the new client.
- Added `Phase` type (`"commentary"` | `"final_answer"`) to `UniMessage` and `UniEvent` in both Python (`types.py`) and TypeScript (`types.ts`).
- Modified `transform_model_output_to_uni_event` to detect the `phase` field on message output items from `response.output_item.added` events and propagate the phase to subsequent streaming events.
- Updated `transform_uni_message_to_model_input` to include the `phase` field at the same level as `role` when replaying assistant history, as required by the GPT-5.4 API.
- Overrode `streaming_response_stateful` in `GPT5_4Client` to split accumulated events by phase: when two consecutive output items carry different phase values, they are stored as separate `UniMessage` entries in history rather than being concatenated, preserving model quality on multi-step tasks.
- Updated test files to use `gpt-5.4` as the default OpenAI model name.
