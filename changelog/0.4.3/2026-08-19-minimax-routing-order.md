# MiniMax routing moved into the client-type chain

- **Date:** 2026-08-19
- **Type:** refactor
- **Scope:** `auto_client`
- **PR:** [#174](https://github.com/Prism-Shadow/agenthub/pull/174)

[中文版](2026-08-19-minimax-routing-order.zh.md)

## What changed

- `auto_client.py` and `autoClient.ts` test `client_type == "minimax-m3"` after the Kimi
  branch instead of ahead of the whole chain, so the MiniMax branch sits with the other
  model-family branches. The test itself is untouched, and no other branch matches
  `minimax-m3`, so every client type routes to the same client as before.
