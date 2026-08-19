# MiniMax 的路由分支移入 client type 判断链

- **Date:** 2026-08-19
- **Type:** refactor
- **Scope:** `auto_client`
- **PR:** [#174](https://github.com/Prism-Shadow/agenthub/pull/174)

[English](2026-08-19-minimax-routing-order.md)

## 变更内容

- `auto_client.py` 与 `autoClient.ts` 把 `client_type == "minimax-m3"` 的判断从整条判断链之前
  移到 Kimi 分支之后，使 MiniMax 分支与其他模型家族的分支排在一起。判断条件本身未变，且没有其他
  分支会匹配 `minimax-m3`，因此所有 client type 的路由结果与此前一致。
