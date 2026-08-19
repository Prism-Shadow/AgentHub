# 0.4.3

[中文版](README.zh.md)

- [2026-08-19] The e2e suites run one client per worker, so distinct models are tested in parallel. ([details](2026-08-19-parallel-e2e-suites.md), [#174](https://github.com/Prism-Shadow/agenthub/pull/174))
- [2026-08-19] The MiniMax routing branch moved into the client-type chain, after the Kimi branch. ([details](2026-08-19-minimax-routing-order.md), [#174](https://github.com/Prism-Shadow/agenthub/pull/174))
- [2026-08-18] Every streaming client skips the heartbeat events gateways inject on long generations instead of failing the stream. ([details](2026-08-18-gateway-heartbeats.md), [#174](https://github.com/Prism-Shadow/agenthub/pull/174))
