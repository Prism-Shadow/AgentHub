# GLM-5.3

> Source: https://docs.bigmodel.cn/cn/guide/models/text/glm-5.3 (snapshot 2026-08-14,
> pre-launch: "GLM-5.3 模型 API 即将上线" — complete calling examples and the experience
> center follow after launch)

Zhipu's latest flagship base model, built on the same foundation as GLM-5.2 with
substantially scaled post-training (long-context task environments dozens of times
larger, extended post-training cycles). Headline claims: ~50% improved coding experience
over GLM-5.2 and top ranking among open-source models on Terminal Bench 3.0.

- **Model ID:** `glm-5.3` (API not yet live at snapshot time)
- **Input modality:** Text
- **Output modality:** Text
- **Context window:** 1M tokens
- **Maximum output tokens:** 128K
- **Capabilities:** multiple thinking modes, real-time streaming, function calling,
  context caching, structured output (JSON), MCP protocol support
- **Pricing:** not published at snapshot time

Benchmark movements cited by the vendor (GLM-5.2 → GLM-5.3): Terminal Bench 3.0
4.6 → 28.3, DeepSWE v1.1 46.2 → 66.9, Agents' Last Exam 23.8 → 28.5, CyberGym 84.5%,
ExploitBench 24.4% → 54.4%.
