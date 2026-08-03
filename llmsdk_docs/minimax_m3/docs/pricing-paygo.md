> ## Documentation Index
>
> Fetch the complete documentation index at: [/docs/llms.txt](https://platform.minimax.io/docs/llms.txt)
>
> Use this file to discover all available pages before exploring further.

[Skip to main content](https://platform.minimax.io/docs/guides/pricing-paygo#content-area)

Pay-as-you-go uses standard Open Platform API Keys and consumes your account balance by actual usage. Credits are a separate prepaid balance used through a Subscription Key with the same resource coverage as Token Plan. For Credits pricing and usage behavior, see [Token Plan pricing](https://platform.minimax.io/docs/guides/pricing-token-plan).

## [​](https://platform.minimax.io/docs/guides/pricing-paygo\#llm)  LLM

[Recharge Now](https://platform.minimax.io/user-center/payment/balance)

- Standard

- Priority\*


| Model | Input | Output | Prompt caching Read |
| --- | --- | --- | --- |
| **MiniMax-M3**<br>≤ 512k input tokens Permanent 50% off | ~~$0.60~~ $0.30 / M tokens | ~~$2.40~~ $1.20 / M tokens | ~~$0.12~~ $0.06 / M tokens |
| **MiniMax-M3**<br>\> 512k input tokens\* Permanent 50% off | ~~$1.20~~ $0.60 / M tokens | ~~$4.80~~ $2.40 / M tokens | ~~$0.24~~ $0.12 / M tokens |

| Model | Input | Output | Prompt caching Read |
| --- | --- | --- | --- |
| **MiniMax-M3**<br>≤ 512k input tokens Permanent 50% off | ~~$0.90~~ $0.45 / M tokens | ~~$3.60~~ $1.80 / M tokens | ~~$0.18~~ $0.09 / M tokens |
| **MiniMax-M3**<br>\> 512k input tokens Permanent 50% off | ~~$1.80~~ $0.90 / M tokens | ~~$7.20~~ $3.60 / M tokens | ~~$0.36~~ $0.18 / M tokens |

\\* Priority provides priority admission for faster response times and improved request reliability. Set `service_tier` to `priority` to enable it. Pricing is 1.5x standard.

| Model | Input | Output | Prompt caching Read | Prompt caching Write |
| --- | --- | --- | --- | --- |
| **MiniMax-M2.7** | $0.3 / M tokens | $1.2 / M tokens | $0.06 / M tokens | $0.375 / M tokens |
| **MiniMax-M2.7-highspeed** | $0.6 / M tokens | $2.4 / M tokens | $0.06 / M tokens | $0.375 / M tokens |

Legacy Models

| Model | Input | Output | Prompt caching Read | Prompt caching Write |
| --- | --- | --- | --- | --- |
| **MiniMax-M2.5** | $0.3 / M tokens | $1.2 / M tokens | $0.03 / M tokens | $0.375 / M tokens |
| **MiniMax-M2.5-highspeed** | $0.6 / M tokens | $2.4 / M tokens | $0.03 / M tokens | $0.375 / M tokens |
| **MiniMax-M2.1** | $0.3 / M tokens | $1.2 / M tokens | $0.03 / M tokens | $0.375 / M tokens |
| **MiniMax-M2.1-highspeed** | $0.6 / M tokens | $2.4 / M tokens | $0.03 / M tokens | $0.375 / M tokens |
| **MiniMax-M2** | $0.3 / M tokens | $1.2 / M tokens | $0.03 / M tokens | $0.375 / M tokens |

Note:

1. The billing item is token count; the token-to-character ratio varies slightly depending on the usage scenario, subject to actual consumption
2. Token to English word ratio (estimate): approximately 750 English words consume 1000 tokens

## [​](https://platform.minimax.io/docs/guides/pricing-paygo\#audio)  Audio

[Recharge Now](https://platform.minimax.io/user-center/payment/balance)

| API | Model | Price |
| --- | --- | --- |
| **T2A** | speech-2.8-turbo | $60/M characters |
| **T2A** | speech-2.8-hd | $100/M characters |
| **Rapid Voice Cloning** | All Models | $1.5 per voice |
| **Voice Design** | All Models | $3 per voice |

Legacy Models

| API | Model | Price |
| --- | --- | --- |
| **T2A** | speech-2.6-turbo / speech-02-turbo | $60/M characters |
| **T2A** | speech-2.6-hd / speech-02-hd | $100/M characters |

## [​](https://platform.minimax.io/docs/guides/pricing-paygo\#video)  Video

[Recharge Now](https://platform.minimax.io/user-center/payment/balance)**Video Generation - Output Pricing**

| **Model / API** | **Resolution** | **Billing Rules** | **List Price** |
| --- | --- | --- | --- |
| MiniMax-H3 | 2K | Billed per second | $0.13 / second |
| MiniMax-H3 | 768P | Billed per second | $0.08 / second |

**Video Generation - Input Material Pricing**

| **Model / API** | **Material Type** | **Billing Rules** |
| --- | --- | --- |
| MiniMax-H3 | Audio | Free |
| MiniMax-H3 | Image | First **5 images** free; **$0.04 per additional image** |
| MiniMax-H3 | Video | Billed by input video duration and output video resolution: **2K $0.13/sec**, **768P $0.08/sec** |

**Video Regeneration - Output Pricing**Regenerate a previously produced 768P video into 2K, billed per second of the regenerated output.

| **Model / API** | **Resolution** | **Billing Rules** | **List Price** |
| --- | --- | --- | --- |
| MiniMax-H3-Regeneration | 768P → 2K | Billed per second of the regenerated output | $0.05 / second |

**Video Regeneration - Input Material Pricing**The input materials used in the original 768P generation task will be billed again.

| **Model / API** | **Material Type** | **Billing Rules** |
| --- | --- | --- |
| MiniMax-H3-Regeneration | Audio | Free |
| MiniMax-H3-Regeneration | Image | First **5 images** free; **$0.025 per additional image** |
| MiniMax-H3-Regeneration | Video | Billed by input video duration from the original 768P task: **$0.05 / second** |

**H3-Context-IR Task Pricing**

| **Model / API** | **Input Price** | **Output Price** |
| --- | --- | --- |
| MiniMax-H3-Context-IR | $0.90 / M tokens | $3.60 / M tokens |

Legacy Models

| Model | Price |
| --- | --- |
| MiniMax-Hailuo-2.3-Fast | $0.19 per 768P, 6s video |
| MiniMax-Hailuo-2.3-Fast | $0.32 per 768P, 10s video |
| MiniMax-Hailuo-2.3-Fast | $0.33 per 1080P, 6s video |
| MiniMax-Hailuo-2.3 | $0.28 per 768P, 6s video |
| MiniMax-Hailuo-2.3 | $0.56 per 768P, 10s video |
| MiniMax-Hailuo-2.3 | $0.49 per 1080P, 6s video |
| MiniMax-Hailuo-02 | $0.28 per 768P, 6s video |
| MiniMax-Hailuo-02 | $0.56 per 768P, 10s video |
| MiniMax-Hailuo-02 | $0.49 per 1080P, 6s video |
| MiniMax-Hailuo-02 | $0.10 per 512P, 6s video |
| MiniMax-Hailuo-02 | $0.15 per 512P, 10s video |

## [​](https://platform.minimax.io/docs/guides/pricing-paygo\#music)  Music

[Recharge Now](https://platform.minimax.io/user-center/payment/balance)

| Model | Description | Price |
| --- | --- | --- |
| Music-3.0-free | RPM = 3 | Free |
| Music-3.0 | RPM = 120, contact sales to increase | $0.15/up-to-5 minutes music |
| Music-2.6-free | RPM = 3 | Free |
| Music-2.6 | RPM = 120, contact sales to increase | $0.15/up-to-5 minutes music |
| Lyrics Generation | Lyrics generation/editing | $0.01/per song |

Legacy Models

| Model | Description | Price |
| --- | --- | --- |
| Music-2.5+ | Instrumental unlocked, break through style boundaries | $0.15/up-to-5 minutes music |
| Music-2.5 | Direct the detail, define the real | $0.15/up-to-5 minutes music |
| Music-2.0 | Enhanced musical expression | $0.03/up-to-5 minutes music |

## [​](https://platform.minimax.io/docs/guides/pricing-paygo\#image)  Image

[Recharge Now](https://platform.minimax.io/user-center/payment/balance)

| Model | Price |
| --- | --- |
| image-01 | $0.0035 per image |

## [​](https://platform.minimax.io/docs/guides/pricing-paygo\#mcp)  MCP

[Recharge Now](https://platform.minimax.io/user-center/payment/balance)

| Model | Input Price |
| --- | --- |
| **API-vlm** | $0.01 / request |

When API-vlm is called through Token Plan, usage deducts from the included Token Plan quota according to its pay-as-you-go price. If the included quota is exhausted and purchased Credits are available, additional usage can be automatically covered by purchased Credits.

🔔 **Pricing Update Notice** — Effective July 22, 2026, the API-vlm price will be adjusted to $0.01 per call. Accordingly, the token quota deducted per API-vlm call under Token Plan subscriptions will decrease, allowing the same plan to support more calls. API endpoints and model capabilities remain unchanged — no code changes required.

## [​](https://platform.minimax.io/docs/guides/pricing-paygo\#server-tools)  Server Tools

[Recharge Now](https://platform.minimax.io/user-center/payment/balance)

| Server Tool | Description | Price |
| --- | --- | --- |
| **web\_search** | Web search; the model runs the search on the server and answers based on the results. See [Server Tools](https://platform.minimax.io/docs/guides/server-tools). | $0.01 / request |

[Overview](https://platform.minimax.io/docs/pricing/overview) [Audio Subscription](https://platform.minimax.io/docs/guides/pricing-speech)

⌘I