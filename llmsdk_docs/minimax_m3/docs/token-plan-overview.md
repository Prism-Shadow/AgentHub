> ## Documentation Index
> Fetch the complete documentation index at: https://platform.minimax.io/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# Token Plan Overview

> Token Plan subscription and usage overview

<img src="https://file.cdn.minimax.io/public/token-plan-hero-v4.png" alt="Token Plan" style={{ display: "block", margin: "0 auto", maxWidth: "100%", borderRadius: "12px" }} />

## Welcome to the Token Plan!

MiniMax is one of the few AI labs that develops frontier models across the full spectrum of modalities: language, speech, video, music, and image. The [Token Plan](https://platform.minimax.io/subscribe/token-plan) extends upon our former Coding Plan by providing included Token Plan usage beyond language models, allowing more creative agents and applications to be built and used.

## Core Advantages

<CardGroup cols={3}>
  <Card title="Full-Spectrum Multimodal" icon="layers">
    One subscription covers eligible MiniMax resources through a shared usage bar.
  </Card>

  <Card title="Built for Agents" icon="zap">
    Plans are designed for long-context agent, coding, and multimodal workflows.
  </Card>

  <Card title="Cost-Effective" icon="piggy-bank">
    A flat subscription fee includes broad resource coverage while keeping usage predictable.
  </Card>
</CardGroup>

## Subscription Key

Each user has a dedicated **Subscription Key** for each Team they belong to. This key can exist before the Team has purchased Token Plan seats or Credits. If no resources are available to the user, the key has no usable paid resources. Once a Token Plan seat is assigned, or Credits access is available, the same Subscription Key can use those resources.

For subscription and Credits rules, see [Token Plan pricing](/docs/guides/pricing-token-plan). For team usage, see [Token Plan for Teams](/docs/guides/pricing-token-plan-team).

## Usage Quota

The [Token Plan](https://platform.minimax.io/subscribe/token-plan) usage quota is shown as a usage bar in the console. For API endpoints that have pay-as-you-go pricing, usage deducts from the included Token Plan quota according to the corresponding endpoint pricing.

|                   | **Plus**                          | **Max**                                      | **Ultra**                                   |
| :---------------- | :-------------------------------- | :------------------------------------------- | :------------------------------------------ |
| **Price**         | **\$20 /month**                   | **\$50 /month**                              | **\$120 /month**                            |
| **Best for**      | Personal projects and prototyping | Daily coding with agents and multimodal work | Heavy Agent workflows and extended sessions |
| **Quota windows** | 5-hour rolling and weekly windows | 5-hour rolling and weekly windows            | 5-hour rolling and weekly windows           |
| **Agent usage**   | 3-4 agents                        | 4-5 agents                                   | 6-7 agents                                  |

<div style={{ fontSize: "11px", color: "#6b7280", lineHeight: 1.35, marginTop: "-16px", marginBottom: "12px" }}>Available model coverage includes the full MiniMax lineup (M3 / M2.7 / image / speech / music). A small number of special models (MiniMax H3, voice design, rapid voice cloning, etc.) are not currently supported. Credits usage is unrestricted.</div>

<Tip>
  To call supported multimodal resources with your Subscription Key, see the [MiniMax CLI guide](https://platform.minimax.io/docs/token-plan/minimax-cli).
</Tip>

## Getting Started

<Steps>
  <Step title="Subscribe or get assigned resources">
    Visit the [Token Plan](https://platform.minimax.io/subscribe/token-plan) subscription page to buy an individual subscription or Credits in your Default Team, or join a Team where a Token Plan seat or shared Credits are available.
  </Step>

  <Step title="Get your Subscription Key">
    Navigate to [Account / Token Plan](https://platform.minimax.io/user-center/payment/token-plan) to view your available resources and get your **Subscription Key**.
  </Step>
</Steps>

<Info>
  **Important Notes**

  * The Subscription Key is used for Token Plan subscriptions and purchased Credits.
  * The Subscription Key is not interchangeable with pay-as-you-go API Keys.
  * A Subscription Key can exist before any paid resource is available. It becomes usable when the user has a Token Plan seat or Credits access.
  * Please protect your API Key to prevent any loss of resources.
</Info>

## Use in AI Agents and Coding Tools

Pick your tool and follow the integration guide:

<CardGroup cols={3}>
  <Card title="OpenClaw" icon="bot" href="/docs/token-plan/openclaw" />

  <Card title="Claude Code" icon="terminal" href="/docs/token-plan/claude-code" />

  <Card title="Cursor" icon="square-code" href="/docs/token-plan/cursor" />

  <Card title="TRAE" icon="code" href="/docs/token-plan/trae" />

  <Card title="Hermes Agent" icon="sparkles" href="/docs/token-plan/hermes-agent" />
</CardGroup>

For other tools, see [Other Tools](/docs/token-plan/other-tools).

## After Reaching the Usage Limit

When you reach the 5-hour rolling quota or weekly quota, you have the following options:

1. **Use purchased Credits**:
   If purchased Credits are available, usage within Token Plan resource coverage can be automatically covered by purchased Credits.
2. **Upgrade or get another assignment**:
   Upgrade your subscription, or ask the Team Owner or Admin to assign a higher available plan.
3. **Switch to Pay-As-You-Go**:
   If you wish to continue without rate limits, you can replace your Subscription Key with your [pay-as-you-go API Key](https://platform.minimax.io/user-center/basic-information/interface-key). This will switch the tool to a pay-as-you-go model based on actual token usage, which will consume your API account balance.
4. **Wait for the quota window to reset**:
   The included Token Plan quota uses 5-hour rolling and weekly windows. Unused subscription quota does not carry over to the next billing cycle.

## Next Steps

<CardGroup cols={3}>
  <Card title="Quickstart" icon="rocket" href="/docs/token-plan/quickstart">
    Run your first MiniMax API call in 5 minutes.
  </Card>

  <Card title="FAQ" icon="info" href="/docs/token-plan/faq">
    Common questions on quotas, billing, switching, refunds.
  </Card>

  <Card title="Promotions" icon="gift" href="/docs/token-plan/promotion">
    Current discount campaigns.
  </Card>
</CardGroup>
