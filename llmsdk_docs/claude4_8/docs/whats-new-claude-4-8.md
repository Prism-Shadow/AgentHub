# What's new in Claude Opus 4.8

Overview of new features and behavior changes in Claude Opus 4.8.

---

<NextOpus /> is Anthropic's most capable generally available model to date. It builds on Claude Opus 4.7. This page summarizes everything new at launch, including fast mode (research preview on the Claude API) and a lower 1,024-token minimum cacheable prompt length.

## New model

| Model | API model ID | Description |
|:------|:-------------|:------------|
| <NextOpus /> | <NextOpusId /> | Anthropic's most capable model for complex reasoning, long-horizon agentic coding, and high-autonomy work |

<NextOpus /> supports the [1M token context window](/docs/en/build-with-claude/context-windows) by default on the Claude API, Amazon Bedrock, and Vertex AI (200k on Microsoft Foundry), 128k max output tokens, [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking), and the same set of tools and platform features as Claude Opus 4.7.

For complete pricing and specs, see the [models overview](/docs/en/about-claude/models/overview).

## New features

### Mid-conversation system messages

<NextOpus /> accepts `role: "system"` messages immediately after a user turn in the `messages` array (subject to [placement rules](/docs/en/build-with-claude/mid-conversation-system-messages#limitations)). This lets you append updated instructions later in a long-running conversation without restating the full system prompt, which preserves [prompt cache](/docs/en/build-with-claude/prompt-caching) hits on the earlier turns and reduces input cost on agentic loops. No beta header is required. See [Mid-conversation system messages](/docs/en/build-with-claude/mid-conversation-system-messages) for usage details.

### Refusal stop details

The `stop_details` object on refusal responses (available since Claude Opus 4.7) is now publicly documented. When Claude declines to complete a request, this object describes the category of refusal, in addition to the existing `refusal` stop reason, making it easier for your application to tell apart different classes of declined request and to route the user to the right next step. No beta header is required. See [Handling stop reasons](/docs/en/build-with-claude/handling-stop-reasons) for the category list and handling guidance.

### Effort defaults

The [effort parameter](/docs/en/build-with-claude/effort) default on <NextOpus /> is `high` on all surfaces, including the Claude API and Claude Code. If you set effort explicitly today, your setting is unchanged. See [Effort](/docs/en/build-with-claude/effort) for per-level guidance.

### Fast mode

[Fast mode](/docs/en/build-with-claude/fast-mode) is now available for <NextOpus /> as a research preview on the Claude API. Set `speed: "fast"` to get up to 2.5x higher output tokens per second from the same model at premium pricing. See [Fast mode](/docs/en/build-with-claude/fast-mode) for access, supported models, and pricing.

### Lower prompt cache minimum

The minimum cacheable prompt length on <NextOpus /> is 1,024 tokens, lower than on Claude Opus 4.7. Prompts that were too short to cache on Claude Opus 4.7 can now create cache entries with no code changes. See [Prompt caching](/docs/en/build-with-claude/prompt-caching#cache-limitations) for per-model minimums.

## API constraints inherited from Claude Opus 4.7

<Note>
These constraints are unchanged from Claude Opus 4.7, so code that already runs on Claude Opus 4.7 needs no changes. They apply to the Messages API only; Claude Managed Agents are unaffected.
</Note>

### Sampling parameters not supported

Setting `temperature`, `top_p`, or `top_k` to a non-default value returns a 400 error on <NextOpus />, same as on Claude Opus 4.7. Omit these parameters and use prompting to guide the model's behavior.

### Adaptive thinking is the only thinking mode

Like Claude Opus 4.7, <NextOpus /> does not support extended thinking budgets. Setting `thinking: {"type": "enabled", "budget_tokens": N}` returns a 400 error. Use [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) and the [effort parameter](/docs/en/build-with-claude/effort) to control thinking depth.

```python Python
# Before (Opus 4.6 or earlier)
thinking = {"type": "enabled", "budget_tokens": 32000}

# After (Opus 4.7 and later)
thinking = {"type": "adaptive"}
output_config = {"effort": "high"}
```

## Capability improvements

### Improvement areas

Compared with Claude Opus 4.7, <NextOpus /> targets behavioral improvements in:

- **Long-horizon agentic coding**, including better long-context handling, fewer compactions, and better [compaction](/docs/en/build-with-claude/compaction) recovery.
- **Reasoning effort calibration**, with more reliable behavior at each effort level across a range of domains.
- **Tool triggering**, with fewer cases of skipping a tool call that the task required.

### Adaptive thinking

With [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) enabled, <NextOpus /> triggers reasoning only when it judges the turn needs it. On simple lookups and short agentic steps it responds directly; on complex multi-step problems it reasons before answering. This reduces wasted thinking tokens on bimodal workloads compared to Claude Opus 4.7 at the same effort level. As on Claude Opus 4.7, thinking is off unless you explicitly set `thinking: {type: "adaptive"}` in your request.

## Behavior changes

These are not API breaking changes but may require prompt updates. See [Migrating to <NextOpus />](/docs/en/about-claude/models/migration-guide#migrating-from-claude-opus-47) for full guidance.

- **Fewer wasted thinking tokens** at the same effort level when adaptive thinking is enabled, because the model decides per turn whether to think.
- **Better tool triggering.** The model is less likely to skip a tool call the task required, an issue some users reported on Claude Opus 4.7.
- **Better compaction handling and long-context quality.** Long agentic traces stay on task with fewer derailments after compaction.

## Migration guide

For step-by-step migration instructions and the full migration checklist, see [Migrating to <NextOpus />](/docs/en/about-claude/models/migration-guide#migrating-from-claude-opus-47). If you use Claude Code or the Agent SDK, the [Claude API skill](/docs/en/agents-and-tools/agent-skills/claude-api-skill) can apply these migration steps to your codebase automatically.

## Next steps

<CardGroup>
  <Card title="Migration guide" icon="arrow-right" href="/docs/en/about-claude/models/migration-guide#migrating-from-claude-opus-47">
    Step-by-step upgrade instructions from Claude Opus 4.7.
  </Card>
  <Card title="Effort" icon="sliders" href="/docs/en/build-with-claude/effort">
    Per-level effort guidance, including the new defaults.
  </Card>
  <Card title="Adaptive thinking" icon="brain" href="/docs/en/build-with-claude/adaptive-thinking">
    The only supported thinking-on mode on <NextOpus />.
  </Card>
  <Card title="Prompt caching" icon="database" href="/docs/en/build-with-claude/prompt-caching">
    How mid-conversation system messages preserve cache hits.
  </Card>
  <Card title="Handling stop reasons" icon="shield" href="/docs/en/build-with-claude/handling-stop-reasons">
    Refusal stop details and how to handle them.
  </Card>
  <Card title="Fast mode" icon="bolt" href="/docs/en/build-with-claude/fast-mode">
    Higher output speed at premium pricing.
  </Card>
</CardGroup>
