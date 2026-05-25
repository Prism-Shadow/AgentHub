# What's new in Claude Opus 4.7

Overview of new features, breaking changes, and behavior changes in Claude Opus 4.7.

---

Claude Opus 4.7 is our most capable generally available model to date. It is highly autonomous and performs exceptionally well on long-horizon agentic work, knowledge work, vision tasks, and memory tasks. This page summarizes everything new at launch.

## New model

| Model | API model ID | Description |
|:------|:-------------|:------------|
| Claude Opus 4.7 | `claude-opus-4-7` | Our most capable generally available model for complex reasoning and agentic coding |

Claude Opus 4.7 supports the [1M token context window](/docs/en/build-with-claude/context-windows), 128k max output tokens, [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking), and the same set of tools and platform features as Claude Opus 4.6.

For complete pricing and specs, see the [models overview](/docs/en/about-claude/models/overview).

## New features

### High-resolution image support

Claude Opus 4.7 is our first Claude model with high-resolution image support. Maximum image resolution has increased to **2576px / 3.75MP** (increased from our previous limit of 1568px / 1.15MP). This change should unlock performance gains on vision-heavy workloads, and is particularly important for computer use and screenshot/artifact/document understanding workflows.

Additionally, operations like mapping coordinates to images are now simpler — the model's coordinates are 1\:1 with actual pixels, so there's no scale-factor math required.

High-res images use more tokens. If the additional image fidelity is unnecessary, downsample images before sending to Claude to avoid token-usage increases.

Beyond resolution, Claude Opus 4.7 also improves on:

- **Low-level perception** — pointing, measuring, counting, and similar tasks.
- **Image localization** — natural-image bounding-box localization and detection are improved.

See [Images and vision](/docs/en/build-with-claude/vision) for details.

### New `xhigh` effort level

The [effort parameter](/docs/en/build-with-claude/effort) allows you to tune Claude's intelligence vs. token spend, trading off capability for faster speed and lower costs. Start with the new `xhigh` effort level for coding and agentic use cases, and use a minimum of `high` effort for most intelligence-sensitive use cases. See [Recommended effort levels for Claude Opus 4.7](/docs/en/build-with-claude/effort#recommended-effort-levels-for-claude-opus-4-7) for per-level guidance. (Messages API only; Claude Managed Agents handles effort automatically.)

### Task budgets (beta)

Claude Opus 4.7 introduces [task budgets](/docs/en/build-with-claude/task-budgets). A task budget gives Claude a rough estimate of how many tokens to target for a full agentic loop, including thinking, tool calls, tool results, and final output. The model sees a running countdown and uses it to prioritize work and finish the task gracefully as the budget is consumed. To use, set the beta header `task-budgets-2026-03-13` and add the following to your output config:

```python Python
response = client.beta.messages.create(
    model="claude-opus-4-7",
    max_tokens=128000,
    output_config={
        "effort": "high",
        "task_budget": {"type": "tokens", "total": 128000},
    },
    messages=[
        {"role": "user", "content": "Review the codebase and propose a refactor plan."}
    ],
    betas=["task-budgets-2026-03-13"],
)
```

You may need to experiment with different task budgets for your use case. If the model is given a task budget that is too restrictive for a given task, it may complete the task less thoroughly or refuse to do the task entirely.

For open-ended agentic tasks where quality matters more than speed, do not set a task budget; reserve task budgets for workloads where you need the model to scope its work to a token allowance. The minimum value for a task budget is 20k tokens.

This is not a hard cap; it's a suggestion that the model is aware of. This is distinct from `max_tokens`, which is a hard per-request cap on generated tokens (`max_tokens` is not passed to the model, and the model is not aware of it), while `task_budget` is an advisory cap across the full agentic loop. Use `task_budget` when you want the model to self-moderate, and `max_tokens` as a hard per-request ceiling to cap usage.

## Breaking changes

<Note>
These breaking changes apply to the Messages API only. If you use Claude Managed Agents, there are no breaking API changes for Claude Opus 4.7.
</Note>

### Extended thinking budgets removed

Extended thinking budgets are removed in Claude Opus 4.7. Setting `thinking: {"type": "enabled", "budget_tokens": N}` will return a 400 error. [Adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) is the only thinking-on mode, and in our internal evaluations it reliably outperforms extended thinking.

```python Python
# Before (Opus 4.6)
thinking = {"type": "enabled", "budget_tokens": 32000}

# After (Opus 4.7)
thinking = {"type": "adaptive"}
output_config = {"effort": "high"}
```

Adaptive thinking is **off by default** on Claude Opus 4.7. Requests with no `thinking` field run without thinking. Set `thinking: {type: "adaptive"}` explicitly to enable it.

### Sampling parameters removed

Starting with Claude Opus 4.7, setting `temperature`, `top_p`, or `top_k` to any non-default value will return a 400 error. The safest migration path is to omit these parameters entirely from requests, and to use prompting to guide the model's behavior. If you were using `temperature = 0` for determinism, note that it never guaranteed identical outputs.

### Thinking content omitted by default

Starting with Claude Opus 4.7, thinking content is omitted from the response by default. Thinking blocks still appear in the response stream, but their `thinking` field will be empty unless the caller explicitly opts in. This is a silent change — no error is raised — and response latency will be slightly improved. If reasoning outputs are needed, you can set `display` to `"summarized"` and opt back in with a one-line change:

```python Python
thinking = {
    "type": "adaptive",
    "display": "summarized",  # or "omitted" (default)
}
```

<Note>
If your product streams reasoning to users, the new default will appear as a long pause before output begins. Set `"display": "summarized"` to restore visible progress during thinking.
</Note>

### Updated token counting

Claude Opus 4.7 uses a new tokenizer, contributing to its improved performance on a wide range of tasks. This new tokenizer may use roughly 1x to 1.35x as many tokens when processing text compared to previous models (up to ~35% more, varying by content), and [`/v1/messages/count_tokens`](/docs/en/build-with-claude/token-counting) will return a different number of tokens for Claude Opus 4.7 than it did for Claude Opus 4.6. The token efficiency of Claude Opus 4.7 can vary by workload shape. Prompting interventions, `task_budget`, and `effort` can help control costs and ensure appropriate token usage. Keep in mind that these controls may trade off model intelligence.

We suggest updating your `max_tokens` parameters to give additional headroom, including compaction triggers. Claude Opus 4.7 provides a 1M context window at standard API pricing with no long-context premium.

## Capability improvements

### Knowledge work

Claude Opus 4.7 shows meaningful gains on knowledge-worker tasks, particularly where the model needs to visually verify its own outputs:

- **.docx redlining and .pptx editing** — improved at producing and self-checking tracked changes and slide layouts.
- **Charts and figure analysis** — improved at programmatic tool-calling with image-processing libraries (e.g. PIL) to analyze charts and figures, including pixel-level data transcription.

If existing prompts have mitigations in these areas (e.g. "double-check the slide layout before returning"), try removing that scaffolding and re-baselining.

### Memory

Claude Opus 4.7 is better at writing and using file-system-based memory. If an agent maintains a scratchpad, notes file, or structured memory store across turns, that agent should improve at jotting down notes to itself and leveraging its notes in future tasks. To give Claude a managed scratchpad without building your own, use the client-side [memory tool](/docs/en/agents-and-tools/tool-use/memory-tool).

### Vision

See [High-resolution image support](#high-resolution-image-support) above.

## Behavior changes

These are not API breaking changes but may require prompt updates. See [Migrating to Claude Opus 4.7](/docs/en/about-claude/models/migration-guide#migrating-to-claude-opus-4-7) for full guidance.

- **More literal instruction following**, particularly at lower effort levels. The model will not silently generalize an instruction from one item to another, and will not infer requests you didn't make.
- **Response length calibrates to perceived task complexity** rather than defaulting to a fixed verbosity.
- **Fewer tool calls by default,** using reasoning more. Raising effort increases tool usage.
- **More direct, opinionated tone** with less validation-forward phrasing and fewer emoji than Claude Opus 4.6's warmer style.
- **More regular progress updates** to the user throughout long agentic traces. If you've added scaffolding to force interim status messages, try removing it.
- **Fewer subagents spawned by default.** Steerable through prompting.
- **Real-time cybersecurity safeguards:** requests that involve prohibited or high-risk topics may lead to refusals. For legitimate security work, apply to the [Cyber Verification Program](https://claude.com/form/cyber-use-case).

## Migration guide

For step-by-step migration instructions and the full migration checklist, see [Migrating to Claude Opus 4.7](/docs/en/about-claude/models/migration-guide#migrating-to-claude-opus-4-7). If you use Claude Code or the Agent SDK, the [Claude API skill](/docs/en/agents-and-tools/agent-skills/claude-api-skill) can apply these migration steps to your codebase automatically.

## Next steps

<CardGroup>
  <Card title="Task budgets" icon="gauge" href="/docs/en/build-with-claude/task-budgets">
    Give Claude an advisory token budget across a full agentic loop.
  </Card>
  <Card title="Adaptive thinking" icon="brain" href="/docs/en/build-with-claude/adaptive-thinking">
    The only supported thinking-on mode on Claude Opus 4.7.
  </Card>
  <Card title="Effort" icon="sliders" href="/docs/en/build-with-claude/effort#recommended-effort-levels-for-claude-opus-4-7">
    Per-level effort guidance for Claude Opus 4.7.
  </Card>
  <Card title="Images and vision" icon="image" href="/docs/en/build-with-claude/vision">
    High-resolution image support and 1\:1 coordinate mapping.
  </Card>
  <Card title="Migration guide" icon="arrow-right" href="/docs/en/about-claude/models/migration-guide#migrating-to-claude-opus-4-7">
    Step-by-step upgrade instructions.
  </Card>
</CardGroup>
