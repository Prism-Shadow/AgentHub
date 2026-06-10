# Migration guide

Guide for migrating to the latest Claude models from previous Claude versions

---

<Note>
This guide covers migrating [Messages API](/docs/en/build-with-claude/working-with-messages) code. If you use [Claude Managed Agents](/docs/en/managed-agents/overview), no changes beyond updating the model name are required.
</Note>

<Tip>
  **Automate your migration with the Claude API skill.** In Claude Code, run `/claude-api migrate` to invoke the bundled [Claude API skill](/docs/en/agents-and-tools/agent-skills/claude-api-skill#migrating-to-a-newer-claude-model). It works for any target model on this page:

  ```text
  /claude-api migrate this project to claude-opus-4-8
  ```

  The skill applies the model ID swap and, as needed, breaking parameter changes, prefill replacement, and effort calibration for your target model across your codebase, then produces a checklist of items to verify manually. It asks you to confirm the migration scope (entire working directory, a subdirectory, or a specific file list) before editing any files. The skill also detects Amazon Bedrock, Vertex AI, Claude Platform on AWS, and Microsoft Foundry clients and adjusts model ID formats and feature changes for each platform.
</Tip>

## Migrating from Claude Mythos Preview to Claude Mythos 5 \{#migrating-from-claude-mythos-preview}

[Claude Mythos 5](https://anthropic.com/glasswing) is the access-gated successor to [Claude Mythos Preview](https://anthropic.com/glasswing), the invitation-only research preview. For general availability, see Claude Fable 5. Migration is mostly drop-in: Claude Mythos 5 uses the same [Messages API](/docs/en/build-with-claude/working-with-messages) and the same [tool use](/docs/en/agents-and-tools/tool-use/overview) patterns as Claude Mythos Preview. The key changes are the features that are no longer available (listed in the next section) and thinking output. Token counts are roughly unchanged: Claude Mythos 5 uses the same tokenizer as Claude Mythos Preview. For the Claude Mythos Preview retirement timeline, see [Model deprecations](/docs/en/about-claude/model-deprecations).

### Update your model name

```python
model = "claude-mythos-preview"  # Before
model = "claude-mythos-5"  # After
```

### Features not available on Claude Mythos 5

1. **Extended thinking and thinking token budgets:** Manual extended thinking (`thinking: {type: "enabled", budget_tokens: N}`) is not supported on `claude-mythos-5` and returns a 400 error. [Adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) is always on: the model determines when and how much to think on each request, and no `thinking` configuration is required. `thinking: {type: "disabled"}` returns an error. `budget_tokens` has no direct replacement: thinking is adaptive, and the [effort parameter](/docs/en/build-with-claude/effort) is a separate output-level control, not a thinking budget.

    Before (Claude Mythos Preview):

    ```python
    client.messages.create(
        model="claude-mythos-preview",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": "..."}],
    )
    ```

    After (Claude Mythos 5):


    ```python nocheck
    client.messages.create(
        model="claude-mythos-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": "..."}],
    )
    ```

2. **Assistant prefill:** Prefilling the assistant message is not supported on `claude-mythos-5` and returns a 400 error, the same as on Claude Mythos Preview. Use system prompt instructions instead.

3. **Thinking output:** On `claude-mythos-5`, the raw chain of thought is never returned, but thinking blocks still carry readable summarized text when `thinking.display` is set to `summarized`. Pass thinking blocks back unchanged when continuing a conversation on the same model. See [Thinking output on Claude Fable 5 and Claude Mythos 5](/docs/en/build-with-claude/adaptive-thinking#thinking-output-on-claude-fable-5-and-claude-mythos-5).

### Token counting and billing

`claude-mythos-5` uses the same tokenizer as `claude-mythos-preview` (the tokenizer introduced with Claude Opus 4.7). Token counts are roughly unchanged when migrating from `claude-mythos-preview`. The same content can tokenize to roughly 30% more tokens compared with models before Claude Opus 4.7, varying by content and workload shape.

[`/v1/messages/count_tokens`](/docs/en/build-with-claude/token-counting) returns roughly unchanged values for `claude-mythos-5` compared with `claude-mythos-preview`. Re-baseline cost and latency on your own workloads.

### Migration checklist

- [ ] Update the model name from `claude-mythos-preview` to `claude-mythos-5`.
- [ ] Remove manual extended thinking configuration (`thinking: {type: "enabled", budget_tokens: N}`). Adaptive thinking is always on, and no `thinking` field is required.
- [ ] Remove any `thinking: {type: "disabled"}` configuration. Disabling thinking returns an error on `claude-mythos-5`.
- [ ] Remove `budget_tokens`. It has no direct replacement: thinking is adaptive, and the `effort` parameter is a separate output-level control, not a thinking budget.
- [ ] Verify any code that parses the `thinking` field treats it as display text only and passes thinking blocks back unchanged when continuing on the same model. `thinking.display` defaults to `"omitted"` on `claude-mythos-5`, the same as on Claude Mythos Preview; set `display: "summarized"` to receive readable summaries. See [Thinking output on Claude Fable 5 and Claude Mythos 5](/docs/en/build-with-claude/adaptive-thinking#thinking-output-on-claude-fable-5-and-claude-mythos-5).
- [ ] If you replay conversation history on another model, strip `thinking` and `redacted_thinking` blocks from prior assistant turns first. Thinking blocks from `claude-mythos-5` are tied to the model that produced them; models other than Claude Fable 5 and Claude Mythos 5 silently ignore them; stripping keeps cross-model requests minimal and uniform.
- [ ] Re-baseline token counts and costs on your own workloads. Token counts are roughly unchanged when migrating from `claude-mythos-preview`.

## Migrating from Claude Opus 4.8 to Claude Fable 5 \{#migrating-from-claude-opus-48}

[Claude Fable 5](/docs/en/about-claude/models/introducing-claude-fable-5-and-claude-mythos-5) is Anthropic's most capable widely released model, generally available on the Claude API, [Claude Platform on AWS](/docs/en/build-with-claude/claude-platform-on-aws), [Amazon Bedrock](/docs/en/build-with-claude/claude-in-amazon-bedrock), [Vertex AI](/docs/en/build-with-claude/claude-on-vertex-ai), and [Microsoft Foundry](/docs/en/build-with-claude/claude-in-microsoft-foundry).

Migration is mostly drop-in: Claude Fable 5 uses the same [Messages API](/docs/en/build-with-claude/working-with-messages) and the same [tool use](/docs/en/agents-and-tools/tool-use/overview) patterns as Claude Opus 4.8, and supports the same [1M token context window](/docs/en/build-with-claude/context-windows) by default and [128k max output tokens](/docs/en/about-claude/models/overview). The key changes are always-on [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking), thinking output, safety classifier refusals, and pricing. Token counts are roughly unchanged: Claude Fable 5 uses the same tokenizer as Claude Opus 4.8.

<Note>
Claude Fable 5 is priced at $10 per million input tokens and $50 per million output tokens, compared with $5 and $25 for Claude Opus 4.8. See [Claude pricing](/docs/en/about-claude/pricing) for details.
</Note>

<Note>
If your code is on Claude Opus 4.7 or earlier, first apply [Migrating from Claude Opus 4.7 to Claude Opus 4.8](#migrating-from-claude-opus-47) and, for models earlier than Claude Opus 4.7, the [Claude Opus 4.7 migration steps](#migrating-to-claude-opus-4-7). Those sections cover breaking changes (sampling parameters rejected, manual extended thinking rejected, prefill removed, new tokenizer) that this section does not repeat.
</Note>

### Update your model name

```python
model = "claude-opus-4-8"  # Before
model = "claude-fable-5"  # After
```

### What changed

The items in this section describe the API and behavior differences worth checking after you swap the model ID.

1. **Adaptive thinking is always on:** [Adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) is the only thinking mode on `claude-fable-5`: the model determines when and how much to think on each request, and no `thinking` configuration is required. `thinking: {type: "disabled"}` returns an error. On Claude Opus 4.8, requests without a `thinking` field run without thinking; on `claude-fable-5`, those requests run with adaptive thinking. `max_tokens` remains a hard limit on total output, thinking plus response text, so revisit it for workloads that ran without thinking on Claude Opus 4.8. See [Cost control](/docs/en/build-with-claude/adaptive-thinking#cost-control). Use the [effort parameter](/docs/en/build-with-claude/effort) to control thinking depth.

    Before (Claude Opus 4.8):

    ```python
    client.messages.create(
        model="claude-opus-4-8",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": "..."}],
    )
    ```

    After (Claude Fable 5):

    ```python
    client.messages.create(
        model="claude-fable-5",
        max_tokens=16000,
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": "..."}],
    )
    ```

2. **Extended thinking and thinking budgets (unchanged):** Manual extended thinking (`thinking: {type: "enabled", budget_tokens: N}`) is not supported on `claude-fable-5` and returns a 400 error, the same as on Claude Opus 4.8. `budget_tokens` has no direct replacement: thinking is adaptive, and the [effort parameter](/docs/en/build-with-claude/effort) is a separate output-level control, not a thinking budget.

3. **Assistant prefill (unchanged):** Prefilling the assistant message is not supported on `claude-fable-5` and returns a 400 error, the same as on Claude Opus 4.8. Use system prompt instructions instead.

4. **Thinking output:** On `claude-fable-5`, the raw chain of thought is never returned, but thinking blocks still carry readable summarized text when `thinking.display` is set to `summarized`. Pass thinking blocks back unchanged when continuing a conversation on the same model. See [Thinking output on Claude Fable 5 and Claude Mythos 5](/docs/en/build-with-claude/adaptive-thinking#thinking-output-on-claude-fable-5-and-claude-mythos-5).

5. **Safety classifiers and the `refusal` stop reason:** `claude-fable-5` runs safety classifiers on requests and during response generation. When a classifier declines a request, the Messages API returns `stop_reason: "refusal"`, and the `stop_details.category` field reports which classifier fired (`"cyber"`, `"bio"`, `"reasoning_extraction"`, or `null` when the refusal maps to no named category). A refused response is a successful HTTP 200 response, not an error, and you are not billed for the input tokens of a request refused before any output is generated. When a classifier fires mid-stream, the input and already-streamed output are billed; discard the partial output. To re-run refused requests on another model automatically, pass the opt-in `fallbacks` parameter (in beta on the Claude API and Claude Platform on AWS; not available on the Message Batches API or on Amazon Bedrock, Vertex AI, and Microsoft Foundry; on those three platforms run the retry client-side or use the SDK refusal-fallback middleware). See [Handling stop reasons](/docs/en/build-with-claude/refusals-and-fallback).

6. **Start at `high` effort:** The [effort parameter](/docs/en/build-with-claude/effort) default remains `high`. On Claude Opus 4.8, the recommendation for coding and high-autonomy work is to set `xhigh` explicitly. On `claude-fable-5`, use `high` as the default for most tasks and reserve `xhigh` for the most capability-sensitive workloads: lower effort settings on `claude-fable-5` still perform well and often exceed `xhigh` performance on prior models. Reduce effort if a task completes but takes longer than necessary. See [Prompting Claude Fable 5](/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5#consider-all-effort-levels).

7. **Lower prompt caching minimum:** The minimum cacheable prompt length on `claude-fable-5` is 512 tokens, lower than the 1,024 tokens on Claude Opus 4.8. Prompts that were too short to cache on Claude Opus 4.8 can now create cache entries, with no code changes required. On Amazon Bedrock, the minimum for `claude-fable-5` is 1,024 tokens. See [Prompt caching](/docs/en/build-with-claude/prompt-caching#cache-limitations) for per-model minimums.

### Migration checklist

- [ ] Update the model name from `claude-opus-4-8` to `claude-fable-5`.
- [ ] Remove any `thinking: {type: "disabled"}` configuration. Disabling thinking returns an error on `claude-fable-5`, and requests without a `thinking` field run with adaptive thinking.
- [ ] If you removed manual extended thinking and assistant prefills during earlier migrations, no action is needed: both remain unsupported on `claude-fable-5`.
- [ ] Verify any code that parses the `thinking` field treats it as display text only and passes thinking blocks back unchanged when continuing on the same model. `thinking.display` defaults to `"omitted"` on `claude-fable-5`, the same as on Claude Opus 4.8; set `display: "summarized"` to receive readable summaries. See [Thinking output on Claude Fable 5 and Claude Mythos 5](/docs/en/build-with-claude/adaptive-thinking#thinking-output-on-claude-fable-5-and-claude-mythos-5).
- [ ] If you replay conversation history on another model, strip `thinking` and `redacted_thinking` blocks from prior assistant turns first. Thinking blocks from `claude-fable-5` are tied to the model that produced them; models other than Claude Fable 5 and Claude Mythos 5 silently ignore them; stripping keeps cross-model requests minimal and uniform. The exception is redeeming a [fallback credit](/docs/en/build-with-claude/fallback-credit), which requires the request body echoed under that feature's exact rules.
- [ ] Handle `stop_reason: "refusal"` and read the `stop_details.category` field. To re-run refused requests on another model automatically, consider the opt-in `fallbacks` parameter (beta). See [Handling stop reasons](/docs/en/build-with-claude/refusals-and-fallback).
- [ ] Re-evaluate your `effort` setting. Start at `high` for most tasks, including workloads that ran at `xhigh` on Claude Opus 4.8.
- [ ] Re-baseline cost and latency on your own workloads. Token counts are roughly unchanged when migrating from `claude-opus-4-8`; per-token pricing differs.

## Migrating from Claude Opus 4.7 to Claude Opus 4.8 \{#migrating-from-claude-opus-47}

Claude Opus 4.8 is Anthropic's most capable Opus-tier model. It builds on Claude Opus 4.7.

Claude Opus 4.8 should have strong out-of-the-box performance on existing Claude Opus 4.7 prompts and evals. There are no breaking API changes for code already running on Claude Opus 4.7. It supports the same set of features as Claude Opus 4.7, including the [1M token context window](/docs/en/build-with-claude/context-windows), [128k max output tokens](/docs/en/about-claude/models/overview), [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking), [prompt caching](/docs/en/build-with-claude/prompt-caching), [batch processing](/docs/en/build-with-claude/batch-processing), the [Files API](/docs/en/build-with-claude/files), [PDF support](/docs/en/build-with-claude/pdf-support), [vision](/docs/en/build-with-claude/vision), and the full set of server-side and client-side [tools](/docs/en/agents-and-tools/tool-use/overview). It also adds [mid-conversation system messages](/docs/en/about-claude/models/whats-new-claude-4-8#mid-conversation-system-messages) and publicly documents [refusal stop details](/docs/en/about-claude/models/whats-new-claude-4-8#refusal-stop-details).

<Note>
If your code is on Claude Opus 4.6 or earlier, also apply the [Claude Opus 4.7 migration steps](#migrating-to-claude-opus-4-7) below before upgrading to Claude Opus 4.8. Those steps include breaking changes (sampling parameters rejected, manual extended thinking rejected, new tokenizer) that the 4.8 upgrade alone does not cover.
</Note>

<Note>
On Microsoft Foundry, Claude Opus 4.8 has a 200k-token context window at launch. The 1M context window applies on the Claude API, Amazon Bedrock, and Vertex AI. See [Claude in Microsoft Foundry](/docs/en/build-with-claude/claude-in-microsoft-foundry).
</Note>

### Update your model name

```python
# Opus migration
model = "claude-opus-4-7"  # Before
model = "claude-opus-4-8"  # After
```

### What changed

These are not breaking changes. Code that runs on Claude Opus 4.7 continues to work unchanged on Claude Opus 4.8. The items below describe behavior differences worth checking after you swap the model ID.

1. **Sampling parameters (unchanged):** Setting `temperature`, `top_p`, or `top_k` to a non-default value returns a 400 error on Claude Opus 4.8, the same as on Claude Opus 4.7. The SDK request types still define these fields for compatibility with earlier models, so code that sets them type-checks, but the API rejects the request server-side. If you removed these parameters when migrating to Opus 4.7, no further changes are needed.

2. **Effort default is `high`:** The [effort parameter](/docs/en/build-with-claude/effort) default on Claude Opus 4.8 is `high` across all surfaces, including Claude Code and the Messages API. If you already set effort explicitly, your setting is unchanged. For coding and high-autonomy work, set `xhigh` explicitly. Re-evaluate your effort setting against your latency and cost budget.

3. **1M context window is the default:** Claude Opus 4.8 serves the full 1M token [context window](/docs/en/build-with-claude/context-windows) by default with no beta header and no long-context premium. If your client passes a context-window beta header for compatibility with older models, you can remove it on Claude Opus 4.8.

4. **Mid-conversation system messages:** Claude Opus 4.8 accepts `role: "system"` messages immediately after a user turn in the `messages` array (subject to [placement rules](/docs/en/build-with-claude/mid-conversation-system-messages#limitations)). Use the top-level `system` field for instructions that apply from the start. Earlier models, including Claude Opus 4.7, reject `role: "system"` in `messages` with a 400 error. If you maintain code paths that rebuild the full message history to update instructions, you can simplify them and preserve [prompt cache](/docs/en/build-with-claude/prompt-caching) hits on earlier turns.

5. **Refusal stop details:** The `stop_details` object on refusal responses (available since Claude Opus 4.7) is now publicly documented. When the model declines a request, it identifies the category of refusal, in addition to the existing `refusal` stop reason. No beta header is required, and there is no opt-out. See [Handling stop reasons](/docs/en/build-with-claude/handling-stop-reasons).

6. **Lower prompt caching minimum:** The minimum cacheable prompt length on Claude Opus 4.8 is 1,024 tokens, lower than on Claude Opus 4.7. Prompts that were too short to cache on Claude Opus 4.7 can now create cache entries, with no code changes required. See [Prompt caching](/docs/en/build-with-claude/prompt-caching#cache-limitations) for per-model minimums.

7. **Effort levels recalibrated:** The token allocation behind each effort level changes on Claude Opus 4.8 compared to Claude Opus 4.7: `medium` allows somewhat more thinking, `high` somewhat less, and `xhigh` substantially more. If you tuned an effort level against Claude Opus 4.7 cost or latency, re-baseline at the same level before adjusting it. See [Effort](/docs/en/build-with-claude/effort).

### Migration checklist

- [ ] Update model name from `claude-opus-4-7` to `claude-opus-4-8` (or update aliases).
- [ ] If you removed sampling parameters during the Opus 4.7 migration, no action is needed. If you re-added them with a 400-retry path, remove that retry path.
- [ ] Re-evaluate your `effort` setting. The default is `high` across all surfaces; for coding and high-autonomy work, set `xhigh` explicitly.
- [ ] Remove any context-window beta header. The 1M context window is the default on the Claude API, Amazon Bedrock, and Vertex AI (200k on Microsoft Foundry).
- [ ] If you rebuild conversation history to update instructions, consider switching to a mid-conversation system message to preserve prompt cache hits.
- [ ] Verify your stop-reason handling reads `stop_details` on refusals (available since Claude Opus 4.7; now publicly documented).
- [ ] Re-baseline cost and latency at your chosen effort level.

## Migrating to Claude Opus 4.7

Claude Opus 4.7 is highly autonomous and performs exceptionally well on long-horizon agentic work, knowledge work, vision tasks, and memory tasks.

Claude Opus 4.7 should have strong out-of-the-box performance on existing Claude Opus 4.6 prompts and evals at the same `$5 / $25` per MTok pricing, but there are a handful of behavioral and API changes worth knowing about as you migrate. It supports the same set of features as Claude Opus 4.6, including:

- [1M token context window](/docs/en/build-with-claude/context-windows) at standard API pricing with no long-context premium
- [128k max output tokens](/docs/en/about-claude/models/overview)
- [Adaptive thinking](/docs/en/build-with-claude/adaptive-thinking)
- [Prompt caching](/docs/en/build-with-claude/prompt-caching)
- [Batch processing](/docs/en/build-with-claude/batch-processing)
- [Files API](/docs/en/build-with-claude/files)
- [PDF support](/docs/en/build-with-claude/pdf-support)
- [Vision](/docs/en/build-with-claude/vision)
- The full set of server-side and client-side [tools](/docs/en/agents-and-tools/tool-use/overview) ([bash](/docs/en/agents-and-tools/tool-use/bash-tool), [code execution](/docs/en/agents-and-tools/tool-use/code-execution-tool), [computer use](/docs/en/agents-and-tools/tool-use/computer-use-tool), [text editor](/docs/en/agents-and-tools/tool-use/text-editor-tool), [web search](/docs/en/agents-and-tools/tool-use/web-search-tool), [web fetch](/docs/en/agents-and-tools/tool-use/web-fetch-tool), [MCP connector](/docs/en/agents-and-tools/mcp-connector), [memory](/docs/en/agents-and-tools/tool-use/memory-tool))

### Update your model name

```python
# Opus migration
model = "claude-opus-4-6"  # Before
model = "claude-opus-4-7"  # After
```

### Breaking changes

1. **Extended thinking removed:** `thinking: {type: "enabled", budget_tokens: N}` is no longer supported on Claude Opus 4.7 or later models and returns a 400 error. Switch to [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) (`thinking: {type: "adaptive"}`) and use the [effort parameter](/docs/en/build-with-claude/effort) to control thinking depth. Adaptive thinking is **off by default** on Claude Opus 4.7: requests with no `thinking` field run without thinking, matching Opus 4.6 behavior. Set `thinking: {type: "adaptive"}` explicitly to enable it.

    Before (Claude Opus 4.6):

    ```python
    client.messages.create(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
        messages=[{"role": "user", "content": "..."}],
    )
    ```

    After (Claude Opus 4.7):

    ```python
    client.messages.create(
        model="claude-opus-4-7",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},  # or "max", "xhigh", "medium", "low"
        messages=[{"role": "user", "content": "..."}],
    )
    ```

    Adaptive thinking is steerable through prompting. For guidance on tuning when the model over- or under-thinks, see [Calibrating effort and thinking depth](/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-4-8#calibrating-effort-and-thinking-depth).

2. **Sampling parameters removed:** Setting `temperature`, `top_p`, or `top_k` to any non-default value on Claude Opus 4.7 returns a 400 error. The safest migration path is to omit these parameters entirely from request payloads. Prompting is the recommended way to guide model behavior on Claude Opus 4.7. If you were using `temperature = 0` for determinism, note that it never guaranteed identical outputs on prior models.

3. **Thinking content omitted by default:** Thinking blocks still appear in the response stream on Claude Opus 4.7, but their `thinking` field is empty unless you explicitly opt in. This is a silent change from Claude Opus 4.6, where the default was to return summarized thinking text. To restore summarized thinking content on Claude Opus 4.7, set `thinking.display` to `"summarized"`:

    ```python
    thinking = {
        "type": "adaptive",
        "display": "summarized",
    }
    ```

    The default is `"omitted"` on Claude Opus 4.7. If your product streams reasoning to users, the new default appears as a long pause before output begins; set `display: "summarized"` to restore visible progress during thinking. See [Extended thinking](/docs/en/build-with-claude/extended-thinking#controlling-thinking-display) for details.

4. **Updated token counting:** Claude Opus 4.7 uses a new tokenizer, contributing to its improved performance on a wide range of tasks. The new tokenizer may use roughly 1x to 1.35x as many tokens when processing text compared to previous models (up to ~35% more, varying by content).

    [`/v1/messages/count_tokens`](/docs/en/build-with-claude/token-counting) will return a different number of tokens for Claude Opus 4.7 than it did for Claude Opus 4.6. Token efficiency can vary by workload shape.

    Prompting interventions, `task_budget`, and `effort` can help control costs and ensure appropriate token usage. These controls may trade off model intelligence. Update your `max_tokens` parameters to give additional headroom, including compaction triggers. Claude Opus 4.7 provides a 1M context window at standard API pricing with no long-context premium.

5. **Prefill removal (carried over from Opus 4.6):** Prefilling assistant messages returns a 400 error on Claude Opus 4.7. Use [structured outputs](/docs/en/build-with-claude/structured-outputs), system prompt instructions, or `output_config.format` instead.

### Choosing an effort level

The [effort parameter](/docs/en/build-with-claude/effort) allows you to tune Claude's intelligence vs. token spend, trading off capability for faster speed and lower costs. Start with the new `xhigh` effort level for coding and agentic use cases, and use a minimum of `high` effort for most intelligence-sensitive use cases. Experiment with other effort levels to further tune token usage and intelligence:

- **`max`:** Max effort can deliver performance gains in some use cases, but may show diminishing returns from increased token usage. This setting can also sometimes be prone to overthinking. Test max effort for intelligence-demanding tasks.
- **`xhigh` (new):** Extra high effort is the best setting for most coding and agentic use cases.
- **`high`:** This setting balances token usage and intelligence. For most intelligence-sensitive use cases, use a minimum of `high` effort.
- **`medium`:** Good for cost-sensitive use cases that need to reduce token usage while trading off intelligence.
- **`low`:** Reserve for short, scoped tasks and latency-sensitive workloads that are not intelligence-sensitive.

Effort is more important for this model than for any prior Opus. Experiment with it actively when you upgrade.

### Behavior changes

Claude Opus 4.7 has several behavioral differences from Claude Opus 4.6 that are not API breaking changes but may require prompt updates or scaffolding removal.

1. **Response length varies by use case:** Claude Opus 4.7 calibrates response length to how complex it judges the task to be, rather than defaulting to a fixed verbosity. This usually means shorter answers on simple lookups and much longer ones on open-ended analysis.

    If your product depends on a certain style or verbosity of output, you may need to tune your prompts. For example, to decrease verbosity, add: "Provide concise, focused responses. Skip non-essential context, and keep examples minimal." If you see specific kinds of over-explaining, add targeted instructions in your prompt to prevent them.

    Positive examples showing how Claude can communicate with the appropriate level of concision tend to be more effective than negative examples or instructions that tell the model what not to do.

2. **More literal instruction following:** Claude Opus 4.7 interprets prompts more literally and explicitly than Claude Opus 4.6, particularly at lower effort levels. It will not silently generalize an instruction from one item to another, and it will not infer requests you didn't make. The upside of this literalism is precision and less thrash. It generally performs better for API use cases with carefully tuned prompts, structured extraction, and pipelines where you want predictable behavior. A prompt and harness review may be especially helpful for migration to Claude Opus 4.7.

3. **More direct tone:** As with any new model, prose style on long-form writing may shift. Claude Opus 4.7 is more direct and opinionated, with less validation-forward phrasing and fewer emoji than Claude Opus 4.6's warmer style. If your product relies on a specific voice, re-evaluate style prompts against the new baseline.

4. **Built-in progress updates in agentic traces:** Claude Opus 4.7 provides more regular, higher-quality updates to the user throughout long agentic traces. If you've added scaffolding to force interim status messages ("After every 3 tool calls, summarize progress"), try removing it. If you find that the length or contents of Claude Opus 4.7's user-facing updates are not well-calibrated to your use case, explicitly describe what these updates should look like in the prompt and provide examples.

5. **Fewer subagents spawned by default:** Claude Opus 4.7 tends to spawn fewer subagents by default. However, this behavior is steerable through prompting; give Claude Opus 4.7 explicit guidance around when subagents are desirable.

6. **Stricter effort calibration:** Meaningfully changing from Claude Opus 4.6, Claude Opus 4.7 respects [effort levels](/docs/en/build-with-claude/effort) strictly, especially at the low end. At `low` and `medium`, the model scopes its work to what was asked rather than going above and beyond.

    This is good for latency and cost, but on moderately complex tasks running at `low` effort there is some risk of under-thinking. If you observe shallow reasoning on complex problems, raise effort to `high` or `xhigh` rather than prompting around it.

    If you need to keep effort at `low` for latency, add targeted guidance: "This task involves multi-step reasoning. Think carefully through the problem before responding." See [Recommended effort levels for Claude Opus 4.7](/docs/en/build-with-claude/effort#recommended-effort-levels-for-claude-opus-4-7).

7. **Fewer tool calls by default:** Claude Opus 4.7 has a tendency to use tools less often than Claude Opus 4.6 and to use reasoning more. This produces better results in most cases.

    To increase tool usage, raise the effort setting. `high` or `xhigh` effort settings show substantially more tool usage in agentic search and coding. You can also adjust your prompt to explicitly instruct the model about when and how to properly use its tools.

8. **Real-time cybersecurity safeguards:** Newly added in Claude Opus 4.7, requests that involve prohibited or high-risk topics may lead to refusals. For legitimate security work such as penetration testing, vulnerability research, or red-teaming, apply to the [Cyber Verification Program](https://claude.com/form/cyber-use-case) to request reduced restrictions. See [Safeguards, warnings, and appeals](https://support.claude.com/en/articles/8241253-safeguards-warnings-and-appeals) for background.

9. **High-resolution image support:** Claude Opus 4.7 is the first Claude model with high-resolution image support. Maximum image resolution is 2576 pixels on the long edge, up from 1568 pixels on prior models. This unlocks gains on vision-heavy workloads and is particularly valuable for computer use, screenshot understanding, and document analysis.

    High-resolution support is automatic and requires no beta header or client-side opt-in. Two things to plan for:

    - Full-resolution images can use up to approximately 3x more image tokens than on prior models (up to 4,784 tokens per image, compared to the previous cap of roughly 1,600 tokens per image). Re-budget `max_tokens` and cost expectations for image-heavy workloads, or downsample before sending if you do not need the additional fidelity.
    - Pointing and bounding-box coordinates returned by the model are 1\:1 with actual image pixels on Claude Opus 4.7, so no scale-factor conversion is required.

    See [High-resolution image support on Claude Opus 4.7](/docs/en/build-with-claude/vision#high-resolution-image-support-on-claude-opus-4-7) for details.

### Recommended changes

These are not required but will improve your experience:

1. **Re-evaluate `max_tokens`:** Because the same text produces a higher token count on Claude Opus 4.7, update your `max_tokens` parameters to give additional headroom, including compaction triggers. Prompting interventions, [`task_budget`](/docs/en/build-with-claude/task-budgets), and [`effort`](/docs/en/build-with-claude/effort) can help control costs and ensure appropriate token usage.

2. **Audit token-count expectations:** Any code path that estimates tokens client-side or assumes a fixed token-to-character ratio should be re-tested against Claude Opus 4.7. Use the [Token counting endpoint](/docs/en/build-with-claude/token-counting) to verify.

3. **Adopt [task budgets](/docs/en/build-with-claude/task-budgets) (beta):** Claude Opus 4.7 introduces task budgets. These budgets let you inform Claude how many tokens it has for a full agentic loop, including thinking, tool calls, tool results, and final output. The model sees a running countdown and uses it to prioritize work and finish the task gracefully as the budget is consumed. To use, set the beta header `task-budgets-2026-03-13` and add the following to your output config:

    ```python
    output_config = {
        "effort": "high",
        "task_budget": {"type": "tokens", "total": 128000},
    }
    ```

    You may need to experiment with different task budgets for your use case. If the model is given a task budget that is too restrictive, it may complete the task less thoroughly, referencing its budget as the constraint.

    For open-ended agentic tasks where quality matters more than speed, do not set a task budget. Reserve task budgets for workloads where you need the model to scope its work to a token allowance. The minimum value for a task budget is 20k tokens.

    A task budget is not a hard cap; it's a suggestion that the model is aware of. It differs from `max_tokens`:

    - **`task_budget`:** an advisory cap across the full agentic loop. The model sees it and uses it to pace itself.
    - **`max_tokens`:** a hard per-request ceiling on generated tokens. It is not passed to the model, so the model is not aware of it.

    Use `task_budget` when you want the model to self-moderate, and `max_tokens` as a hard ceiling to cap usage.

4. **Set a large `max_tokens` at `max` or `xhigh` effort:** If you are running Claude Opus 4.7 at `max` or `xhigh` effort, set a large max output token budget so the model has room to think and act across its subagents and tool calls. Start at 64k tokens and tune from there.

5. **Downsample images if high resolution is unnecessary:** Claude Opus 4.7 supports images up to 2576px / 3.75MP. High-res images use more tokens. If the additional image fidelity is unnecessary, downsample images before sending to Claude to avoid token-usage increases. See [Images and vision](/docs/en/build-with-claude/vision).

### Migration checklist

- [ ] Update model name from `claude-opus-4-6` to `claude-opus-4-7` (or update aliases).
- [ ] Remove `temperature`, `top_p`, and `top_k` from request payloads.
- [ ] Replace `thinking: {type: "enabled", budget_tokens: N}` with `thinking: {type: "adaptive"}` plus the [effort parameter](/docs/en/build-with-claude/effort).
- [ ] Remove any assistant-message prefills.
- [ ] If your UI displays thinking content, explicitly opt in to thinking summarization.
- [ ] Re-benchmark end-to-end cost and latency under the updated tokenization.
- [ ] Re-tune `max_tokens` to account for the updated tokenization.
- [ ] Re-test any client-side token-count estimations.
- [ ] If your application sends images, re-budget for [high-resolution image support](/docs/en/build-with-claude/vision#high-resolution-image-support-on-claude-opus-4-7) (up to approximately 3x more image tokens per full-resolution image). Downsample before sending if you do not need the additional fidelity.
- [ ] If you consume pointing or bounding-box coordinates from the model, remove any scale-factor conversion; coordinates are 1\:1 with actual image pixels on Claude Opus 4.7.
- [ ] Review prompts for the behavior changes above (response length, literalism, tone, progress updates, subagents, effort calibration, tool triggering, cyber safeguards, high-resolution image handling).
- [ ] Re-baseline response length with existing length-control prompts removed, then tune explicitly.
- [ ] If using `xhigh` or `max` effort, raise `max_tokens` to at least 64k as a starting point.
- [ ] Consider adopting task budgets (beta) for agentic workflows.
- [ ] If your product does legitimate security work, apply to the [Cyber Verification Program](https://claude.com/form/cyber-use-case) for access to lower restrictions on cyber content.

## Migrating to Claude Opus 4.7 from Opus 4.5 or earlier

If you are migrating from Claude Opus 4.5, Opus 4.1 (deprecated), or an earlier model directly to Claude Opus 4.7, apply **all of the [Opus 4.7 changes above](#migrating-to-claude-opus-4-7)** plus the cumulative changes in this section that took effect between Opus 4.5 and Opus 4.7. If you are migrating from Opus 4.6, you only need the [Opus 4.7 section above](#migrating-to-claude-opus-4-7).

### Update your model name

```python
# Opus migration
model = "claude-opus-4-5"  # Before
model = "claude-opus-4-7"  # After
```

### Breaking changes

1. **Prefill removal** is covered in the [Opus 4.7 breaking changes](#breaking-changes) above.

2. **Tool parameter quoting:** Claude Opus 4.6 and later models may produce slightly different JSON string escaping in tool call arguments (e.g., different handling of Unicode escapes or forward slash escaping). If you parse tool call `input` as a raw string rather than using a JSON parser, verify your parsing logic. Standard JSON parsers (like `json.loads()` or `JSON.parse()`) handle these differences automatically.

### Recommended changes

These changes improve your experience on Opus 4.7. Items marked **(required on Opus 4.7)** were optional recommendations when Opus 4.6 launched but are now mandatory; the rest remain recommended.

1. **Migrate to adaptive thinking (required on Opus 4.7):** `thinking: {type: "enabled", budget_tokens: N}` returns a 400 error on Claude Opus 4.7. Switch to `thinking: {type: "adaptive"}` and use the [effort parameter](/docs/en/build-with-claude/effort) to control thinking depth. See [Adaptive thinking](/docs/en/build-with-claude/adaptive-thinking).

   <CodeGroup>
   ```bash cURL
   curl -sS https://api.anthropic.com/v1/messages \
     -H "content-type: application/json" \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" \
     -d '{
       "model": "claude-opus-4-7",
       "max_tokens": 16000,
       "thinking": {"type": "adaptive"},
       "output_config": {"effort": "high"},
       "messages": [{"role": "user", "content": "Your prompt here"}]
     }'
   ```

   ```python Before hidelines={1..3}
   import anthropic

   client = anthropic.Anthropic()
   response = client.beta.messages.create(
       model="claude-opus-4-5",
       max_tokens=16000,
       thinking={"type": "enabled", "budget_tokens": 32000},
       betas=["interleaved-thinking-2025-05-14"],
       messages=[{"role": "user", "content": "Your prompt here"}],
   )
   ```

   ```python After
   response = client.messages.create(
       model="claude-opus-4-7",
       max_tokens=16000,
       thinking={"type": "adaptive"},
       output_config={"effort": "high"},
       messages=[{"role": "user", "content": "Your prompt here"}],
   )
   ```

   ```bash CLI
   ant messages create <<'YAML'
   model: claude-opus-4-7
   max_tokens: 16000
   thinking:
     type: adaptive
   output_config:
     effort: high
   messages:
     - role: user
       content: Your prompt here
   YAML
   ```

   ```typescript TypeScript hidelines={1..2}
   import Anthropic from "@anthropic-ai/sdk";

   const client = new Anthropic();

   const response = await client.messages.create({
     model: "claude-opus-4-7",
     max_tokens: 16000,
     thinking: { type: "adaptive" },
     output_config: { effort: "high" },
     messages: [{ role: "user", content: "Your prompt here" }]
   });
   ```

   ```csharp C#
   using Anthropic;
   using Anthropic.Models.Messages;

   AnthropicClient client = new();

   var parameters = new MessageCreateParams
   {
       Model = Model.ClaudeOpus4_7,
       MaxTokens = 16000,
       Thinking = new ThinkingConfigAdaptive(),
       OutputConfig = new OutputConfig { Effort = Effort.High },
       Messages = [new() { Role = Role.User, Content = "Your prompt here" }]
   };

   var response = await client.Messages.Create(parameters);
   Console.WriteLine(response);
   ```

   ```go Go hidelines={1..11,-1}
   package main

   import (
   	"context"
   	"fmt"
   	"log"

   	"github.com/anthropics/anthropic-sdk-go"
   )

   func main() {
   	client := anthropic.NewClient()

   	response, err := client.Messages.New(context.TODO(), anthropic.MessageNewParams{
   		Model:     anthropic.ModelClaudeOpus4_7,
   		MaxTokens: 16000,
   		Thinking: anthropic.ThinkingConfigParamUnion{
   			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
   		},
   		OutputConfig: anthropic.OutputConfigParam{
   			Effort: anthropic.OutputConfigEffortHigh,
   		},
   		Messages: []anthropic.MessageParam{
   			anthropic.NewUserMessage(anthropic.NewTextBlock("Your prompt here")),
   		},
   	})
   	if err != nil {
   		log.Fatal(err)
   	}
   	fmt.Println(response)
   }
   ```

   ```java Java hidelines={1..5,8..10,-2..}
   import com.anthropic.client.AnthropicClient;
   import com.anthropic.client.okhttp.AnthropicOkHttpClient;
   import com.anthropic.models.messages.MessageCreateParams;
   import com.anthropic.models.messages.Message;
   import com.anthropic.models.messages.Model;
   import com.anthropic.models.messages.OutputConfig;
   import com.anthropic.models.messages.ThinkingConfigAdaptive;

   public class AdaptiveThinkingExample {
       public static void main(String[] args) {
           AnthropicClient client = AnthropicOkHttpClient.fromEnv();

           MessageCreateParams params = MessageCreateParams.builder()
               .model(Model.CLAUDE_OPUS_4_7)
               .maxTokens(16000L)
               .thinking(ThinkingConfigAdaptive.builder().build())
               .outputConfig(OutputConfig.builder()
                   .effort(OutputConfig.Effort.HIGH)
                   .build())
               .addUserMessage("Your prompt here")
               .build();

           Message response = client.messages().create(params);
           System.out.println(response);
       }
   }
   ```

   ```php PHP hidelines={1..4}
   <?php

   use Anthropic\Client;

   $client = new Client();

   $response = $client->messages->create(
       maxTokens: 16000,
       messages: [['role' => 'user', 'content' => 'Your prompt here']],
       model: 'claude-opus-4-7',
       thinking: ['type' => 'adaptive'],
       outputConfig: ['effort' => 'high'],
   );
   ```

   ```ruby Ruby hidelines={1..2}
   require "anthropic"

   client = Anthropic::Client.new

   response = client.messages.create(
     model: "claude-opus-4-7",
     max_tokens: 16000,
     thinking: { type: "adaptive" },
     output_config: { effort: "high" },
     messages: [{ role: "user", content: "Your prompt here" }]
   )
   ```
   </CodeGroup>

   Note that the migration also moves from `client.beta.messages.create` to `client.messages.create`. Adaptive thinking and effort are GA features and do not require the beta SDK namespace or any beta headers.

2. **Remove effort beta header:** The effort parameter is now GA. Remove `betas=["effort-2025-11-24"]` from your requests.

3. **Remove fine-grained tool streaming beta header:** Fine-grained tool streaming is now GA. Remove `betas=["fine-grained-tool-streaming-2025-05-14"]` from your requests.

4. **Remove interleaved thinking beta header:** Adaptive thinking automatically enables interleaved thinking on Claude Opus 4.7, Opus 4.6, and Sonnet 4.6. Remove `betas=["interleaved-thinking-2025-05-14"]` from your requests. The header is still functional on Sonnet 4.6 with manual extended thinking, but manual mode is deprecated.

5. **Migrate to output_config.format:** If using structured outputs, update `output_format={...}` to `output_config={"format": {...}}`. The old parameter remains functional but is deprecated and will be removed in a future model release.

### Migrating from Claude 4.1 or earlier

If you're migrating from Opus 4.1 (deprecated), Sonnet 4 (deprecated), or earlier models directly to Claude Opus 4.7, apply the Claude Opus 4.7 changes at the top of this guide and the cumulative changes above plus the additional changes in this section.

```python
# From Opus 4.1
model = "claude-opus-4-1-20250805"  # Before
model = "claude-opus-4-7"  # After

# From Sonnet 4
model = "claude-sonnet-4-20250514"  # Before
model = "claude-opus-4-7"  # After

# From Sonnet 3.7
model = "claude-3-7-sonnet-20250219"  # Before
model = "claude-opus-4-7"  # After
```

#### Additional breaking changes

1. **Remove sampling parameters**

   <Warning>
   This is a breaking change when migrating from Claude 3.x models.
   </Warning>

   Starting with Claude Opus 4.7, setting `temperature`, `top_p`, or `top_k` to any non-default value will return a 400 error. The safest migration path is to omit these parameters entirely from requests, and to use prompting to guide the model's behavior. If you were using `temperature = 0` for determinism, note that it never guaranteed identical outputs.


   ```python Python nocheck
   # Before - This will error in Claude 4+ models
   response = client.messages.create(
       model="claude-3-7-sonnet-20250219",
       temperature=0.7,
       top_p=0.9,  # Non-default sampling params return 400 on Opus 4.7
       # ...
   )

   # After
   response = client.messages.create(
       model="claude-opus-4-7",
       # ...
   )
   ```

2. **Update tool versions**

   <Warning>
   This is a breaking change when migrating from Claude 3.x models.
   </Warning>

   Update to the latest tool versions. Remove any code using the `undo_edit` command.

   ```python
   # Before
   tools = [{"type": "text_editor_20250124", "name": "str_replace_editor"}]

   # After
   tools = [{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}]
   ```

   - **Text editor:** Use `text_editor_20250728` and `str_replace_based_edit_tool`. See [Text editor tool documentation](/docs/en/agents-and-tools/tool-use/text-editor-tool) for details.
   - **Code execution:** Upgrade to `code_execution_20250825`. See [Code execution tool documentation](/docs/en/agents-and-tools/tool-use/code-execution-tool#upgrade-to-latest-tool-version) for migration instructions.

3. **Handle the `refusal` stop reason**

   Update your application to [handle `refusal` stop reasons](/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals):


   ```python Python nocheck
   response = client.messages.create(...)

   if response.stop_reason == "refusal":
       # Handle refusal appropriately
       pass
   ```

4. **Handle the `model_context_window_exceeded` stop reason**

   Claude 4.5+ models return a `model_context_window_exceeded` stop reason when generation stops due to hitting the context window limit, rather than the requested `max_tokens` limit. Update your application to handle this new stop reason:


   ```python Python nocheck
   response = client.messages.create(...)

   if response.stop_reason == "model_context_window_exceeded":
       # Handle context window limit appropriately
       pass
   ```

5. **Verify tool parameter handling (trailing newlines)**

   Claude 4.5+ models preserve trailing newlines in tool call string parameters that were previously stripped. If your tools rely on exact string matching against tool call parameters, verify your logic handles trailing newlines correctly.

6. **Update your prompts for behavioral changes**

   Claude 4+ models have a more concise, direct communication style and require explicit direction. Review [prompting best practices](/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) for optimization guidance.

#### Additional recommended changes

- **Remove legacy beta headers:** Remove `token-efficient-tools-2025-02-19` and `output-128k-2025-02-19`. All Claude 4+ models have built-in token-efficient tool use and these headers have no effect.

### Migration checklist (from Opus 4.5 or earlier)

- [ ] Update model ID to `claude-opus-4-7`
- [ ] Apply all [Opus 4.7 breaking changes](#migrating-to-claude-opus-4-7) (extended thinking removed, sampling parameters removed, thinking display omitted by default, updated tokenization)
- [ ] **BREAKING:** Remove assistant message prefills (returns 400 error); use structured outputs or `output_config.format` instead
- [ ] **BREAKING on Opus 4.7:** Replace `thinking: {type: "enabled", budget_tokens: N}` with `thinking: {type: "adaptive"}` plus the [effort parameter](/docs/en/build-with-claude/effort) (returns 400 on Opus 4.7)
- [ ] Verify tool call JSON parsing uses a standard JSON parser
- [ ] Remove `effort-2025-11-24` beta header (effort is now GA)
- [ ] Remove `fine-grained-tool-streaming-2025-05-14` beta header
- [ ] Remove `interleaved-thinking-2025-05-14` beta header (adaptive thinking enables interleaved thinking automatically)
- [ ] Migrate `output_format` to `output_config.format` (if applicable)
- [ ] If migrating from Claude 4.1 or earlier: remove `temperature`, `top_p`, and `top_k` (non-default values return 400 on Opus 4.7)
- [ ] If migrating from Claude 4.1 or earlier: update tool versions (`text_editor_20250728`, `code_execution_20250825`)
- [ ] If migrating from Claude 4.1 or earlier: handle `refusal` stop reason
- [ ] If migrating from Claude 4.1 or earlier: handle `model_context_window_exceeded` stop reason
- [ ] If migrating from Claude 4.1 or earlier: verify tool string parameter handling for trailing newlines
- [ ] If migrating from Claude 4.1 or earlier: remove legacy beta headers (`token-efficient-tools-2025-02-19`, `output-128k-2025-02-19`)
- [ ] Review and update prompts following [prompting best practices](/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- [ ] Test in development environment before production deployment

---

## Migrating to Claude Sonnet 4.6

Claude Sonnet 4.6 combines strong intelligence with fast performance, featuring improved agentic search capabilities and free code execution when used with web search or web fetch. It is ideal for everyday coding, analysis, and content tasks.

For a complete overview of capabilities, see the [models overview](/docs/en/about-claude/models/overview).

<Note>
Sonnet 4.6 pricing is $3 per million input tokens, $15 per million output tokens. See [Claude pricing](/docs/en/about-claude/pricing) for details.
</Note>

**Update your model name:**

```python
# From Sonnet 4.5
model = "claude-sonnet-4-5"  # Before
model = "claude-sonnet-4-6"  # After

# From Sonnet 4
model = "claude-sonnet-4-20250514"  # Before
model = "claude-sonnet-4-6"  # After
```

### Breaking changes

#### When migrating from Sonnet 4.5

1. **Prefilling assistant messages is no longer supported**

   <Warning>
   This is a breaking change when migrating from Sonnet 4.5 or earlier.
   </Warning>

   Prefilling assistant messages returns a `400` error on Sonnet 4.6. Use [structured outputs](/docs/en/build-with-claude/structured-outputs), system prompt instructions, or `output_config.format` instead.

   **Common prefill use cases and migrations:**

   - **Controlling output formatting** (forcing JSON/YAML output): Use [structured outputs](/docs/en/build-with-claude/structured-outputs) or tools with enum fields for classification tasks.

   - **Eliminating preambles** (removing "Here is..." phrases): Add direct instructions in the system prompt: "Respond directly without preamble. Do not start with phrases like 'Here is...', 'Based on...', etc."

   - **Avoiding bad refusals:** Claude is much better at appropriate refusals now. Clear prompting in the user message without prefill should be sufficient.

   - **Continuations** (resuming interrupted responses): Move the continuation to the user message: "Your previous response was interrupted and ended with `[previous_response]`. Continue from where you left off."

   - **Context hydration / role consistency** (refreshing context in long conversations): Inject what were previously prefilled-assistant reminders into the user turn instead.

2. **Tool parameter JSON escaping may differ**

   <Warning>
   This is a breaking change when migrating from Sonnet 4.5 or earlier.
   </Warning>

   JSON string escaping in tool parameters may differ from previous models. Standard JSON parsers handle this automatically, but custom string-based parsing may need updates.

#### When migrating from Claude 3.x

3. **Update sampling parameters**

   <Warning>
   This is a breaking change when migrating from Claude 3.x models.
   </Warning>

   Use only `temperature` OR `top_p`, not both.

4. **Update tool versions**

   <Warning>
   This is a breaking change when migrating from Claude 3.x models.
   </Warning>

   Update to the latest tool versions (`text_editor_20250728`, `code_execution_20250825`). Remove any code using the `undo_edit` command.

5. **Handle the `refusal` stop reason**

   Update your application to [handle `refusal` stop reasons](/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals).

6. **Update your prompts for behavioral changes**

   Claude 4 models have a more concise, direct communication style. Review [prompting best practices](/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) for optimization guidance.

### Recommended changes

1. **Remove `fine-grained-tool-streaming-2025-05-14` beta header:** Fine-grained tool streaming is now GA on Sonnet 4.6 and no longer requires a beta header.
2. **Migrate `output_format` to `output_config.format`:** The `output_format` parameter is deprecated. Use `output_config.format` instead.

### Migrating from Sonnet 4.5

Consider migrating from Sonnet 4.5 to Sonnet 4.6, which delivers more intelligence at the same price point.

<Warning>
Sonnet 4.6 defaults to an effort level of `high`, in contrast to Sonnet 4.5 which had no effort parameter. Consider adjusting the effort parameter as you migrate from Sonnet 4.5 to Sonnet 4.6. If not explicitly set, you may experience higher latency with the default effort level.
</Warning>

#### If you're not using extended thinking

If you're not using extended thinking on Sonnet 4.5, you can continue without it on Sonnet 4.6. You should explicitly set effort to the level appropriate for your use case. At `low` effort with thinking disabled, you can expect similar or better performance relative to Sonnet 4.5 with no extended thinking.

<CodeGroup>
```bash cURL
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-sonnet-4-6",
    "max_tokens": 8192,
    "output_config": {
        "effort": "low"
    },
    "messages": [
        {
            "role": "user",
            "content": "Your prompt here"
        }
    ]
}'
```

```bash CLI
ant messages create <<'YAML'
model: claude-sonnet-4-6
max_tokens: 8192
output_config:
  effort: low
messages:
  - role: user
    content: Your prompt here
YAML
```

```python Python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    output_config={"effort": "low"},
    messages=[{"role": "user", "content": "Your prompt here"}],
)
```

```typescript TypeScript
const response = await client.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 8192,
  output_config: { effort: "low" },
  messages: [{ role: "user", content: "Your prompt here" }]
});
```

```csharp C#
using Anthropic;
using Anthropic.Models.Messages;

AnthropicClient client = new();

var parameters = new MessageCreateParams
{
    Model = Model.ClaudeSonnet4_6,
    MaxTokens = 8192,
    OutputConfig = new OutputConfig
    {
        Effort = Effort.Low
    },
    Messages = [new() { Role = Role.User, Content = "Your prompt here" }]
};
var message = await client.Messages.Create(parameters);
Console.WriteLine(message);
```

```go Go hidelines={1..11,-1}
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/anthropics/anthropic-sdk-go"
)

func main() {
	client := anthropic.NewClient()

	response, err := client.Messages.New(context.TODO(), anthropic.MessageNewParams{
		Model:     anthropic.Model("claude-sonnet-4-6"),
		MaxTokens: 8192,
		OutputConfig: anthropic.OutputConfigParam{
			Effort: anthropic.OutputConfigEffortLow,
		},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Your prompt here")),
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(response.Content[0].Text)
}
```

```java Java hidelines={1..5,7..9,-2..}
import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.Message;
import com.anthropic.models.messages.Model;
import com.anthropic.models.messages.OutputConfig;

public class Main {
    public static void main(String[] args) {
        AnthropicClient client = AnthropicOkHttpClient.fromEnv();

        MessageCreateParams params = MessageCreateParams.builder()
            .model(Model.CLAUDE_SONNET_4_6)
            .maxTokens(8192L)
            .outputConfig(OutputConfig.builder()
                .effort(OutputConfig.Effort.LOW)
                .build())
            .addUserMessage("Your prompt here")
            .build();

        Message response = client.messages().create(params);
        response.content().stream()
            .flatMap(block -> block.text().stream())
            .forEach(textBlock -> System.out.println(textBlock.text()));
    }
}
```

```php PHP hidelines={1..4}
<?php

use Anthropic\Client;

$client = new Client();

$message = $client->messages->create(
    maxTokens: 8192,
    messages: [['role' => 'user', 'content' => 'Your prompt here']],
    model: 'claude-sonnet-4-6',
    outputConfig: ['effort' => 'low'],
);
echo $message->content[0]->text;
```

```ruby Ruby hidelines={1..2}
require "anthropic"

client = Anthropic::Client.new

message = client.messages.create(
  model: "claude-sonnet-4-6",
  max_tokens: 8192,
  output_config: {
    effort: "low"
  },
  messages: [
    { role: "user", content: "Your prompt here" }
  ]
)
puts message.content.first.text
```
</CodeGroup>

#### If you're using extended thinking

If you're using extended thinking with `budget_tokens` on Sonnet 4.5, it is still functional on Sonnet 4.6 but is deprecated. Migrate to [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) with the [effort parameter](/docs/en/build-with-claude/effort).

##### Migrating to adaptive thinking

[Adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) is the recommended replacement for `budget_tokens` on Sonnet 4.6. It is particularly well suited to the following workload patterns:

- **Autonomous multi-step agents:** coding agents that turn requirements into working software, data analysis pipelines, and bug finding where the model runs independently across many steps. Adaptive thinking lets the model calibrate its reasoning per step, staying on path over longer trajectories. For these workloads, start at `high` effort. If latency or token usage is a concern, scale down to `medium`.
- **Computer use agents:** Sonnet 4.6 achieved best-in-class accuracy on computer use evaluations using adaptive mode.
- **Bimodal workloads:** a mix of easy and hard tasks where adaptive skips thinking on simple queries and reasons deeply on complex ones.

When using adaptive thinking, evaluate `medium` and `high` effort on your tasks. The right level depends on your workload's tradeoff between quality, latency, and token usage.

<CodeGroup>
```bash cURL
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-sonnet-4-6",
    "max_tokens": 64000,
    "thinking": {
        "type": "adaptive"
    },
    "output_config": {
        "effort": "medium"
    },
    "messages": [
        {
            "role": "user",
            "content": "Your prompt here"
        }
    ]
}'
```

```bash CLI nocheck
ant messages create <<'YAML'
model: claude-sonnet-4-6
max_tokens: 64000
thinking:
  type: adaptive
output_config:
  effort: medium
messages:
  - role: user
    content: Your prompt here
YAML
```

```python Python nocheck
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=64000,
    thinking={"type": "adaptive"},
    output_config={"effort": "medium"},
    messages=[{"role": "user", "content": "Your prompt here"}],
)
```

```typescript TypeScript nocheck
const response = await client.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 64000,
  thinking: { type: "adaptive" },
  output_config: { effort: "medium" },
  messages: [{ role: "user", content: "Your prompt here" }]
});
```

```csharp C# nocheck
using Anthropic;
using Anthropic.Models.Messages;

AnthropicClient client = new();

var parameters = new MessageCreateParams
{
    Model = Model.ClaudeSonnet4_6,
    MaxTokens = 64000,
    Thinking = new ThinkingConfigAdaptive(),
    OutputConfig = new OutputConfig { Effort = Effort.Medium },
    Messages = [new() { Role = Role.User, Content = "Your prompt here" }]
};

var message = await client.Messages.Create(parameters);
Console.WriteLine(message);
```

```go Go nocheck hidelines={1..11,-1}
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/anthropics/anthropic-sdk-go"
)

func main() {
	client := anthropic.NewClient()

	response, err := client.Messages.New(context.TODO(), anthropic.MessageNewParams{
		Model:     "claude-sonnet-4-6",
		MaxTokens: 64000,
		Thinking: anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{},
		},
		OutputConfig: anthropic.OutputConfigParam{
			Effort: anthropic.OutputConfigEffortMedium,
		},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Your prompt here")),
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(response)
}
```

```java Java nocheck hidelines={1..5,8..10,-2..}
import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.Message;
import com.anthropic.models.messages.Model;
import com.anthropic.models.messages.OutputConfig;
import com.anthropic.models.messages.ThinkingConfigAdaptive;

public class Main {
    public static void main(String[] args) {
        AnthropicClient client = AnthropicOkHttpClient.fromEnv();

        MessageCreateParams params = MessageCreateParams.builder()
            .model(Model.CLAUDE_SONNET_4_6)
            .maxTokens(64000L)
            .thinking(ThinkingConfigAdaptive.builder().build())
            .outputConfig(OutputConfig.builder()
                .effort(OutputConfig.Effort.MEDIUM)
                .build())
            .addUserMessage("Your prompt here")
            .build();

        Message response = client.messages().create(params);
        System.out.println(response);
    }
}
```

```php PHP hidelines={1..4} nocheck
<?php

use Anthropic\Client;

$client = new Client();

$message = $client->messages->create(
    maxTokens: 64000,
    messages: [['role' => 'user', 'content' => 'Your prompt here']],
    model: 'claude-sonnet-4-6',
    thinking: ['type' => 'adaptive'],
    outputConfig: ['effort' => 'medium'],
);

echo array_find($message->content, fn($block) => $block->type === 'text')->text;
```

```ruby Ruby nocheck hidelines={1..2}
require "anthropic"

client = Anthropic::Client.new

message = client.messages.create(
  model: "claude-sonnet-4-6",
  max_tokens: 64000,
  thinking: {
    type: "adaptive"
  },
  output_config: {
    effort: "medium"
  },
  messages: [
    { role: "user", content: "Your prompt here" }
  ]
)
puts message.content.find { |block| block.type == :text }.text
```
</CodeGroup>

<Note>
If you see inconsistent behavior or quality regressions with adaptive thinking, try lowering the [effort](/docs/en/build-with-claude/effort) setting or using `max_tokens` as a hard limit first. Extended thinking with `budget_tokens` is still functional on Sonnet 4.6 but is deprecated and no longer recommended.
</Note>

##### Keeping budget_tokens during migration

If you need to keep `budget_tokens` temporarily while migrating, a budget around 16k tokens provides headroom for harder problems without risk of runaway token usage. This configuration is deprecated and will be removed in a future model release.

###### Coding and agentic use cases

For agentic coding, frontend design, tool-heavy workflows, and complex enterprise workflows, start with `medium` effort. If you find latency is too high, consider reducing effort to `low`. If you need higher intelligence, consider increasing effort to `high` or migrating to Opus 4.7.

<CodeGroup>
```bash cURL
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "anthropic-beta: interleaved-thinking-2025-05-14" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-sonnet-4-6",
    "max_tokens": 16384,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 16384
    },
    "output_config": {
        "effort": "medium"
    },
    "messages": [
        {
            "role": "user",
            "content": "Your prompt here"
        }
    ]
}'
```

```bash CLI
ant beta:messages create --beta interleaved-thinking-2025-05-14 <<'YAML'
model: claude-sonnet-4-6
max_tokens: 16384
thinking:
  type: enabled
  budget_tokens: 16384
output_config:
  effort: medium
messages:
  - role: user
    content: Your prompt here
YAML
```

```python Python
response = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16384,
    thinking={"type": "enabled", "budget_tokens": 16384},
    output_config={"effort": "medium"},
    betas=["interleaved-thinking-2025-05-14"],
    messages=[{"role": "user", "content": "Your prompt here"}],
)
```

```typescript TypeScript
const response = await client.beta.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 16384,
  thinking: { type: "enabled", budget_tokens: 16384 },
  output_config: { effort: "medium" },
  betas: ["interleaved-thinking-2025-05-14"],
  messages: [{ role: "user", content: "Your prompt here" }]
});
```

```csharp C#
using Anthropic;
using Anthropic.Models.Beta;
using Anthropic.Models.Beta.Messages;

AnthropicClient client = new();

var parameters = new MessageCreateParams
{
    Model = "claude-sonnet-4-6",
    MaxTokens = 16384,
    Thinking = new BetaThinkingConfigEnabled { BudgetTokens = 16384 },
    OutputConfig = new BetaOutputConfig
    {
        Effort = Effort.Medium
    },
    Betas = [AnthropicBeta.InterleavedThinking2025_05_14],
    Messages = [new() { Role = Role.User, Content = "Your prompt here" }]
};

var message = await client.Beta.Messages.Create(parameters);
Console.WriteLine(message);
```

```go Go hidelines={1..11,-1}
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/anthropics/anthropic-sdk-go"
)

func main() {
	client := anthropic.NewClient()

	response, err := client.Beta.Messages.New(context.TODO(), anthropic.BetaMessageNewParams{
		Model:     "claude-sonnet-4-6",
		MaxTokens: 16384,
		Thinking:  anthropic.BetaThinkingConfigParamOfEnabled(16384),
		OutputConfig: anthropic.BetaOutputConfigParam{
			Effort: anthropic.BetaOutputConfigEffortMedium,
		},
		Messages: []anthropic.BetaMessageParam{
			anthropic.NewBetaUserMessage(anthropic.NewBetaTextBlock("Your prompt here")),
		},
		Betas: []anthropic.AnthropicBeta{anthropic.AnthropicBetaInterleavedThinking2025_05_14},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(response)
}
```

```java Java hidelines={1..6,9..11,-2..}
import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.models.beta.messages.MessageCreateParams;
import com.anthropic.models.beta.messages.BetaMessage;
import com.anthropic.models.messages.Model;
import com.anthropic.models.beta.AnthropicBeta;
import com.anthropic.models.beta.messages.BetaThinkingConfigEnabled;
import com.anthropic.models.beta.messages.BetaOutputConfig;

public class Main {
    public static void main(String[] args) {
        AnthropicClient client = AnthropicOkHttpClient.fromEnv();

        MessageCreateParams params = MessageCreateParams.builder()
            .model(Model.CLAUDE_SONNET_4_6)
            .maxTokens(16384L)
            .thinking(BetaThinkingConfigEnabled.builder()
                .budgetTokens(16384L)
                .build())
            .outputConfig(BetaOutputConfig.builder()
                .effort(BetaOutputConfig.Effort.MEDIUM)
                .build())
            .addBeta(AnthropicBeta.INTERLEAVED_THINKING_2025_05_14)
            .addUserMessage("Your prompt here")
            .build();

        BetaMessage response = client.beta().messages().create(params);
        System.out.println(response);
    }
}
```

```php PHP hidelines={1..4}
<?php

use Anthropic\Client;

$client = new Client();

$message = $client->beta->messages->create(
    maxTokens: 16384,
    messages: [['role' => 'user', 'content' => 'Your prompt here']],
    model: 'claude-sonnet-4-6',
    thinking: ['type' => 'enabled', 'budget_tokens' => 16384],
    outputConfig: ['effort' => 'medium'],
    betas: ['interleaved-thinking-2025-05-14'],
);

echo array_find($message->content, fn($block) => $block->type === 'text')->text;
```

```ruby Ruby hidelines={1..2}
require "anthropic"

client = Anthropic::Client.new

message = client.beta.messages.create(
  model: "claude-sonnet-4-6",
  max_tokens: 16384,
  thinking: {
    type: "enabled",
    budget_tokens: 16384
  },
  output_config: {
    effort: "medium"
  },
  betas: ["interleaved-thinking-2025-05-14"],
  messages: [
    { role: "user", content: "Your prompt here" }
  ]
)
puts message.content.find { |block| block.type == :text }.text
```
</CodeGroup>

###### Chat and non-coding use cases

For chat, content generation, search, classification, and other non-coding tasks, start with `low` effort with extended thinking. If you need more depth, increase effort to `medium`.

<CodeGroup>
```bash cURL
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "anthropic-beta: interleaved-thinking-2025-05-14" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-sonnet-4-6",
    "max_tokens": 8192,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 16384
    },
    "output_config": {
        "effort": "low"
    },
    "messages": [
        {
            "role": "user",
            "content": "Your prompt here"
        }
    ]
}'
```

```bash CLI
ant beta:messages create --beta interleaved-thinking-2025-05-14 <<'YAML'
model: claude-sonnet-4-6
max_tokens: 8192
thinking:
  type: enabled
  budget_tokens: 16384
output_config:
  effort: low
messages:
  - role: user
    content: Your prompt here
YAML
```

```python Python
response = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=8192,
    thinking={"type": "enabled", "budget_tokens": 16384},
    output_config={"effort": "low"},
    betas=["interleaved-thinking-2025-05-14"],
    messages=[{"role": "user", "content": "Your prompt here"}],
)
```

```typescript TypeScript
const response = await client.beta.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 8192,
  thinking: { type: "enabled", budget_tokens: 16384 },
  output_config: { effort: "low" },
  betas: ["interleaved-thinking-2025-05-14"],
  messages: [{ role: "user", content: "Your prompt here" }]
});
```

```csharp C#
using Anthropic;
using Anthropic.Models.Beta;
using Anthropic.Models.Beta.Messages;

AnthropicClient client = new();

var parameters = new MessageCreateParams
{
    Model = "claude-sonnet-4-6",
    MaxTokens = 8192,
    Thinking = new BetaThinkingConfigEnabled { BudgetTokens = 16384 },
    OutputConfig = new BetaOutputConfig
    {
        Effort = Effort.Low
    },
    Betas = [AnthropicBeta.InterleavedThinking2025_05_14],
    Messages = [new() { Role = Role.User, Content = "Your prompt here" }]
};

var message = await client.Beta.Messages.Create(parameters);
Console.WriteLine(message);
```

```go Go hidelines={1..11,-1}
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/anthropics/anthropic-sdk-go"
)

func main() {
	client := anthropic.NewClient()

	response, err := client.Beta.Messages.New(context.TODO(), anthropic.BetaMessageNewParams{
		Model:     "claude-sonnet-4-6",
		MaxTokens: 8192,
		Thinking:  anthropic.BetaThinkingConfigParamOfEnabled(16384),
		OutputConfig: anthropic.BetaOutputConfigParam{
			Effort: anthropic.BetaOutputConfigEffortLow,
		},
		Messages: []anthropic.BetaMessageParam{
			anthropic.NewBetaUserMessage(anthropic.NewBetaTextBlock("Your prompt here")),
		},
		Betas: []anthropic.AnthropicBeta{anthropic.AnthropicBetaInterleavedThinking2025_05_14},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(response)
}
```

```java Java hidelines={1..6,9..11,-2..}
import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;
import com.anthropic.models.beta.messages.MessageCreateParams;
import com.anthropic.models.beta.messages.BetaMessage;
import com.anthropic.models.messages.Model;
import com.anthropic.models.beta.AnthropicBeta;
import com.anthropic.models.beta.messages.BetaThinkingConfigEnabled;
import com.anthropic.models.beta.messages.BetaOutputConfig;

public class Main {
    public static void main(String[] args) {
        AnthropicClient client = AnthropicOkHttpClient.fromEnv();

        MessageCreateParams params = MessageCreateParams.builder()
            .model(Model.CLAUDE_SONNET_4_6)
            .maxTokens(8192L)
            .thinking(BetaThinkingConfigEnabled.builder()
                .budgetTokens(16384L)
                .build())
            .outputConfig(BetaOutputConfig.builder()
                .effort(BetaOutputConfig.Effort.LOW)
                .build())
            .addBeta(AnthropicBeta.INTERLEAVED_THINKING_2025_05_14)
            .addUserMessage("Your prompt here")
            .build();

        BetaMessage response = client.beta().messages().create(params);
        System.out.println(response);
    }
}
```

```php PHP hidelines={1..4}
<?php

use Anthropic\Client;

$client = new Client();

$message = $client->beta->messages->create(
    maxTokens: 8192,
    messages: [['role' => 'user', 'content' => 'Your prompt here']],
    model: 'claude-sonnet-4-6',
    thinking: ['type' => 'enabled', 'budget_tokens' => 16384],
    outputConfig: ['effort' => 'low'],
    betas: ['interleaved-thinking-2025-05-14'],
);

echo array_find($message->content, fn($block) => $block->type === 'text')->text;
```

```ruby Ruby hidelines={1..2}
require "anthropic"

client = Anthropic::Client.new

message = client.beta.messages.create(
  model: "claude-sonnet-4-6",
  max_tokens: 8192,
  thinking: {
    type: "enabled",
    budget_tokens: 16384
  },
  output_config: {
    effort: "low"
  },
  betas: ["interleaved-thinking-2025-05-14"],
  messages: [
    { role: "user", content: "Your prompt here" }
  ]
)
puts message.content.find { |block| block.type == :text }.text
```
</CodeGroup>

### Sonnet 4.6 migration checklist

- [ ] Update model ID to `claude-sonnet-4-6`
- [ ] **BREAKING:** Remove assistant message prefilling; use structured outputs or `output_config.format` instead
- [ ] **BREAKING:** Verify tool parameter JSON parsing handles escaping differences
- [ ] **BREAKING:** Update tool versions to latest (`text_editor_20250728`, `code_execution_20250825`); legacy versions are not supported (if migrating from 3.x)
- [ ] **BREAKING:** Remove any code using the `undo_edit` command (if applicable)
- [ ] **BREAKING:** Update sampling parameters to use only `temperature` OR `top_p`, not both (if migrating from 3.x)
- [ ] Handle new `refusal` stop reason in your application
- [ ] Remove `fine-grained-tool-streaming-2025-05-14` beta header (now GA)
- [ ] Migrate `output_format` to `output_config.format`
- [ ] Review and update prompts following [prompting best practices](/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- [ ] **Recommended:** Migrate from `thinking: {type: "enabled", budget_tokens: N}` to `thinking: {type: "adaptive"}` with the [effort parameter](/docs/en/build-with-claude/effort) (`budget_tokens` is deprecated and will be removed in a future release)
- [ ] Test in development environment before production deployment

---

## Migrating to Claude Sonnet 4.5

Claude Sonnet 4.5 combines strong intelligence with fast performance, making it ideal for everyday coding, analysis, and content tasks.

For a complete overview of capabilities, see the [models overview](/docs/en/about-claude/models/overview).

<Note>
Sonnet 4.5 pricing is $3 per million input tokens, $15 per million output tokens. See [Claude pricing](/docs/en/about-claude/pricing) for details.
</Note>

**Update your model name:**

```python
# From Sonnet 4
model = "claude-sonnet-4-20250514"  # Before
model = "claude-sonnet-4-5-20250929"  # After

# From Sonnet 3.7
model = "claude-3-7-sonnet-20250219"  # Before
model = "claude-sonnet-4-5-20250929"  # After
```

### Breaking changes

These breaking changes apply when migrating from Claude 3.x Sonnet models.

1. **Update sampling parameters**

   <Warning>
   This is a breaking change when migrating from Claude 3.x models.
   </Warning>

   Use only `temperature` OR `top_p`, not both.

2. **Update tool versions**

   <Warning>
   This is a breaking change when migrating from Claude 3.x models.
   </Warning>

   Update to the latest tool versions (`text_editor_20250728`, `code_execution_20250825`). Remove any code using the `undo_edit` command.

3. **Handle the `refusal` stop reason**

   Update your application to [handle `refusal` stop reasons](/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals).

4. **Update your prompts for behavioral changes**

   Claude 4 models have a more concise, direct communication style. Review [prompting best practices](/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) for optimization guidance.

### Sonnet 4.5 migration checklist

- [ ] Update model ID to `claude-sonnet-4-5-20250929`
- [ ] **BREAKING:** Update tool versions to latest (`text_editor_20250728`, `code_execution_20250825`); legacy versions are not supported (if migrating from 3.x)
- [ ] **BREAKING:** Remove any code using the `undo_edit` command (if applicable)
- [ ] **BREAKING:** Update sampling parameters to use only `temperature` OR `top_p`, not both (if migrating from 3.x)
- [ ] Handle new `refusal` stop reason in your application
- [ ] Review and update prompts following [prompting best practices](/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- [ ] Consider enabling extended thinking for complex reasoning tasks
- [ ] Test in development environment before production deployment

---

## Migrating to Claude Haiku 4.5

Claude Haiku 4.5 is the fastest and most intelligent Haiku model with near-frontier performance, delivering premium model quality for interactive applications and high-volume processing.

For a complete overview of capabilities, see the [models overview](/docs/en/about-claude/models/overview).

<Note>
Haiku 4.5 pricing is $1 per million input tokens, $5 per million output tokens. See [Claude pricing](/docs/en/about-claude/pricing) for details.
</Note>

**Update your model name:**

```python
# From Haiku 3.5
model = "claude-3-5-haiku-20241022"  # Before
model = "claude-haiku-4-5-20251001"  # After
```

**Review new rate limits:** Haiku 4.5 has separate rate limits from Haiku 3.5. See [Rate limits documentation](/docs/en/api/rate-limits) for details.

<Tip>
For significant performance improvements on coding and reasoning tasks, consider enabling extended thinking with `thinking: {type: "enabled", budget_tokens: N}`.
</Tip>

<Note>
Extended thinking impacts [prompt caching](/docs/en/build-with-claude/prompt-caching#caching-with-thinking-blocks) efficiency.

Extended thinking is deprecated in Claude 4.6 models and removed in Claude Opus 4.7. If using newer models, use [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) instead.
</Note>

**Explore new capabilities:** See the [models overview](/docs/en/about-claude/models/overview) for details on context awareness, increased output capacity (64k tokens), higher intelligence, and improved speed.

### Breaking changes

These breaking changes apply when migrating from Claude 3.x Haiku models.

1. **Update sampling parameters**

   <Warning>
   This is a breaking change when migrating from Claude 3.x models.
   </Warning>

   Use only `temperature` OR `top_p`, not both.

2. **Update tool versions**

   <Warning>
   This is a breaking change when migrating from Claude 3.x models.
   </Warning>

   Update to the latest tool versions (`text_editor_20250728`, `code_execution_20250825`). Remove any code using the `undo_edit` command.

3. **Handle the `refusal` stop reason**

   Update your application to [handle `refusal` stop reasons](/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals).

4. **Update your prompts for behavioral changes**

   Claude 4 models have a more concise, direct communication style. Review [prompting best practices](/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) for optimization guidance.

### Haiku 4.5 migration checklist

- [ ] Update model ID to `claude-haiku-4-5-20251001`
- [ ] **BREAKING:** Update tool versions to latest (`text_editor_20250728`, `code_execution_20250825`); legacy versions are not supported
- [ ] **BREAKING:** Remove any code using the `undo_edit` command (if applicable)
- [ ] **BREAKING:** Update sampling parameters to use only `temperature` OR `top_p`, not both
- [ ] Handle new `refusal` stop reason in your application
- [ ] Review and adjust for new rate limits (separate from Haiku 3.5)
- [ ] Review and update prompts following [prompting best practices](/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- [ ] Consider enabling extended thinking for complex reasoning tasks
- [ ] Test in development environment before production deployment

---

## Get help

- Check the [API documentation](/docs/en/api/overview) for detailed specifications
- Review [model capabilities](/docs/en/about-claude/models/overview) for performance comparisons
- Review [API release notes](/docs/en/release-notes/api) for API updates
- Contact support if you encounter any issues during migration
