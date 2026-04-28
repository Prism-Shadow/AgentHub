---
latestModelInfo:
  model: gpt-5.5
  migrationGuide: /api/docs/guides/upgrading-to-gpt-5p5.md
  promptingGuide: /api/docs/guides/prompt-guidance.md
---

# Using GPT-5.5

## Introduction

GPT-5.5 raises the baseline for complex production workflows. It’s a strong fit for coding use cases, tool-heavy agents, grounded assistants, long-context retrieval, product-spec-to-plan workflows, and customer-facing workflows where execution quality and response polish are critical.

To get the most out of GPT-5.5, treat it as a new model family to tune for, not a drop-in replacement for `gpt-5.2` or `gpt-5.5`. Begin migration with a fresh baseline instead of carrying over every instruction from an older prompt stack. Start with the smallest prompt that preserves the product contract, then tune reasoning effort, verbosity, tool descriptions, and output format against representative examples.

GPT-5.5 supports all API features that were already available with GPT-5.5, including [prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching), [hosted tools](https://developers.openai.com/api/docs/guides/tools#available-tools), [tool search](https://developers.openai.com/api/docs/guides/tools-tool-search), [compaction](https://developers.openai.com/api/docs/guides/compaction), and `phase` handling for manually replayed assistant items.

See the [GPT-5.5 Prompting Guide](https://developers.openai.com/api/docs/guides/prompt-guidance?model=gpt-5.5) for examples of successful prompting patterns.

## What's new

- **More efficient reasoning:** GPT-5.5 reaches strong results with fewer reasoning tokens than prior models, even at the same reasoning effort. This is especially useful in complex, tool-heavy, or multi-step workflows where token savings compound.
- **Stronger task execution with outcome-first prompts:** GPT-5.5 is better at working from a clear goal, preserving constraints, and turning product intent into concrete next steps. Describe the expected outcome, success criteria, allowed side effects, evidence rules, and output shape. Avoid step-by-step process guidance unless the exact path matters.
- **Stronger and more precise tool use:** GPT-5.5 is especially useful on large tool surfaces, multi-step service workflows, and long-running agent tasks. It tends to be more precise in tool selection and argument use.
- **Tone is often more polished, but can be more direct:** GPT-5.5 often produces warmer, more readable answers with less prompt scaffolding.

## Behavioral changes

1. **Reasoning effort now defaults to `medium`:** GPT-5.5 defaults to `medium` reasoning effort. Treat `medium` as the recommended balanced starting point for quality, reliability, latency, and cost. For latency-sensitive workflows, evaluate `low` before `none` when tool use, planning, search, or multi-step decision making still matters. Reserve `none` for latency-critical tasks that don't need reasoning or multi-chained tool calls, such as lightweight voice turns, fast information retrieval, and classification. Increase to `high` or `xhigh` only when evals show a measurable quality gain that justifies the extra latency and cost. See the [Reasoning models documentation](https://developers.openai.com/api/docs/guides/reasoning) for more details on recommended settings.

   Higher reasoning effort isn't automatically better. If the task has conflicting instructions, weak stopping criteria, or open-ended tool access, higher effort can lead to overthinking, unnecessary searching, or output quality regressions. Increase effort only when evals show a measurable quality gain.

2. **Image inputs preserve more visual detail by default:** GPT-5.5 updates the default handling for image inputs to preserve more visual detail and improve computer use performance. When `image_detail` is unset or set to `auto`, the model now uses `original` behavior, preserving images without resizing up to 10,240,000 pixels or a 6,000-pixel dimension limit. For `high`, specify the value directly; it preserves images without resizing up to 2,500,000 pixels or a 2,048-pixel dimension limit. `low` now focuses on context efficiency and resizes images above a 512-pixel dimension limit more aggressively than previous models. See the [Images and vision documentation](https://developers.openai.com/api/docs/guides/images-vision).

3. **Improved instruction following:** GPT-5.5 interprets prompts in a literal and thorough manner, enabling specific, descriptive instructions when the product requires them. Define success criteria and stopping rules, especially for long-running, tool-heavy, or evidence-gathering workflows. See [Write outcome-first prompts](https://developers.openai.com/api/docs/guides/prompt-guidance?model=gpt-5.5#outcome-first-prompts-and-stopping-conditions) and [Keep the right specificity](https://developers.openai.com/api/docs/guides/prompt-guidance?model=gpt-5.5#formatting).

4. **Default style is more concise and direct:** GPT-5.5 tends to be efficient, direct, and task-oriented by default. This is useful for many production workflows, but customer-facing or conversational experiences may need explicit personality, warmth, rationale, and formatting guidance. Use `text.verbosity` intentionally: `medium` is the default, and `low` is often a better starting point for concise responses. See the [GPT-5.5 prompting guide](https://developers.openai.com/api/docs/guides/prompt-guidance?model=gpt-5.5).

5. **Coding workflows need stronger orchestration:** GPT-5.5 is better suited to complex coding tasks that require planning, tool use, codebase navigation, verification, and multi-step execution. For coding agents, be explicit about reuse, subagent delegation, test expectations, acceptance criteria, and when to continue versus ask for help.

## Migration quickstart

### Automated migration with Codex

Codex can apply the recommended changes in this guide with the [OpenAI Docs Skill](https://github.com/openai/skills/tree/main/skills/.curated/openai-docs).

```text
$openai-docs migrate this project to gpt-5.5
```

To use this skill in other coding agents, download it from the [OpenAI skills repository](https://github.com/openai/skills/tree/main/skills/.curated/openai-docs).

### API and model parameters

- Update the model slug to `gpt-5.5`.
- Use the Responses API for any reasoning, tool-calling, or multi-turn use case.
- Tune `reasoning.effort`. Use `low` for efficient reasoning, `medium` for a balanced point on the latency/performance curve, `high` for complex agentic tasks that require hard reasoning and where latency matters less, and `xhigh` for the hardest asynchronous agentic tasks or evals that test the bounds of model intelligence. See the [Reasoning models documentation](https://developers.openai.com/api/docs/guides/reasoning).
- To configure for more concise responses, set `text.verbosity` to `low`. On GPT-5.5, this will result in proportionally more concise responses than `low` verbosity with GPT-5.5.
- For tool-heavy or long-running workflows, verify that your application handles `phase`, preambles, and assistant-item replay correctly.
- Benchmark against other models on accuracy, token consumption, and end-to-end latency.

### Prompting

- State the expected outcome and success criteria.
- Reduce or remove detailed step-by-step process guidance. Let GPT-5.5 choose the path unless the product requires that path.
- Remove output schema definitions from the prompt where possible. Use [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) instead.
- Optimize your prompt for caching: [static parts first, dynamic parts last](https://developers.openai.com/api/docs/guides/prompt-caching).
- Drop the current date. The model is already aware of the current UTC date.
- Review and optimize your prompts based on the guidance in [Prompting GPT-5.5](https://developers.openai.com/api/docs/guides/prompt-guidance?model=gpt-5.5).

## Using reasoning models

This guidance applies to GPT-5 series models and is worth revisiting whenever teams move workloads onto reasoning models. GPT-5.5 carries forward many capabilities that first appeared in earlier models, but they're still worth reviewing if you are moving from an earlier GPT-5 model, GPT-4.1, or a reasoning model such as o3.

Teams can overlook these features because they sit partly in API configuration and orchestration rather than in the prompt itself. Used together, the Responses API, reasoning controls, verbosity, structured outputs, prompt caching, tool design, hosted tools, and state management help reasoning models deliver their best intelligence, reliability, latency, and cost profile.

- **Responses API:** GPT-5.5 works best in the [Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses). Use `previous_response_id` for multi-turn state handling. For stateless or Zero Data Retention flows, pass back the relevant returned output items each turn. See [Passing context from the previous response](https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response) for details.
- **Reasoning effort:** Use `reasoning.effort` to choose between `low`, `medium`, `high`, or `xhigh`. The default is `medium`, but many workloads will perform well with `low`. Reserve `none` for use cases where low latency is more important than intelligence. See [Reasoning Models](https://developers.openai.com/api/docs/guides/reasoning) for detailed recommendations.
- **Verbosity:** Use `text.verbosity` to control output length. Treat final answer length as separate from reasoning quality; specify word budgets, section counts, table widths, or JSON-only output where needed.
- **Structured Outputs:** Avoid describing the expected output schema in the prompt. Use [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) for automatic validation and increased accuracy.
- **Prompt caching:** [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) works automatically for eligible long prompts and can reduce latency and input-token cost. To maximize cache hits, keep stable content at the beginning of the request. Put dynamic user-specific context near the end. For repeated traffic with common prefixes, use `prompt_cache_key` consistently and track `usage.prompt_tokens_details.cached_tokens`.
- **Tool calling:** GPT-5.5 supports the same tool-calling patterns as GPT-5.5, including function tools and tool-heavy agent workflows. Put most tool-specific guidance in the tool descriptions themselves: what the tool does, when to use it, required inputs, side effects, retry safety, and common error modes. Add tool-specific context to system instructions only when it applies across tools or materially changes the agent's operating policy.
- **Hosted tools and tool search:** Prefer [OpenAI-hosted tools](https://developers.openai.com/api/docs/guides/tools) where they fit the workflow, such as web search, file search, code interpreter, image generation, and computer use. Hosted tools reduce custom orchestration burden and keep common tool patterns aligned with the Responses API and Agents SDK. Use custom function tools when you need to call your own systems, enforce domain-specific side effects, or expose internal business workflows. For large tool catalogs, consider using [tool search](https://developers.openai.com/api/docs/guides/tools-tool-search) to defer tool definitions and load only the relevant subset.
- **Tool preambles:** Preambles can improve chat UX because the user sees an initial, useful status update before the model generates the final response. They also make tool use easier to follow: the model can state what it's about to check or do, then continue from that same assistant state after tool results arrive.
- **`phase` handling:** If your application manually manages Responses state by passing output items back each turn instead of using `previous_response_id`, preserve the `phase` parameter on returned assistant output items and pass it back unchanged. This is especially important when using reasoning effort, preambles, or repeated tool calls. See [Phase parameter](https://developers.openai.com/api/docs/guides/reasoning#phase-parameter).
- **Compaction:** For long-running agents, use [conversation/state compaction](https://developers.openai.com/api/docs/guides/compaction) intentionally. Preserve completed actions, active assumptions, IDs, tool outcomes, unresolved blockers, and the next concrete goal.
- **Agents SDK:** For new agentic systems, use the latest [Agents SDK](https://developers.openai.com/api/docs/guides/agents) patterns for tool orchestration, tracing, handoffs, and state management rather than rebuilding orchestration from scratch.
- **Current date:** GPT-5.5 is aware of the current date in UTC. You don't need to add the current date to system instructions. Add explicit date or timezone context only when the application needs a business-specific timezone, policy-effective date, user-local date, or other non-UTC reference point.