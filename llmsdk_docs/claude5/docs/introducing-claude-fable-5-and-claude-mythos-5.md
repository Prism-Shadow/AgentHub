# Introducing Claude Fable 5 and Claude Mythos 5

Claude Fable 5 and Claude Mythos 5 capabilities, API changes, and availability.

---

Claude Fable 5 is Anthropic's most capable widely released model, built for the most demanding reasoning and long-horizon agentic work. Claude Mythos 5 shares the same capabilities without the safety classifiers and is available only in limited release through [Project Glasswing](https://anthropic.com/glasswing).

## Models

| Model | API model ID | Description |
|:------|:-------------|:------------|
| Claude Fable 5 | `claude-fable-5` | Anthropic's most capable widely released model, for the most demanding reasoning and long-horizon agentic work |
| Claude Mythos 5 | `claude-mythos-5` | Shares Claude Fable 5's capabilities without the safety classifiers. Available through Project Glasswing. Successor to Claude Mythos Preview. |

Claude Fable 5 and Claude Mythos 5 support a [1M token context window](/docs/en/build-with-claude/context-windows) by default and up to 128k output tokens per request.

Claude Fable 5 and Claude Mythos 5 are priced at $10 per million input tokens and $50 per million output tokens. For specs across all current models, see the [models overview](/docs/en/about-claude/models/overview).

## Refusals, fallback, and billing on Claude Fable 5

Claude Fable 5 includes safety classifiers that can decline certain requests. The sections below summarize what that means for your integration; each links to the full guide.

### Refusals

When Claude Fable 5 declines a request, the Messages API returns `stop_reason: "refusal"` as a successful HTTP 200 response, not an error. The response also reports which classifier declined the request. See [Refusals and fallback](/docs/en/build-with-claude/refusals-and-fallback) for response shapes and handling guidance.

### Fallback

A request that Claude Fable 5 refuses can usually be served by another Claude model. Pass the `fallbacks` parameter to have the API retry for you (in beta on the Claude API and Claude Platform on AWS), or use the SDK middleware (TypeScript, Python, Go, Java, and C#) to retry from the client on any platform. See [Server-side fallback](/docs/en/build-with-claude/refusals-and-fallback#server-side-fallback), [client-side fallback](/docs/en/build-with-claude/refusals-and-fallback#client-side-fallback), and [SDK middleware](/docs/en/cli-sdks-libraries/middleware).

### Billing

You are not billed for a request that is refused before any output is generated. When you retry on another model, fallback credit refunds the prompt-cache cost of switching. See [Fallback credit](/docs/en/build-with-claude/fallback-credit).

## Availability

Claude Fable 5 and Claude Mythos 5 both become available on June 9, 2026:

- **Claude Fable 5** is generally available on the Claude API, [Claude Platform on AWS](/docs/en/build-with-claude/claude-platform-on-aws), [Amazon Bedrock](/docs/en/build-with-claude/claude-in-amazon-bedrock), [Vertex AI](/docs/en/build-with-claude/claude-on-vertex-ai), and [Microsoft Foundry](/docs/en/build-with-claude/claude-in-microsoft-foundry).
- **Claude Mythos 5** is not generally available: it is offered in limited availability to approved customers in [Project Glasswing](https://anthropic.com/glasswing). For access, contact your Anthropic, AWS, or Google Cloud account team. Customers without access to Claude Mythos 5 can use Claude Fable 5, the generally available Mythos-class model.

Claude Fable 5 and Claude Mythos 5 are designated [Covered Models](https://support.claude.com/en/articles/15425695), which carry 30-day data retention and are not available under zero data retention. See [Model-specific data retention requirements](/docs/en/manage-claude/api-and-data-retention#model-specific-data-retention-requirements).

## Working with Claude Fable 5 and Claude Mythos 5

### Prompting

Claude Fable 5 responds to the same prompting techniques as other Claude models, with a few differences in how to structure long-context prompts and reasoning instructions. See [Prompting Claude Fable 5](/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5).

## Messages API on Claude Fable 5 and Claude Mythos 5

<Note>
The behaviors in this section are specific to Claude Fable 5 and Claude Mythos 5. The Messages API is unchanged for Opus, Sonnet, and Haiku models.
</Note>

### Adaptive thinking is always on

[Adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) is the only thinking mode on Claude Fable 5 and Claude Mythos 5. It applies whenever the `thinking` parameter is unset, and `thinking: {"type": "disabled"}` is not supported. Use the [effort parameter](/docs/en/build-with-claude/effort) to control thinking depth.

### Raw thinking content is never returned

The raw chain of thought is never returned on Claude Fable 5 and Claude Mythos 5. `thinking.display` defaults to `"omitted"`, which returns thinking blocks with an empty `thinking` field; set `display: "summarized"` to receive readable summarized thinking. Pass thinking blocks back unchanged in multi-turn conversations on the same model. See [thinking output on Claude Fable 5 and Claude Mythos 5](/docs/en/build-with-claude/adaptive-thinking#thinking-output-on-claude-fable-5-and-claude-mythos-5) for cross-model handling.

## Supported features

At launch, Claude Fable 5 and Claude Mythos 5 support:

- [Effort](/docs/en/build-with-claude/effort)
- [Task budgets](/docs/en/build-with-claude/task-budgets) (beta: set the `task-budgets-2026-03-13` header)
- The [memory tool](/docs/en/agents-and-tools/tool-use/memory-tool)
- Tool result clearing through [context editing](/docs/en/build-with-claude/context-editing) (beta: set the `context-management-2025-06-27` header)
- [Compaction](/docs/en/build-with-claude/compaction)
- [Vision](/docs/en/build-with-claude/vision)

## Migrating from earlier models

If you're migrating from Claude Mythos Preview, see [Migrating from Claude Mythos Preview to Claude Mythos 5](/docs/en/about-claude/models/migration-guide#migrating-from-claude-mythos-preview) for step-by-step instructions.

If you're migrating from Claude Opus 4.8, see [Migrating from Claude Opus 4.8 to Claude Fable 5](/docs/en/about-claude/models/migration-guide#migrating-from-claude-opus-48).

## Next steps

<CardGroup>
  <Card title="Migration guide" icon="arrow-right" href="/docs/en/about-claude/models/migration-guide">
    Step-by-step upgrade instructions from Claude Opus 4.8 and Claude Mythos Preview.
  </Card>
  <Card title="Models overview" icon="settings" href="/docs/en/about-claude/models/overview">
    Specs and comparison for all current Claude models.
  </Card>
  <Card title="Adaptive thinking" icon="brain" href="/docs/en/build-with-claude/adaptive-thinking">
    The only thinking mode on Claude Fable 5 and Claude Mythos 5.
  </Card>
  <Card title="Refusals and fallback" icon="shield" href="/docs/en/build-with-claude/refusals-and-fallback">
    How Claude Fable 5 declines requests, and how to retry on another model.
  </Card>
  <Card title="Fallback credit" icon="coins" href="/docs/en/build-with-claude/fallback-credit">
    Avoid paying the prompt-cache cost twice on a retry.
  </Card>
  <Card title="Fallback and billing cookbook" icon="book-open" href="https://platform.claude.com/cookbook/fable-5-fallback-billing-guide">
    A worked end-to-end example of refusal handling, fallback, and billing.
  </Card>
  <Card title="Effort" icon="sliders" href="/docs/en/build-with-claude/effort">
    Control thinking depth and cost on Claude Fable 5 and Claude Mythos 5.
  </Card>
  <Card title="Prompting Claude Fable 5" icon="terminal" href="/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5">
    Fable-specific prompting techniques.
  </Card>
</CardGroup>
