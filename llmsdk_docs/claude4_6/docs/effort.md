# Effort

Control how many tokens Claude uses when responding with the effort parameter, trading off between response thoroughness and token efficiency.

---

The effort parameter allows you to control how eager Claude is about spending tokens when responding to requests. This gives you the ability to trade off between response thoroughness and token efficiency, all with a single model. The effort parameter is generally available on all supported models with no beta header required.

<Note>
  The effort parameter is supported by Claude Opus 4.6, Claude Sonnet 4.6, and Claude Opus 4.5.
</Note>

<Tip>
For Claude Opus 4.6 and Sonnet 4.6, effort replaces `budget_tokens` as the recommended way to control thinking depth. Combine effort with [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) (`thinking: {type: "adaptive"}`) for the best experience. While `budget_tokens` is still accepted on Opus 4.6 and Sonnet 4.6, it is deprecated and will be removed in a future model release. At `high` (default) and `max` effort, Claude will almost always think. At lower effort levels, it may skip thinking for simpler problems.
</Tip>

## How effort works

By default, Claude uses high effort, spending as many tokens as needed for excellent results. You can raise the effort level to `max` for the absolute highest capability, or lower it to be more conservative with token usage, optimizing for speed and cost while accepting some reduction in capability.

<Tip>
Setting `effort` to `"high"` produces exactly the same behavior as omitting the `effort` parameter entirely.
</Tip>

The effort parameter affects **all tokens** in the response, including:

- Text responses and explanations
- Tool calls and function arguments
- Extended thinking (when enabled)

This approach has two major advantages:

1. It doesn't require thinking to be enabled in order to use it.
2. It can affect all token spend including tool calls. For example, lower effort would mean Claude makes fewer tool calls. This gives a much greater degree of control over efficiency.

### Effort levels

| Level    | Description                                                                                                                      | Typical use case                                                                      |
| -------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `max`    | Absolute maximum capability with no constraints on token spending. Opus 4.6 only. Requests using `max` on other models return an error. | Tasks requiring the deepest possible reasoning and most thorough analysis |
| `high`   | High capability. Equivalent to not setting the parameter. | Complex reasoning, difficult coding problems, agentic tasks                           |
| `medium` | Balanced approach with moderate token savings. | Agentic tasks that require a balance of speed, cost, and performance                                                         |
| `low`    | Most efficient. Significant token savings with some capability reduction. | Simpler tasks that need the best speed and lowest costs, such as subagents                     |

<Note>
Effort is a behavioral signal, not a strict token budget. At lower effort levels, Claude will still think on sufficiently difficult problems, but it will think less than it would at higher effort levels for the same problem.
</Note>

### Recommended effort levels for Sonnet 4.6

Sonnet 4.6 defaults to `high` effort. Explicitly set effort when using Sonnet 4.6 to avoid unexpected latency:

- **Medium effort** (recommended default): Best balance of speed, cost, and performance for most applications. Suitable for agentic coding, tool-heavy workflows, and code generation.
- **Low effort:** For high-volume or latency-sensitive workloads. Suitable for chat and non-coding use cases where faster turnaround is prioritized.
- **High effort:** For tasks requiring maximum intelligence from Sonnet 4.6.

## Basic usage

<CodeGroup>
```bash Shell
curl https://api.anthropic.com/v1/messages \
    --header "x-api-key: $ANTHROPIC_API_KEY" \
    --header "anthropic-version: 2023-06-01" \
    --header "content-type: application/json" \
    --data '{
        "model": "claude-opus-4-6",
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": "Analyze the trade-offs between microservices and monolithic architectures"
        }],
        "output_config": {
            "effort": "medium"
        }
    }'
```

```python Python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": "Analyze the trade-offs between microservices and monolithic architectures",
        }
    ],
    output_config={"effort": "medium"},
)

print(response.content[0].text)
```

```typescript TypeScript hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const response = await client.messages.create({
  model: "claude-opus-4-6",
  max_tokens: 4096,
  messages: [
    {
      role: "user",
      content: "Analyze the trade-offs between microservices and monolithic architectures"
    }
  ],
  output_config: {
    effort: "medium"
  }
});

const textBlock = response.content.find(
  (block): block is Anthropic.TextBlock => block.type === "text"
);
console.log(textBlock?.text);
```

```csharp C#
using System;
using System.Threading.Tasks;
using Anthropic;
using Anthropic.Models.Messages;

class Program
{
    static async Task Main(string[] args)
    {
        AnthropicClient client = new();

        var parameters = new MessageCreateParams
        {
            Model = Model.ClaudeOpus4_6,
            MaxTokens = 4096,
            Messages = [new() { Role = Role.User, Content = "Analyze the trade-offs between microservices and monolithic architectures" }],
            OutputConfig = new OutputConfig
            {
                Effort = Effort.Medium
            }
        };

        var message = await client.Messages.Create(parameters);
        Console.WriteLine(message);
    }
}
```

</CodeGroup>

## When to adjust the effort parameter

- Use **max effort** when you need the absolute highest capability with no constraints: the most thorough reasoning and deepest analysis. Only available on Opus 4.6; requests using `max` on other models return an error.
- Use **high effort** (the default) when you need Claude's best work: complex reasoning, nuanced analysis, difficult coding problems, or any task where quality is the top priority.
- Use **medium effort** as a balanced option when you want solid performance without the full token expenditure of high effort.
- Use **low effort** when you're optimizing for speed (because Claude answers with fewer tokens) or cost. For example, simple classification tasks, quick lookups, or high-volume use cases where marginal quality improvements don't justify additional latency or spend.

## Effort with tool use

When using tools, the effort parameter affects both the explanations around tool calls and the tool calls themselves. Lower effort levels tend to:

- Combine multiple operations into fewer tool calls
- Make fewer tool calls
- Proceed directly to action without preamble
- Use terse confirmation messages after completion

Higher effort levels may:

- Make more tool calls
- Explain the plan before taking action
- Provide detailed summaries of changes
- Include more comprehensive code comments

## Effort with extended thinking

The effort parameter works alongside extended thinking. Its behavior depends on the model:

- **Claude Opus 4.6** uses [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) (`thinking: {type: "adaptive"}`), where effort is the recommended control for thinking depth. While `budget_tokens` is still accepted on Opus 4.6, it is deprecated and will be removed in a future release. At `high` and `max` effort, Claude almost always thinks deeply. At lower levels, it may skip thinking for simpler problems.
- **Claude Sonnet 4.6** supports both [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) (where effort controls thinking depth) and manual thinking with [interleaved mode](/docs/en/build-with-claude/extended-thinking#interleaved-thinking) (`thinking: {type: "enabled", budget_tokens: N}`).
- **Claude Opus 4.5 and other Claude 4 models** use manual thinking (`thinking: {type: "enabled", budget_tokens: N}`), where effort works alongside the thinking token budget. Set the effort level for your task, then set the thinking token budget based on task complexity.

The effort parameter can be used with or without extended thinking enabled. When used without thinking, it still controls overall token spend for text responses and tool calls.

## Best practices

1. **Start with high:** Use lower effort levels to trade off performance for token efficiency.
2. **Use low for speed-sensitive or simple tasks:** When latency matters or tasks are straightforward, low effort can significantly reduce response times and costs.
3. **Test your use case:** The impact of effort levels varies by task type. Evaluate performance on your specific use cases before deploying.
4. **Consider dynamic effort:** Adjust effort based on task complexity. Simple queries may warrant low effort while agentic coding and complex reasoning benefit from high effort.
