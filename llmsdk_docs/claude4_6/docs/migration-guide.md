# Migration guide

Guide for migrating to Claude 4.6 models from previous Claude versions

---

## Migrating to Claude 4.6

Claude Opus 4.6 is a near drop-in replacement for Claude 4.5, with a few breaking changes to be aware of. For a full list of new features, see [What's new in Claude 4.6](/docs/en/about-claude/models/whats-new-claude-4-6).

### Update your model name

```python
# Opus migration
model = "claude-opus-4-5"  # Before
model = "claude-opus-4-6"  # After
```

### Breaking changes

1. **Prefill removal:** Prefilling assistant messages returns a 400 error on Claude 4.6 models. Use [structured outputs](/docs/en/build-with-claude/structured-outputs), system prompt instructions, or `output_config.format` instead.

2. **Tool parameter quoting:** Claude 4.6 models may produce slightly different JSON string escaping in tool call arguments (e.g., different handling of Unicode escapes or forward slash escaping). If you parse tool call `input` as a raw string rather than using a JSON parser, verify your parsing logic. Standard JSON parsers (like `json.loads()` or `JSON.parse()`) handle these differences automatically.

### Recommended changes

These are not required but will improve your experience:

1. **Migrate to adaptive thinking:** `thinking: {type: "enabled", budget_tokens: N}` is deprecated on Claude 4.6 models and will be removed in a future model release. Switch to `thinking: {type: "adaptive"}` and use the [effort parameter](/docs/en/build-with-claude/effort) to control thinking depth. See [Adaptive thinking](/docs/en/build-with-claude/adaptive-thinking).

   <CodeGroup>

   ```python Before nocheck
   response = client.beta.messages.create(
       model="claude-opus-4-5",
       max_tokens=16000,
       thinking={"type": "enabled", "budget_tokens": 32000},
       betas=["interleaved-thinking-2025-05-14"],
       messages=[...],
   )
   ```


   ```python After nocheck
   response = client.messages.create(
       model="claude-opus-4-6",
       max_tokens=16000,
       thinking={"type": "adaptive"},
       output_config={"effort": "high"},
       messages=[{"role": "user", "content": "Your prompt here"}],
   )
   ```

   ```typescript TypeScript hidelines={1..4}
   import Anthropic from "@anthropic-ai/sdk";

   const client = new Anthropic();

   const response = await client.messages.create({
     model: "claude-opus-4-6",
     max_tokens: 16000,
     thinking: { type: "adaptive" },
     output_config: { effort: "high" },
     messages: [{ role: "user", content: "Your prompt here" }]
   } as unknown as Anthropic.MessageCreateParamsNonStreaming);
   ```

   ```csharp C#
   using Anthropic;
   using Anthropic.Models.Messages;

   public class Program
   {
       public static async Task Main(string[] args)
       {
           AnthropicClient client = new();

           var parameters = new MessageCreateParams
           {
               Model = Model.ClaudeOpus4_6,
               MaxTokens = 16000,
               Thinking = new ThinkingConfigAdaptive(),
               OutputConfig = new OutputConfig { Effort = Effort.High },
               Messages = [new() { Role = Role.User, Content = "Your prompt here" }]
           };

           var response = await client.Messages.Create(parameters);
           Console.WriteLine(response);
       }
   }
   ```
   </CodeGroup>

   Note that the migration also moves from `client.beta.messages.create` to `client.messages.create`. Adaptive thinking and effort are GA features and do not require the beta SDK namespace or any beta headers.

2. **Remove effort beta header:** The effort parameter is now GA. Remove `betas=["effort-2025-11-24"]` from your requests.

3. **Remove fine-grained tool streaming beta header:** Fine-grained tool streaming is now GA. Remove `betas=["fine-grained-tool-streaming-2025-05-14"]` from your requests.

4. **Remove interleaved thinking beta header (Opus 4.6 only):** Adaptive thinking automatically enables interleaved thinking on Opus 4.6. Remove `betas=["interleaved-thinking-2025-05-14"]` from your Opus 4.6 requests. Note: Sonnet 4.6 continues to support this beta header with manual extended thinking.

5. **Migrate to output_config.format:** If using structured outputs, update `output_format={...}` to `output_config={"format": {...}}`. The old parameter remains functional but is deprecated and will be removed in a future model release.

### Migrating from Claude 4.1 or earlier to Claude 4.6

If you're migrating from Opus 4.1, Sonnet 4, or earlier models directly to Claude 4.6, apply the Claude 4.6 breaking changes above plus the additional changes in this section.

```python
# From Opus 4.1
model = "claude-opus-4-1-20250805"  # Before
model = "claude-opus-4-6"  # After

# From Sonnet 4
model = "claude-sonnet-4-20250514"  # Before
model = "claude-opus-4-6"  # After

# From Sonnet 3.7
model = "claude-3-7-sonnet-20250219"  # Before
model = "claude-opus-4-6"  # After
```

#### Additional breaking changes

1. **Update sampling parameters**

   <Warning>
   This is a breaking change when migrating from Claude 3.x models.
   </Warning>

   Use only `temperature` OR `top_p`, not both:


   ```python nocheck
   # Before - This will error in Claude 4+ models
   response = client.messages.create(
       model="claude-3-7-sonnet-20250219",
       temperature=0.7,
       top_p=0.9,  # Cannot use both
       # ...
   )

   # After
   response = client.messages.create(
       model="claude-opus-4-6",
       temperature=0.7,  # Use temperature OR top_p, not both
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


   ```python nocheck
   response = client.messages.create(...)

   if response.stop_reason == "refusal":
       # Handle refusal appropriately
       pass
   ```

4. **Handle the `model_context_window_exceeded` stop reason**

   Claude 4.5+ models return a `model_context_window_exceeded` stop reason when generation stops due to hitting the context window limit, rather than the requested `max_tokens` limit. Update your application to handle this new stop reason:


   ```python nocheck
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

### Claude 4.6 migration checklist

- [ ] Update model ID to `claude-opus-4-6`
- [ ] **BREAKING:** Remove assistant message prefills (returns 400 error); use structured outputs or `output_config.format` instead
- [ ] **Recommended:** Migrate from `thinking: {type: "enabled", budget_tokens: N}` to `thinking: {type: "adaptive"}` with the [effort parameter](/docs/en/build-with-claude/effort) (`budget_tokens` is deprecated and will be removed in a future release)
- [ ] Verify tool call JSON parsing uses a standard JSON parser
- [ ] Remove `effort-2025-11-24` beta header (effort is now GA)
- [ ] Remove `fine-grained-tool-streaming-2025-05-14` beta header
- [ ] Remove `interleaved-thinking-2025-05-14` beta header (Opus 4.6 only; Sonnet 4.6 still supports it)
- [ ] Migrate `output_format` to `output_config.format` (if applicable)
- [ ] If migrating from Claude 4.1 or earlier: update sampling parameters to use only `temperature` OR `top_p`
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
```bash Shell
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

```python Python nocheck
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
    }
}
```
</CodeGroup>

#### If you're using extended thinking

If you're using extended thinking on Sonnet 4.5, it continues to be supported on Sonnet 4.6 with no changes needed to your thinking configuration. Consider keeping a thinking budget around 16k tokens. In practice, most tasks don't use that much, but it provides headroom for harder problems without risk of runaway token usage.

##### Coding and agentic use cases

For agentic coding, frontend design, tool-heavy workflows, and complex enterprise workflows, start with `medium` effort. If you find latency is too high, consider reducing effort to `low`. If you need higher intelligence, consider increasing effort to `high` or migrating to Opus 4.6.

<CodeGroup>
```bash Shell
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

```python Python nocheck
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
            Model = "claude-sonnet-4-6",
            MaxTokens = 16384,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 16384),
            OutputConfig = new OutputConfig
            {
                Effort = Effort.Medium
            },
            Messages = [new() { Role = Role.User, Content = "Your prompt here" }]
        };

        var message = await client.Messages.Create(parameters);
        Console.WriteLine(message);
    }
}
```
</CodeGroup>

##### Chat and non-coding use cases

For chat, content generation, search, classification, and other non-coding tasks, start with `low` effort with extended thinking. If you need more depth, increase effort to `medium`.

<CodeGroup>
```bash Shell
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

```python Python nocheck
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
            Model = "claude-sonnet-4-6",
            MaxTokens = 8192,
            Thinking = new ThinkingConfigEnabled(budgetTokens: 16384),
            OutputConfig = new OutputConfig
            {
                Effort = Effort.Low
            },
            Messages = [new() { Role = Role.User, Content = "Your prompt here" }]
        };

        var message = await client.Messages.Create(parameters);
        Console.WriteLine(message);
    }
}
```
</CodeGroup>

##### When to try adaptive thinking

The migration paths above use extended thinking with `budget_tokens` for predictable token usage. If your workload fits one of the following patterns, consider trying [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) instead:

- **Autonomous multi-step agents:** coding agents that turn requirements into working software, data analysis pipelines, and bug finding where the model runs independently across many steps. Adaptive thinking lets the model calibrate its reasoning per step, staying on path over longer trajectories. For these workloads, start at `high` effort. If latency or token usage is a concern, scale down to `medium`.
- **Computer use agents:** Sonnet 4.6 achieved best-in-class accuracy on computer use evaluations using adaptive mode.
- **Bimodal workloads:** a mix of easy and hard tasks where adaptive skips thinking on simple queries and reasons deeply on complex ones.

When using adaptive thinking, evaluate `medium` and `high` effort on your tasks. The right level depends on your workload's tradeoff between quality, latency, and token usage.

<CodeGroup>
```bash Shell
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

```csharp C#
using Anthropic;
using Anthropic.Models.Messages;
using System;
using System.Threading.Tasks;

class Program
{
    static async Task Main(string[] args)
    {
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
    }
}
```
</CodeGroup>

<Note>
If you see inconsistent behavior or quality regressions with adaptive thinking, switch to extended thinking with `budget_tokens`. This provides more predictable results with a cap on thinking costs.
</Note>

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
- [ ] Consider enabling extended thinking or adaptive thinking for complex reasoning tasks
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

# From Haiku 3
model = "claude-3-haiku-20240307"  # Before
model = "claude-haiku-4-5-20251001"  # After
```

**Review new rate limits:** Haiku 4.5 has separate rate limits from Haiku 3.5 and Haiku 3. See [Rate limits documentation](/docs/en/api/rate-limits) for details.

<Tip>
For significant performance improvements on coding and reasoning tasks, consider enabling extended thinking with `thinking: {type: "enabled", budget_tokens: N}`.
</Tip>

<Note>
Extended thinking impacts [prompt caching](/docs/en/build-with-claude/prompt-caching#caching-with-thinking-blocks) efficiency.

Extended thinking is deprecated in Claude 4.6 or newer models. If using newer models, use [adaptive thinking](/docs/en/build-with-claude/adaptive-thinking) instead.
</Note>

**Explore new capabilities:** See the [models overview](/docs/en/about-claude/models/overview) for details on context awareness, increased output capacity (64K tokens), higher intelligence, and improved speed.

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
