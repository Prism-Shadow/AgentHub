# Using GPT-5.4

import {
  Bolt,
  Code,
  Sparkles,
} from "@components/react/oai/platform/ui/Icon.react";




export const fastResponses = <>

GPT-5.4 has a new reasoning mode: `none` for low-latency interactions. By default, GPT-5.4 reasoning is set to `none`.

<br />
<br />

This behavior will more closely (but not exactly!) match non-reasoning models like&nbsp;

[GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1). We expect GPT-5.4 to produce
more intelligent responses than GPT-4.1, but when speed and maximum context
length are paramount, you might consider using GPT-4.1 instead.

Fast, low latency response options

```javascript
import OpenAI from "openai";
const openai = new OpenAI();

const result = await openai.responses.create({
  model: "gpt-5.4",
  input: "Write a haiku about code.",
  reasoning: { effort: "low" },
  text: { verbosity: "low" },
});

console.log(result.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

result = client.responses.create(
    model="gpt-5.4",
    input="Write a haiku about code.",
    reasoning={ "effort": "low" },
    text={ "verbosity": "low" },
)

print(result.output_text)
```

```bash
curl https://api.openai.com/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $OPENAI_API_KEY" \\
  -d '{
    "model": "gpt-5.4",
    "input": "Write a haiku about code.",
    "reasoning": { "effort": "low" }
  }'
```


</>;

export const goodResponses = <>

GPT-5.4 is great at reasoning through complex tasks. <strong>For complex tasks like coding and multi-step planning,
use high reasoning effort.</strong>

<br />
<br />

Use these configurations when replacing tasks you might have used o3 to tackle.
We expect GPT-5.4 to produce better results than o3 and o4-mini under most circumstances.

Slower, high reasoning tasks

```javascript
import OpenAI from "openai";
const openai = new OpenAI();

const result = await openai.responses.create({
  model: "gpt-5.4",
  input: "Find the null pointer exception: ...your code here...",
  reasoning: { effort: "high" },
});

console.log(result.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

result = client.responses.create(
    model="gpt-5.4",
    input="Find the null pointer exception: ...your code here...",
    reasoning={ "effort": "high" },
)

print(result.output_text)
```

```bash
curl https://api.openai.com/v1/responses \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $OPENAI_API_KEY" \\
  -d '{
    "model": "gpt-5.4",
    "input": "Find the null pointer exception: ...your code here...",
    "reasoning": { "effort": "high" }
  }'
```


</>;

[GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4) is our most capable frontier model yet, delivering higher-quality outputs with fewer iterations across ChatGPT, the API, and Codex. It helps people and teams analyze complex information, build production software, and automate multi-step workflows.

In practice, `gpt-5.4` is the default model for both broad general-purpose work and most coding tasks. Start there when you want one model that can move between software engineering, reasoning, writing, and tool use in the same workflow.

This guide covers key features of the GPT-5 model family and how to get the most out of GPT-5.4.

## Key improvements

Compared with the previous GPT-5.2 model, GPT-5.4 shows improvements in:

- Coding, document understanding, tool use, and instruction following
- Image perception and multimodal tasks
- Long-running task execution and multi-step agent workflows
- Token efficiency and end-to-end performance on tool-heavy workloads
- Agentic web search and multi-source synthesis, especially for hard-to-locate information
- Document-heavy and spreadsheet-heavy business workflows in customer service, analytics, and finance

GPT-5.4 brings the coding capabilities of GPT-5.3-Codex to our flagship frontier model. Developers can generate production-quality code, build polished front-end UI, follow repo-specific patterns, and handle multi-file changes with fewer retries. It also has a strong out-of-the-box coding personality, so teams spend less time on prompt tuning.

For agentic workloads, GPT-5.4 reduces end-to-end time across multi-step trajectories and often completes tasks with fewer tokens and tool calls. This makes agents more responsive and lowers the cost of operating complex workflows at scale in the API and Codex.

### New features in GPT-5.4

Like earlier GPT-5 models, GPT-5.4 supports custom tools, parameters to control verbosity and reasoning, and an allowed tools list. GPT-5.4 also introduces several capabilities that make it easier to build powerful agent systems, operate over larger bodies of information, and run more reliable automated workflows:

- **`tool_search` in the API:** GPT-5.4 improves tool search for larger tool ecosystems by using deferred tool loading. This makes tools searchable, loads only the relevant definitions, reduces token usage, and improves tool selection accuracy in real deployments. Learn more in the [tool search guide](https://developers.openai.com/api/docs/guides/tools-tool-search).
- **1M token context window:** GPT-5.4 supports up to a 1M token context window, making it easier to analyze entire codebases, long document collections, or extended agent trajectories in a single request. Read more in the [1M context window](#1m-context-window) section.
- **Built-in computer use:** GPT-5.4 is the first mainline model with built-in computer-use capabilities, enabling agents to interact directly with software to complete, verify, and fix tasks in a build-run-verify-fix loop. Learn more in the [computer use guide](https://developers.openai.com/api/docs/guides/tools-computer-use).
- **Native compaction support:** GPT-5.4 is the first mainline model trained to support compaction, enabling longer agent trajectories while preserving key context.

## Meet the models

In general, `gpt-5.4` is the default model for your most important work across both general-purpose tasks and coding. It replaces the previous `gpt-5.2` model in the API, and `gpt-5.3-codex` in Codex. The model powering ChatGPT is `gpt-5-chat-latest`. For more difficult problems, `gpt-5.4-pro` uses more compute to think longer and provide consistently better answers.

For a smaller model, use `gpt-5-mini`.

To help you pick the model that best fits your use case, consider these tradeoffs:

| Variant                                       | Best for                                                                                                             |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [`gpt-5.4`](https://developers.openai.com/api/docs/models/gpt-5.4)         | General-purpose work, including complex reasoning, broad world knowledge, and code-heavy or multi-step agentic tasks |
| [`gpt-5.4-pro`](https://developers.openai.com/api/docs/models/gpt-5.4-pro) | Tough problems that may take longer to solve and need deeper reasoning                                               |
| [`gpt-5-mini`](https://developers.openai.com/api/docs/models/gpt-5-mini)   | Cost-optimized reasoning and chat; balances speed, cost, and capability                                              |
| [`gpt-5-nano`](https://developers.openai.com/api/docs/models/gpt-5-nano)   | High-throughput tasks, especially straightforward instruction-following or classification                            |

### Lower reasoning effort

The `reasoning.effort` parameter controls how many reasoning tokens the model generates before producing a response. Earlier reasoning models like o3 supported only `low`, `medium`, and `high`: `low` favored speed and fewer tokens, while `high` favored more thorough reasoning.

Starting with GPT-5.2, the lowest setting is `none` to provide lower-latency interactions. This is the default setting in GPT-5.2 and newer models. If you need more thinking, slowly increase to `medium` and experiment with results.

With reasoning effort set to `none`, prompting is important. To improve the model's reasoning quality, even with the default settings, encourage it to “think” or outline its steps before answering.

Reasoning effort set to none

```bash
curl --request POST \
  --url https://api.openai.com/v1/responses \
  --header "Authorization: Bearer $OPENAI_API_KEY" \
  --header 'Content-type: application/json' \
  --data '{
        "model": "gpt-5.4",
        "input": "How much gold would it take to coat the Statue of Liberty in a 1mm layer?",
        "reasoning": {
                "effort": "none"
        }
}'
```

```javascript
import OpenAI from "openai";
const openai = new OpenAI();

const response = await openai.responses.create({
  model: "gpt-5.4",
  input: "How much gold would it take to coat the Statue of Liberty in a 1mm layer?",
  reasoning: {
    effort: "none"
  }
});

console.log(response);
```

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5.4",
    input="How much gold would it take to coat the Statue of Liberty in a 1mm layer?",
    reasoning={
        "effort": "none"
    }
)

print(response)
```


### Verbosity

Verbosity determines how many output tokens are generated. Lowering the number of tokens reduces overall latency. While the model's reasoning approach stays mostly the same, the model finds ways to answer more concisely—which can either improve or diminish answer quality, depending on your use case. Here are some scenarios for both ends of the verbosity spectrum:

- **High verbosity:** Use when you need the model to provide thorough explanations of documents or perform extensive code refactoring.
- **Low verbosity:** Best for situations where you want concise answers or simple code generation, such as SQL queries.

GPT-5 made this option configurable as one of `high`, `medium`, or `low`. With GPT-5.4, verbosity remains configurable and defaults to `medium`.

When generating code with GPT-5.4, `medium` and `high` verbosity levels yield longer, more structured code with inline explanations, while `low` verbosity produces shorter, more concise code with minimal commentary.

Control verbosity

```bash
curl --request POST \
  --url https://api.openai.com/v1/responses \
  --header "Authorization: Bearer $OPENAI_API_KEY" \
  --header 'Content-type: application/json' \
  --data '{
  "model": "gpt-5.4",
  "input": "What is the answer to the ultimate question of life, the universe, and everything?",
  "text": {
    "verbosity": "low"
  }
}'
```

```javascript
import OpenAI from "openai";
const openai = new OpenAI();

const response = await openai.responses.create({
  model: "gpt-5.4",
  input: "What is the answer to the ultimate question of life, the universe, and everything?",
  text: {
    verbosity: "low"
  }
});

console.log(response);
```

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-5.4",
    input="What is the answer to the ultimate question of life, the universe, and everything?",
    text={
        "verbosity": "low"
    }
)

print(response)
```


You can still steer verbosity through prompting after setting it to `low` in the API. The verbosity parameter defines a general token range at the system prompt level, but the actual output is flexible to both developer and user prompts within that range.

#### 1M context window

1M token context window was introduced with GPT-5.4, making it easier to analyze entire codebases, long document collections, or extended agent trajectories in a single request.

We have separate standard pricing for requests under 272K and over 272K tokens, available in the [pricing docs](https://developers.openai.com/api/docs/pricing). If you use [priority processing](https://developers.openai.com/api/docs/guides/priority-processing), any prompt above 272K tokens is automatically processed at standard rates.

Long context pricing stacks with other pricing modifiers such as data residency and batch.

We have different rate limits for requests under 272K tokens and over 272K tokens; this is available on the [GPT-5.4 model page](https://developers.openai.com/api/docs/models/gpt-5.4).

## Using tools with GPT-5.4

GPT-5.4 has been post-trained on specific tools. See the [tools docs](https://developers.openai.com/api/docs/guides/tools) for more specific guidance.

### Computer use tool

Computer use lets GPT-5.4 operate software through the user interface by inspecting screenshots and returning structured actions for your harness to execute. It is a good fit for browser or desktop workflows where a person could complete the task through the UI, such as navigating a site, filling out forms, or validating that a change actually worked.

Use it in an isolated browser or VM, and keep a human in the loop for high-impact actions. The full guide covers the built-in Responses API loop, custom harness patterns, and code-execution-based setups.

[

<span slot="icon">
      </span>
    Learn how to run the built-in computer tool safely and integrate it with
    your own harness.

](https://developers.openai.com/api/docs/guides/tools-computer-use)

### Tool search tool

Tool search lets GPT-5.4 defer large tool surfaces until runtime so the model loads only the definitions it needs. This is most useful when you have many functions, namespaces, or MCP tools and want to reduce token usage, preserve cache performance, and improve latency without exposing every schema up front.

Use hosted tool search when the candidate tools are already known at request time, or client-executed tool search when your application needs to decide what to load dynamically. The full guide also covers best practices for namespaces, MCP servers, and deferred loading.

[

<span slot="icon">
      </span>
    Learn how to defer tool definitions and load the right subset at runtime.

](https://developers.openai.com/api/docs/guides/tools-tool-search)

### Custom tools

When the GPT-5 model family launched, we introduced a new capability called custom tools, which lets models send any raw text as tool call input but still constrain outputs if desired. This tool behavior remains true in GPT-5.4.

[

<span slot="icon">
      </span>
    Learn about custom tools in the function calling guide.

](https://developers.openai.com/api/docs/guides/function-calling)

#### Freeform inputs

Define your tool with `type: custom` to enable models to send plaintext inputs directly to your tools, rather than being limited to structured JSON. The model can send any raw text—code, SQL queries, shell commands, configuration files, or long-form prose—directly to your tool.

```bash
{
    "type": "custom",
    "name": "code_exec",
    "description": "Executes arbitrary python code",
}

```

#### Constraining outputs

GPT-5.4 supports context-free grammars (CFGs) for custom tools, letting you provide a Lark grammar to constrain outputs to a specific syntax or DSL. Attaching a CFG (e.g., a SQL or DSL grammar) ensures the assistant's text matches your grammar.

This enables precise, constrained tool calls or structured responses and lets you enforce strict syntactic or domain-specific formats directly in GPT-5.4's function calling, improving control and reliability for complex or constrained domains.

#### Best practices for custom tools

- **Write concise, explicit tool descriptions**. The model chooses what to send based on your description; state clearly if you want it to always call the tool.
- **Validate outputs on the server side**. Freeform strings are powerful but require safeguards against injection or unsafe commands.

### Allowed tools

The `allowed_tools` parameter under `tool_choice` lets you pass N tool definitions but restrict the model to only M (&lt; N) of them. List your full toolkit in `tools`, and then use an `allowed_tools` block to name the subset and specify a mode—either `auto` (the model may pick any of those) or `required` (the model must invoke one).

[

<span slot="icon">
      </span>
    Learn about the allowed tools option in the function calling guide.

](https://developers.openai.com/api/docs/guides/function-calling)

By separating all possible tools from the subset that can be used _now_, you gain greater safety, predictability, and improved prompt caching. You also avoid brittle prompt engineering, such as hard-coded call order. GPT-5.4 dynamically invokes or requires specific functions mid-conversation while reducing the risk of unintended tool usage over long contexts.

|                  | **Standard Tools**                        | **Allowed Tools**                                             |
| ---------------- | ----------------------------------------- | ------------------------------------------------------------- |
| Model's universe | All tools listed under **`"tools": […]`** | Only the subset under **`"tools": […]`** in **`tool_choice`** |
| Tool invocation  | Model may or may not call any tool        | Model restricted to (or required to call) chosen tools        |
| Purpose          | Declare available capabilities            | Constrain which capabilities are actually used                |

```bash
  "tool_choice": {
    "type": "allowed_tools",
    "mode": "auto",
    "tools": [
      { "type": "function", "name": "get_weather" },
      { "type": "function", "name": "search_docs" }
    ]
  }
}'
```

For a more detailed overview of all of these new features, see the [prompt guidance for GPT-5.4](https://developers.openai.com/api/docs/guides/prompt-guidance).

### Preambles

Preambles are brief, user-visible explanations that GPT-5.4 generates before invoking any tool or function, outlining its intent or plan (e.g., “why I'm calling this tool”). They appear after the chain-of-thought and before the actual tool call, providing transparency into the model's reasoning and enhancing debuggability, user confidence, and fine-grained steerability.

By letting GPT-5.4 “think out loud” before each tool call, preambles boost tool-calling accuracy (and overall task success) without bloating reasoning overhead. To enable preambles, add a system or developer instruction—for example: “Before you call a tool, explain why you are calling it.” GPT-5.4 prepends a concise rationale to each specified tool call. The model may also output multiple messages between tool calls, which can enhance the interaction experience—particularly for minimal reasoning or latency-sensitive use cases.

For more on using preambles, see the [GPT-5 prompting cookbook](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide#tool-preambles).

## Migration guidance

GPT-5.4 is our best model yet, and it works best with the Responses API, which supports passing chain of thought (CoT) between turns to improve performance. Read below to migrate from your current model or API.

### Migrating from other models to GPT-5.4

Use the [OpenAI Docs
  skill](https://github.com/openai/skills/tree/main/skills/.system/openai-docs)
  when migrating existing prompts or workflows to GPT-5.4. It's available in our
  public skills repository and the Codex desktop app.

While the model should be close to a drop-in replacement for GPT-5.2, there are a few key changes to call out. See [Prompt guidance for GPT-5.4](https://developers.openai.com/api/docs/guides/prompt-guidance) for specific updates to make in your prompts.

Using GPT-5 models with the Responses API provides improved intelligence because of the API's design. The Responses API can pass the previous turn's CoT to the model. This leads to fewer generated reasoning tokens, higher cache hit rates, and less latency. To learn more, see an [in-depth guide](https://developers.openai.com/cookbook/examples/responses_api/reasoning_items) on the benefits of the Responses API.

When migrating to GPT-5.4 from an older OpenAI model, start by experimenting with reasoning levels and prompting strategies. Based on our testing, we recommend using our [prompt optimizer](https://platform.openai.com/chat/edit?models=gpt-5.4&optimize=true)—which automatically updates your prompts for GPT-5.4 based on our best practices—and following this model-specific guidance:

- **gpt-5.2**: `gpt-5.4` with default settings is meant to be a drop-in replacement.
- **o3**: `gpt-5.4` with `medium` or `high` reasoning. Start with `medium` reasoning with prompt tuning, then increase to `high` if you aren't getting the results you want.
- **gpt-4.1**: `gpt-5.4` with `none` reasoning. Start with `none` and tune your prompts; increase if you need better performance.
- **o4-mini or gpt-4.1-mini**: `gpt-5-mini` with prompt tuning is a great replacement.
- **gpt-4.1-nano**: `gpt-5-nano` with prompt tuning is a great replacement.

### New `phase` parameter

For long-running or tool-heavy GPT-5.4 flows in the Responses API, use the assistant message `phase` field to avoid early stopping and other misbehavior.

`phase` is optional at the API level, but we highly recommend using it. Use `phase: "commentary"` for intermediate assistant updates (such as preambles before tool calls) and `phase: "final_answer"` for the completed answer. Do not add `phase` to user messages.

If you use `previous_response_id`, that is usually the simplest path because
  prior assistant state is preserved. If you replay assistant history manually,
  preserve each original `phase` value.

Missing or dropped `phase` can cause preambles to be treated as final answers, which can degrade quality on multi-step tasks. For additional guidance and examples, see the [GPT-5.4 prompting guide](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5-4_prompting_guide#phase).

Round-trip assistant phase values

```javascript
import OpenAI from "openai";
const client = new OpenAI();

const response = await client.responses.create({
  model: "gpt-5.4",
  input: [
    {
      role: "assistant",
      phase: "commentary",
      content:
        "I’ll inspect the logs and then summarize root cause and remediation.",
    },
    {
      role: "assistant",
      phase: "final_answer",
      content: "Root cause: cache invalidation race.",
    },
    {
      role: "user",
      content: "Great—now give me a rollout-safe fix plan.",
    },
  ],
});

console.log(response.output_text);
```

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.4",
    input=[
        {
            "role": "assistant",
            "phase": "commentary",
            "content": "I’ll inspect the logs and then summarize root cause and remediation.",
        },
        {
            "role": "assistant",
            "phase": "final_answer",
            "content": "Root cause: cache invalidation race.",
        },
        {
            "role": "user",
            "content": "Great—now give me a rollout-safe fix plan.",
        },
    ],
)

print(response.output_text)
```


### GPT-5.4 parameter compatibility

The following parameters are **only supported** when using GPT-5.4 with reasoning effort set to `none`:

- `temperature`
- `top_p`
- `logprobs`

Requests to GPT-5.4 or GPT-5.2 with any other reasoning effort setting, or to older GPT-5 models (e.g., `gpt-5`, `gpt-5-mini`, `gpt-5-nano`) that include these fields will raise an error.

To achieve similar results with reasoning effort set higher, or with another GPT-5 family model, try these alternative parameters:

- **Reasoning depth:** `reasoning: { effort: "none" | "low" | "medium" | "high" | "xhigh" }`
- **Output verbosity:** `text: { verbosity: "low" | "medium" | "high" }`
- **Output length:** `max_output_tokens`

### Migrating from Chat Completions to Responses API

The biggest difference, and main reason to migrate from Chat Completions to the Responses API for GPT-5.4, is support for passing chain of thought (CoT) between turns. See a full [comparison of the APIs](https://developers.openai.com/api/docs/guides/responses-vs-chat-completions).

Passing CoT exists only in the Responses API, and we've seen improved intelligence, fewer generated reasoning tokens, higher cache hit rates, and lower latency as a result of doing so. Most other parameters remain at parity, though the formatting is different. Here's how new parameters are handled differently between Chat Completions and the Responses API:

**Reasoning effort**



<div data-content-switcher-pane data-value="responses">
    <div class="hidden">Responses API</div>
    </div>
  <div data-content-switcher-pane data-value="chat" hidden>
    <div class="hidden">Chat Completions</div>
    </div>



**Verbosity**



<div data-content-switcher-pane data-value="responses">
    <div class="hidden">Responses API</div>
    </div>
  <div data-content-switcher-pane data-value="chat" hidden>
    <div class="hidden">Chat Completions</div>
    </div>



**Custom tools**



<div data-content-switcher-pane data-value="responses">
    <div class="hidden">Responses API</div>
    </div>
  <div data-content-switcher-pane data-value="chat" hidden>
    <div class="hidden">Chat Completions</div>
    </div>



## Prompting guidance

We specifically designed GPT-5.4 to excel at coding and agentic tasks. We also recommend iterating on prompts for GPT-5.4 with the prompt optimizer.

<div className="mt-4 flex flex-col gap-2">
  [

<span slot="icon">
        </span>
      Craft the perfect prompt for GPT-5.4 in the dashboard

](https://platform.openai.com/chat/edit?optimize=true)

[

<span slot="icon">
      </span>
    Learn prompt patterns and migration tips for GPT-5.4

](https://developers.openai.com/api/docs/guides/prompt-guidance)

  <a href="https://cookbook.openai.com/examples/gpt-5/gpt-5_frontend">


<span slot="icon">
        </span>
      See prompt samples specific to frontend development for GPT-5 family of
      models


  </a>
</div>

### GPT-5.4 is a reasoning model

Reasoning models like GPT-5.4 break problems down step by step, producing an internal chain of thought that encodes their reasoning. To maximize performance, pass these reasoning items back to the model: this avoids re-reasoning and keeps interactions closer to the model's training distribution. In multi-turn conversations, passing a `previous_response_id` automatically makes earlier reasoning items available. This is especially important when using tools—for example, when a function call requires an extra round trip. In these cases, either include them with `previous_response_id` or add them directly to `input`.

Learn more about reasoning models and how to get the most out of them in our [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning).

## Further reading

[GPT-5.4 prompting guide](https://developers.openai.com/api/docs/guides/prompt-guidance)

[GPT-5.3-Codex prompting guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide)

[GPT-5.4 blog post](https://openai.com/index/introducing-gpt-5-4/)

[GPT-5 frontend guide](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_frontend)

[GPT-5 model family: new features guide](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_new_params_and_tools)

[Cookbook on reasoning models](https://developers.openai.com/cookbook/examples/responses_api/reasoning_items)

[Comparison of Responses API vs. Chat Completions](https://developers.openai.com/api/docs/guides/migrate-to-responses)

## FAQ

1. **How are these models integrated into ChatGPT?**

   In ChatGPT, there are three models: GPT‑5 Instant, GPT‑5 Thinking, and GPT‑5 Pro. Based on the user's question, a routing layer selects the best model to use. Users can also invoke reasoning directly through the ChatGPT UI.

   All three ChatGPT models (Instant, Thinking, and Pro) have a new knowledge cutoff of August 2025. For users, this means GPT-5.4 starts with a more current understanding of the world, so answers are more accurate and useful, with more relevant examples and context, even before turning to web search.

1. **Will these models be supported in Codex?**

   Yes, `gpt-5.4` is the newest model that powers Codex and Codex CLI. You can also use this as a standalone model for building agentic coding applications.

1. **How does GPT-5.4 compare to GPT-5.3-Codex?**

   [GPT-5.3-Codex](https://developers.openai.com/api/docs/models/gpt-5.3-codex) is specifically designed for use in coding environments such as Codex. GPT-5.4 is designed for both general-purpose work and coding, making it the better default when your workflow spans software engineering plus planning, writing, or other business tasks. GPT-5.3-Codex is only available in the Responses API and supports `low`, `medium`, `high`, and `xhigh` reasoning effort settings along with function calling, structured outputs, streaming, and prompt caching. It doesn't support all GPT-5.4 parameters or API surfaces.

1. **What is the deprecation plan for previous models?**

   Any model deprecations will be posted on our [deprecations page](https://developers.openai.com/api/docs/deprecations#page-top). We'll send advanced notice of any model deprecations.

1. **What are the reasoning efforts supported?**
   - GPT 5 supports minimal, low, medium (default), and high.
   - GPT 5.2 supports none (default), low, medium, and high.
   - GPT 5.4 supports none (default), low, medium, high, and xhigh.
