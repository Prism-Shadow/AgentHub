# Reasoning models

Source: https://developers.openai.com/api/docs/guides/reasoning (Responses API; applies to GPT-5.6 variants, GPT-5.5, and GPT-5.4)

Page outline: Get started with reasoning / Reasoning effort / Reasoning mode / How reasoning works / Managing the context window / Controlling costs / Allocating space for reasoning / Handling incomplete responses / Preserve reasoning across calls / Continue reasoning with stored responses / Preserve reasoning without stored responses / Reasoning summaries / `phase` parameter / Advice on prompting / Use case examples

## Reasoning effort

The `reasoning.effort` parameter guides how much the model thinks: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`. Lower values prioritize speed and cost; higher values enable more thorough analysis. Defaults vary by model; GPT-5.5 defaults to `medium`.

## Reasoning summaries

Models can emit summaries of their internal reasoning when `reasoning.summary` is set (`concise`, `detailed`, or `auto`). Summaries appear in the `summary` array of reasoning output items and require explicit opt-in.

## Preserve reasoning without stored responses

When `store` is `false`, reasoning items include an `encrypted_content` property by default. Pass these encrypted tokens back on later calls to preserve reasoning context without storing responses server-side.

## `phase` parameter

> For long-running or tool-heavy flows with GPT-5.5 and GPT-5.4 in the Responses API, use the assistant message `phase` field to avoid early stopping.

- Use `phase: "commentary"` for intermediate assistant updates, such as preambles before tool calls.
- Use `phase: "final_answer"` for the completed answer.
- These are the only two `phase` values.
- Don't add `phase` to user messages.
- `phase` is optional at the API level, but OpenAI recommends using it. When replaying assistant history manually, preserve each original `phase` value. Missing or dropped `phase` can cause preambles to be treated as final answers in those workflows.
- Using `previous_response_id` is usually the simplest path because prior assistant state is preserved.

Official example (input array with `phase` on assistant messages):

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.6",
    input=[
        {
            "role": "assistant",
            "phase": "commentary",
            "content": "I'll inspect the logs and then summarize root cause and remediation.",
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
