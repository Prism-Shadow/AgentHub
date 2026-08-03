# MiniMax M3 Python Quick Start

MiniMax documents an OpenAI Responses API-compatible endpoint at `https://api.minimax.io/v1/responses`.

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="<MiniMax API Key or Token Plan Subscription Key>",
    base_url="https://api.minimax.io/v1",
)

stream = await client.responses.create(
    model="MiniMax-M3",
    input="Explain the purpose of a hash function.",
    reasoning={"effort": "minimal"},
    stream=True,
)

async for event in stream:
    print(event)
```

Use a Token Plan **Subscription Key** for Token Plan quota/Credits, or a separate API Key for pay-as-you-go billing. The keys are not interchangeable.

For multi-turn tool use, preserve and replay the full assistant response, including reasoning output and function-call items, before adding matching function-call output items.

Sources:

- https://platform.minimax.io/docs/api-reference/responses-create
- https://platform.minimax.io/docs/guides/text-m3-function-call
- https://platform.minimax.io/docs/token-plan/intro
