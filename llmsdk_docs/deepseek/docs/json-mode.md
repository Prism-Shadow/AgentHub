# JSON Mode

JSON mode asks DeepSeek to return valid JSON strings for structured output use cases.

## Requirements

To enable JSON output:

1. Set `response_format` to `{"type": "json_object"}`.
2. Include the word `json` in the system or user prompt.
3. Provide an example of the desired JSON shape in the prompt.
4. Set `max_tokens` high enough to avoid truncating the JSON response.

The API may occasionally return empty content in JSON mode. If that happens, adjust the prompt and retry.

## Sample Code

```python
import json
from openai import OpenAI

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

system_prompt = """
The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format.

EXAMPLE INPUT:
Which is the highest mountain in the world? Mount Everest.
EXAMPLE JSON OUTPUT:
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
}
"""

user_prompt = "Which is the longest river in the world? The Nile River."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    response_format={"type": "json_object"},
)

print(json.loads(response.choices[0].message.content))
```

Expected output:

```json
{
  "question": "Which is the longest river in the world?",
  "answer": "The Nile River"
}
```
