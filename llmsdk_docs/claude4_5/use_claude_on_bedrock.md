# Use Claude on Amazon Bedrock

Source: https://docs.aws.amazon.com/bedrock/latest/userguide/api-inference-examples-claude-messages-code-examples.html

---

The following code examples show how to send a text message to Anthropic Claude, using the Converse API and the Messages API (`invoke_model`).

## Prerequisites

- An active AWS account with access to Amazon Bedrock
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) configured with appropriate credentials
- Python 3.8+ with `boto3` installed: `pip install boto3`
- Model access enabled for the Anthropic Claude model in the AWS Bedrock console

---

## Converse API

The following examples use the Converse API to send a text message to Claude.

### Basic text message (Python)

```python
# Use the Conversation API to send a text message to Anthropic Claude.

import boto3
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Claude 3 Haiku.
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Start a conversation with the user message.
user_message = "Describe the purpose of a 'hello world' program in one line."
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)
```

### Streaming response (Python)

```python
# Use the Conversation API to send a text message to Anthropic Claude
# and print the response stream.

import boto3
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Claude 3 Haiku.
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Start a conversation with the user message.
user_message = "Describe the purpose of a 'hello world' program in one line."
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    streaming_response = client.converse_stream(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    # Extract and print the streamed response text in real-time.
    for chunk in streaming_response["stream"]:
        if "contentBlockDelta" in chunk:
            text = chunk["contentBlockDelta"]["delta"]["text"]
            print(text, end="", flush=True)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)
```

---

## Messages API (`invoke_model`)

The following examples use the `invoke_model` endpoint to send a text message to Claude using the native Anthropic Messages API format.

### Basic text message (Python)

```python
# Use the native inference API to send a text message to Anthropic Claude.

import boto3
import json
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Claude 3 Haiku.
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Define the prompt for the model.
prompt = "Describe the purpose of a 'hello world' program in one line."

# Format the request payload using the model's native structure.
native_request = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "messages": [
        {
            "role": "user",
            "content": prompt,
        }
    ],
}

# Convert the native request to JSON.
request = json.dumps(native_request)

try:
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# Decode the response body.
model_response = json.loads(response["body"].read())

# Extract and print the response text.
response_text = model_response["content"][0]["text"]
print(response_text)
```

### Streaming response (Python)

```python
# Use the native inference API to send a text message to Anthropic Claude
# and print the response stream.

import boto3
import json
from botocore.exceptions import ClientError

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Claude 3 Haiku.
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Define the prompt for the model.
prompt = "Describe the purpose of a 'hello world' program in one line."

# Format the request payload using the model's native structure.
native_request = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "messages": [
        {
            "role": "user",
            "content": prompt,
        }
    ],
}

# Convert the native request to JSON.
request = json.dumps(native_request)

try:
    # Invoke the model with the request.
    streaming_response = client.invoke_model_with_response_stream(
        modelId=model_id, body=request
    )

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)

# Extract and print the response text in real-time.
for event in streaming_response["body"]:
    chunk = json.loads(event["chunk"]["bytes"])
    if chunk["type"] == "content_block_delta":
        print(chunk["delta"].get("text", ""), end="", flush=True)
```

---

## Request Body Format

When using `invoke_model`, the request body must use the Anthropic Messages API format:

```json
{
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1024,
    "system": "You are a helpful assistant.",
    "messages": [
        {
            "role": "user",
            "content": "Hello, Claude"
        }
    ]
}
```

### Key Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `anthropic_version` | string | Yes | Must be `"bedrock-2023-05-31"` for Bedrock |
| `max_tokens` | integer | Yes | Maximum number of tokens to generate |
| `messages` | array | Yes | Array of message objects with `role` and `content` |
| `system` | string | No | System prompt to set context for Claude |
| `temperature` | float | No | Controls randomness (0.0–1.0) |
| `top_p` | float | No | Nucleus sampling parameter (0.0–1.0) |
| `top_k` | integer | No | Top-k sampling parameter |
| `stop_sequences` | array | No | List of sequences that stop generation |

---

## Response Format

A successful `invoke_model` response body contains:

```json
{
    "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "Hello! How can I help you today?"
        }
    ],
    "model": "claude-3-haiku-20240307",
    "stop_reason": "end_turn",
    "stop_sequence": null,
    "usage": {
        "input_tokens": 12,
        "output_tokens": 15
    }
}
```

---

## Supported Claude Model IDs on Bedrock

| Model | Model ID |
|-------|----------|
| Claude 3.5 Sonnet v2 | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| Claude 3.5 Haiku | `anthropic.claude-3-5-haiku-20241022-v1:0` |
| Claude 3 Opus | `anthropic.claude-3-opus-20240229-v1:0` |
| Claude 3 Sonnet | `anthropic.claude-3-sonnet-20240229-v1:0` |
| Claude 3 Haiku | `anthropic.claude-3-haiku-20240307-v1:0` |

> **Note:** Model availability varies by AWS Region. Check the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for the latest supported models and regions.

---

## Multi-turn Conversations

To have a multi-turn conversation with Claude, include the full conversation history in the `messages` array:

```python
import boto3
import json

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

conversation_history = []

def chat(user_message):
    conversation_history.append({"role": "user", "content": user_message})

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": conversation_history,
    }

    response = client.invoke_model(
        modelId=model_id, body=json.dumps(native_request)
    )
    model_response = json.loads(response["body"].read())
    assistant_message = model_response["content"][0]["text"]

    conversation_history.append({"role": "assistant", "content": assistant_message})
    return assistant_message

print(chat("Hello, my name is Alice."))
print(chat("What is my name?"))
```

---

## Using the Anthropic SDK with Bedrock

As an alternative to using `boto3` directly, you can use the official Anthropic Python SDK with Bedrock support:

```python
pip install anthropic[bedrock]
```

```python
from anthropic import AnthropicBedrock

client = AnthropicBedrock(
    # aws_profile="my-profile",          # optional: AWS profile name
    # aws_region="us-east-1",            # optional: AWS region
    # aws_access_key="...",              # optional: explicit credentials
    # aws_secret_key="...",
    # aws_session_token="...",
)

message = client.messages.create(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ],
)
print(message.content)
```

The Anthropic SDK handles authentication automatically using the standard AWS credential chain (environment variables, `~/.aws/credentials`, IAM role, etc.).

---

## Additional Resources

- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/)
- [Anthropic Claude on Amazon Bedrock](https://aws.amazon.com/bedrock/claude/)
- [Bedrock Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/)
- [Anthropic SDK for Bedrock](https://github.com/anthropics/anthropic-sdk-python#aws-bedrock)
