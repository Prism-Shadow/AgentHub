# Using the Messages API

Practical patterns and examples for using the Messages API effectively

---

This guide covers common patterns for working with the Messages API, including basic requests, multi-turn conversations, prefill techniques, and vision capabilities. For complete API specifications, see the [Messages API reference](/docs/en/api/messages).

<Note>
This feature is [Zero Data Retention (ZDR)](/docs/en/build-with-claude/zero-data-retention) eligible. When your organization has a ZDR arrangement, data sent through this feature is not stored after the API response is returned.
</Note>

## Basic request and response

<CodeGroup>
  ```bash Shell
  #!/bin/sh
  curl https://api.anthropic.com/v1/messages \
       --header "x-api-key: $ANTHROPIC_API_KEY" \
       --header "anthropic-version: 2023-06-01" \
       --header "content-type: application/json" \
       --data \
  '{
      "model": "claude-opus-4-6",
      "max_tokens": 1024,
      "messages": [
          {"role": "user", "content": "Hello, Claude"}
      ]
  }'
  ```

  ```python Python
  import anthropic

  message = anthropic.Anthropic().messages.create(
      model="claude-opus-4-6",
      max_tokens=1024,
      messages=[{"role": "user", "content": "Hello, Claude"}],
  )
  print(message)
  ```

  ```typescript TypeScript hidelines={1..4}
  import Anthropic from "@anthropic-ai/sdk";

  const anthropic = new Anthropic();

  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 1024,
    messages: [{ role: "user", content: "Hello, Claude" }]
  });
  console.log(message);
  ```

  ```csharp C#
  using System;
  using System.Threading.Tasks;
  using Anthropic;
  using Anthropic.Models.Messages;

  class Program
  {
      static async Task Main()
      {
          AnthropicClient client = new();

          var parameters = new MessageCreateParams
          {
              Model = Model.ClaudeOpus4_6,
              MaxTokens = 1024,
              Messages = [new() { Role = Role.User, Content = "Hello, Claude" }]
          };
          var message = await client.Messages.Create(parameters);
          Console.WriteLine(message);
      }
  }
  ```
</CodeGroup>

```json JSON
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello!"
    }
  ],
  "model": "claude-opus-4-6",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 6
  }
}
```

## Multiple conversational turns

The Messages API is stateless, which means that you always send the full conversational history to the API. You can use this pattern to build up a conversation over time. Earlier conversational turns don't necessarily need to actually originate from Claude. You can use synthetic `assistant` messages.

<CodeGroup>
```bash Shell
#!/bin/sh
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-opus-4-6",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, Claude"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Can you describe LLMs to me?"}

    ]
}'
```

```python Python hidelines={1..2,-1}
import anthropic

message = anthropic.Anthropic().messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Can you describe LLMs to me?"},
    ],
)
print(message)
```

```typescript TypeScript hidelines={1..4}
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic();

await anthropic.messages.create({
  model: "claude-opus-4-6",
  max_tokens: 1024,
  messages: [
    { role: "user", content: "Hello, Claude" },
    { role: "assistant", content: "Hello!" },
    { role: "user", content: "Can you describe LLMs to me?" }
  ]
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
            Model = Model.ClaudeOpus4_6,
            MaxTokens = 1024,
            Messages =
            [
                new() { Role = Role.User, Content = "Hello, Claude" },
                new() { Role = Role.Assistant, Content = "Hello!" },
                new() { Role = Role.User, Content = "Can you describe LLMs to me?" }
            ]
        };

        var message = await client.Messages.Create(parameters);
        Console.WriteLine(message);
    }
}
```
</CodeGroup>

```json JSON
{
  "id": "msg_018gCsTGsXkYJVqYPxTgDHBU",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Sure, I'd be happy to provide..."
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 30,
    "output_tokens": 309
  }
}
```

## Putting words in Claude's mouth

You can pre-fill part of Claude's response in the last position of the input messages list. This can be used to shape Claude's response. The example below uses `"max_tokens": 1` to get a single multiple choice answer from Claude.

<CodeGroup>
  ```bash Shell
  #!/bin/sh
  curl https://api.anthropic.com/v1/messages \
       --header "x-api-key: $ANTHROPIC_API_KEY" \
       --header "anthropic-version: 2023-06-01" \
       --header "content-type: application/json" \
       --data \
  '{
      "model": "claude-opus-4-6",
      "max_tokens": 1,
      "messages": [
          {"role": "user", "content": "What is latin for Ant? (A) Apoidea, (B) Rhopalocera, (C) Formicidae"},
          {"role": "assistant", "content": "The answer is ("}
      ]
  }'
  ```

  ```python Python hidelines={1..2,-1}
  import anthropic

  message = anthropic.Anthropic().messages.create(
      model="claude-opus-4-6",
      max_tokens=1,
      messages=[
          {
              "role": "user",
              "content": "What is latin for Ant? (A) Apoidea, (B) Rhopalocera, (C) Formicidae",
          },
          {"role": "assistant", "content": "The answer is ("},
      ],
  )
  print(message)
  ```

  ```typescript TypeScript hidelines={1..4}
  import Anthropic from "@anthropic-ai/sdk";

  const anthropic = new Anthropic();

  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 1,
    messages: [
      {
        role: "user",
        content: "What is latin for Ant? (A) Apoidea, (B) Rhopalocera, (C) Formicidae"
      },
      { role: "assistant", content: "The answer is (" }
    ]
  });
  console.log(message);
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
              MaxTokens = 1,
              Messages = [
                  new() { Role = Role.User, Content = "What is latin for Ant? (A) Apoidea, (B) Rhopalocera, (C) Formicidae" },
                  new() { Role = Role.Assistant, Content = "The answer is (" }
              ]
          };

          var message = await client.Messages.Create(parameters);
          Console.WriteLine(message);
      }
  }
  ```
</CodeGroup>

```json JSON
{
  "id": "msg_01Q8Faay6S7QPTvEUUQARt7h",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "C"
    }
  ],
  "model": "claude-opus-4-6",
  "stop_reason": "max_tokens",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 42,
    "output_tokens": 1
  }
}
```

<Warning>
Prefilling is deprecated and not supported on Claude Opus 4.6, Claude Sonnet 4.6, and Claude Sonnet 4.5. Use [structured outputs](/docs/en/build-with-claude/structured-outputs) or system prompt instructions instead.
</Warning>

## Vision

Claude can read both text and images in requests. Both `base64` and `url` source types are supported for images, along with the `image/jpeg`, `image/png`, `image/gif`, and `image/webp` media types. See the [vision guide](/docs/en/build-with-claude/vision) for more details.

<CodeGroup>
  ```bash Shell
  #!/bin/sh

  # Option 1: Base64-encoded image
  IMAGE_URL="https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
  IMAGE_MEDIA_TYPE="image/jpeg"
  IMAGE_BASE64=$(curl "$IMAGE_URL" | base64)

  curl https://api.anthropic.com/v1/messages \
       --header "x-api-key: $ANTHROPIC_API_KEY" \
       --header "anthropic-version: 2023-06-01" \
       --header "content-type: application/json" \
       --data \
  '{
      "model": "claude-opus-4-6",
      "max_tokens": 1024,
      "messages": [
          {"role": "user", "content": [
              {"type": "image", "source": {
                  "type": "base64",
                  "media_type": "'$IMAGE_MEDIA_TYPE'",
                  "data": "'$IMAGE_BASE64'"
              }},
              {"type": "text", "text": "What is in the above image?"}
          ]}
      ]
  }'

  # Option 2: URL-referenced image
  curl https://api.anthropic.com/v1/messages \
       --header "x-api-key: $ANTHROPIC_API_KEY" \
       --header "anthropic-version: 2023-06-01" \
       --header "content-type: application/json" \
       --data \
  '{
      "model": "claude-opus-4-6",
      "max_tokens": 1024,
      "messages": [
          {"role": "user", "content": [
              {"type": "image", "source": {
                  "type": "url",
                  "url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
              }},
              {"type": "text", "text": "What is in the above image?"}
          ]}
      ]
  }'
  ```


  ```python Python nocheck hidelines={-1}
  import anthropic
  import base64
  import httpx

  # Option 1: Base64-encoded image
  image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
  image_media_type = "image/jpeg"
  image_data = base64.standard_b64encode(httpx.get(image_url).content).decode("utf-8")

  message = anthropic.Anthropic().messages.create(
      model="claude-opus-4-6",
      max_tokens=1024,
      messages=[
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "source": {
                          "type": "base64",
                          "media_type": image_media_type,
                          "data": image_data,
                      },
                  },
                  {"type": "text", "text": "What is in the above image?"},
              ],
          }
      ],
  )
  print(message)

  # Option 2: URL-referenced image
  message_from_url = anthropic.Anthropic().messages.create(
      model="claude-opus-4-6",
      max_tokens=1024,
      messages=[
          {
              "role": "user",
              "content": [
                  {
                      "type": "image",
                      "source": {
                          "type": "url",
                          "url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg",
                      },
                  },
                  {"type": "text", "text": "What is in the above image?"},
              ],
          }
      ],
  )
  print(message_from_url)
  ```

  ```typescript TypeScript hidelines={1..4}
  import Anthropic from "@anthropic-ai/sdk";

  const anthropic = new Anthropic();

  // Option 1: Base64-encoded image
  const image_url =
    "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg";
  const image_media_type = "image/jpeg";
  const image_array_buffer = await (await fetch(image_url)).arrayBuffer();
  const image_data = Buffer.from(image_array_buffer).toString("base64");

  const message = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 1024,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: image_media_type,
              data: image_data
            }
          },
          {
            type: "text",
            text: "What is in the above image?"
          }
        ]
      }
    ]
  });
  console.log(message);

  // Option 2: URL-referenced image
  const messageFromUrl = await anthropic.messages.create({
    model: "claude-opus-4-6",
    max_tokens: 1024,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "url",
              url: "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
            }
          },
          {
            type: "text",
            text: "What is in the above image?"
          }
        ]
      }
    ]
  });
  console.log(messageFromUrl);
  ```

  ```csharp C#
  using System;
  using System.Collections.Generic;
  using System.Net.Http;
  using System.Threading.Tasks;
  using Anthropic;
  using Anthropic.Models.Messages;

  public class Program
  {
      public static async Task Main(string[] args)
      {
          AnthropicClient client = new();

          // Option 1: Base64-encoded image
          string imageUrl = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg";

          using HttpClient httpClient = new();
          byte[] imageBytes = await httpClient.GetByteArrayAsync(imageUrl);
          string imageData = Convert.ToBase64String(imageBytes);

          var parameters = new MessageCreateParams
          {
              Model = Model.ClaudeOpus4_6,
              MaxTokens = 1024,
              Messages =
              [
                  new()
                  {
                      Role = Role.User,
                      Content = new MessageParamContent(new List<ContentBlockParam>
                      {
                          new ContentBlockParam(new ImageBlockParam(
                              new ImageBlockParamSource(new Base64ImageSource()
                              {
                                  Data = imageData,
                                  MediaType = MediaType.ImageJpeg,
                              })
                          )),
                          new ContentBlockParam(new TextBlockParam("What is in the above image?")),
                      }),
                  }
              ]
          };

          var message = await client.Messages.Create(parameters);
          Console.WriteLine(message);

          // Option 2: URL-referenced image
          var parametersFromUrl = new MessageCreateParams
          {
              Model = Model.ClaudeOpus4_6,
              MaxTokens = 1024,
              Messages =
              [
                  new()
                  {
                      Role = Role.User,
                      Content = new MessageParamContent(new List<ContentBlockParam>
                      {
                          new ContentBlockParam(new ImageBlockParam(
                              new ImageBlockParamSource(new UrlImageSource()
                              {
                                  Url = new Uri("https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"),
                              })
                          )),
                          new ContentBlockParam(new TextBlockParam("What is in the above image?")),
                      }),
                  }
              ]
          };

          var messageFromUrl = await client.Messages.Create(parametersFromUrl);
          Console.WriteLine(messageFromUrl);
      }
  }
  ```
</CodeGroup>

```json JSON
{
  "id": "msg_01EcyWo6m4hyW8KHs2y2pei5",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "This image shows an ant, specifically a close-up view of an ant. The ant is shown in detail, with its distinct head, antennae, and legs clearly visible. The image is focused on capturing the intricate details and features of the ant, likely taken with a macro lens to get an extreme close-up perspective."
    }
  ],
  "model": "claude-opus-4-6",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 1551,
    "output_tokens": 71
  }
}
```

## Tool use and computer use

See the [tool use guide](/docs/en/agents-and-tools/tool-use/overview) for examples of how to use tools with the Messages API.
See the [computer use guide](/docs/en/agents-and-tools/tool-use/computer-use-tool) for examples of how to control desktop computer environments with the Messages API.
For guaranteed JSON output, see [Structured Outputs](/docs/en/build-with-claude/structured-outputs).
