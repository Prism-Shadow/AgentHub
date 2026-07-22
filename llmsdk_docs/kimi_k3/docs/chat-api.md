> Fetch the complete documentation index at: https://platform.kimi.com/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# 创建对话补全

> 为聊天消息创建补全结果。支持标准聊天、Partial Mode 和 Tool Use（函数调用）。

创建一个对话补全请求，模型将根据输入的消息列表生成回复。

<Accordion title="Content 字段说明">
  `content` 字段支持以下两种形式：

  **纯文本字符串**

  ```json theme={null}
  { "content": "你好" }
  ```

  **对象数组**（用于多模态输入）

  数组中每个元素通过 `type` 字段区分类型：

  ```json theme={null}
  {
      "content": [
          { "type": "text", "text": "描述这张图片" },
          { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } },
          { "type": "video_url", "video_url": { "url": "data:video/mp4;base64,..." } }
      ]
  }
  ```

  其中 `image_url` 和 `video_url` 也支持直接传入字符串，效果等同于对象形式中的 `url` 字段：

  ```json theme={null}
  { "type": "image_url", "image_url": "data:image/png;base64,..." }
  ```

  #### 参数说明

  数组中每个元素的字段说明如下：

  | 参数名称        | 是否必须                   | 说明                                           | 类型                                         |
  | ----------- | ---------------------- | -------------------------------------------- | ------------------------------------------ |
  | `type`      | required               | 内容类型                                         | `"text"` \| `"image_url"` \| `"video_url"` |
  | `text`      | 当 `type=text` 时必填      | 文本内容                                         | string                                     |
  | `image_url` | 当 `type=image_url` 时必填 | 用于传输图片，支持对象形式 `{"url": "..."}` 或直接传入 URL 字符串 | object \| string                           |
  | `video_url` | 当 `type=video_url` 时必填 | 用于传输视频，支持对象形式 `{"url": "..."}` 或直接传入 URL 字符串 | object \| string                           |

  当 `image_url` 传入对象时，其字段说明如下：

  | 参数名称  | 是否必须     | 说明                              | 类型     |
  | ----- | -------- | ------------------------------- | ------ |
  | `url` | required | 使用 base64 编码或通过 file id 指定的图片内容 | string |

  当 `video_url` 传入对象时，其字段说明如下：

  | 参数名称  | 是否必须     | 说明                                                             | 类型     |
  | ----- | -------- | -------------------------------------------------------------- | ------ |
  | `url` | required | 使用 base64 编码或通过 file id 指定的视频内容，例如 `data:video/mp4;base64,...` | string |

  <Note>
    无论使用对象形式（`url` 字段）还是字符串简写，均支持以下两种格式：

    * base64 编码：`data:image/png;base64,...` 或 `data:video/mp4;base64,...`
    * 文件引用：`ms://<file_id>`

    详见[使用 Kimi 视觉模型](/docs/guide/use-kimi-vision-model)。
  </Note>

  #### 调用示例

  <CodeGroup>
    ```python python expandable theme={null}
    import os
    import base64

    from openai import OpenAI
    from openai.types.chat import ChatCompletion

    client: OpenAI = OpenAI(
        api_key=os.environ.get("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.cn/v1",
    )

    # 对图片进行 base64 编码
    with open("您的图片地址", "rb") as f:
        img_base: str = base64.b64encode(f.read()).decode("utf-8")

    response: ChatCompletion = client.chat.completions.create(
        model="kimi-k2.6",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "请描述这个图片",
                    },
                ],
            }
        ],
    )
    print(response.choices[0].message.content)
    ```

    ```bash curl expandable theme={null}
    curl https://api.moonshot.cn/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $MOONSHOT_API_KEY" \
        -d '{
            "model": "kimi-k2.6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQ..."
                            }
                        },
                        {
                            "type": "text",
                            "text": "请描述这个图片"
                        }
                    ]
                }
            ]
        }'
    ```

    ```javascript node.js expandable theme={null}
    const fs = require("fs");
    const OpenAI = require("openai");

    const client = new OpenAI({
        apiKey: process.env.MOONSHOT_API_KEY,
        baseURL: "https://api.moonshot.cn/v1",
    });

    async function main() {
        // 对图片进行 base64 编码
        const imgBase = fs.readFileSync("您的图片地址").toString("base64");

        const response = await client.chat.completions.create({
            model: "kimi-k2.6",
            messages: [
                {
                    role: "user",
                    content: [
                        {
                            type: "image_url",
                            image_url: {
                                url: `data:image/jpeg;base64,${imgBase}`,
                            },
                        },
                        {
                            type: "text",
                            text: "请描述这个图片",
                        },
                    ],
                },
            ],
        });
        console.log(response.choices[0].message.content);
    }

    main();
    ```
  </CodeGroup>
</Accordion>

<Accordion title="响应格式">
  ### 非流式响应

  ```json theme={null}
  {
      "id": "cmpl-04ea926191a14749b7f2c7a48a68abc6",
      "object": "chat.completion",
      "created": 1698999496,
      "model": "kimi-k2.6",
      "choices": [
          {
              "index": 0,
              "message": {
                  "role": "assistant",
                  "content": "你好，李雷！1+1等于2。如果你有其他问题，请随时提问！"
              },
              "finish_reason": "stop"
          }
      ],
      "usage": {
          "prompt_tokens": 19,
          "completion_tokens": 21,
          "total_tokens": 40,
          "cached_tokens": 10
      }
  }
  ```

  ### 流式响应

  ```text theme={null}
  data: {"id":"cmpl-xxx","object":"chat.completion.chunk","created":1698999575,"model":"kimi-k2.6","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

  data: {"id":"cmpl-xxx","object":"chat.completion.chunk","created":1698999575,"model":"kimi-k2.6","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}

  ...

  data: {"id":"cmpl-xxx","object":"chat.completion.chunk","created":1698999575,"model":"kimi-k2.6","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":19,"completion_tokens":13,"total_tokens":32,"cached_tokens":12}}

  data: [DONE]
  ```

  <Note>
    响应示例中的模型名称会根据请求中的 model 参数返回。当使用 `kimi-k2.6` 模型时，响应中的 `"model"` 字段将显示为 `"kimi-k2.6"`。
  </Note>
</Accordion>

<Accordion title="多轮对话">
  Kimi API 是无状态的，本身不具有记忆功能。要实现多轮对话，需在每次请求时把前一轮的 assistant 回复（以及工具执行结果，如适用）原样追加到 `messages` 数组中再发送。

  ```python theme={null}
  messages = [
      {"role": "system", "content": "你是 Kimi。"},
      {"role": "user", "content": "你好，我叫李雷。"}
  ]

  completion = client.chat.completions.create(model="kimi-k2.6", messages=messages)
  reply = completion.choices[0].message

  # 将 assistant 回复追加回 messages，供下一轮使用
  messages.append({"role": "assistant", "content": reply.content})
  messages.append({"role": "user", "content": "1+1 等于多少？"})
  ```

  当对话历史过长时，建议只保留最近的若干条消息，或做消息压缩，以避免超出模型的上下文长度限制。

  <Note>
    详见 [配置多轮对话参数](/docs/guide/engage-in-multi-turn-conversations-using-kimi-api)。
  </Note>
</Accordion>

<Accordion title="JSON Mode">
  通过 `response_format` 参数可约束模型输出格式：

  * `{"type": "text"}`（默认）：普通文本输出
  * `{"type": "json_object"}`：强制输出合法 JSON Object
  * `{"type": "json_schema", "json_schema": {...}}`：按给定 JSON Schema 输出结构化数据（Structured Output）

  使用 `json_object` 时，**必须在 system prompt 或 user prompt 中明确描述期望的 JSON 字段和类型**，否则模型可能输出不符合预期的结果。

  ```json theme={null}
  {
    "model": "kimi-k2.6",
    "messages": [
      {"role": "system", "content": "请输出 JSON，包含 title、author、summary 字段。"},
      {"role": "user", "content": "总结这篇文章..."}
    ],
    "response_format": {"type": "json_object"}
  }
  ```

  <Note>
    详见 [使用 Kimi API 的 JSON Mode](/docs/guide/use-json-mode-feature-of-kimi-api)。
  </Note>
</Accordion>

<Accordion title="工具调用 Tool Use">
  通过 `tools` 参数传入 JSON Schema 定义的外部工具，模型可决定在适当时机调用它们。

  **请求示例**

  ```json theme={null}
  {
    "model": "kimi-k2.6",
    "messages": [{"role": "user", "content": "北京今天天气怎么样？"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "获取指定城市的天气",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
          }
        }
      }
    ]
  }
  ```

  **响应中的 `tool_calls`**

  当 `finish_reason` 为 `"tool_calls"` 时，模型返回 `tool_calls` 数组，包含 `id`、`function.name` 和 `function.arguments`：

  ```json theme={null}
  {
    "choices": [{
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
          "id": "call_xxx",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\":\"北京\"}"
          }
        }]
      },
      "finish_reason": "tool_calls"
    }]
  }
  ```

  **提交工具执行结果**

  在本地执行工具后，将结果通过 `role="tool"` 消息追加到 `messages` 中（`tool_call_id` 必须与请求中的 `id` 对应）：

  ```json theme={null}
  {"role": "tool", "tool_call_id": "call_xxx", "content": "晴，25°C"}
  ```

  <Note>
    详见 [使用 Kimi API 完成工具调用](/docs/guide/use-kimi-api-to-complete-tool-calls)。
  </Note>
</Accordion>

<Accordion title="思考模式与 Preserved Thinking">
  `kimi-k2.6` 和 `kimi-k2.7-code` 支持思考模式，模型在输出最终答案前会先输出推理过程（`reasoning_content`）。K3 始终进行推理，使用顶层 `reasoning_effort`（当前唯一值为 `"max"`）。

  **K2.x 请求参数**

  | 字段              | 类型                          | 说明                                                                                                                                    |
  | --------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
  | `thinking.type` | `"enabled"` \| `"disabled"` | 思考开关（`kimi-k2.7-code` 始终为 `enabled`，不可关闭）                                                                                             |
  | `thinking.keep` | `null` \| `"all"`           | Preserved Thinking：是否将历史轮次的 `reasoning_content` 保留在上下文。`kimi-k2.6` 默认 `null`（不保留），可选 `"all"`；`kimi-k2.7-code` 固定为 `"all"`（保留），传入其他值报错 |

  **响应字段**

  非流式响应中，`choices[0].message` 包含：

  | 字段                  | 说明                |
  | ------------------- | ----------------- |
  | `content`           | 最终答案              |
  | `reasoning_content` | 推理过程（仅在思考模式启用时返回） |

  ```json theme={null}
  {
    "choices": [{
      "message": {
        "role": "assistant",
        "content": "1+1 等于 2。",
        "reasoning_content": "用户问的是基础数学问题，直接相加即可。"
      }
    }]
  }
  ```

  <Warning>
    多轮对话中若使用思考模式，请务必将每一轮 assistant 消息的 `reasoning_content` 原样保留在 `messages` 中，否则模型可能丢失推理上下文。
  </Warning>

  <Note>
    详见 [配置思考模式](/docs/guide/use-kimi-k2-thinking-model)。
  </Note>
</Accordion>

<Accordion title="流式输出 Streaming">
  设置 `stream: true` 可启用流式输出，模型会以 Server-Sent Events (SSE) 格式逐段返回生成的内容。推荐在聊天、代码生成、长文本输出等实时性要求高的场景中使用。

  ```python theme={null}
  completion = client.chat.completions.create(
      model="kimi-k2.6",
      messages=[{"role": "user", "content": "请解释什么是递归。"}],
      stream=True
  )

  for chunk in completion:
      if chunk.choices[0].delta.content:
          print(chunk.choices[0].delta.content, end="")
  ```

  **SSE 响应格式**

  每一行以 `data:` 开头，内容为 JSON 对象。当 `finish_reason` 为 `null` 时，内容在 `delta.content` 中累加；当 `finish_reason` 不为 `null` 时，表示输出结束：

  ```text theme={null}
  data: {"id":"cmpl-xxx","object":"chat.completion.chunk","created":1698999575,"model":"kimi-k2.6","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

  data: {"id":"cmpl-xxx","object":"chat.completion.chunk","created":1698999575,"model":"kimi-k2.6","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}

  data: {"id":"cmpl-xxx","object":"chat.completion.chunk","created":1698999575,"model":"kimi-k2.6","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":19,"completion_tokens":13,"total_tokens":32,"cached_tokens":12}}

  data: [DONE]
  ```

  **`stream_options`**

  通过 `stream_options: {"include_usage": true}` 可在最后一个 chunk（`data: [DONE]` 之前）额外获取 `usage` 字段，显示本次请求的 Token 消耗：

  ```python theme={null}
  stream=True,
  stream_options={"include_usage": True}
  ```

  <Note>
    详见 [利用 Kimi API 的流式输出功能](/docs/guide/utilize-the-streaming-output-feature-of-kimi-api)。
  </Note>
</Accordion>

<Accordion title="Partial Mode">
  Partial Mode（Prefill）允许你在 `messages` 的最后一条 assistant 消息中预填输出前缀，从而引导模型按照你期望的格式或方向继续生成。

  **开启方式**

  在 `messages` 数组末尾添加一条 `role="assistant"` 的消息，并设置 `partial: true`：

  ````python theme={null}
  completion = client.chat.completions.create(
      model="kimi-k2.6",
      messages=[
          {"role": "user", "content": "用 Python 实现快速排序。"},
          {"role": "assistant", "content": "```python\n", "partial": True}
      ]
  )
  ````

  模型会从 \`\`\`\`python\n\` 之后继续生成代码，而不是先输出解释文字再写代码。

  **常见用途**

  * 强制模型以特定格式开头（如 JSON 的 `{`、代码块的 \`\`\`\`python\`）
  * 角色扮演中保持角色名称前缀（配合 `name` 字段）
  * 在 `finish_reason="length"` 时，用相同的前缀续写被截断的内容

  <Warning>
    请勿将 Partial Mode 与 `response_format={"type": "json_object"}` 混用，否则可能获得预期外的模型回复。如需引导 JSON 输出，建议直接使用 [Structured Output](/docs/guide/response_format) 或单独设置 `partial: true` 并预填 `{`。
  </Warning>

  <Note>
    详见 [使用 Kimi API 的 Partial Mode](/docs/guide/use-partial-mode-feature-of-kimi-api)。
  </Note>
</Accordion>


## OpenAPI

````yaml POST /v1/chat/completions
openapi: 3.1.0
info:
  title: Moonshot AI API
  version: 1.0.0
  description: Moonshot AI / Kimi 大语言模型服务 API
servers:
  - url: https://api.moonshot.cn
    description: 生产环境
security: []
paths:
  /v1/chat/completions:
    post:
      tags:
        - Chat
      summary: 创建聊天补全
      description: 为聊天消息创建补全结果。支持标准聊天、Partial Mode 和 Tool Use（函数调用）。
      requestBody:
        required: true
        content:
          application/json:
            schema:
              oneOf:
                - $ref: '#/components/schemas/KimiK3ChatRequest'
                - $ref: '#/components/schemas/KimiK27CodeChatRequest'
                - $ref: '#/components/schemas/KimiK26ChatRequest'
                - $ref: '#/components/schemas/KimiK25ChatRequest'
                - $ref: '#/components/schemas/MoonshotV1ChatRequest'
              discriminator:
                propertyName: model
                mapping:
                  kimi-k3:
                    $ref: '#/components/schemas/KimiK3ChatRequest'
                  kimi-k2.7-code:
                    $ref: '#/components/schemas/KimiK27CodeChatRequest'
                  kimi-k2.7-code-highspeed:
                    $ref: '#/components/schemas/KimiK27CodeChatRequest'
                  kimi-k2.6:
                    $ref: '#/components/schemas/KimiK26ChatRequest'
                  kimi-k2.5:
                    $ref: '#/components/schemas/KimiK25ChatRequest'
                  moonshot-v1-8k:
                    $ref: '#/components/schemas/MoonshotV1ChatRequest'
                  moonshot-v1-32k:
                    $ref: '#/components/schemas/MoonshotV1ChatRequest'
                  moonshot-v1-128k:
                    $ref: '#/components/schemas/MoonshotV1ChatRequest'
                  moonshot-v1-auto:
                    $ref: '#/components/schemas/MoonshotV1ChatRequest'
                  moonshot-v1-8k-vision-preview:
                    $ref: '#/components/schemas/MoonshotV1ChatRequest'
                  moonshot-v1-32k-vision-preview:
                    $ref: '#/components/schemas/MoonshotV1ChatRequest'
                  moonshot-v1-128k-vision-preview:
                    $ref: '#/components/schemas/MoonshotV1ChatRequest'
      responses:
        '200':
          description: 聊天补全响应
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatCompletionResponse'
            text/event-stream:
              schema:
                $ref: '#/components/schemas/ChatCompletionChunk'
        '400':
          description: 请求错误 - 参数无效
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '401':
          description: 未授权 - API 密钥无效或缺失
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
        '500':
          description: 服务器错误
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'
      security:
        - bearerAuth: []
components:
  schemas:
    KimiK3ChatRequest:
      title: kimi-k3
      allOf:
        - $ref: '#/components/schemas/ChatRequestCommon'
        - type: object
          properties:
            model:
              type: string
              description: 模型 ID
              enum:
                - kimi-k3
              default: kimi-k3
            messages:
              type: array
              description: >-
                Kimi K3 对话消息列表。除标准消息外，还可在任意对话位置插入 {"role": "system", "tools":
                [...]} 消息动态加载工具；该动态工具消息不包含 content 字段，并且只影响后续对话。
              items:
                $ref: '#/components/schemas/KimiK3Message'
            reasoning_effort:
              type: string
              enum:
                - low
                - high
                - max
              default: max
              description: >-
                Kimi K3 始终启用思考，并开启 Preserved Thinking。思考强度支持 low、high 和 max，默认为
                max。
          required:
            - model
            - messages
    KimiK27CodeChatRequest:
      title: kimi-k2.7-code
      allOf:
        - $ref: '#/components/schemas/ChatRequestBase'
        - type: object
          properties:
            model:
              type: string
              description: >-
                模型 ID。可选 `kimi-k2.7-code` 或其高速版
                `kimi-k2.7-code-highspeed`；两者为同一模型、参数完全一致，高速版输出速度约 180
                Tokens/s（短上下文场景可达 260 Tokens/s）。
              enum:
                - kimi-k2.7-code
                - kimi-k2.7-code-highspeed
              default: kimi-k2.7-code
            thinking:
              type: object
              description: >-
                控制 kimi-k2.7-code 模型是否启用思考能力，以及是否完整保留多轮对话中的
                reasoning_content。可选参数，默认值为 {"type": "enabled", "keep": "all"}。


                与 kimi-k2.6 的差异：

                - `type` 仅支持 `"enabled"`。与 kimi-k2.6 不同，不支持 `"disabled"` —
                传入会报错。该模型始终开启思考。

                - `keep` 仅接受合法值 `"all"`；不传或传 `"all"` 时服务端均按 `"all"`
                处理，传入其他非法值会报错。因此该模型始终启用 Preserved Thinking。
              properties:
                type:
                  type: string
                  enum:
                    - enabled
                  description: >-
                    启用思考能力。对 kimi-k2.7-code 仅接受 `"enabled"`；传入 `"disabled"`
                    会报错。这与同时支持 `"disabled"` 的 kimi-k2.6 不同。
                keep:
                  type:
                    - string
                    - 'null'
                  enum:
                    - all
                    - null
                  description: >-
                    控制是否保留历史对话轮次（previous turns）的 reasoning_content，从而启用
                    Preserved Thinking。


                    - 对 kimi-k2.7-code，该参数仅接受合法值 `"all"`：传 `"all"`、传 `null`
                    或不传表现完全一致，服务端均按 `"all"` 处理 — 始终保留历史轮次的
                    reasoning_content；传入其他非法值会报错。这与 kimi-k2.6 不同，k2.6 默认为
                    `null`（除非显式设为 `"all"`，否则不保留历史思考）。

                    - 由于 Preserved Thinking 始终开启，请把每一轮历史 assistant 消息中的
                    reasoning_content 原样保留在 messages 中。

                    - 注意：该参数只影响历史轮次的 reasoning_content；不改变模型在当前 turn
                    内是否产生/输出思考内容（由 `type` 控制）。关于使用方式的最佳实践，详见 [Preserved
                    Thinking](/guide/use-kimi-k2-thinking-model#preserved-thinking)。
              required:
                - type
              additionalProperties: false
          required:
            - model
    KimiK26ChatRequest:
      title: kimi-k2.6
      allOf:
        - $ref: '#/components/schemas/ChatRequestBase'
        - type: object
          properties:
            model:
              type: string
              description: 模型 ID
              enum:
                - kimi-k2.6
              default: kimi-k2.6
            thinking:
              type: object
              description: >-
                控制 kimi-k2.6 模型是否启用思考能力，以及是否完整保留多轮对话中的
                reasoning_content。可选参数，默认值为 {"type": "enabled"}。
              properties:
                type:
                  type: string
                  enum:
                    - enabled
                    - disabled
                  description: 启用或禁用思考能力
                keep:
                  type:
                    - string
                    - 'null'
                  enum:
                    - all
                    - null
                  description: >-
                    控制是否保留历史对话轮次（previous turns）的 reasoning_content，从而启用
                    Preserved Thinking。默认为 `null`，即不保留历史轮次的思考内容。


                    - `null`（默认）或不传：服务端会忽略历史 turns 的 reasoning_content。

                    - `"all"`：保留历史 turns 的 reasoning_content 并随上下文一同提供给模型，启用
                    Preserved Thinking。使用时需把每一轮历史 assistant 消息中的
                    reasoning_content 原样保留在 messages 中。推荐与 `type: "enabled"`
                    搭配使用。

                    - 注意：该参数只影响历史轮次的 reasoning_content；不改变模型在当前 turn
                    内是否产生/输出思考内容（由 `type` 控制）。关于使用方式的最佳实践，详见 [Preserved
                    Thinking](/guide/use-kimi-k2-thinking-model#preserved-thinking)。
              required:
                - type
              additionalProperties: false
          required:
            - model
    KimiK25ChatRequest:
      title: kimi-k2.5
      allOf:
        - $ref: '#/components/schemas/ChatRequestBase'
        - type: object
          properties:
            model:
              type: string
              description: 模型 ID
              enum:
                - kimi-k2.5
              default: kimi-k2.5
            thinking:
              type: object
              description: '控制模型是否启用思考能力。可选参数，默认值为 {"type": "enabled"}。'
              properties:
                type:
                  type: string
                  enum:
                    - enabled
                    - disabled
                  description: 启用或禁用思考能力
              required:
                - type
              additionalProperties: false
          required:
            - model
    MoonshotV1ChatRequest:
      title: moonshot-v1
      allOf:
        - $ref: '#/components/schemas/ChatRequestBase'
        - type: object
          properties:
            model:
              type: string
              description: 模型 ID
              enum:
                - moonshot-v1-8k
                - moonshot-v1-32k
                - moonshot-v1-128k
                - moonshot-v1-auto
                - moonshot-v1-8k-vision-preview
                - moonshot-v1-32k-vision-preview
                - moonshot-v1-128k-vision-preview
              default: moonshot-v1-128k
            temperature:
              type: number
              format: float
              description: 采样温度，范围 0 到 1。较高的值（如 0.7）使输出更随机，较低的值（如 0.2）使输出更集中和确定。默认值为 0.0。
              default: 0
              minimum: 0
              maximum: 1
            top_p:
              type: number
              format: float
              description: >-
                另一种采样方法，模型考虑累积概率质量为 top_p 的 Token 结果。例如 0.1 表示仅考虑概率质量前 10% 的
                Token。通常建议只修改此参数或 temperature 其中之一。默认值为 1.0。
              default: 1
              minimum: 0
              maximum: 1
            'n':
              type: integer
              description: 每条输入消息生成的结果数量。默认为 1，不超过 5。当温度非常接近 0 时，只能返回 1 个结果。
              default: 1
              minimum: 1
              maximum: 5
            presence_penalty:
              type: number
              format: float
              description: 存在惩罚，范围 -2.0 到 2.0。正值会根据 Token 是否出现在文本中进行惩罚，增加模型讨论新话题的可能性
              default: 0
              minimum: -2
              maximum: 2
            frequency_penalty:
              type: number
              format: float
              description: 频率惩罚，范围 -2.0 到 2.0。正值会根据 Token 在文本中的现有频率进行惩罚，降低模型逐字重复相同短语的可能性
              default: 0
              minimum: -2
              maximum: 2
          required:
            - model
    ChatCompletionResponse:
      type: object
      properties:
        id:
          type: string
          description: 补全结果的唯一标识符
        object:
          type: string
          description: 对象类型
          example: chat.completion
        created:
          type: integer
          description: 补全创建时的 Unix 时间戳
        model:
          type: string
          description: 用于补全的模型
        choices:
          type: array
          description: 补全选项列表
          items:
            type: object
            properties:
              index:
                type: integer
              message:
                type: object
                properties:
                  role:
                    type: string
                    enum:
                      - assistant
                  content:
                    type:
                      - string
                      - 'null'
                    description: 助手的消息内容
                  tool_calls:
                    type: array
                    description: 模型发起的工具调用
                    items:
                      type: object
                      properties:
                        id:
                          type: string
                        type:
                          type: string
                          enum:
                            - function
                        function:
                          type: object
                          properties:
                            name:
                              type: string
                            arguments:
                              type: string
                              description: 函数参数的 JSON 字符串
                  reasoning_content:
                    type:
                      - string
                      - 'null'
                    description: 推理过程（仅在思考模式启用时返回）
              finish_reason:
                type: string
                enum:
                  - stop
                  - length
                  - tool_calls
        usage:
          type: object
          properties:
            prompt_tokens:
              type: integer
              description: 提示中的 Token 数量
            completion_tokens:
              type: integer
              description: 补全中的 Token 数量
            total_tokens:
              type: integer
              description: 使用的总 Token 数量
            cached_tokens:
              type: integer
              description: 命中缓存的 Token 数量
    ChatCompletionChunk:
      type: object
      properties:
        id:
          type: string
          description: 补全结果的唯一标识符
        object:
          type: string
          description: 对象类型
          example: chat.completion.chunk
        created:
          type: integer
          description: 补全创建时的 Unix 时间戳
        model:
          type: string
          description: 用于补全的模型
        choices:
          type: array
          description: 补全选项列表
          items:
            $ref: '#/components/schemas/ChoiceDelta'
        usage:
          type:
            - object
            - 'null'
          description: Token 使用统计。包含 usage 的最终 chunk 中为对象，其他 chunk 中为 null
          properties:
            prompt_tokens:
              type: integer
              description: 提示中的 Token 数量
            completion_tokens:
              type: integer
              description: 补全中的 Token 数量
            total_tokens:
              type: integer
              description: 使用的总 Token 数量
            cached_tokens:
              type: integer
              description: 命中缓存的 Token 数量
    ErrorResponse:
      type: object
      properties:
        error:
          type: object
          properties:
            message:
              type: string
              description: 描述错误原因的错误消息
            type:
              type: string
              description: 错误类型
            code:
              type: string
              description: 错误码
          required:
            - message
      required:
        - error
    ChatRequestCommon:
      type: object
      properties:
        max_tokens:
          type: integer
          deprecated: true
          description: 已弃用，请使用 max_completion_tokens
        max_completion_tokens:
          type: integer
          description: >-
            聊天补全生成的最大 Token 数量。默认值因模型而异：Kimi K3 默认为 131072，最大可设置为
            1048576。如果结果达到最大 Token 数而未结束，finish reason 将为 "length"；否则为
            "stop"。此值为期望返回的 Token 长度，而非输入加输出的总长度。如果输入加 max_completion_tokens
            超出模型上下文窗口，将返回 invalid_request_error。
        response_format:
          type: object
          description: >-
            控制模型输出格式。默认值为 {"type": "text"}，即纯文本输出。设置为 {"type": "json_object"}
            可启用 JSON 模式，确保输出为合法 JSON 对象（需在 prompt 中引导模型输出 JSON 并指定格式）。设置为
            {"type": "json_schema"} 可启用 Structured Output，按指定的 JSON Schema
            约束输出结构（推荐，需配合 json_schema 字段使用）。如果您在使用 JSON Schema 时遇到校验问题，欢迎到 walle
            GitHub Issues (https://github.com/MoonshotAI/walle/issues) 提交反馈。
          properties:
            type:
              type: string
              enum:
                - text
                - json_object
                - json_schema
              description: >-
                输出格式类型。text：默认，纯文本输出；json_object：保证输出为合法 JSON 对象；json_schema：按指定
                JSON Schema 约束输出（推荐，需配合 json_schema 字段使用）
            json_schema:
              type: object
              description: 当 type 为 json_schema 时使用，定义输出应遵循的 JSON Schema
              properties:
                name:
                  type: string
                  description: Schema 名称，用于标识
                strict:
                  type: boolean
                  default: true
                  description: >-
                    是否严格按 schema 约束输出。默认为 true。为 true 时 schema 需符合 MFJS
                    规范，不符合会返回错误或 warning；为 false 时仅保证输出为合法 JSON 对象，不强制约束内部结构。
                schema:
                  type: object
                  description: >-
                    JSON Schema 对象，定义输出应遵循的结构。需符合 MFJS（Moonshot Flavored JSON
                    Schema）规范。可使用 walle CLI 工具自检：go install
                    github.com/moonshotai/walle/cmd/walle@latest && walle
                    -schema '你的schema' -level strict
                  additionalProperties: true
              required:
                - name
                - schema
        stop:
          oneOf:
            - type: string
            - type: array
              items:
                type: string
              maxItems: 5
          default: null
          description: 停用词，完全匹配时将停止输出。匹配到的词本身不会被输出。最多允许 5 个字符串，每个不超过 32 字节
        stream:
          type: boolean
          default: false
          description: 是否以流式方式返回响应，默认 false
        stream_options:
          type: object
          description: 流式响应选项
          properties:
            include_usage:
              type: boolean
              default: false
              description: >-
                如果设置，将在 data: [DONE] 消息之前额外发送一个 chunk。该 chunk 的 usage 字段显示整个请求的
                Token 使用统计，choices 字段为空数组。其他所有 chunk 也会包含 usage 字段，但值为
                null。注意：如果流中断，可能无法收到包含总 Token 用量的最终 chunk
        tools:
          type: array
          description: 模型可调用的工具列表
          items:
            $ref: '#/components/schemas/ToolDefinition'
        prompt_cache_key:
          type: string
          default: null
          description: >-
            用于缓存相似请求的响应以优化缓存命中率。对于 Coding Agent，通常是代表单个会话的 session id 或 task
            id；退出并恢复会话时应保持不变。对于 Kimi Code Plan，此字段为必填以提高缓存命中率。对于其他多轮对话
            Agent，也建议使用此字段
        safety_identifier:
          type: string
          description: 用于检测可能违反使用政策的用户的稳定标识符。应为唯一标识每个用户的字符串。建议对用户名或邮箱进行哈希处理以避免发送可识别信息
        tool_choice:
          oneOf:
            - type: string
              enum:
                - auto
                - none
                - required
              description: auto：模型自行决定是否调用工具；none：不调用工具；required：强制调用工具
            - type: object
              description: 强制调用指定工具
              properties:
                type:
                  type: string
                  enum:
                    - function
                function:
                  type: object
                  properties:
                    name:
                      type: string
                      description: 要调用的函数名称
                  required:
                    - name
              required:
                - type
                - function
          description: >-
            控制模型是否调用工具。`auto`（默认）：模型自行决定是否调用工具；`none`：不调用工具；`required`：强制调用工具；也可传入特定函数对象强制调用指定工具。
    KimiK3Message:
      oneOf:
        - $ref: '#/components/schemas/Message'
          title: 标准消息
        - $ref: '#/components/schemas/KimiK3DynamicToolMessage'
          title: 动态工具消息
      description: Kimi K3 对话消息。既支持标准消息，也支持不含 content、通过 tools 声明动态工具的 system 消息。
    ChatRequestBase:
      type: object
      properties:
        messages:
          type: array
          description: >-
            包含迄今为止对话的消息列表。每个元素格式为 {"role": "user", "content": "你好"}。role 支持
            system、user、assistant、tool 其一，content 不得为空。content 字段可以是 string，也可以是
            array[object]（用于多模态输入）
          items:
            $ref: '#/components/schemas/Message'
        max_tokens:
          type: integer
          deprecated: true
          description: 已弃用，请使用 max_completion_tokens
        max_completion_tokens:
          type: integer
          description: >-
            聊天补全生成的最大 Token 数量。默认值因模型而异：Kimi K3 默认为 131072，最大可设置为
            1048576。如果结果达到最大 Token 数而未结束，finish reason 将为 "length"；否则为
            "stop"。此值为期望返回的 Token 长度，而非输入加输出的总长度。如果输入加 max_completion_tokens
            超出模型上下文窗口，将返回 invalid_request_error。
        response_format:
          type: object
          description: >-
            控制模型输出格式。默认值为 {"type": "text"}，即纯文本输出。设置为 {"type": "json_object"}
            可启用 JSON 模式，确保输出为合法 JSON 对象（需在 prompt 中引导模型输出 JSON 并指定格式）。设置为
            {"type": "json_schema"} 可启用 Structured Output，按指定的 JSON Schema
            约束输出结构（推荐，需配合 json_schema 字段使用）。如果您在使用 JSON Schema 时遇到校验问题，欢迎到 walle
            GitHub Issues (https://github.com/MoonshotAI/walle/issues) 提交反馈。
          properties:
            type:
              type: string
              enum:
                - text
                - json_object
                - json_schema
              description: >-
                输出格式类型。text：默认，纯文本输出；json_object：保证输出为合法 JSON 对象；json_schema：按指定
                JSON Schema 约束输出（推荐，需配合 json_schema 字段使用）
            json_schema:
              type: object
              description: 当 type 为 json_schema 时使用，定义输出应遵循的 JSON Schema
              properties:
                name:
                  type: string
                  description: Schema 名称，用于标识
                strict:
                  type: boolean
                  default: true
                  description: >-
                    是否严格按 schema 约束输出。默认为 true。为 true 时 schema 需符合 MFJS
                    规范，不符合会返回错误或 warning；为 false 时仅保证输出为合法 JSON 对象，不强制约束内部结构。
                schema:
                  type: object
                  description: >-
                    JSON Schema 对象，定义输出应遵循的结构。需符合 MFJS（Moonshot Flavored JSON
                    Schema）规范。可使用 walle CLI 工具自检：go install
                    github.com/moonshotai/walle/cmd/walle@latest && walle
                    -schema '你的schema' -level strict
                  additionalProperties: true
              required:
                - name
                - schema
        stop:
          oneOf:
            - type: string
            - type: array
              items:
                type: string
              maxItems: 5
          default: null
          description: 停用词，完全匹配时将停止输出。匹配到的词本身不会被输出。最多允许 5 个字符串，每个不超过 32 字节
        stream:
          type: boolean
          default: false
          description: 是否以流式方式返回响应，默认 false
        stream_options:
          type: object
          description: 流式响应选项
          properties:
            include_usage:
              type: boolean
              default: false
              description: >-
                如果设置，将在 data: [DONE] 消息之前额外发送一个 chunk。该 chunk 的 usage 字段显示整个请求的
                Token 使用统计，choices 字段为空数组。其他所有 chunk 也会包含 usage 字段，但值为
                null。注意：如果流中断，可能无法收到包含总 Token 用量的最终 chunk
        tools:
          type: array
          description: 模型可调用的工具列表
          items:
            $ref: '#/components/schemas/ToolDefinition'
        prompt_cache_key:
          type: string
          default: null
          description: >-
            用于缓存相似请求的响应以优化缓存命中率。对于 Coding Agent，通常是代表单个会话的 session id 或 task
            id；退出并恢复会话时应保持不变。对于 Kimi Code Plan，此字段为必填以提高缓存命中率。对于其他多轮对话
            Agent，也建议使用此字段
        safety_identifier:
          type: string
          description: 用于检测可能违反使用政策的用户的稳定标识符。应为唯一标识每个用户的字符串。建议对用户名或邮箱进行哈希处理以避免发送可识别信息
        tool_choice:
          oneOf:
            - type: string
              enum:
                - auto
                - none
                - required
              description: auto：模型自行决定是否调用工具；none：不调用工具；required：强制调用工具
            - type: object
              description: 强制调用指定工具
              properties:
                type:
                  type: string
                  enum:
                    - function
                function:
                  type: object
                  properties:
                    name:
                      type: string
                      description: 要调用的函数名称
                  required:
                    - name
              required:
                - type
                - function
          description: >-
            控制模型是否调用工具。`auto`（默认）：模型自行决定是否调用工具；`none`：不调用工具；`required`：强制调用工具；也可传入特定函数对象强制调用指定工具。
      required:
        - messages
    ChoiceDelta:
      type: object
      properties:
        index:
          type: integer
          description: 选项索引
        delta:
          type: object
          description: 增量内容对象
          properties:
            role:
              type: string
              description: 消息角色（仅在首个 chunk 中出现）
            content:
              type: string
              description: 消息内容片段
            tool_calls:
              type: array
              description: 模型发起的工具调用片段
              items:
                type: object
                properties:
                  id:
                    type: string
                  type:
                    type: string
                    enum:
                      - function
                  function:
                    type: object
                    properties:
                      name:
                        type: string
                      arguments:
                        type: string
                        description: 函数参数的 JSON 字符串
        finish_reason:
          type:
            - string
            - 'null'
          enum:
            - stop
            - length
            - tool_calls
            - null
          description: 停止原因，仅在最后一个 chunk 中出现
        usage:
          type: object
          description: Token 使用统计（可选）
          properties:
            prompt_tokens:
              type: integer
              description: 提示中的 Token 数量
            completion_tokens:
              type: integer
              description: 补全中的 Token 数量
            total_tokens:
              type: integer
              description: 使用的总 Token 数量
    ToolDefinition:
      type: object
      properties:
        type:
          type: string
          enum:
            - function
        function:
          type: object
          properties:
            name:
              type: string
              description: 函数名称。必须符合正则表达式：^[a-zA-Z_][a-zA-Z0-9-_]{2,63}$
              pattern: ^[a-zA-Z_][a-zA-Z0-9-_]{2,63}$
            description:
              type: string
              description: 函数功能描述
            parameters:
              type: object
              description: >-
                函数参数，JSON Schema 格式。需符合 [MFJS（Moonshot Flavored JSON
                Schema）规范](https://github.com/MoonshotAI/walle/blob/main/docs/mfjs-spec.zh.md)。
              additionalProperties: true
            strict:
              type: boolean
              default: true
              description: >-
                是否严格按 parameters schema 约束工具调用参数的输出。默认为 true。设为 false 时仅保证输出为合法
                JSON 对象，不强制约束内部结构。
          required:
            - name
            - parameters
      required:
        - type
        - function
    Message:
      type: object
      properties:
        role:
          type: string
          enum:
            - system
            - user
            - assistant
            - tool
          example: user
          description: 消息发送者的角色，支持 system、user、assistant、tool
        content:
          oneOf:
            - type: string
            - type: array
              items:
                oneOf:
                  - title: text
                    type: object
                    properties:
                      type:
                        type: string
                        enum:
                          - text
                      text:
                        type: string
                    required:
                      - type
                      - text
                  - title: image_url
                    type: object
                    properties:
                      type:
                        type: string
                        enum:
                          - image_url
                      image_url:
                        oneOf:
                          - type: object
                            properties:
                              url:
                                type: string
                            required:
                              - url
                          - type: string
                    required:
                      - type
                      - image_url
                  - title: video_url
                    type: object
                    properties:
                      type:
                        type: string
                        enum:
                          - video_url
                      video_url:
                        oneOf:
                          - type: object
                            properties:
                              url:
                                type: string
                            required:
                              - url
                          - type: string
                    required:
                      - type
                      - video_url
          example: 你好
          description: 消息内容。可以是纯文本字符串，也可以是包含 text/image_url/video_url 类型的对象数组（用于多模态输入）
        name:
          type: string
          default: null
          description: 消息发送者的名称（可选）
        partial:
          type: boolean
          default: false
          description: 在最后一条 assistant 消息中设置为 true 以启用 Partial Mode
      required:
        - role
        - content
    KimiK3DynamicToolMessage:
      type: object
      description: >-
        Kimi K3 动态加载工具消息。该消息只能使用 system 角色，通过 tools 字段在当前对话位置声明后续对话可用的工具，且不包含
        content 字段。
      properties:
        role:
          type: string
          enum:
            - system
          description: 动态加载工具消息的角色，固定为 system
        tools:
          type: array
          description: 从该消息位置开始可供模型调用的工具列表
          items:
            $ref: '#/components/schemas/ToolDefinition'
      required:
        - role
        - tools
      additionalProperties: false
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      description: >-
        Authorization 请求头需要一个 Bearer 令牌。使用 MOONSHOT_API_KEY 作为令牌。这是一个服务端密钥，请在
        [API 密钥页面](https://platform.kimi.com/console/api-keys) 生成。

````