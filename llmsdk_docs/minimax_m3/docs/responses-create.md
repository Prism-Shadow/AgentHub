> ## Documentation Index
> Fetch the complete documentation index at: https://platform.minimax.io/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# Create Response

> Call MiniMax models via the OpenAI Responses API compatible main endpoint. Generates model replies, supports streaming and non-streaming.

## Reasoning Control

For `MiniMax-M3`, the `reasoning` field controls whether the response can include reasoning output.

* If `reasoning` is omitted, reasoning is disabled by default and the response does not include an output item with `type: "reasoning"`.
* `reasoning: {"effort": "none"}` is the default behavior and disables reasoning output for `MiniMax-M3`.
* Values `minimal`, `low`, `medium`, and `high` are accepted for compatibility and enable reasoning output, but they do not tune MiniMax-M3's reasoning depth.
* For M2.x models, reasoning cannot be disabled; `reasoning: {"effort": "none"}` is accepted but reasoning remains on.

```json theme={null}
{
  "model": "MiniMax-M3",
  "input": "Which is larger, 9.11 or 9.9?"
}
```

```json theme={null}
{
  "model": "MiniMax-M3",
  "input": "Which is larger, 9.11 or 9.9?",
  "reasoning": {
    "effort": "minimal"
  }
}
```


## OpenAPI

````yaml api-reference/text/api/openapi-responses.json POST /v1/responses
openapi: 3.1.0
info:
  title: MiniMax Responses API
  description: >-
    MiniMax OpenAI Responses API compatible endpoints, supporting chat
    generation and token estimation
  license:
    name: MIT
  version: 1.0.0
servers:
  - url: https://api.minimax.io
security:
  - bearerAuth: []
paths:
  /v1/responses:
    post:
      tags:
        - Responses
      summary: Create Response
      operationId: createResponse
      parameters:
        - name: Content-Type
          in: header
          required: true
          description: Media type of the request body. Must be set to `application/json`
          schema:
            type: string
            enum:
              - application/json
            default: application/json
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateResponseReq'
            examples:
              SimpleText:
                summary: Simple text input
                value:
                  model: MiniMax-M3
                  input: Hello!
              ConversationHistory:
                summary: Full conversation history
                value:
                  model: MiniMax-M3
                  instructions: You are a technical writing assistant.
                  input:
                    - role: user
                      content: >-
                        We are building a social product for overseas users. I
                        prefer Go + React Native + PostgreSQL. What do you
                        think?
              Streaming:
                summary: Streaming output
                value:
                  model: MiniMax-M3
                  input: Hello!
                  stream: true
              FunctionCall:
                summary: Function call
                value:
                  model: MiniMax-M3
                  input: What is the weather in Boston today?
                  tools:
                    - type: function
                      name: get_current_weather
                      description: Get the current weather in a given location
                      parameters:
                        type: object
                        properties:
                          location:
                            type: string
                            description: The city and state, e.g. San Francisco, CA
                          unit:
                            type: string
                            enum:
                              - celsius
                              - fahrenheit
                        required:
                          - location
                          - unit
              MultiTurnFunctionCall:
                summary: Multi-turn function call
                value:
                  model: MiniMax-M3
                  input:
                    - type: message
                      role: user
                      content: What is the weather in Beijing?
                    - type: function_call
                      call_id: call_abc123
                      name: get_weather
                      arguments: '{"city":"Beijing"}'
                    - type: function_call_output
                      call_id: call_abc123
                      output: Sunny, 22°C
                  tools:
                    - type: function
                      name: get_weather
                      description: Get current weather for a city
                      parameters:
                        type: object
                        properties:
                          city:
                            type: string
                        required:
                          - city
        required: true
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CreateResponseResp'
              examples:
                Response:
                  value:
                    id: abc123
                    object: response
                    created_at: 1764000000
                    model: MiniMax-M3
                    status: completed
                    output:
                      - id: abc123_msg
                        type: message
                        status: completed
                        role: assistant
                        content:
                          - type: output_text
                            text: Hello! I'm MiniMax. How can I help you today?
                            annotations: []
                    output_text: Hello! I'm MiniMax. How can I help you today?
                    usage:
                      input_tokens: 8
                      input_tokens_details:
                        cached_tokens: 0
                      output_tokens: 14
                      output_tokens_details:
                        reasoning_tokens: 0
                      total_tokens: 22
                    parallel_tool_calls: true
                    store: false
                    truncation: disabled
components:
  schemas:
    CreateResponseReq:
      type: object
      required:
        - model
        - input
      properties:
        model:
          type: string
          description: Model name to invoke, e.g. `MiniMax-M3`
          example: MiniMax-M3
        service_tier:
          type: string
          description: >-
            Service tier for request admission. Supported values are `standard`
            and `priority`. If omitted, the request uses the `standard` tier.
            The `priority` [price](/guides/pricing-paygo) is 1.5 times the
            `standard` price and ensures priority admission so the request is
            processed ahead of other requests, leading to faster responses and
            fewer failures.
          enum:
            - standard
            - priority
          default: standard
        input:
          description: >-
            Conversation content. Supports either a simple text or a full
            conversation history array
          oneOf:
            - type: string
              description: Simple text input
            - type: array
              description: Full conversation history
              items:
                $ref: '#/components/schemas/InputItem'
        instructions:
          type: string
          description: System instructions
        max_output_tokens:
          type: integer
          description: Maximum output token count
        temperature:
          type: number
          format: float
          description: Sampling temperature, range (0, 1]
          default: 1
          minimum: 0
          maximum: 1
        top_p:
          type: number
          format: float
          description: Nucleus sampling, range (0, 1]
          default: 0.95
          minimum: 0
          maximum: 1
        stream:
          type: boolean
          description: Set to `true` to enable SSE streaming response
          default: false
        tools:
          type: array
          description: Tool list
          items:
            $ref: '#/components/schemas/Tool'
        tool_choice:
          type: string
          enum:
            - none
            - auto
          description: >-
            Tool selection strategy: `none` means no tool will be called; `auto`
            lets the model decide whether to call tools
        metadata:
          type: object
          description: Request metadata. Both keys and values are strings
          additionalProperties:
            type: string
        prompt_cache_key:
          type: string
          description: Prompt cache routing identifier
        text:
          type: object
          description: Output format control
          properties:
            format:
              type: object
              properties:
                type:
                  type: string
                  enum:
                    - text
                  default: text
                  description: Output format type
        reasoning:
          type: object
          description: >-
            Reasoning control. For MiniMax-M3, the default is `none`, which
            disables reasoning. Set `effort` to a non-`none` value (`minimal`,
            `low`, `medium`, or `high`) to enable Adaptive Thinking, but this
            does not tune MiniMax-M3's reasoning depth. For M2.x models,
            reasoning cannot be disabled.
          properties:
            effort:
              type: string
              enum:
                - minimal
                - low
                - medium
                - high
                - none
              default: none
          required: []
    CreateResponseResp:
      type: object
      required:
        - id
        - object
        - created_at
        - model
        - status
        - output
      properties:
        id:
          type: string
          description: Response ID
          example: abc123
        object:
          type: string
          enum:
            - response
          description: Object type, always `response`
        created_at:
          type: integer
          description: Response creation time (Unix seconds)
        model:
          type: string
          description: Actual model that processed the request
        status:
          type: string
          enum:
            - completed
            - incomplete
            - failed
          description: Response status
        output:
          type: array
          description: Model output list
          items:
            $ref: '#/components/schemas/OutputItem'
        output_text:
          type: string
          nullable: true
          description: Convenience field. Concatenation of all text outputs
        usage:
          $ref: '#/components/schemas/Usage'
        error:
          type: object
          nullable: true
          description: Error info, only returned when `status=failed`
          properties:
            code:
              type: string
              description: Error code
            message:
              type: string
              description: Human-readable error description
        incomplete_details:
          type: object
          nullable: true
          description: Reason for incompletion, only returned when `status=incomplete`
          properties:
            reason:
              type: string
              enum:
                - max_output_tokens
                - content_filter
        parallel_tool_calls:
          type: boolean
          description: Whether parallel tool calls are supported
        store:
          type: boolean
          description: Whether the response is persisted
        truncation:
          type: string
          enum:
            - disabled
          description: Context truncation strategy
    InputItem:
      type: object
      description: >-
        Conversation history item. The `type` field determines the shape:
        `message` (default) / `function_call` / `function_call_output` /
        `reasoning`
      properties:
        type:
          type: string
          enum:
            - message
            - function_call
            - function_call_output
            - reasoning
          default: message
          description: Item type
        role:
          type: string
          enum:
            - user
            - assistant
            - system
            - developer
            - tool
          description: Message role (only when `type` is `message`)
        content:
          description: >-
            Message content; string or multimodal parts array (only when `type`
            is `message`)
          oneOf:
            - type: string
            - type: array
              items:
                $ref: '#/components/schemas/ContentPart'
        call_id:
          type: string
          description: >-
            Tool call ID (only when `type` is `function_call` or
            `function_call_output`)
        name:
          type: string
          description: Function name (only when `type` is `function_call`)
        arguments:
          type: string
          description: >-
            Function arguments as a JSON string (only when `type` is
            `function_call`)
        output:
          description: Tool return result (only when `type` is `function_call_output`)
          oneOf:
            - type: string
            - type: array
              items:
                $ref: '#/components/schemas/ContentPart'
        summary:
          type: array
          description: Reasoning segment array (only when `type` is `reasoning`)
          items:
            type: object
            properties:
              type:
                type: string
                enum:
                  - summary_text
              text:
                type: string
                description: Reasoning text
    Tool:
      type: object
      required:
        - type
        - name
      properties:
        type:
          type: string
          enum:
            - function
          description: Tool type
        name:
          type: string
          description: Function name
        description:
          type: string
          description: Function description, helps the model decide when to call it
        parameters:
          type: object
          description: Function parameter definition in JSON Schema format
    OutputItem:
      oneOf:
        - title: Message
          type: object
          description: Assistant reply
          properties:
            id:
              type: string
              description: '`<id>_msg` format'
            type:
              type: string
              enum:
                - message
            status:
              type: string
              enum:
                - completed
            role:
              type: string
              enum:
                - assistant
            content:
              type: array
              items:
                type: object
                properties:
                  type:
                    type: string
                    enum:
                      - output_text
                  text:
                    type: string
                    description: Text generated by the model
                  annotations:
                    type: array
                    description: Reference annotations
        - title: Reasoning
          type: object
          description: Reasoning output (only returned when reasoning is enabled)
          properties:
            id:
              type: string
              description: '`<id>_rs` format'
            type:
              type: string
              enum:
                - reasoning
            status:
              type: string
              enum:
                - completed
            summary:
              type: array
              description: Reasoning summary
            content:
              type: array
              items:
                type: object
                properties:
                  type:
                    type: string
                    enum:
                      - reasoning_text
                  text:
                    type: string
                    description: Reasoning text
        - title: Function Call
          type: object
          description: Function call
          properties:
            id:
              type: string
              description: '`<id>_fc_<index>` format'
            type:
              type: string
              enum:
                - function_call
            status:
              type: string
              enum:
                - completed
            call_id:
              type: string
              description: Tool call ID, used to correlate with `function_call_output`
            name:
              type: string
              description: Function name
            arguments:
              type: string
              description: Function arguments as a JSON string
    Usage:
      type: object
      properties:
        input_tokens:
          type: integer
          description: Input token count
        input_tokens_details:
          type: object
          properties:
            cached_tokens:
              type: integer
              description: Prompt cache hit tokens
        output_tokens:
          type: integer
          description: Output token count
        output_tokens_details:
          type: object
          properties:
            reasoning_tokens:
              type: integer
              description: >-
                Tokens consumed by reasoning (only counted when reasoning is
                enabled)
        total_tokens:
          type: integer
          description: Total token count
    ContentPart:
      type: object
      required:
        - type
      description: Message content part
      properties:
        type:
          type: string
          enum:
            - input_text
            - output_text
            - input_image
            - input_video
          description: |-
            Content part type:
            - `input_text` / `output_text`: Text part
            - `input_image`: Image input
            - `input_video`: Video input
        text:
          type: string
          description: Text content (when `type` is `input_text` / `output_text`)
        image_url:
          description: >-
            Image input (when `type` is `input_image`). Supported formats: JPEG,
            PNG, GIF, WEBP
          oneOf:
            - type: string
            - type: object
              required:
                - url
              properties:
                url:
                  type: string
                  description: Image URL or Base64 encoding
                detail:
                  type: string
                  enum:
                    - low
                    - default
                    - high
                  default: default
                  description: Image understanding precision tier
        video_url:
          description: >-
            Video input (when `type` is `input_video`). Supported formats: MP4,
            AVI, MOV, MKV
          oneOf:
            - type: string
            - type: object
              required:
                - url
              properties:
                url:
                  type: string
                  description: >-
                    Video URL or Base64 encoding. Use the File API to upload
                    large files
                fps:
                  type: number
                  format: float
                  default: 1
                  minimum: 0.2
                  maximum: 5
                  description: Frame extraction rate
                detail:
                  type: string
                  enum:
                    - low
                    - default
                    - high
                  default: default
                  description: Video understanding precision tier
                max_long_side_pixel:
                  type: integer
                  description: Pixel constraint on the longest side of video frames
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: |-
        `HTTP: Bearer Auth`
         - Security Scheme Type: http
         - HTTP Authorization Scheme: Bearer API_key, used to authenticate your account. View it in [Account Management > API Keys](https://platform.minimax.io/user-center/basic-information/interface-key)

````