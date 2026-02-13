> ## Documentation Index
> Fetch the complete documentation index at: https://docs.z.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Chat Completion

> Create a chat completion model that generates AI replies for given conversation messages. It supports multimodal inputs (text, images, audio, video, file), offers configurable parameters (like temperature, max tokens, tool use), and supports both streaming and non-streaming output modes.



## OpenAPI

````yaml POST /paas/v4/chat/completions
openapi: 3.0.1
info:
  title: Z.AI API
  description: Z.AI API available endpoints
  license:
    name: Z.AI Developer Agreement and Policy
    url: https://chat.z.ai/legal-agreement/terms-of-service
  version: 1.0.0
  contact:
    name: Z.AI Developers
    url: https://chat.z.ai/legal-agreement/privacy-policy
    email: user_feedback@z.ai
servers:
  - url: https://api.z.ai/api
    description: Production server
security:
  - bearerAuth: []
paths:
  /paas/v4/chat/completions:
    post:
      description: >-
        Create a chat completion model that generates AI replies for given
        conversation messages. It supports multimodal inputs (text, images,
        audio, video, file), offers configurable parameters (like temperature,
        max tokens, tool use), and supports both streaming and non-streaming
        output modes.
      parameters:
        - $ref: '#/components/parameters/AcceptLanguage'
      requestBody:
        content:
          application/json:
            schema:
              oneOf:
                - $ref: '#/components/schemas/ChatCompletionTextRequest'
                  title: Text Model
                - $ref: '#/components/schemas/ChatCompletionVisionRequest'
                  title: Vision Model
            examples:
              Basic Example:
                value:
                  model: glm-5
                  messages:
                    - role: system
                      content: You are a useful AI assistant.
                    - role: user
                      content: >-
                        Please tell us about the development of artificial
                        intelligence.
                  temperature: 1
                  stream: false
              Stream Example:
                value:
                  model: glm-5
                  messages:
                    - role: user
                      content: Write a poem about spring.
                  temperature: 1
                  stream: true
              Thinking Example:
                value:
                  model: glm-5
                  messages:
                    - role: user
                      content: Write a poem about spring.
                  thinking:
                    type: enabled
                  stream: true
              Multi Conversation:
                value:
                  model: glm-5
                  messages:
                    - role: system
                      content: You are a professional programming assistant.
                    - role: user
                      content: What is recursion?
                    - role: assistant
                      content: >-
                        Recursion is a programming technique where a function
                        calls itself to solve a problem... What is recursion
                    - role: user
                      content: Can you give me an example of Python recursion?
                  stream: true
              Image Visual Example:
                value:
                  model: glm-4.6v
                  messages:
                    - role: user
                      content:
                        - type: image_url
                          image_url:
                            url: https://cdn.bigmodel.cn/static/logo/register.png
                        - type: image_url
                          image_url:
                            url: https://cdn.bigmodel.cn/static/logo/api-key.png
                        - type: text
                          text: What are the pics talk about?
              Video Visual Example:
                value:
                  model: glm-4.6v
                  messages:
                    - role: user
                      content:
                        - type: video_url
                          video_url:
                            url: >-
                              https://cdn.bigmodel.cn/agent-demos/lark/113123.mov
                        - type: text
                          text: What are the video show about?
              File Visual Example:
                value:
                  model: glm-4.6v
                  messages:
                    - role: user
                      content:
                        - type: file_url
                          file_url:
                            url: https://cdn.bigmodel.cn/static/demo/demo2.txt
                        - type: file_url
                          file_url:
                            url: https://cdn.bigmodel.cn/static/demo/demo1.pdf
                        - type: text
                          text: What are the files show about?
              Function Call Example:
                value:
                  model: glm-5
                  messages:
                    - role: user
                      content: >-
                        Is there an example of how the weather in Beijing is
                        today?
                  tools:
                    - type: function
                      function:
                        name: get_weather
                        description: Get weather information for the specified city.
                        parameters:
                          type: object
                          properties:
                            city:
                              type: string
                              description: City Name
                          required:
                            - city
                  tool_choice: auto
                  temperature: 0.3
        required: true
      responses:
        '200':
          description: Processing successful
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatCompletionResponse'
        default:
          description: The request has failed.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
components:
  parameters:
    AcceptLanguage:
      name: Accept-Language
      in: header
      schema:
        type: string
        description: Config desired response language for HTTP requests.
        default: en-US,en
        example: en-US,en
        enum:
          - en-US,en
      required: false
  schemas:
    ChatCompletionTextRequest:
      required:
        - model
        - messages
      type: object
      properties:
        model:
          type: string
          description: >-
            The model code to be called. GLM-5 are the latest flagship model
            series, foundational models specifically designed for agent
            applications.
          example: glm-5
          default: glm-5
          enum:
            - glm-5
            - glm-4.7
            - glm-4.7-flash
            - glm-4.7-flashx
            - glm-4.6
            - glm-4.5
            - glm-4.5-air
            - glm-4.5-x
            - glm-4.5-airx
            - glm-4.5-flash
            - glm-4-32b-0414-128k
        messages:
          type: array
          description: >-
            The current conversation message list as the model’s prompt input,
            provided in JSON array format, e.g.,`{“role”: “user”, “content”:
            “Hello”}`. Possible message types include system messages, user
            messages, assistant messages, and tool messages. Note: The input
            must not consist of system messages or assistant messages only.
          items:
            oneOf:
              - title: User Message
                type: object
                properties:
                  role:
                    type: string
                    enum:
                      - user
                    description: Role of the message author
                    default: user
                  content:
                    oneOf:
                      - type: string
                        description: Text message content
                        example: >-
                          What opportunities and challenges will the Chinese
                          large model industry face in 2025?
                required:
                  - role
                  - content
              - title: System Message
                type: object
                properties:
                  role:
                    type: string
                    enum:
                      - system
                    description: Role of the message author
                    default: system
                  content:
                    oneOf:
                      - type: string
                        description: Message text content
                        example: You are a helpful assistant.
                required:
                  - role
                  - content
              - title: Assistant Message
                type: object
                description: Can include tool calls
                properties:
                  role:
                    type: string
                    enum:
                      - assistant
                    description: Role of the message author
                    default: assistant
                  content:
                    oneOf:
                      - type: string
                        description: Text message content
                        example: I'll help you with that analysis.
                  tool_calls:
                    type: array
                    description: >-
                      Tool call messages generated by the model. When this field
                      is provided, content is usually empty.
                    items:
                      type: object
                      properties:
                        id:
                          type: string
                          description: Tool call ID
                        type:
                          type: string
                          description: Tool type, supports web_search, retrieval, function
                          enum:
                            - function
                            - web_search
                            - retrieval
                        function:
                          type: object
                          description: >-
                            Function call information, not empty when type is
                            function
                          properties:
                            name:
                              type: string
                              description: Function name
                            arguments:
                              type: string
                              description: Function parameters, JSON format string
                          required:
                            - name
                            - arguments
                      required:
                        - id
                        - type
                required:
                  - role
              - title: Tool Message
                type: object
                properties:
                  role:
                    type: string
                    enum:
                      - tool
                    description: Role of the message author
                    default: tool
                  content:
                    oneOf:
                      - type: string
                        description: Message text content
                        example: 'Function executed successfully with result: ...'
                  tool_call_id:
                    type: string
                    description: Indicates the tool call ID corresponding to this message
                required:
                  - role
                  - content
                  - tool_call_id
          minItems: 1
        request_id:
          type: string
          description: >-
            Passed by the user side, needs to be unique; used to distinguish
            each request. If not provided by the user side, the platform will
            generate one by default.
        do_sample:
          type: boolean
          example: true
          default: true
          description: >-
            When do_sample is true, sampling strategy is enabled; when do_sample
            is false, sampling strategy parameters such as temperature and top_p
            will not take effect. Default value is `true`.
        stream:
          type: boolean
          example: false
          default: false
          description: >-
            This parameter should be set to false or omitted when using
            synchronous call. It indicates that the model returns all content at
            once after generating all content. Default value is false. If set to
            true, the model will return the generated content in chunks via
            standard Event Stream. When the Event Stream ends, a `data: [DONE]`
            message will be returned.
        thinking:
          $ref: '#/components/schemas/ChatThinking'
        temperature:
          type: number
          description: >-
            Sampling temperature, controls the randomness of the output, must be
            a positive number within the range: `[0.0, 1.0]`. The GLM-5 GLM-4.7
            GLM-4.6 series default value is `1.0`, GLM-4.5 series default value
            is `0.6`, GLM-4-32B-0414-128K default value is `0.75`.
          format: float
          example: 1
          default: 1
          minimum: 0
          maximum: 1
        top_p:
          type: number
          description: >-
            Another method of temperature sampling, value range is: `[0.01,
            1.0]`. The GLM-5, GLM-4.7, GLM-4.6, GLM-4.5 series default value is
            `0.95`, GLM-4-32B-0414-128K default value is `0.9`.
          format: float
          example: 0.95
          default: 0.95
          minimum: 0.01
          maximum: 1
        max_tokens:
          type: integer
          description: >-
            The maximum number of tokens for model output, the GLM-5 GLM-4.7
            GLM-4.6 series supports 128K maximum output, the GLM-4.5 series
            supports 96K maximum output, the GLM-4.6v series supports 32K
            maximum output, the GLM-4.5v series supports 16K maximum output,
            GLM-4-32B-0414-128K supports 16K maximum output.
          example: 1024
          minimum: 1
          maximum: 131072
        tool_stream:
          type: boolean
          example: false
          default: false
          description: >-
            Whether to enable streaming response for Function Calls. Default
            value is false. Only supported by GLM-4.6. Refer the [Stream Tool
            Call](/guides/tools/stream-tool)
        tools:
          type: array
          description: >
            A list of tools the model may call. Currently, only functions are
            supported as a tool. Use this to provide a list of functions the
            model may generate JSON inputs for. A max of 128 functions are
            supported.
          items:
            anyOf:
              - $ref: '#/components/schemas/FunctionToolSchema'
              - $ref: '#/components/schemas/RetrievalToolSchema'
              - $ref: '#/components/schemas/WebSearchToolSchema'
        tool_choice:
          oneOf:
            - type: string
              enum:
                - auto
              description: >-
                Used to control how the model selects which function to call.
                This is only applicable when the tool type is function. The
                default value is auto, and only auto is supported.
          description: Controls how the model selects a tool.
        stop:
          type: array
          description: >-
            Stop word list. Generation stops when the model encounters any
            specified string. Currently, only one stop word is supported, in the
            format ["stop_word1"].
          items:
            type: string
          maxItems: 1
        response_format:
          type: object
          description: >-
            Specifies the response format of the model. Defaults to text.
            Supports two formats:{ "type": "text" } plain text mode, returns
            natural language text, { "type": "json_object" } JSON mode, returns
            valid JSON data. When using JSON mode, it’s recommended to clearly
            request JSON output in the prompt.
          properties:
            type:
              type: string
              enum:
                - text
                - json_object
              default: text
              description: >-
                Output format type: text for plain text, json_object for
                JSON-formatted output.
          required:
            - type
        user_id:
          type: string
          description: >-
            Unique ID for the end user, 6–128 characters. Avoid using sensitive
            information.
          minLength: 6
          maxLength: 128
    ChatCompletionVisionRequest:
      required:
        - model
        - messages
      type: object
      properties:
        model:
          type: string
          description: >-
            The model code to be called. GLM-4.6V are the new generation of
            visual reasoning models. `AutoGLM-Phone-Multilingual` is mobile
            intelligent assistant model.
          example: glm-4.6v
          default: glm-4.6v
          enum:
            - glm-4.6v
            - autoglm-phone-multilingual
            - glm-4.6v-flash
            - glm-4.6v-flashx
            - glm-4.5v
        messages:
          type: array
          description: >-
            The current conversation message list as the model’s prompt input,
            provided in JSON array format, e.g.,`{“role”: “user”, “content”:
            “Hello”}`. Possible message types include system messages, user
            messages. Note: The input must not consist of system or assistant
            messages only.
          items:
            oneOf:
              - title: User Message
                type: object
                properties:
                  role:
                    type: string
                    enum:
                      - user
                    description: Role of the message author
                    default: user
                  content:
                    oneOf:
                      - type: array
                        description: >-
                          Multimodal message content, supports text, images,
                          video, file
                        items:
                          $ref: '#/components/schemas/VisionMultimodalContentItem'
                      - type: string
                        description: >-
                          Text message content (can switch to multimodal message
                          above)
                        example: >-
                          What opportunities and challenges will the Chinese
                          large model industry face in 2025?
                required:
                  - role
                  - content
              - title: System Message
                type: object
                properties:
                  role:
                    type: string
                    enum:
                      - system
                    description: Role of the message author
                    default: system
                  content:
                    oneOf:
                      - type: string
                        description: Message text content
                        example: You are a helpful assistant.
                required:
                  - role
                  - content
              - title: Assistant Message
                type: object
                description: Can include tool calls
                properties:
                  role:
                    type: string
                    enum:
                      - assistant
                    description: Role of the message author
                    default: assistant
                  content:
                    oneOf:
                      - type: string
                        description: Text message content
                        example: I'll help you with that analysis.
                required:
                  - role
          minItems: 1
        request_id:
          type: string
          description: >-
            Passed by the user side, needs to be unique; used to distinguish
            each request. If not provided by the user side, the platform will
            generate one by default.
        do_sample:
          type: boolean
          example: true
          default: true
          description: >-
            When do_sample is true, sampling strategy is enabled; when do_sample
            is false, sampling strategy parameters such as temperature and top_p
            will not take effect. Default value is `true`.
        stream:
          type: boolean
          example: false
          default: false
          description: >-
            This parameter should be set to false or omitted when using
            synchronous call. It indicates that the model returns all content at
            once after generating all content. Default value is false. If set to
            true, the model will return the generated content in chunks via
            standard Event Stream. When the Event Stream ends, a `data: [DONE]`
            message will be returned.
        thinking:
          $ref: '#/components/schemas/ChatThinking'
        temperature:
          type: number
          description: >-
            Sampling temperature, controls the randomness of the output, must be
            a positive number within the range: `[0.0, 1.0]`. The GLM-4.6V,
            GLM-4.5V series default value is `0.8`, the
            autoglm-phone-multilingual default value is `0.0`.
          format: float
          example: 0.8
          default: 0.8
          minimum: 0
          maximum: 1
        top_p:
          type: number
          description: >-
            Another method of temperature sampling, value range is: `[0.01,
            1.0]`, value range is: `[0.01, 1.0]`. The GLM-4.6V, GLM-4.5V series
            default value is `0.6`, the autoglm-phone-multilingual default value
            is `0.85`.
          format: float
          example: 0.6
          default: 0.6
          minimum: 0.01
          maximum: 1
        max_tokens:
          type: integer
          description: >-
            The maximum number of tokens for model output, the GLM-4.6V series
            supports 32K maximum output, the GLM-4.5V series supports 16K
            maximum output, the autoglm-phone-multilingual supports 4K maximum
            output.
          example: 1024
          minimum: 1
          maximum: 16384
        tools:
          type: array
          description: >
            A list of tools the model may call. Only support by GLM-4.6V series
            and autoglm-phone-multilingual. Use this to provide a list of
            functions the model may generate JSON inputs for. A max of 128
            functions are supported.
          items:
            anyOf:
              - $ref: '#/components/schemas/FunctionToolSchema'
        tool_choice:
          oneOf:
            - type: string
              enum:
                - auto
              description: >-
                Used to control how the model selects which function to call.
                This is only applicable when the tool type is function. The
                default value is auto, and only auto is supported.
          description: Controls how the model selects a tool.
        stop:
          type: array
          description: >-
            Stop word list. Generation stops when the model encounters any
            specified string. Currently, only one stop word is supported, in the
            format ["stop_word1"].
          items:
            type: string
          maxItems: 1
        user_id:
          type: string
          description: >-
            Unique ID for the end user, 6–128 characters. Avoid using sensitive
            information.
          minLength: 6
          maxLength: 128
    ChatCompletionResponse:
      type: object
      properties:
        id:
          type: string
          description: Task ID
        request_id:
          description: Request ID
          type: string
        created:
          description: Request creation time, Unix timestamp in seconds
          type: integer
        model:
          description: Model name
          type: string
        choices:
          type: array
          description: List of model responses
          items:
            type: object
            properties:
              index:
                type: integer
                description: Result index.
              message:
                $ref: '#/components/schemas/ChatCompletionResponseMessage'
              finish_reason:
                type: string
                description: >-
                  Reason for model inference termination. Can be ‘stop’,
                  ‘tool_calls’, ‘length’, ‘sensitive’, or ‘network_error’.
        usage:
          type: object
          description: Token usage statistics returned when the model call ends.
          properties:
            prompt_tokens:
              type: number
              description: Number of tokens in user input
            completion_tokens:
              type: number
              description: Number of output tokens
            prompt_tokens_details:
              type: object
              properties:
                cached_tokens:
                  type: number
                  description: Number of tokens served from cache
            total_tokens:
              type: integer
              description: Total number of tokens
        web_search:
          description: Search results.
          type: array
          items:
            $ref: '#/components/schemas/WebSearchObjectResponse'
    Error:
      required:
        - code
        - message
      type: object
      description: The request has failed.
      properties:
        code:
          type: integer
          format: int32
          description: Error code.
        message:
          type: string
          description: Error message.
    ChatThinking:
      type: object
      description: >-
        Only supported by GLM-4.5 series and higher models. This parameter is
        used to control whether the model enable the chain of thought.
      properties:
        type:
          type: string
          description: >-
            Whether to enable the chain of thought(When enabled, GLM-5 GLM-4.7
            GLM-4.5V will think compulsorily, while GLM-4.6, GLM-4.6V, GLM-4.5
            and others will automatically determine whether to think), default:
            enabled
          default: enabled
          enum:
            - enabled
            - disabled
        clear_thinking:
          type: boolean
          description: >-
            Default value is True. Controls whether to clear `reasoning_content`
            from previous conversation turns. View more in [Thinking
            Mode](/guides/capabilities/thinking-mode).
             - `true` (default): For this request, the system ignores/removes `reasoning_content` from prior turns, and only keeps non-reasoning context (e.g., user/assistant visible text, tool calls, and tool results). This is recommended for general chat or lightweight tasks to reduce context length and cost.
             - `false`: Retains `reasoning_content` from prior turns and includes it in the context sent to the model. To enable Preserved Thinking, you must forward the full, unmodified, and correctly ordered historical `reasoning_content` in `messages`. Missing, truncated, rewritten, or reordered blocks may degrade performance or prevent the feature from taking effect.
             - Notes: This parameter only affects cross-turn historical thinking blocks; it does not change whether the model generates/returns thinking in the current turn.
          default: true
          example: true
    FunctionToolSchema:
      type: object
      title: Function Call
      properties:
        type:
          type: string
          default: function
          enum:
            - function
        function:
          $ref: '#/components/schemas/FunctionObject'
      required:
        - type
        - function
      additionalProperties: false
    RetrievalToolSchema:
      type: object
      title: Retrieval
      properties:
        type:
          type: string
          default: retrieval
          enum:
            - retrieval
        retrieval:
          $ref: '#/components/schemas/RetrievalObject'
      required:
        - type
        - retrieval
      additionalProperties: false
    WebSearchToolSchema:
      type: object
      title: Web Search
      properties:
        type:
          type: string
          default: web_search
          enum:
            - web_search
        web_search:
          $ref: '#/components/schemas/WebSearchObject'
      required:
        - type
        - web_search
      additionalProperties: false
    VisionMultimodalContentItem:
      oneOf:
        - title: Text
          type: object
          properties:
            type:
              type: string
              enum:
                - text
              description: Content type is text
              default: text
            text:
              type: string
              description: Text content
          required:
            - type
            - text
          additionalProperties: false
        - title: Image
          type: object
          properties:
            type:
              type: string
              enum:
                - image_url
              description: Content type is image URL
              default: image_url
            image_url:
              type: object
              description: Image information
              properties:
                url:
                  type: string
                  description: >-
                    Image URL or Base64 encoding. Image size limit is under 5M
                    per image, with pixels not exceeding 6000*6000. Supports
                    jpg, png, jpeg formats.
              required:
                - url
              additionalProperties: false
          required:
            - type
            - image_url
          additionalProperties: false
        - title: Video
          type: object
          properties:
            type:
              type: string
              enum:
                - video_url
              description: Content type is video URL
              default: video_url
            video_url:
              type: object
              description: Video information.
              properties:
                url:
                  type: string
                  description: >-
                    Video URL address.The video size is limited to within 200
                    MB, and the format supports `mp4`，`mkv`，`mov`.
              required:
                - url
              additionalProperties: false
          required:
            - type
            - video_url
          additionalProperties: false
        - title: File
          type: object
          properties:
            type:
              type: string
              enum:
                - file_url
              description: >-
                Content type is file URL, not support passing both the
                `file_url` and `image_url` or `video_url` parameters at the same
                time.
              default: file_url
            file_url:
              type: object
              description: File information.
              properties:
                url:
                  type: string
                  description: >-
                    File URL address. Only GLM-4.6V, GLM-4.5V supported.
                    Supports formats such as pdf、txt、word、jsonl、xlsx、pptx, with
                    a maximum of 50.
              required:
                - url
              additionalProperties: false
          required:
            - type
            - file_url
          additionalProperties: false
    ChatCompletionResponseMessage:
      type: object
      properties:
        role:
          type: string
          description: Current conversation role, default is ‘assistant’ (model)
          example: assistant
        content:
          type: string
          description: >-
            Current conversation content. Hits function is null, otherwise
            returns model inference result.

            For the GLM-4.5V series models, the output may contain the reasoning
            process tags `<think> </think>` or the text boundary tags
            `<|begin_of_box|> <|end_of_box|>`.
        reasoning_content:
          type: string
          description: Reasoning content, supports by GLM-4.5 series.
        tool_calls:
          type: array
          description: >-
            Function names and parameters generated by the model that should be
            called.
          items:
            $ref: '#/components/schemas/ChatCompletionResponseMessageToolCall'
    WebSearchObjectResponse:
      type: object
      properties:
        title:
          type: string
          description: Title.
        content:
          type: string
          description: Content summary.
        link:
          type: string
          description: Result URL.
        media:
          type: string
          description: Website name.
        icon:
          type: string
          description: Website icon.
        refer:
          type: string
          description: Index number.
        publish_date:
          type: string
          description: Website publication date.
    FunctionObject:
      type: object
      properties:
        name:
          type: string
          description: >-
            The name of the function to be called. Must be a-z, A-Z, 0-9, or
            contain underscores and dashes, with a maximum length of 64.
          minLength: 1
          maxLength: 64
          pattern: ^[a-zA-Z0-9_-]+$
        description:
          type: string
          description: >-
            A description of what the function does, used by the model to choose
            when and how to call the function.
        parameters:
          $ref: '#/components/schemas/FunctionParameters'
      required:
        - name
        - description
        - parameters
    RetrievalObject:
      type: object
      properties:
        knowledge_id:
          type: string
          description: Knowledge base ID, created or obtained from the platform
        prompt_template:
          type: string
          description: >-
            Prompt template for requesting the model, a custom request template
            containing placeholders `{{ knowledge }}` and `{{ question }}`.
            Default template: Search for the answer to the question
            `{{question}}` in the document `{{ knowledge }}`. If an answer is
            found, respond only using statements from the document; if no answer
            is found, use your own knowledge to answer and inform the user that
            the information is not from the document. Do not repeat the
            question, start the answer directly.
      required:
        - knowledge_id
    WebSearchObject:
      type: object
      properties:
        enable:
          type: boolean
          description: |-
            Whether to enable search functionality.
            Default is `false`. Set to true to `enable`.
        search_engine:
          type: string
          description: |-
            Type of search engine.
            Default is `search_pro_jina`. Supports: `search_pro_jina`.
          enum:
            - search_pro_jina
        search_query:
          type: string
          description: Force trigger a search
        count:
          type: integer
          description: |
            Number of returned results
            Range: `1-50`, max `50` results per search
            Default is `10`
            Supported engines: `search_pro_jina`
          minimum: 1
          maximum: 50
        search_domain_filter:
          type: string
          description: >-
            Limits search results to specified whitelisted domains. Whitelist:
            input domains directly (e.g., www.example.com)

            Supported engines: `search_pro_jina`
        search_recency_filter:
          type: string
          description: |-
            Limits search to a specific time range.
            Default is `noLimit`
            Values:
            `oneDay`, within a day
            `oneWeek`, within a week
            `oneMonth`, within a month
            `oneYear`, within a year
            `noLimit`, no limit (default)
            Supported engines: `search_pro_jina`
          enum:
            - oneDay
            - oneWeek
            - oneMonth
            - oneYear
            - noLimit
        content_size:
          type: string
          description: >-
            Number of characters for webpage summaries.

            Default is `medium`

            `medium`: Balanced mode for most queries. 400-600 characters

            `high`: Maximizes context for comprehensive answers, 2500
            characters.
          enum:
            - medium
            - high
        result_sequence:
          type: string
          description: >-
            Specifies whether search results are shown before or after model
            response. Options: `before`, `after`. Default is `after`
          enum:
            - before
            - after
        search_result:
          type: boolean
          description: |-
            Whether to return search results in the response.
            Default is `false`
        require_search:
          type: boolean
          description: |-
            Whether to force model response based on search result.
            Default is `false`
        search_prompt:
          type: string
          description: >-
            Prompt to customize how search results are processed.

            Default Prompt:

            `You are an intelligent Q&A expert with the ability to synthesize
            information, recognize time, understand semantics, and clean
            contradictory data. The current date is {{current_date}}. Use this
            as the only time reference. Based on the following information,
            provide a comprehensive and accurate answer to the user's
            question.Only extract valuable content for the answer. Ensure the
            answer is timely and authoritative. State the answer directly
            without citing data sources or internal processes.`
      required:
        - search_engine
    ChatCompletionResponseMessageToolCall:
      type: object
      properties:
        function:
          type: object
          description: >-
            Contains the function name and JSON format parameters generated by
            the model.
          properties:
            name:
              type: string
              description: Model-generated function name.
            arguments:
              type: object
              description: >-
                JSON format of the function call parameters generated by the
                model. Validate the parameters before calling the function.
          required:
            - name
            - arguments
        id:
          type: string
          description: Unique identifier for the hit function.
        type:
          type: string
          description: Tool type called by the model, currently only supports ‘function’.
    FunctionParameters:
      type: object
      description: >-
        Parameters defined using JSON Schema. Must pass a JSON Schema object to
        accurately define accepted parameters. Omit if no parameters are needed
        when calling the function.
      additionalProperties: true
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      description: >-
        Use the following format for authentication: Bearer [<your api
        key>](https://z.ai/manage-apikey/apikey-list)

````
