> ## Documentation Index
> Fetch the complete documentation index at: https://openrouter.ai/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# Create a response

> Creates a streaming or non-streaming response using OpenResponses API format



## OpenAPI

````yaml /openapi/openapi.yaml post /responses
openapi: 3.1.0
info:
  contact:
    email: support@openrouter.ai
    name: OpenRouter Support
    url: https://openrouter.ai/docs
  description: OpenAI-compatible API with additional OpenRouter features
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT
  title: OpenRouter API
  version: 1.0.0
servers:
  - description: Production server
    url: https://openrouter.ai/api/v1
    x-speakeasy-server-id: production
security:
  - apiKey: []
tags:
  - description: API key management endpoints
    name: API Keys
  - description: Analytics and usage endpoints
    name: Analytics
  - description: Anthropic Messages endpoints
    name: Anthropic Messages
  - description: BYOK endpoints
    name: BYOK
  - description: Benchmarks endpoints
    name: Benchmarks
  - description: Chat completion endpoints
    name: Chat
  - description: Task classification market-share endpoints
    name: Classifications
  - description: Credit management endpoints
    name: Credits
  - description: >-
      Public OpenRouter usage datasets. Data returned by these endpoints is
      licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/):
      reuse and republish it, including commercially, with attribution to
      OpenRouter.
    name: Datasets
  - description: Text embedding endpoints
    name: Embeddings
  - description: Endpoint information
    name: Endpoints
  - description: Files endpoints
    name: Files
  - description: Generation history endpoints
    name: Generations
  - description: Guardrails endpoints
    name: Guardrails
  - description: Images endpoints
    name: Images
  - description: Model information endpoints
    name: Models
  - description: OAuth authentication endpoints
    name: OAuth
  - description: Observability endpoints
    name: Observability
  - description: Organization endpoints
    name: Organization
  - description: Presets endpoints
    name: Presets
  - description: Provider information endpoints
    name: Providers
  - description: Rerank endpoints
    name: Rerank
  - description: OpenAI-compatible Responses API endpoints
    name: Responses
  - description: SCIM endpoints
    name: SCIM
  - description: Speech-to-text endpoints
    name: STT
    x-displayName: Transcriptions
  - description: Text-to-speech endpoints
    name: TTS
    x-displayName: Speech
  - description: Video Generation endpoints
    name: Video Generation
  - description: Workspaces endpoints
    name: Workspaces
  - description: beta.Analytics endpoints
    name: beta.Analytics
externalDocs:
  description: OpenRouter Documentation
  url: https://openrouter.ai/docs
paths:
  /responses:
    post:
      tags:
        - Responses
      summary: Create a response
      description: >-
        Creates a streaming or non-streaming response using OpenResponses API
        format
      operationId: createResponses
      parameters:
        - description: >-
            Opt-in to surface routing metadata on the response under
            `openrouter_metadata`. Defaults to `disabled`. The legacy header
            `X-OpenRouter-Experimental-Metadata` is also accepted for backward
            compatibility.
          example: enabled
          in: header
          name: X-OpenRouter-Metadata
          required: false
          schema:
            $ref: '#/components/schemas/MetadataLevel'
      requestBody:
        content:
          application/json:
            example:
              input: Tell me a joke
              model: openai/gpt-4o
            schema:
              $ref: '#/components/schemas/ResponsesRequest'
        required: true
      responses:
        '200':
          content:
            application/json:
              example:
                completed_at: 1700000010
                created_at: 1700000000
                error: null
                frequency_penalty: null
                id: resp_abc123
                incomplete_details: null
                instructions: null
                max_output_tokens: null
                metadata: null
                model: openai/gpt-4o
                object: response
                output:
                  - content:
                      - annotations: []
                        text: >-
                          Why did the chicken cross the road? To get to the
                          other side!
                        type: output_text
                    id: msg-abc123
                    role: assistant
                    status: completed
                    type: message
                parallel_tool_calls: true
                presence_penalty: null
                status: completed
                temperature: null
                tool_choice: auto
                tools: []
                top_p: null
                usage:
                  input_tokens: 10
                  input_tokens_details:
                    cached_tokens: 0
                  output_tokens: 20
                  output_tokens_details:
                    reasoning_tokens: 0
                  total_tokens: 30
              schema:
                $ref: '#/components/schemas/OpenResponsesResult'
            text/event-stream:
              example:
                data:
                  content_index: 0
                  delta: Hello
                  item_id: item-1
                  logprobs: []
                  output_index: 0
                  sequence_number: 4
                  type: response.output_text.delta
              schema:
                $ref: '#/components/schemas/ResponsesStreamingResponse'
              x-speakeasy-sse-sentinel: '[DONE]'
          description: Successful response
        '400':
          content:
            application/json:
              example:
                error:
                  code: 400
                  message: Invalid request parameters
              schema:
                $ref: '#/components/schemas/BadRequestResponse'
          description: Bad Request - Invalid request parameters or malformed input
        '401':
          content:
            application/json:
              example:
                error:
                  code: 401
                  message: Missing Authentication header
              schema:
                $ref: '#/components/schemas/UnauthorizedResponse'
          description: Unauthorized - Authentication required or invalid credentials
        '402':
          content:
            application/json:
              example:
                error:
                  code: 402
                  message: >-
                    Insufficient credits. Add more using
                    https://openrouter.ai/credits
              schema:
                $ref: '#/components/schemas/PaymentRequiredResponse'
          description: Payment Required - Insufficient credits or quota to complete request
        '403':
          content:
            application/json:
              examples:
                guardrail-blocked:
                  summary: Guardrail blocked the request
                  value:
                    error:
                      code: 403
                      message: 'Request blocked: prompt injection patterns detected'
                      metadata:
                        patterns:
                          - ignore all previous instructions
                    openrouter_metadata:
                      attempt: 1
                      endpoints:
                        available:
                          - model: openai/gpt-4o
                            provider: OpenAI
                            selected: false
                        total: 1
                      is_byok: false
                      pipeline:
                        - data:
                            action: blocked
                            detected: true
                            engines:
                              - regex
                            patterns:
                              - ignore all previous instructions
                          guardrail_id: grd_abc123
                          guardrail_scope: api-key
                          name: regex_pi_detection
                          summary: >-
                            Blocked: prompt injection detected (1 pattern
                            matched)
                          type: guardrail
                      region: iad
                      requested: openai/gpt-4o
                      strategy: direct
                      summary: available=1
                insufficient-permissions:
                  summary: Insufficient permissions
                  value:
                    error:
                      code: 403
                      message: Only management keys can perform this operation
              schema:
                $ref: '#/components/schemas/ForbiddenResponse'
          description: >-
            Forbidden - Authentication successful but insufficient permissions,
            or a guardrail blocked the request. When guardrails block and the
            `X-OpenRouter-Metadata: enabled` header is present, the response
            includes `openrouter_metadata` with full routing context and a
            `pipeline` array containing guardrail stage details.
        '404':
          content:
            application/json:
              example:
                error:
                  code: 404
                  message: Resource not found
              schema:
                $ref: '#/components/schemas/NotFoundResponse'
          description: Not Found - Resource does not exist
        '408':
          content:
            application/json:
              example:
                error:
                  code: 408
                  message: Operation timed out. Please try again later.
              schema:
                $ref: '#/components/schemas/RequestTimeoutResponse'
          description: Request Timeout - Operation exceeded time limit
        '413':
          content:
            application/json:
              example:
                error:
                  code: 413
                  message: Request payload too large
              schema:
                $ref: '#/components/schemas/PayloadTooLargeResponse'
          description: Payload Too Large - Request payload exceeds size limits
        '422':
          content:
            application/json:
              example:
                error:
                  code: 422
                  message: Invalid argument
              schema:
                $ref: '#/components/schemas/UnprocessableEntityResponse'
          description: Unprocessable Entity - Semantic validation failure
        '429':
          content:
            application/json:
              example:
                error:
                  code: 429
                  message: Rate limit exceeded
              schema:
                $ref: '#/components/schemas/TooManyRequestsResponse'
          description: Too Many Requests - Rate limit exceeded
        '500':
          content:
            application/json:
              example:
                error:
                  code: 500
                  message: Internal Server Error
              schema:
                $ref: '#/components/schemas/InternalServerResponse'
          description: Internal Server Error - Unexpected server error
        '502':
          content:
            application/json:
              example:
                error:
                  code: 502
                  message: Provider returned error
              schema:
                $ref: '#/components/schemas/BadGatewayResponse'
          description: Bad Gateway - Provider/upstream API failure
        '503':
          content:
            application/json:
              example:
                error:
                  code: 503
                  message: Service temporarily unavailable
              schema:
                $ref: '#/components/schemas/ServiceUnavailableResponse'
          description: Service Unavailable - Service temporarily unavailable
        '524':
          content:
            application/json:
              example:
                error:
                  code: 524
                  message: Request timed out. Please try again later.
              schema:
                $ref: '#/components/schemas/EdgeNetworkTimeoutResponse'
          description: Infrastructure Timeout - Provider request timed out at edge network
        '529':
          content:
            application/json:
              example:
                error:
                  code: 529
                  message: Provider returned error
              schema:
                $ref: '#/components/schemas/ProviderOverloadedResponse'
          description: Provider Overloaded - Provider is temporarily overloaded
components:
  schemas:
    MetadataLevel:
      description: >-
        Opt-in level for surfacing routing metadata on the response under
        `openrouter_metadata`.
      enum:
        - disabled
        - enabled
      example: enabled
      type: string
    ResponsesRequest:
      description: Request schema for Responses endpoint
      example:
        input:
          - content: Hello, how are you?
            role: user
            type: message
        model: anthropic/claude-4.5-sonnet-20250929
        temperature: 0.7
        tools:
          - description: Get the current weather in a given location
            name: get_current_weather
            parameters:
              properties:
                location:
                  type: string
              type: object
            type: function
        top_p: 0.9
      properties:
        background:
          type:
            - boolean
            - 'null'
        cache_control:
          $ref: '#/components/schemas/AnthropicCacheControlDirective'
        debug:
          $ref: '#/components/schemas/ChatDebugOptions'
        frequency_penalty:
          format: double
          type:
            - number
            - 'null'
        image_config:
          $ref: '#/components/schemas/ImageConfig'
        include:
          items:
            $ref: '#/components/schemas/ResponseIncludesEnum'
          type:
            - array
            - 'null'
        input:
          $ref: '#/components/schemas/Inputs'
        instructions:
          type:
            - string
            - 'null'
        max_output_tokens:
          type:
            - integer
            - 'null'
        max_tool_calls:
          description: >-
            Maximum number of server-tool (e.g. `openrouter:web_search`) agent
            steps the model may take during a request. Defaults to 30, which is
            also the maximum. Ignored when `stop_server_tools_when` is set.
          example: 30
          type:
            - integer
            - 'null'
        metadata:
          $ref: '#/components/schemas/RequestMetadata'
        modalities:
          description: >-
            Output modalities for the response. Supported values are "text" and
            "image".
          example:
            - text
            - image
          items:
            $ref: '#/components/schemas/OutputModalityEnum'
          type: array
        model:
          type: string
        models:
          items:
            type: string
          type: array
        parallel_tool_calls:
          type:
            - boolean
            - 'null'
        plugins:
          description: >-
            Plugins you want to enable for this request, including their
            settings.
          items:
            discriminator:
              mapping:
                auto-beta-router:
                  $ref: '#/components/schemas/AutoBetaRouterPlugin'
                auto-router:
                  $ref: '#/components/schemas/AutoRouterPlugin'
                context-compression:
                  $ref: '#/components/schemas/ContextCompressionPlugin'
                file-parser:
                  $ref: '#/components/schemas/FileParserPlugin'
                fusion:
                  $ref: '#/components/schemas/FusionPlugin'
                moderation:
                  $ref: '#/components/schemas/ModerationPlugin'
                pareto-router:
                  $ref: '#/components/schemas/ParetoRouterPlugin'
                response-healing:
                  $ref: '#/components/schemas/ResponseHealingPlugin'
                web:
                  $ref: '#/components/schemas/WebSearchPlugin'
                web-fetch:
                  $ref: '#/components/schemas/WebFetchPlugin'
              propertyName: id
            oneOf:
              - $ref: '#/components/schemas/AutoRouterPlugin'
              - $ref: '#/components/schemas/AutoBetaRouterPlugin'
              - $ref: '#/components/schemas/ModerationPlugin'
              - $ref: '#/components/schemas/WebSearchPlugin'
              - $ref: '#/components/schemas/WebFetchPlugin'
              - $ref: '#/components/schemas/FileParserPlugin'
              - $ref: '#/components/schemas/ResponseHealingPlugin'
              - $ref: '#/components/schemas/ContextCompressionPlugin'
              - $ref: '#/components/schemas/ParetoRouterPlugin'
              - $ref: '#/components/schemas/FusionPlugin'
          type: array
        presence_penalty:
          format: double
          type:
            - number
            - 'null'
        previous_response_id:
          description: >-
            Not supported. The Responses API is stateless: no responses are
            stored, so a previous response cannot be referenced. Requests with a
            non-null value are rejected with a 400 error. Send the full
            conversation history in `input` instead.
        prompt:
          $ref: '#/components/schemas/StoredPromptTemplate'
        prompt_cache_key:
          type:
            - string
            - 'null'
        prompt_cache_options:
          $ref: '#/components/schemas/PromptCacheOptions'
        provider:
          $ref: '#/components/schemas/ProviderPreferences'
        reasoning:
          $ref: '#/components/schemas/ReasoningConfig'
        route:
          $ref: '#/components/schemas/DeprecatedRoute'
        safety_identifier:
          description: >-
            Recommended per-end-user identifier for abuse isolation. Use a
            stable ID, hash, or pseudonym. When a provider requires a user
            identity, OpenRouter folds it into the hashed identity sent upstream
            and never forwards it raw. If omitted, requests use an account-level
            identity, so provider policy blocks can affect the whole account.
          example: user-123
          type:
            - string
            - 'null'
        service_tier:
          default: auto
          description: >-
            The service tier to use for processing this request. `fast` is
            accepted as an alias for `priority`.
          enum:
            - auto
            - default
            - fast
            - flex
            - priority
            - scale
            - null
          type:
            - string
            - 'null'
        session_id:
          description: >-
            A unique identifier for grouping related requests (e.g., a
            conversation or agent workflow). When provided, OpenRouter uses it
            as the sticky routing key, routing all requests in the session to
            the same provider to maximize prompt cache hits. Also used for
            observability grouping. If provided in both the request body and the
            x-session-id header, the body value takes precedence. Maximum of 256
            characters.
          maxLength: 256
          type: string
        stop_server_tools_when:
          $ref: '#/components/schemas/StopServerToolsWhen'
        store:
          const: false
          default: false
          type: boolean
        stream:
          default: false
          type: boolean
        temperature:
          format: double
          type:
            - number
            - 'null'
        text:
          $ref: '#/components/schemas/TextExtendedConfig'
        tool_choice:
          $ref: '#/components/schemas/OpenAIResponsesToolChoice'
        tools:
          items:
            anyOf:
              - allOf:
                  - $ref: '#/components/schemas/FunctionTool'
                  - properties: {}
                    type: object
                description: Function tool definition
                example:
                  description: Get the current weather in a location
                  name: get_weather
                  parameters:
                    properties:
                      location:
                        description: The city and state
                        type: string
                      unit:
                        enum:
                          - celsius
                          - fahrenheit
                        type: string
                    required:
                      - location
                    type: object
                  type: function
              - $ref: '#/components/schemas/Preview_WebSearchServerTool'
              - $ref: '#/components/schemas/Preview_20250311_WebSearchServerTool'
              - $ref: '#/components/schemas/Legacy_WebSearchServerTool'
              - $ref: '#/components/schemas/WebSearchServerTool'
              - $ref: '#/components/schemas/FileSearchServerTool'
              - $ref: '#/components/schemas/ComputerUseServerTool'
              - $ref: '#/components/schemas/CodeInterpreterServerTool'
              - $ref: '#/components/schemas/McpServerTool'
              - $ref: '#/components/schemas/ImageGenerationServerTool'
              - $ref: '#/components/schemas/CodexLocalShellTool'
              - $ref: '#/components/schemas/ShellServerTool'
              - $ref: '#/components/schemas/ApplyPatchServerTool'
              - $ref: '#/components/schemas/CustomTool'
              - $ref: '#/components/schemas/NamespaceTool'
              - $ref: '#/components/schemas/AdvisorServerTool_OpenRouter'
              - $ref: '#/components/schemas/SubagentServerTool_OpenRouter'
              - $ref: '#/components/schemas/DatetimeServerTool'
              - $ref: '#/components/schemas/FilesServerTool'
              - $ref: '#/components/schemas/FusionServerTool_OpenRouter'
              - $ref: '#/components/schemas/ImageGenerationServerTool_OpenRouter'
              - $ref: '#/components/schemas/SearchModelsServerTool_OpenRouter'
              - $ref: '#/components/schemas/WebFetchServerTool'
              - $ref: '#/components/schemas/WebSearchServerTool_OpenRouter'
              - $ref: '#/components/schemas/ApplyPatchServerTool_OpenRouter'
              - $ref: '#/components/schemas/BashServerTool'
              - $ref: '#/components/schemas/ShellServerTool_OpenRouter'
          type: array
        top_k:
          type: integer
        top_logprobs:
          type:
            - integer
            - 'null'
        top_p:
          format: double
          type:
            - number
            - 'null'
        trace:
          $ref: '#/components/schemas/TraceConfig'
        truncation:
          $ref: '#/components/schemas/OpenAIResponsesTruncation'
        user:
          description: >-
            A unique identifier representing your end-user, which helps
            distinguish between different users of your app. This allows your
            app to identify specific users in case of abuse reports, preventing
            your entire app from being affected by the actions of individual
            users. Maximum of 256 characters.
          maxLength: 256
          type: string
      type: object
    OpenResponsesResult:
      allOf:
        - $ref: '#/components/schemas/BaseResponsesResult'
        - properties:
            error_type:
              $ref: '#/components/schemas/ApiErrorType'
            openrouter_metadata:
              $ref: '#/components/schemas/OpenRouterMetadata'
            output:
              items:
                $ref: '#/components/schemas/OutputItems'
              type: array
            service_tier:
              type:
                - string
                - 'null'
            text:
              $ref: '#/components/schemas/TextExtendedConfig'
            usage:
              $ref: '#/components/schemas/Usage'
          type: object
      description: Complete non-streaming response from the Responses API
      example:
        completed_at: 1704067210
        created_at: 1704067200
        error: null
        frequency_penalty: null
        id: resp-abc123
        incomplete_details: null
        instructions: null
        max_output_tokens: null
        metadata: null
        model: gpt-4
        object: response
        output:
          - content:
              - annotations: []
                text: Hello! How can I help you today?
                type: output_text
            id: msg-abc123
            role: assistant
            status: completed
            type: message
        parallel_tool_calls: true
        presence_penalty: null
        status: completed
        temperature: null
        tool_choice: auto
        tools: []
        top_p: null
        usage:
          input_tokens: 10
          input_tokens_details:
            cached_tokens: 0
          output_tokens: 25
          output_tokens_details:
            reasoning_tokens: 0
          total_tokens: 35
    ResponsesStreamingResponse:
      example:
        data:
          content_index: 0
          delta: Hello
          item_id: item-1
          logprobs: []
          output_index: 0
          sequence_number: 4
          type: response.output_text.delta
      properties:
        data:
          $ref: '#/components/schemas/StreamEvents'
      required:
        - data
      type: object
    BadRequestResponse:
      description: Bad Request - Invalid request parameters or malformed input
      example:
        error:
          code: 400
          message: Invalid request parameters
      properties:
        error:
          $ref: '#/components/schemas/BadRequestResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    UnauthorizedResponse:
      description: Unauthorized - Authentication required or invalid credentials
      example:
        error:
          code: 401
          message: Missing Authentication header
      properties:
        error:
          $ref: '#/components/schemas/UnauthorizedResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    PaymentRequiredResponse:
      description: Payment Required - Insufficient credits or quota to complete request
      example:
        error:
          code: 402
          message: Insufficient credits. Add more using https://openrouter.ai/credits
      properties:
        error:
          $ref: '#/components/schemas/PaymentRequiredResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    ForbiddenResponse:
      description: Forbidden - Authentication successful but insufficient permissions
      example:
        error:
          code: 403
          message: Only management keys can perform this operation
      properties:
        error:
          $ref: '#/components/schemas/ForbiddenResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    NotFoundResponse:
      description: Not Found - Resource does not exist
      example:
        error:
          code: 404
          message: Resource not found
      properties:
        error:
          $ref: '#/components/schemas/NotFoundResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    RequestTimeoutResponse:
      description: Request Timeout - Operation exceeded time limit
      example:
        error:
          code: 408
          message: Operation timed out. Please try again later.
      properties:
        error:
          $ref: '#/components/schemas/RequestTimeoutResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    PayloadTooLargeResponse:
      description: Payload Too Large - Request payload exceeds size limits
      example:
        error:
          code: 413
          message: Request payload too large
      properties:
        error:
          $ref: '#/components/schemas/PayloadTooLargeResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    UnprocessableEntityResponse:
      description: Unprocessable Entity - Semantic validation failure
      example:
        error:
          code: 422
          message: Invalid argument
      properties:
        error:
          $ref: '#/components/schemas/UnprocessableEntityResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    TooManyRequestsResponse:
      description: Too Many Requests - Rate limit exceeded
      example:
        error:
          code: 429
          message: Rate limit exceeded
      properties:
        error:
          $ref: '#/components/schemas/TooManyRequestsResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    InternalServerResponse:
      description: Internal Server Error - Unexpected server error
      example:
        error:
          code: 500
          message: Internal Server Error
      properties:
        error:
          $ref: '#/components/schemas/InternalServerResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    BadGatewayResponse:
      description: Bad Gateway - Provider/upstream API failure
      example:
        error:
          code: 502
          message: Provider returned error
      properties:
        error:
          $ref: '#/components/schemas/BadGatewayResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    ServiceUnavailableResponse:
      description: Service Unavailable - Service temporarily unavailable
      example:
        error:
          code: 503
          message: Service temporarily unavailable
      properties:
        error:
          $ref: '#/components/schemas/ServiceUnavailableResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    EdgeNetworkTimeoutResponse:
      description: Infrastructure Timeout - Provider request timed out at edge network
      example:
        error:
          code: 524
          message: Request timed out. Please try again later.
      properties:
        error:
          $ref: '#/components/schemas/EdgeNetworkTimeoutResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    ProviderOverloadedResponse:
      description: Provider Overloaded - Provider is temporarily overloaded
      example:
        error:
          code: 529
          message: Provider returned error
      properties:
        error:
          $ref: '#/components/schemas/ProviderOverloadedResponseErrorData'
        openrouter_metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
        user_id:
          type:
            - string
            - 'null'
      required:
        - error
      type: object
    AnthropicCacheControlDirective:
      description: >-
        Enable automatic prompt caching. When set at the top level, the system
        automatically applies cache breakpoints to the last cacheable block in
        the request. When set on an individual content block, it marks an
        explicit cache breakpoint; block-level markers also work on OpenAI
        models that support explicit prompt caching — OpenRouter converts them
        to the provider's native format.
      example:
        type: ephemeral
      properties:
        ttl:
          $ref: '#/components/schemas/AnthropicCacheControlTtl'
        type:
          enum:
            - ephemeral
          type: string
      required:
        - type
      type: object
    ChatDebugOptions:
      description: Debug options for inspecting request transformations (streaming only)
      example:
        echo_upstream_body: true
      properties:
        echo_upstream_body:
          description: >-
            If true, includes the transformed upstream request body in a debug
            chunk at the start of the stream. Only works with streaming mode.
          example: true
          type: boolean
      type: object
    ImageConfig:
      additionalProperties:
        anyOf:
          - type: string
          - format: double
            type: number
          - items: {}
            type: array
      description: >-
        Provider-specific image configuration options. Keys and values vary by
        model/provider. See
        https://openrouter.ai/docs/guides/overview/multimodal/image-generation
        for more details.
      example:
        aspect_ratio: '16:9'
        quality: high
      type: object
    ResponseIncludesEnum:
      enum:
        - file_search_call.results
        - message.input_image.image_url
        - computer_call_output.output.image_url
        - reasoning.encrypted_content
        - code_interpreter_call.outputs
      example: file_search_call.results
      type: string
    Inputs:
      anyOf:
        - type: string
        - items:
            anyOf:
              - $ref: '#/components/schemas/ReasoningItem'
              - $ref: '#/components/schemas/EasyInputMessage'
              - $ref: '#/components/schemas/InputMessageItem'
              - $ref: '#/components/schemas/FunctionCallItem'
              - $ref: '#/components/schemas/FunctionCallOutputItem'
              - $ref: '#/components/schemas/ApplyPatchCallItem'
              - $ref: '#/components/schemas/ApplyPatchCallOutputItem'
              - allOf:
                  - $ref: '#/components/schemas/OutputMessageItem'
                  - properties:
                      content:
                        anyOf:
                          - items:
                              anyOf:
                                - $ref: '#/components/schemas/ResponseOutputText'
                                - $ref: >-
                                    #/components/schemas/OpenAIResponsesRefusalContent
                            type: array
                          - type: string
                          - type: 'null'
                      type:
                        default: message
                        enum:
                          - message
                        type: string
                    type: object
                description: An output message item
                example:
                  content:
                    - annotations: []
                      text: Hello! How can I help you?
                      type: output_text
                  id: msg-123
                  role: assistant
                  status: completed
                  type: message
              - allOf:
                  - $ref: '#/components/schemas/OutputReasoningItem'
                  - properties:
                      summary:
                        items:
                          $ref: '#/components/schemas/ReasoningSummaryText'
                        type:
                          - array
                          - 'null'
                    type: object
                description: An output item containing reasoning
                example:
                  content:
                    - text: First, we analyze the problem...
                      type: reasoning_text
                  format: anthropic-claude-v1
                  id: reasoning-123
                  signature: EvcBCkgIChABGAIqQKkSDbRuVEQUk9qN1odC098l9SEj...
                  status: completed
                  summary:
                    - text: Analyzed the problem and found the optimal solution.
                      type: summary_text
                  type: reasoning
              - $ref: '#/components/schemas/OutputFunctionCallItem'
              - $ref: '#/components/schemas/OutputCustomToolCallItem'
              - $ref: '#/components/schemas/OutputWebSearchCallItem'
              - $ref: '#/components/schemas/OutputFileSearchCallItem'
              - $ref: '#/components/schemas/OutputImageGenerationCallItem'
              - $ref: '#/components/schemas/OutputCodeInterpreterCallItem'
              - $ref: '#/components/schemas/OutputComputerCallItem'
              - $ref: '#/components/schemas/OutputDatetimeItem'
              - $ref: '#/components/schemas/OutputWebSearchServerToolItem'
              - $ref: '#/components/schemas/OutputCodeInterpreterServerToolItem'
              - $ref: '#/components/schemas/OutputFileSearchServerToolItem'
              - $ref: '#/components/schemas/OutputImageGenerationServerToolItem'
              - $ref: '#/components/schemas/OutputBrowserUseServerToolItem'
              - $ref: '#/components/schemas/OutputBashServerToolItem'
              - $ref: '#/components/schemas/OutputTextEditorServerToolItem'
              - $ref: '#/components/schemas/OutputApplyPatchServerToolItem'
              - $ref: '#/components/schemas/OutputWebFetchServerToolItem'
              - $ref: '#/components/schemas/OutputToolSearchServerToolItem'
              - $ref: '#/components/schemas/OutputMemoryServerToolItem'
              - $ref: '#/components/schemas/OutputMcpServerToolItem'
              - $ref: '#/components/schemas/OutputSearchModelsServerToolItem'
              - $ref: '#/components/schemas/OutputFusionServerToolItem'
              - $ref: '#/components/schemas/OutputAdvisorServerToolItem'
              - $ref: '#/components/schemas/OutputSubagentServerToolItem'
              - $ref: '#/components/schemas/OutputFilesServerToolItem'
              - $ref: '#/components/schemas/LocalShellCallItem'
              - $ref: '#/components/schemas/LocalShellCallOutputItem'
              - $ref: '#/components/schemas/ShellCallItem'
              - $ref: '#/components/schemas/ShellCallOutputItem'
              - $ref: '#/components/schemas/McpListToolsItem'
              - $ref: '#/components/schemas/McpApprovalRequestItem'
              - $ref: '#/components/schemas/McpApprovalResponseItem'
              - $ref: '#/components/schemas/McpCallItem'
              - $ref: '#/components/schemas/CustomToolCallItem'
              - $ref: '#/components/schemas/CustomToolCallOutputItem'
              - $ref: '#/components/schemas/CompactionItem'
              - $ref: '#/components/schemas/ContextCompactionItem'
              - $ref: '#/components/schemas/ItemReferenceItem'
              - $ref: '#/components/schemas/AdditionalToolsItem'
              - $ref: '#/components/schemas/AgentMessageItem'
          type: array
      description: Input for a response request - can be a string or array of items
      example:
        - content: What is the weather today?
          role: user
    RequestMetadata:
      additionalProperties:
        maxLength: 512
        type: string
      description: >-
        Metadata key-value pairs for the request. Keys must be ≤64 characters
        and cannot contain brackets. Values must be ≤512 characters. Maximum 16
        pairs allowed.
      example:
        session_id: abc-def-ghi
        user_id: '123'
      type:
        - object
        - 'null'
    OutputModalityEnum:
      enum:
        - text
        - image
      example: text
      type: string
    AutoBetaRouterPlugin:
      example:
        allowed_models:
          - anthropic/*
          - openai/*
        cost_tier: low
        enabled: true
        excluded_models:
          - openai/gpt-4o
        id: auto-beta-router
      properties:
        allowed_models:
          description: >-
            List of model patterns to filter which models the auto-beta-router
            can route between. Supports wildcards (e.g., "anthropic/*" matches
            all Anthropic models). Up to 1024 patterns, each at most 1024
            characters, with 65536 total characters across all patterns. When
            not specified, every model ranked for the classified task type is a
            candidate, falling back to a default model set when rankings are
            unavailable.
          example:
            - anthropic/*
            - openai/gpt-4o
            - google/*
          items:
            maxLength: 1024
            type: string
          maxItems: 1024
          type: array
        cost_quality_tradeoff:
          deprecated: true
          description: >-
            Deprecated: Use cost_tier instead. Balances routing between cost and
            quality on a 0-10 scale. The auto-beta-router ranks models for the
            classified task type by community spend share, then filters
            candidates by their average cost per generation for that task.
            Higher values favor cheaper models: 10 keeps only models around the
            cheapest 10th percentile, while 0 permits models up to the 90th
            percentile for cost. Defaults to 9 when no cost setting is provided.
            It remains supported and retains ceiling behavior, but cost_tier
            takes precedence when both are provided.
          example: 9
          maximum: 10
          minimum: 0
          type: integer
        cost_tier:
          description: >-
            Named cost/quality setting. For auto-beta-router, tiers select
            cost-percentile bands: low = [0, 20), medium = [20, 40), high = [40,
            60), xhigh = [60, 80), and max = [80, 100]. Takes precedence over
            the deprecated numeric cost_quality_tradeoff when both are provided.
          enum:
            - low
            - medium
            - high
            - xhigh
            - max
          example: low
          type: string
        enabled:
          description: >-
            Set to false to disable the auto-beta-router plugin for this
            request. Defaults to true.
          type: boolean
        excluded_models:
          description: >-
            List of model patterns to exclude from auto-beta-router selection.
            Supports wildcards (e.g., "meta-llama/*" excludes all Llama models).
            Up to 1024 patterns, each at most 1024 characters, with 65536 total
            characters across all patterns. Applied after allowed_models, so an
            excluded pattern always wins over an allowed one.
          example:
            - openai/gpt-4o
            - meta-llama/*
          items:
            maxLength: 1024
            type: string
          maxItems: 1024
          type: array
        id:
          enum:
            - auto-beta-router
          type: string
      required:
        - id
      type: object
    AutoRouterPlugin:
      example:
        allowed_models:
          - anthropic/*
          - openai/*
        cost_tier: low
        enabled: true
        excluded_models:
          - openai/gpt-4o
        id: auto-router
        pin_model: false
      properties:
        allowed_models:
          description: >-
            List of model patterns to filter which models the auto-router can
            route between. Supports wildcards (e.g., "anthropic/*" matches all
            Anthropic models). Up to 1024 patterns, each at most 1024
            characters, with 65536 total characters across all patterns. When
            not specified, every model ranked for the classified task type is a
            candidate, falling back to a default model set when rankings are
            unavailable.
          example:
            - anthropic/*
            - openai/gpt-4o
            - google/*
          items:
            maxLength: 1024
            type: string
          maxItems: 1024
          type: array
        cost_quality_tradeoff:
          deprecated: true
          description: >-
            Deprecated: Use cost_tier instead. Balances routing between cost and
            quality on a 0-10 scale. The auto-router ranks models for the
            classified task type by community spend share, then filters
            candidates by their average cost per generation for that task.
            Higher values favor cheaper models: 10 keeps only models around the
            cheapest 10th percentile, while 0 permits models up to the 90th
            percentile for cost. Defaults to 9 when no cost setting is provided.
            It remains supported and retains ceiling behavior, but cost_tier
            takes precedence when both are provided.
          example: 9
          maximum: 10
          minimum: 0
          type: integer
        cost_tier:
          description: >-
            Named cost/quality setting. Tiers select cost-percentile bands: low
            = [0, 20), medium = [20, 40), high = [40, 60), xhigh = [60, 80), and
            max = [80, 100]. Takes precedence over the deprecated numeric
            cost_quality_tradeoff when both are provided.
          enum:
            - low
            - medium
            - high
            - xhigh
            - max
          example: low
          type: string
        enabled:
          description: >-
            Set to false to disable the auto-router plugin for this request.
            Defaults to true.
          type: boolean
        excluded_models:
          description: >-
            List of model patterns to exclude from auto-router selection.
            Supports wildcards (e.g., "meta-llama/*" excludes all Llama models).
            Up to 1024 patterns, each at most 1024 characters, with 65536 total
            characters across all patterns. Applied after allowed_models, so an
            excluded pattern always wins over an allowed one.
          example:
            - openai/gpt-4o
            - meta-llama/*
          items:
            maxLength: 1024
            type: string
          maxItems: 1024
          type: array
        id:
          enum:
            - auto-router
          type: string
        pin_model:
          description: >-
            When true, reuses the model from the most recent assistant message's
            `model` attribute for subsequent turns. Defaults to false.
          example: false
          type: boolean
      required:
        - id
      type: object
    ContextCompressionPlugin:
      example:
        enabled: true
        engine: middle-out
        id: context-compression
      properties:
        enabled:
          description: >-
            Set to false to disable the context-compression plugin for this
            request. Defaults to true.
          type: boolean
        engine:
          $ref: '#/components/schemas/ContextCompressionEngine'
        id:
          enum:
            - context-compression
          type: string
      required:
        - id
      type: object
    FileParserPlugin:
      example:
        enabled: true
        id: file-parser
        pdf:
          engine: cloudflare-ai
      properties:
        enabled:
          description: >-
            Set to false to disable the file-parser plugin for this request.
            Defaults to true.
          type: boolean
        id:
          enum:
            - file-parser
          type: string
        pdf:
          $ref: '#/components/schemas/PDFParserOptions'
      required:
        - id
      type: object
    FusionPlugin:
      example:
        analysis_models:
          - ~anthropic/claude-opus-latest
          - ~openai/gpt-latest
          - ~google/gemini-pro-latest
        enabled: true
        id: fusion
        model: ~anthropic/claude-opus-latest
      properties:
        analysis_models:
          description: >-
            Slugs of models to run in parallel as the "expert panel" the analyst
            analyzes. Each model receives the same user prompt with web_search +
            web_fetch enabled. Capped at 8 models to bound cost amplification.
            When omitted, defaults to the Quality preset from the /labs/fusion
            UI (~anthropic/claude-opus-latest, ~openai/gpt-latest,
            ~google/gemini-pro-latest).
          example:
            - ~anthropic/claude-opus-latest
            - ~openai/gpt-latest
            - ~google/gemini-pro-latest
          items:
            type: string
          maxItems: 8
          minItems: 1
          type: array
        enabled:
          description: >-
            Set to false to disable the fusion plugin for this request. Defaults
            to true.
          type: boolean
        id:
          enum:
            - fusion
          type: string
        max_tool_calls:
          description: >-
            Maximum number of tool-calling steps each panelist (analysis model)
            and the analyst model may take during their agentic web-research
            loop. Models with web_search/web_fetch enabled iterate until they
            produce a text response or hit this ceiling. Defaults to 4. Capped
            at 16.
          example: 12
          maximum: 16
          minimum: 1
          type: integer
        model:
          description: >-
            Slug of the model that performs both the analyst step (with
            web_search + web_fetch) and the final synthesis. When omitted,
            defaults to the first model in the Quality preset.
          example: ~anthropic/claude-opus-latest
          type: string
        preset:
          description: >-
            A curated OpenRouter fusion preset (slugs follow `<task>-<tier>`,
            e.g. `general-high`). Expands server-side into the preset's
            analysis_models panel and analyst model, so callers never name
            individual models. Explicitly provided `analysis_models` / `model`
            take precedence.
          enum:
            - general-high
            - general-budget
            - general-fast
          example: general-high
          type: string
        tools:
          description: >-
            Server tools available to panelist and analyst inner calls. Each
            entry uses the same `{ type, parameters? }` shorthand as the outer
            Chat Completions request. When omitted, defaults to `[{ type:
            "openrouter:web_search" }, { type: "openrouter:web_fetch" }]`. Pass
            an empty array to disable tools entirely (panelists answer from
            parametric knowledge only).
          example:
            - parameters:
                excluded_domains:
                  - example.com
              type: openrouter:web_search
            - type: openrouter:web_fetch
          items:
            properties:
              parameters:
                additionalProperties:
                  anyOf:
                    - type: string
                    - format: double
                      type: number
                    - type: boolean
                    - type: 'null'
                    - items:
                        anyOf:
                          - type: string
                          - format: double
                            type: number
                          - type: boolean
                          - type: 'null'
                          - type: 'null'
                      type: array
                    - additionalProperties:
                        anyOf:
                          - type: string
                          - format: double
                            type: number
                          - type: boolean
                          - type: 'null'
                          - type: 'null'
                      type: object
                    - type: 'null'
                description: >-
                  Optional configuration forwarded as the tool's `parameters`
                  object.
                type: object
              type:
                description: >-
                  Server tool type identifier (e.g. "openrouter:web_search",
                  "openrouter:web_fetch").
                type: string
            required:
              - type
            type: object
          maxItems: 8
          type: array
      required:
        - id
      type: object
    ModerationPlugin:
      example:
        id: moderation
      properties:
        id:
          enum:
            - moderation
          type: string
      required:
        - id
      type: object
    ParetoRouterPlugin:
      example:
        enabled: true
        id: pareto-router
        max_price: 5
        price_source: prompt
      properties:
        enabled:
          description: >-
            Set to false to disable the pareto-router plugin for this request.
            Defaults to true.
          type: boolean
        id:
          enum:
            - pareto-router
          type: string
        max_price:
          description: >-
            Maximum input price in USD per million tokens. When set,
            quality-tier selection (min_coding_score) is bypassed: the router
            computes the Pareto frontier over the top coding models and routes
            to the best-scoring frontier model priced at or below this cap,
            falling back through cheaper frontier models, then non-frontier
            models. Enforced against the price source given by price_source.
            Returns 404 when no candidate satisfies the cap.
          example: 5
          format: double
          minimum: 0
          type: number
        min_coding_score:
          description: >-
            Minimum coding quality score between 0 and 1. Maps to internal
            quality tiers: >= 0.66 → high (top coding models), >= 0.33 → medium
            (strong modern flagships), < 0.33 → low (capable coders above the
            median). Omit to default to the highest tier (equivalent to >=
            0.66). Not used when max_price is set (price-based selection takes
            over).
          example: 0.8
          format: double
          maximum: 1
          minimum: 0
          type: number
        price_source:
          description: >-
            Price source for the Pareto frontier cost axis and for enforcing
            max_price. "prompt" uses catalog list price
            (endpoint.pricing.prompt). "weighted_avg" uses traffic-weighted
            effective input price from ClickHouse, falling back to prompt price
            for models without traffic data. Defaults to "prompt".
          enum:
            - prompt
            - weighted_avg
          type: string
      required:
        - id
      type: object
    ResponseHealingPlugin:
      example:
        enabled: true
        id: response-healing
      properties:
        enabled:
          description: >-
            Set to false to disable the response-healing plugin for this
            request. Defaults to true.
          type: boolean
        id:
          enum:
            - response-healing
          type: string
      required:
        - id
      type: object
    WebSearchPlugin:
      example:
        enabled: true
        id: web
        max_results: 5
      properties:
        enabled:
          description: >-
            Set to false to disable the web-search plugin for this request.
            Defaults to true.
          type: boolean
        engine:
          $ref: '#/components/schemas/WebSearchEngine'
        exclude_domains:
          description: >-
            A list of domains to exclude from web search results. Supports
            wildcards (e.g. "*.substack.com") and path filtering (e.g.
            "openai.com/blog").
          example:
            - example.com
            - '*.substack.com'
            - openai.com/blog
          items:
            type: string
          type: array
        id:
          enum:
            - web
          type: string
        include_domains:
          description: >-
            A list of domains to restrict web search results to. Supports
            wildcards (e.g. "*.substack.com") and path filtering (e.g.
            "openai.com/blog").
          example:
            - example.com
            - '*.substack.com'
            - openai.com/blog
          items:
            type: string
          type: array
        max_results:
          type: integer
        max_uses:
          description: >-
            Maximum number of times the model can invoke web search in a single
            turn. Passed through to native providers that support it (e.g.
            Anthropic).
          type: integer
        mode:
          $ref: '#/components/schemas/WebSearchMode'
        search_prompt:
          type: string
        user_location:
          allOf:
            - $ref: '#/components/schemas/WebSearchUserLocation'
            - description: >-
                Approximate user location for location-biased search results.
                Passed through to native providers that support it (e.g.
                Anthropic).
              example:
                city: San Francisco
                country: US
                region: California
                timezone: America/Los_Angeles
                type: approximate
              required:
                - type
      required:
        - id
      type: object
    WebFetchPlugin:
      example:
        id: web-fetch
        max_uses: 10
      properties:
        allowed_domains:
          description: Only fetch from these domains.
          items:
            type: string
          type: array
        blocked_domains:
          description: Never fetch from these domains.
          items:
            type: string
          type: array
        id:
          enum:
            - web-fetch
          type: string
        max_content_tokens:
          description: >-
            Maximum content length in approximate tokens. Content exceeding this
            limit is truncated.
          type: integer
        max_uses:
          description: >-
            Maximum number of web fetches per request. Once exceeded, the tool
            returns an error.
          type: integer
      required:
        - id
      type: object
    StoredPromptTemplate:
      example:
        id: prompt-abc123
        variables:
          name: John
      properties:
        id:
          type: string
        variables:
          additionalProperties:
            anyOf:
              - type: string
              - $ref: '#/components/schemas/InputText'
              - $ref: '#/components/schemas/InputImage'
              - $ref: '#/components/schemas/InputFile'
          type:
            - object
            - 'null'
      required:
        - id
      type:
        - object
        - 'null'
    PromptCacheOptions:
      description: >-
        Request-level prompt-cache controls. `mode: "explicit"` disables
        OpenAI-managed breakpoints so only blocks marked with
        `prompt_cache_breakpoint` are cached. Only supported by OpenAI GPT-5.6
        and newer.
      example:
        mode: explicit
        ttl: 30m
      properties:
        mode:
          enum:
            - explicit
          type: string
        ttl:
          type:
            - string
            - 'null'
      required:
        - mode
      type:
        - object
        - 'null'
    ProviderPreferences:
      additionalProperties: false
      description: >-
        When multiple model providers are available, optionally indicate your
        routing preference.
      example:
        allow_fallbacks: true
      properties:
        allow_fallbacks:
          description: >
            Whether to allow backup providers to serve requests

            - true: (default) when the primary provider (or your custom
            providers in "order") is unavailable, use the next best provider.

            - false: use only the primary/custom provider, and return the
            upstream error if it's unavailable.
          type:
            - boolean
            - 'null'
        data_collection:
          description: >-
            Data collection setting. If no available model provider meets the
            requirement, your request will return an error.

            - allow: (default) allow providers which store user data
            non-transiently and may train on it


            - deny: use only providers which do not collect user data.
          enum:
            - deny
            - allow
            - null
          example: allow
          type:
            - string
            - 'null'
        enforce_distillable_text:
          description: >-
            Whether to restrict routing to only models that allow text
            distillation. When true, only models where the author has allowed
            distillation will be used.
          example: true
          type:
            - boolean
            - 'null'
        ignore:
          description: >-
            List of provider slugs to ignore. If provided, this list is merged
            with your account-wide ignored provider settings for this request.
          example:
            - openai
            - anthropic
          items:
            anyOf:
              - $ref: '#/components/schemas/ProviderName'
              - type: string
          type:
            - array
            - 'null'
        max_price:
          description: >-
            The object specifying the maximum price you want to pay for this
            request. USD price per million tokens, for prompt and completion.
          properties:
            audio:
              description: Maximum price in USD per audio unit
              type: string
            completion:
              description: Maximum price in USD per million completion tokens
              type: string
            image:
              description: Maximum price in USD per image
              type: string
            prompt:
              description: Maximum price in USD per million prompt tokens
              type: string
            request:
              description: Maximum price in USD per request
              type: string
          type: object
        only:
          description: >-
            List of provider slugs to allow. If provided, this list is merged
            with your account-wide allowed provider settings for this request.
          example:
            - openai
            - anthropic
          items:
            anyOf:
              - $ref: '#/components/schemas/ProviderName'
              - type: string
          type:
            - array
            - 'null'
        order:
          description: >-
            An ordered list of provider slugs. The router will attempt to use
            the first provider in the subset of this list that supports your
            requested model, and fall back to the next if it is unavailable. If
            no providers are available, the request will fail with an error
            message.
          example:
            - openai
            - anthropic
          items:
            anyOf:
              - $ref: '#/components/schemas/ProviderName'
              - type: string
          type:
            - array
            - 'null'
        preferred_max_latency:
          $ref: '#/components/schemas/PreferredMaxLatency'
        preferred_min_throughput:
          $ref: '#/components/schemas/PreferredMinThroughput'
        quantizations:
          description: A list of quantization levels to filter the provider by.
          items:
            $ref: '#/components/schemas/Quantization'
          type:
            - array
            - 'null'
        require_parameters:
          description: >-
            Whether to filter providers to only those that support the
            parameters you've provided. If this setting is omitted or set to
            false, then providers will receive only the parameters they support,
            and ignore the rest.
          type:
            - boolean
            - 'null'
        sort:
          anyOf:
            - $ref: '#/components/schemas/ProviderSort'
            - $ref: '#/components/schemas/ProviderSortConfig'
            - type: 'null'
          description: >-
            The sorting strategy to use for this request, if "order" is not
            specified. When set, no load balancing is performed.
          example: price
        zdr:
          description: >-
            Whether to restrict routing to only ZDR (Zero Data Retention)
            endpoints. When true, only endpoints that do not retain prompts will
            be used.
          example: true
          type:
            - boolean
            - 'null'
      type:
        - object
        - 'null'
    ReasoningConfig:
      anyOf:
        - allOf:
            - $ref: '#/components/schemas/BaseReasoningConfig'
            - properties:
                enabled:
                  type:
                    - boolean
                    - 'null'
                max_tokens:
                  type:
                    - integer
                    - 'null'
              type: object
        - type: 'null'
      description: Configuration for reasoning mode in the response
      example:
        enabled: true
        summary: auto
    DeprecatedRoute:
      deprecated: true
      description: >-
        **DEPRECATED** Use providers.sort.partition instead.
        Backwards-compatible alias for providers.sort.partition. Accepts legacy
        values: "fallback" (maps to "model"), "sort" (maps to "none").
      enum:
        - fallback
        - sort
        - null
      example: fallback
      type:
        - string
        - 'null'
      x-fern-ignore: true
      x-speakeasy-deprecation-message: Use providers.sort.partition instead.
      x-speakeasy-ignore: true
    StopServerToolsWhen:
      description: >-
        Stop conditions for the server-tool agent loop. Any condition firing
        halts the loop (OR logic). When set, this overrides `max_tool_calls`.
        When a condition fires while the model is still emitting tool calls, the
        pending tool calls are executed and one final turn is made with tool
        calls disabled so the response ends with a natural-language answer
        instead of an unfinished tool call.
      example:
        - step_count: 5
          type: step_count_is
        - max_cost_in_dollars: 0.5
          type: max_cost
      items:
        $ref: '#/components/schemas/StopServerToolsWhenCondition'
      minItems: 1
      type: array
    TextExtendedConfig:
      allOf:
        - $ref: '#/components/schemas/TextConfig'
        - properties:
            verbosity:
              enum:
                - low
                - medium
                - high
                - xhigh
                - max
                - null
              type:
                - string
                - 'null'
          type: object
      description: Text output configuration including format and verbosity
      example:
        format:
          type: text
    OpenAIResponsesToolChoice:
      anyOf:
        - enum:
            - auto
          type: string
        - enum:
            - none
          type: string
        - enum:
            - required
          type: string
        - properties:
            name:
              type: string
            type:
              enum:
                - function
              type: string
          required:
            - type
            - name
          type: object
        - properties:
            type:
              anyOf:
                - enum:
                    - web_search_preview_2025_03_11
                  type: string
                - enum:
                    - web_search_preview
                  type: string
          required:
            - type
          type: object
        - $ref: '#/components/schemas/ToolChoiceAllowed'
        - properties:
            type:
              enum:
                - apply_patch
              type: string
          required:
            - type
          type: object
        - properties:
            type:
              enum:
                - shell
              type: string
          required:
            - type
          type: object
      example: auto
    FunctionTool:
      description: Function tool definition
      example:
        description: Get the current weather in a location
        name: get_weather
        parameters:
          properties:
            location:
              description: The city and state
              type: string
            unit:
              enum:
                - celsius
                - fahrenheit
              type: string
          required:
            - location
          type: object
        type: function
      properties:
        description:
          type:
            - string
            - 'null'
        name:
          type: string
        parameters:
          additionalProperties: {}
          type:
            - object
            - 'null'
        strict:
          type:
            - boolean
            - 'null'
        type:
          enum:
            - function
          type: string
      required:
        - type
        - name
        - parameters
      type: object
    Preview_WebSearchServerTool:
      description: Web search preview tool configuration
      example:
        type: web_search_preview
      properties:
        engine:
          $ref: '#/components/schemas/WebSearchEngineEnum'
        filters:
          $ref: '#/components/schemas/WebSearchDomainFilter'
        max_results:
          description: >-
            Maximum number of search results to return per search call. Defaults
            to 5. Applies to Exa, Firecrawl, Parallel, and Perplexity engines;
            ignored with native provider search. Perplexity supports a maximum
            of 20; values above 20 are clamped.
          example: 5
          type: integer
        max_uses:
          description: >-
            Maximum number of web searches the model may perform in a single
            request. Once reached, further search calls return an error result
            instead of executing. Applies to the Exa, Firecrawl, Parallel, and
            Perplexity engines. With native provider search, forwarded only to
            Anthropic (as `max_uses`); other native search providers have no
            equivalent parameter and ignore it.
          example: 3
          type: integer
        mode:
          $ref: '#/components/schemas/WebSearchMode'
        search_context_size:
          $ref: '#/components/schemas/SearchContextSizeEnum'
        type:
          enum:
            - web_search_preview
          type: string
        user_location:
          $ref: '#/components/schemas/Preview_WebSearchUserLocation'
      required:
        - type
      type: object
    Preview_20250311_WebSearchServerTool:
      description: Web search preview tool configuration (2025-03-11 version)
      example:
        type: web_search_preview_2025_03_11
      properties:
        engine:
          $ref: '#/components/schemas/WebSearchEngineEnum'
        filters:
          $ref: '#/components/schemas/WebSearchDomainFilter'
        max_results:
          description: >-
            Maximum number of search results to return per search call. Defaults
            to 5. Applies to Exa, Firecrawl, Parallel, and Perplexity engines;
            ignored with native provider search. Perplexity supports a maximum
            of 20; values above 20 are clamped.
          example: 5
          type: integer
        max_uses:
          description: >-
            Maximum number of web searches the model may perform in a single
            request. Once reached, further search calls return an error result
            instead of executing. Applies to the Exa, Firecrawl, Parallel, and
            Perplexity engines. With native provider search, forwarded only to
            Anthropic (as `max_uses`); other native search providers have no
            equivalent parameter and ignore it.
          example: 3
          type: integer
        mode:
          $ref: '#/components/schemas/WebSearchMode'
        search_context_size:
          $ref: '#/components/schemas/SearchContextSizeEnum'
        type:
          enum:
            - web_search_preview_2025_03_11
          type: string
        user_location:
          $ref: '#/components/schemas/Preview_WebSearchUserLocation'
      required:
        - type
      type: object
    Legacy_WebSearchServerTool:
      description: Web search tool configuration
      example:
        engine: auto
        filters:
          allowed_domains:
            - example.com
        type: web_search
      properties:
        engine:
          $ref: '#/components/schemas/WebSearchEngineEnum'
        filters:
          $ref: '#/components/schemas/WebSearchDomainFilter'
        max_results:
          description: >-
            Maximum number of search results to return per search call. Defaults
            to 5. Applies to Exa, Firecrawl, Parallel, and Perplexity engines;
            ignored with native provider search. Perplexity supports a maximum
            of 20; values above 20 are clamped.
          example: 5
          type: integer
        max_uses:
          description: >-
            Maximum number of web searches the model may perform in a single
            request. Once reached, further search calls return an error result
            instead of executing. Applies to the Exa, Firecrawl, Parallel, and
            Perplexity engines. With native provider search, forwarded only to
            Anthropic (as `max_uses`); other native search providers have no
            equivalent parameter and ignore it.
          example: 3
          type: integer
        mode:
          $ref: '#/components/schemas/WebSearchMode'
        search_context_size:
          $ref: '#/components/schemas/SearchContextSizeEnum'
        type:
          enum:
            - web_search
          type: string
        user_location:
          $ref: '#/components/schemas/WebSearchUserLocation'
      required:
        - type
      type: object
    WebSearchServerTool:
      description: Web search tool configuration (2025-08-26 version)
      example:
        engine: auto
        filters:
          allowed_domains:
            - example.com
        type: web_search_2025_08_26
      properties:
        engine:
          $ref: '#/components/schemas/WebSearchEngineEnum'
        filters:
          $ref: '#/components/schemas/WebSearchDomainFilter'
        max_results:
          description: >-
            Maximum number of search results to return per search call. Defaults
            to 5. Applies to Exa, Firecrawl, Parallel, and Perplexity engines;
            ignored with native provider search. Perplexity supports a maximum
            of 20; values above 20 are clamped.
          example: 5
          type: integer
        max_uses:
          description: >-
            Maximum number of web searches the model may perform in a single
            request. Once reached, further search calls return an error result
            instead of executing. Applies to the Exa, Firecrawl, Parallel, and
            Perplexity engines. With native provider search, forwarded only to
            Anthropic (as `max_uses`); other native search providers have no
            equivalent parameter and ignore it.
          example: 3
          type: integer
        mode:
          $ref: '#/components/schemas/WebSearchMode'
        search_context_size:
          $ref: '#/components/schemas/SearchContextSizeEnum'
        type:
          enum:
            - web_search_2025_08_26
          type: string
        user_location:
          $ref: '#/components/schemas/WebSearchUserLocation'
      required:
        - type
      type: object
    FileSearchServerTool:
      description: File search tool configuration
      example:
        type: file_search
        vector_store_ids:
          - vs_abc123
      properties:
        filters:
          anyOf:
            - properties:
                key:
                  type: string
                type:
                  enum:
                    - eq
                    - ne
                    - gt
                    - gte
                    - lt
                    - lte
                  type: string
                value:
                  anyOf:
                    - type: string
                    - format: double
                      type: number
                    - type: boolean
                    - items:
                        anyOf:
                          - type: string
                          - format: double
                            type: number
                      type: array
              required:
                - key
                - type
                - value
              type: object
            - $ref: '#/components/schemas/CompoundFilter'
            - type: 'null'
        max_num_results:
          type: integer
        ranking_options:
          properties:
            ranker:
              enum:
                - auto
                - default-2024-11-15
              type: string
            score_threshold:
              format: double
              type: number
          type: object
        type:
          enum:
            - file_search
          type: string
        vector_store_ids:
          items:
            type: string
          type: array
      required:
        - type
        - vector_store_ids
      type: object
    ComputerUseServerTool:
      description: Computer use preview tool configuration
      example:
        display_height: 768
        display_width: 1024
        environment: linux
        type: computer_use_preview
      properties:
        display_height:
          type: integer
        display_width:
          type: integer
        environment:
          enum:
            - windows
            - mac
            - linux
            - ubuntu
            - browser
          type: string
        type:
          enum:
            - computer_use_preview
          type: string
      required:
        - type
        - display_height
        - display_width
        - environment
      type: object
    CodeInterpreterServerTool:
      description: Code interpreter tool configuration
      example:
        container: auto
        type: code_interpreter
      properties:
        container:
          anyOf:
            - type: string
            - properties:
                file_ids:
                  items:
                    type: string
                  type: array
                memory_limit:
                  enum:
                    - 1g
                    - 4g
                    - 16g
                    - 64g
                    - null
                  type:
                    - string
                    - 'null'
                type:
                  enum:
                    - auto
                  type: string
              required:
                - type
              type: object
        type:
          enum:
            - code_interpreter
          type: string
      required:
        - type
        - container
      type: object
    McpServerTool:
      description: MCP (Model Context Protocol) tool configuration
      example:
        server_label: my-server
        server_url: https://example.com/mcp
        type: mcp
      properties:
        allowed_tools:
          anyOf:
            - items:
                type: string
              type: array
            - properties:
                read_only:
                  type: boolean
                tool_names:
                  items:
                    type: string
                  type: array
              type: object
            - type: 'null'
        authorization:
          type: string
        connector_id:
          enum:
            - connector_dropbox
            - connector_gmail
            - connector_googlecalendar
            - connector_googledrive
            - connector_microsoftteams
            - connector_outlookcalendar
            - connector_outlookemail
            - connector_sharepoint
          type: string
        headers:
          additionalProperties:
            type: string
          type:
            - object
            - 'null'
        require_approval:
          anyOf:
            - properties:
                always:
                  properties:
                    tool_names:
                      items:
                        type: string
                      type: array
                  type: object
                never:
                  properties:
                    tool_names:
                      items:
                        type: string
                      type: array
                  type: object
              type: object
            - enum:
                - always
              type: string
            - enum:
                - never
              type: string
            - type: 'null'
        server_description:
          type: string
        server_label:
          type: string
        server_url:
          type: string
        type:
          enum:
            - mcp
          type: string
      required:
        - type
        - server_label
      type: object
    ImageGenerationServerTool:
      description: Image generation tool configuration
      example:
        quality: high
        type: image_generation
      properties:
        background:
          enum:
            - transparent
            - opaque
            - auto
          type: string
        input_fidelity:
          enum:
            - high
            - low
            - null
          type:
            - string
            - 'null'
        input_image_mask:
          properties:
            file_id:
              type: string
            image_url:
              type: string
          type: object
        model:
          type: string
        moderation:
          enum:
            - auto
            - low
          type: string
        output_compression:
          type: integer
        output_format:
          enum:
            - png
            - webp
            - jpeg
          type: string
        partial_images:
          type: integer
        quality:
          enum:
            - low
            - medium
            - high
            - auto
          type: string
        size:
          type: string
        type:
          enum:
            - image_generation
          type: string
      required:
        - type
      type: object
    CodexLocalShellTool:
      description: Local shell tool configuration
      example:
        type: local_shell
      properties:
        type:
          enum:
            - local_shell
          type: string
      required:
        - type
      type: object
    ShellServerTool:
      description: Shell tool configuration
      example:
        type: shell
      properties:
        type:
          enum:
            - shell
          type: string
      required:
        - type
      type: object
    ApplyPatchServerTool:
      description: Apply patch tool configuration
      example:
        type: apply_patch
      properties:
        type:
          enum:
            - apply_patch
          type: string
      required:
        - type
      type: object
    CustomTool:
      description: Custom tool configuration
      example:
        name: my_tool
        type: custom
      properties:
        description:
          type: string
        format:
          anyOf:
            - properties:
                type:
                  enum:
                    - text
                  type: string
              required:
                - type
              type: object
            - properties:
                definition:
                  type: string
                syntax:
                  enum:
                    - lark
                    - regex
                  type: string
                type:
                  enum:
                    - grammar
                  type: string
              required:
                - type
                - definition
                - syntax
              type: object
        name:
          type: string
        type:
          enum:
            - custom
          type: string
      required:
        - type
        - name
      type: object
    NamespaceTool:
      description: Groups function/custom tools under a shared namespace
      example:
        description: Tools for spawning and managing sub-agents.
        name: multi_agent_v1
        tools:
          - name: spawn_agent
            type: function
        type: namespace
      properties:
        description:
          type: string
        name:
          type: string
        tools:
          items:
            anyOf:
              - $ref: '#/components/schemas/NamespaceFunctionTool'
              - $ref: '#/components/schemas/CustomTool'
          type: array
        type:
          enum:
            - namespace
          type: string
      required:
        - type
        - name
        - description
        - tools
      type: object
    AdvisorServerTool_OpenRouter:
      description: >-
        OpenRouter built-in server tool: consults a higher-intelligence advisor
        model (any OpenRouter model) for guidance mid-generation and returns its
        response. The advisor may run as a sub-agent with its own tools. Include
        multiple entries to offer several named advisors; at most one entry may
        omit `name` to act as the default advisor.
      example:
        parameters:
          model: ~anthropic/claude-opus-latest
          name: reviewer
        type: openrouter:advisor
      properties:
        parameters:
          $ref: '#/components/schemas/AdvisorServerToolConfig'
        type:
          enum:
            - openrouter:advisor
          type: string
      required:
        - type
      type: object
    SubagentServerTool_OpenRouter:
      description: >-
        OpenRouter built-in server tool: delegates self-contained tasks to a
        smaller, cheaper, faster worker model (any OpenRouter model)
        mid-generation and returns its outcome. The worker may run as a
        sub-agent with its own tools.
      example:
        parameters:
          model: ~anthropic/claude-haiku-latest
        type: openrouter:subagent
      properties:
        parameters:
          $ref: '#/components/schemas/SubagentServerToolConfig'
        type:
          enum:
            - openrouter:subagent
          type: string
      required:
        - type
      type: object
    DatetimeServerTool:
      description: 'OpenRouter built-in server tool: returns the current date and time'
      example:
        parameters:
          timezone: America/New_York
        type: openrouter:datetime
      properties:
        parameters:
          $ref: '#/components/schemas/DatetimeServerToolConfig'
        type:
          enum:
            - openrouter:datetime
          type: string
      required:
        - type
      type: object
    FilesServerTool:
      description: >-
        OpenRouter built-in server tool: read, write, edit, and list workspace
        files via the Files API. Requires an authenticated request; files come
        from the API key's workspace (or the default workspace for keys without
        one).
      example:
        parameters: {}
        type: openrouter:files
      properties:
        parameters:
          $ref: '#/components/schemas/FilesServerToolConfig'
        type:
          enum:
            - openrouter:files
          type: string
      required:
        - type
      type: object
    FusionServerTool_OpenRouter:
      description: >-
        OpenRouter built-in server tool: fans out the user prompt to a panel of
        analysis models, then asks an analyst model to summarize their
        collective output as structured JSON the outer model can synthesize
        from.
      example:
        parameters:
          analysis_models:
            - ~anthropic/claude-opus-latest
            - ~openai/gpt-latest
        type: openrouter:fusion
      properties:
        parameters:
          $ref: '#/components/schemas/FusionServerToolConfig'
        type:
          enum:
            - openrouter:fusion
          type: string
      required:
        - type
      type: object
    ImageGenerationServerTool_OpenRouter:
      description: >-
        OpenRouter built-in server tool: generates images from text prompts
        using an image generation model
      example:
        parameters:
          model: openai/gpt-5-image
          quality: high
          size: 1024x1024
        type: openrouter:image_generation
      properties:
        parameters:
          $ref: '#/components/schemas/ImageGenerationServerToolConfig'
        type:
          enum:
            - openrouter:image_generation
          type: string
      required:
        - type
      type: object
    SearchModelsServerTool_OpenRouter:
      description: >-
        OpenRouter built-in server tool: searches and filters AI models
        available on OpenRouter
      example:
        parameters:
          max_results: 5
        type: openrouter:experimental__search_models
      properties:
        parameters:
          $ref: '#/components/schemas/SearchModelsServerToolConfig'
        type:
          enum:
            - openrouter:experimental__search_models
          type: string
      required:
        - type
      type: object
    WebFetchServerTool:
      description: >-
        OpenRouter built-in server tool: fetches full content from a URL (web
        page or PDF)
      example:
        parameters:
          max_uses: 10
        type: openrouter:web_fetch
      properties:
        parameters:
          $ref: '#/components/schemas/WebFetchServerToolConfig'
        type:
          enum:
            - openrouter:web_fetch
          type: string
      required:
        - type
      type: object
    WebSearchServerTool_OpenRouter:
      description: >-
        OpenRouter built-in server tool: searches the web for current
        information
      example:
        parameters:
          max_results: 5
        type: openrouter:web_search
      properties:
        parameters:
          $ref: '#/components/schemas/WebSearchServerToolConfig'
        type:
          enum:
            - openrouter:web_search
          type: string
      required:
        - type
      type: object
    ApplyPatchServerTool_OpenRouter:
      description: >-
        OpenRouter built-in server tool: validates V4A diff patches for file
        operations (create, update, delete). Restricted to the Responses API.
      example:
        type: openrouter:apply_patch
      properties:
        parameters:
          $ref: '#/components/schemas/ApplyPatchServerToolConfig'
        type:
          enum:
            - openrouter:apply_patch
          type: string
      required:
        - type
      type: object
    BashServerTool:
      description: >-
        OpenRouter built-in server tool: runs shell commands server-side in a
        sandboxed container
      example:
        parameters:
          environment:
            type: container_auto
        type: openrouter:bash
      properties:
        parameters:
          $ref: '#/components/schemas/BashServerToolConfig'
        type:
          enum:
            - openrouter:bash
          type: string
      required:
        - type
      type: object
    ShellServerTool_OpenRouter:
      description: >-
        OpenRouter built-in server tool: runs shell commands server-side in a
        sandboxed container (a sandbox-backed clone of OpenAI's hosted shell
        tool)
      example:
        parameters:
          engine: openrouter
          environment:
            type: container_auto
        type: openrouter:shell
      properties:
        parameters:
          $ref: '#/components/schemas/ShellServerToolConfig'
        type:
          enum:
            - openrouter:shell
          type: string
      required:
        - type
      type: object
    TraceConfig:
      additionalProperties: {}
      description: >-
        Metadata for observability and tracing. Known keys (trace_id,
        trace_name, span_name, generation_name, parent_span_id) have special
        handling. Additional keys are passed through as custom metadata to
        configured broadcast destinations.
      example:
        trace_id: trace-abc123
        trace_name: my-app-trace
      properties:
        generation_name:
          type: string
        parent_span_id:
          type: string
        span_name:
          type: string
        trace_id:
          type: string
        trace_name:
          type: string
      type: object
    OpenAIResponsesTruncation:
      enum:
        - auto
        - disabled
        - null
      example: auto
      type:
        - string
        - 'null'
    BaseResponsesResult:
      example:
        completed_at: 1704067210
        created_at: 1704067200
        error: null
        frequency_penalty: null
        id: resp-abc123
        incomplete_details: null
        instructions: null
        max_output_tokens: null
        metadata: null
        model: gpt-4
        object: response
        output: []
        parallel_tool_calls: true
        presence_penalty: null
        status: completed
        temperature: null
        tool_choice: auto
        tools: []
        top_p: null
      properties:
        background:
          type:
            - boolean
            - 'null'
        completed_at:
          type:
            - integer
            - 'null'
        created_at:
          type: integer
        error:
          $ref: '#/components/schemas/ResponsesErrorField'
        frequency_penalty:
          format: double
          type:
            - number
            - 'null'
        id:
          type: string
        incomplete_details:
          $ref: '#/components/schemas/IncompleteDetails'
        instructions:
          $ref: '#/components/schemas/BaseInputs'
        max_output_tokens:
          type:
            - integer
            - 'null'
        max_tool_calls:
          type:
            - integer
            - 'null'
        metadata:
          $ref: '#/components/schemas/RequestMetadata'
        model:
          type: string
        object:
          enum:
            - response
          type: string
        output:
          items:
            discriminator:
              mapping:
                apply_patch_call:
                  $ref: '#/components/schemas/OutputItemApplyPatchCall'
                code_interpreter_call:
                  $ref: '#/components/schemas/OutputItemCodeInterpreterCall'
                custom_tool_call:
                  $ref: '#/components/schemas/OutputItemCustomToolCall'
                file_search_call:
                  $ref: '#/components/schemas/OutputItemFileSearchCall'
                function_call:
                  $ref: '#/components/schemas/OutputItemFunctionCall'
                image_generation_call:
                  $ref: '#/components/schemas/OutputItemImageGenerationCall'
                message:
                  $ref: '#/components/schemas/OutputMessage'
                reasoning:
                  $ref: '#/components/schemas/OutputItemReasoning'
                web_search_call:
                  $ref: '#/components/schemas/OutputItemWebSearchCall'
              propertyName: type
            oneOf:
              - $ref: '#/components/schemas/OutputMessage'
              - $ref: '#/components/schemas/OutputItemReasoning'
              - $ref: '#/components/schemas/OutputItemFunctionCall'
              - $ref: '#/components/schemas/OutputItemCustomToolCall'
              - $ref: '#/components/schemas/OutputItemWebSearchCall'
              - $ref: '#/components/schemas/OutputItemFileSearchCall'
              - $ref: '#/components/schemas/OutputItemImageGenerationCall'
              - $ref: '#/components/schemas/OutputItemApplyPatchCall'
              - $ref: '#/components/schemas/OutputItemCodeInterpreterCall'
          type: array
        output_text:
          type: string
        parallel_tool_calls:
          type: boolean
        presence_penalty:
          format: double
          type:
            - number
            - 'null'
        previous_response_id:
          type:
            - string
            - 'null'
        prompt:
          $ref: '#/components/schemas/StoredPromptTemplate'
        prompt_cache_key:
          type:
            - string
            - 'null'
        prompt_cache_options:
          $ref: '#/components/schemas/PromptCacheOptions'
        reasoning:
          $ref: '#/components/schemas/BaseReasoningConfig'
        safety_identifier:
          type:
            - string
            - 'null'
        service_tier:
          $ref: '#/components/schemas/ServiceTier'
        status:
          $ref: '#/components/schemas/OpenAIResponsesResponseStatus'
        store:
          type: boolean
        temperature:
          format: double
          type:
            - number
            - 'null'
        text:
          $ref: '#/components/schemas/TextConfig'
        tool_choice:
          $ref: '#/components/schemas/OpenAIResponsesToolChoice'
        tools:
          items:
            oneOf:
              - allOf:
                  - $ref: '#/components/schemas/FunctionTool'
                  - properties: {}
                    type: object
                description: Function tool definition
                example:
                  description: Get the current weather in a location
                  name: get_weather
                  parameters:
                    properties:
                      location:
                        description: The city and state
                        type: string
                      unit:
                        enum:
                          - celsius
                          - fahrenheit
                        type: string
                    required:
                      - location
                    type: object
                  type: function
              - $ref: '#/components/schemas/Preview_WebSearchServerTool'
              - $ref: '#/components/schemas/Preview_20250311_WebSearchServerTool'
              - $ref: '#/components/schemas/Legacy_WebSearchServerTool'
              - $ref: '#/components/schemas/WebSearchServerTool'
              - $ref: '#/components/schemas/FileSearchServerTool'
              - $ref: '#/components/schemas/ComputerUseServerTool'
              - $ref: '#/components/schemas/CodeInterpreterServerTool'
              - $ref: '#/components/schemas/McpServerTool'
              - $ref: '#/components/schemas/ImageGenerationServerTool'
              - $ref: '#/components/schemas/CodexLocalShellTool'
              - $ref: '#/components/schemas/ShellServerTool'
              - $ref: '#/components/schemas/ApplyPatchServerTool'
              - $ref: '#/components/schemas/CustomTool'
              - $ref: '#/components/schemas/NamespaceTool'
          type: array
        top_logprobs:
          type: integer
        top_p:
          format: double
          type:
            - number
            - 'null'
        truncation:
          $ref: '#/components/schemas/Truncation'
        usage:
          $ref: '#/components/schemas/OpenAIResponsesUsage'
        user:
          type:
            - string
            - 'null'
      required:
        - id
        - object
        - created_at
        - model
        - status
        - completed_at
        - output
        - error
        - incomplete_details
        - temperature
        - top_p
        - presence_penalty
        - frequency_penalty
        - instructions
        - metadata
        - tools
        - tool_choice
        - parallel_tool_calls
      type: object
    ApiErrorType:
      description: Canonical OpenRouter error type, stable across all API formats
      enum:
        - context_length_exceeded
        - max_tokens_exceeded
        - token_limit_exceeded
        - string_too_long
        - authentication
        - permission_denied
        - payment_required
        - rate_limit_exceeded
        - provider_overloaded
        - provider_unavailable
        - invalid_request
        - invalid_prompt
        - not_found
        - precondition_failed
        - payload_too_large
        - unprocessable
        - content_policy_violation
        - refusal
        - invalid_image
        - image_too_large
        - image_too_small
        - unsupported_image_format
        - image_not_found
        - image_download_failed
        - server
        - timeout
        - unmapped
      example: rate_limit_exceeded
      type: string
    OpenRouterMetadata:
      example:
        attempt: 1
        endpoints:
          available:
            - model: openai/gpt-4o
              provider: OpenAI
              selected: true
          total: 1
        is_byok: false
        region: iad
        requested: openai/gpt-4o
        strategy: direct
        summary: available=1, selected=OpenAI
      properties:
        attempt:
          type: integer
        attempts:
          items:
            $ref: '#/components/schemas/RouterAttempt'
          type: array
        endpoints:
          $ref: '#/components/schemas/EndpointsMetadata'
        is_byok:
          type: boolean
        params:
          $ref: '#/components/schemas/RouterParams'
        pipeline:
          items:
            $ref: '#/components/schemas/PipelineStage'
          type: array
        region:
          type:
            - string
            - 'null'
        requested:
          type: string
        strategy:
          $ref: '#/components/schemas/RoutingStrategy'
        summary:
          type: string
      required:
        - requested
        - strategy
        - region
        - summary
        - attempt
        - is_byok
        - endpoints
      type: object
    OutputItems:
      description: An output item from the response
      discriminator:
        mapping:
          apply_patch_call:
            $ref: '#/components/schemas/OutputApplyPatchCallItem'
          code_interpreter_call:
            $ref: '#/components/schemas/OutputCodeInterpreterCallItem'
          computer_call:
            $ref: '#/components/schemas/OutputComputerCallItem'
          custom_tool_call:
            $ref: '#/components/schemas/OutputCustomToolCallItem'
          file_search_call:
            $ref: '#/components/schemas/OutputFileSearchCallItem'
          function_call:
            $ref: '#/components/schemas/OutputFunctionCallItem'
          image_generation_call:
            $ref: '#/components/schemas/OutputImageGenerationCallItem'
          message:
            $ref: '#/components/schemas/OutputMessageItem'
          openrouter:advisor:
            $ref: '#/components/schemas/OutputAdvisorServerToolItem'
          openrouter:apply_patch:
            $ref: '#/components/schemas/OutputApplyPatchServerToolItem'
          openrouter:bash:
            $ref: '#/components/schemas/OutputBashServerToolItem'
          openrouter:browser_use:
            $ref: '#/components/schemas/OutputBrowserUseServerToolItem'
          openrouter:code_interpreter:
            $ref: '#/components/schemas/OutputCodeInterpreterServerToolItem'
          openrouter:datetime:
            $ref: '#/components/schemas/OutputDatetimeItem'
          openrouter:experimental__search_models:
            $ref: '#/components/schemas/OutputSearchModelsServerToolItem'
          openrouter:file_search:
            $ref: '#/components/schemas/OutputFileSearchServerToolItem'
          openrouter:files:
            $ref: '#/components/schemas/OutputFilesServerToolItem'
          openrouter:fusion:
            $ref: '#/components/schemas/OutputFusionServerToolItem'
          openrouter:image_generation:
            $ref: '#/components/schemas/OutputImageGenerationServerToolItem'
          openrouter:mcp:
            $ref: '#/components/schemas/OutputMcpServerToolItem'
          openrouter:memory:
            $ref: '#/components/schemas/OutputMemoryServerToolItem'
          openrouter:subagent:
            $ref: '#/components/schemas/OutputSubagentServerToolItem'
          openrouter:text_editor:
            $ref: '#/components/schemas/OutputTextEditorServerToolItem'
          openrouter:tool_search:
            $ref: '#/components/schemas/OutputToolSearchServerToolItem'
          openrouter:web_fetch:
            $ref: '#/components/schemas/OutputWebFetchServerToolItem'
          openrouter:web_search:
            $ref: '#/components/schemas/OutputWebSearchServerToolItem'
          reasoning:
            $ref: '#/components/schemas/OutputReasoningItem'
          shell_call:
            $ref: '#/components/schemas/OutputShellCallItem'
          shell_call_output:
            $ref: '#/components/schemas/OutputShellCallOutputItem'
          web_search_call:
            $ref: '#/components/schemas/OutputWebSearchCallItem'
        propertyName: type
      example:
        content:
          - text: Hello! How can I help you today?
            type: output_text
        id: msg-abc123
        role: assistant
        status: completed
        type: message
      oneOf:
        - $ref: '#/components/schemas/OutputMessageItem'
        - $ref: '#/components/schemas/OutputReasoningItem'
        - $ref: '#/components/schemas/OutputFunctionCallItem'
        - $ref: '#/components/schemas/OutputWebSearchCallItem'
        - $ref: '#/components/schemas/OutputFileSearchCallItem'
        - $ref: '#/components/schemas/OutputImageGenerationCallItem'
        - $ref: '#/components/schemas/OutputCodeInterpreterCallItem'
        - $ref: '#/components/schemas/OutputComputerCallItem'
        - $ref: '#/components/schemas/OutputDatetimeItem'
        - $ref: '#/components/schemas/OutputWebSearchServerToolItem'
        - $ref: '#/components/schemas/OutputCodeInterpreterServerToolItem'
        - $ref: '#/components/schemas/OutputFileSearchServerToolItem'
        - $ref: '#/components/schemas/OutputImageGenerationServerToolItem'
        - $ref: '#/components/schemas/OutputBrowserUseServerToolItem'
        - $ref: '#/components/schemas/OutputBashServerToolItem'
        - $ref: '#/components/schemas/OutputTextEditorServerToolItem'
        - $ref: '#/components/schemas/OutputApplyPatchServerToolItem'
        - $ref: '#/components/schemas/OutputApplyPatchCallItem'
        - $ref: '#/components/schemas/OutputShellCallItem'
        - $ref: '#/components/schemas/OutputShellCallOutputItem'
        - $ref: '#/components/schemas/OutputWebFetchServerToolItem'
        - $ref: '#/components/schemas/OutputToolSearchServerToolItem'
        - $ref: '#/components/schemas/OutputMemoryServerToolItem'
        - $ref: '#/components/schemas/OutputMcpServerToolItem'
        - $ref: '#/components/schemas/OutputSearchModelsServerToolItem'
        - $ref: '#/components/schemas/OutputFusionServerToolItem'
        - $ref: '#/components/schemas/OutputAdvisorServerToolItem'
        - $ref: '#/components/schemas/OutputSubagentServerToolItem'
        - $ref: '#/components/schemas/OutputFilesServerToolItem'
        - $ref: '#/components/schemas/OutputCustomToolCallItem'
    Usage:
      anyOf:
        - allOf:
            - $ref: '#/components/schemas/OpenAIResponsesUsage'
            - properties:
                cost:
                  description: Cost of the completion
                  format: double
                  type:
                    - number
                    - 'null'
                cost_details:
                  properties:
                    upstream_inference_cost:
                      format: double
                      type:
                        - number
                        - 'null'
                    upstream_inference_input_cost:
                      format: double
                      type: number
                    upstream_inference_output_cost:
                      format: double
                      type: number
                  required:
                    - upstream_inference_input_cost
                    - upstream_inference_output_cost
                  type: object
                is_byok:
                  description: >-
                    Whether a request was made using a Bring Your Own Key
                    configuration
                  type: boolean
                server_tool_use_details:
                  $ref: '#/components/schemas/ServerToolUseDetails'
              type: object
        - type: 'null'
      description: Token usage information for the response
      example:
        cost: 0.0012
        cost_details:
          upstream_inference_cost: null
          upstream_inference_input_cost: 0.0008
          upstream_inference_output_cost: 0.0004
        input_tokens: 10
        input_tokens_details:
          cached_tokens: 0
        output_tokens: 25
        output_tokens_details:
          reasoning_tokens: 0
        total_tokens: 35
    StreamEvents:
      description: Union of all possible event types emitted during response streaming
      discriminator:
        mapping:
          error:
            $ref: '#/components/schemas/ErrorEvent'
          response.apply_patch_call_operation_diff.delta:
            $ref: '#/components/schemas/ApplyPatchCallOperationDiffDeltaEvent'
          response.apply_patch_call_operation_diff.done:
            $ref: '#/components/schemas/ApplyPatchCallOperationDiffDoneEvent'
          response.code_interpreter_call_code.delta:
            $ref: '#/components/schemas/CodeInterpreterCallCodeDeltaEvent'
          response.code_interpreter_call_code.done:
            $ref: '#/components/schemas/CodeInterpreterCallCodeDoneEvent'
          response.code_interpreter_call.completed:
            $ref: '#/components/schemas/CodeInterpreterCallCompletedEvent'
          response.code_interpreter_call.in_progress:
            $ref: '#/components/schemas/CodeInterpreterCallInProgressEvent'
          response.code_interpreter_call.interpreting:
            $ref: '#/components/schemas/CodeInterpreterCallInterpretingEvent'
          response.completed:
            $ref: '#/components/schemas/StreamEventsResponseCompleted'
          response.content_part.added:
            $ref: '#/components/schemas/ContentPartAddedEvent'
          response.content_part.done:
            $ref: '#/components/schemas/ContentPartDoneEvent'
          response.created:
            $ref: '#/components/schemas/OpenResponsesCreatedEvent'
          response.custom_tool_call_input.delta:
            $ref: '#/components/schemas/CustomToolCallInputDeltaEvent'
          response.custom_tool_call_input.done:
            $ref: '#/components/schemas/CustomToolCallInputDoneEvent'
          response.debug:
            $ref: '#/components/schemas/DebugEvent'
          response.failed:
            $ref: '#/components/schemas/StreamEventsResponseFailed'
          response.function_call_arguments.delta:
            $ref: '#/components/schemas/FunctionCallArgsDeltaEvent'
          response.function_call_arguments.done:
            $ref: '#/components/schemas/FunctionCallArgsDoneEvent'
          response.fusion_call.analysis.completed:
            $ref: '#/components/schemas/FusionCallAnalysisCompletedEvent'
          response.fusion_call.analysis.in_progress:
            $ref: '#/components/schemas/FusionCallAnalysisInProgressEvent'
          response.fusion_call.completed:
            $ref: '#/components/schemas/FusionCallCompletedEvent'
          response.fusion_call.in_progress:
            $ref: '#/components/schemas/FusionCallInProgressEvent'
          response.fusion_call.panel.added:
            $ref: '#/components/schemas/FusionCallPanelAddedEvent'
          response.fusion_call.panel.completed:
            $ref: '#/components/schemas/FusionCallPanelCompletedEvent'
          response.fusion_call.panel.delta:
            $ref: '#/components/schemas/FusionCallPanelDeltaEvent'
          response.fusion_call.panel.failed:
            $ref: '#/components/schemas/FusionCallPanelFailedEvent'
          response.fusion_call.panel.reasoning.delta:
            $ref: '#/components/schemas/FusionCallPanelReasoningDeltaEvent'
          response.image_generation_call.completed:
            $ref: '#/components/schemas/ImageGenCallCompletedEvent'
          response.image_generation_call.generating:
            $ref: '#/components/schemas/ImageGenCallGeneratingEvent'
          response.image_generation_call.in_progress:
            $ref: '#/components/schemas/ImageGenCallInProgressEvent'
          response.image_generation_call.partial_image:
            $ref: '#/components/schemas/ImageGenCallPartialImageEvent'
          response.in_progress:
            $ref: '#/components/schemas/OpenResponsesInProgressEvent'
          response.incomplete:
            $ref: '#/components/schemas/StreamEventsResponseIncomplete'
          response.output_item.added:
            $ref: '#/components/schemas/StreamEventsResponseOutputItemAdded'
          response.output_item.done:
            $ref: '#/components/schemas/StreamEventsResponseOutputItemDone'
          response.output_text.annotation.added:
            $ref: '#/components/schemas/AnnotationAddedEvent'
          response.output_text.delta:
            $ref: '#/components/schemas/TextDeltaEvent'
          response.output_text.done:
            $ref: '#/components/schemas/TextDoneEvent'
          response.reasoning_summary_part.added:
            $ref: '#/components/schemas/ReasoningSummaryPartAddedEvent'
          response.reasoning_summary_part.done:
            $ref: '#/components/schemas/ReasoningSummaryPartDoneEvent'
          response.reasoning_summary_text.delta:
            $ref: '#/components/schemas/ReasoningSummaryTextDeltaEvent'
          response.reasoning_summary_text.done:
            $ref: '#/components/schemas/ReasoningSummaryTextDoneEvent'
          response.reasoning_text.delta:
            $ref: '#/components/schemas/ReasoningDeltaEvent'
          response.reasoning_text.done:
            $ref: '#/components/schemas/ReasoningDoneEvent'
          response.refusal.delta:
            $ref: '#/components/schemas/RefusalDeltaEvent'
          response.refusal.done:
            $ref: '#/components/schemas/RefusalDoneEvent'
          response.web_search_call.completed:
            $ref: '#/components/schemas/WebSearchCallCompletedEvent'
          response.web_search_call.in_progress:
            $ref: '#/components/schemas/WebSearchCallInProgressEvent'
          response.web_search_call.searching:
            $ref: '#/components/schemas/WebSearchCallSearchingEvent'
        propertyName: type
      example:
        response:
          completed_at: null
          created_at: 1704067200
          error: null
          frequency_penalty: null
          id: resp-abc123
          incomplete_details: null
          instructions: null
          max_output_tokens: null
          metadata: null
          model: gpt-4
          object: response
          output: []
          parallel_tool_calls: true
          presence_penalty: null
          status: in_progress
          temperature: null
          tool_choice: auto
          tools: []
          top_p: null
        sequence_number: 0
        type: response.created
      oneOf:
        - $ref: '#/components/schemas/OpenResponsesCreatedEvent'
        - $ref: '#/components/schemas/OpenResponsesInProgressEvent'
        - $ref: '#/components/schemas/StreamEventsResponseCompleted'
        - $ref: '#/components/schemas/StreamEventsResponseIncomplete'
        - $ref: '#/components/schemas/StreamEventsResponseFailed'
        - $ref: '#/components/schemas/ErrorEvent'
        - $ref: '#/components/schemas/StreamEventsResponseOutputItemAdded'
        - $ref: '#/components/schemas/StreamEventsResponseOutputItemDone'
        - $ref: '#/components/schemas/ContentPartAddedEvent'
        - $ref: '#/components/schemas/ContentPartDoneEvent'
        - $ref: '#/components/schemas/TextDeltaEvent'
        - $ref: '#/components/schemas/TextDoneEvent'
        - $ref: '#/components/schemas/RefusalDeltaEvent'
        - $ref: '#/components/schemas/RefusalDoneEvent'
        - $ref: '#/components/schemas/AnnotationAddedEvent'
        - $ref: '#/components/schemas/FunctionCallArgsDeltaEvent'
        - $ref: '#/components/schemas/FunctionCallArgsDoneEvent'
        - $ref: '#/components/schemas/ReasoningDeltaEvent'
        - $ref: '#/components/schemas/ReasoningDoneEvent'
        - $ref: '#/components/schemas/ReasoningSummaryPartAddedEvent'
        - $ref: '#/components/schemas/ReasoningSummaryPartDoneEvent'
        - $ref: '#/components/schemas/ReasoningSummaryTextDeltaEvent'
        - $ref: '#/components/schemas/ReasoningSummaryTextDoneEvent'
        - $ref: '#/components/schemas/ImageGenCallInProgressEvent'
        - $ref: '#/components/schemas/ImageGenCallGeneratingEvent'
        - $ref: '#/components/schemas/ImageGenCallPartialImageEvent'
        - $ref: '#/components/schemas/ImageGenCallCompletedEvent'
        - $ref: '#/components/schemas/CodeInterpreterCallInProgressEvent'
        - $ref: '#/components/schemas/CodeInterpreterCallInterpretingEvent'
        - $ref: '#/components/schemas/CodeInterpreterCallCompletedEvent'
        - $ref: '#/components/schemas/CodeInterpreterCallCodeDeltaEvent'
        - $ref: '#/components/schemas/CodeInterpreterCallCodeDoneEvent'
        - $ref: '#/components/schemas/WebSearchCallInProgressEvent'
        - $ref: '#/components/schemas/WebSearchCallSearchingEvent'
        - $ref: '#/components/schemas/WebSearchCallCompletedEvent'
        - $ref: '#/components/schemas/CustomToolCallInputDeltaEvent'
        - $ref: '#/components/schemas/CustomToolCallInputDoneEvent'
        - $ref: '#/components/schemas/ApplyPatchCallOperationDiffDeltaEvent'
        - $ref: '#/components/schemas/ApplyPatchCallOperationDiffDoneEvent'
        - $ref: '#/components/schemas/FusionCallInProgressEvent'
        - $ref: '#/components/schemas/FusionCallPanelAddedEvent'
        - $ref: '#/components/schemas/FusionCallPanelDeltaEvent'
        - $ref: '#/components/schemas/FusionCallPanelReasoningDeltaEvent'
        - $ref: '#/components/schemas/FusionCallPanelCompletedEvent'
        - $ref: '#/components/schemas/FusionCallPanelFailedEvent'
        - $ref: '#/components/schemas/FusionCallAnalysisInProgressEvent'
        - $ref: '#/components/schemas/FusionCallAnalysisCompletedEvent'
        - $ref: '#/components/schemas/FusionCallCompletedEvent'
        - $ref: '#/components/schemas/DebugEvent'
    BadRequestResponseErrorData:
      description: Error data for BadRequestResponse
      example:
        code: 400
        message: Invalid request parameters
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    UnauthorizedResponseErrorData:
      description: Error data for UnauthorizedResponse
      example:
        code: 401
        message: Missing Authentication header
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    PaymentRequiredResponseErrorData:
      description: Error data for PaymentRequiredResponse
      example:
        code: 402
        message: Insufficient credits. Add more using https://openrouter.ai/credits
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    ForbiddenResponseErrorData:
      description: Error data for ForbiddenResponse
      example:
        code: 403
        message: Only management keys can perform this operation
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    NotFoundResponseErrorData:
      description: Error data for NotFoundResponse
      example:
        code: 404
        message: Resource not found
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    RequestTimeoutResponseErrorData:
      description: Error data for RequestTimeoutResponse
      example:
        code: 408
        message: Operation timed out. Please try again later.
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    PayloadTooLargeResponseErrorData:
      description: Error data for PayloadTooLargeResponse
      example:
        code: 413
        message: Request payload too large
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    UnprocessableEntityResponseErrorData:
      description: Error data for UnprocessableEntityResponse
      example:
        code: 422
        message: Invalid argument
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    TooManyRequestsResponseErrorData:
      description: Error data for TooManyRequestsResponse
      example:
        code: 429
        message: Rate limit exceeded
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    InternalServerResponseErrorData:
      description: Error data for InternalServerResponse
      example:
        code: 500
        message: Internal Server Error
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    BadGatewayResponseErrorData:
      description: Error data for BadGatewayResponse
      example:
        code: 502
        message: Provider returned error
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    ServiceUnavailableResponseErrorData:
      description: Error data for ServiceUnavailableResponse
      example:
        code: 503
        message: Service temporarily unavailable
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    EdgeNetworkTimeoutResponseErrorData:
      description: Error data for EdgeNetworkTimeoutResponse
      example:
        code: 524
        message: Request timed out. Please try again later.
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    ProviderOverloadedResponseErrorData:
      description: Error data for ProviderOverloadedResponse
      example:
        code: 529
        message: Provider returned error
      properties:
        code:
          type: integer
        message:
          type: string
        metadata:
          additionalProperties: {}
          type:
            - object
            - 'null'
      required:
        - code
        - message
      type: object
    AnthropicCacheControlTtl:
      enum:
        - 5m
        - 1h
      example: 5m
      type: string
    ReasoningItem:
      allOf:
        - $ref: '#/components/schemas/OutputItemReasoning'
        - properties:
            content:
              items:
                $ref: '#/components/schemas/ReasoningTextContent'
              type:
                - array
                - 'null'
            format:
              $ref: '#/components/schemas/ReasoningFormat'
            signature:
              type:
                - string
                - 'null'
          type: object
      description: Reasoning output item with signature and format extensions
      example:
        id: reasoning-abc123
        summary:
          - text: Step by step analysis
            type: summary_text
        type: reasoning
    EasyInputMessage:
      example:
        content: What is the weather today?
        role: user
      properties:
        content:
          anyOf:
            - items:
                oneOf:
                  - $ref: '#/components/schemas/InputText'
                  - allOf:
                      - $ref: '#/components/schemas/InputImage'
                      - properties: {}
                        type: object
                    description: Image input content item
                    example:
                      detail: auto
                      image_url: https://example.com/image.jpg
                      type: input_image
                  - $ref: '#/components/schemas/InputFile'
                  - $ref: '#/components/schemas/InputAudio'
                  - $ref: '#/components/schemas/InputVideo'
              type: array
            - type: string
            - type: 'null'
        phase:
          anyOf:
            - enum:
                - commentary
              type: string
            - enum:
                - final_answer
              type: string
            - type: 'null'
          description: >-
            The phase of an assistant message. Use `commentary` for an
            intermediate assistant message and `final_answer` for the final
            assistant message. For follow-up requests with models like
            `gpt-5.3-codex` and later, preserve and resend phase on all
            assistant messages. Omitting it can degrade performance. Not used
            for user messages.
          example: final_answer
        role:
          anyOf:
            - enum:
                - user
              type: string
            - enum:
                - system
              type: string
            - enum:
                - assistant
              type: string
            - enum:
                - developer
              type: string
        type:
          enum:
            - message
          type: string
      required:
        - role
      type: object
    InputMessageItem:
      example:
        content:
          - text: Hello, how are you?
            type: input_text
        id: msg-abc123
        role: user
        type: message
      properties:
        content:
          items:
            oneOf:
              - $ref: '#/components/schemas/InputText'
              - allOf:
                  - $ref: '#/components/schemas/InputImage'
                  - properties: {}
                    type: object
                description: Image input content item
                example:
                  detail: auto
                  image_url: https://example.com/image.jpg
                  type: input_image
              - $ref: '#/components/schemas/InputFile'
              - $ref: '#/components/schemas/InputAudio'
              - $ref: '#/components/schemas/InputVideo'
          type:
            - array
            - 'null'
        id:
          type: string
        role:
          anyOf:
            - enum:
                - user
              type: string
            - enum:
                - system
              type: string
            - enum:
                - developer
              type: string
        type:
          enum:
            - message
          type: string
      required:
        - role
      type: object
    FunctionCallItem:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponseFunctionToolCall'
        - properties: {}
          required:
            - id
          type: object
      description: A function call initiated by the model
      example:
        arguments: '{"location":"San Francisco"}'
        call_id: call-abc123
        id: call-abc123
        name: get_weather
        status: completed
        type: function_call
    FunctionCallOutputItem:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponseFunctionToolCallOutput'
        - properties:
            output:
              anyOf:
                - type: string
                - items:
                    oneOf:
                      - $ref: '#/components/schemas/InputText'
                      - allOf:
                          - $ref: '#/components/schemas/InputImage'
                          - properties: {}
                            type: object
                        description: Image input content item
                        example:
                          detail: auto
                          image_url: https://example.com/image.jpg
                          type: input_image
                      - $ref: '#/components/schemas/InputFile'
                  type: array
          type: object
      description: The output from a function call execution
      example:
        call_id: call-abc123
        id: output-abc123
        output: '{"temperature":72,"conditions":"sunny"}'
        status: completed
        type: function_call_output
    ApplyPatchCallItem:
      description: >-
        A tool call emitted by the model requesting a V4A patch operation. The
        client applies the patch and echoes an `apply_patch_call_output` on the
        next turn.
      example:
        call_id: call_abc123
        id: apc_abc123
        operation:
          diff: |-
            @@ function main() {
            +  console.log("hi");
             }
          path: /src/main.ts
          type: update_file
        status: completed
        type: apply_patch_call
      properties:
        call_id:
          type: string
        id:
          type:
            - string
            - 'null'
        operation:
          $ref: '#/components/schemas/ApplyPatchCallOperation'
        status:
          $ref: '#/components/schemas/ApplyPatchCallStatus'
        type:
          enum:
            - apply_patch_call
          type: string
      required:
        - type
        - call_id
        - status
        - operation
      type: object
    ApplyPatchCallOutputItem:
      description: >-
        The client's echo of an `apply_patch_call` after applying the patch.
        `output` is an optional human-readable log; `status` is `completed` when
        the patch was applied successfully, `failed` otherwise.
      example:
        call_id: call_abc123
        output: Applied patch to /src/main.ts
        status: completed
        type: apply_patch_call_output
      properties:
        call_id:
          type: string
        id:
          type:
            - string
            - 'null'
        output:
          type:
            - string
            - 'null'
        status:
          enum:
            - completed
            - failed
          type: string
        type:
          enum:
            - apply_patch_call_output
          type: string
      required:
        - type
        - call_id
        - status
      type: object
    OutputMessageItem:
      allOf:
        - $ref: '#/components/schemas/OutputMessage'
        - properties: {}
          type: object
      description: An output message item
      example:
        content:
          - annotations: []
            text: Hello! How can I help you?
            type: output_text
        id: msg-123
        role: assistant
        status: completed
        type: message
    ResponseOutputText:
      example:
        annotations:
          - end_index: 42
            start_index: 0
            title: Paris - Wikipedia
            type: url_citation
            url: https://en.wikipedia.org/wiki/Paris
        text: The capital of France is Paris.
        type: output_text
      properties:
        annotations:
          items:
            $ref: '#/components/schemas/OpenAIResponsesAnnotation'
          type: array
        logprobs:
          items:
            properties:
              bytes:
                items:
                  type: integer
                type: array
              logprob:
                format: double
                type: number
              token:
                type: string
              top_logprobs:
                items:
                  properties:
                    bytes:
                      items:
                        type: integer
                      type: array
                    logprob:
                      format: double
                      type: number
                    token:
                      type: string
                  required:
                    - token
                    - bytes
                    - logprob
                  type: object
                type: array
            required:
              - token
              - bytes
              - logprob
              - top_logprobs
            type: object
          type: array
        text:
          type: string
        type:
          enum:
            - output_text
          type: string
      required:
        - type
        - text
      type: object
    OpenAIResponsesRefusalContent:
      example:
        refusal: I'm sorry, I cannot assist with that request
        type: refusal
      properties:
        refusal:
          type: string
        type:
          enum:
            - refusal
          type: string
      required:
        - type
        - refusal
      type: object
    OutputReasoningItem:
      allOf:
        - $ref: '#/components/schemas/OutputItemReasoning'
        - properties:
            content:
              items:
                $ref: '#/components/schemas/ReasoningTextContent'
              type:
                - array
                - 'null'
            format:
              $ref: '#/components/schemas/ReasoningFormat'
            signature:
              description: A signature for the reasoning content, used for verification
              example: EvcBCkgIChABGAIqQKkSDbRuVEQUk9qN1odC098l9SEj...
              type:
                - string
                - 'null'
          type: object
      description: An output item containing reasoning
      example:
        content:
          - text: First, we analyze the problem...
            type: reasoning_text
        format: anthropic-claude-v1
        id: reasoning-123
        signature: EvcBCkgIChABGAIqQKkSDbRuVEQUk9qN1odC098l9SEj...
        status: completed
        summary:
          - text: Analyzed the problem and found the optimal solution.
            type: summary_text
        type: reasoning
    ReasoningSummaryText:
      example:
        text: Analyzed the problem using first principles
        type: summary_text
      properties:
        text:
          type: string
        type:
          enum:
            - summary_text
          type: string
      required:
        - type
        - text
      type: object
    OutputFunctionCallItem:
      allOf:
        - $ref: '#/components/schemas/OutputItemFunctionCall'
        - properties: {}
          type: object
      example:
        arguments: '{"location":"San Francisco"}'
        call_id: call-abc123
        id: fc-abc123
        name: get_weather
        status: completed
        type: function_call
    OutputCustomToolCallItem:
      description: >-
        A call to a custom (freeform-grammar) tool created by the model —
        distinct from `function_call`. Used for tools like Codex CLI's
        `apply_patch` whose payload is opaque text rather than JSON arguments.
      example:
        call_id: call-abc123
        id: ctc-abc123
        input: |-
          *** Begin Patch
          *** End Patch
        name: apply_patch
        type: custom_tool_call
      properties:
        call_id:
          type: string
        id:
          type: string
        input:
          type: string
        name:
          type: string
        namespace:
          description: >-
            Namespace qualifier for tools registered as part of a namespace tool
            group (e.g. an MCP server)
          type: string
        type:
          enum:
            - custom_tool_call
          type: string
      required:
        - type
        - name
        - input
        - call_id
      type: object
    OutputWebSearchCallItem:
      allOf:
        - $ref: '#/components/schemas/OutputItemWebSearchCall'
        - additionalProperties: {}
          properties: {}
          type: object
      example:
        id: ws-abc123
        status: completed
        type: web_search_call
    OutputFileSearchCallItem:
      allOf:
        - $ref: '#/components/schemas/OutputItemFileSearchCall'
        - properties: {}
          type: object
      example:
        id: fs-abc123
        queries:
          - search term
        results: []
        status: completed
        type: file_search_call
    OutputImageGenerationCallItem:
      allOf:
        - $ref: '#/components/schemas/OutputItemImageGenerationCall'
        - properties:
            prompt:
              description: >-
                The prompt (possibly rewritten) that the image was generated
                from.
              type: string
          type: object
      example:
        id: img-abc123
        result: null
        status: completed
        type: image_generation_call
    OutputCodeInterpreterCallItem:
      allOf:
        - $ref: '#/components/schemas/CodeInterpreterCallItem'
        - additionalProperties: {}
          properties: {}
          type: object
      description: A code interpreter execution call with outputs
      example:
        code: print("hello")
        container_id: ctr-xyz789
        id: ci-abc123
        outputs:
          - logs: |
              hello
            type: logs
        status: completed
        type: code_interpreter_call
    OutputComputerCallItem:
      example:
        action:
          type: screenshot
        call_id: call-abc123
        id: cu-abc123
        pending_safety_checks: []
        status: completed
        type: computer_call
      properties:
        action: {}
        call_id:
          type: string
        id:
          type: string
        pending_safety_checks:
          items:
            properties:
              code:
                type: string
              id:
                type: string
              message:
                type: string
            required:
              - id
              - code
              - message
            type: object
          type: array
        status:
          enum:
            - completed
            - incomplete
            - in_progress
          type: string
        type:
          enum:
            - computer_call
          type: string
      required:
        - type
        - call_id
        - status
        - pending_safety_checks
      type: object
    OutputDatetimeItem:
      description: An openrouter:datetime server tool output item
      example:
        datetime: '2026-03-12T14:30:00.000Z'
        id: dt_tmp_abc123
        status: completed
        timezone: UTC
        type: openrouter:datetime
      properties:
        datetime:
          description: ISO 8601 datetime string
          type: string
        id:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        timezone:
          description: IANA timezone name
          type: string
        type:
          enum:
            - openrouter:datetime
          type: string
      required:
        - status
        - type
        - datetime
        - timezone
      type: object
    OutputWebSearchServerToolItem:
      description: An openrouter:web_search server tool output item
      example:
        action:
          query: latest AI news
          type: search
        id: ws_tmp_abc123
        status: completed
        type: openrouter:web_search
      properties:
        action:
          description: >-
            The search action performed, matching OpenAI web_search_call.action
            shape. Includes the query the model issued and optional source URLs
            returned by the search provider.
          properties:
            query:
              type: string
            sources:
              items:
                properties:
                  type:
                    enum:
                      - url
                    type: string
                  url:
                    type: string
                required:
                  - type
                  - url
                type: object
              type: array
            type:
              enum:
                - search
              type: string
          required:
            - type
            - query
          type: object
        id:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:web_search
          type: string
      required:
        - status
        - type
      type: object
    OutputCodeInterpreterServerToolItem:
      description: An openrouter:code_interpreter server tool output item
      example:
        code: print("hello")
        id: ci_tmp_abc123
        language: python
        status: completed
        stdout: |
          hello
        type: openrouter:code_interpreter
      properties:
        code:
          type: string
        exitCode:
          type: integer
        id:
          type: string
        language:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        stderr:
          type: string
        stdout:
          type: string
        type:
          enum:
            - openrouter:code_interpreter
          type: string
      required:
        - status
        - type
      type: object
    OutputFileSearchServerToolItem:
      description: An openrouter:file_search server tool output item
      example:
        id: fs_tmp_abc123
        queries:
          - search term
        status: completed
        type: openrouter:file_search
      properties:
        id:
          type: string
        queries:
          items:
            type: string
          type: array
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:file_search
          type: string
      required:
        - status
        - type
      type: object
    OutputImageGenerationServerToolItem:
      description: An openrouter:image_generation server tool output item
      example:
        id: ig_tmp_abc123
        imageUrl: https://example.com/image.png
        result: https://example.com/image.png
        status: completed
        type: openrouter:image_generation
      properties:
        id:
          type: string
        imageB64:
          type: string
        imageUrl:
          type: string
        prompt:
          description: The prompt (possibly rewritten) that the image was generated from.
          type: string
        result:
          description: >-
            The generated image as a base64-encoded string or URL, matching
            OpenAI image_generation_call format
          type:
            - string
            - 'null'
        revisedPrompt:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:image_generation
          type: string
      required:
        - status
        - type
      type: object
    OutputBrowserUseServerToolItem:
      description: An openrouter:browser_use server tool output item
      example:
        action: screenshot
        id: bu_tmp_abc123
        status: completed
        type: openrouter:browser_use
      properties:
        action:
          type: string
        id:
          type: string
        screenshotB64:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:browser_use
          type: string
      required:
        - status
        - type
      type: object
    OutputBashServerToolItem:
      description: An openrouter:bash server tool output item
      example:
        command: ls -la
        exitCode: 0
        id: bash_tmp_abc123
        status: completed
        stdout: |
          total 0
        type: openrouter:bash
      properties:
        arguments:
          description: The raw tool-call arguments string as emitted by the model.
          type: string
        call_id:
          description: The model-generated tool call id from the originating turn.
          type: string
        command:
          type: string
        exitCode:
          type: integer
        id:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        stderr:
          type: string
        stdout:
          type: string
        type:
          enum:
            - openrouter:bash
          type: string
      required:
        - status
        - type
      type: object
    OutputTextEditorServerToolItem:
      description: An openrouter:text_editor server tool output item
      example:
        command: view
        filePath: /src/main.ts
        id: te_tmp_abc123
        status: completed
        type: openrouter:text_editor
      properties:
        command:
          enum:
            - view
            - create
            - str_replace
            - insert
          type: string
        filePath:
          type: string
        id:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:text_editor
          type: string
      required:
        - status
        - type
      type: object
    OutputApplyPatchServerToolItem:
      description: >-
        An openrouter:apply_patch server tool output item. The turn halts when
        validation succeeds so the client can apply the patch and echo an
        `apply_patch_call_output` on the next turn.
      example:
        call_id: call_abc123
        id: apc_abc123
        operation:
          diff: |-
            @@ function main() {
            +  console.log("hi");
             }
          path: /src/main.ts
          type: update_file
        status: completed
        type: openrouter:apply_patch
      properties:
        call_id:
          type: string
        id:
          type: string
        operation:
          $ref: '#/components/schemas/ApplyPatchCallOperation'
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:apply_patch
          type: string
      required:
        - status
        - type
      type: object
    OutputWebFetchServerToolItem:
      description: An openrouter:web_fetch server tool output item
      example:
        httpStatus: 200
        id: wf_tmp_abc123
        status: completed
        title: Example Domain
        type: openrouter:web_fetch
        url: https://example.com
      properties:
        content:
          type: string
        error:
          description: The error message if the fetch failed.
          type: string
        httpStatus:
          description: The HTTP status code returned by the upstream URL fetch.
          type: integer
        id:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        title:
          type: string
        type:
          enum:
            - openrouter:web_fetch
          type: string
        url:
          type: string
      required:
        - status
        - type
      type: object
    OutputToolSearchServerToolItem:
      description: An openrouter:tool_search server tool output item
      example:
        id: ts_tmp_abc123
        query: weather tools
        status: completed
        type: openrouter:tool_search
      properties:
        id:
          type: string
        query:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:tool_search
          type: string
      required:
        - status
        - type
      type: object
    OutputMemoryServerToolItem:
      description: An openrouter:memory server tool output item
      example:
        action: read
        id: mem_tmp_abc123
        key: user_preference
        status: completed
        type: openrouter:memory
      properties:
        action:
          enum:
            - read
            - write
            - delete
          type: string
        id:
          type: string
        key:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:memory
          type: string
        value: {}
      required:
        - status
        - type
      type: object
    OutputMcpServerToolItem:
      description: An openrouter:mcp server tool output item
      example:
        id: mcp_tmp_abc123
        serverLabel: my-server
        status: completed
        toolName: get_data
        type: openrouter:mcp
      properties:
        id:
          type: string
        serverLabel:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        toolName:
          type: string
        type:
          enum:
            - openrouter:mcp
          type: string
      required:
        - status
        - type
      type: object
    OutputSearchModelsServerToolItem:
      description: An openrouter:experimental__search_models server tool output item
      example:
        arguments: '{"query":"Claude Opus"}'
        id: sm_tmp_abc123
        query: Claude Opus
        status: completed
        type: openrouter:experimental__search_models
      properties:
        arguments:
          description: >-
            The JSON arguments submitted to the search tool (e.g.
            {"query":"Claude"})
          type: string
        id:
          type: string
        query:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:experimental__search_models
          type: string
      required:
        - status
        - type
      type: object
    OutputFusionServerToolItem:
      description: An openrouter:fusion server tool output item
      example:
        id: st_tmp_abc123
        status: completed
        type: openrouter:fusion
      properties:
        analysis:
          $ref: '#/components/schemas/FusionAnalysisResult'
        error:
          description: >-
            Error message when the fusion run did not produce an analysis
            result.
          type: string
        failed_models:
          description: >-
            Models that were requested as part of the analysis panel but did not
            produce a response. Present when at least one requested analysis
            model failed. The fusion result is still usable but was produced
            from a degraded panel.
          items:
            properties:
              error:
                description: Error message describing why the model failed.
                type: string
              model:
                description: Slug of the analysis model that failed.
                type: string
              status_code:
                description: >-
                  HTTP status code from the upstream response, when available
                  (e.g. 402, 429).
                type: integer
            required:
              - model
              - error
            type: object
          type: array
        failure_reason:
          description: >-
            Typed failure reason when the fusion run failed. Possible values
            include: all_panels_failed, insufficient_credits, rate_limited,
            invalid_model, judge_not_valid_json, judge_schema_mismatch,
            judge_upstream_error, judge_empty_completion. The four
            analysis-stage codes keep their pre-rename `judge_` spelling so
            existing consumers keep matching. The consumer-cancellation code is
            `cancelled`.
          type: string
        id:
          type: string
        responses:
          description: >-
            Analysis models that produced a response in this fusion run, with
            each model's full panel content.
          items:
            properties:
              content:
                type: string
              model:
                type: string
            required:
              - model
            type: object
          type: array
        sources:
          description: >-
            Web pages the analysis panels and analyst retrieved via web search
            during this fusion run, deduplicated by URL across the whole run.
            Present when at least one model cited a source.
          items:
            $ref: '#/components/schemas/FusionSource'
          type: array
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:fusion
          type: string
      required:
        - status
        - type
      type: object
    OutputAdvisorServerToolItem:
      description: An openrouter:advisor server tool output item
      example:
        id: st_tmp_abc123
        status: completed
        type: openrouter:advisor
      properties:
        advice:
          description: >-
            The advisor model's response (the advice text returned to the
            executor).
          type: string
        error:
          description: Error message when the advisor call did not produce advice.
          type: string
        id:
          type: string
        instance_name:
          description: >-
            Provider-safe function name of the specific advisor instance that
            produced this item (e.g. `openrouter_advisor__1`). Present only when
            more than one advisor tool is configured; omitted for the default
            single advisor. Echo this field back unchanged so the advisor's
            cross-request memory stays namespaced to the correct instance. This
            identity is positional: it is derived from the index of the advisor
            entry in the request `tools` array, so clients must keep the order
            of advisor tool entries stable across requests in a conversation.
            Reordering or inserting advisor entries shifts these names and
            causes each advisor's cross-request memory to be attributed to the
            wrong instance.
          example: openrouter_advisor__1
          type: string
        model:
          description: Slug of the advisor model that was consulted.
          type: string
        prompt:
          description: The prompt the executor sent to the advisor.
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:advisor
          type: string
      required:
        - status
        - type
      type: object
    OutputSubagentServerToolItem:
      description: An openrouter:subagent server tool output item
      example:
        id: st_tmp_abc123
        status: completed
        type: openrouter:subagent
      properties:
        error:
          description: Error message when the subagent task did not produce an outcome.
          type: string
        id:
          type: string
        instance_name:
          description: >-
            Provider-safe function name of the specific subagent instance that
            produced this item (e.g. `openrouter_subagent__1`). Present only on
            items from non-default instances — the second and later subagent
            entries in the request `tools` array. The first (default) instance
            omits it, even when multiple subagents are configured. When a
            replayed item echoes this field back, the transcript rehydrates the
            call under that instance's tool. This identity is positional: it is
            derived from the index of the subagent entry in the request `tools`
            array, so keep the order of subagent entries stable across requests
            in a conversation.
          example: openrouter_subagent__1
          type: string
        model:
          description: Slug of the worker model that executed the task.
          type: string
        name:
          description: >-
            Configured name of the subagent that executed the task (the `name`
            on its tool entry). Present only for named subagents; omitted for an
            unnamed (default) subagent.
          example: summarizer
          type: string
        outcome:
          description: >-
            The worker model's result (the outcome text returned to the
            delegating model).
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        task_description:
          description: The task description the delegating model sent to the worker.
          type: string
        task_name:
          description: The short task identifier the delegating model supplied.
          type: string
        type:
          enum:
            - openrouter:subagent
          type: string
      required:
        - status
        - type
      type: object
    OutputFilesServerToolItem:
      description: An openrouter:files server tool output item
      example:
        filename: notes.txt
        id: fl_tmp_abc123
        operation: read
        result: '{"id":"file_abc","filename":"notes.txt","content":"hello"}'
        status: completed
        type: openrouter:files
      properties:
        error:
          description: Error message when the file operation failed.
          type: string
        file_id:
          description: The target file id supplied in the tool-call arguments.
          type: string
        filename:
          description: The target filename supplied in the tool-call arguments.
          type: string
        id:
          type: string
        operation:
          description: The file operation performed (list, read, write, or edit).
          type: string
        result:
          description: JSON-serialized result of the file operation.
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - openrouter:files
          type: string
      required:
        - status
        - type
      type: object
    LocalShellCallItem:
      description: A local shell command execution call
      example:
        action:
          command:
            - ls
            - '-la'
          env:
            PATH: /usr/bin
          timeout_ms: 5000
          type: exec
        call_id: call-abc123
        id: shell-abc123
        status: completed
        type: local_shell_call
      properties:
        action:
          properties:
            command:
              items:
                type: string
              type: array
            env:
              additionalProperties:
                type: string
              type: object
            timeout_ms:
              type:
                - integer
                - 'null'
            type:
              enum:
                - exec
              type: string
            user:
              type:
                - string
                - 'null'
            working_directory:
              type:
                - string
                - 'null'
          required:
            - type
            - command
            - env
          type: object
        call_id:
          type: string
        id:
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - local_shell_call
          type: string
      required:
        - type
        - id
        - call_id
        - action
        - status
      type: object
    LocalShellCallOutputItem:
      description: Output from a local shell command execution
      example:
        id: output-abc123
        output: |-
          total 24
          drwxr-xr-x  5 user  staff  160 Jan  1 12:00 .
        status: completed
        type: local_shell_call_output
      properties:
        id:
          type: string
        output:
          type: string
        status:
          anyOf:
            - $ref: '#/components/schemas/ToolCallStatus'
            - type: 'null'
        type:
          enum:
            - local_shell_call_output
          type: string
      required:
        - type
        - id
        - output
      type: object
    ShellCallItem:
      description: A shell command execution call (newer variant)
      example:
        action:
          commands:
            - ls
            - '-la'
          max_output_length: 10000
        call_id: call-abc123
        status: completed
        type: shell_call
      properties:
        action:
          properties:
            commands:
              items:
                type: string
              type: array
            max_output_length:
              type:
                - integer
                - 'null'
            timeout_ms:
              type:
                - integer
                - 'null'
          required:
            - commands
          type: object
        call_id:
          type: string
        environment: {}
        id:
          type:
            - string
            - 'null'
        status:
          anyOf:
            - $ref: '#/components/schemas/ToolCallStatus'
            - type: 'null'
        type:
          enum:
            - shell_call
          type: string
      required:
        - type
        - call_id
        - action
      type: object
    ShellCallOutputItem:
      description: Output from a shell command execution (newer variant)
      example:
        call_id: call-abc123
        output:
          - content: |
              total 0
            type: stdout
        status: completed
        type: shell_call_output
      properties:
        call_id:
          type: string
        id:
          type:
            - string
            - 'null'
        max_output_length:
          type:
            - integer
            - 'null'
        output:
          items:
            additionalProperties: {}
            properties:
              content:
                type:
                  - string
                  - 'null'
              exit_code:
                type:
                  - integer
                  - 'null'
              type:
                type: string
            required:
              - type
            type: object
          type: array
        status:
          anyOf:
            - $ref: '#/components/schemas/ToolCallStatus'
            - type: 'null'
        type:
          enum:
            - shell_call_output
          type: string
      required:
        - type
        - call_id
        - output
      type: object
    McpListToolsItem:
      description: List of available MCP tools from a server
      example:
        id: mcp-list-abc123
        server_label: database-server
        tools:
          - description: Execute a database query
            input_schema:
              properties:
                query:
                  type: string
              type: object
            name: query_database
        type: mcp_list_tools
      properties:
        error:
          type:
            - string
            - 'null'
        id:
          type: string
        server_label:
          type: string
        tools:
          items:
            properties:
              annotations: {}
              description:
                type:
                  - string
                  - 'null'
              input_schema:
                additionalProperties: {}
                type: object
              name:
                type: string
            required:
              - name
              - input_schema
            type: object
          type: array
        type:
          enum:
            - mcp_list_tools
          type: string
      required:
        - type
        - id
        - server_label
        - tools
      type: object
    McpApprovalRequestItem:
      description: Request for approval to execute an MCP tool
      example:
        arguments: '{"id":"123"}'
        id: approval-abc123
        name: delete_record
        server_label: database-server
        type: mcp_approval_request
      properties:
        arguments:
          type: string
        id:
          type: string
        name:
          type: string
        server_label:
          type: string
        type:
          enum:
            - mcp_approval_request
          type: string
      required:
        - type
        - id
        - name
        - arguments
        - server_label
      type: object
    McpApprovalResponseItem:
      description: User response to an MCP tool approval request
      example:
        approval_request_id: approval-abc123
        approve: true
        reason: Approved for execution
        type: mcp_approval_response
      properties:
        approval_request_id:
          type: string
        approve:
          type: boolean
        id:
          type:
            - string
            - 'null'
        reason:
          type:
            - string
            - 'null'
        type:
          enum:
            - mcp_approval_response
          type: string
      required:
        - type
        - approval_request_id
        - approve
      type: object
    McpCallItem:
      description: An MCP tool call with its output or error
      example:
        arguments: '{"query":"SELECT * FROM users"}'
        id: mcp-call-abc123
        name: query_database
        output: '[{"id":1,"name":"Alice"}]'
        server_label: database-server
        type: mcp_call
      properties:
        arguments:
          type: string
        error:
          type:
            - string
            - 'null'
        id:
          type: string
        name:
          type: string
        output:
          type:
            - string
            - 'null'
        server_label:
          type: string
        type:
          enum:
            - mcp_call
          type: string
      required:
        - type
        - id
        - name
        - arguments
        - server_label
      type: object
    CustomToolCallItem:
      description: >-
        A call to a custom (freeform-grammar) tool created by the model —
        distinct from `function_call`. Used for tools like Codex CLI's
        `apply_patch` whose payload is opaque text rather than JSON arguments.
      example:
        call_id: call-abc123
        id: ctc-abc123
        input: |-
          *** Begin Patch
          *** End Patch
        name: apply_patch
        type: custom_tool_call
      properties:
        call_id:
          type: string
        id:
          type: string
        input:
          type: string
        name:
          type: string
        namespace:
          description: >-
            Namespace qualifier for tools registered as part of a namespace tool
            group (e.g. an MCP server)
          type: string
        type:
          enum:
            - custom_tool_call
          type: string
      required:
        - type
        - call_id
        - name
        - input
      type: object
    CustomToolCallOutputItem:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponseCustomToolCallOutput'
        - properties:
            output:
              anyOf:
                - type: string
                - items:
                    oneOf:
                      - $ref: '#/components/schemas/InputText'
                      - allOf:
                          - $ref: '#/components/schemas/InputImage'
                          - properties: {}
                            type: object
                        description: Image input content item
                        example:
                          detail: auto
                          image_url: https://example.com/image.jpg
                          type: input_image
                      - $ref: '#/components/schemas/InputFile'
                  type: array
          type: object
      description: >-
        The output from a custom (freeform-grammar) tool call execution. Mirrors
        `function_call_output` but is matched to a `custom_tool_call` rather
        than a `function_call`.
      example:
        call_id: call-abc123
        id: ctco-abc123
        output: patch applied successfully
        type: custom_tool_call_output
    CompactionItem:
      additionalProperties: {}
      description: A context compaction marker with encrypted summary
      example:
        encrypted_content: enc_abc123...
        type: compaction
      properties:
        encrypted_content:
          type: string
        id:
          type:
            - string
            - 'null'
        type:
          enum:
            - compaction
          type: string
      required:
        - type
        - encrypted_content
      type: object
    ContextCompactionItem:
      additionalProperties: {}
      description: A context compaction marker with an optional encrypted summary
      example:
        encrypted_content: enc_abc123...
        type: context_compaction
      properties:
        encrypted_content:
          type:
            - string
            - 'null'
        id:
          type:
            - string
            - 'null'
        type:
          enum:
            - context_compaction
          type: string
      required:
        - type
      type: object
    ItemReferenceItem:
      description: A reference to a previous response item by ID
      example:
        id: msg-abc123
        type: item_reference
      properties:
        id:
          type: string
        type:
          enum:
            - item_reference
          type: string
      required:
        - type
        - id
      type: object
    AdditionalToolsItem:
      description: Additional tools made available to the model at this point in the input
      example:
        role: developer
        tools:
          - name: get_weather
            parameters:
              properties:
                location:
                  type: string
              type: object
            type: function
        type: additional_tools
      properties:
        id:
          type:
            - string
            - 'null'
        role:
          enum:
            - unknown
            - user
            - assistant
            - system
            - critic
            - discriminator
            - developer
            - tool
          type: string
        tools:
          items:
            anyOf:
              - allOf:
                  - $ref: '#/components/schemas/FunctionTool'
                  - properties: {}
                    type: object
                description: Function tool definition
                example:
                  description: Get the current weather in a location
                  name: get_weather
                  parameters:
                    properties:
                      location:
                        description: The city and state
                        type: string
                      unit:
                        enum:
                          - celsius
                          - fahrenheit
                        type: string
                    required:
                      - location
                    type: object
                  type: function
              - $ref: '#/components/schemas/Preview_WebSearchServerTool'
              - $ref: '#/components/schemas/Preview_20250311_WebSearchServerTool'
              - $ref: '#/components/schemas/Legacy_WebSearchServerTool'
              - $ref: '#/components/schemas/WebSearchServerTool'
              - $ref: '#/components/schemas/FileSearchServerTool'
              - $ref: '#/components/schemas/ComputerUseServerTool'
              - $ref: '#/components/schemas/CodeInterpreterServerTool'
              - $ref: '#/components/schemas/McpServerTool'
              - $ref: '#/components/schemas/ImageGenerationServerTool'
              - $ref: '#/components/schemas/CodexLocalShellTool'
              - $ref: '#/components/schemas/ShellServerTool'
              - $ref: '#/components/schemas/ApplyPatchServerTool'
              - $ref: '#/components/schemas/CustomTool'
              - $ref: '#/components/schemas/NamespaceTool'
              - $ref: '#/components/schemas/AdvisorServerTool_OpenRouter'
              - $ref: '#/components/schemas/SubagentServerTool_OpenRouter'
              - $ref: '#/components/schemas/DatetimeServerTool'
              - $ref: '#/components/schemas/FilesServerTool'
              - $ref: '#/components/schemas/FusionServerTool_OpenRouter'
              - $ref: '#/components/schemas/ImageGenerationServerTool_OpenRouter'
              - $ref: '#/components/schemas/SearchModelsServerTool_OpenRouter'
              - $ref: '#/components/schemas/WebFetchServerTool'
              - $ref: '#/components/schemas/WebSearchServerTool_OpenRouter'
              - $ref: '#/components/schemas/ApplyPatchServerTool_OpenRouter'
              - $ref: '#/components/schemas/BashServerTool'
              - $ref: '#/components/schemas/ShellServerTool_OpenRouter'
              - additionalProperties: {}
                properties:
                  type:
                    type: string
                required:
                  - type
                type: object
          type: array
        type:
          enum:
            - additional_tools
          type: string
      required:
        - type
        - role
        - tools
      type: object
    AgentMessageItem:
      additionalProperties: {}
      description: A message routed between agents in a multi-agent session
      example:
        author: /root/worker
        content:
          - text: Task complete.
            type: input_text
        recipient: /root
        type: agent_message
      properties:
        agent:
          additionalProperties: {}
          properties:
            agent_name:
              type: string
          required:
            - agent_name
          type:
            - object
            - 'null'
        author:
          type: string
        content:
          items:
            oneOf:
              - $ref: '#/components/schemas/InputText'
              - allOf:
                  - $ref: '#/components/schemas/InputImage'
                  - properties: {}
                    type: object
                description: Image input content item
                example:
                  detail: auto
                  image_url: https://example.com/image.jpg
                  type: input_image
              - additionalProperties: {}
                properties:
                  encrypted_content:
                    type: string
                  type:
                    enum:
                      - encrypted_content
                    type: string
                required:
                  - type
                  - encrypted_content
                type: object
          type: array
        id:
          type:
            - string
            - 'null'
        recipient:
          type: string
        type:
          enum:
            - agent_message
          type: string
      required:
        - type
        - author
        - recipient
        - content
      type: object
    ContextCompressionEngine:
      description: The compression engine to use. Defaults to "middle-out".
      enum:
        - middle-out
      example: middle-out
      type: string
    PDFParserOptions:
      description: Options for PDF parsing.
      example:
        engine: cloudflare-ai
      properties:
        engine:
          $ref: '#/components/schemas/PDFParserEngine'
      type: object
    WebSearchEngine:
      description: The search engine to use for web search.
      enum:
        - native
        - exa
        - firecrawl
        - parallel
        - perplexity
      example: exa
      type: string
    WebSearchMode:
      description: >-
        Engine-native search mode. Exa supports instant, fast, auto (default),
        deep-lite, deep, and deep-reasoning. Parallel supports turbo, fast,
        basic (default), and advanced. Modes unsupported by the selected engine
        are ignored.
      enum:
        - instant
        - fast
        - auto
        - deep-lite
        - deep
        - deep-reasoning
        - turbo
        - basic
        - advanced
      example: auto
      type: string
    WebSearchUserLocation:
      description: User location information for web search
      example:
        city: San Francisco
        country: USA
        region: California
        timezone: America/Los_Angeles
        type: approximate
      properties:
        city:
          type:
            - string
            - 'null'
        country:
          type:
            - string
            - 'null'
        region:
          type:
            - string
            - 'null'
        timezone:
          type:
            - string
            - 'null'
        type:
          enum:
            - approximate
          type: string
      type:
        - object
        - 'null'
    InputText:
      description: Text input content item
      example:
        text: Hello, how can I help you?
        type: input_text
      properties:
        prompt_cache_breakpoint:
          $ref: '#/components/schemas/PromptCacheBreakpoint'
        text:
          type: string
        type:
          enum:
            - input_text
          type: string
      required:
        - type
        - text
      type: object
    InputImage:
      description: Image input content item
      example:
        detail: auto
        image_url: https://example.com/image.jpg
        type: input_image
      properties:
        detail:
          enum:
            - auto
            - high
            - low
            - original
          type: string
        image_url:
          type:
            - string
            - 'null'
        type:
          enum:
            - input_image
          type: string
      required:
        - type
        - detail
      type: object
    InputFile:
      description: File input content item
      example:
        file_id: file-abc123
        filename: document.pdf
        type: input_file
      properties:
        file_data:
          type: string
        file_id:
          type:
            - string
            - 'null'
        file_url:
          type: string
        filename:
          type: string
        type:
          enum:
            - input_file
          type: string
      required:
        - type
      type: object
    ProviderName:
      enum:
        - Modal
        - AkashML
        - AI21
        - AionLabs
        - Alibaba
        - Ambient
        - Baidu
        - Amazon Bedrock
        - Amazon Nova
        - Anthropic
        - Arcee AI
        - AtlasCloud
        - Avian
        - Azure
        - BaseTen
        - BytePlus
        - Black Forest Labs
        - Cerebras
        - Chutes
        - Cirrascale
        - Claude Platform on AWS
        - Clarifai
        - Cloudflare
        - Cohere
        - CoreWeave
        - Crucible
        - Crusoe
        - Darkbloom
        - Databricks
        - Decart
        - Deepgram
        - DeepInfra
        - DeepSeek
        - DekaLLM
        - DigitalOcean
        - Featherless
        - Fireworks
        - Fish Audio
        - Friendli
        - GMICloud
        - Google
        - Google AI Studio
        - Groq
        - HeyGen
        - Inception
        - Inceptron
        - InferenceNet
        - Ionstream
        - Infermatic
        - Io Net
        - Inferact vLLM
        - Inflection
        - Liquid
        - Mara
        - Mancer 2
        - Meta
        - Minimax
        - ModelRun
        - Mistral
        - Modular
        - Moonshot AI
        - Morph
        - VoyageAI by MongoDB
        - NCompass
        - Nebius
        - Nex AGI
        - NextBit
        - Novita
        - Nvidia
        - OpenAI
        - OpenInference
        - Parasail
        - Poolside
        - Perceptron
        - Perplexity
        - Phala
        - Recraft
        - Reka
        - Relace
        - Sail Research
        - Sakana AI
        - SambaNova
        - Seed
        - SiliconFlow
        - Sourceful
        - StepFun
        - Stealth
        - StreamLake
        - Switchpoint
        - Tencent
        - Tenstorrent
        - Thinking Machines
        - Together
        - Upstage
        - Venice
        - Wafer
        - WandB
        - Quiver
        - Krea
        - Runway
        - Xiaomi
        - xAI
        - Z.AI
        - FakeProvider
      example: OpenAI
      type: string
    PreferredMaxLatency:
      anyOf:
        - format: double
          type: number
        - $ref: '#/components/schemas/PercentileLatencyCutoffs'
        - type: 'null'
      description: >-
        Preferred maximum latency (in seconds). Can be a number (applies to p50)
        or an object with percentile-specific cutoffs. Endpoints above the
        threshold(s) may still be used, but are deprioritized in routing. When
        using fallback models, this may cause a fallback model to be used
        instead of the primary model if it meets the threshold.
      example: 5
    PreferredMinThroughput:
      anyOf:
        - format: double
          type: number
        - $ref: '#/components/schemas/PercentileThroughputCutoffs'
        - type: 'null'
      description: >-
        Preferred minimum throughput (in tokens per second). Can be a number
        (applies to p50) or an object with percentile-specific cutoffs.
        Endpoints below the threshold(s) may still be used, but are
        deprioritized in routing. When using fallback models, this may cause a
        fallback model to be used instead of the primary model if it meets the
        threshold.
      example: 100
    Quantization:
      enum:
        - int4
        - int8
        - fp4
        - mxfp4
        - nvfp4
        - fp6
        - fp8
        - mxfp8
        - fp16
        - bf16
        - fp32
        - unknown
      example: fp16
      type: string
    ProviderSort:
      description: The provider sorting strategy (price, throughput, latency)
      enum:
        - price
        - throughput
        - latency
        - exacto
      example: price
      type: string
    ProviderSortConfig:
      description: The provider sorting strategy (price, throughput, latency)
      example:
        by: price
        partition: model
      properties:
        by:
          description: The provider sorting strategy (price, throughput, latency)
          enum:
            - price
            - throughput
            - latency
            - exacto
            - null
          example: price
          type:
            - string
            - 'null'
        partition:
          description: >-
            Partitioning strategy for sorting: "model" (default) groups
            endpoints by model before sorting (fallback models remain
            fallbacks), "none" sorts all endpoints together regardless of model.
          enum:
            - model
            - none
            - null
          example: model
          type:
            - string
            - 'null'
      type: object
    BaseReasoningConfig:
      example:
        effort: medium
        summary: auto
      properties:
        context:
          $ref: '#/components/schemas/ReasoningContext'
        effort:
          $ref: '#/components/schemas/ReasoningEffort'
        mode:
          $ref: '#/components/schemas/ReasoningMode'
        summary:
          $ref: '#/components/schemas/ReasoningSummaryVerbosity'
      type:
        - object
        - 'null'
    StopServerToolsWhenCondition:
      description: A single condition that, when met, halts the server-tool agent loop.
      discriminator:
        mapping:
          finish_reason_is:
            $ref: '#/components/schemas/StopServerToolsWhenFinishReasonIs'
          has_tool_call:
            $ref: '#/components/schemas/StopServerToolsWhenHasToolCall'
          max_cost:
            $ref: '#/components/schemas/StopServerToolsWhenMaxCost'
          max_tokens_used:
            $ref: '#/components/schemas/StopServerToolsWhenMaxTokensUsed'
          step_count_is:
            $ref: '#/components/schemas/StopServerToolsWhenStepCountIs'
        propertyName: type
      example:
        step_count: 5
        type: step_count_is
      oneOf:
        - $ref: '#/components/schemas/StopServerToolsWhenStepCountIs'
        - $ref: '#/components/schemas/StopServerToolsWhenHasToolCall'
        - $ref: '#/components/schemas/StopServerToolsWhenMaxTokensUsed'
        - $ref: '#/components/schemas/StopServerToolsWhenMaxCost'
        - $ref: '#/components/schemas/StopServerToolsWhenFinishReasonIs'
    TextConfig:
      description: Text output configuration including format and verbosity
      example:
        format:
          type: text
        verbosity: medium
      properties:
        format:
          $ref: '#/components/schemas/Formats'
        verbosity:
          enum:
            - high
            - low
            - medium
            - null
          type:
            - string
            - 'null'
      type: object
    ToolChoiceAllowed:
      description: Constrains the model to a pre-defined set of allowed tools
      example:
        mode: auto
        tools:
          - name: get_weather
            type: function
        type: allowed_tools
      properties:
        mode:
          anyOf:
            - enum:
                - auto
              type: string
            - enum:
                - required
              type: string
        tools:
          items:
            additionalProperties: {}
            type: object
          type: array
        type:
          enum:
            - allowed_tools
          type: string
      required:
        - type
        - mode
        - tools
      type: object
    WebSearchEngineEnum:
      description: >-
        Which search engine to use. "auto" (default) uses native if the provider
        supports it, otherwise Exa. "native" forces the provider's built-in
        search. "exa" forces the Exa search API. "firecrawl" uses Firecrawl
        (requires BYOK). "parallel" uses the Parallel search API. "perplexity"
        uses the Perplexity Search API (raw ranked results).
      enum:
        - native
        - exa
        - parallel
        - firecrawl
        - perplexity
        - auto
      example: auto
      type: string
    WebSearchDomainFilter:
      example:
        allowed_domains:
          - example.com
        blocked_domains:
          - spam.com
        excluded_domains:
          - spam.com
      properties:
        allowed_domains:
          items:
            type: string
          type:
            - array
            - 'null'
        blocked_domains:
          items:
            type: string
          type:
            - array
            - 'null'
        excluded_domains:
          items:
            type: string
          type:
            - array
            - 'null'
      type:
        - object
        - 'null'
    SearchContextSizeEnum:
      description: Size of the search context for web search tools
      enum:
        - low
        - medium
        - high
      example: medium
      type: string
    Preview_WebSearchUserLocation:
      example:
        city: San Francisco
        country: USA
        region: California
        timezone: America/Los_Angeles
        type: approximate
      properties:
        city:
          type:
            - string
            - 'null'
        country:
          type:
            - string
            - 'null'
        region:
          type:
            - string
            - 'null'
        timezone:
          type:
            - string
            - 'null'
        type:
          enum:
            - approximate
          type: string
      required:
        - type
      type:
        - object
        - 'null'
    CompoundFilter:
      description: A compound filter that combines multiple comparison or compound filters
      example:
        filters:
          - key: author
            type: eq
            value: Alice
        type: and
      properties:
        filters:
          items:
            additionalProperties: {}
            type: object
          type: array
        type:
          enum:
            - and
            - or
          type: string
      required:
        - type
        - filters
      type: object
    NamespaceFunctionTool:
      description: A function tool grouped inside a namespace tool
      example:
        name: spawn_agent
        type: function
      properties:
        allowed_callers:
          items:
            enum:
              - direct
              - programmatic
            type: string
          type:
            - array
            - 'null'
        defer_loading:
          type: boolean
        description:
          type:
            - string
            - 'null'
        name:
          type: string
        output_schema:
          additionalProperties: {}
          type:
            - object
            - 'null'
        parameters:
          additionalProperties: {}
          type:
            - object
            - 'null'
        strict:
          type:
            - boolean
            - 'null'
        type:
          enum:
            - function
          type: string
      required:
        - type
        - name
      type: object
    AdvisorServerToolConfig:
      description: Configuration for one openrouter:advisor server tool entry.
      example:
        model: ~anthropic/claude-opus-latest
        name: reviewer
      properties:
        forward_transcript:
          description: >-
            When true, the full parent conversation is forwarded to the advisor
            so it sees the same context the executor does (and the tool-call
            `prompt`, if given, is appended as a final user turn). When false or
            omitted, the advisor receives only the `prompt` the executor passes
            in the tool call.
          example: false
          type: boolean
        instructions:
          description: >-
            System instructions for the advisor sub-agent. When omitted, the
            advisor responds with no system prompt of its own.
          example: You are a senior staff engineer. Give a focused, decisive plan.
          type: string
        max_completion_tokens:
          description: >-
            Maximum number of output tokens (including reasoning) the advisor
            may produce. When omitted, the provider's default applies.
          example: 2048
          type: integer
        max_tool_calls:
          description: >-
            Maximum number of tool-calling steps the advisor sub-agent may take
            during its agentic loop. Capped at 25. Only relevant when the
            advisor is given tools.
          example: 5
          maximum: 25
          minimum: 1
          type: integer
        model:
          description: >-
            Slug of the advisor model to consult (any OpenRouter model). When
            omitted, the executor can choose it via the tool call's `model`
            argument; if neither is set, the model from the outer API request is
            used. The advisor tool itself cannot be the advisor model.
          example: ~anthropic/claude-opus-latest
          type: string
        name:
          description: >-
            Optional name for this advisor. The model sees one tool per named
            advisor (and one default for an unnamed entry). Names must be unique
            across advisor entries. Letters, digits, spaces, underscores, and
            dashes; trimmed; 1–64 chars.
          example: reviewer
          maxLength: 64
          minLength: 1
          pattern: ^[a-zA-Z0-9 _-]+$
          type: string
        reasoning:
          $ref: '#/components/schemas/AdvisorReasoning'
        stream:
          description: >-
            When true, the advisor's advice streams incrementally as it is
            produced. In the Responses API this emits
            `response.output_text.delta` events targeting the advisor output
            item; the final `advice` field is still set on the completed item.
            Has no effect on the Chat Completions API (where the advice arrives
            only as the final tool result). When false or omitted, the advice
            arrives only as the final result.
          example: false
          type: boolean
        temperature:
          description: >-
            Sampling temperature forwarded to the advisor call. When omitted,
            the provider's default applies.
          example: 0.7
          format: double
          type: number
        tools:
          description: >-
            Tools the advisor sub-agent may use while forming its advice. The
            advisor runs as an agentic sub-agent over these tools, then returns
            its text. Only OpenRouter server tools are supported — function
            tools are rejected — and the list must not include the advisor tool
            itself.
          items:
            $ref: '#/components/schemas/AdvisorNestedTool'
          type: array
      type: object
    SubagentServerToolConfig:
      description: Configuration for one openrouter:subagent server tool entry.
      example:
        model: ~anthropic/claude-haiku-latest
        name: summarizer
      properties:
        instructions:
          description: >-
            System instructions for the subagent. When omitted, the subagent
            responds with no system prompt of its own.
          example: >-
            You are a fast, focused worker. Complete the task exactly as
            described.
          type: string
        max_completion_tokens:
          description: >-
            Maximum number of output tokens (including reasoning) the subagent
            may produce. When omitted, the provider's default applies.
          example: 2048
          type: integer
        max_tool_calls:
          description: >-
            Maximum number of tool-calling steps the subagent may take during
            its agentic loop. Capped at 25. Only relevant when the subagent is
            given tools. Accepted and validated but not yet enforced on the
            subagent call.
          example: 5
          maximum: 25
          minimum: 1
          type: integer
        model:
          description: >-
            Slug of the model that executes delegated tasks (any OpenRouter
            model). Typically a smaller, cheaper, faster model than the one
            delegating. When omitted, the model from the outer API request is
            used. The subagent tool itself cannot be the subagent model.
          example: ~anthropic/claude-haiku-latest
          type: string
        name:
          description: >-
            Optional name for this subagent. The model sees one tool per named
            subagent (and one default for an unnamed entry). Names must be
            unique across subagent entries. Letters, digits, spaces,
            underscores, and dashes; trimmed; 1–64 chars.
          example: summarizer
          maxLength: 64
          minLength: 1
          pattern: ^[a-zA-Z0-9 _-]+$
          type: string
        reasoning:
          $ref: '#/components/schemas/SubagentReasoning'
        temperature:
          description: >-
            Sampling temperature forwarded to the subagent call. When omitted,
            the provider's default applies.
          example: 0.7
          format: double
          type: number
        tools:
          description: >-
            Tools the subagent may use while executing a delegated task. The
            subagent runs as an agentic sub-agent over these tools, then returns
            its outcome. Only OpenRouter server tools are supported — function
            tools are rejected — and the list must not include the subagent tool
            itself.
          items:
            $ref: '#/components/schemas/SubagentNestedTool'
          type: array
      type: object
    DatetimeServerToolConfig:
      description: Configuration for the openrouter:datetime server tool
      example:
        timezone: America/New_York
      properties:
        timezone:
          description: IANA timezone name (e.g. "America/New_York"). Defaults to UTC.
          example: America/New_York
          type: string
      type: object
    FilesServerToolConfig:
      description: Configuration for the openrouter:files server tool
      example: {}
      properties: {}
      type: object
    FusionServerToolConfig:
      description: Configuration for the openrouter:fusion server tool.
      example:
        analysis_models:
          - ~anthropic/claude-opus-latest
          - ~openai/gpt-latest
          - ~google/gemini-pro-latest
      properties:
        analysis_models:
          description: >-
            Slugs of models to run in parallel as the analysis panel. Each model
            receives the user prompt with openrouter:web_search and
            openrouter:web_fetch enabled, then an analyst model summarizes the
            collective output into structured analysis JSON. Capped at 8 models
            to bound cost amplification. Defaults to the Quality preset from
            /labs/fusion.
          example:
            - ~anthropic/claude-opus-latest
            - ~openai/gpt-latest
            - ~google/gemini-pro-latest
          items:
            type: string
          maxItems: 8
          minItems: 1
          type: array
        cache_control:
          $ref: '#/components/schemas/AnthropicCacheControlDirective'
        max_completion_tokens:
          description: >-
            Maximum number of output tokens (including reasoning tokens) each
            panelist and the analyst model may produce per inner call. Controls
            the total output budget so reasoning-heavy models like GPT-5.5 do
            not exhaust their token allowance before producing visible text.
            Defaults to 16000 when omitted.
          example: 16384
          type: integer
        max_tool_calls:
          description: >-
            Maximum number of tool-calling steps each panelist (analysis model)
            and the analyst model may take during their agentic web-research
            loop. Models with web_search/web_fetch enabled iterate until they
            produce a text response or hit this ceiling. Defaults to 4. Capped
            at 16.
          example: 12
          maximum: 16
          minimum: 1
          type: integer
        model:
          description: >-
            Slug of the analyst model that produces the structured analysis
            JSON. Defaults to the model used in the outer API request.
          example: ~anthropic/claude-opus-latest
          type: string
        reasoning:
          description: >-
            Reasoning configuration forwarded to panelist and analyst inner
            calls. Use this to control reasoning effort and token budget for
            models that support extended thinking.
          properties:
            effort:
              description: Reasoning effort level for panelist and analyst inner calls.
              enum:
                - max
                - xhigh
                - high
                - medium
                - low
                - minimal
                - none
              type: string
            max_tokens:
              description: >-
                Maximum number of reasoning tokens each panelist and analyst
                model may use. Helps bound cost when models allocate too much
                budget to chain-of-thought.
              type: integer
          type: object
        temperature:
          description: >-
            Temperature forwarded to panelist inner calls. The analyst always
            runs at temperature 0 regardless of this value. When omitted, the
            provider's default applies.
          example: 0.7
          format: double
          type: number
        tools:
          description: >-
            Server tools available to panelist and analyst inner calls. Each
            entry uses the same `{ type, parameters? }` shorthand as the outer
            Chat Completions request. When omitted, defaults to `[{ type:
            "openrouter:web_search" }, { type: "openrouter:web_fetch" }]`. Pass
            an empty array to disable tools entirely (panelists answer from
            parametric knowledge only).
          example:
            - parameters:
                excluded_domains:
                  - example.com
              type: openrouter:web_search
            - type: openrouter:web_fetch
          items:
            properties:
              parameters:
                additionalProperties: {}
                description: >-
                  Optional configuration forwarded as the tool's `parameters`
                  object.
                type: object
              type:
                description: >-
                  Server tool type identifier (e.g. "openrouter:web_search",
                  "openrouter:web_fetch").
                type: string
            required:
              - type
            type: object
          maxItems: 8
          type: array
      type: object
    ImageGenerationServerToolConfig:
      additionalProperties:
        anyOf:
          - type: string
          - format: double
            type: number
          - items: {}
            type: array
      description: >-
        Configuration for the openrouter:image_generation server tool. Accepts
        all image_config params (aspect_ratio, quality, size, background,
        output_format, output_compression, moderation, etc.) plus a model field.
      example:
        aspect_ratio: '16:9'
        model: openai/gpt-5-image
        quality: high
      properties:
        model:
          description: >-
            Which image generation model to use (e.g. "openai/gpt-5-image").
            Defaults to "openai/gpt-5-image".
          example: openai/gpt-5-image
          type: string
      type: object
    SearchModelsServerToolConfig:
      description: Configuration for the openrouter:experimental__search_models server tool
      example:
        max_results: 5
      properties:
        max_results:
          description: Maximum number of models to return. Defaults to 5, max 20.
          example: 5
          type: integer
      type: object
    WebFetchServerToolConfig:
      description: Configuration for the openrouter:web_fetch server tool
      example:
        max_content_tokens: 100000
        max_uses: 10
      properties:
        allowed_domains:
          description: Only fetch from these domains.
          items:
            type: string
          type: array
        blocked_domains:
          description: Never fetch from these domains.
          items:
            type: string
          type: array
        engine:
          $ref: '#/components/schemas/WebFetchEngineEnum'
        max_content_tokens:
          description: >-
            Maximum content length in approximate tokens. Content exceeding this
            limit is truncated.
          example: 100000
          type: integer
        max_uses:
          description: >-
            Maximum number of web fetches per request. Once exceeded, the tool
            returns an error.
          example: 10
          type: integer
      type: object
    WebSearchServerToolConfig:
      description: Configuration for the openrouter:web_search server tool
      example:
        max_results: 5
        search_context_size: medium
      properties:
        allowed_domains:
          description: >-
            Limit search results to these domains. Supported by Exa, Firecrawl,
            Parallel, Perplexity, and most native providers (Anthropic, OpenAI,
            xAI). Cannot be used with excluded_domains.
          items:
            type: string
          type: array
        engine:
          $ref: '#/components/schemas/WebSearchEngineEnum'
        excluded_domains:
          description: >-
            Exclude search results from these domains. Supported by Exa,
            Firecrawl, Parallel, Perplexity, Anthropic, OpenAI, and xAI. Cannot
            be used with allowed_domains.
          items:
            type: string
          type: array
        max_characters:
          description: >-
            Exact maximum number of characters of content per search result.
            Applies to the Exa, Parallel, and Perplexity engines; ignored with
            native provider search and Firecrawl. For Exa, caps highlight
            content per result. For Parallel, caps excerpt content per result
            (default 1,500 when omitted). For Perplexity, maps to the native
            `max_tokens_per_page` parameter (converted from characters to
            tokens) and trims the response to the exact character cap. When both
            `max_characters` and `search_context_size` are set, `max_characters`
            takes precedence. When omitted, falls back to `search_context_size`
            mapping (Exa) or engine defaults (Parallel, Perplexity).
          example: 2000
          type: integer
        max_results:
          description: >-
            Maximum number of search results to return per search call. Defaults
            to 5. Applies to Exa, Firecrawl, Parallel, and Perplexity engines;
            ignored with native provider search. Perplexity supports a maximum
            of 20; values above 20 are clamped.
          example: 5
          type: integer
        max_total_results:
          description: >-
            Maximum total number of search results across all search calls in a
            single request. Once this limit is reached, the tool will stop
            returning new results. Useful for controlling cost and context size
            in agentic loops. Defaults to 50 when not specified.
          example: 50
          type: integer
        max_uses:
          description: >-
            Maximum number of web searches the model may perform in a single
            request. Once reached, further search calls return an error result
            instead of executing. Applies to the Exa, Firecrawl, Parallel, and
            Perplexity engines. With native provider search, forwarded only to
            Anthropic (as `max_uses`); other native search providers have no
            equivalent parameter and ignore it.
          example: 3
          type: integer
        mode:
          $ref: '#/components/schemas/WebSearchMode'
        search_context_size:
          $ref: '#/components/schemas/SearchQualityLevel'
        user_location:
          $ref: '#/components/schemas/WebSearchUserLocationServerTool'
      type: object
    ApplyPatchServerToolConfig:
      description: Configuration for the openrouter:apply_patch server tool
      example:
        engine: auto
      properties:
        engine:
          $ref: '#/components/schemas/ApplyPatchEngineEnum'
      type: object
    BashServerToolConfig:
      description: Configuration for the openrouter:bash server tool
      example:
        environment:
          type: container_auto
      properties:
        engine:
          $ref: '#/components/schemas/BashServerToolEngine'
        environment:
          $ref: '#/components/schemas/BashServerToolEnvironment'
        sleep_after_seconds:
          $ref: '#/components/schemas/SandboxSleepAfterSeconds'
      type: object
    ShellServerToolConfig:
      description: Configuration for the openrouter:shell server tool
      example:
        engine: openrouter
        environment:
          type: container_auto
      properties:
        engine:
          $ref: '#/components/schemas/ShellServerToolEngine'
        environment:
          $ref: '#/components/schemas/ShellServerToolEnvironment'
        sleep_after_seconds:
          $ref: '#/components/schemas/SandboxSleepAfterSeconds'
      type: object
    ResponsesErrorField:
      description: Error information returned from the API
      example:
        code: rate_limit_exceeded
        message: Rate limit exceeded. Please try again later.
      properties:
        code:
          enum:
            - server_error
            - rate_limit_exceeded
            - invalid_prompt
            - vector_store_timeout
            - invalid_image
            - invalid_image_format
            - invalid_base64_image
            - invalid_image_url
            - image_too_large
            - image_too_small
            - image_parse_error
            - image_content_policy_violation
            - invalid_image_mode
            - image_file_too_large
            - unsupported_image_media_type
            - empty_image_file
            - failed_to_download_image
            - image_file_not_found
            - bio_policy
            - data_residency_mismatch
          type: string
        message:
          type: string
      required:
        - code
        - message
      type:
        - object
        - 'null'
    IncompleteDetails:
      example:
        reason: max_output_tokens
      properties:
        reason:
          enum:
            - max_output_tokens
            - content_filter
          type: string
      type:
        - object
        - 'null'
    BaseInputs:
      anyOf:
        - type: string
        - items:
            anyOf:
              - properties:
                  content:
                    anyOf:
                      - items:
                          discriminator:
                            mapping:
                              input_audio:
                                $ref: '#/components/schemas/InputAudio'
                              input_file:
                                $ref: '#/components/schemas/InputFile'
                              input_image:
                                $ref: '#/components/schemas/InputImage'
                              input_text:
                                $ref: '#/components/schemas/InputText'
                            propertyName: type
                          oneOf:
                            - $ref: '#/components/schemas/InputText'
                            - $ref: '#/components/schemas/InputImage'
                            - $ref: '#/components/schemas/InputFile'
                            - $ref: '#/components/schemas/InputAudio'
                        type: array
                      - type: string
                  phase:
                    anyOf:
                      - enum:
                          - commentary
                        type: string
                      - enum:
                          - final_answer
                        type: string
                      - type: 'null'
                  role:
                    anyOf:
                      - enum:
                          - user
                        type: string
                      - enum:
                          - system
                        type: string
                      - enum:
                          - assistant
                        type: string
                      - enum:
                          - developer
                        type: string
                  type:
                    enum:
                      - message
                    type: string
                required:
                  - role
                  - content
                type: object
              - $ref: '#/components/schemas/OpenAIResponseInputMessageItem'
              - $ref: '#/components/schemas/OpenAIResponseFunctionToolCallOutput'
              - $ref: '#/components/schemas/OpenAIResponseFunctionToolCall'
              - $ref: '#/components/schemas/OutputItemImageGenerationCall'
              - $ref: '#/components/schemas/OutputMessage'
              - $ref: '#/components/schemas/OpenAIResponseCustomToolCall'
              - $ref: '#/components/schemas/OpenAIResponseCustomToolCallOutput'
              - $ref: '#/components/schemas/ApplyPatchCallItem'
              - $ref: '#/components/schemas/ApplyPatchCallOutputItem'
          type: array
        - type: 'null'
      example:
        - content: What is the weather today?
          role: user
    OutputItemApplyPatchCall:
      example:
        call_id: call_abc123
        id: apc_abc123
        operation:
          diff: |-
            @@ function main() {
            +  console.log("hi");
             }
          path: /src/main.ts
          type: update_file
        status: completed
        type: apply_patch_call
      properties:
        call_id:
          type: string
        created_by:
          type: string
        id:
          type: string
        operation:
          discriminator:
            mapping:
              create_file:
                $ref: '#/components/schemas/ApplyPatchCreateFileOperation'
              delete_file:
                $ref: '#/components/schemas/ApplyPatchDeleteFileOperation'
              update_file:
                $ref: '#/components/schemas/ApplyPatchUpdateFileOperation'
            propertyName: type
          oneOf:
            - $ref: '#/components/schemas/ApplyPatchCreateFileOperation'
            - $ref: '#/components/schemas/ApplyPatchUpdateFileOperation'
            - $ref: '#/components/schemas/ApplyPatchDeleteFileOperation'
        status:
          enum:
            - in_progress
            - completed
          type: string
        type:
          enum:
            - apply_patch_call
          type: string
      required:
        - type
        - id
        - call_id
        - operation
        - status
      type: object
    OutputItemCodeInterpreterCall:
      additionalProperties: {}
      example:
        code: print("hello")
        id: ci_abc123
        outputs:
          - logs: |
              hello
            type: logs
        status: completed
        type: code_interpreter_call
      properties:
        code:
          type:
            - string
            - 'null'
        container_id:
          type: string
        id:
          type: string
        outputs:
          items:
            discriminator:
              mapping:
                file:
                  $ref: '#/components/schemas/CodeInterpreterFileOutput'
                image:
                  $ref: '#/components/schemas/CodeInterpreterImageOutput'
                logs:
                  $ref: '#/components/schemas/CodeInterpreterLogsOutput'
              propertyName: type
            oneOf:
              - $ref: '#/components/schemas/CodeInterpreterLogsOutput'
              - $ref: '#/components/schemas/CodeInterpreterImageOutput'
              - $ref: '#/components/schemas/CodeInterpreterFileOutput'
          type:
            - array
            - 'null'
        status:
          enum:
            - in_progress
            - completed
            - incomplete
            - interpreting
            - failed
          type: string
        type:
          enum:
            - code_interpreter_call
          type: string
      required:
        - type
        - id
        - status
      type: object
    OutputItemCustomToolCall:
      example:
        call_id: call-abc123
        id: ctc-abc123
        input: |-
          *** Begin Patch
          *** End Patch
        name: apply_patch
        type: custom_tool_call
      properties:
        call_id:
          type: string
        id:
          type: string
        input:
          type: string
        name:
          type: string
        namespace:
          description: >-
            Namespace qualifier for tools registered as part of a namespace tool
            group (e.g. an MCP server)
          type: string
        type:
          enum:
            - custom_tool_call
          type: string
      required:
        - type
        - name
        - input
        - call_id
      type: object
    OutputItemFileSearchCall:
      example:
        id: filesearch-abc123
        queries:
          - machine learning algorithms
          - neural networks
        status: completed
        type: file_search_call
      properties:
        id:
          type: string
        queries:
          items:
            type: string
          type: array
        status:
          $ref: '#/components/schemas/WebSearchStatus'
        type:
          enum:
            - file_search_call
          type: string
      required:
        - type
        - id
        - queries
        - status
      type: object
    OutputItemFunctionCall:
      example:
        arguments: '{"location":"San Francisco","unit":"celsius"}'
        call_id: call-abc123
        id: call-abc123
        name: get_weather
        type: function_call
      properties:
        arguments:
          type: string
        call_id:
          type: string
        id:
          type: string
        name:
          type: string
        namespace:
          description: >-
            Namespace qualifier for tools registered as part of a namespace tool
            group (e.g. an MCP server)
          type: string
        status:
          anyOf:
            - enum:
                - completed
              type: string
            - enum:
                - incomplete
              type: string
            - enum:
                - in_progress
              type: string
        type:
          enum:
            - function_call
          type: string
      required:
        - type
        - name
        - arguments
        - call_id
      type: object
    OutputItemImageGenerationCall:
      example:
        id: imagegen-abc123
        result: >-
          iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==
        status: completed
        type: image_generation_call
      properties:
        id:
          type: string
        result:
          default: null
          type:
            - string
            - 'null'
        status:
          $ref: '#/components/schemas/ImageGenerationStatus'
        type:
          enum:
            - image_generation_call
          type: string
      required:
        - type
        - id
        - status
      type: object
    OutputMessage:
      example:
        content:
          - text: Hello! How can I help you today?
            type: output_text
        id: msg-abc123
        role: assistant
        status: completed
        type: message
      properties:
        content:
          items:
            anyOf:
              - $ref: '#/components/schemas/ResponseOutputText'
              - $ref: '#/components/schemas/OpenAIResponsesRefusalContent'
          type: array
        id:
          type: string
        phase:
          anyOf:
            - enum:
                - commentary
              type: string
            - enum:
                - final_answer
              type: string
            - type: 'null'
          description: >-
            The phase of an assistant message. Use `commentary` for an
            intermediate assistant message and `final_answer` for the final
            assistant message. For follow-up requests with models like
            `gpt-5.3-codex` and later, preserve and resend phase on all
            assistant messages. Omitting it can degrade performance. Not used
            for user messages.
        role:
          enum:
            - assistant
          type: string
        status:
          anyOf:
            - enum:
                - completed
              type: string
            - enum:
                - incomplete
              type: string
            - enum:
                - in_progress
              type: string
        type:
          enum:
            - message
          type: string
      required:
        - id
        - role
        - type
        - content
      type: object
    OutputItemReasoning:
      example:
        id: reasoning-abc123
        summary:
          - text: Analyzed the problem using first principles
            type: summary_text
        type: reasoning
      properties:
        content:
          items:
            $ref: '#/components/schemas/ReasoningTextContent'
          type: array
        encrypted_content:
          type:
            - string
            - 'null'
        id:
          type: string
        status:
          anyOf:
            - enum:
                - completed
              type: string
            - enum:
                - incomplete
              type: string
            - enum:
                - in_progress
              type: string
        summary:
          items:
            $ref: '#/components/schemas/ReasoningSummaryText'
          type: array
        type:
          enum:
            - reasoning
          type: string
      required:
        - type
        - id
        - summary
      type: object
    OutputItemWebSearchCall:
      additionalProperties: {}
      example:
        action:
          query: OpenAI API
          type: search
        id: search-abc123
        status: completed
        type: web_search_call
      properties:
        action:
          oneOf:
            - properties:
                queries:
                  items:
                    type: string
                  type: array
                query:
                  type: string
                sources:
                  items:
                    $ref: '#/components/schemas/WebSearchSource'
                  type: array
                type:
                  enum:
                    - search
                  type: string
              required:
                - type
                - query
              type: object
            - properties:
                type:
                  enum:
                    - open_page
                  type: string
                url:
                  type:
                    - string
                    - 'null'
              required:
                - type
              type: object
            - properties:
                pattern:
                  type: string
                type:
                  enum:
                    - find_in_page
                  type: string
                url:
                  type: string
              required:
                - type
                - pattern
                - url
              type: object
        id:
          type: string
        status:
          $ref: '#/components/schemas/WebSearchStatus'
        type:
          enum:
            - web_search_call
          type: string
      required:
        - type
        - id
        - status
      type: object
    ServiceTier:
      enum:
        - auto
        - default
        - flex
        - priority
        - scale
        - null
      example: default
      type:
        - string
        - 'null'
    OpenAIResponsesResponseStatus:
      enum:
        - completed
        - incomplete
        - in_progress
        - failed
        - cancelled
        - queued
      example: completed
      type: string
    Truncation:
      enum:
        - auto
        - disabled
        - null
      example: auto
      type:
        - string
        - 'null'
    OpenAIResponsesUsage:
      example:
        input_tokens: 100
        input_tokens_details:
          cached_tokens: 0
        output_tokens: 50
        output_tokens_details:
          reasoning_tokens: 0
        total_tokens: 150
      properties:
        input_tokens:
          type: integer
        input_tokens_details:
          properties:
            cache_write_tokens:
              type:
                - integer
                - 'null'
            cached_tokens:
              type: integer
          required:
            - cached_tokens
          type: object
        output_tokens:
          type: integer
        output_tokens_details:
          properties:
            reasoning_tokens:
              type: integer
          required:
            - reasoning_tokens
          type: object
        total_tokens:
          type: integer
      required:
        - input_tokens
        - input_tokens_details
        - output_tokens
        - output_tokens_details
        - total_tokens
      type: object
    RouterAttempt:
      example:
        model: openai/gpt-4o
        provider: OpenAI
        status: 200
      properties:
        model:
          type: string
        provider:
          type: string
        status:
          type: integer
      required:
        - provider
        - model
        - status
      type: object
    EndpointsMetadata:
      example:
        available:
          - model: openai/gpt-4o
            provider: OpenAI
            selected: true
        total: 3
      properties:
        available:
          items:
            $ref: '#/components/schemas/EndpointInfo'
          type: array
        total:
          type: integer
      required:
        - total
        - available
      type: object
    RouterParams:
      additionalProperties: {}
      example:
        version_group: anthropic/claude-sonnet-4
      properties:
        quality_floor:
          format: double
          type: number
        throughput_floor:
          format: double
          type: number
        version_group:
          type: string
      type: object
    PipelineStage:
      example:
        data:
          action: redacted
          engines:
            - presidio
          flagged: true
          matched_entity_types:
            - EMAIL
            - PHONE
        name: content-filter
        summary: PII redacted via Presidio (EMAIL, PHONE)
        type: guardrail
      properties:
        cost_usd:
          format: double
          type:
            - number
            - 'null'
        data:
          additionalProperties: {}
          type: object
        guardrail_id:
          type: string
        guardrail_scope:
          type: string
        name:
          type: string
        summary:
          type: string
        type:
          $ref: '#/components/schemas/PipelineStageType'
      required:
        - type
        - name
      type: object
    RoutingStrategy:
      enum:
        - direct
        - auto
        - free
        - latest
        - alias
        - fallback
        - pareto
        - bodybuilder
        - fusion
      example: direct
      type: string
    OutputApplyPatchCallItem:
      description: >-
        A native `apply_patch_call` output item matching OpenAI's Responses API
        shape. Emitted when the client requested the `apply_patch` shorthand.
      example:
        call_id: call_abc123
        id: apc_abc123
        operation:
          diff: |-
            @@ function main() {
            +  console.log("hi");
             }
          path: /src/main.ts
          type: update_file
        status: completed
        type: apply_patch_call
      properties:
        call_id:
          type: string
        id:
          type: string
        operation:
          $ref: '#/components/schemas/ApplyPatchCallOperation'
        status:
          $ref: '#/components/schemas/ApplyPatchCallStatus'
        type:
          enum:
            - apply_patch_call
          type: string
      required:
        - type
        - id
        - call_id
        - status
        - operation
      type: object
    OutputShellCallItem:
      description: >-
        A native `shell_call` output item matching OpenAI's Responses API shape.
        Emitted for the sandbox-backed `shell` tool.
      example:
        action:
          commands:
            - echo hello
          max_output_length: null
          timeout_ms: null
        call_id: call_abc123
        id: shc_abc123
        status: completed
        type: shell_call
      properties:
        action:
          properties:
            commands:
              items:
                type: string
              type: array
            max_output_length:
              type:
                - integer
                - 'null'
            timeout_ms:
              type:
                - integer
                - 'null'
          required:
            - commands
            - max_output_length
            - timeout_ms
          type: object
        call_id:
          type: string
        id:
          type: string
        status:
          $ref: '#/components/schemas/ShellCallStatus'
        type:
          enum:
            - shell_call
          type: string
      required:
        - type
        - id
        - call_id
        - status
      type: object
    OutputShellCallOutputItem:
      description: >-
        A native `shell_call_output` item matching OpenAI's Responses API shape.
        Carries per-command stdout, stderr, and the exit/timeout outcome.
      example:
        call_id: call_abc123
        id: sho_abc123
        output:
          - outcome:
              exit_code: 0
              type: exit
            stderr: ''
            stdout: |
              hello
        status: completed
        type: shell_call_output
      properties:
        call_id:
          type: string
        id:
          type: string
        max_output_length:
          type:
            - integer
            - 'null'
        output:
          items:
            properties:
              outcome:
                oneOf:
                  - properties:
                      exit_code:
                        type: integer
                      type:
                        enum:
                          - exit
                        type: string
                    required:
                      - type
                      - exit_code
                    type: object
                  - properties:
                      type:
                        enum:
                          - timeout
                        type: string
                    required:
                      - type
                    type: object
              stderr:
                type: string
              stdout:
                type: string
            required:
              - stdout
              - stderr
              - outcome
            type: object
          type: array
        status:
          $ref: '#/components/schemas/ShellCallStatus'
        type:
          enum:
            - shell_call_output
          type: string
      required:
        - type
        - id
        - call_id
        - status
        - output
      type: object
    ServerToolUseDetails:
      description: Usage for server-side tool execution (e.g., web search)
      example:
        tool_calls_executed: 2
        tool_calls_requested: 2
        web_search_requests: 2
      properties:
        tool_calls_executed:
          description: >-
            Number of OpenRouter server tool calls that executed and produced a
            result.
          type:
            - integer
            - 'null'
        tool_calls_requested:
          description: >-
            Total number of OpenRouter server-orchestrated tool calls the model
            requested, across all tool types. Provider-native tools (e.g. native
            web search) are not counted here.
          type:
            - integer
            - 'null'
        web_search_requests:
          description: >-
            Number of web searches performed by server-side tools. For
            server-orchestrated tool calls a web search is also counted in
            tool_calls_requested; provider-native web search may report
            web_search_requests only. Do not sum the two.
          type:
            - integer
            - 'null'
      type:
        - object
        - 'null'
    ErrorEvent:
      allOf:
        - $ref: '#/components/schemas/BaseErrorEvent'
        - properties: {}
          type: object
      description: Event emitted when an error occurs during streaming
      example:
        code: rate_limit_exceeded
        message: Rate limit exceeded. Please try again later.
        param: null
        sequence_number: 2
        type: error
    ApplyPatchCallOperationDiffDeltaEvent:
      description: >-
        Incremental chunk of `operation.diff` for an `apply_patch_call`. Matches
        OpenAI's streaming shape.
      example:
        delta: |
          +console.log("hi");
        item_id: apc_abc123
        output_index: 0
        sequence_number: 5
        type: response.apply_patch_call_operation_diff.delta
      properties:
        delta:
          type: string
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.apply_patch_call_operation_diff.delta
          type: string
      required:
        - type
        - item_id
        - output_index
        - delta
        - sequence_number
      type: object
    ApplyPatchCallOperationDiffDoneEvent:
      description: >-
        Emitted when `operation.diff` streaming completes for an
        `apply_patch_call`.
      example:
        diff: |
          @@
          +console.log("hi");
        item_id: apc_abc123
        output_index: 0
        sequence_number: 12
        type: response.apply_patch_call_operation_diff.done
      properties:
        diff:
          type: string
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.apply_patch_call_operation_diff.done
          type: string
      required:
        - type
        - item_id
        - output_index
        - diff
        - sequence_number
      type: object
    CodeInterpreterCallCodeDeltaEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesCodeInterpreterCallCodeDelta'
        - properties: {}
          type: object
      description: Incremental chunk of code being streamed for a `code_interpreter_call`.
      example:
        delta: print("hello")
        item_id: ci-abc123
        output_index: 0
        sequence_number: 3
        type: response.code_interpreter_call_code.delta
    CodeInterpreterCallCodeDoneEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesCodeInterpreterCallCodeDone'
        - properties: {}
          type: object
      description: Emitted when code streaming completes for a `code_interpreter_call`.
      example:
        code: print("hello")
        item_id: ci-abc123
        output_index: 0
        sequence_number: 8
        type: response.code_interpreter_call_code.done
    CodeInterpreterCallCompletedEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesCodeInterpreterCallCompleted'
        - properties: {}
          type: object
      description: Code interpreter call completed
      example:
        item_id: ci-abc123
        output_index: 0
        sequence_number: 10
        type: response.code_interpreter_call.completed
    CodeInterpreterCallInProgressEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesCodeInterpreterCallInProgress'
        - properties: {}
          type: object
      description: Code interpreter call in progress
      example:
        item_id: ci-abc123
        output_index: 0
        sequence_number: 2
        type: response.code_interpreter_call.in_progress
    CodeInterpreterCallInterpretingEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesCodeInterpreterCallInterpreting'
        - properties: {}
          type: object
      description: Code interpreter is executing the code
      example:
        item_id: ci-abc123
        output_index: 0
        sequence_number: 9
        type: response.code_interpreter_call.interpreting
    StreamEventsResponseCompleted:
      allOf:
        - $ref: '#/components/schemas/CompletedEvent'
        - properties:
            response:
              $ref: '#/components/schemas/OpenResponsesResult'
          type: object
      description: Event emitted when a response has completed successfully
      example:
        response:
          completed_at: null
          created_at: 1704067200
          error: null
          frequency_penalty: null
          id: resp-abc123
          incomplete_details: null
          instructions: null
          max_output_tokens: null
          metadata: null
          model: gpt-4
          object: response
          output: []
          parallel_tool_calls: true
          presence_penalty: null
          status: completed
          temperature: null
          tool_choice: auto
          tools: []
          top_p: null
        sequence_number: 10
        type: response.completed
    ContentPartAddedEvent:
      allOf:
        - $ref: '#/components/schemas/BaseContentPartAddedEvent'
        - properties:
            part:
              anyOf:
                - $ref: '#/components/schemas/ResponseOutputText'
                - $ref: '#/components/schemas/ReasoningTextContent'
                - $ref: '#/components/schemas/OpenAIResponsesRefusalContent'
          type: object
      description: Event emitted when a new content part is added to an output item
      example:
        content_index: 0
        item_id: item-1
        output_index: 0
        part:
          annotations: []
          text: ''
          type: output_text
        sequence_number: 3
        type: response.content_part.added
    ContentPartDoneEvent:
      allOf:
        - $ref: '#/components/schemas/BaseContentPartDoneEvent'
        - properties:
            part:
              anyOf:
                - $ref: '#/components/schemas/ResponseOutputText'
                - $ref: '#/components/schemas/ReasoningTextContent'
                - $ref: '#/components/schemas/OpenAIResponsesRefusalContent'
          type: object
      description: Event emitted when a content part is complete
      example:
        content_index: 0
        item_id: item-1
        output_index: 0
        part:
          annotations: []
          text: Hello! How can I help you?
          type: output_text
        sequence_number: 7
        type: response.content_part.done
    OpenResponsesCreatedEvent:
      allOf:
        - $ref: '#/components/schemas/CreatedEvent'
        - properties:
            response:
              $ref: '#/components/schemas/OpenResponsesResult'
          type: object
      description: Event emitted when a response is created
      example:
        response:
          completed_at: null
          created_at: 1704067200
          error: null
          frequency_penalty: null
          id: resp-abc123
          incomplete_details: null
          instructions: null
          max_output_tokens: null
          metadata: null
          model: gpt-4
          object: response
          output: []
          parallel_tool_calls: true
          presence_penalty: null
          status: in_progress
          temperature: null
          tool_choice: auto
          tools: []
          top_p: null
        sequence_number: 0
        type: response.created
    CustomToolCallInputDeltaEvent:
      allOf:
        - $ref: '#/components/schemas/BaseCustomToolCallInputDeltaEvent'
        - properties: {}
          type: object
      description: >-
        Event emitted when a custom tool call's freeform input is being
        streamed. Mirrors `response.function_call_arguments.delta` but for
        `custom` tools whose input is opaque text rather than JSON arguments.
      example:
        delta: '*** Begin Patch'
        item_id: item-1
        output_index: 0
        sequence_number: 4
        type: response.custom_tool_call_input.delta
    CustomToolCallInputDoneEvent:
      allOf:
        - $ref: '#/components/schemas/BaseCustomToolCallInputDoneEvent'
        - properties: {}
          type: object
      description: >-
        Event emitted when a custom tool call's freeform input streaming is
        complete. Mirrors `response.function_call_arguments.done` but for
        `custom` tools.
      example:
        input: |-
          *** Begin Patch
          *** End Patch
        item_id: item-1
        output_index: 0
        sequence_number: 6
        type: response.custom_tool_call_input.done
    DebugEvent:
      description: >-
        Debug event emitted when debug.echo_upstream_body is true. Contains the
        transformed upstream request body or timing milestones.
      example:
        debug:
          echo_upstream_body:
            messages: []
            model: anthropic/claude-sonnet-4
        sequence_number: 1
        type: response.debug
      properties:
        debug:
          properties:
            echo_upstream_body:
              additionalProperties: {}
              type: object
            timings:
              properties:
                epoch_ms:
                  type: integer
                event:
                  enum:
                    - adapter_request
                    - upstream_headers_received
                    - first_token_received
                    - upstream_body_ended
                  type: string
                start_ms:
                  type: integer
              required:
                - start_ms
                - event
                - epoch_ms
              type: object
          type: object
        sequence_number:
          type: integer
        type:
          enum:
            - response.debug
          type: string
      required:
        - type
        - debug
        - sequence_number
      type: object
    StreamEventsResponseFailed:
      allOf:
        - $ref: '#/components/schemas/FailedEvent'
        - properties:
            response:
              $ref: '#/components/schemas/OpenResponsesResult'
          type: object
      description: Event emitted when a response has failed
      example:
        response:
          completed_at: null
          created_at: 1704067200
          error: null
          frequency_penalty: null
          id: resp-abc123
          incomplete_details: null
          instructions: null
          max_output_tokens: null
          metadata: null
          model: gpt-4
          object: response
          output: []
          parallel_tool_calls: true
          presence_penalty: null
          status: failed
          temperature: null
          tool_choice: auto
          tools: []
          top_p: null
        sequence_number: 3
        type: response.failed
    FunctionCallArgsDeltaEvent:
      allOf:
        - $ref: '#/components/schemas/BaseFunctionCallArgsDeltaEvent'
        - properties: {}
          type: object
      description: Event emitted when function call arguments are being streamed
      example:
        delta: '{"city": "..."}'
        item_id: item-1
        output_index: 0
        sequence_number: 4
        type: response.function_call_arguments.delta
    FunctionCallArgsDoneEvent:
      allOf:
        - $ref: '#/components/schemas/BaseFunctionCallArgsDoneEvent'
        - properties: {}
          type: object
      description: Event emitted when function call arguments streaming is complete
      example:
        arguments: '{"city": "San Francisco", "units": "celsius"}'
        item_id: item-1
        name: get_weather
        output_index: 0
        sequence_number: 6
        type: response.function_call_arguments.done
    FusionCallAnalysisCompletedEvent:
      description: Emitted when the fusion analyst completes with the structured analysis.
      example:
        analysis:
          blind_spots: []
          consensus: []
          contradictions: []
          partial_coverage: []
          unique_insights: []
        item_id: st_fusion_abc
        output_index: 0
        sequence_number: 40
        type: response.fusion_call.analysis.completed
      properties:
        analysis:
          $ref: '#/components/schemas/FusionAnalysisResult'
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.fusion_call.analysis.completed
          type: string
      required:
        - type
        - analysis
        - output_index
        - item_id
        - sequence_number
      type: object
    FusionCallAnalysisInProgressEvent:
      description: >-
        Emitted when the fusion analyst starts producing the structured
        analysis.
      example:
        analyst_model: openai/gpt-5
        item_id: st_fusion_abc
        judge_model: openai/gpt-5
        output_index: 0
        sequence_number: 25
        type: response.fusion_call.analysis.in_progress
      properties:
        analyst_model:
          description: Slug of the model producing the structured analysis.
          type: string
        item_id:
          type: string
        judge_model:
          deprecated: true
          description: >-
            Deprecated alias of `analyst_model`, kept so existing consumers keep
            working. Always carries the same value. Use `analyst_model`.
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.fusion_call.analysis.in_progress
          type: string
      required:
        - type
        - analyst_model
        - judge_model
        - output_index
        - item_id
        - sequence_number
      type: object
    FusionCallCompletedEvent:
      description: Emitted when the openrouter:fusion tool call finishes.
      example:
        item_id: st_fusion_abc
        output_index: 0
        sequence_number: 41
        type: response.fusion_call.completed
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.fusion_call.completed
          type: string
      required:
        - type
        - output_index
        - item_id
        - sequence_number
      type: object
    FusionCallInProgressEvent:
      description: Emitted when an openrouter:fusion tool call begins executing.
      example:
        item_id: st_fusion_abc
        output_index: 0
        sequence_number: 3
        type: response.fusion_call.in_progress
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.fusion_call.in_progress
          type: string
      required:
        - type
        - output_index
        - item_id
        - sequence_number
      type: object
    FusionCallPanelAddedEvent:
      description: Emitted when a fusion analysis-panel model starts.
      example:
        item_id: st_fusion_abc
        model: openai/gpt-5
        output_index: 0
        sequence_number: 4
        type: response.fusion_call.panel.added
      properties:
        item_id:
          type: string
        model:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.fusion_call.panel.added
          type: string
      required:
        - type
        - model
        - output_index
        - item_id
        - sequence_number
      type: object
    FusionCallPanelCompletedEvent:
      description: Emitted when a fusion panel model finishes with its full content.
      example:
        content: Full panel response text...
        item_id: st_fusion_abc
        model: openai/gpt-5
        output_index: 0
        sequence_number: 20
        type: response.fusion_call.panel.completed
      properties:
        content:
          type: string
        item_id:
          type: string
        model:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.fusion_call.panel.completed
          type: string
      required:
        - type
        - model
        - content
        - output_index
        - item_id
        - sequence_number
      type: object
    FusionCallPanelDeltaEvent:
      description: Incremental content token from a fusion panel model.
      example:
        delta: Carbon taxes
        item_id: st_fusion_abc
        model: openai/gpt-5
        output_index: 0
        sequence_number: 5
        type: response.fusion_call.panel.delta
      properties:
        delta:
          type: string
        item_id:
          type: string
        model:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.fusion_call.panel.delta
          type: string
      required:
        - type
        - model
        - delta
        - output_index
        - item_id
        - sequence_number
      type: object
    FusionCallPanelFailedEvent:
      description: Emitted when a fusion panel model fails.
      example:
        error: Upstream provider error
        item_id: st_fusion_abc
        model: openai/gpt-5
        output_index: 0
        sequence_number: 18
        status_code: 502
        type: response.fusion_call.panel.failed
      properties:
        error:
          type: string
        item_id:
          type: string
        model:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        status_code:
          type: integer
        type:
          enum:
            - response.fusion_call.panel.failed
          type: string
      required:
        - type
        - model
        - error
        - output_index
        - item_id
        - sequence_number
      type: object
    FusionCallPanelReasoningDeltaEvent:
      description: Incremental reasoning token from a fusion panel model.
      example:
        delta: Considering both sides
        item_id: st_fusion_abc
        model: openai/gpt-5
        output_index: 0
        sequence_number: 6
        type: response.fusion_call.panel.reasoning.delta
      properties:
        delta:
          type: string
        item_id:
          type: string
        model:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.fusion_call.panel.reasoning.delta
          type: string
      required:
        - type
        - model
        - delta
        - output_index
        - item_id
        - sequence_number
      type: object
    ImageGenCallCompletedEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesImageGenCallCompleted'
        - properties: {}
          type: object
      description: Image generation call completed
      example:
        item_id: call-123
        output_index: 0
        sequence_number: 4
        type: response.image_generation_call.completed
    ImageGenCallGeneratingEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesImageGenCallGenerating'
        - properties: {}
          type: object
      description: Image generation call is generating
      example:
        item_id: call-123
        output_index: 0
        sequence_number: 2
        type: response.image_generation_call.generating
    ImageGenCallInProgressEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesImageGenCallInProgress'
        - properties: {}
          type: object
      description: Image generation call in progress
      example:
        item_id: call-123
        output_index: 0
        sequence_number: 1
        type: response.image_generation_call.in_progress
    ImageGenCallPartialImageEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesImageGenCallPartialImage'
        - properties: {}
          type: object
      description: Image generation call with partial image
      example:
        item_id: call-123
        output_index: 0
        partial_image_b64: base64encodedimage...
        partial_image_index: 0
        sequence_number: 3
        type: response.image_generation_call.partial_image
    OpenResponsesInProgressEvent:
      allOf:
        - $ref: '#/components/schemas/InProgressEvent'
        - properties:
            response:
              $ref: '#/components/schemas/OpenResponsesResult'
          type: object
      description: Event emitted when a response is in progress
      example:
        response:
          completed_at: null
          created_at: 1704067200
          error: null
          frequency_penalty: null
          id: resp-abc123
          incomplete_details: null
          instructions: null
          max_output_tokens: null
          metadata: null
          model: gpt-4
          object: response
          output: []
          parallel_tool_calls: true
          presence_penalty: null
          status: in_progress
          temperature: null
          tool_choice: auto
          tools: []
          top_p: null
        sequence_number: 1
        type: response.in_progress
    StreamEventsResponseIncomplete:
      allOf:
        - $ref: '#/components/schemas/IncompleteEvent'
        - properties:
            response:
              $ref: '#/components/schemas/OpenResponsesResult'
          type: object
      description: Event emitted when a response is incomplete
      example:
        response:
          completed_at: null
          created_at: 1704067200
          error: null
          frequency_penalty: null
          id: resp-abc123
          incomplete_details: null
          instructions: null
          max_output_tokens: null
          metadata: null
          model: gpt-4
          object: response
          output: []
          parallel_tool_calls: true
          presence_penalty: null
          status: incomplete
          temperature: null
          tool_choice: auto
          tools: []
          top_p: null
        sequence_number: 5
        type: response.incomplete
    StreamEventsResponseOutputItemAdded:
      allOf:
        - $ref: '#/components/schemas/OutputItemAddedEvent'
        - properties:
            item:
              $ref: '#/components/schemas/OutputItems'
          type: object
      description: Event emitted when a new output item is added to the response
      example:
        item:
          content: []
          id: item-1
          role: assistant
          status: in_progress
          type: message
        output_index: 0
        sequence_number: 2
        type: response.output_item.added
    StreamEventsResponseOutputItemDone:
      allOf:
        - $ref: '#/components/schemas/OutputItemDoneEvent'
        - properties:
            item:
              $ref: '#/components/schemas/OutputItems'
          type: object
      description: Event emitted when an output item is complete
      example:
        item:
          content:
            - annotations: []
              text: Hello! How can I help you?
              type: output_text
          id: item-1
          role: assistant
          status: completed
          type: message
        output_index: 0
        sequence_number: 8
        type: response.output_item.done
    AnnotationAddedEvent:
      allOf:
        - $ref: '#/components/schemas/BaseAnnotationAddedEvent'
        - properties: {}
          type: object
      description: Event emitted when a text annotation is added to output
      example:
        annotation:
          end_index: 7
          start_index: 0
          title: Example
          type: url_citation
          url: https://example.com
        annotation_index: 0
        content_index: 0
        item_id: item-1
        output_index: 0
        sequence_number: 5
        type: response.output_text.annotation.added
    TextDeltaEvent:
      allOf:
        - $ref: '#/components/schemas/BaseTextDeltaEvent'
        - properties:
            logprobs:
              items:
                $ref: '#/components/schemas/StreamLogprob'
              type: array
          type: object
      description: Event emitted when a text delta is streamed
      example:
        content_index: 0
        delta: Hello
        item_id: item-1
        logprobs: []
        output_index: 0
        sequence_number: 4
        type: response.output_text.delta
    TextDoneEvent:
      allOf:
        - $ref: '#/components/schemas/BaseTextDoneEvent'
        - properties:
            logprobs:
              items:
                $ref: '#/components/schemas/StreamLogprob'
              type: array
          type: object
      description: Event emitted when text streaming is complete
      example:
        content_index: 0
        item_id: item-1
        logprobs: []
        output_index: 0
        sequence_number: 6
        text: Hello! How can I help you?
        type: response.output_text.done
    ReasoningSummaryPartAddedEvent:
      allOf:
        - $ref: '#/components/schemas/BaseReasoningSummaryPartAddedEvent'
        - properties: {}
          type: object
      description: Event emitted when a reasoning summary part is added
      example:
        item_id: item-1
        output_index: 0
        part:
          text: ''
          type: summary_text
        sequence_number: 3
        summary_index: 0
        type: response.reasoning_summary_part.added
    ReasoningSummaryPartDoneEvent:
      allOf:
        - $ref: '#/components/schemas/BaseReasoningSummaryPartDoneEvent'
        - properties: {}
          type: object
      description: Event emitted when a reasoning summary part is complete
      example:
        item_id: item-1
        output_index: 0
        part:
          text: Analyzing the problem step by step to find the optimal solution.
          type: summary_text
        sequence_number: 7
        summary_index: 0
        type: response.reasoning_summary_part.done
    ReasoningSummaryTextDeltaEvent:
      allOf:
        - $ref: '#/components/schemas/BaseReasoningSummaryTextDeltaEvent'
        - properties: {}
          type: object
      description: Event emitted when reasoning summary text delta is streamed
      example:
        delta: Analyzing
        item_id: item-1
        output_index: 0
        sequence_number: 4
        summary_index: 0
        type: response.reasoning_summary_text.delta
    ReasoningSummaryTextDoneEvent:
      allOf:
        - $ref: '#/components/schemas/BaseReasoningSummaryTextDoneEvent'
        - properties: {}
          type: object
      description: Event emitted when reasoning summary text streaming is complete
      example:
        item_id: item-1
        output_index: 0
        sequence_number: 6
        summary_index: 0
        text: Analyzing the problem step by step to find the optimal solution.
        type: response.reasoning_summary_text.done
    ReasoningDeltaEvent:
      allOf:
        - $ref: '#/components/schemas/BaseReasoningDeltaEvent'
        - properties: {}
          type: object
      description: Event emitted when reasoning text delta is streamed
      example:
        content_index: 0
        delta: First, we need
        item_id: item-1
        output_index: 0
        sequence_number: 4
        type: response.reasoning_text.delta
    ReasoningDoneEvent:
      allOf:
        - $ref: '#/components/schemas/BaseReasoningDoneEvent'
        - properties: {}
          type: object
      description: Event emitted when reasoning text streaming is complete
      example:
        content_index: 0
        item_id: item-1
        output_index: 0
        sequence_number: 6
        text: >-
          First, we need to identify the key components and then combine them
          logically.
        type: response.reasoning_text.done
    RefusalDeltaEvent:
      allOf:
        - $ref: '#/components/schemas/BaseRefusalDeltaEvent'
        - properties: {}
          type: object
      description: Event emitted when a refusal delta is streamed
      example:
        content_index: 0
        delta: I'm sorry
        item_id: item-1
        output_index: 0
        sequence_number: 4
        type: response.refusal.delta
    RefusalDoneEvent:
      allOf:
        - $ref: '#/components/schemas/BaseRefusalDoneEvent'
        - properties: {}
          type: object
      description: Event emitted when refusal streaming is complete
      example:
        content_index: 0
        item_id: item-1
        output_index: 0
        refusal: I'm sorry, but I can't assist with that request.
        sequence_number: 6
        type: response.refusal.done
    WebSearchCallCompletedEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesSearchCompleted'
        - properties: {}
          type: object
      description: Web search call completed
      example:
        item_id: ws-123
        output_index: 0
        sequence_number: 3
        type: response.web_search_call.completed
    WebSearchCallInProgressEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesWebSearchCallInProgress'
        - properties: {}
          type: object
      description: Web search call in progress
      example:
        item_id: ws-123
        output_index: 0
        sequence_number: 1
        type: response.web_search_call.in_progress
    WebSearchCallSearchingEvent:
      allOf:
        - $ref: '#/components/schemas/OpenAIResponsesWebSearchCallSearching'
        - properties: {}
          type: object
      description: Web search call is searching
      example:
        item_id: ws-123
        output_index: 0
        sequence_number: 2
        type: response.web_search_call.searching
    ReasoningTextContent:
      example:
        text: Let me think step by step about this problem...
        type: reasoning_text
      properties:
        text:
          type: string
        type:
          enum:
            - reasoning_text
          type: string
      required:
        - type
        - text
      type: object
    ReasoningFormat:
      enum:
        - unknown
        - openai-responses-v1
        - azure-openai-responses-v1
        - bedrock-openai-responses-v1
        - xai-responses-v1
        - meta-responses-v1
        - anthropic-claude-v1
        - google-gemini-v1
        - null
      example: unknown
      type:
        - string
        - 'null'
    InputAudio:
      description: Audio input content item
      example:
        input_audio:
          data: SGVsbG8gV29ybGQ=
          format: mp3
        type: input_audio
      properties:
        input_audio:
          properties:
            data:
              type: string
            format:
              enum:
                - mp3
                - wav
              type: string
          required:
            - data
            - format
          type: object
        type:
          enum:
            - input_audio
          type: string
      required:
        - type
        - input_audio
      type: object
    InputVideo:
      description: Video input content item
      example:
        type: input_video
        video_url: https://example.com/video.mp4
      properties:
        type:
          enum:
            - input_video
          type: string
        video_url:
          description: A base64 data URL or remote URL that resolves to a video file
          type: string
      required:
        - type
        - video_url
      type: object
    OpenAIResponseFunctionToolCall:
      example:
        arguments: '{"location":"San Francisco"}'
        call_id: call-abc123
        id: fc-abc123
        name: get_weather
        status: completed
        type: function_call
      properties:
        arguments:
          type: string
        call_id:
          type: string
        id:
          type: string
        name:
          type: string
        namespace:
          description: >-
            Namespace qualifier for tools registered as part of a namespace tool
            group (e.g. an MCP server)
          type: string
        status:
          $ref: '#/components/schemas/ToolCallStatus'
        type:
          enum:
            - function_call
          type: string
      required:
        - type
        - call_id
        - name
        - arguments
      type: object
    OpenAIResponseFunctionToolCallOutput:
      example:
        call_id: call-abc123
        output: '{"temperature":72,"conditions":"sunny"}'
        type: function_call_output
      properties:
        call_id:
          type: string
        id:
          type:
            - string
            - 'null'
        output:
          anyOf:
            - type: string
            - items:
                discriminator:
                  mapping:
                    input_file:
                      $ref: '#/components/schemas/InputFile'
                    input_image:
                      $ref: '#/components/schemas/InputImage'
                    input_text:
                      $ref: '#/components/schemas/InputText'
                  propertyName: type
                oneOf:
                  - $ref: '#/components/schemas/InputText'
                  - $ref: '#/components/schemas/InputImage'
                  - $ref: '#/components/schemas/InputFile'
              type: array
        status:
          anyOf:
            - $ref: '#/components/schemas/ToolCallStatus'
            - type: 'null'
        type:
          enum:
            - function_call_output
          type: string
      required:
        - type
        - call_id
        - output
      type: object
    ApplyPatchCallOperation:
      description: >-
        The patch operation requested by an `apply_patch_call`. `create_file`
        and `update_file` carry a V4A diff; `delete_file` omits it.
      discriminator:
        mapping:
          create_file:
            $ref: '#/components/schemas/ApplyPatchCreateFileOperation'
          delete_file:
            $ref: '#/components/schemas/ApplyPatchDeleteFileOperation'
          update_file:
            $ref: '#/components/schemas/ApplyPatchUpdateFileOperation'
        propertyName: type
      example:
        diff: |-
          @@ function main() {
          +  console.log("hi");
           }
        path: /src/main.ts
        type: update_file
      oneOf:
        - $ref: '#/components/schemas/ApplyPatchCreateFileOperation'
        - $ref: '#/components/schemas/ApplyPatchUpdateFileOperation'
        - $ref: '#/components/schemas/ApplyPatchDeleteFileOperation'
    ApplyPatchCallStatus:
      description: Lifecycle state of an `apply_patch_call` output item.
      enum:
        - in_progress
        - completed
      example: completed
      type: string
    OpenAIResponsesAnnotation:
      anyOf:
        - $ref: '#/components/schemas/FileCitation'
        - $ref: '#/components/schemas/URLCitation'
        - $ref: '#/components/schemas/FilePath'
      example:
        file_id: file-abc123
        filename: research_paper.pdf
        index: 0
        type: file_citation
    CodeInterpreterCallItem:
      allOf:
        - $ref: '#/components/schemas/OutputItemCodeInterpreterCall'
        - additionalProperties: {}
          properties: {}
          type: object
      description: A code interpreter execution call with outputs
      example:
        code: print("Hello, World!")
        container_id: container-xyz789
        id: code-abc123
        outputs:
          - logs: Hello, World!
            type: logs
        status: completed
        type: code_interpreter_call
    ToolCallStatus:
      enum:
        - in_progress
        - completed
        - incomplete
      example: completed
      type: string
    FusionAnalysisResult:
      description: Structured analysis produced by the fusion analyst model.
      example:
        blind_spots:
          - No model considered the impact on existing API consumers.
        consensus:
          - All panel models agree the request is asking for a concise summary.
        contradictions:
          - stances:
              - model: openai/gpt-5
                stance: Favors an incremental rollout.
              - model: anthropic/claude-sonnet-4.5
                stance: Favors a single coordinated migration.
            topic: Recommended approach
        partial_coverage:
          - models:
              - openai/gpt-5
            point: Only one model addressed the rollback strategy.
        unique_insights:
          - insight: >-
              Highlighted a backwards-compatibility risk the other models
              missed.
            model: anthropic/claude-sonnet-4.5
      properties:
        blind_spots:
          items:
            type: string
          type: array
        consensus:
          items:
            type: string
          type: array
        contradictions:
          items:
            properties:
              stances:
                items:
                  properties:
                    model:
                      type: string
                    stance:
                      type: string
                  required:
                    - model
                    - stance
                  type: object
                type: array
              topic:
                type: string
            required:
              - topic
              - stances
            type: object
          type: array
        partial_coverage:
          items:
            properties:
              models:
                items:
                  type: string
                type: array
              point:
                type: string
            required:
              - models
              - point
            type: object
          type: array
        unique_insights:
          items:
            properties:
              insight:
                type: string
              model:
                type: string
            required:
              - model
              - insight
            type: object
          type: array
      required:
        - consensus
        - contradictions
        - partial_coverage
        - unique_insights
        - blind_spots
      type: object
    FusionSource:
      description: A web page retrieved via web search during a fusion run.
      example:
        title: Example article title
        url: https://example.com/article
      properties:
        title:
          description: Title of the retrieved web page.
          type: string
        url:
          description: URL of the web page a panel or the analyst retrieved during the run.
          type: string
      required:
        - url
        - title
      type: object
    OpenAIResponseCustomToolCallOutput:
      example:
        call_id: call-abc123
        output: patch applied successfully
        type: custom_tool_call_output
      properties:
        call_id:
          type: string
        id:
          type: string
        output:
          anyOf:
            - type: string
            - items:
                discriminator:
                  mapping:
                    input_file:
                      $ref: '#/components/schemas/InputFile'
                    input_image:
                      $ref: '#/components/schemas/InputImage'
                    input_text:
                      $ref: '#/components/schemas/InputText'
                  propertyName: type
                oneOf:
                  - $ref: '#/components/schemas/InputText'
                  - $ref: '#/components/schemas/InputImage'
                  - $ref: '#/components/schemas/InputFile'
              type: array
        type:
          enum:
            - custom_tool_call_output
          type: string
      required:
        - type
        - call_id
        - output
      type: object
    PDFParserEngine:
      anyOf:
        - enum:
            - mistral-ocr
            - native
            - cloudflare-ai
          type: string
        - enum:
            - pdf-text
          type: string
      description: >-
        The engine to use for parsing PDF files. "pdf-text" is deprecated and
        automatically redirected to "cloudflare-ai".
      example: cloudflare-ai
    PromptCacheBreakpoint:
      description: >-
        Marks an explicit prompt-cache boundary on this content block
        (OpenAI-style). Everything through the block carrying this marker is
        part of the candidate cached prefix. Supported natively by OpenAI
        GPT-5.6 and newer; on providers that use Anthropic-style
        `cache_control`, OpenRouter converts the marker to that format
        automatically.
      example:
        mode: explicit
      properties:
        mode:
          enum:
            - explicit
          type: string
      required:
        - mode
      type:
        - object
        - 'null'
    PercentileLatencyCutoffs:
      description: >-
        Percentile-based latency cutoffs. All specified cutoffs must be met for
        an endpoint to be preferred.
      example:
        p50: 5
        p90: 10
      properties:
        p50:
          description: Maximum p50 latency (seconds)
          format: double
          type:
            - number
            - 'null'
        p75:
          description: Maximum p75 latency (seconds)
          format: double
          type:
            - number
            - 'null'
        p90:
          description: Maximum p90 latency (seconds)
          format: double
          type:
            - number
            - 'null'
        p99:
          description: Maximum p99 latency (seconds)
          format: double
          type:
            - number
            - 'null'
      type: object
    PercentileThroughputCutoffs:
      description: >-
        Percentile-based throughput cutoffs. All specified cutoffs must be met
        for an endpoint to be preferred.
      example:
        p50: 100
        p90: 50
      properties:
        p50:
          description: Minimum p50 throughput (tokens/sec)
          format: double
          type:
            - number
            - 'null'
        p75:
          description: Minimum p75 throughput (tokens/sec)
          format: double
          type:
            - number
            - 'null'
        p90:
          description: Minimum p90 throughput (tokens/sec)
          format: double
          type:
            - number
            - 'null'
        p99:
          description: Minimum p99 throughput (tokens/sec)
          format: double
          type:
            - number
            - 'null'
      type: object
    ReasoningContext:
      description: >-
        Controls which reasoning is available to the model. `auto` uses the
        model default (same as omitting); `all_turns` includes reasoning from
        earlier turns passed in input; `current_turn` limits to the current turn
        only. Only supported by OpenAI GPT-5.6 and newer.
      enum:
        - auto
        - all_turns
        - current_turn
        - null
      example: all_turns
      type:
        - string
        - 'null'
    ReasoningEffort:
      enum:
        - max
        - xhigh
        - high
        - medium
        - low
        - minimal
        - none
        - null
      example: medium
      type:
        - string
        - 'null'
    ReasoningMode:
      description: >-
        Selects the reasoning mode. `standard` is the default; `pro` engages
        deeper reasoning on models that support it, billed at standard token
        rates. Only supported by OpenAI GPT-5.6 and newer.
      enum:
        - standard
        - pro
        - null
      example: standard
      type:
        - string
        - 'null'
    ReasoningSummaryVerbosity:
      enum:
        - auto
        - concise
        - detailed
        - null
      example: auto
      type:
        - string
        - 'null'
    StopServerToolsWhenFinishReasonIs:
      description: Stop when the upstream model emits this finish reason (e.g. `length`).
      example:
        reason: length
        type: finish_reason_is
      properties:
        reason:
          minLength: 1
          type: string
        type:
          enum:
            - finish_reason_is
          type: string
      required:
        - type
        - reason
      type: object
    StopServerToolsWhenHasToolCall:
      description: Stop after a tool with this name has been called.
      example:
        tool_name: finalize
        type: has_tool_call
      properties:
        tool_name:
          minLength: 1
          type: string
        type:
          enum:
            - has_tool_call
          type: string
      required:
        - type
        - tool_name
      type: object
    StopServerToolsWhenMaxCost:
      description: Stop once cumulative cost across the loop exceeds this dollar threshold.
      example:
        max_cost_in_dollars: 0.5
        type: max_cost
      properties:
        max_cost_in_dollars:
          format: double
          type: number
        type:
          enum:
            - max_cost
          type: string
      required:
        - type
        - max_cost_in_dollars
      type: object
    StopServerToolsWhenMaxTokensUsed:
      description: Stop once cumulative token usage across the loop exceeds this threshold.
      example:
        max_tokens: 10000
        type: max_tokens_used
      properties:
        max_tokens:
          type: integer
        type:
          enum:
            - max_tokens_used
          type: string
      required:
        - type
        - max_tokens
      type: object
    StopServerToolsWhenStepCountIs:
      description: Stop after the agent loop has executed this many steps.
      example:
        step_count: 5
        type: step_count_is
      properties:
        step_count:
          type: integer
        type:
          enum:
            - step_count_is
          type: string
      required:
        - type
        - step_count
      type: object
    Formats:
      anyOf:
        - $ref: '#/components/schemas/FormatTextConfig'
        - $ref: '#/components/schemas/FormatJsonObjectConfig'
        - $ref: '#/components/schemas/FormatJsonSchemaConfig'
      description: Text response format configuration
      example:
        type: text
    AdvisorReasoning:
      description: >-
        Reasoning configuration forwarded to the advisor call. Use this to
        control reasoning effort and token budget for models that support
        extended thinking.
      example:
        effort: high
      properties:
        effort:
          description: Reasoning effort level for the advisor call.
          enum:
            - max
            - xhigh
            - high
            - medium
            - low
            - minimal
            - none
          type: string
        max_tokens:
          description: Maximum number of reasoning tokens the advisor may use.
          type: integer
      type: object
    AdvisorNestedTool:
      additionalProperties: {}
      description: >-
        A tool made available to the advisor sub-agent. Only OpenRouter server
        tools (e.g. openrouter:web_search) are supported; function tools are
        rejected because the advisor has no way to execute them. The advisor
        tool may not list itself.
      example:
        type: openrouter:web_search
      properties:
        parameters:
          additionalProperties: {}
          type: object
        type:
          type: string
      required:
        - type
      type: object
    SubagentReasoning:
      description: >-
        Reasoning configuration forwarded to the subagent call. Use this to
        control reasoning effort and token budget for models that support
        extended thinking.
      example:
        effort: low
      properties:
        effort:
          description: Reasoning effort level for the subagent call.
          enum:
            - max
            - xhigh
            - high
            - medium
            - low
            - minimal
            - none
          type: string
        max_tokens:
          description: >-
            Maximum number of reasoning tokens the subagent may use. Accepted
            and validated but not yet forwarded to the subagent call.
          type: integer
      type: object
    SubagentNestedTool:
      additionalProperties: {}
      description: >-
        A tool made available to the subagent. Only OpenRouter server tools
        (e.g. openrouter:web_search) are supported; function tools are rejected
        because the worker has no way to execute them. The subagent tool may not
        list itself.
      example:
        type: openrouter:web_search
      properties:
        parameters:
          additionalProperties: {}
          type: object
        type:
          type: string
      required:
        - type
      type: object
    WebFetchEngineEnum:
      description: >-
        Which fetch engine to use. "auto" (default) uses native if the provider
        supports it, otherwise Exa. "native" forces the provider's built-in
        fetch. "exa" uses Exa Contents API. "openrouter" uses direct HTTP fetch.
        "firecrawl" uses Firecrawl scrape (requires BYOK). "parallel" uses the
        Parallel extract API.
      enum:
        - auto
        - native
        - openrouter
        - exa
        - parallel
        - firecrawl
      example: auto
      type: string
    SearchQualityLevel:
      description: >-
        How much context to retrieve per result. Applies to Exa, Parallel, and
        Perplexity engines; ignored with native provider search and Firecrawl.
        For Exa, pins a fixed per-result character cap (low=5,000,
        medium=15,000, high=30,000); when omitted, Exa picks an adaptive size
        per query and document (typically ~2,000–4,000 characters per result).
        For Parallel, controls the total characters across all results; when
        omitted, Parallel uses its own default size. For Perplexity, maps
        directly to the Search API's native search_context_size parameter.
        Overridden by `max_characters` when both are set.
      enum:
        - low
        - medium
        - high
      example: medium
      type: string
    WebSearchUserLocationServerTool:
      description: Approximate user location for location-biased results.
      example:
        city: San Francisco
        country: US
        region: California
        timezone: America/Los_Angeles
        type: approximate
      properties:
        city:
          type:
            - string
            - 'null'
        country:
          type:
            - string
            - 'null'
        region:
          type:
            - string
            - 'null'
        timezone:
          type:
            - string
            - 'null'
        type:
          enum:
            - approximate
          type: string
      type: object
    ApplyPatchEngineEnum:
      description: >-
        Which apply_patch engine to use. "auto" (default) uses native
        passthrough when the endpoint advertises native apply_patch support,
        otherwise falls back to OpenRouter's HITL validator. "native" forces
        native passthrough — when the endpoint does not support native, the
        request falls back to HITL. "openrouter" always runs the HITL validator.
        Native passthrough streams the diff incrementally via
        `apply_patch_call_operation_diff.delta` events; HITL buffers the diff
        for atomic delivery as a single delta.
      enum:
        - auto
        - native
        - openrouter
      example: auto
      type: string
    BashServerToolEngine:
      description: >-
        Which bash engine to use. "openrouter" runs commands server-side in the
        OpenRouter sandbox. "auto" (default) and "native" use native
        passthrough, returning the tool call to your application to run
        client-side; OpenRouter does not execute the commands.
      enum:
        - auto
        - native
        - openrouter
      example: auto
      type: string
    BashServerToolEnvironment:
      description: Execution environment for the bash server tool.
      discriminator:
        mapping:
          container_auto:
            $ref: '#/components/schemas/ContainerAutoEnvironment'
          container_reference:
            $ref: '#/components/schemas/ContainerReferenceEnvironment'
        propertyName: type
      example:
        type: container_auto
      oneOf:
        - $ref: '#/components/schemas/ContainerAutoEnvironment'
        - $ref: '#/components/schemas/ContainerReferenceEnvironment'
    SandboxSleepAfterSeconds:
      description: >-
        How long (in seconds) the container stays warm after its last command
        before sleeping, freeing its capacity slot. Idle-based: each command
        renews the timer. Defaults to 900 (15 minutes); capped at 2592000 (30
        days).
      example: 900
      type: integer
    ShellServerToolEngine:
      description: >-
        Which shell engine to use. "openrouter" runs commands server-side in the
        OpenRouter sandbox. "auto" (default) keeps the provider's native hosted
        shell when available (OpenAI); on other providers the call is routed to
        the OpenRouter sandbox.
      enum:
        - auto
        - openrouter
      example: openrouter
      type: string
    ShellServerToolEnvironment:
      description: >-
        Server-side execution environment for the shell tool. Only
        container-backed environments are supported; "local" shells are not.
      discriminator:
        mapping:
          container_auto:
            $ref: '#/components/schemas/ContainerAutoEnvironment'
          container_reference:
            $ref: '#/components/schemas/ContainerReferenceEnvironment'
        propertyName: type
      example:
        type: container_auto
      oneOf:
        - $ref: '#/components/schemas/ContainerAutoEnvironment'
        - $ref: '#/components/schemas/ContainerReferenceEnvironment'
    OpenAIResponseInputMessageItem:
      example:
        content:
          - text: Hello, how are you?
            type: input_text
        id: msg-abc123
        role: user
        type: message
      properties:
        content:
          items:
            discriminator:
              mapping:
                input_audio:
                  $ref: '#/components/schemas/InputAudio'
                input_file:
                  $ref: '#/components/schemas/InputFile'
                input_image:
                  $ref: '#/components/schemas/InputImage'
                input_text:
                  $ref: '#/components/schemas/InputText'
              propertyName: type
            oneOf:
              - $ref: '#/components/schemas/InputText'
              - $ref: '#/components/schemas/InputImage'
              - $ref: '#/components/schemas/InputFile'
              - $ref: '#/components/schemas/InputAudio'
          type: array
        id:
          type: string
        role:
          anyOf:
            - enum:
                - user
              type: string
            - enum:
                - system
              type: string
            - enum:
                - developer
              type: string
        type:
          enum:
            - message
          type: string
      required:
        - id
        - role
        - content
      type: object
    OpenAIResponseCustomToolCall:
      example:
        call_id: call-abc123
        id: ctc-abc123
        input: |-
          *** Begin Patch
          *** End Patch
        name: apply_patch
        type: custom_tool_call
      properties:
        call_id:
          type: string
        id:
          type: string
        input:
          type: string
        name:
          type: string
        namespace:
          description: >-
            Namespace qualifier for tools registered as part of a namespace tool
            group (e.g. an MCP server)
          type: string
        type:
          enum:
            - custom_tool_call
          type: string
      required:
        - type
        - call_id
        - name
        - input
      type: object
    ApplyPatchCreateFileOperation:
      description: >-
        The `create_file` variant of an `apply_patch_call.operation`. Carries a
        V4A diff describing the new file contents.
      example:
        diff: |
          @@
          +console.log("hi");
        path: /src/main.ts
        type: create_file
      properties:
        diff:
          type: string
        path:
          type: string
        type:
          enum:
            - create_file
          type: string
      required:
        - type
        - path
        - diff
      type: object
    ApplyPatchDeleteFileOperation:
      description: >-
        The `delete_file` variant of an `apply_patch_call.operation`. Identifies
        the file to remove; no diff is required.
      example:
        path: /src/main.ts
        type: delete_file
      properties:
        path:
          type: string
        type:
          enum:
            - delete_file
          type: string
      required:
        - type
        - path
      type: object
    ApplyPatchUpdateFileOperation:
      description: >-
        The `update_file` variant of an `apply_patch_call.operation`. Carries a
        V4A diff describing edits to an existing file.
      example:
        diff: |-
          @@ function main() {
          +  console.log("hi");
           }
        path: /src/main.ts
        type: update_file
      properties:
        diff:
          type: string
        path:
          type: string
        type:
          enum:
            - update_file
          type: string
      required:
        - type
        - path
        - diff
      type: object
    CodeInterpreterFileOutput:
      example:
        download_url: https://example.com/download/file_abc123
        filename: summary.txt
        id: file_abc123
        type: file
      properties:
        download_url:
          description: >-
            Provider-issued download URL for the generated file. Typically
            signed and time-limited (see `expires_at`); fetch promptly rather
            than persisting the URL.
          type: string
        error_code:
          description: >-
            Provider error code when generating or persisting the file failed;
            null or absent on success.
          type:
            - string
            - 'null'
        expires_at:
          description: >-
            When the `download_url` stops working, as an ISO 8601 timestamp.
            After this time the file must be regenerated.
          type: string
        filename:
          type: string
        id:
          type: string
        media_type:
          type: string
        sha256:
          type: string
        size_bytes:
          type: integer
        status:
          description: >-
            Provider-reported readiness of the generated file (e.g. `ready`).
            Values other than `ready` indicate the artifact may not be
            downloadable.
          type: string
        type:
          enum:
            - file
          type: string
      required:
        - type
      type: object
    CodeInterpreterImageOutput:
      example:
        type: image
        url: https://example.com/plot.png
      properties:
        type:
          enum:
            - image
          type: string
        url:
          type: string
      required:
        - type
        - url
      type: object
    CodeInterpreterLogsOutput:
      example:
        logs: |
          hello
        type: logs
      properties:
        logs:
          type: string
        type:
          enum:
            - logs
          type: string
      required:
        - type
        - logs
      type: object
    WebSearchStatus:
      enum:
        - completed
        - searching
        - in_progress
        - failed
      example: completed
      type: string
    ImageGenerationStatus:
      enum:
        - in_progress
        - completed
        - generating
        - failed
      example: completed
      type: string
    WebSearchSource:
      example:
        type: url
        url: https://example.com/article
      properties:
        type:
          enum:
            - url
          type: string
        url:
          type: string
      required:
        - type
        - url
      type: object
    EndpointInfo:
      example:
        model: openai/gpt-4o
        provider: OpenAI
        selected: true
      properties:
        model:
          type: string
        provider:
          type: string
        selected:
          type: boolean
      required:
        - provider
        - model
        - selected
      type: object
    PipelineStageType:
      description: >-
        Categorical kind of a pipeline stage. Multiple plugins can share a type
        (e.g. all guardrail-level plugins emit `guardrail`); the `name` field
        disambiguates which plugin emitted it.
      enum:
        - guardrail
        - plugin
        - server_tools
        - response_healing
        - context_compression
      example: guardrail
      type: string
    ShellCallStatus:
      description: Status of a shell call or its output.
      enum:
        - in_progress
        - completed
        - incomplete
      example: completed
      type: string
    BaseErrorEvent:
      description: Event emitted when an error occurs during streaming
      example:
        code: rate_limit_exceeded
        message: Rate limit exceeded. Please try again later.
        param: null
        sequence_number: 2
        type: error
      properties:
        code:
          type:
            - string
            - 'null'
        message:
          type: string
        param:
          type:
            - string
            - 'null'
        sequence_number:
          type: integer
        type:
          enum:
            - error
          type: string
      required:
        - type
        - code
        - message
        - param
        - sequence_number
      type: object
    OpenAIResponsesCodeInterpreterCallCodeDelta:
      example:
        delta: print("hello")
        item_id: ci_abc123
        output_index: 0
        sequence_number: 2
        type: response.code_interpreter_call_code.delta
      properties:
        delta:
          type: string
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.code_interpreter_call_code.delta
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
        - delta
      type: object
    OpenAIResponsesCodeInterpreterCallCodeDone:
      example:
        code: print("hello")
        item_id: ci_abc123
        output_index: 0
        sequence_number: 3
        type: response.code_interpreter_call_code.done
      properties:
        code:
          type: string
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.code_interpreter_call_code.done
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
        - code
      type: object
    OpenAIResponsesCodeInterpreterCallCompleted:
      example:
        item_id: ci_abc123
        output_index: 0
        sequence_number: 3
        type: response.code_interpreter_call.completed
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.code_interpreter_call.completed
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
      type: object
    OpenAIResponsesCodeInterpreterCallInProgress:
      example:
        item_id: ci_abc123
        output_index: 0
        sequence_number: 1
        type: response.code_interpreter_call.in_progress
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.code_interpreter_call.in_progress
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
      type: object
    OpenAIResponsesCodeInterpreterCallInterpreting:
      example:
        item_id: ci_abc123
        output_index: 0
        sequence_number: 2
        type: response.code_interpreter_call.interpreting
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.code_interpreter_call.interpreting
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
      type: object
    CompletedEvent:
      description: Event emitted when a response has completed successfully
      example:
        response:
          completed_at: null
          created_at: 1704067200
          error: null
          frequency_penalty: null
          id: resp-abc123
          incomplete_details: null
          instructions: null
          max_output_tokens: null
          metadata: null
          model: gpt-4
          object: response
          output:
            - content:
                - annotations: []
                  text: Hello! How can I help you?
                  type: output_text
              id: item-1
              role: assistant
              status: completed
              type: message
          parallel_tool_calls: true
          presence_penalty: null
          status: completed
          temperature: null
          tool_choice: auto
          tools: []
          top_p: null
        sequence_number: 10
        type: response.completed
      properties:
        response:
          $ref: '#/components/schemas/BaseResponsesResult'
        sequence_number:
          type: integer
        type:
          enum:
            - response.completed
          type: string
      required:
        - type
        - response
        - sequence_number
      type: object
    BaseContentPartAddedEvent:
      description: Event emitted when a new content part is added to an output item
      example:
        content_index: 0
        item_id: item-1
        output_index: 0
        part:
          annotations: []
          text: ''
          type: output_text
        sequence_number: 3
        type: response.content_part.added
      properties:
        content_index:
          type: integer
        item_id:
          type: string
        output_index:
          type: integer
        part:
          anyOf:
            - $ref: '#/components/schemas/ResponseOutputText'
            - $ref: '#/components/schemas/OpenAIResponsesRefusalContent'
        sequence_number:
          type: integer
        type:
          enum:
            - response.content_part.added
          type: string
      required:
        - type
        - output_index
        - item_id
        - content_index
        - part
        - sequence_number
      type: object
    BaseContentPartDoneEvent:
      description: Event emitted when a content part is complete
      example:
        content_index: 0
        item_id: item-1
        output_index: 0
        part:
          annotations: []
          text: Hello! How can I help you?
          type: output_text
        sequence_number: 7
        type: response.content_part.done
      properties:
        content_index:
          type: integer
        item_id:
          type: string
        output_index:
          type: integer
        part:
          anyOf:
            - $ref: '#/components/schemas/ResponseOutputText'
            - $ref: '#/components/schemas/OpenAIResponsesRefusalContent'
        sequence_number:
          type: integer
        type:
          enum:
            - response.content_part.done
          type: string
      required:
        - type
        - output_index
        - item_id
        - content_index
        - part
        - sequence_number
      type: object
    CreatedEvent:
      description: Event emitted when a response is created
      example:
        response:
          completed_at: null
          created_at: 1704067200
          error: null
          frequency_penalty: null
          id: resp-abc123
          incomplete_details: null
          instructions: null
          max_output_tokens: null
          metadata: null
          model: gpt-4
          object: response
          output: []
          parallel_tool_calls: true
          presence_penalty: null
          status: in_progress
          temperature: null
          tool_choice: auto
          tools: []
          top_p: null
        sequence_number: 0
        type: response.created
      properties:
        response:
          $ref: '#/components/schemas/BaseResponsesResult'
        sequence_number:
          type: integer
        type:
          enum:
            - response.created
          type: string
      required:
        - type
        - response
        - sequence_number
      type: object
    BaseCustomToolCallInputDeltaEvent:
      description: >-
        Event emitted when a custom tool call's freeform input is being
        streamed. Mirrors `response.function_call_arguments.delta` but for
        `custom` tools whose input is opaque text rather than JSON arguments.
      example:
        delta: '*** Begin Patch'
        item_id: item-1
        output_index: 0
        sequence_number: 4
        type: response.custom_tool_call_input.delta
      properties:
        delta:
          type: string
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.custom_tool_call_input.delta
          type: string
      required:
        - type
        - item_id
        - output_index
        - delta
        - sequence_number
      type: object
    BaseCustomToolCallInputDoneEvent:
      description: >-
        Event emitted when a custom tool call's freeform input streaming is
        complete. Mirrors `response.function_call_arguments.done` but for
        `custom` tools.
      example:
        input: |-
          *** Begin Patch
          *** End Patch
        item_id: item-1
        output_index: 0
        sequence_number: 6
        type: response.custom_tool_call_input.done
      properties:
        input:
          type: string
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.custom_tool_call_input.done
          type: string
      required:
        - type
        - item_id
        - output_index
        - input
        - sequence_number
      type: object
    FailedEvent:
      description: Event emitted when a response has failed
      example:
        response:
          completed_at: null
          created_at: 1704067200
          error: null
          frequency_penalty: null
          id: resp-abc123
          incomplete_details: null
          instructions: null
          max_output_tokens: null
          metadata: null
          model: gpt-4
          object: response
          output: []
          parallel_tool_calls: true
          presence_penalty: null
          status: failed
          temperature: null
          tool_choice: auto
          tools: []
          top_p: null
        sequence_number: 3
        type: response.failed
      properties:
        response:
          $ref: '#/components/schemas/BaseResponsesResult'
        sequence_number:
          type: integer
        type:
          enum:
            - response.failed
          type: string
      required:
        - type
        - response
        - sequence_number
      type: object
    BaseFunctionCallArgsDeltaEvent:
      description: Event emitted when function call arguments are being streamed
      example:
        delta: '{"city": "..."}'
        item_id: item-1
        output_index: 0
        sequence_number: 4
        type: response.function_call_arguments.delta
      properties:
        delta:
          type: string
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.function_call_arguments.delta
          type: string
      required:
        - type
        - item_id
        - output_index
        - delta
        - sequence_number
      type: object
    BaseFunctionCallArgsDoneEvent:
      description: Event emitted when function call arguments streaming is complete
      example:
        arguments: '{"city": "San Francisco", "units": "celsius"}'
        item_id: item-1
        name: get_weather
        output_index: 0
        sequence_number: 6
        type: response.function_call_arguments.done
      properties:
        arguments:
          type: string
        item_id:
          type: string
        name:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.function_call_arguments.done
          type: string
      required:
        - type
        - item_id
        - output_index
        - name
        - arguments
        - sequence_number
      type: object
    OpenAIResponsesImageGenCallCompleted:
      example:
        item_id: ig_abc123
        output_index: 0
        sequence_number: 4
        type: response.image_generation_call.completed
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.image_generation_call.completed
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
      type: object
    OpenAIResponsesImageGenCallGenerating:
      example:
        item_id: ig_abc123
        output_index: 0
        sequence_number: 2
        type: response.image_generation_call.generating
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.image_generation_call.generating
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
      type: object
    OpenAIResponsesImageGenCallInProgress:
      example:
        item_id: ig_abc123
        output_index: 0
        sequence_number: 1
        type: response.image_generation_call.in_progress
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.image_generation_call.in_progress
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
      type: object
    OpenAIResponsesImageGenCallPartialImage:
      example:
        item_id: ig_abc123
        output_index: 0
        partial_image_b64: iVBORw0KGgo...
        partial_image_index: 0
        sequence_number: 3
        type: response.image_generation_call.partial_image
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        partial_image_b64:
          type: string
        partial_image_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.image_generation_call.partial_image
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
        - partial_image_b64
        - partial_image_index
      type: object
    InProgressEvent:
      description: Event emitted when a response is in progress
      example:
        response:
          completed_at: null
          created_at: 1704067200
          error: null
          frequency_penalty: null
          id: resp-abc123
          incomplete_details: null
          instructions: null
          max_output_tokens: null
          metadata: null
          model: gpt-4
          object: response
          output: []
          parallel_tool_calls: true
          presence_penalty: null
          status: in_progress
          temperature: null
          tool_choice: auto
          tools: []
          top_p: null
        sequence_number: 1
        type: response.in_progress
      properties:
        response:
          $ref: '#/components/schemas/BaseResponsesResult'
        sequence_number:
          type: integer
        type:
          enum:
            - response.in_progress
          type: string
      required:
        - type
        - response
        - sequence_number
      type: object
    IncompleteEvent:
      description: Event emitted when a response is incomplete
      example:
        response:
          completed_at: null
          created_at: 1704067200
          error: null
          frequency_penalty: null
          id: resp-abc123
          incomplete_details: null
          instructions: null
          max_output_tokens: null
          metadata: null
          model: gpt-4
          object: response
          output: []
          parallel_tool_calls: true
          presence_penalty: null
          status: incomplete
          temperature: null
          tool_choice: auto
          tools: []
          top_p: null
        sequence_number: 5
        type: response.incomplete
      properties:
        response:
          $ref: '#/components/schemas/BaseResponsesResult'
        sequence_number:
          type: integer
        type:
          enum:
            - response.incomplete
          type: string
      required:
        - type
        - response
        - sequence_number
      type: object
    OutputItemAddedEvent:
      description: Event emitted when a new output item is added to the response
      example:
        item:
          content: []
          id: item-1
          role: assistant
          status: in_progress
          type: message
        output_index: 0
        sequence_number: 2
        type: response.output_item.added
      properties:
        item:
          discriminator:
            mapping:
              apply_patch_call:
                $ref: '#/components/schemas/OutputItemApplyPatchCall'
              code_interpreter_call:
                $ref: '#/components/schemas/OutputItemCodeInterpreterCall'
              custom_tool_call:
                $ref: '#/components/schemas/OutputItemCustomToolCall'
              file_search_call:
                $ref: '#/components/schemas/OutputItemFileSearchCall'
              function_call:
                $ref: '#/components/schemas/OutputItemFunctionCall'
              image_generation_call:
                $ref: '#/components/schemas/OutputItemImageGenerationCall'
              message:
                $ref: '#/components/schemas/OutputMessage'
              reasoning:
                $ref: '#/components/schemas/OutputItemReasoning'
              web_search_call:
                $ref: '#/components/schemas/OutputItemWebSearchCall'
            propertyName: type
          oneOf:
            - $ref: '#/components/schemas/OutputMessage'
            - $ref: '#/components/schemas/OutputItemReasoning'
            - $ref: '#/components/schemas/OutputItemFunctionCall'
            - $ref: '#/components/schemas/OutputItemCustomToolCall'
            - $ref: '#/components/schemas/OutputItemWebSearchCall'
            - $ref: '#/components/schemas/OutputItemFileSearchCall'
            - $ref: '#/components/schemas/OutputItemImageGenerationCall'
            - $ref: '#/components/schemas/OutputItemApplyPatchCall'
            - $ref: '#/components/schemas/OutputItemCodeInterpreterCall'
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.output_item.added
          type: string
      required:
        - type
        - output_index
        - item
        - sequence_number
      type: object
    OutputItemDoneEvent:
      description: Event emitted when an output item is complete
      example:
        item:
          content:
            - annotations: []
              text: Hello! How can I help you?
              type: output_text
          id: item-1
          role: assistant
          status: completed
          type: message
        output_index: 0
        sequence_number: 8
        type: response.output_item.done
      properties:
        item:
          discriminator:
            mapping:
              apply_patch_call:
                $ref: '#/components/schemas/OutputItemApplyPatchCall'
              code_interpreter_call:
                $ref: '#/components/schemas/OutputItemCodeInterpreterCall'
              custom_tool_call:
                $ref: '#/components/schemas/OutputItemCustomToolCall'
              file_search_call:
                $ref: '#/components/schemas/OutputItemFileSearchCall'
              function_call:
                $ref: '#/components/schemas/OutputItemFunctionCall'
              image_generation_call:
                $ref: '#/components/schemas/OutputItemImageGenerationCall'
              message:
                $ref: '#/components/schemas/OutputMessage'
              reasoning:
                $ref: '#/components/schemas/OutputItemReasoning'
              web_search_call:
                $ref: '#/components/schemas/OutputItemWebSearchCall'
            propertyName: type
          oneOf:
            - $ref: '#/components/schemas/OutputMessage'
            - $ref: '#/components/schemas/OutputItemReasoning'
            - $ref: '#/components/schemas/OutputItemFunctionCall'
            - $ref: '#/components/schemas/OutputItemCustomToolCall'
            - $ref: '#/components/schemas/OutputItemWebSearchCall'
            - $ref: '#/components/schemas/OutputItemFileSearchCall'
            - $ref: '#/components/schemas/OutputItemImageGenerationCall'
            - $ref: '#/components/schemas/OutputItemApplyPatchCall'
            - $ref: '#/components/schemas/OutputItemCodeInterpreterCall'
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.output_item.done
          type: string
      required:
        - type
        - output_index
        - item
        - sequence_number
      type: object
    BaseAnnotationAddedEvent:
      description: Event emitted when a text annotation is added to output
      example:
        annotation:
          end_index: 7
          start_index: 0
          title: Example
          type: url_citation
          url: https://example.com
        annotation_index: 0
        content_index: 0
        item_id: item-1
        output_index: 0
        sequence_number: 5
        type: response.output_text.annotation.added
      properties:
        annotation:
          $ref: '#/components/schemas/OpenAIResponsesAnnotation'
        annotation_index:
          type: integer
        content_index:
          type: integer
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.output_text.annotation.added
          type: string
      required:
        - type
        - output_index
        - item_id
        - content_index
        - sequence_number
        - annotation_index
        - annotation
      type: object
    BaseTextDeltaEvent:
      description: Event emitted when a text delta is streamed
      example:
        content_index: 0
        delta: Hello
        item_id: item-1
        logprobs: []
        output_index: 0
        sequence_number: 4
        type: response.output_text.delta
      properties:
        content_index:
          type: integer
        delta:
          type: string
        item_id:
          type: string
        logprobs:
          items:
            $ref: '#/components/schemas/OpenResponsesLogProbs'
          type: array
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.output_text.delta
          type: string
      required:
        - type
        - logprobs
        - output_index
        - item_id
        - content_index
        - delta
        - sequence_number
      type: object
    StreamLogprob:
      allOf:
        - $ref: '#/components/schemas/OpenResponsesLogProbs'
        - properties:
            top_logprobs:
              items:
                $ref: '#/components/schemas/StreamLogprobTopLogprob'
              type: array
          type: object
      description: Log probability information for a token
      example:
        bytes:
          - 72
          - 101
          - 108
          - 108
          - 111
        logprob: -0.5
        token: Hello
        top_logprobs: []
    BaseTextDoneEvent:
      description: Event emitted when text streaming is complete
      example:
        content_index: 0
        item_id: item-1
        logprobs: []
        output_index: 0
        sequence_number: 6
        text: Hello! How can I help you?
        type: response.output_text.done
      properties:
        content_index:
          type: integer
        item_id:
          type: string
        logprobs:
          items:
            $ref: '#/components/schemas/OpenResponsesLogProbs'
          type: array
        output_index:
          type: integer
        sequence_number:
          type: integer
        text:
          type: string
        type:
          enum:
            - response.output_text.done
          type: string
      required:
        - type
        - output_index
        - item_id
        - content_index
        - text
        - sequence_number
        - logprobs
      type: object
    BaseReasoningSummaryPartAddedEvent:
      description: Event emitted when a reasoning summary part is added
      example:
        item_id: item-1
        output_index: 0
        part:
          text: ''
          type: summary_text
        sequence_number: 3
        summary_index: 0
        type: response.reasoning_summary_part.added
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        part:
          $ref: '#/components/schemas/ReasoningSummaryText'
        sequence_number:
          type: integer
        summary_index:
          type: integer
        type:
          enum:
            - response.reasoning_summary_part.added
          type: string
      required:
        - type
        - output_index
        - item_id
        - summary_index
        - part
        - sequence_number
      type: object
    BaseReasoningSummaryPartDoneEvent:
      description: Event emitted when a reasoning summary part is complete
      example:
        item_id: item-1
        output_index: 0
        part:
          text: Analyzing the problem step by step to find the optimal solution.
          type: summary_text
        sequence_number: 7
        summary_index: 0
        type: response.reasoning_summary_part.done
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        part:
          $ref: '#/components/schemas/ReasoningSummaryText'
        sequence_number:
          type: integer
        summary_index:
          type: integer
        type:
          enum:
            - response.reasoning_summary_part.done
          type: string
      required:
        - type
        - output_index
        - item_id
        - summary_index
        - part
        - sequence_number
      type: object
    BaseReasoningSummaryTextDeltaEvent:
      description: Event emitted when reasoning summary text delta is streamed
      example:
        delta: Analyzing
        item_id: item-1
        output_index: 0
        sequence_number: 4
        summary_index: 0
        type: response.reasoning_summary_text.delta
      properties:
        delta:
          type: string
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        summary_index:
          type: integer
        type:
          enum:
            - response.reasoning_summary_text.delta
          type: string
      required:
        - type
        - item_id
        - output_index
        - summary_index
        - delta
        - sequence_number
      type: object
    BaseReasoningSummaryTextDoneEvent:
      description: Event emitted when reasoning summary text streaming is complete
      example:
        item_id: item-1
        output_index: 0
        sequence_number: 6
        summary_index: 0
        text: Analyzing the problem step by step to find the optimal solution.
        type: response.reasoning_summary_text.done
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        summary_index:
          type: integer
        text:
          type: string
        type:
          enum:
            - response.reasoning_summary_text.done
          type: string
      required:
        - type
        - item_id
        - output_index
        - summary_index
        - text
        - sequence_number
      type: object
    BaseReasoningDeltaEvent:
      description: Event emitted when reasoning text delta is streamed
      example:
        content_index: 0
        delta: First, we need
        item_id: item-1
        output_index: 0
        sequence_number: 4
        type: response.reasoning_text.delta
      properties:
        content_index:
          type: integer
        delta:
          type: string
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.reasoning_text.delta
          type: string
      required:
        - type
        - output_index
        - item_id
        - content_index
        - delta
        - sequence_number
      type: object
    BaseReasoningDoneEvent:
      description: Event emitted when reasoning text streaming is complete
      example:
        content_index: 0
        item_id: item-1
        output_index: 0
        sequence_number: 6
        text: >-
          First, we need to identify the key components and then combine them
          logically.
        type: response.reasoning_text.done
      properties:
        content_index:
          type: integer
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        text:
          type: string
        type:
          enum:
            - response.reasoning_text.done
          type: string
      required:
        - type
        - output_index
        - item_id
        - content_index
        - text
        - sequence_number
      type: object
    BaseRefusalDeltaEvent:
      description: Event emitted when a refusal delta is streamed
      example:
        content_index: 0
        delta: I'm sorry
        item_id: item-1
        output_index: 0
        sequence_number: 4
        type: response.refusal.delta
      properties:
        content_index:
          type: integer
        delta:
          type: string
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.refusal.delta
          type: string
      required:
        - type
        - output_index
        - item_id
        - content_index
        - delta
        - sequence_number
      type: object
    BaseRefusalDoneEvent:
      description: Event emitted when refusal streaming is complete
      example:
        content_index: 0
        item_id: item-1
        output_index: 0
        refusal: I'm sorry, but I can't assist with that request.
        sequence_number: 6
        type: response.refusal.done
      properties:
        content_index:
          type: integer
        item_id:
          type: string
        output_index:
          type: integer
        refusal:
          type: string
        sequence_number:
          type: integer
        type:
          enum:
            - response.refusal.done
          type: string
      required:
        - type
        - output_index
        - item_id
        - content_index
        - refusal
        - sequence_number
      type: object
    OpenAIResponsesSearchCompleted:
      example:
        item_id: ws_abc123
        output_index: 0
        sequence_number: 5
        type: response.web_search_call.completed
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.web_search_call.completed
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
      type: object
    OpenAIResponsesWebSearchCallInProgress:
      example:
        item_id: ws_abc123
        output_index: 0
        sequence_number: 1
        type: response.web_search_call.in_progress
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.web_search_call.in_progress
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
      type: object
    OpenAIResponsesWebSearchCallSearching:
      example:
        item_id: ws_abc123
        output_index: 0
        sequence_number: 2
        type: response.web_search_call.searching
      properties:
        item_id:
          type: string
        output_index:
          type: integer
        sequence_number:
          type: integer
        type:
          enum:
            - response.web_search_call.searching
          type: string
      required:
        - type
        - item_id
        - output_index
        - sequence_number
      type: object
    FileCitation:
      example:
        file_id: file-abc123
        filename: research_paper.pdf
        index: 0
        type: file_citation
      properties:
        file_id:
          type: string
        filename:
          type: string
        index:
          type: integer
        type:
          enum:
            - file_citation
          type: string
      required:
        - type
        - file_id
        - filename
        - index
      type: object
    URLCitation:
      example:
        content: >-
          OpenRouter provides a unified API for accessing LLMs from multiple
          providers.
        end_index: 42
        start_index: 0
        title: OpenRouter Documentation
        type: url_citation
        url: https://openrouter.ai/docs
      properties:
        content:
          type: string
        end_index:
          type: integer
        start_index:
          type: integer
        title:
          type: string
        type:
          enum:
            - url_citation
          type: string
        url:
          type: string
      required:
        - type
        - url
        - title
        - start_index
        - end_index
      type: object
    FilePath:
      example:
        file_id: file-xyz789
        index: 0
        type: file_path
      properties:
        file_id:
          type: string
        index:
          type: integer
        type:
          enum:
            - file_path
          type: string
      required:
        - type
        - file_id
        - index
      type: object
    FormatTextConfig:
      description: Plain text response format
      example:
        type: text
      properties:
        type:
          enum:
            - text
          type: string
      required:
        - type
      type: object
    FormatJsonObjectConfig:
      description: JSON object response format
      example:
        type: json_object
      properties:
        type:
          enum:
            - json_object
          type: string
      required:
        - type
      type: object
    FormatJsonSchemaConfig:
      description: JSON schema constrained response format
      example:
        description: User information schema
        name: user_info
        schema:
          properties:
            age:
              type: number
            name:
              type: string
          required:
            - name
          type: object
        type: json_schema
      properties:
        description:
          type: string
        name:
          type: string
        schema:
          additionalProperties: {}
          type: object
        strict:
          type:
            - boolean
            - 'null'
        type:
          enum:
            - json_schema
          type: string
      required:
        - type
        - name
        - schema
      type: object
    ContainerAutoEnvironment:
      description: An OpenRouter-managed, auto-provisioned ephemeral container.
      example:
        type: container_auto
      properties:
        type:
          enum:
            - container_auto
          type: string
      required:
        - type
      type: object
    ContainerReferenceEnvironment:
      description: Reference to a previously created container to reuse.
      example:
        container_id: cntr_abc123
        type: container_reference
      properties:
        container_id:
          description: Identifier of an existing container to reuse (max 20 characters).
          example: cntr_abc123
          maxLength: 20
          minLength: 1
          pattern: ^[\w-]+$
          type: string
        type:
          enum:
            - container_reference
          type: string
      required:
        - type
        - container_id
      type: object
    OpenResponsesLogProbs:
      description: Log probability information for a token
      example:
        logprob: -0.1
        token: world
        top_logprobs:
          - logprob: -0.5
            token: hello
      properties:
        bytes:
          items:
            type: integer
          type: array
        logprob:
          format: double
          type: number
        token:
          type: string
        top_logprobs:
          items:
            $ref: '#/components/schemas/OpenResponsesTopLogprobs'
          type: array
      required:
        - logprob
        - token
      type: object
    StreamLogprobTopLogprob:
      allOf:
        - $ref: '#/components/schemas/OpenResponsesTopLogprobs'
        - properties: {}
          type: object
      description: Alternative token with its log probability
      example:
        bytes:
          - 72
          - 101
          - 108
          - 108
          - 111
        logprob: -0.5
        token: Hello
    OpenResponsesTopLogprobs:
      description: Alternative token with its log probability
      example:
        logprob: -0.5
        token: hello
      properties:
        bytes:
          items:
            type: integer
          type: array
        logprob:
          format: double
          type: number
        token:
          type: string
      type: object
  securitySchemes:
    apiKey:
      description: API key as bearer token in Authorization header
      scheme: bearer
      type: http

````