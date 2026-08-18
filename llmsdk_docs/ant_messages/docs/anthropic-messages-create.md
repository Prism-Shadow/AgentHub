---
title: Create a Message
url: https://platform.claude.com/docs/en/api/typescript/beta/messages/create
---

## Create a Message

`client.beta.messages.create(MessageCreateParamsparams, RequestOptionsoptions?): BetaMessage | Stream<BetaRawMessageStreamEvent>`

**post** `/v1/messages`

Send a structured list of input messages with text and/or image content, and the model will generate the next message in the conversation.

The Messages API can be used for either single queries or stateless multi-turn conversations.

Learn more about the Messages API in our [user guide](https://platform.claude.com/docs/en/get-started)

### Parameters

- `MessageCreateParams = MessageCreateParamsNonStreaming | MessageCreateParamsStreaming`

  - `MessageCreateParamsBase`

    - `max_tokens: number`

      Body param: The maximum number of tokens to generate before stopping.

      Note that our models may stop _before_ reaching this maximum. This parameter only specifies the absolute maximum number of tokens to generate.

      Set to `0` to populate the [prompt cache](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#pre-warming-the-cache) without generating a response.

      Different models have different maximum values for this parameter.  See [models](https://platform.claude.com/docs/en/about-claude/models/overview) for details.

    - `messages: Array<BetaMessageParam>`

      Body param: Input messages.

      Our models are trained to operate on alternating `user` and `assistant` conversational turns. When creating a new `Message`, you specify the prior conversational turns with the `messages` parameter, and the model then generates the next `Message` in the conversation. Consecutive `user` or `assistant` turns in your request will be combined into a single turn.

      Each input message must be an object with a `role` and `content`. You can specify a single `user`-role message, or you can include multiple `user` and `assistant` messages.

      If the final message uses the `assistant` role, the response content will continue immediately from the content in that message. This can be used to constrain part of the model's response.

      Example with a single `user` message:

      ```json
      [{"role": "user", "content": "Hello, Claude"}]
      ```

      Example with multiple conversational turns:

      ```json
      [
        {"role": "user", "content": "Hello there."},
        {"role": "assistant", "content": "Hi, I'm Claude. How can I help you?"},
        {"role": "user", "content": "Can you explain LLMs in plain English?"},
      ]
      ```

      Example with a partially-filled response from Claude:

      ```json
      [
        {"role": "user", "content": "What's the Greek name for Sun? (A) Sol (B) Helios (C) Sun"},
        {"role": "assistant", "content": "The best answer is ("},
      ]
      ```

      Each input message `content` may be either a single `string` or an array of content blocks, where each block has a specific `type`. Using a `string` for `content` is shorthand for an array of one content block of type `"text"`. The following input messages are equivalent:

      ```json
      {"role": "user", "content": "Hello, Claude"}
      ```

      ```json
      {"role": "user", "content": [{"type": "text", "text": "Hello, Claude"}]}
      ```

      See [input examples](https://platform.claude.com/docs/en/build-with-claude/working-with-messages).

      Note that if you want to include a [system prompt](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#give-claude-a-role), you can use the top-level `system` parameter — there is no `"system"` role for input messages in the Messages API.

      There is a limit of 100,000 messages in a single request.

      - `content: string | Array<BetaContentBlockParam>`

        - `string`

        - `Array<BetaContentBlockParam>`

          - `BetaTextBlockParam`

            - `text: string`

            - `type: "text"`

              - `"text"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

              - `type: "ephemeral"`

                - `"ephemeral"`

              - `ttl?: "5m" | "1h"`

                The time-to-live for the cache control breakpoint.

                This may be one the following values:

                - `5m`: 5 minutes
                - `1h`: 1 hour

                Defaults to `5m`. See [prompt caching pricing](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) for details.

                - `"5m"`

                - `"1h"`

            - `citations?: Array<BetaTextCitationParam> | null`

              - `BetaCitationCharLocationParam`

                - `cited_text: string`

                - `document_index: number`

                - `document_title: string | null`

                - `end_char_index: number`

                - `start_char_index: number`

                - `type: "char_location"`

                  - `"char_location"`

              - `BetaCitationPageLocationParam`

                - `cited_text: string`

                - `document_index: number`

                - `document_title: string | null`

                - `end_page_number: number`

                - `start_page_number: number`

                - `type: "page_location"`

                  - `"page_location"`

              - `BetaCitationContentBlockLocationParam`

                - `cited_text: string`

                  The full text of the cited block range, concatenated.

                  Always equals the contents of `content[start_block_index:end_block_index]` joined together. The text block is the minimal citable unit; this field is never a substring of a single block. Not counted toward output tokens, and not counted toward input tokens when sent back in subsequent turns.

                - `document_index: number`

                - `document_title: string | null`

                - `end_block_index: number`

                  Exclusive 0-based end index of the cited block range in the source's `content` array.

                  Always greater than `start_block_index`; a single-block citation has `end_block_index = start_block_index + 1`.

                - `start_block_index: number`

                  0-based index of the first cited block in the source's `content` array.

                - `type: "content_block_location"`

                  - `"content_block_location"`

              - `BetaCitationWebSearchResultLocationParam`

                - `cited_text: string`

                - `encrypted_index: string`

                - `title: string | null`

                - `type: "web_search_result_location"`

                  - `"web_search_result_location"`

                - `url: string`

              - `BetaCitationSearchResultLocationParam`

                - `cited_text: string`

                  The full text of the cited block range, concatenated.

                  Always equals the contents of `content[start_block_index:end_block_index]` joined together. The text block is the minimal citable unit; this field is never a substring of a single block. Not counted toward output tokens, and not counted toward input tokens when sent back in subsequent turns.

                - `end_block_index: number`

                  Exclusive 0-based end index of the cited block range in the source's `content` array.

                  Always greater than `start_block_index`; a single-block citation has `end_block_index = start_block_index + 1`.

                - `search_result_index: number`

                  0-based index of the cited search result among all `search_result` content blocks in the request, in the order they appear across messages and tool results.

                  Counted separately from `document_index`; server-side web search results are not included in this count.

                - `source: string`

                - `start_block_index: number`

                  0-based index of the first cited block in the source's `content` array.

                - `title: string | null`

                - `type: "search_result_location"`

                  - `"search_result_location"`

          - `BetaImageBlockParam`

            - `source: BetaBase64ImageSource | BetaURLImageSource | BetaFileImageSource`

              - `BetaBase64ImageSource`

                - `data: string`

                - `media_type: "image/jpeg" | "image/png" | "image/gif" | "image/webp"`

                  - `"image/jpeg"`

                  - `"image/png"`

                  - `"image/gif"`

                  - `"image/webp"`

                - `type: "base64"`

                  - `"base64"`

              - `BetaURLImageSource`

                - `type: "url"`

                  - `"url"`

                - `url: string`

              - `BetaFileImageSource`

                - `file_id: string`

                - `type: "file"`

                  - `"file"`

            - `type: "image"`

              - `"image"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

          - `BetaRequestDocumentBlock`

            - `source: BetaBase64PDFSource | BetaPlainTextSource | BetaContentBlockSource | 2 more`

              - `BetaBase64PDFSource`

                - `data: string`

                - `media_type: "application/pdf"`

                  - `"application/pdf"`

                - `type: "base64"`

                  - `"base64"`

              - `BetaPlainTextSource`

                - `data: string`

                - `media_type: "text/plain"`

                  - `"text/plain"`

                - `type: "text"`

                  - `"text"`

              - `BetaContentBlockSource`

                - `content: string | Array<BetaContentBlockSourceContent>`

                  - `string`

                  - `Array<BetaContentBlockSourceContent>`

                    - `BetaTextBlockParam`

                    - `BetaImageBlockParam`

                - `type: "content"`

                  - `"content"`

              - `BetaURLPDFSource`

                - `type: "url"`

                  - `"url"`

                - `url: string`

              - `BetaFileDocumentSource`

                - `file_id: string`

                - `type: "file"`

                  - `"file"`

            - `type: "document"`

              - `"document"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

            - `citations?: BetaCitationsConfigParam | null`

              - `enabled?: boolean`

            - `context?: string | null`

            - `title?: string | null`

          - `BetaSearchResultBlockParam`

            - `content: Array<BetaTextBlockParam>`

              - `text: string`

              - `type: "text"`

              - `cache_control?: BetaCacheControlEphemeral | null`

                Create a cache control breakpoint at this content block.

              - `citations?: Array<BetaTextCitationParam> | null`

            - `source: string`

            - `title: string`

            - `type: "search_result"`

              - `"search_result"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

            - `citations?: BetaCitationsConfigParam`

          - `BetaThinkingBlockParam`

            - `signature: string`

              The `signature` value of this thinking block, exactly as returned by the API in a previous response. Used to verify that the block was generated by Claude.

              Thinking blocks must be passed back unmodified and in their original order; a modified block results in a 400 `invalid_request_error`.

            - `thinking: string`

              The `thinking` text of this block as returned by the API.

            - `type: "thinking"`

              - `"thinking"`

          - `BetaRedactedThinkingBlockParam`

            - `data: string`

              The `data` value of this redacted thinking block, exactly as returned by the API in a previous response. Opaque and encrypted; pass it back unchanged.

            - `type: "redacted_thinking"`

              - `"redacted_thinking"`

          - `BetaToolUseBlockParam`

            - `id: string`

            - `input: Record<string, unknown>`

            - `name: string`

            - `type: "tool_use"`

              - `"tool_use"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

            - `caller?: BetaDirectCaller | BetaServerToolCaller | BetaServerToolCaller20260120`

              Tool invocation directly from the model.

              - `BetaDirectCaller`

                Tool invocation directly from the model.

                - `type: "direct"`

                  - `"direct"`

              - `BetaServerToolCaller`

                Tool invocation generated by a server-side tool.

                - `tool_id: string`

                - `type: "code_execution_20250825"`

                  - `"code_execution_20250825"`

              - `BetaServerToolCaller20260120`

                - `tool_id: string`

                - `type: "code_execution_20260120"`

                  - `"code_execution_20260120"`

          - `BetaToolResultBlockParam`

            - `tool_use_id: string`

            - `type: "tool_result"`

              - `"tool_result"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

            - `content?: string | Array<BetaTextBlockParam | BetaImageBlockParam | BetaSearchResultBlockParam | 2 more>`

              - `string`

              - `Array<BetaTextBlockParam | BetaImageBlockParam | BetaSearchResultBlockParam | 2 more>`

                - `BetaTextBlockParam`

                - `BetaImageBlockParam`

                - `BetaSearchResultBlockParam`

                - `BetaRequestDocumentBlock`

                - `BetaToolReferenceBlockParam`

                  Tool reference block that can be included in tool_result content.

                  - `tool_name: string`

                  - `type: "tool_reference"`

                    - `"tool_reference"`

                  - `cache_control?: BetaCacheControlEphemeral | null`

                    Create a cache control breakpoint at this content block.

            - `is_error?: boolean`

          - `BetaServerToolUseBlockParam`

            - `id: string`

            - `input: Record<string, unknown>`

            - `name: "advisor" | "web_search" | "web_fetch" | 5 more`

              - `"advisor"`

              - `"web_search"`

              - `"web_fetch"`

              - `"code_execution"`

              - `"bash_code_execution"`

              - `"text_editor_code_execution"`

              - `"tool_search_tool_regex"`

              - `"tool_search_tool_bm25"`

            - `type: "server_tool_use"`

              - `"server_tool_use"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

            - `caller?: BetaDirectCaller | BetaServerToolCaller | BetaServerToolCaller20260120`

              Tool invocation directly from the model.

              - `BetaDirectCaller`

                Tool invocation directly from the model.

              - `BetaServerToolCaller`

                Tool invocation generated by a server-side tool.

              - `BetaServerToolCaller20260120`

          - `BetaWebSearchToolResultBlockParam`

            - `content: BetaWebSearchToolResultBlockParamContent`

              - `Array<BetaWebSearchResultBlockParam>`

                - `encrypted_content: string`

                - `title: string`

                - `type: "web_search_result"`

                  - `"web_search_result"`

                - `url: string`

                - `page_age?: string | null`

              - `BetaWebSearchToolRequestError`

                - `error_code: BetaWebSearchToolResultErrorCode`

                  - `"invalid_tool_input"`

                  - `"unavailable"`

                  - `"max_uses_exceeded"`

                  - `"too_many_requests"`

                  - `"query_too_long"`

                  - `"request_too_large"`

                - `type: "web_search_tool_result_error"`

                  - `"web_search_tool_result_error"`

            - `tool_use_id: string`

            - `type: "web_search_tool_result"`

              - `"web_search_tool_result"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

            - `caller?: BetaDirectCaller | BetaServerToolCaller | BetaServerToolCaller20260120`

              Tool invocation directly from the model.

              - `BetaDirectCaller`

                Tool invocation directly from the model.

              - `BetaServerToolCaller`

                Tool invocation generated by a server-side tool.

              - `BetaServerToolCaller20260120`

          - `BetaWebFetchToolResultBlockParam`

            - `content: BetaWebFetchToolResultErrorBlockParam | BetaWebFetchBlockParam`

              - `BetaWebFetchToolResultErrorBlockParam`

                - `error_code: BetaWebFetchToolResultErrorCode`

                  - `"invalid_tool_input"`

                  - `"url_too_long"`

                  - `"url_not_allowed"`

                  - `"url_not_in_prior_context"`

                  - `"url_not_accessible"`

                  - `"unsupported_content_type"`

                  - `"too_many_requests"`

                  - `"max_uses_exceeded"`

                  - `"unavailable"`

                - `type: "web_fetch_tool_result_error"`

                  - `"web_fetch_tool_result_error"`

              - `BetaWebFetchBlockParam`

                - `content: BetaRequestDocumentBlock`

                - `type: "web_fetch_result"`

                  - `"web_fetch_result"`

                - `url: string`

                  Fetched content URL

                - `retrieved_at?: string | null`

                  ISO 8601 timestamp when the content was retrieved

            - `tool_use_id: string`

            - `type: "web_fetch_tool_result"`

              - `"web_fetch_tool_result"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

            - `caller?: BetaDirectCaller | BetaServerToolCaller | BetaServerToolCaller20260120`

              Tool invocation directly from the model.

              - `BetaDirectCaller`

                Tool invocation directly from the model.

              - `BetaServerToolCaller`

                Tool invocation generated by a server-side tool.

              - `BetaServerToolCaller20260120`

          - `BetaAdvisorToolResultBlockParam`

            - `content: BetaAdvisorToolResultErrorParam | BetaAdvisorResultBlockParam | BetaAdvisorRedactedResultBlockParam`

              - `BetaAdvisorToolResultErrorParam`

                - `error_code: "max_uses_exceeded" | "prompt_too_long" | "too_many_requests" | 4 more`

                  - `"max_uses_exceeded"`

                  - `"prompt_too_long"`

                  - `"too_many_requests"`

                  - `"overloaded"`

                  - `"unavailable"`

                  - `"execution_time_exceeded"`

                  - `"model_not_found"`

                - `type: "advisor_tool_result_error"`

                  - `"advisor_tool_result_error"`

              - `BetaAdvisorResultBlockParam`

                - `text: string`

                - `type: "advisor_result"`

                  - `"advisor_result"`

                - `stop_reason?: string | null`

              - `BetaAdvisorRedactedResultBlockParam`

                - `encrypted_content: string`

                  Opaque blob produced by a prior response; must be round-tripped verbatim.

                - `type: "advisor_redacted_result"`

                  - `"advisor_redacted_result"`

                - `stop_reason?: string | null`

            - `tool_use_id: string`

            - `type: "advisor_tool_result"`

              - `"advisor_tool_result"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

          - `BetaCodeExecutionToolResultBlockParam`

            - `content: BetaCodeExecutionToolResultBlockParamContent`

              Code execution result with encrypted stdout for PFC + web_search results.

              - `BetaCodeExecutionToolResultErrorParam`

                - `error_code: BetaCodeExecutionToolResultErrorCode`

                  - `"invalid_tool_input"`

                  - `"unavailable"`

                  - `"too_many_requests"`

                  - `"execution_time_exceeded"`

                - `type: "code_execution_tool_result_error"`

                  - `"code_execution_tool_result_error"`

              - `BetaCodeExecutionResultBlockParam`

                - `content: Array<BetaCodeExecutionOutputBlockParam>`

                  - `file_id: string`

                  - `type: "code_execution_output"`

                    - `"code_execution_output"`

                - `return_code: number`

                - `stderr: string`

                - `stdout: string`

                - `type: "code_execution_result"`

                  - `"code_execution_result"`

              - `BetaEncryptedCodeExecutionResultBlockParam`

                Code execution result with encrypted stdout for PFC + web_search results.

                - `content: Array<BetaCodeExecutionOutputBlockParam>`

                  - `file_id: string`

                  - `type: "code_execution_output"`

                - `encrypted_stdout: string`

                - `return_code: number`

                - `stderr: string`

                - `type: "encrypted_code_execution_result"`

                  - `"encrypted_code_execution_result"`

            - `tool_use_id: string`

            - `type: "code_execution_tool_result"`

              - `"code_execution_tool_result"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

          - `BetaBashCodeExecutionToolResultBlockParam`

            - `content: BetaBashCodeExecutionToolResultErrorParam | BetaBashCodeExecutionResultBlockParam`

              - `BetaBashCodeExecutionToolResultErrorParam`

                - `error_code: "invalid_tool_input" | "unavailable" | "too_many_requests" | 2 more`

                  - `"invalid_tool_input"`

                  - `"unavailable"`

                  - `"too_many_requests"`

                  - `"execution_time_exceeded"`

                  - `"output_file_too_large"`

                - `type: "bash_code_execution_tool_result_error"`

                  - `"bash_code_execution_tool_result_error"`

              - `BetaBashCodeExecutionResultBlockParam`

                - `content: Array<BetaBashCodeExecutionOutputBlockParam>`

                  - `file_id: string`

                  - `type: "bash_code_execution_output"`

                    - `"bash_code_execution_output"`

                - `return_code: number`

                - `stderr: string`

                - `stdout: string`

                - `type: "bash_code_execution_result"`

                  - `"bash_code_execution_result"`

            - `tool_use_id: string`

            - `type: "bash_code_execution_tool_result"`

              - `"bash_code_execution_tool_result"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

          - `BetaTextEditorCodeExecutionToolResultBlockParam`

            - `content: BetaTextEditorCodeExecutionToolResultErrorParam | BetaTextEditorCodeExecutionViewResultBlockParam | BetaTextEditorCodeExecutionCreateResultBlockParam | BetaTextEditorCodeExecutionStrReplaceResultBlockParam`

              - `BetaTextEditorCodeExecutionToolResultErrorParam`

                - `error_code: "invalid_tool_input" | "unavailable" | "too_many_requests" | 2 more`

                  - `"invalid_tool_input"`

                  - `"unavailable"`

                  - `"too_many_requests"`

                  - `"execution_time_exceeded"`

                  - `"file_not_found"`

                - `type: "text_editor_code_execution_tool_result_error"`

                  - `"text_editor_code_execution_tool_result_error"`

                - `error_message?: string | null`

              - `BetaTextEditorCodeExecutionViewResultBlockParam`

                - `content: string`

                - `file_type: "text" | "image" | "pdf"`

                  - `"text"`

                  - `"image"`

                  - `"pdf"`

                - `type: "text_editor_code_execution_view_result"`

                  - `"text_editor_code_execution_view_result"`

                - `num_lines?: number | null`

                - `start_line?: number | null`

                - `total_lines?: number | null`

              - `BetaTextEditorCodeExecutionCreateResultBlockParam`

                - `is_file_update: boolean`

                - `type: "text_editor_code_execution_create_result"`

                  - `"text_editor_code_execution_create_result"`

              - `BetaTextEditorCodeExecutionStrReplaceResultBlockParam`

                - `type: "text_editor_code_execution_str_replace_result"`

                  - `"text_editor_code_execution_str_replace_result"`

                - `lines?: Array<string> | null`

                - `new_lines?: number | null`

                - `new_start?: number | null`

                - `old_lines?: number | null`

                - `old_start?: number | null`

            - `tool_use_id: string`

            - `type: "text_editor_code_execution_tool_result"`

              - `"text_editor_code_execution_tool_result"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

          - `BetaToolSearchToolResultBlockParam`

            - `content: BetaToolSearchToolResultErrorParam | BetaToolSearchToolSearchResultBlockParam`

              - `BetaToolSearchToolResultErrorParam`

                - `error_code: "invalid_tool_input" | "unavailable" | "too_many_requests" | "execution_time_exceeded"`

                  - `"invalid_tool_input"`

                  - `"unavailable"`

                  - `"too_many_requests"`

                  - `"execution_time_exceeded"`

                - `type: "tool_search_tool_result_error"`

                  - `"tool_search_tool_result_error"`

                - `error_message?: string | null`

              - `BetaToolSearchToolSearchResultBlockParam`

                - `tool_references: Array<BetaToolReferenceBlockParam>`

                  - `tool_name: string`

                  - `type: "tool_reference"`

                  - `cache_control?: BetaCacheControlEphemeral | null`

                    Create a cache control breakpoint at this content block.

                - `type: "tool_search_tool_search_result"`

                  - `"tool_search_tool_search_result"`

            - `tool_use_id: string`

            - `type: "tool_search_tool_result"`

              - `"tool_search_tool_result"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

          - `BetaMCPToolUseBlockParam`

            - `id: string`

            - `input: Record<string, unknown>`

            - `name: string`

            - `server_name: string`

              The name of the MCP server

            - `type: "mcp_tool_use"`

              - `"mcp_tool_use"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

          - `BetaRequestMCPToolResultBlockParam`

            - `tool_use_id: string`

            - `type: "mcp_tool_result"`

              - `"mcp_tool_result"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

            - `content?: string | Array<BetaTextBlockParam>`

              - `string`

              - `Array<BetaTextBlockParam>`

                - `text: string`

                - `type: "text"`

                - `cache_control?: BetaCacheControlEphemeral | null`

                  Create a cache control breakpoint at this content block.

                - `citations?: Array<BetaTextCitationParam> | null`

            - `is_error?: boolean`

          - `BetaContainerUploadBlockParam`

            A content block that represents a file to be uploaded to the container
            Files uploaded via this block will be available in the container's input directory.

            - `file_id: string`

            - `type: "container_upload"`

              - `"container_upload"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

          - `BetaCompactionBlockParam`

            A compaction block containing summary of previous context.

            Users should round-trip these blocks from responses to subsequent requests
            to maintain context across compaction boundaries.

            When content is None, the block represents a failed compaction. The server
            treats these as no-ops. Empty string content is not allowed.

            - `type: "compaction"`

              - `"compaction"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

            - `content?: string | null`

              Summary of previously compacted content, or null if compaction failed

            - `encrypted_content?: string | null`

              Opaque metadata from prior compaction, to be round-tripped verbatim

          - `BetaMidConversationSystemBlockParam`

            System instructions that appear mid-conversation.

            Use this block to provide or update system-level instructions at a specific
            point in the conversation, rather than only via the top-level `system` parameter.

            - `content: Array<BetaTextBlockParam | BetaRequestToolAdditionBlock | BetaRequestToolRemovalBlock>`

              System instruction text blocks.

              - `BetaTextBlockParam`

              - `BetaRequestToolAdditionBlock`

                Mid-conversation directive to surface a declared tool.

                `tool` references a tool (or MCP toolset) by name from the request's
                `tools`; it is offered to the model from this point in the
                conversation onward.

                - `tool: BetaToolChangeToolReference | BetaToolChangeMCPToolReference | BetaToolChangeMCPToolsetReference`

                  Reference to a single tool the caller declared directly in
                  `tools[]`. Does not accept the composed `{server}_{name}` form the
                  server assigns to MCP-resolved tools — use `mcp_tool_reference` or
                  `mcp_toolset_reference` for those.

                  - `BetaToolChangeToolReference`

                    Reference to a single tool the caller declared directly in
                    `tools[]`. Does not accept the composed `{server}_{name}` form the
                    server assigns to MCP-resolved tools — use `mcp_tool_reference` or
                    `mcp_toolset_reference` for those.

                    - `name: string`

                    - `type: "tool_reference"`

                      - `"tool_reference"`

                  - `BetaToolChangeMCPToolReference`

                    Reference to a single MCP tool by its server and remote name — the
                    same `server_name`/`name` pair `mcp_tool_use` carries.

                    - `name: string`

                    - `server_name: string`

                    - `type: "mcp_tool_reference"`

                      - `"mcp_tool_reference"`

                  - `BetaToolChangeMCPToolsetReference`

                    Reference to every tool in the named MCP server's toolset.

                    - `server_name: string`

                    - `type: "mcp_toolset_reference"`

                      - `"mcp_toolset_reference"`

                - `type: "tool_addition"`

                  - `"tool_addition"`

                - `cache_control?: BetaCacheControlEphemeral | null`

                  Create a cache control breakpoint at this content block.

              - `BetaRequestToolRemovalBlock`

                Mid-conversation directive to withdraw a tool.

                `tool` references a tool (or MCP toolset) by name from the request's
                `tools`; it is no longer offered to the model from this point in the
                conversation onward.

                - `tool: BetaToolChangeToolReference | BetaToolChangeMCPToolReference | BetaToolChangeMCPToolsetReference`

                  Reference to a single tool the caller declared directly in
                  `tools[]`. Does not accept the composed `{server}_{name}` form the
                  server assigns to MCP-resolved tools — use `mcp_tool_reference` or
                  `mcp_toolset_reference` for those.

                  - `BetaToolChangeToolReference`

                    Reference to a single tool the caller declared directly in
                    `tools[]`. Does not accept the composed `{server}_{name}` form the
                    server assigns to MCP-resolved tools — use `mcp_tool_reference` or
                    `mcp_toolset_reference` for those.

                  - `BetaToolChangeMCPToolReference`

                    Reference to a single MCP tool by its server and remote name — the
                    same `server_name`/`name` pair `mcp_tool_use` carries.

                  - `BetaToolChangeMCPToolsetReference`

                    Reference to every tool in the named MCP server's toolset.

                - `type: "tool_removal"`

                  - `"tool_removal"`

                - `cache_control?: BetaCacheControlEphemeral | null`

                  Create a cache control breakpoint at this content block.

            - `type: "mid_conv_system"`

              - `"mid_conv_system"`

            - `cache_control?: BetaCacheControlEphemeral | null`

              Create a cache control breakpoint at this content block.

          - `BetaRequestToolAdditionBlock`

            Mid-conversation directive to surface a declared tool.

            `tool` references a tool (or MCP toolset) by name from the request's
            `tools`; it is offered to the model from this point in the
            conversation onward.

          - `BetaRequestToolRemovalBlock`

            Mid-conversation directive to withdraw a tool.

            `tool` references a tool (or MCP toolset) by name from the request's
            `tools`; it is no longer offered to the model from this point in the
            conversation onward.

          - `BetaFallbackBlockParam`

            A `fallback` block echoed back from a prior response.

            Accepted in `messages[].content` and not rendered into the prompt; not
            validated against the request's `fallbacks` chain or top-level `model`.

            Echo the assistant turn back verbatim, including this block in its
            original position. The block marks the boundary between content produced
            before and after a fallback hop, and the server relies on that boundary
            to validate the turn: when thinking runs flank the boundary, omitting
            the block merges them into one span the server cannot validate (the
            request is rejected), and moving it into the middle of a single run is
            likewise rejected; between non-thinking blocks the block's placement has
            no validation effect.

            - `from: BetaFallbackInfoParam`

              Identifies one hop of a fallback transition.

              - `model: Model`

                The model that will complete your prompt.

                See [models](https://docs.anthropic.com/en/docs/models-overview) for additional details and options.

                - `"claude-sonnet-5" | "claude-fable-5" | "claude-mythos-5" | 12 more`

                  - `"claude-sonnet-5"`

                    High-performance model for coding and agents

                  - `"claude-fable-5"`

                    Next generation of intelligence for the hardest knowledge work and coding problems

                  - `"claude-mythos-5"`

                    Most capable model for cybersecurity and biology research

                  - `"claude-opus-5"`

                    Powerful intelligence for long-running agents and coding

                  - `"claude-opus-4-8"`

                    Powerful intelligence for long-running agents and coding

                  - `"claude-opus-4-7"`

                    Powerful intelligence for long-running agents and coding

                  - `"claude-mythos-preview"`

                    New class of intelligence, strongest in coding and cybersecurity

                  - `"claude-opus-4-6"`

                    Powerful intelligence for long-running agents and coding

                  - `"claude-sonnet-4-6"`

                    Best combination of speed and intelligence

                  - `"claude-haiku-4-5"`

                    Fastest model with near-frontier intelligence

                  - `"claude-haiku-4-5-20251001"`

                    Fastest model with near-frontier intelligence

                  - `"claude-opus-4-5"`

                    Powerful intelligence for long-running agents and coding

                  - `"claude-opus-4-5-20251101"`

                    Powerful intelligence for long-running agents and coding

                  - `"claude-sonnet-4-5"`

                    High-performance model for agents and coding

                  - `"claude-sonnet-4-5-20250929"`

                    High-performance model for agents and coding

                - `(string & {})`

            - `to: BetaFallbackInfoParam`

              Identifies one hop of a fallback transition.

            - `type: "fallback"`

              - `"fallback"`

            - `trigger?: unknown`

              The response block's `trigger`, echoed verbatim. Accepted and ignored by the server; any object or `null` is allowed.

      - `role: "user" | "assistant" | "system"`

        - `"user"`

        - `"assistant"`

        - `"system"`

    - `model: Model`

      Body param: The model that will complete your prompt.

      See [models](https://docs.anthropic.com/en/docs/models-overview) for additional details and options.

    - `cache_control?: BetaCacheControlEphemeral | null`

      Body param: Top-level cache control automatically applies a cache_control marker to the last cacheable block in the request.

    - `container?: BetaContainerParams | string | null`

      Body param: Container identifier for reuse across requests.

      - `BetaContainerParams`

        Container parameters with skills to be loaded.

        - `id?: string | null`

          Container id

        - `skills?: Array<BetaSkillParams> | null`

          List of skills to load in the container

          - `skill_id: string`

            Skill ID

          - `type: "anthropic" | "custom"`

            Type of skill - either 'anthropic' (built-in) or 'custom' (user-defined)

            - `"anthropic"`

            - `"custom"`

          - `version?: string`

            Skill version or 'latest' for most recent version

      - `string`

    - `context_management?: BetaContextManagementConfig | null`

      Body param: Context management configuration.

      This allows you to control how Claude manages context across multiple requests, such as whether to clear function results or not.

      - `edits?: Array<BetaClearToolUses20250919Edit | BetaClearThinking20251015Edit | BetaCompact20260112Edit>`

        List of context management edits to apply

        - `BetaClearToolUses20250919Edit`

          - `type: "clear_tool_uses_20250919"`

            - `"clear_tool_uses_20250919"`

          - `clear_at_least?: BetaInputTokensClearAtLeast | null`

            Minimum number of tokens that must be cleared when triggered. Context will only be modified if at least this many tokens can be removed.

            - `type: "input_tokens"`

              - `"input_tokens"`

            - `value: number`

          - `clear_tool_inputs?: boolean | Array<string> | null`

            Whether to clear all tool inputs (bool) or specific tool inputs to clear (list)

            - `boolean`

            - `Array<string>`

          - `exclude_tools?: Array<string> | null`

            Tool names whose uses are preserved from clearing

          - `keep?: BetaToolUsesKeep`

            Number of tool uses to retain in the conversation

            - `type: "tool_uses"`

              - `"tool_uses"`

            - `value: number`

          - `trigger?: BetaInputTokensTrigger | BetaToolUsesTrigger`

            Condition that triggers the context management strategy

            - `BetaInputTokensTrigger`

              - `type: "input_tokens"`

                - `"input_tokens"`

              - `value: number`

            - `BetaToolUsesTrigger`

              - `type: "tool_uses"`

                - `"tool_uses"`

              - `value: number`

        - `BetaClearThinking20251015Edit`

          - `type: "clear_thinking_20251015"`

            - `"clear_thinking_20251015"`

          - `keep?: BetaThinkingTurns | BetaAllThinkingTurns | "all"`

            Number of most recent assistant turns to keep thinking blocks for. Older turns will have their thinking blocks removed.

            - `BetaThinkingTurns`

              - `type: "thinking_turns"`

                - `"thinking_turns"`

              - `value: number`

            - `BetaAllThinkingTurns`

              - `type: "all"`

                - `"all"`

            - `"all"`

              - `"all"`

        - `BetaCompact20260112Edit`

          Automatically compact older context when reaching the configured trigger threshold.

          - `type: "compact_20260112"`

            - `"compact_20260112"`

          - `instructions?: string | null`

            Additional instructions for summarization.

          - `pause_after_compaction?: boolean`

            Whether to pause after compaction and return the compaction block to the user.

          - `trigger?: BetaInputTokensTrigger | null`

            When to trigger compaction. Defaults to 150000 input tokens.

    - `diagnostics?: BetaDiagnosticsParam | null`

      Body param: Request-level diagnostics. Currently carries the previous response
      id for prompt-cache divergence reporting.

      - `previous_message_id?: string | null`

        The `id` (`msg_...`) from this client's previous /v1/messages response. The server compares that request's prompt fingerprint against this one and returns `diagnostics.cache_miss_reason` when the prompt-cache prefix could not be reused. Pass `null` on the first turn to opt in without a prior message to compare.

    - `fallback_credit_token?: string | BetaFallbackCreditTokenParam | null`

      Body param: The `fallback_credit_token` from a prior refusal's `stop_details`.

      When a preceding request was refused and returned a `fallback_credit_token`,
      pass that code here on the retry to have the retry's cache-creation tokens
      for the prefix that was warm on the refused model billed at the cache-read
      rate. Must be redeemed by the same organization and workspace, with the same
      request body (optionally extended by one appended `assistant` message whose
      content is the partial text — with any trailing whitespace stripped from
      the final text block — and paired server-tool blocks streamed before the
      refusal; the appended-assistant form is not available for requests with
      `output_format` set or forced `tool_choice`), on an eligible fallback
      model, on the same platform,
      and within 5 minutes of the refusal; a mismatch is a 400. A token minted
      mid-server-tool-loop whose partial content was continuable may only be
      redeemed with the appended-assistant form — if an exact-body retry is
      rejected with a 400 saying the token must be redeemed by continuing the
      partial response, retry with the appended-assistant form instead.

      When the appended-assistant form is used on a model that otherwise disallows
      assistant-turn prefill, this token also authorizes that one prefill.

      - `string`

      - `BetaFallbackCreditTokenParam`

        Object form of `fallback_credit_token`: the token plus a redemption
        mode.

        Requires `anthropic-beta: fallback-credit-2026-07-01`; without that
        header the field accepts the bare string only. The bare string and the
        mode-less object are equivalent (both select `strict`), so wrapping
        an existing token changes nothing by itself.

        - `token: string`

          The opaque `fallback_credit_token` from a prior refusal's `stop_details` — the same string the bare-string form carries.

        - `mode?: "strict" | "best_effort"`

          How a failing token affects the retry. `strict` (the default, and the bare-string behavior): a failing redemption is a 400 and the retry is not served. `best_effort`: the retry is served either way — a token-layer failure no longer rejects the request; the retry proceeds at normal price and the outcome is reported on the response's `usage.fallback_credit`. Two failures stay hard in both modes: a malformed token, and combining `fallback_credit_token` with `fallbacks`.

          - `"strict"`

          - `"best_effort"`

    - `fallbacks?: BetaFallbacksParam | null`

      Body param: Opt-in server-side retry on one or more substitute models when the requested model declines for policy reasons. Tried in order: if the first entry also declines, the second is tried, and so on. The string "default" requests the requested model's server-defined default fallback configuration.

      - `Array<BetaFallbackParam>`

        - `model: Model`

          The model that will complete your prompt.

          See [models](https://docs.anthropic.com/en/docs/models-overview) for additional details and options.

        - `max_tokens?: number | null`

        - `output_config?: BetaOutputConfig | null`

          - `effort?: "low" | "medium" | "high" | 2 more | null`

            All possible effort levels.

            - `"low"`

            - `"medium"`

            - `"high"`

            - `"xhigh"`

            - `"max"`

          - `format?: BetaJSONOutputFormat | null`

            A schema to specify Claude's output format in responses. See [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)

            - `schema: Record<string, unknown>`

              The JSON schema of the format

            - `type: "json_schema"`

              - `"json_schema"`

          - `task_budget?: BetaTokenTaskBudget | null`

            User-configurable total token budget across contexts.

            - `total: number`

              Total token budget across all contexts in the session.

            - `type: "tokens"`

              The budget type. Currently only 'tokens' is supported.

              - `"tokens"`

            - `remaining?: number | null`

              Remaining tokens in the budget. Use this to track usage across contexts when implementing compaction client-side. Defaults to total if not provided.

        - `speed?: "standard" | "fast" | null`

          Inference speed mode. `fast` provides significantly faster output token generation at premium pricing. Not all models support `fast`; invalid combinations are rejected at create time.

          - `"standard"`

          - `"fast"`

        - `thinking?: BetaThinkingConfigEnabled | BetaThinkingConfigDisabled | BetaThinkingConfigAdaptive | null`

          - `BetaThinkingConfigEnabled`

            - `budget_tokens: number`

              Determines how many tokens Claude can use for its internal reasoning process. Larger budgets can enable more thorough analysis for complex problems, improving response quality.

              Must be ≥1024 and less than `max_tokens`.

              See [extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) for details.

            - `type: "enabled"`

              - `"enabled"`

            - `display?: "summarized" | "omitted" | null`

              Controls how thinking content appears in the response. When set to `summarized`, thinking is returned normally. When set to `omitted`, thinking content is redacted but a signature is returned for multi-turn continuity. Defaults to `summarized`.

              - `"summarized"`

              - `"omitted"`

          - `BetaThinkingConfigDisabled`

            - `type: "disabled"`

              - `"disabled"`

          - `BetaThinkingConfigAdaptive`

            - `type: "adaptive"`

              - `"adaptive"`

            - `display?: "summarized" | "omitted" | null`

              Controls how thinking content appears in the response. When set to `summarized`, thinking is returned normally. When set to `omitted`, thinking content is redacted but a signature is returned for multi-turn continuity. Defaults to `summarized`.

              - `"summarized"`

              - `"omitted"`

      - `"default"`

        - `"default"`

    - `inference_geo?: string | null`

      Body param: Specifies the geographic region for inference processing. If not specified, the workspace's `default_inference_geo` is used.

    - `mcp_servers?: Array<BetaRequestMCPServerURLDefinition>`

      Body param: MCP servers to be utilized in this request

      - `name: string`

      - `type: "url"`

        - `"url"`

      - `url: string`

      - `authorization_token?: string | null`

      - `tool_configuration?: BetaRequestMCPServerToolConfiguration | null`

        - `allowed_tools?: Array<string> | null`

        - `enabled?: boolean | null`

    - `metadata?: BetaMetadata`

      Body param: An object describing metadata about the request.

      - `user_id?: string | null`

        An external identifier for the user who is associated with the request.

        This should be a uuid, hash value, or other opaque identifier. Anthropic may use this id to help detect abuse. Do not include any identifying information such as name, email address, or phone number.

    - `output_config?: BetaOutputConfig`

      Body param: Configuration options for the model's output, such as the output format.

    - `output_format?: BetaJSONOutputFormat | null`

      Body param: Deprecated: Use `output_config.format` instead. See [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)

      A schema to specify Claude's output format in responses. This parameter will be removed in a future release.

    - `service_tier?: "auto" | "standard_only"`

      Body param: Determines whether to use priority capacity (if available) or standard capacity for this request.

      Anthropic offers different levels of service for your API requests. See [service-tiers](https://platform.claude.com/docs/en/api/service-tiers) for details.

      - `"auto"`

      - `"standard_only"`

    - `speed?: "standard" | "fast" | null`

      Body param: Inference speed mode. `fast` provides significantly faster output token generation at premium pricing. Not all models support `fast`; invalid combinations are rejected at create time.

      - `"standard"`

      - `"fast"`

    - `stop_sequences?: Array<string>`

      Body param: Custom text sequences that will cause the model to stop generating.

      Our models will normally stop when they have naturally completed their turn, which will result in a response `stop_reason` of `"end_turn"`.

      If you want the model to stop generating when it encounters custom strings of text, you can use the `stop_sequences` parameter. If the model encounters one of the custom sequences, the response `stop_reason` value will be `"stop_sequence"` and the response `stop_sequence` value will contain the matched stop sequence.

    - `stream?: false`

      Body param: Whether to incrementally stream the response using server-sent events.

      See [streaming](https://platform.claude.com/docs/en/build-with-claude/streaming) for details.

      - `false`

    - `system?: string | Array<BetaTextBlockParam>`

      Body param: System prompt.

      A system prompt is a way of providing context and instructions to Claude, such as specifying a particular goal or role. See our [guide to system prompts](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#give-claude-a-role).

      - `string`

      - `Array<BetaTextBlockParam>`

        - `text: string`

        - `type: "text"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `citations?: Array<BetaTextCitationParam> | null`

    - `temperature?: number`

      Body param: Amount of randomness injected into the response.

      Defaults to `1.0`. Ranges from `0.0` to `1.0`. Use `temperature` closer to `0.0` for analytical / multiple choice, and closer to `1.0` for creative and generative tasks.

      Note that even with `temperature` of `0.0`, the results will not be fully deterministic.

    - `thinking?: BetaThinkingConfigParam`

      Body param: Configuration for enabling Claude's extended thinking.

      When enabled, responses include `thinking` content blocks showing Claude's thinking process before the final answer. Requires a minimum budget of 1,024 tokens and counts towards your `max_tokens` limit.

      See [extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) for details.

      - `BetaThinkingConfigEnabled`

      - `BetaThinkingConfigDisabled`

      - `BetaThinkingConfigAdaptive`

    - `tool_choice?: BetaToolChoice`

      Body param: How the model should use the provided tools. The model can use a specific tool, any available tool, decide by itself, or not use tools at all.

      - `BetaToolChoiceAuto`

        The model will automatically decide whether to use tools.

        - `type: "auto"`

          - `"auto"`

        - `disable_parallel_tool_use?: boolean`

          Whether to disable parallel tool use.

          Defaults to `false`. If set to `true`, the model will output at most one tool use.

      - `BetaToolChoiceAny`

        The model will use any available tools.

        - `type: "any"`

          - `"any"`

        - `disable_parallel_tool_use?: boolean`

          Whether to disable parallel tool use.

          Defaults to `false`. If set to `true`, the model will output exactly one tool use.

      - `BetaToolChoiceTool`

        The model will use the specified tool with `tool_choice.name`.

        - `name: string`

          The name of the tool to use.

        - `type: "tool"`

          - `"tool"`

        - `disable_parallel_tool_use?: boolean`

          Whether to disable parallel tool use.

          Defaults to `false`. If set to `true`, the model will output exactly one tool use.

      - `BetaToolChoiceNone`

        The model will not be allowed to use tools.

        - `type: "none"`

          - `"none"`

    - `tools?: Array<BetaToolUnion>`

      Body param: Definitions of tools that the model may use.

      If you include `tools` in your API request, the model may return `tool_use` content blocks that represent the model's use of those tools. You can then run those tools using the tool input generated by the model and then optionally return results back to the model using `tool_result` content blocks.

      There are two types of tools: **client tools** and **server tools**. The behavior described below applies to client tools. For [server tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools), see their individual documentation as each has its own behavior (e.g., the [web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)).

      Each tool definition includes:

      * `name`: Name of the tool.
      * `description`: Optional, but strongly-recommended description of the tool.
      * `input_schema`: [JSON schema](https://json-schema.org/draft/2020-12) for the tool `input` shape that the model will produce in `tool_use` output content blocks.

      For example, if you defined `tools` as:

      ```json
      [
        {
          "name": "get_stock_price",
          "description": "Get the current stock price for a given ticker symbol.",
          "input_schema": {
            "type": "object",
            "properties": {
              "ticker": {
                "type": "string",
                "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
              }
            },
            "required": ["ticker"]
          }
        }
      ]
      ```

      And then asked the model "What's the S&P 500 at today?", the model might produce `tool_use` content blocks in the response like this:

      ```json
      [
        {
          "type": "tool_use",
          "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
          "name": "get_stock_price",
          "input": { "ticker": "^GSPC" }
        }
      ]
      ```

      You might then run your `get_stock_price` tool with `{"ticker": "^GSPC"}` as an input, and return the following back to the model in a subsequent `user` message:

      ```json
      [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
          "content": "259.75 USD"
        }
      ]
      ```

      Tools can be used for workflows that include running client-side tools and functions, or more generally whenever you want the model to produce a particular JSON structure of output.

      See our [guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) for more details.

      - `BetaTool`

        - `input_schema: InputSchema`

          [JSON schema](https://json-schema.org/draft/2020-12) for this tool's input.

          This defines the shape of the `input` that your tool accepts and that the model will produce.

          - `type: "object"`

            - `"object"`

          - `properties?: Record<string, unknown> | null`

          - `required?: Array<string> | null`

        - `name: string`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `description?: string`

          Description of what this tool does.

          Tool descriptions should be as detailed as possible. The more information that the model has about what the tool is and how to use it, the better it will perform. You can use natural language descriptions to reinforce important aspects of the tool input JSON schema.

        - `eager_input_streaming?: boolean | null`

          Enable eager input streaming for this tool. When true, tool input parameters will be streamed incrementally as they are generated, and types will be inferred on-the-fly rather than buffering the full JSON output. When false, streaming is disabled for this tool even if the fine-grained-tool-streaming beta is active. When null (default), uses the default behavior based on beta headers.

        - `input_examples?: Array<Record<string, unknown>>`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

        - `type?: "custom" | null`

          - `"custom"`

      - `BetaToolBash20241022`

        - `name: "bash"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"bash"`

        - `type: "bash_20241022"`

          - `"bash_20241022"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `input_examples?: Array<Record<string, unknown>>`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaToolBash20250124`

        - `name: "bash"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"bash"`

        - `type: "bash_20250124"`

          - `"bash_20250124"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `input_examples?: Array<Record<string, unknown>>`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaCodeExecutionTool20250522`

        - `name: "code_execution"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"code_execution"`

        - `type: "code_execution_20250522"`

          - `"code_execution_20250522"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaCodeExecutionTool20250825`

        - `name: "code_execution"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"code_execution"`

        - `type: "code_execution_20250825"`

          - `"code_execution_20250825"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaCodeExecutionTool20260120`

        Code execution tool with REPL state persistence (daemon mode + gVisor checkpoint).

        - `name: "code_execution"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"code_execution"`

        - `type: "code_execution_20260120"`

          - `"code_execution_20260120"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaCodeExecutionTool20260521`

        Code execution tool with REPL state persistence.

        - `name: "code_execution"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"code_execution"`

        - `type: "code_execution_20260521"`

          - `"code_execution_20260521"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaToolComputerUse20241022`

        - `display_height_px: number`

          The height of the display in pixels.

        - `display_width_px: number`

          The width of the display in pixels.

        - `name: "computer"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"computer"`

        - `type: "computer_20241022"`

          - `"computer_20241022"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `display_number?: number | null`

          The X11 display number (e.g. 0, 1) for the display.

        - `input_examples?: Array<Record<string, unknown>>`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaMemoryTool20250818`

        - `name: "memory"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"memory"`

        - `type: "memory_20250818"`

          - `"memory_20250818"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `input_examples?: Array<Record<string, unknown>>`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaToolComputerUse20250124`

        - `display_height_px: number`

          The height of the display in pixels.

        - `display_width_px: number`

          The width of the display in pixels.

        - `name: "computer"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"computer"`

        - `type: "computer_20250124"`

          - `"computer_20250124"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `display_number?: number | null`

          The X11 display number (e.g. 0, 1) for the display.

        - `input_examples?: Array<Record<string, unknown>>`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaToolTextEditor20241022`

        - `name: "str_replace_editor"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"str_replace_editor"`

        - `type: "text_editor_20241022"`

          - `"text_editor_20241022"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `input_examples?: Array<Record<string, unknown>>`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaToolComputerUse20251124`

        - `display_height_px: number`

          The height of the display in pixels.

        - `display_width_px: number`

          The width of the display in pixels.

        - `name: "computer"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"computer"`

        - `type: "computer_20251124"`

          - `"computer_20251124"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `display_number?: number | null`

          The X11 display number (e.g. 0, 1) for the display.

        - `enable_zoom?: boolean`

          Whether to enable an action to take a zoomed-in screenshot of the screen.

        - `input_examples?: Array<Record<string, unknown>>`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaToolTextEditor20250124`

        - `name: "str_replace_editor"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"str_replace_editor"`

        - `type: "text_editor_20250124"`

          - `"text_editor_20250124"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `input_examples?: Array<Record<string, unknown>>`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaToolTextEditor20250429`

        - `name: "str_replace_based_edit_tool"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"str_replace_based_edit_tool"`

        - `type: "text_editor_20250429"`

          - `"text_editor_20250429"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `input_examples?: Array<Record<string, unknown>>`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaToolTextEditor20250728`

        - `name: "str_replace_based_edit_tool"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"str_replace_based_edit_tool"`

        - `type: "text_editor_20250728"`

          - `"text_editor_20250728"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `input_examples?: Array<Record<string, unknown>>`

        - `max_characters?: number | null`

          Maximum number of characters to display when viewing a file. If not specified, defaults to displaying the full file.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaWebSearchTool20250305`

        - `name: "web_search"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"web_search"`

        - `type: "web_search_20250305"`

          - `"web_search_20250305"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `allowed_domains?: Array<string> | null`

          If provided, only these domains will be included in results. Cannot be used alongside `blocked_domains`.

        - `blocked_domains?: Array<string> | null`

          If provided, these domains will never appear in results. Cannot be used alongside `allowed_domains`.

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `max_uses?: number | null`

          Maximum number of times the tool can be used in the API request.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

        - `user_location?: BetaUserLocation | null`

          Parameters for the user's location. Used to provide more relevant search results.

          - `type: "approximate"`

            - `"approximate"`

          - `city?: string | null`

            The city of the user.

          - `country?: string | null`

            The two letter [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the user.

          - `region?: string | null`

            The region of the user.

          - `timezone?: string | null`

            The [IANA timezone](https://nodatime.org/TimeZones) of the user.

      - `BetaWebFetchTool20250910`

        - `name: "web_fetch"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"web_fetch"`

        - `type: "web_fetch_20250910"`

          - `"web_fetch_20250910"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `allowed_domains?: Array<string> | null`

          List of domains to allow fetching from

        - `blocked_domains?: Array<string> | null`

          List of domains to block fetching from

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `citations?: BetaCitationsConfigParam | null`

          Citations configuration for fetched documents. Citations are disabled by default.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `max_content_tokens?: number | null`

          Maximum number of tokens used by including web page text content in the context. The limit is approximate and does not apply to binary content such as PDFs.

        - `max_uses?: number | null`

          Maximum number of times the tool can be used in the API request.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaWebSearchTool20260209`

        - `name: "web_search"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"web_search"`

        - `type: "web_search_20260209"`

          - `"web_search_20260209"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `allowed_domains?: Array<string> | null`

          If provided, only these domains will be included in results. Cannot be used alongside `blocked_domains`.

        - `blocked_domains?: Array<string> | null`

          If provided, these domains will never appear in results. Cannot be used alongside `allowed_domains`.

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `max_uses?: number | null`

          Maximum number of times the tool can be used in the API request.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

        - `user_location?: BetaUserLocation | null`

          Parameters for the user's location. Used to provide more relevant search results.

      - `BetaWebFetchTool20260209`

        - `name: "web_fetch"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"web_fetch"`

        - `type: "web_fetch_20260209"`

          - `"web_fetch_20260209"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `allowed_domains?: Array<string> | null`

          List of domains to allow fetching from

        - `blocked_domains?: Array<string> | null`

          List of domains to block fetching from

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `citations?: BetaCitationsConfigParam | null`

          Citations configuration for fetched documents. Citations are disabled by default.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `max_content_tokens?: number | null`

          Maximum number of tokens used by including web page text content in the context. The limit is approximate and does not apply to binary content such as PDFs.

        - `max_uses?: number | null`

          Maximum number of times the tool can be used in the API request.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaWebFetchTool20260309`

        Web fetch tool with use_cache parameter for bypassing cached content.

        - `name: "web_fetch"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"web_fetch"`

        - `type: "web_fetch_20260309"`

          - `"web_fetch_20260309"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `allowed_domains?: Array<string> | null`

          List of domains to allow fetching from

        - `blocked_domains?: Array<string> | null`

          List of domains to block fetching from

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `citations?: BetaCitationsConfigParam | null`

          Citations configuration for fetched documents. Citations are disabled by default.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `max_content_tokens?: number | null`

          Maximum number of tokens used by including web page text content in the context. The limit is approximate and does not apply to binary content such as PDFs.

        - `max_uses?: number | null`

          Maximum number of times the tool can be used in the API request.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

        - `use_cache?: boolean`

          Whether to use cached content. Set to false to bypass the cache and fetch fresh content. Only set to false when the user explicitly requests fresh content or when fetching rapidly-changing sources.

      - `BetaWebSearchTool20260318`

        - `name: "web_search"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"web_search"`

        - `type: "web_search_20260318"`

          - `"web_search_20260318"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `allowed_domains?: Array<string> | null`

          If provided, only these domains will be included in results. Cannot be used alongside `blocked_domains`.

        - `blocked_domains?: Array<string> | null`

          If provided, these domains will never appear in results. Cannot be used alongside `allowed_domains`.

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `max_uses?: number | null`

          Maximum number of times the tool can be used in the API request.

        - `response_inclusion?: "full" | "excluded"`

          How this tool's result blocks appear in the API response when the result was consumed by a completed code_execution call in the same turn. 'full' returns the complete content (default). 'excluded' drops the nested server_tool_use and result block pair entirely. Results from direct calls, or from code_execution calls that paused before completing, are always returned in full so they can be sent back on the next turn.

          - `"full"`

          - `"excluded"`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

        - `user_location?: BetaUserLocation | null`

          Parameters for the user's location. Used to provide more relevant search results.

      - `BetaWebFetchTool20260318`

        - `name: "web_fetch"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"web_fetch"`

        - `type: "web_fetch_20260318"`

          - `"web_fetch_20260318"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `allowed_domains?: Array<string> | null`

          List of domains to allow fetching from

        - `blocked_domains?: Array<string> | null`

          List of domains to block fetching from

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `citations?: BetaCitationsConfigParam | null`

          Citations configuration for fetched documents. Citations are disabled by default.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `max_content_tokens?: number | null`

          Maximum number of tokens used by including web page text content in the context. The limit is approximate and does not apply to binary content such as PDFs.

        - `max_uses?: number | null`

          Maximum number of times the tool can be used in the API request.

        - `response_inclusion?: "full" | "excluded"`

          How this tool's result blocks appear in the API response when the result was consumed by a completed code_execution call in the same turn. 'full' returns the complete content (default). 'excluded' drops the nested server_tool_use and result block pair entirely. Results from direct calls, or from code_execution calls that paused before completing, are always returned in full so they can be sent back on the next turn.

          - `"full"`

          - `"excluded"`

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

        - `use_cache?: boolean`

          Whether to use cached content. Set to false to bypass the cache and fetch fresh content. Only set to false when the user explicitly requests fresh content or when fetching rapidly-changing sources.

      - `BetaAdvisorTool20260301`

        - `model: Model`

          The model that will complete your prompt.

          See [models](https://docs.anthropic.com/en/docs/models-overview) for additional details and options.

        - `name: "advisor"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"advisor"`

        - `type: "advisor_20260301"`

          - `"advisor_20260301"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `caching?: BetaCacheControlEphemeral | null`

          Caching for the advisor's own prompt. When set, each advisor call writes a cache entry at the given TTL so subsequent calls in the same conversation read the stable prefix. When omitted, the advisor prompt is not cached.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `max_tokens?: number | null`

          Bounds the advisor's total output (thinking + text) per call. When the advisor hits this cap, the returned advisor_result or advisor_redacted_result block carries stop_reason='max_tokens', and a truncation note is appended to the advice text the worker model sees (inside the encrypted blob in redacted mode). When set, the server also emits a remaining-tokens budget block in the advisor's prompt so the advisor self-shapes toward the cap. When omitted, the advisor model's default output cap applies and no budget block is emitted.

        - `max_uses?: number | null`

          Maximum number of times the tool can be used in the API request.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaToolSearchToolBm25_20251119`

        - `name: "tool_search_tool_bm25"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"tool_search_tool_bm25"`

        - `type: "tool_search_tool_bm25_20251119" | "tool_search_tool_bm25"`

          - `"tool_search_tool_bm25_20251119"`

          - `"tool_search_tool_bm25"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaToolSearchToolRegex20251119`

        - `name: "tool_search_tool_regex"`

          Name of the tool.

          This is how the tool will be called by the model and in `tool_use` blocks.

          - `"tool_search_tool_regex"`

        - `type: "tool_search_tool_regex_20251119" | "tool_search_tool_regex"`

          - `"tool_search_tool_regex_20251119"`

          - `"tool_search_tool_regex"`

        - `allowed_callers?: Array<"direct" | "code_execution_20250825" | "code_execution_20260120" | "code_execution_20260521">`

          - `"direct"`

          - `"code_execution_20250825"`

          - `"code_execution_20260120"`

          - `"code_execution_20260521"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `defer_loading?: boolean`

          If true, tool will not be included in initial system prompt. Only loaded when returned via tool_reference from tool search.

        - `strict?: boolean`

          When true, guarantees schema validation on tool names and inputs

      - `BetaMCPToolset`

        Configuration for a group of tools from an MCP server.

        Allows configuring enabled status and defer_loading for all tools
        from an MCP server, with optional per-tool overrides.

        - `mcp_server_name: string`

          Name of the MCP server to configure tools for

        - `type: "mcp_toolset"`

          - `"mcp_toolset"`

        - `cache_control?: BetaCacheControlEphemeral | null`

          Create a cache control breakpoint at this content block.

        - `configs?: Record<string, BetaMCPToolConfig> | null`

          Configuration overrides for specific tools, keyed by tool name

          - `defer_loading?: boolean`

          - `enabled?: boolean`

        - `default_config?: BetaMCPToolDefaultConfig`

          Default configuration applied to all tools from this server

          - `defer_loading?: boolean`

          - `enabled?: boolean`

    - `top_k?: number`

      Body param: Only sample from the top K options for each subsequent token.

      Used to remove "long tail" low probability responses. [Learn more technical details here](https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277).

      Recommended for advanced use cases only.

    - `top_p?: number`

      Body param: Use nucleus sampling.

      In nucleus sampling, we compute the cumulative distribution over all the options for each subsequent token in decreasing probability order and cut it off once it reaches a particular probability specified by `top_p`.

      Recommended for advanced use cases only.

    - `betas?: Array<AnthropicBeta>`

      Header param: Optional header to specify the beta version(s) you want to use.

      - `(string & {})`

      - `"message-batches-2024-09-24" | "prompt-caching-2024-07-31" | "computer-use-2024-10-22" | 30 more`

        - `"message-batches-2024-09-24"`

        - `"prompt-caching-2024-07-31"`

        - `"computer-use-2024-10-22"`

        - `"computer-use-2025-01-24"`

        - `"pdfs-2024-09-25"`

        - `"token-counting-2024-11-01"`

        - `"token-efficient-tools-2025-02-19"`

        - `"output-128k-2025-02-19"`

        - `"files-api-2025-04-14"`

        - `"mcp-client-2025-04-04"`

        - `"mcp-client-2025-11-20"`

        - `"dev-full-thinking-2025-05-14"`

        - `"interleaved-thinking-2025-05-14"`

        - `"code-execution-2025-05-22"`

        - `"extended-cache-ttl-2025-04-11"`

        - `"context-1m-2025-08-07"`

        - `"context-management-2025-06-27"`

        - `"model-context-window-exceeded-2025-08-26"`

        - `"skills-2025-10-02"`

        - `"fast-mode-2026-02-01"`

        - `"output-300k-2026-03-24"`

        - `"user-profiles-2026-03-24"`

        - `"advisor-tool-2026-03-01"`

        - `"managed-agents-2026-04-01"`

        - `"cache-diagnosis-2026-04-07"`

        - `"dreaming-2026-04-21"`

        - `"thinking-token-count-2026-05-13"`

        - `"server-side-fallback-2026-06-01"`

        - `"server-side-fallback-2026-07-01"`

        - `"fallback-credit-2026-06-01"`

        - `"fallback-credit-2026-07-01"`

        - `"agent-memory-2026-07-22"`

        - `"mid-conversation-tool-changes-2026-07-01"`

    - `user_profile_id?: string`

      Header param: The user profile ID to attribute this request to. Use when acting on behalf of a party other than your organization. Requires the `user-profiles` beta header.

  - `MessageCreateParamsNonStreaming extends MessageCreateParamsBase`

    - `stream?: false`

      Body param: Whether to incrementally stream the response using server-sent events.

      See [streaming](https://platform.claude.com/docs/en/build-with-claude/streaming) for details.

  - `MessageCreateParamsStreaming extends MessageCreateParamsBase`

    - `stream: true`

      Body param: Whether to incrementally stream the response using server-sent events.

      See [streaming](https://platform.claude.com/docs/en/build-with-claude/streaming) for details.

      - `true`

### Returns

- `BetaMessage`

  - `id: string`

    Unique object identifier.

    The format and length of IDs may change over time.

  - `container: BetaContainer | null`

    Information about the container used in the request (for the code execution tool)

    - `id: string`

      Identifier for the container used in this request

    - `expires_at: string`

      The time at which the container will expire.

    - `skills: Array<BetaSkill> | null`

      Skills loaded in the container

      - `skill_id: string`

        Skill ID

      - `type: "anthropic" | "custom"`

        Type of skill - either 'anthropic' (built-in) or 'custom' (user-defined)

        - `"anthropic"`

        - `"custom"`

      - `version: string`

        Skill version or 'latest' for most recent version

  - `content: Array<BetaContentBlock>`

    Content generated by the model.

    This is an array of content blocks, each of which has a `type` that determines its shape.

    Example:

    ```json
    [{"type": "text", "text": "Hi, I'm Claude."}]
    ```

    If the request input `messages` ended with an `assistant` turn, then the response `content` will continue directly from that last turn. You can use this to constrain the model's output.

    For example, if the input `messages` were:

    ```json
    [
      {"role": "user", "content": "What's the Greek name for Sun? (A) Sol (B) Helios (C) Sun"},
      {"role": "assistant", "content": "The best answer is ("}
    ]
    ```

    Then the response `content` might be:

    ```json
    [{"type": "text", "text": "B)"}]
    ```

    - `BetaTextBlock`

      - `citations: Array<BetaTextCitation> | null`

        Citations supporting the text block.

        The type of citation returned will depend on the type of document being cited. Citing a PDF results in `page_location`, plain text results in `char_location`, and content document results in `content_block_location`.

        - `BetaCitationCharLocation`

          - `cited_text: string`

          - `document_index: number`

          - `document_title: string | null`

          - `end_char_index: number`

          - `file_id: string | null`

          - `start_char_index: number`

          - `type: "char_location"`

            - `"char_location"`

        - `BetaCitationPageLocation`

          - `cited_text: string`

          - `document_index: number`

          - `document_title: string | null`

          - `end_page_number: number`

          - `file_id: string | null`

          - `start_page_number: number`

          - `type: "page_location"`

            - `"page_location"`

        - `BetaCitationContentBlockLocation`

          - `cited_text: string`

            The full text of the cited block range, concatenated.

            Always equals the contents of `content[start_block_index:end_block_index]` joined together. The text block is the minimal citable unit; this field is never a substring of a single block. Not counted toward output tokens, and not counted toward input tokens when sent back in subsequent turns.

          - `document_index: number`

          - `document_title: string | null`

          - `end_block_index: number`

            Exclusive 0-based end index of the cited block range in the source's `content` array.

            Always greater than `start_block_index`; a single-block citation has `end_block_index = start_block_index + 1`.

          - `file_id: string | null`

          - `start_block_index: number`

            0-based index of the first cited block in the source's `content` array.

          - `type: "content_block_location"`

            - `"content_block_location"`

        - `BetaCitationsWebSearchResultLocation`

          - `cited_text: string`

          - `encrypted_index: string`

          - `title: string | null`

          - `type: "web_search_result_location"`

            - `"web_search_result_location"`

          - `url: string`

        - `BetaCitationSearchResultLocation`

          - `cited_text: string`

            The full text of the cited block range, concatenated.

            Always equals the contents of `content[start_block_index:end_block_index]` joined together. The text block is the minimal citable unit; this field is never a substring of a single block. Not counted toward output tokens, and not counted toward input tokens when sent back in subsequent turns.

          - `end_block_index: number`

            Exclusive 0-based end index of the cited block range in the source's `content` array.

            Always greater than `start_block_index`; a single-block citation has `end_block_index = start_block_index + 1`.

          - `search_result_index: number`

            0-based index of the cited search result among all `search_result` content blocks in the request, in the order they appear across messages and tool results.

            Counted separately from `document_index`; server-side web search results are not included in this count.

          - `source: string`

          - `start_block_index: number`

            0-based index of the first cited block in the source's `content` array.

          - `title: string | null`

          - `type: "search_result_location"`

            - `"search_result_location"`

      - `text: string`

      - `type: "text"`

        - `"text"`

    - `BetaThinkingBlock`

      - `signature: string`

        A value used to verify that this thinking block was generated by Claude when it is passed back to the API.

        This is an opaque field and should not be interpreted or parsed. When passing thinking blocks back to the API (required when using tools with extended thinking), pass them back exactly as received, with this field intact.

        See [extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) for details.

      - `thinking: string`

        The text of Claude's thinking process for this block.

      - `type: "thinking"`

        - `"thinking"`

    - `BetaRedactedThinkingBlock`

      - `data: string`

        The contents of this redacted thinking block, returned when portions of the model's thinking were safety-redacted. This field is opaque and encrypted, with no readable content.

        Pass `redacted_thinking` blocks back to the API unchanged when continuing a multi-turn conversation.

        See [extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking#redacted-thinking-blocks) for details.

      - `type: "redacted_thinking"`

        - `"redacted_thinking"`

    - `BetaToolUseBlock`

      - `id: string`

      - `input: Record<string, unknown>`

      - `name: string`

      - `type: "tool_use"`

        - `"tool_use"`

      - `caller?: BetaDirectCaller | BetaServerToolCaller | BetaServerToolCaller20260120`

        Tool invocation directly from the model.

        - `BetaDirectCaller`

          Tool invocation directly from the model.

          - `type: "direct"`

            - `"direct"`

        - `BetaServerToolCaller`

          Tool invocation generated by a server-side tool.

          - `tool_id: string`

          - `type: "code_execution_20250825"`

            - `"code_execution_20250825"`

        - `BetaServerToolCaller20260120`

          - `tool_id: string`

          - `type: "code_execution_20260120"`

            - `"code_execution_20260120"`

    - `BetaServerToolUseBlock`

      - `id: string`

      - `input: Record<string, unknown>`

      - `name: "advisor" | "web_search" | "web_fetch" | 5 more`

        - `"advisor"`

        - `"web_search"`

        - `"web_fetch"`

        - `"code_execution"`

        - `"bash_code_execution"`

        - `"text_editor_code_execution"`

        - `"tool_search_tool_regex"`

        - `"tool_search_tool_bm25"`

      - `type: "server_tool_use"`

        - `"server_tool_use"`

      - `caller?: BetaDirectCaller | BetaServerToolCaller | BetaServerToolCaller20260120`

        Tool invocation directly from the model.

        - `BetaDirectCaller`

          Tool invocation directly from the model.

        - `BetaServerToolCaller`

          Tool invocation generated by a server-side tool.

        - `BetaServerToolCaller20260120`

    - `BetaWebSearchToolResultBlock`

      - `content: BetaWebSearchToolResultBlockContent`

        - `BetaWebSearchToolResultError`

          - `error_code: BetaWebSearchToolResultErrorCode`

            - `"invalid_tool_input"`

            - `"unavailable"`

            - `"max_uses_exceeded"`

            - `"too_many_requests"`

            - `"query_too_long"`

            - `"request_too_large"`

          - `type: "web_search_tool_result_error"`

            - `"web_search_tool_result_error"`

        - `Array<BetaWebSearchResultBlock>`

          - `encrypted_content: string`

          - `page_age: string | null`

          - `title: string`

          - `type: "web_search_result"`

            - `"web_search_result"`

          - `url: string`

      - `tool_use_id: string`

      - `type: "web_search_tool_result"`

        - `"web_search_tool_result"`

      - `caller?: BetaDirectCaller | BetaServerToolCaller | BetaServerToolCaller20260120`

        Tool invocation directly from the model.

        - `BetaDirectCaller`

          Tool invocation directly from the model.

        - `BetaServerToolCaller`

          Tool invocation generated by a server-side tool.

        - `BetaServerToolCaller20260120`

    - `BetaWebFetchToolResultBlock`

      - `content: BetaWebFetchToolResultErrorBlock | BetaWebFetchBlock`

        - `BetaWebFetchToolResultErrorBlock`

          - `error_code: BetaWebFetchToolResultErrorCode`

            - `"invalid_tool_input"`

            - `"url_too_long"`

            - `"url_not_allowed"`

            - `"url_not_in_prior_context"`

            - `"url_not_accessible"`

            - `"unsupported_content_type"`

            - `"too_many_requests"`

            - `"max_uses_exceeded"`

            - `"unavailable"`

          - `type: "web_fetch_tool_result_error"`

            - `"web_fetch_tool_result_error"`

        - `BetaWebFetchBlock`

          - `content: BetaDocumentBlock`

            - `citations: BetaCitationConfig | null`

              Citation configuration for the document

              - `enabled: boolean`

            - `source: BetaBase64PDFSource | BetaPlainTextSource`

              - `BetaBase64PDFSource`

                - `data: string`

                - `media_type: "application/pdf"`

                  - `"application/pdf"`

                - `type: "base64"`

                  - `"base64"`

              - `BetaPlainTextSource`

                - `data: string`

                - `media_type: "text/plain"`

                  - `"text/plain"`

                - `type: "text"`

                  - `"text"`

            - `title: string | null`

              The title of the document

            - `type: "document"`

              - `"document"`

          - `retrieved_at: string | null`

            ISO 8601 timestamp when the content was retrieved

          - `type: "web_fetch_result"`

            - `"web_fetch_result"`

          - `url: string`

            Fetched content URL

      - `tool_use_id: string`

      - `type: "web_fetch_tool_result"`

        - `"web_fetch_tool_result"`

      - `caller?: BetaDirectCaller | BetaServerToolCaller | BetaServerToolCaller20260120`

        Tool invocation directly from the model.

        - `BetaDirectCaller`

          Tool invocation directly from the model.

        - `BetaServerToolCaller`

          Tool invocation generated by a server-side tool.

        - `BetaServerToolCaller20260120`

    - `BetaAdvisorToolResultBlock`

      - `content: BetaAdvisorToolResultError | BetaAdvisorResultBlock | BetaAdvisorRedactedResultBlock`

        - `BetaAdvisorToolResultError`

          - `error_code: "max_uses_exceeded" | "prompt_too_long" | "too_many_requests" | 4 more`

            - `"max_uses_exceeded"`

            - `"prompt_too_long"`

            - `"too_many_requests"`

            - `"overloaded"`

            - `"unavailable"`

            - `"execution_time_exceeded"`

            - `"model_not_found"`

          - `type: "advisor_tool_result_error"`

            - `"advisor_tool_result_error"`

        - `BetaAdvisorResultBlock`

          - `stop_reason: string | null`

            The advisor sub-inference's stop reason (same values as the top-level message `stop_reason`). `max_tokens` indicates the advisor's output was truncated at the tool's `max_tokens` value or the advisor model's policy cap.

          - `text: string`

          - `type: "advisor_result"`

            - `"advisor_result"`

        - `BetaAdvisorRedactedResultBlock`

          - `encrypted_content: string`

            Opaque blob containing the advisor's output. Round-trip verbatim; do not inspect or modify.

          - `stop_reason: string | null`

            The advisor sub-inference's stop reason (same values as the top-level message `stop_reason`).

          - `type: "advisor_redacted_result"`

            - `"advisor_redacted_result"`

      - `tool_use_id: string`

      - `type: "advisor_tool_result"`

        - `"advisor_tool_result"`

    - `BetaCodeExecutionToolResultBlock`

      - `content: BetaCodeExecutionToolResultBlockContent`

        Code execution result with encrypted stdout for PFC + web_search results.

        - `BetaCodeExecutionToolResultError`

          - `error_code: BetaCodeExecutionToolResultErrorCode`

            - `"invalid_tool_input"`

            - `"unavailable"`

            - `"too_many_requests"`

            - `"execution_time_exceeded"`

          - `type: "code_execution_tool_result_error"`

            - `"code_execution_tool_result_error"`

        - `BetaCodeExecutionResultBlock`

          - `content: Array<BetaCodeExecutionOutputBlock>`

            - `file_id: string`

            - `type: "code_execution_output"`

              - `"code_execution_output"`

          - `return_code: number`

          - `stderr: string`

          - `stdout: string`

          - `type: "code_execution_result"`

            - `"code_execution_result"`

        - `BetaEncryptedCodeExecutionResultBlock`

          Code execution result with encrypted stdout for PFC + web_search results.

          - `content: Array<BetaCodeExecutionOutputBlock>`

            - `file_id: string`

            - `type: "code_execution_output"`

          - `encrypted_stdout: string`

          - `return_code: number`

          - `stderr: string`

          - `type: "encrypted_code_execution_result"`

            - `"encrypted_code_execution_result"`

      - `tool_use_id: string`

      - `type: "code_execution_tool_result"`

        - `"code_execution_tool_result"`

    - `BetaBashCodeExecutionToolResultBlock`

      - `content: BetaBashCodeExecutionToolResultError | BetaBashCodeExecutionResultBlock`

        - `BetaBashCodeExecutionToolResultError`

          - `error_code: "invalid_tool_input" | "unavailable" | "too_many_requests" | 2 more`

            - `"invalid_tool_input"`

            - `"unavailable"`

            - `"too_many_requests"`

            - `"execution_time_exceeded"`

            - `"output_file_too_large"`

          - `type: "bash_code_execution_tool_result_error"`

            - `"bash_code_execution_tool_result_error"`

        - `BetaBashCodeExecutionResultBlock`

          - `content: Array<BetaBashCodeExecutionOutputBlock>`

            - `file_id: string`

            - `type: "bash_code_execution_output"`

              - `"bash_code_execution_output"`

          - `return_code: number`

          - `stderr: string`

          - `stdout: string`

          - `type: "bash_code_execution_result"`

            - `"bash_code_execution_result"`

      - `tool_use_id: string`

      - `type: "bash_code_execution_tool_result"`

        - `"bash_code_execution_tool_result"`

    - `BetaTextEditorCodeExecutionToolResultBlock`

      - `content: BetaTextEditorCodeExecutionToolResultError | BetaTextEditorCodeExecutionViewResultBlock | BetaTextEditorCodeExecutionCreateResultBlock | BetaTextEditorCodeExecutionStrReplaceResultBlock`

        - `BetaTextEditorCodeExecutionToolResultError`

          - `error_code: "invalid_tool_input" | "unavailable" | "too_many_requests" | 2 more`

            - `"invalid_tool_input"`

            - `"unavailable"`

            - `"too_many_requests"`

            - `"execution_time_exceeded"`

            - `"file_not_found"`

          - `error_message: string | null`

          - `type: "text_editor_code_execution_tool_result_error"`

            - `"text_editor_code_execution_tool_result_error"`

        - `BetaTextEditorCodeExecutionViewResultBlock`

          - `content: string`

          - `file_type: "text" | "image" | "pdf"`

            - `"text"`

            - `"image"`

            - `"pdf"`

          - `num_lines: number | null`

          - `start_line: number | null`

          - `total_lines: number | null`

          - `type: "text_editor_code_execution_view_result"`

            - `"text_editor_code_execution_view_result"`

        - `BetaTextEditorCodeExecutionCreateResultBlock`

          - `is_file_update: boolean`

          - `type: "text_editor_code_execution_create_result"`

            - `"text_editor_code_execution_create_result"`

        - `BetaTextEditorCodeExecutionStrReplaceResultBlock`

          - `lines: Array<string> | null`

          - `new_lines: number | null`

          - `new_start: number | null`

          - `old_lines: number | null`

          - `old_start: number | null`

          - `type: "text_editor_code_execution_str_replace_result"`

            - `"text_editor_code_execution_str_replace_result"`

      - `tool_use_id: string`

      - `type: "text_editor_code_execution_tool_result"`

        - `"text_editor_code_execution_tool_result"`

    - `BetaToolSearchToolResultBlock`

      - `content: BetaToolSearchToolResultError | BetaToolSearchToolSearchResultBlock`

        - `BetaToolSearchToolResultError`

          - `error_code: "invalid_tool_input" | "unavailable" | "too_many_requests" | "execution_time_exceeded"`

            - `"invalid_tool_input"`

            - `"unavailable"`

            - `"too_many_requests"`

            - `"execution_time_exceeded"`

          - `error_message: string | null`

          - `type: "tool_search_tool_result_error"`

            - `"tool_search_tool_result_error"`

        - `BetaToolSearchToolSearchResultBlock`

          - `tool_references: Array<BetaToolReferenceBlock>`

            - `tool_name: string`

            - `type: "tool_reference"`

              - `"tool_reference"`

          - `type: "tool_search_tool_search_result"`

            - `"tool_search_tool_search_result"`

      - `tool_use_id: string`

      - `type: "tool_search_tool_result"`

        - `"tool_search_tool_result"`

    - `BetaMCPToolUseBlock`

      - `id: string`

      - `input: Record<string, unknown>`

      - `name: string`

        The name of the MCP tool

      - `server_name: string`

        The name of the MCP server

      - `type: "mcp_tool_use"`

        - `"mcp_tool_use"`

    - `BetaMCPToolResultBlock`

      - `content: string | Array<BetaTextBlock>`

        - `string`

        - `Array<BetaTextBlock>`

          - `citations: Array<BetaTextCitation> | null`

            Citations supporting the text block.

            The type of citation returned will depend on the type of document being cited. Citing a PDF results in `page_location`, plain text results in `char_location`, and content document results in `content_block_location`.

          - `text: string`

          - `type: "text"`

      - `is_error: boolean`

      - `tool_use_id: string`

      - `type: "mcp_tool_result"`

        - `"mcp_tool_result"`

    - `BetaContainerUploadBlock`

      Response model for a file uploaded to the container.

      - `file_id: string`

      - `type: "container_upload"`

        - `"container_upload"`

    - `BetaCompactionBlock`

      A compaction block returned when autocompact is triggered.

      When content is None, it indicates the compaction failed to produce a valid
      summary (e.g., malformed output from the model). Clients may round-trip
      compaction blocks with null content; the server treats them as no-ops.

      - `content: string | null`

        Summary of compacted content, or null if compaction failed

      - `encrypted_content: string | null`

        Opaque metadata from prior compaction, to be round-tripped verbatim

      - `type: "compaction"`

        - `"compaction"`

    - `BetaFallbackBlock`

      Marks the point in `content` where one model's output gives way to the next.

      One block appears per hop where a preceding model actually ran this turn and
      declined. A turn where no preceding model ran and declined has no such
      boundary and carries no block — the signal for whether a fallback model
      served the response is the presence of a `fallback_message` entry in
      `usage.iterations`, not this block.

      The block is treated like a server-tool content block for streaming: it
      arrives via the standard `content_block_start` / `content_block_stop`
      pair and carries no deltas.

      - `from: BetaFallbackInfo`

        The model whose output ends at this point — the model that declined at this hop. When the declining hop is the requested model, its `model` echoes the top-level `model` string the caller sent (alias or canonical); when the declining hop is a fallback model, its `model` is that model's canonical id.

        - `model: Model`

          The model that will complete your prompt.

          See [models](https://docs.anthropic.com/en/docs/models-overview) for additional details and options.

          - `"claude-sonnet-5" | "claude-fable-5" | "claude-mythos-5" | 12 more`

            - `"claude-sonnet-5"`

              High-performance model for coding and agents

            - `"claude-fable-5"`

              Next generation of intelligence for the hardest knowledge work and coding problems

            - `"claude-mythos-5"`

              Most capable model for cybersecurity and biology research

            - `"claude-opus-5"`

              Powerful intelligence for long-running agents and coding

            - `"claude-opus-4-8"`

              Powerful intelligence for long-running agents and coding

            - `"claude-opus-4-7"`

              Powerful intelligence for long-running agents and coding

            - `"claude-mythos-preview"`

              New class of intelligence, strongest in coding and cybersecurity

            - `"claude-opus-4-6"`

              Powerful intelligence for long-running agents and coding

            - `"claude-sonnet-4-6"`

              Best combination of speed and intelligence

            - `"claude-haiku-4-5"`

              Fastest model with near-frontier intelligence

            - `"claude-haiku-4-5-20251001"`

              Fastest model with near-frontier intelligence

            - `"claude-opus-4-5"`

              Powerful intelligence for long-running agents and coding

            - `"claude-opus-4-5-20251101"`

              Powerful intelligence for long-running agents and coding

            - `"claude-sonnet-4-5"`

              High-performance model for agents and coding

            - `"claude-sonnet-4-5-20250929"`

              High-performance model for agents and coding

          - `(string & {})`

      - `to: BetaFallbackInfo`

        The fallback model producing the content that follows this block. Its `model` is always the canonical id.

      - `trigger: BetaFallbackRefusalTrigger`

        What caused the `from` model to hand over at this hop.

        - `category: "cyber" | "bio" | "frontier_llm" | 2 more | null`

          The policy category that triggered a refusal.

          - `"cyber"`

            The request could enable cyber harm, such as malware or exploit development. Benign cybersecurity work can also trigger this category.

          - `"bio"`

            The request could enable biological harm, such as dangerous lab methods. Beneficial life sciences work can also trigger this category.

          - `"frontier_llm"`

            The request could assist the development of competing AI models, which is restricted under [Anthropic's commercial terms](https://www.anthropic.com/legal/commercial-terms). Benign machine learning work can also trigger this category.

          - `"reasoning_extraction"`

            The request asks the model to reproduce its internal reasoning in the response text. To get reasoning in a structured form instead, use [adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking).

          - `"general_harms"`

            The request could be related to an area that was determined as harmful. Benign work might sometimes trigger this category.

        - `type: "refusal"`

          - `"refusal"`

      - `type: "fallback"`

        - `"fallback"`

  - `context_management: BetaContextManagementResponse | null`

    Context management response.

    Information about context management strategies applied during the request.

    - `applied_edits: Array<BetaClearToolUses20250919EditResponse | BetaClearThinking20251015EditResponse>`

      List of context management edits that were applied.

      - `BetaClearToolUses20250919EditResponse`

        - `cleared_input_tokens: number`

          Number of input tokens cleared by this edit.

        - `cleared_tool_uses: number`

          Number of tool uses that were cleared.

        - `type: "clear_tool_uses_20250919"`

          The type of context management edit applied.

          - `"clear_tool_uses_20250919"`

      - `BetaClearThinking20251015EditResponse`

        - `cleared_input_tokens: number`

          Number of input tokens cleared by this edit.

        - `cleared_thinking_turns: number`

          Number of thinking turns that were cleared.

        - `type: "clear_thinking_20251015"`

          The type of context management edit applied.

          - `"clear_thinking_20251015"`

  - `diagnostics: BetaDiagnostics | null`

    Response envelope for request-level diagnostics. Present (possibly
    null) whenever the caller supplied `diagnostics` on the request.

    - `cache_miss_reason: BetaCacheMissModelChanged | BetaCacheMissSystemChanged | BetaCacheMissToolsChanged | 3 more | null`

      Explains why the prompt cache could not fully reuse the prefix from the request identified by `diagnostics.previous_message_id`. `null` means diagnosis is still pending — the response was serialized before the background comparison completed.

      - `BetaCacheMissModelChanged`

        - `cache_missed_input_tokens: number`

          Approximate number of input tokens that would have been read from cache had the prefix matched the previous request.

        - `type: "model_changed"`

          - `"model_changed"`

      - `BetaCacheMissSystemChanged`

        - `cache_missed_input_tokens: number`

          Approximate number of input tokens that would have been read from cache had the prefix matched the previous request.

        - `type: "system_changed"`

          - `"system_changed"`

      - `BetaCacheMissToolsChanged`

        - `cache_missed_input_tokens: number`

          Approximate number of input tokens that would have been read from cache had the prefix matched the previous request.

        - `type: "tools_changed"`

          - `"tools_changed"`

      - `BetaCacheMissMessagesChanged`

        - `cache_missed_input_tokens: number`

          Approximate number of input tokens that would have been read from cache had the prefix matched the previous request.

        - `type: "messages_changed"`

          - `"messages_changed"`

      - `BetaCacheMissPreviousMessageNotFound`

        - `type: "previous_message_not_found"`

          - `"previous_message_not_found"`

      - `BetaCacheMissUnavailable`

        - `type: "unavailable"`

          - `"unavailable"`

  - `model: Model`

    The model that will complete your prompt.

    See [models](https://docs.anthropic.com/en/docs/models-overview) for additional details and options.

  - `role: "assistant"`

    Conversational role of the generated message.

    This will always be `"assistant"`.

    - `"assistant"`

  - `stop_details: BetaRefusalStopDetails | null`

    Structured information about a refusal.

    - `category: "cyber" | "bio" | "frontier_llm" | 2 more | null`

      The policy category that triggered a refusal.

      - `"cyber"`

        The request could enable cyber harm, such as malware or exploit development. Benign cybersecurity work can also trigger this category.

      - `"bio"`

        The request could enable biological harm, such as dangerous lab methods. Beneficial life sciences work can also trigger this category.

      - `"frontier_llm"`

        The request could assist the development of competing AI models, which is restricted under [Anthropic's commercial terms](https://www.anthropic.com/legal/commercial-terms). Benign machine learning work can also trigger this category.

      - `"reasoning_extraction"`

        The request asks the model to reproduce its internal reasoning in the response text. To get reasoning in a structured form instead, use [adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking).

      - `"general_harms"`

        The request could be related to an area that was determined as harmful. Benign work might sometimes trigger this category.

    - `explanation: string | null`

      Human-readable explanation of the refusal.

      This text is not guaranteed to be stable. `null` when no explanation is available for the category.

    - `fallback_credit_token: string | null`

      Opaque code that refunds the cache-miss cost when retrying this refused
      request on the fallback model. Pass it as `fallback_credit_token` on the
      retry request. Expires 5 minutes after the refusal.

      The retry is sent either with the same request body (`system`, `messages`,
      `tools`, and other render-shaping fields), or with the same body plus one
      appended `assistant` message whose content is the partial text (with any
      trailing whitespace stripped from the final text block) and paired
      server-tool blocks from this refusal — which also authorizes that
      appended turn as an assistant-prefill continuation on models that otherwise
      disallow prefill. A token minted mid-server-tool-loop whose partial content
      was continuable may only be redeemed the second way — if a same-body retry
      is rejected with a 400 saying the token must be redeemed by continuing the
      partial response, retry the second way instead. Either way: same workspace,
      same platform; a mismatch is a 400. Resending a token for an already-warm
      prefix is permitted but yields no additional credit.

      `null` when the refused model isn't eligible for a fallback credit.

    - `fallback_has_prefill_claim: boolean | null`

      Whether the accompanying `fallback_credit_token` may be redeemed with the
      appended-assistant retry form. Only set when `fallback_credit_token` is
      present.

      `true`: retry by resending the same request body plus one appended
      `assistant` message whose content is this response's `content` with any
      trailing whitespace stripped from the final text block and unpaired
      `tool_use` blocks omitted (the same appended-turn shape described on
      `fallback_credit_token`), with the token attached. `false`: retry by
      resending the original request body unchanged, with the token attached —
      the appended-assistant form is not available for this refusal (no
      continuable partial content, or the request uses `output_format` or a
      `tool_choice` that forces tool use). One exception: when the request used
      `output_format` or a forced `tool_choice` and the refusal arrived after
      server tools (including MCP connector tools) had already executed, the
      token may not be redeemable by either retry form; if the exact-body retry
      is then rejected with a 400 saying the token must be redeemed by
      continuing the partial response, discard the token and retry without it.

      Advisory: if an appended-assistant retry is rejected with a 400 despite
      `true`, fall back to resending the original request body with the token.

    - `recommended_model: string | null`

      The server's suggested retry target for this refusal. Populated when a fallback attempt could not be made (the fallback model's rate limit was exhausted, or it was overloaded); names the fallback model the caller can retry directly. Null otherwise.

    - `type: "refusal"`

      - `"refusal"`

  - `stop_reason: BetaStopReason | null`

    The reason that we stopped.

    This may be one the following values:

    * `"end_turn"`: the model reached a natural stopping point
    * `"max_tokens"`: we exceeded the requested `max_tokens` or the model's maximum
    * `"stop_sequence"`: one of your provided custom `stop_sequences` was generated
    * `"tool_use"`: the model invoked one or more tools
    * `"pause_turn"`: we paused a long-running turn. You may provide the response back as-is in a subsequent request to let the model continue.
    * `"refusal"`: when streaming classifiers intervene to handle potential policy violations
    * `"model_context_window_exceeded"`: we exceeded the model's context window

    In non-streaming mode this value is always non-null. In streaming mode, it is null in the `message_start` event and non-null otherwise.

    - `"end_turn"`

    - `"max_tokens"`

    - `"stop_sequence"`

    - `"tool_use"`

    - `"pause_turn"`

    - `"compaction"`

    - `"refusal"`

    - `"model_context_window_exceeded"`

  - `stop_sequence: string | null`

    Which custom stop sequence was generated, if any.

    This value will be a non-null string if one of your custom stop sequences was generated.

  - `type: "message"`

    Object type.

    For Messages, this is always `"message"`.

    - `"message"`

  - `usage: BetaUsage`

    Billing and rate-limit usage.

    Anthropic's API bills and rate-limits by token counts, as tokens represent the underlying cost to our systems.

    Under the hood, the API transforms requests into a format suitable for the model. The model's output then goes through a parsing stage before becoming an API response. As a result, the token counts in `usage` will not match one-to-one with the exact visible content of an API request or response.

    For example, `output_tokens` will be non-zero, even for an empty string response from Claude.

    Total input tokens in a request is the summation of `input_tokens`, `cache_creation_input_tokens`, and `cache_read_input_tokens`.

    - `cache_creation: BetaCacheCreation | null`

      Breakdown of cached tokens by TTL

      - `ephemeral_1h_input_tokens: number`

        The number of input tokens used to create the 1 hour cache entry.

      - `ephemeral_5m_input_tokens: number`

        The number of input tokens used to create the 5 minute cache entry.

    - `cache_creation_input_tokens: number | null`

      The number of input tokens used to create the cache entry.

    - `cache_read_input_tokens: number | null`

      The number of input tokens read from the cache.

    - `fallback_credit: BetaFallbackCreditUsage | null`

      Outcome of the `fallback_credit_token` presented on this request.

      - `status: BetaFallbackCreditRedeemed | BetaFallbackCreditNotApplied`

        Whether the fallback-credit reprice was applied to this response's billing.

        A union discriminated on `type`. `redeemed`: the retry is billed as if
        the conversation had been on the retry model all along — including when the
        resulting shift is zero because there was nothing to move. `not_applied`:
        no reprice was applied; the arm's `reason` says why.

        - `BetaFallbackCreditRedeemed`

          The reprice was applied: the retry is billed as if the conversation
          had been on the retry model all along.

          - `type: "redeemed"`

            - `"redeemed"`

        - `BetaFallbackCreditNotApplied`

          No reprice was applied; `reason` says why.

          - `reason: "body_mismatch" | "continuation_excluded" | "continuation_only" | 9 more`

            Why the reprice was not applied.

            A closed enum; additions to the redemption-check vocabulary arrive as
            deliberate schema updates.

            - `"body_mismatch"`

            - `"continuation_excluded"`

            - `"continuation_only"`

            - `"expired"`

            - `"invalid_target_model"`

            - `"not_enabled"`

            - `"reprice_unavailable"`

            - `"temporarily_unavailable"`

            - `"variant_fields_present"`

            - `"wrong_organization"`

            - `"wrong_platform"`

            - `"wrong_workspace"`

          - `type: "not_applied"`

            - `"not_applied"`

          - `remove_to_redeem?: Array<string> | null`

            Request fields to remove before retrying, so the retry can redeem this
            token.

            Present exactly when `reason` is `variant_fields_present` — never null,
            never an empty array; absent otherwise. Fields are named only from your own request, and only after
            the sealed variant hash matched. A served best-effort retry has already
            been billed at normal price; nothing redeems retroactively, but a corrected
            re-send inside the token's five-minute window can still redeem.

    - `inference_geo: string | null`

      The geographic region where inference was performed for this request.

    - `input_tokens: number`

      The number of input tokens which were used.

    - `iterations: BetaIterationsUsage | null`

      Per-iteration token usage breakdown.

      Each entry represents one sampling iteration, with its own input/output token counts and cache statistics. This allows you to:

      - Determine which iterations exceeded long context thresholds (>=200k tokens)
      - Calculate the true context window size from the last iteration
      - Understand token accumulation across server-side tool use loops

      - `BetaMessageIterationUsage`

        Token usage for a sampling iteration.

        - `cache_creation: BetaCacheCreation | null`

          Breakdown of cached tokens by TTL

        - `cache_creation_input_tokens: number`

          The number of input tokens used to create the cache entry.

        - `cache_read_input_tokens: number`

          The number of input tokens read from the cache.

        - `input_tokens: number`

          The number of input tokens which were used.

        - `model: Model`

          The model that will complete your prompt.

          See [models](https://docs.anthropic.com/en/docs/models-overview) for additional details and options.

        - `output_tokens: number`

          The number of output tokens which were used.

        - `type: "message"`

          Usage for a sampling iteration

          - `"message"`

      - `BetaCompactionIterationUsage`

        Token usage for a compaction iteration.

        - `cache_creation: BetaCacheCreation | null`

          Breakdown of cached tokens by TTL

        - `cache_creation_input_tokens: number`

          The number of input tokens used to create the cache entry.

        - `cache_read_input_tokens: number`

          The number of input tokens read from the cache.

        - `input_tokens: number`

          The number of input tokens which were used.

        - `output_tokens: number`

          The number of output tokens which were used.

        - `type: "compaction"`

          Usage for a compaction iteration

          - `"compaction"`

      - `BetaAdvisorMessageIterationUsage`

        Token usage for an advisor sub-inference iteration.

        - `cache_creation: BetaCacheCreation | null`

          Breakdown of cached tokens by TTL

        - `cache_creation_input_tokens: number`

          The number of input tokens used to create the cache entry.

        - `cache_read_input_tokens: number`

          The number of input tokens read from the cache.

        - `input_tokens: number`

          The number of input tokens which were used.

        - `model: Model`

          The model that will complete your prompt.

          See [models](https://docs.anthropic.com/en/docs/models-overview) for additional details and options.

        - `output_tokens: number`

          The number of output tokens which were used.

        - `type: "advisor_message"`

          Usage for an advisor sub-inference iteration

          - `"advisor_message"`

      - `BetaFallbackMessageIterationUsage`

        Token usage for the fallback-model attempt of a server-side fallback request.

        Produced in place of a `message` entry for whichever hop served the
        response. A declined hop produces the existing `message` entry. Whether
        a fallback model served the response is signalled by the presence of this
        entry in `usage.iterations`.

        - `cache_creation: BetaCacheCreation | null`

          Breakdown of cached tokens by TTL

        - `cache_creation_input_tokens: number`

          The number of input tokens used to create the cache entry.

        - `cache_read_input_tokens: number`

          The number of input tokens read from the cache.

        - `input_tokens: number`

          The number of input tokens which were used.

        - `model: Model`

          The model that will complete your prompt.

          See [models](https://docs.anthropic.com/en/docs/models-overview) for additional details and options.

        - `output_tokens: number`

          The number of output tokens which were used.

        - `type: "fallback_message"`

          Usage for the fallback-model attempt that served the response

          - `"fallback_message"`

    - `output_tokens: number`

      The number of output tokens which were used.

    - `output_tokens_details: BetaOutputTokensDetails | null`

      Breakdown of output tokens by category.

      `output_tokens` remains the inclusive, authoritative total used for billing.
      This object provides a read-only decomposition for observability — for example,
      how many of the billed output tokens were spent on internal reasoning that may
      have been summarized before being returned to you.

      - `thinking_tokens: number`

        Number of output tokens the model generated as internal reasoning, including
        the thinking-block delimiter tokens.

        Reflects the raw reasoning the model produced, not the (possibly shorter)
        summarized thinking text returned in the response body. Computed by
        re-tokenizing the raw reasoning text, so it may differ from the model's exact
        generation count by a small number of tokens. Always ≤ `output_tokens`;
        `output_tokens - thinking_tokens` approximates the non-reasoning output.

    - `server_tool_use: BetaServerToolUsage | null`

      The number of server tool requests.

      - `web_fetch_requests: number`

        The number of web fetch tool requests.

      - `web_search_requests: number`

        The number of web search tool requests.

    - `service_tier: "standard" | "priority" | "batch" | null`

      If the request used the priority, standard, or batch tier.

      - `"standard"`

      - `"priority"`

      - `"batch"`

    - `speed: "standard" | "fast" | null`

      Inference speed mode. `fast` provides significantly faster output token generation at premium pricing. Not all models support `fast`; invalid combinations are rejected at create time.

      - `"standard"`

      - `"fast"`

### Example

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: process.env["ANTHROPIC_API_KEY"] // This is the default and can be omitted
});

const betaMessage = await client.beta.messages.create({
  max_tokens: 1024,
  messages: [{ content: "Hello, world", role: "user" }],
  model: "claude-opus-4-6"
});

console.log(betaMessage.id);
```

#### Response

```json
{
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "container": {
    "id": "container_011CpZohnwH4vuy7gazohgSP",
    "expires_at": "2019-12-27T18:11:19.117Z",
    "skills": [
      {
        "skill_id": "pdf",
        "type": "anthropic",
        "version": "latest"
      }
    ]
  },
  "content": [
    {
      "citations": [
        {
          "cited_text": "The grass is green. The sky is blue.",
          "document_index": 0,
          "document_title": "My Document",
          "end_char_index": 0,
          "file_id": "file_011CNha8iCJcU1wXNR6q4V8w",
          "start_char_index": 0,
          "type": "char_location"
        }
      ],
      "text": "Hi! My name is Claude.",
      "type": "text"
    }
  ],
  "context_management": {
    "applied_edits": [
      {
        "cleared_input_tokens": 0,
        "cleared_tool_uses": 0,
        "type": "clear_tool_uses_20250919"
      }
    ]
  },
  "diagnostics": {
    "cache_miss_reason": {
      "cache_missed_input_tokens": 0,
      "type": "model_changed"
    }
  },
  "model": "claude-opus-4-6",
  "role": "assistant",
  "stop_details": {
    "category": "cyber",
    "explanation": "This request was declined because it conflicts with Anthropic's Usage Policy.",
    "fallback_credit_token": "QW50aHJvcGljL0NsYXVkZQ==",
    "fallback_has_prefill_claim": true,
    "recommended_model": "claude-sonnet-4-6",
    "type": "refusal"
  },
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "cache_creation": {
      "ephemeral_1h_input_tokens": 0,
      "ephemeral_5m_input_tokens": 0
    },
    "cache_creation_input_tokens": 2051,
    "cache_read_input_tokens": 2051,
    "fallback_credit": {
      "status": {
        "type": "redeemed"
      }
    },
    "inference_geo": "global",
    "input_tokens": 2095,
    "iterations": [
      {
        "cache_creation": {
          "ephemeral_1h_input_tokens": 0,
          "ephemeral_5m_input_tokens": 0
        },
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "input_tokens": 0,
        "model": "claude-sonnet-5",
        "output_tokens": 0,
        "type": "message"
      }
    ],
    "output_tokens": 503,
    "output_tokens_details": {
      "thinking_tokens": 0
    },
    "server_tool_use": {
      "web_fetch_requests": 2,
      "web_search_requests": 0
    },
    "service_tier": "standard",
    "speed": "standard"
  }
}
```
