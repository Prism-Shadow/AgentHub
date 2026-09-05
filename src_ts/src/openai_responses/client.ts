// Copyright 2025 Prism Shadow. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import OpenAI from "openai";
import type {
  ResponseInputItem,
  ResponseStreamEvent,
  ResponseCreateParamsStreaming,
} from "openai/resources/responses/responses";
import { LLMClient } from "../baseClient";
import {
  parseToolCallArguments,
  UnsupportedParameterError,
} from "../errors";
import {
  EventType,
  FinishReason,
  PartialContentItem,
  ThinkingLevel,
  ToolChoice,
  UniConfig,
  UniEvent,
  UniMessage,
  PromptCaching,
  UsageMetadata,
} from "../types";
import { isDebugEnabled, openaiImageDetail } from "../utils";

/**
 * OpenAI Responses-compatible client implementation.
 */
export class OpenaiResponsesClient extends LLMClient {
  protected _model: string;
  private _client: OpenAI;

  /**
   * Initialize OpenAI Responses-compatible client with model, API key, and base URL.
   */
  constructor(options: {
    model: string;
    apiKey?: string;
    baseUrl?: string | null;
    clientType?: string | null;
    defaultHeaders?: Record<string, string>;
  }) {
    super();
    this._model = options.model;
    const key = options.apiKey || process.env.OPENAI_API_KEY || undefined;
    const url = options.baseUrl || process.env.OPENAI_BASE_URL || undefined;
    this._client = new OpenAI({
      apiKey: key,
      baseURL: url,
      defaultHeaders: options.defaultHeaders,
    });
  }

  /**
   * Convert ThinkingLevel enum to the Responses API reasoning effort.
   *
   * This client is a generic Responses-protocol client, so it serves third-party
   * gateways (Console Go, OpenRouter, ...) as well as OpenAI's own GPT-5.x models
   * when they are routed here through a clientType override. Third-party gateways
   * accept at most "xhigh" -- the Console Go family rejects "max" with
   *   `reasoning.effort`: unknown variant `max`, expected one of
   *   `none`, `minimal`, `low`, `medium`, `high`, `xhigh`
   * -- so MAX is clamped to "xhigh" for every model except OpenAI's own GPT-5.x
   * line, which is the one that genuinely takes "max".
   */
  private _convertThinkingLevelToEffort(thinkingLevel: ThinkingLevel): string {
    if (thinkingLevel !== ThinkingLevel.MAX) {
      return this._effortForLevel(thinkingLevel);
    }
    // OpenAI's own GPT-5.4/5.5/5.6 accept "max"; gateways do not.
    const model = this._model.toLowerCase();
    if (
      model.includes("gpt-5.4") ||
      model.includes("gpt-5.5") ||
      model.includes("gpt-5.6")
    ) {
      return "max";
    }
    return "xhigh";
  }

  private _effortForLevel(thinkingLevel: ThinkingLevel): string {
    const mapping: { [key: string]: string } = {
      [ThinkingLevel.NONE]: "none",
      [ThinkingLevel.LOW]: "low",
      [ThinkingLevel.MEDIUM]: "medium",
      [ThinkingLevel.HIGH]: "high",
      [ThinkingLevel.XHIGH]: "xhigh",
      [ThinkingLevel.MAX]: "xhigh",
    };
    return mapping[thinkingLevel];
  }

  /**
   * Convert ToolChoice to the Responses API tool_choice format with allowed tools support.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _convertToolChoice(toolChoice: ToolChoice): any {
    if (Array.isArray(toolChoice)) {
      return {
        mode: "required",
        tools: toolChoice.map((name) => ({ type: "function", name })),
      };
    }

    return toolChoice;
  }

  /**
   * Convert an image URL to an input_image item, at the detail the API needs
   * to read it.
   */
  private _convertImageUrl(imageUrl: string): {
    type: "input_image";
    image_url: string;
    detail?: "high";
  } {
    const detail = openaiImageDetail(this._model, imageUrl);
    return detail
      ? { type: "input_image", image_url: imageUrl, detail }
      : { type: "input_image", image_url: imageUrl };
  }

  /**
   * Transform universal configuration to OpenAI Responses-compatible configuration.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformUniConfigToModelConfig(config: UniConfig): any {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const openaiConfig: any = {
      model: this._model,
      store: false,
    };

    if (config.system_prompt !== undefined) {
      openaiConfig.instructions = config.system_prompt;
    }

    if (config.max_tokens !== undefined) {
      openaiConfig.max_output_tokens = config.max_tokens;
    }

    if (config.temperature !== undefined) {
      openaiConfig.temperature = config.temperature;
    }

    // Unlike the model-specific Responses clients, the summary stays inside this branch:
    // OpenRouter reads a reasoning object carrying no effort as "reasoning disabled" and
    // refuses it on a forced-thinking model -- "Reasoning is mandatory for this endpoint
    // and cannot be disabled" (400, verified live 2026-09-03 with z-ai/glm-5.3) -- so a
    // summary sent on its own would turn a dropped value into a failed request.
    if (config.thinking_level !== undefined) {
      openaiConfig.reasoning = {
        effort: this._convertThinkingLevelToEffort(config.thinking_level),
      };
      if (config.thinking_summary) {
        openaiConfig.reasoning.summary = "concise";
      }
    }

    if (config.tools !== undefined) {
      openaiConfig.tools = config.tools.map((tool) => ({
        type: "function",
        ...tool,
      }));
    }

    if (config.tool_choice !== undefined) {
      openaiConfig.tool_choice = this._convertToolChoice(config.tool_choice);
    }

    if (config.fast_mode) {
      openaiConfig.service_tier = "priority";
    }

    if (
      config.prompt_caching !== undefined &&
      config.prompt_caching !== PromptCaching.ENABLE
    ) {
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "prompt_caching",
        message: "prompt_caching must be ENABLE for the Responses API.",
      });
    }

    return openaiConfig;
  }

  /**
   * Transform universal message format to OpenAI Responses-compatible input format.
   */
  transformUniMessageToModelInput(
    messages: UniMessage[],
    _signal?: AbortSignal,
  ): ResponseInputItem[] {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const inputList: any[] = [];

    for (const msg of messages) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let contentItems: any[] = [];
      let lastPhase: string | null = null;

      for (const item of msg.content_items) {
        // anything that is not message content becomes an input item of its own, so the
        // text collected so far is flushed first to keep the original order: a server that
        // merges a function call into the adjacent assistant message rejects a call whose
        // output does not follow it (DeepSeek answers "No tool output found for tool call")
        if (
          item.type !== "text" &&
          item.type !== "image_url" &&
          contentItems.length > 0
        ) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const entry: any = { role: msg.role, content: contentItems };
          if (lastPhase !== null) {
            entry.phase = lastPhase;
          }

          inputList.push(entry);
          contentItems = [];
        }

        if (item.type === "text") {
          const phase = item.fidelity?.phase;
          if (msg.role === "assistant" && phase) {
            // split different phases
            if (
              lastPhase !== null &&
              lastPhase !== phase &&
              contentItems.length > 0
            ) {
              inputList.push({
                role: msg.role,
                content: contentItems,
                phase: lastPhase,
              });
              contentItems = [];
            }
            lastPhase = phase;
          }
          if (msg.role === "user") {
            contentItems.push({ type: "input_text", text: item.text });
          } else {
            contentItems.push({ type: "output_text", text: item.text });
          }
        } else if (item.type === "image_url") {
          contentItems.push(this._convertImageUrl(item.image_url));
        } else if (item.type === "thinking") {
          // the wire shape differs by server: OpenAI-style servers stream summaries and
          // demand the summary key back (with encrypted_content preserved), while
          // DeepSeek/Z.AI/MiniMax-style servers accept a reasoning item rebuilt from the
          // thinking text alone as reasoning_text content
          const fidelity = item.fidelity ?? {};
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const reasoning: any = { type: "reasoning", summary: [] };
          if (fidelity.channel === "summary") {
            if (item.thinking) {
              reasoning.summary = [{ type: "summary_text", text: item.thinking }];
            }
          } else if (item.thinking) {
            reasoning.content = [
              { type: "reasoning_text", text: item.thinking },
            ];
          }

          for (const key of ["encrypted_content", "signature", "format"]) {
            if (fidelity[key] != null) {
              reasoning[key] = fidelity[key];
            }
          }

          inputList.push(reasoning);
        } else if (item.type === "tool_call") {
          inputList.push({
            type: "function_call",
            call_id: item.tool_call_id,
            name: item.name,
            arguments: JSON.stringify(item.arguments),
          });
        } else if (item.type === "tool_result") {
          if (!item.tool_call_id) {
            throw new Error("tool_call_id is required for tool result.");
          }

          // NOTE: tool results are input items
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const toolResult: any[] = [{ type: "input_text", text: item.text }];

          if (item.images) {
            for (const imageUrl of item.images) {
              toolResult.push(this._convertImageUrl(imageUrl));
            }
          }

          // Build the function_call_output item. The output is delivered as a plain
          // string for maximum compatibility: some Responses-compatible gateways
          // (e.g. the opencode_go / "Console Go" proxy family) reject the array form
          // with "No function call found for function call output with call_id …"
          // because they key the call's replay by a string output payload.
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const functionCallOutput: any = {
            type: "function_call_output",
            call_id: item.tool_call_id,
            output: typeof item.text === "string" ? item.text : JSON.stringify(toolResult),
          };
          inputList.push(functionCallOutput);
        } else {
          throw new Error(`Unknown item: ${JSON.stringify(item)}`);
        }
      }

      if (contentItems.length > 0) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const entry: any = { role: msg.role, content: contentItems };
        if (lastPhase !== null) {
          entry.phase = lastPhase;
        }
        inputList.push(entry);
      }
    }

    return inputList;
  }

  /**
   * Transform OpenAI Responses-compatible streaming event to universal event format.
   */
  transformModelOutputToUniEvent(modelOutput: ResponseStreamEvent): UniEvent {
    let eventType: EventType | null = null;
    const contentItems: PartialContentItem[] = [];
    let usageMetadata: UsageMetadata | null = null;
    let finishReason: FinishReason | null = null;

    const openaiEventType = modelOutput.type;
    if (openaiEventType === "response.output_text.delta") {
      eventType = "delta";
      contentItems.push({ type: "text", text: modelOutput.delta });
    } else if (
      openaiEventType === "response.reasoning_text.delta" ||
      openaiEventType === "response.reasoning_summary_text.delta"
    ) {
      eventType = "delta";
      contentItems.push({ type: "thinking", thinking: modelOutput.delta });
    } else if (openaiEventType === "response.output_item.added") {
      const item = modelOutput.item;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const phase = (item as any).phase as string | undefined;
      if (item.type === "function_call") {
        eventType = "start";
        contentItems.push({
          type: "partial_tool_call",
          name: item.name,
          arguments: "",
          tool_call_id: item.call_id,
        });
      } else if (item.type === "message" && phase) {
        eventType = "delta";
        contentItems.push({ type: "text", text: "", fidelity: { phase } });
      } else {
        eventType = "unused";
      }
    } else if (openaiEventType === "response.output_item.done") {
      const item = modelOutput.item;
      if (item.type === "reasoning") {
        // record the wire shape of the completed reasoning item so a replay reproduces
        // the channel that carried the thinking plus the fields the server demands back
        eventType = "delta";
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const fidelity: any = {};
        if (item.summary && item.summary.length > 0) {
          fidelity.channel = "summary";
        }
        for (const key of ["encrypted_content", "signature", "format"]) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          if ((item as any)[key] != null) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            fidelity[key] = (item as any)[key];
          }
        }

        contentItems.push({ type: "thinking", thinking: "", fidelity });
      } else {
        eventType = "unused";
      }
    } else if (openaiEventType === "response.function_call_arguments.delta") {
      eventType = "delta";
      contentItems.push({
        type: "partial_tool_call",
        name: "",
        arguments: modelOutput.delta,
        tool_call_id: "",
      });
    } else if (openaiEventType === "response.function_call_arguments.done") {
      eventType = "stop";
    } else if (
      openaiEventType === "response.completed" ||
      openaiEventType === "response.incomplete"
    ) {
      eventType = "stop";
      const response = modelOutput.response;
      const finishReasonMapping: { [key: string]: FinishReason } = {
        completed: "stop",
        incomplete: "length",
      };
      if (response.status) {
        finishReason = finishReasonMapping[response.status] || "unknown";
      }
      if (response.usage) {
        // some servers drop the detail blocks (e.g. MiniMax on truncation), so default to zero
        const cachedTokens = response.usage.input_tokens_details?.cached_tokens || 0;
        const reasoningTokens =
          response.usage.output_tokens_details?.reasoning_tokens || 0;

        usageMetadata = {
          cached_tokens: cachedTokens,
          prompt_tokens: response.usage.input_tokens - cachedTokens,
          thoughts_tokens: reasoningTokens,
          response_tokens: response.usage.output_tokens - reasoningTokens,
        };
      }
    } else if (
      [
        "response.created",
        "response.in_progress",
        "response.output_text.done",
        "response.reasoning_text.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.done",
        "response.content_part.added",
        "response.content_part.done",
        // gateway heartbeat on long generations; carries no content
        "keepalive",
      ].includes(openaiEventType)
    ) {
      eventType = "unused";
        } else if (isDebugEnabled()) {
      throw new Error(`Unknown output: ${JSON.stringify(modelOutput)}`);
    } else {
      // a gateway injects its own events (heartbeats, cost tickers) into the stream, and
      // killing a long generation over one costs more than dropping it
      eventType = "unused";
    }

    return {
      role: "assistant",
      event_type: eventType,
      content_items: contentItems,
      usage_metadata: usageMetadata,
      finish_reason: finishReason,
    };
  }

  /**
   * Estimate the wire bytes of an input list. Used to gate requests against
   * gateway body-size limits (nginx `client_max_body_size`) before they ship.
   */
  _estimateInputBytes(inputList: any[]): number {
    try {
      return JSON.stringify(inputList).length;
    } catch {
      return 0;
    }
  }

  /**
   * Trim the input list so the serialized body stays under `maxBytes`.
   * Strategy: keep the system instructions (first entry) and the most recent
   * user/assistant exchanges, dropping older history in pairs. Tool calls and
   * their outputs are kept together so the function-call replay stays valid.
   */
  _trimInputToFit(inputList: any[], maxBytes: number): any[] {
    if (this._estimateInputBytes(inputList) <= maxBytes) return inputList;

    // Never touch the first entry — it carries the system instructions.
    const head = inputList[0];
    const rest = inputList.slice(1);

    // Drop oldest pairs until we fit. Walk from the front, removing one
    // assistant/user exchange at a time (keep function_call + function_call_output paired).
    let trimmed = rest;
    while (
      trimmed.length > 2 &&
      this._estimateInputBytes([head, ...trimmed]) > maxBytes
    ) {
      // Find the next safe cut point: the earliest index after which we can slice
      // without orphaning a function_call from its function_call_output.
      // Safe cut points: after a function_call_output, or after any non-function_call item.
      // Unsafe: after a function_call (the output must follow).
      let cut = 0; // Default: drop nothing (keep everything)
      for (let i = 0; i < trimmed.length; i++) {
        if (trimmed[i].type === "function_call") {
          // Can't cut here, need to keep looking for the function_call_output
          continue;
        }
        // Safe to cut here (function_call_output or non-function_call item)
        cut = i + 1;
      }
      // If cut is 0, we can't safely drop anything without breaking a pair;
      // force-drop the oldest exchange to avoid an infinite loop.
      if (cut === 0) cut = 1;
      trimmed = trimmed.slice(cut);
    }

    return [head, ...trimmed];
  }

  /**
   * Stream generate using an OpenAI Responses-compatible API with unified conversion methods.
   */
  async *_streamingResponseInternal(options: {
    messages: UniMessage[];
    config: UniConfig;
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent> {
    const openaiConfig = this.transformUniConfigToModelConfig(options.config);
    let inputList = this.transformUniMessageToModelInput(
      options.messages,
      options.signal,
    );

    // Guard against gateways that cap the HTTP body (nginx 413). The Console Go
    // proxy family sits behind nginx with a ~1 MB client_max_body_size; a long
    // history or a verbose tool catalog blows past it. Trim to a safe ceiling.
    const MAX_BODY_BYTES = 900 * 1024;
    if (this._estimateInputBytes(inputList) > MAX_BODY_BYTES) {
      inputList = this._trimInputToFit(inputList, MAX_BODY_BYTES);
    }

    // DEBUG: log the input list to diagnose function_call issues
    if (process.env.DEBUG_RESPONSES) {
      console.error("[DEBUG] Responses input:", JSON.stringify(inputList, null, 2));
    }

    // Validate: every function_call_output must have a matching function_call before it.
    // The Console Go proxy rejects the request with "No function call found for function
    // call output" if any output is orphaned, so strip orphans defensively.
    {
      const seenCallIds = new Set<string>();
      const validated: any[] = [];
      for (const item of inputList) {
        if (item.type === "function_call") {
          seenCallIds.add(item.call_id);
          validated.push(item);
        } else if (item.type === "function_call_output") {
          if (seenCallIds.has(item.call_id)) {
            validated.push(item);
          } else {
            // Orphaned output — drop it to avoid a 400.
            console.error(
              "[openai_responses] Dropping orphaned function_call_output with call_id:",
              item.call_id,
            );
          }
        } else {
          validated.push(item);
        }
      }
      inputList = validated;
    }

    const partialToolCall: {
      name?: string;
      arguments?: string;
      tool_call_id?: string;
    } = {};

    const params: ResponseCreateParamsStreaming = {
      ...openaiConfig,
      input: inputList,
      stream: true,
    };

    const stream = await this._client.responses.create(params, {
      signal: options.signal,
    });
    for await (const event of stream) {
      const uniEvent = this.transformModelOutputToUniEvent(event);

      if (uniEvent.event_type === "start") {
        for (const item of uniEvent.content_items) {
          if (item.type === "partial_tool_call") {
            partialToolCall.name = item.name;
            partialToolCall.arguments = "";
            partialToolCall.tool_call_id = item.tool_call_id;
            yield uniEvent;
          }
        }
      } else if (uniEvent.event_type === "delta") {
        for (const item of uniEvent.content_items) {
          if (item.type === "partial_tool_call") {
            partialToolCall.arguments =
              (partialToolCall.arguments || "") + item.arguments;
          }
        }

        yield uniEvent;
      } else if (uniEvent.event_type === "stop") {
        if (partialToolCall.name && partialToolCall.arguments !== undefined) {
          yield {
            role: "assistant",
            event_type: "delta",
            content_items: [
              {
                type: "tool_call",
                name: partialToolCall.name,
                arguments: parseToolCallArguments(
                  partialToolCall.arguments,
                  this.constructor.name,
                  partialToolCall.name || "",
                  partialToolCall.tool_call_id || "",
                ),
                tool_call_id: partialToolCall.tool_call_id || "",
              },
            ],
            usage_metadata: null,
            finish_reason: null,
          };
          partialToolCall.name = undefined;
          partialToolCall.arguments = undefined;
          partialToolCall.tool_call_id = undefined;
        }

        if (uniEvent.finish_reason || uniEvent.usage_metadata) {
          yield uniEvent;
        }
      }
    }
  }

  /**
   * List the model ids the configured endpoint serves.
   *
   * @returns The model ids, in the order the endpoint returned them.
   */
  async listModels(): Promise<string[]> {
    const models: string[] = [];
    for await (const model of this._client.models.list()) {
      models.push(model.id);
    }

    return models;
  }
}
