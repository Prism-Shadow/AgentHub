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
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionCreateParamsStreaming,
} from "openai/resources/chat/completions";
import { LLMClient } from "../baseClient";
import {
  parseToolCallArguments,
  UnsupportedParameterError,
} from "../errors";
import {
  EventType,
  FinishReason,
  PartialContentItem,
  PromptCaching,
  ThinkingLevel,
  ToolChoice,
  UniConfig,
  UniEvent,
  UniMessage,
  UsageMetadata,
} from "../types";

export class DeepSeekV4Client extends LLMClient {
  protected _model: string;
  private _client: OpenAI;

  constructor(options: {
    model: string;
    apiKey?: string;
    baseUrl?: string | null;
    clientType?: string | null;
    defaultHeaders?: Record<string, string>;
  }) {
    super();
    this._model = options.model;
    const key = options.apiKey || process.env.DEEPSEEK_API_KEY || undefined;
    const url =
      options.baseUrl ||
      process.env.DEEPSEEK_BASE_URL ||
      "https://api.deepseek.com";
    this._client = new OpenAI({
      apiKey: key,
      baseURL: url,
      defaultHeaders: options.defaultHeaders,
    });
  }

  private _convertThinkingLevelToConfig(thinkingLevel: ThinkingLevel): {
    type: string;
  } {
    const mapping: { [key: string]: { type: string } } = {
      [ThinkingLevel.NONE]: { type: "disabled" },
      [ThinkingLevel.LOW]: { type: "enabled" },
      [ThinkingLevel.MEDIUM]: { type: "enabled" },
      [ThinkingLevel.HIGH]: { type: "enabled" },
      [ThinkingLevel.XHIGH]: { type: "enabled" },
      [ThinkingLevel.MAX]: { type: "enabled" },
    };
    return mapping[thinkingLevel];
  }

  /**
   * Convert ThinkingLevel enum to DeepSeek's reasoning_effort.
   *
   * DeepSeek accepts low/high/max and maps medium and xhigh onto high server-side
   * (llmsdk_docs/deepseek_v4/docs/thinking-mode.md), so this sends the value the
   * server would settle on anyway.
   */
  private _convertReasoningEffort(thinkingLevel: ThinkingLevel): string | null {
    const mapping: { [key: string]: string | null } = {
      [ThinkingLevel.NONE]: null,
      [ThinkingLevel.LOW]: "low",
      [ThinkingLevel.MEDIUM]: "high",
      [ThinkingLevel.HIGH]: "high",
      [ThinkingLevel.XHIGH]: "high",
      [ThinkingLevel.MAX]: "max",
    };
    return mapping[thinkingLevel];
  }

  private _convertToolChoice(toolChoice: ToolChoice): string {
    if (toolChoice === "auto" || toolChoice === "none") {
      return toolChoice;
    }
    throw new UnsupportedParameterError({
      client: this.constructor.name,
      parameter: "tool_choice",
      message: 'DeepSeek V4 only supports "auto" and "none" for tool_choice.',
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformUniConfigToModelConfig(config: UniConfig): any {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const deepseekConfig: any = {
      model: this._model,
      stream: true,
      stream_options: { include_usage: true },
    };

    if (config.max_tokens !== undefined) {
      deepseekConfig.max_tokens = config.max_tokens;
    }

    if (config.temperature !== undefined && config.temperature !== 1.0) {
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "temperature",
        message: "DeepSeek V4 does not support setting temperature.",
      });
    }

    if (config.thinking_level !== undefined) {
      const thinkingConfig = this._convertThinkingLevelToConfig(
        config.thinking_level,
      );
      deepseekConfig.extra_body = {
        ...(deepseekConfig.extra_body || {}),
        thinking: thinkingConfig,
      };
      const reasoningEffort = this._convertReasoningEffort(
        config.thinking_level,
      );
      if (reasoningEffort !== null) {
        deepseekConfig.reasoning_effort = reasoningEffort;
      }
    }

    if (config.tools !== undefined) {
      deepseekConfig.tools = config.tools.map((tool) => ({
        type: "function",
        function: tool,
      }));
    }

    if (config.tool_choice !== undefined) {
      deepseekConfig.tool_choice = this._convertToolChoice(config.tool_choice);
    }

    if (config.fast_mode) {
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "fast_mode",
        message: "DeepSeek V4 does not support fast mode.",
      });
    }

    if (
      config.prompt_caching !== undefined &&
      config.prompt_caching !== PromptCaching.ENABLE
    ) {
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "prompt_caching",
        message: "prompt_caching must be ENABLE for DeepSeek.",
      });
    }

    return deepseekConfig;
  }

  transformUniMessageToModelInput(
    messages: UniMessage[],
    _signal?: AbortSignal,
  ): ChatCompletionMessageParam[] {
    const deepseekMessages: ChatCompletionMessageParam[] = [];

    for (const msg of messages) {
      const contentParts: Array<{
        type: string;
        text?: string;
        image_url?: { url: string };
      }> = [];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const toolCalls: any[] = [];
      let thinking = "";

      for (const item of msg.content_items) {
        if (item.type === "text") {
          contentParts.push({ type: "text", text: item.text });
        } else if (item.type === "image_url") {
          throw new Error("DeepSeek does not support image url inputs.");
        } else if (item.type === "thinking") {
          thinking += item.thinking;
        } else if (item.type === "tool_call") {
          toolCalls.push({
            id: item.tool_call_id,
            type: "function",
            function: {
              name: item.name,
              arguments: JSON.stringify(item.arguments, null, 0),
            },
          });
        } else if (item.type === "tool_result") {
          if (!item.tool_call_id) {
            throw new Error("tool_call_id is required for tool result.");
          }

          if (item.images && item.images.length > 0) {
            throw new Error(
              "DeepSeek does not support images in tool results.",
            );
          }

          deepseekMessages.push({
            role: "tool",
            tool_call_id: item.tool_call_id,
            content: [{ type: "text" as const, text: item.text }],
          });
        } else {
          throw new Error(
            `Unknown item type: ${(item as { type: string }).type}`,
          );
        }
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const message: any = { role: msg.role };
      if (contentParts.length > 0) {
        message.content = contentParts;
      }

      if (toolCalls.length > 0) {
        message.tool_calls = toolCalls;
      }

      if (thinking) {
        message.reasoning_content = thinking;
      }

      if (Object.keys(message).length > 1) {
        deepseekMessages.push(message);
      }
    }

    return deepseekMessages;
  }

  transformModelOutputToUniEvent(modelOutput: ChatCompletionChunk): UniEvent {
    let eventType: EventType | null = null;
    const contentItems: PartialContentItem[] = [];
    let usageMetadata: UsageMetadata | null = null;
    let finishReason: FinishReason | null = null;

    // gateways inject content-free heartbeat chunks on long generations, whose
    // choices arrive as undefined rather than an empty list
    if (modelOutput.choices?.length) {
      const choice = modelOutput.choices[0];
      const delta = choice?.delta;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if ((delta as any)?.reasoning_content) {
        eventType = "delta";
        // record the wire field so a replay through another OpenAI-compatible
        // client reproduces the exact field DeepSeek produced
        contentItems.push({
          type: "thinking",
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          thinking: (delta as any).reasoning_content,
          fidelity: { reasoning_field: "reasoning_content" },
        });
      }

      if (delta?.content) {
        eventType = "delta";
        contentItems.push({ type: "text", text: delta.content });
      }

      if (delta?.tool_calls) {
        eventType = "delta";
        for (const toolCall of delta.tool_calls) {
          contentItems.push({
            type: "partial_tool_call",
            name: toolCall.function?.name || "",
            arguments: toolCall.function?.arguments || "",
            tool_call_id: toolCall.id || "",
          });
        }
      }

      if (choice?.finish_reason) {
        eventType = eventType || "stop";
        const finishReasonMapping: { [key: string]: FinishReason } = {
          stop: "stop",
          length: "length",
          tool_calls: "tool_call",
          content_filter: "stop",
        };
        finishReason = finishReasonMapping[choice.finish_reason] || "unknown";
      }
    }

    if (modelOutput.usage) {
      eventType = eventType || "stop";
      const completionTokenDetails =
        modelOutput.usage.completion_tokens_details;
      const reasoningTokens = completionTokenDetails
        ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
          ((completionTokenDetails as any).reasoning_tokens ?? null)
        : null;
      const responseTokens =
        modelOutput.usage.completion_tokens - (reasoningTokens || 0);

      // usage.prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens
      usageMetadata = {
        cached_tokens:
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (modelOutput.usage as any).prompt_cache_hit_tokens ?? null,
        prompt_tokens:
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (modelOutput.usage as any).prompt_cache_miss_tokens ?? null,
        thoughts_tokens: reasoningTokens,
        response_tokens: responseTokens,
      };
    }

    return {
      role: "assistant",
      event_type: eventType as EventType,
      content_items: contentItems,
      usage_metadata: usageMetadata,
      finish_reason: finishReason,
    };
  }

  async *_streamingResponseInternal(options: {
    messages: UniMessage[];
    config: UniConfig;
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent> {
    const deepseekConfig = this.transformUniConfigToModelConfig(options.config);
    const deepseekMessages = this.transformUniMessageToModelInput(
      options.messages,
      options.signal,
    );

    if (options.config.system_prompt) {
      deepseekMessages.unshift({
        role: "system",
        content: options.config.system_prompt,
      });
    }

    const params: ChatCompletionCreateParamsStreaming = {
      ...deepseekConfig,
      messages: deepseekMessages,
      stream: true,
    };

    const stream = await this._client.chat.completions.create(params, {
      signal: options.signal,
    });

    const partialToolCall: {
      name?: string;
      arguments?: string;
      tool_call_id?: string;
    } = {};
    let partialUsage: {
      finish_reason?: FinishReason | null;
      usage_metadata?: UsageMetadata | null;
    } = {};

    for await (const chunk of stream) {
      const event = this.transformModelOutputToUniEvent(chunk);
      partialUsage.finish_reason =
        event.finish_reason || partialUsage.finish_reason;
      partialUsage.usage_metadata =
        event.usage_metadata || partialUsage.usage_metadata;

      if (event.event_type === "delta") {
        for (const item of event.content_items) {
          if (item.type === "partial_tool_call") {
            if (!partialToolCall.name) {
              partialToolCall.name = item.name;
              partialToolCall.arguments = item.arguments;
              partialToolCall.tool_call_id = item.tool_call_id;
            } else if (item.name) {
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
              partialToolCall.name = item.name;
              partialToolCall.arguments = item.arguments;
              partialToolCall.tool_call_id = item.tool_call_id;
            } else {
              partialToolCall.arguments =
                (partialToolCall.arguments || "") + item.arguments;
            }
          }
        }
        yield event;
      } else if (event.event_type === "stop") {
        if (partialToolCall.name) {
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

        if (partialUsage.finish_reason && partialUsage.usage_metadata) {
          yield {
            role: "assistant",
            event_type: "stop",
            content_items: [],
            usage_metadata: partialUsage.usage_metadata,
            finish_reason: partialUsage.finish_reason,
          };
          partialUsage.finish_reason = null;
          partialUsage.usage_metadata = null;
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
