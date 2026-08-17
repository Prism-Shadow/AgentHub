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

import * as path from "path";
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
import { fixOpenrouterUsageMetadata } from "../utils";

/**
 * Kimi K3-specific LLM client implementation using OpenAI-compatible API.
 */
export class KimiK3Client extends LLMClient {
  protected _model: string;
  private _client: OpenAI;

  /**
   * Initialize Kimi K3 client with model and API key.
   */
  constructor(options: {
    model: string;
    apiKey?: string;
    baseUrl?: string | null;
    clientType?: string | null;
  }) {
    super();
    this._model = options.model;
    const key = options.apiKey || process.env.MOONSHOT_API_KEY || undefined;
    const url =
      options.baseUrl ||
      process.env.MOONSHOT_BASE_URL ||
      "https://api.moonshot.cn/v1";
    this._client = new OpenAI({ apiKey: key, baseURL: url });
  }

  /**
   * Detect MIME type from URL extension for image.
   */
  private _detectImageMimeType(url: string): string {
    const ext = path.extname(url).toLowerCase();
    const mimeTypes: { [key: string]: string } = {
      ".bmp": "image/bmp",
      ".gif": "image/gif",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".png": "image/png",
      ".svg": "image/svg+xml",
      ".tiff": "image/tiff",
      ".webp": "image/webp",
    };
    return mimeTypes[ext] || "image/jpeg";
  }

  /**
   * Convert image URL to base64-encoded data URL.
   */
  private async _convertImageUrlToBase64(
    url: string,
    signal?: AbortSignal,
  ): Promise<string> {
    if (url.startsWith("data:")) {
      return url;
    }

    const response = await fetch(url, { signal });
    if (!response.ok) {
      throw new Error(
        `Failed to fetch image: ${response.status} ${response.statusText}`,
      );
    }
    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const mimeType = this._detectImageMimeType(url);
    const base64String = buffer.toString("base64");
    return `data:${mimeType};base64,${base64String}`;
  }

  /**
   * Convert ThinkingLevel enum to Kimi K3's reasoning_effort.
   *
   * K3 cannot disable reasoning, so NONE degrades to the lowest effort
   * instead of throwing.
   */
  private _convertThinkingLevelToReasoningEffort(
    thinkingLevel: ThinkingLevel,
  ): string {
    const mapping: { [key: string]: string } = {
      [ThinkingLevel.NONE]: "low",
      [ThinkingLevel.LOW]: "low",
      [ThinkingLevel.MEDIUM]: "high",
      [ThinkingLevel.HIGH]: "high",
      [ThinkingLevel.XHIGH]: "max",
    };
    return mapping[thinkingLevel];
  }

  /**
   * Convert ToolChoice to OpenAI's tool_choice format.
   */
  private _convertToolChoice(toolChoice: ToolChoice): string {
    if (toolChoice === "auto") {
      return "auto";
    } else if (toolChoice === "none") {
      return "none";
    } else if (toolChoice === "required") {
      return "required";
    } else {
      // forcing a specific tool is incompatible with K3's always-on reasoning
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "tool_choice",
        message:
          'Kimi K3 does not support forcing specific tools; only "auto", "none" and "required" are supported.',
      });
    }
  }

  /**
   * Transform universal configuration to Kimi K3-specific configuration.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformUniConfigToModelConfig(config: UniConfig): any {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const kimiConfig: any = {
      model: this._model,
      stream: true,
      stream_options: { include_usage: true },
    };

    if (config.max_tokens !== undefined) {
      kimiConfig.max_completion_tokens = config.max_tokens;
    }

    if (config.temperature !== undefined && config.temperature !== 1.0) {
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "temperature",
        message: "Kimi K3 does not support setting temperature.",
      });
    }

    if (config.thinking_level !== undefined) {
      kimiConfig.reasoning_effort = this._convertThinkingLevelToReasoningEffort(
        config.thinking_level,
      );
    }

    if (config.tools !== undefined) {
      kimiConfig.tools = config.tools.map((tool) => ({
        type: "function",
        function: tool,
      }));
    }

    if (config.tool_choice !== undefined) {
      kimiConfig.tool_choice = this._convertToolChoice(config.tool_choice);
    }

    if (config.fast_mode) {
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "fast_mode",
        message: "Kimi K3 does not support fast mode.",
      });
    }

    if (
      config.prompt_caching !== undefined &&
      config.prompt_caching !== PromptCaching.ENABLE
    ) {
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "prompt_caching",
        message: "prompt_caching must be ENABLE for Kimi K3.",
      });
    }

    // K3 context caching is automatic; trace_id is intentionally not sent
    // as prompt_cache_key
    return kimiConfig;
  }

  /**
   * Transform universal message format to OpenAI's message format.
   */
  async transformUniMessageToModelInput(
    messages: UniMessage[],
    signal?: AbortSignal,
  ): Promise<ChatCompletionMessageParam[]> {
    const openaiMessages: ChatCompletionMessageParam[] = [];

    for (const msg of messages) {
      const contentParts: Array<{
        type: string;
        text?: string;
        image_url?: { url: string };
      }> = [];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const toolCalls: any[] = [];
      let thinking = "";
      const thinkingFields = new Set<string | undefined>();

      for (const item of msg.content_items) {
        if (item.type === "text") {
          contentParts.push({ type: "text", text: item.text });
        } else if (item.type === "image_url") {
          const base64Image = await this._convertImageUrlToBase64(
            item.image_url,
            signal,
          );
          contentParts.push({
            type: "image_url",
            image_url: { url: base64Image },
          });
        } else if (item.type === "thinking") {
          thinking += item.thinking;
          thinkingFields.add(item.fidelity?.reasoning_field);
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

          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const content: any[] = [{ type: "text", text: item.text }];

          if (item.images && item.images.length > 0) {
            for (const imageUrl of item.images) {
              const base64Image = await this._convertImageUrlToBase64(
                imageUrl,
                signal,
              );
              if (this._client.baseURL.includes("siliconflow.cn")) {
                // siliconflow does not support image_url in tool result
                contentParts.push({
                  type: "image_url",
                  image_url: { url: base64Image },
                });
              } else {
                content.push({
                  type: "image_url",
                  image_url: { url: base64Image },
                });
              }
            }
          }

          openaiMessages.push({
            role: "tool",
            tool_call_id: item.tool_call_id,
            content,
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
        // send thinking back through the exact field the upstream produced (recorded
        // in the item fidelity); servers may reject the spelling they did not emit
        if (
          thinkingFields.size === 1 &&
          thinkingFields.has("reasoning_content")
        ) {
          message.reasoning_content = thinking;
        } else if (
          thinkingFields.size === 1 &&
          thinkingFields.has("reasoning")
        ) {
          message.reasoning = thinking;
        } else {
          message.reasoning_content = thinking; // vLLM & siliconflow compatibility
          message.reasoning = thinking; // openrouter compatibility
        }
      }

      if (Object.keys(message).length > 1) {
        openaiMessages.push(message);
      }
    }

    return openaiMessages;
  }

  /**
   * Transform Kimi K3 model output to universal event format.
   */
  transformModelOutputToUniEvent(modelOutput: ChatCompletionChunk): UniEvent {
    let eventType: EventType | null = null;
    const contentItems: PartialContentItem[] = [];
    let usageMetadata: UsageMetadata | null = null;
    let finishReason: FinishReason | null = null;

    if (modelOutput.choices.length > 0) {
      const choice = modelOutput.choices[0];
      const delta = choice?.delta;

      if (delta?.content) {
        eventType = "delta";
        contentItems.push({ type: "text", text: delta.content });
      }

      // the thinking field name differs by server: vLLM & siliconflow use
      // reasoning_content while openrouter uses reasoning; record the wire
      // field that carried each delta so a replay can reproduce exactly the
      // field the upstream produced
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const reasoningContent = (delta as any)?.reasoning_content;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const reasoning = (delta as any)?.reasoning;
      if (reasoningContent && reasoning) {
        eventType = "delta";
        // ambiguous origin: record no fidelity so a replay sends both fields back
        contentItems.push({ type: "thinking", thinking: reasoningContent });
      } else if (reasoningContent) {
        eventType = "delta";
        contentItems.push({
          type: "thinking",
          thinking: reasoningContent,
          fidelity: { reasoning_field: "reasoning_content" },
        });
      } else if (reasoning) {
        eventType = "delta";
        contentItems.push({
          type: "thinking",
          thinking: reasoning,
          fidelity: { reasoning_field: "reasoning" },
        });
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

      const cachedTokens =
        modelOutput.usage.prompt_tokens_details?.cached_tokens || null;
      const reasoningTokens =
        modelOutput.usage.completion_tokens_details?.reasoning_tokens || null;

      const promptTokens =
        cachedTokens !== null
          ? modelOutput.usage.prompt_tokens - cachedTokens
          : modelOutput.usage.prompt_tokens;
      const responseTokens =
        reasoningTokens !== null
          ? modelOutput.usage.completion_tokens - reasoningTokens
          : modelOutput.usage.completion_tokens;

      usageMetadata = {
        cached_tokens: cachedTokens,
        prompt_tokens: promptTokens,
        thoughts_tokens: reasoningTokens,
        response_tokens: responseTokens,
      };
      usageMetadata = fixOpenrouterUsageMetadata(
        usageMetadata,
        this._client.baseURL,
      );
    }

    return {
      role: "assistant",
      event_type: eventType as EventType,
      content_items: contentItems,
      usage_metadata: usageMetadata,
      finish_reason: finishReason,
    };
  }

  /**
   * Stream generate using Kimi SDK with unified conversion methods.
   */
  async *_streamingResponseInternal(options: {
    messages: UniMessage[];
    config: UniConfig;
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent> {
    const kimiConfig = this.transformUniConfigToModelConfig(options.config);
    const kimiMessages = await this.transformUniMessageToModelInput(
      options.messages,
      options.signal,
    );

    if (options.config.system_prompt) {
      kimiMessages.unshift({
        role: "system",
        content: options.config.system_prompt,
      });
    }

    const params: ChatCompletionCreateParamsStreaming = {
      ...kimiConfig,
      messages: kimiMessages,
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
      // the finish reason and usage metadata should be accumulated
      partialUsage.finish_reason =
        event.finish_reason || partialUsage.finish_reason;
      partialUsage.usage_metadata =
        event.usage_metadata || partialUsage.usage_metadata;
      if (event.event_type === "delta") {
        for (const item of event.content_items) {
          if (item.type === "partial_tool_call") {
            if (!partialToolCall.name) {
              // start a new partial tool call
              partialToolCall.name = item.name;
              partialToolCall.arguments = item.arguments;
              partialToolCall.tool_call_id = item.tool_call_id;
            } else if (item.name) {
              // finish the previous partial tool call
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
              // start a new partial tool call
              partialToolCall.name = item.name;
              partialToolCall.arguments = item.arguments;
              partialToolCall.tool_call_id = item.tool_call_id;
            } else {
              // update partial tool call
              partialToolCall.arguments =
                (partialToolCall.arguments || "") + item.arguments;
            }
          }
        }
        yield event;
      } else if (event.event_type === "stop") {
        if (partialToolCall.name) {
          // finish the partial tool call
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
}
