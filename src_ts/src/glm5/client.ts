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
 * GLM-5-specific LLM client implementation using OpenAI-compatible API.
 */
export class GLM5Client extends LLMClient {
  protected _model: string;
  private _client: OpenAI;

  /**
   * Initialize GLM-5 client with model and API key.
   */
  constructor(options: {
    model: string;
    apiKey?: string;
    baseUrl?: string | null;
    clientType?: string | null;
  }) {
    super();
    this._model = options.model;
    const key = options.apiKey || process.env.GLM_API_KEY || undefined;
    const url =
      options.baseUrl ||
      process.env.GLM_BASE_URL ||
      "https://api.z.ai/api/paas/v4/";
    this._client = new OpenAI({ apiKey: key, baseURL: url });
  }

  /**
   * Convert ThinkingLevel enum to GLM's thinking configuration.
   */
  private _convertThinkingLevelToConfig(thinkingLevel: ThinkingLevel): {
    type: string;
  } {
    const mapping: { [key: string]: { type: string } } = {
      [ThinkingLevel.NONE]: { type: "disabled" },
      [ThinkingLevel.LOW]: { type: "enabled" },
      [ThinkingLevel.MEDIUM]: { type: "enabled" },
      [ThinkingLevel.HIGH]: { type: "enabled" },
    };
    return mapping[thinkingLevel];
  }

  /**
   * Convert ToolChoice to OpenAI's tool_choice format.
   */
  private _convertToolChoice(toolChoice: ToolChoice): string {
    if (toolChoice === "auto") {
      return "auto";
    } else {
      throw new Error('GLM only supports "auto" for tool_choice.');
    }
  }

  /**
   * Transform universal configuration to GLM-specific configuration.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformUniConfigToModelConfig(config: UniConfig): any {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const glmConfig: any = {
      model: this._model,
      stream: true,
      extra_body: { tool_stream: true },
    };

    if (config.max_tokens !== undefined) {
      glmConfig.max_tokens = config.max_tokens;
    }

    if (config.temperature !== undefined) {
      glmConfig.temperature = config.temperature;
    }

    if (config.thinking_level !== undefined) {
      const thinkingConfig = this._convertThinkingLevelToConfig(
        config.thinking_level,
      );
      glmConfig.extra_body = {
        ...(glmConfig.extra_body || {}),
        thinking: thinkingConfig,
      };
    }

    if (config.tools !== undefined) {
      glmConfig.tools = config.tools.map((tool) => ({
        type: "function",
        function: tool,
      }));
    }

    if (config.tool_choice !== undefined) {
      glmConfig.tool_choice = this._convertToolChoice(config.tool_choice);
    }

    if (
      config.prompt_caching !== undefined &&
      config.prompt_caching !== PromptCaching.ENABLE
    ) {
      throw new Error("prompt_caching must be ENABLE for GLM-5.");
    }

    return glmConfig;
  }

  /**
   * Transform universal message format to OpenAI's message format.
   */
  transformUniMessageToModelInput(
    messages: UniMessage[],
  ): ChatCompletionMessageParam[] {
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

      for (const item of msg.content_items) {
        if (item.type === "text") {
          contentParts.push({ type: "text", text: item.text });
        } else if (item.type === "image_url") {
          contentParts.push({
            type: "image_url",
            image_url: { url: item.image_url },
          });
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
            throw new Error("GLM does not support images in tool results.");
          }

          openaiMessages.push({
            role: "tool",
            tool_call_id: item.tool_call_id,
            content: item.text,
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
        message.reasoning = thinking;
      }

      if (Object.keys(message).length > 1) {
        openaiMessages.push(message);
      }
    }

    return openaiMessages;
  }

  /**
   * Transform GLM model output to universal event format.
   */
  transformModelOutputToUniEvent(modelOutput: ChatCompletionChunk): UniEvent {
    let eventType: EventType | null = null;
    const contentItems: PartialContentItem[] = [];
    let usageMetadata: UsageMetadata | null = null;
    let finishReason: FinishReason | null = null;

    const choice = modelOutput.choices[0];
    const delta = choice?.delta;

    if (delta?.content) {
      eventType = "delta";
      contentItems.push({ type: "text", text: delta.content });
    }

    // vLLM & siliconflow compatibility
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if ((delta as any)?.reasoning_content) {
      eventType = "delta";
      contentItems.push({
        type: "thinking",
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        thinking: (delta as any).reasoning_content,
      });
    }
    // openrouter compatibility
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    else if ((delta as any)?.reasoning) {
      eventType = "delta";
      contentItems.push({
        type: "thinking",
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        thinking: (delta as any).reasoning,
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
      eventType = "stop";
      const finishReasonMapping: { [key: string]: FinishReason } = {
        stop: "stop",
        length: "length",
        tool_calls: "stop",
        content_filter: "stop",
      };
      finishReason = finishReasonMapping[choice.finish_reason] || "unknown";
    }

    if (modelOutput.usage) {
      eventType = eventType || "delta";

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
   * Stream generate using GLM SDK with unified conversion methods.
   */
  async *streamingResponse(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncGenerator<UniEvent> {
    const glmConfig = this.transformUniConfigToModelConfig(options.config);
    const glmMessages = this.transformUniMessageToModelInput(options.messages);

    if (options.config.system_prompt) {
      glmMessages.unshift({
        role: "system",
        content: options.config.system_prompt,
      });
    }

    const params: ChatCompletionCreateParamsStreaming = {
      ...glmConfig,
      messages: glmMessages,
      stream: true,
    };

    const stream = await this._client.chat.completions.create(params);

    const partialToolCall: {
      name?: string;
      arguments?: string;
      tool_call_id?: string;
    } = {};

    for await (const chunk of stream) {
      const event = this.transformModelOutputToUniEvent(chunk);
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
                    arguments: JSON.parse(partialToolCall.arguments || "{}"),
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
                arguments: JSON.parse(partialToolCall.arguments || "{}"),
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

        if (event.finish_reason || event.usage_metadata) {
          yield event;
        }
      }
    }
  }
}
