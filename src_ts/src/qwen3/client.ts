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
  ToolChoice,
  UniConfig,
  UniEvent,
  UniMessage,
  UsageMetadata,
} from "../types";

/**
 * Qwen3-specific LLM client implementation using OpenAI-compatible API.
 */
export class Qwen3Client extends LLMClient {
  private _model: string;
  private _client: OpenAI;

  /**
   * Initialize Qwen3 client with model and API key.
   */
  constructor(
    model: string,
    apiKey?: string | null,
    baseUrl?: string | null
  ) {
    super();
    this._model = model;
    const key = apiKey || process.env.QWEN3_API_KEY || undefined;
    const url =
      baseUrl || process.env.QWEN3_BASE_URL || "http://127.0.0.1:8000/v1/";
    this._client = new OpenAI({ apiKey: key, baseURL: url });
  }

  /**
   * Convert ToolChoice to OpenAI's tool_choice format.
   */
  private _convertToolChoice(toolChoice: ToolChoice): string {
    if (toolChoice === "auto") {
      return "auto";
    } else {
      throw new Error("Qwen3 only supports 'auto' for tool_choice.");
    }
  }

  /**
   * Transform universal configuration to Qwen3-specific configuration.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformUniConfigToModelConfig(config: UniConfig): any {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const qwen3Config: any = {
      model: this._model,
      stream: true,
    };

    if (config.max_tokens !== undefined) {
      qwen3Config.max_tokens = config.max_tokens;
    }

    if (config.temperature !== undefined) {
      qwen3Config.temperature = config.temperature;
    }

    if (config.tools !== undefined) {
      qwen3Config.tools = config.tools.map((tool) => ({
        type: "function",
        function: tool,
      }));
    }

    if (config.tool_choice !== undefined) {
      qwen3Config.tool_choice = this._convertToolChoice(config.tool_choice);
    }

    if (
      config.prompt_caching !== undefined &&
      config.prompt_caching !== PromptCaching.ENABLE
    ) {
      throw new Error("prompt_caching must be ENABLE for Qwen3.");
    }

    return qwen3Config;
  }

  /**
   * Transform universal message format to Qwen3-specific message format.
   */
  transformUniMessageToModelInput(
    messages: UniMessage[]
  ): ChatCompletionMessageParam[] {
    const qwen3Messages: ChatCompletionMessageParam[] = [];

    for (const msg of messages) {
      const contentParts: Array<{ type: string; text?: string }> = [];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const toolCalls: any[] = [];
      let thinking = "";

      for (const item of msg.content_items) {
        if (item.type === "text") {
          contentParts.push({ type: "text", text: item.text });
        } else if (item.type === "image_url") {
          throw new Error("Qwen3 does not support image_url.");
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
          qwen3Messages.push({
            role: "tool",
            tool_call_id: item.tool_call_id,
            content: item.result,
          });
        } else {
          throw new Error(`Unknown item type: ${(item as { type: string }).type}`);
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
        qwen3Messages.push(message);
      }
    }

    return qwen3Messages;
  }

  /**
   * Transform Qwen3 model output to universal event format.
   */
  transformModelOutputToUniEvent(modelOutput: ChatCompletionChunk): UniEvent {
    let eventType: EventType | null = null;
    const contentItems: PartialContentItem[] = [];
    let usageMetadata: UsageMetadata | null = null;
    let finishReason: FinishReason | null = null;

    const choice = modelOutput.choices[0];
    const delta = choice?.delta;

    if (delta?.content) {
      if (delta.content === "<tool_call>") {
        eventType = "start";
      } else if (delta.content === "</tool_call>") {
        eventType = "stop";
      } else {
        eventType = "delta";
        contentItems.push({ type: "text", text: delta.content });
      }
    }

    // vLLM & siliconflow compatibility
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if ((delta as any)?.reasoning_content) {
      eventType = "delta";
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      contentItems.push({ type: "thinking", thinking: (delta as any).reasoning_content });
    }
    // openrouter compatibility
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    else if ((delta as any)?.reasoning) {
      eventType = "delta";
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      contentItems.push({ type: "thinking", thinking: (delta as any).reasoning });
    }

    if (delta?.tool_calls) {
      for (const toolCall of delta.tool_calls) {
        eventType = "delta";
        contentItems.push({
          type: "partial_tool_call",
          name: toolCall.function?.name || "",
          arguments: toolCall.function?.arguments || "",
          tool_call_id: toolCall.function?.name || "",
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
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const reasoningTokens = (modelOutput.usage as any).completion_tokens_details?.reasoning_tokens || null;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const cachedTokens = (modelOutput.usage as any).prompt_tokens_details?.cached_tokens || null;

      usageMetadata = {
        prompt_tokens: modelOutput.usage.prompt_tokens || null,
        thoughts_tokens: reasoningTokens,
        response_tokens: modelOutput.usage.completion_tokens || null,
        cached_tokens: cachedTokens,
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

  /**
   * Stream generate using Qwen3 SDK with unified conversion methods.
   */
  async *streamingResponse(
    messages: UniMessage[],
    config: UniConfig
  ): AsyncGenerator<UniEvent> {
    const qwen3Config = this.transformUniConfigToModelConfig(config);
    const qwen3Messages = this.transformUniMessageToModelInput(messages);

    if (config.system_prompt) {
      qwen3Messages.unshift({
        role: "system",
        content: config.system_prompt,
      });
    }

    const params: ChatCompletionCreateParamsStreaming = {
      ...qwen3Config,
      messages: qwen3Messages,
      stream: true,
    };

    const stream = await this._client.chat.completions.create(params);

    const partialToolCall: {
      name?: string;
      arguments?: string;
      tool_call_id?: string;
      data?: string;
    } = {};

    for await (const chunk of stream) {
      const event = this.transformModelOutputToUniEvent(chunk);
      if (event.event_type === "start") {
        partialToolCall.data = "";
      } else if (event.event_type === "delta") {
        if (partialToolCall.data !== undefined) {
          partialToolCall.data += event.content_items[0]?.type === "text"
            ? (event.content_items[0] as { text: string }).text
            : "";
          continue;
        }

        for (const item of event.content_items) {
          if (item.type === "partial_tool_call") {
            if (!partialToolCall.name) {
              partialToolCall.name = item.name;
              partialToolCall.arguments = "";
            } else {
              partialToolCall.arguments =
                (partialToolCall.arguments || "") + item.arguments;
            }
          }
        }

        yield event;
      } else if (event.event_type === "stop") {
        if (partialToolCall.data !== undefined) {
          const toolCall = JSON.parse(partialToolCall.data.trim());
          yield {
            role: "assistant",
            event_type: "delta",
            content_items: [
              {
                type: "partial_tool_call",
                name: toolCall.name,
                arguments: JSON.stringify(toolCall.arguments, null, 0),
                tool_call_id: toolCall.name,
              },
            ],
            usage_metadata: null,
            finish_reason: null,
          };
          yield {
            role: "assistant",
            event_type: "delta",
            content_items: [
              {
                type: "tool_call",
                name: toolCall.name,
                arguments: toolCall.arguments,
                tool_call_id: toolCall.name,
              },
            ],
            usage_metadata: null,
            finish_reason: null,
          };
          partialToolCall.data = undefined;
        }

        if (partialToolCall.name && partialToolCall.arguments) {
          yield {
            role: "assistant",
            event_type: "delta",
            content_items: [
              {
                type: "tool_call",
                name: partialToolCall.name,
                arguments: JSON.parse(partialToolCall.arguments),
                tool_call_id: partialToolCall.name,
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
