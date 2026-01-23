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
  EventType,
  FinishReason,
  PartialContentItem,
  ThinkingLevel,
  ToolChoice,
  UniConfig,
  UniEvent,
  UniMessage,
  UsageMetadata,
} from "../types";

/**
 * GPT-5.2-specific LLM client implementation.
 */
export class GPT5_2Client extends LLMClient {
  protected _model: string;
  private _client: OpenAI;

  /**
   * Initialize GPT-5.2 client with model and API key.
   */
  constructor(options: {
    model: string;
    apiKey?: string;
    baseUrl?: string | null;
    clientType?: string | null;
  }) {
    super();
    this._model = options.model;
    const key = options.apiKey || process.env.OPENAI_API_KEY || undefined;
    const url = options.baseUrl || process.env.OPENAI_BASE_URL || undefined;
    this._client = new OpenAI({ apiKey: key, baseURL: url });
  }

  /**
   * Convert ThinkingLevel enum to OpenAI's reasoning effort.
   */
  private _convertThinkingLevelToEffort(
    thinkingLevel: ThinkingLevel
  ): string {
    const mapping: { [key: string]: string } = {
      [ThinkingLevel.NONE]: "none",
      [ThinkingLevel.LOW]: "low",
      [ThinkingLevel.MEDIUM]: "medium",
      [ThinkingLevel.HIGH]: "high",
    };
    return mapping[thinkingLevel];
  }

  /**
   * Convert ToolChoice to OpenAI's tool_choice format with allowed tools support.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _convertToolChoice(toolChoice: ToolChoice): any {
    if (Array.isArray(toolChoice)) {
      return {
        mode: "required",
        tools: toolChoice.map((name) => ({ type: "function", name })),
      };
    } else if (toolChoice === "none") {
      return "none";
    } else if (toolChoice === "auto") {
      return "auto";
    } else if (toolChoice === "required") {
      return "required";
    }
  }

  /**
   * Transform universal configuration to OpenAI Responses API configuration.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformUniConfigToModelConfig(config: UniConfig): any {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const openaiConfig: any = {
      model: this._model,
      store: false,
      include: ["reasoning.encrypted_content"],
    };

    if (config.system_prompt !== undefined) {
      openaiConfig.instructions = config.system_prompt;
    }

    if (config.max_tokens !== undefined) {
      openaiConfig.max_output_tokens = config.max_tokens;
    }

    if (config.temperature !== undefined && config.temperature !== 1.0) {
      throw new Error("GPT-5.2 does not support setting temperature.");
    }

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

    return openaiConfig;
  }

  /**
   * Transform universal message format to OpenAI Responses API input format.
   */
  transformUniMessageToModelInput(messages: UniMessage[]): ResponseInputItem[] {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const inputList: any[] = [];

    for (const msg of messages) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const contentItems: any[] = [];
      for (const item of msg.content_items) {
        if (item.type === "text") {
          if (msg.role === "user") {
            contentItems.push({ type: "input_text", text: item.text });
          } else {
            contentItems.push({ type: "output_text", text: item.text });
          }
        } else if (item.type === "image_url") {
          contentItems.push({
            type: "input_image",
            image_url: item.image_url,
          });
        } else if (item.type === "thinking") {
          const signatureStr = typeof item.signature === 'string'
            ? item.signature
            : item.signature?.toString() || '{}';
          const signature = JSON.parse(signatureStr);
          inputList.push({
            type: "reasoning",
            id: signature.id,
            summary: item.thinking
              ? [{ type: "summary_text", text: item.thinking }]
              : [],
            encrypted_content: signature.encrypted_content,
          });
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

          inputList.push({
            type: "function_call_output",
            call_id: item.tool_call_id,
            output: item.result,
          });
        } else {
          throw new Error(`Unknown item: ${JSON.stringify(item)}`);
        }
      }

      if (contentItems.length > 0) {
        inputList.push({ role: msg.role, content: contentItems });
      }
    }

    return inputList;
  }

  /**
   * Transform OpenAI Responses API streaming event to universal event format.
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
    } else if (openaiEventType === "response.reasoning_summary_text.delta") {
      eventType = "delta";
      contentItems.push({ type: "thinking", thinking: modelOutput.delta });
    } else if (openaiEventType === "response.output_item.added") {
      const item = modelOutput.item;
      if (item.type === "function_call") {
        eventType = "start";
        contentItems.push({
          type: "partial_tool_call",
          name: item.name,
          arguments: "",
          tool_call_id: item.call_id,
        });
      } else if (item.type === "reasoning") {
        eventType = "delta";
        const signature = {
          id: item.id,
          encrypted_content: item.encrypted_content,
        };
        contentItems.push({
          type: "thinking",
          thinking: "",
          signature: JSON.stringify(signature),
        });
      } else {
        eventType = "unused";
      }
    } else if (openaiEventType === "response.output_item.done") {
      // not sure about the signature of openai, need to check
      const item = modelOutput.item;
      if (item.type === "reasoning") {
        eventType = "delta";
        const signature = {
          id: item.id,
          encrypted_content: item.encrypted_content,
        };
        contentItems.push({
          type: "thinking",
          thinking: "",
          signature: JSON.stringify(signature),
        });
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
    } else if (openaiEventType === "response.completed") {
      eventType = "stop";
      const response = modelOutput.response;
      const finishReasonMapping: { [key: string]: FinishReason } = {
        completed: "stop",
        incomplete: "length",
      };
      if (response.status) {
        finishReason = finishReasonMapping[response.status]
      }
      if (response.usage) {
        usageMetadata = {
          prompt_tokens: response.usage.input_tokens,
          thoughts_tokens: response.usage.output_tokens_details.reasoning_tokens,
          response_tokens: response.usage.output_tokens,
          cached_tokens: response.usage.input_tokens_details.cached_tokens,
        };
      }
    } else if (
      [
        "response.created",
        "response.in_progress",
        "response.output_text.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.done",
        "response.content_part.added",
        "response.content_part.done",
      ].includes(openaiEventType)
    ) {
      eventType = "unused";
    } else {
      throw new Error(`Unknown output: ${JSON.stringify(modelOutput)}`);
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
   * Stream generate using OpenAI Responses API with unified conversion methods.
   */
  async *streamingResponse(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncGenerator<UniEvent> {
    const openaiConfig = this.transformUniConfigToModelConfig(options.config);
    const inputList = this.transformUniMessageToModelInput(options.messages);

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

    const stream = await this._client.responses.create(params);

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
        if (
          partialToolCall.name &&
          partialToolCall.arguments !== undefined
        ) {
          yield {
            role: "assistant",
            event_type: "delta",
            content_items: [
              {
                type: "tool_call",
                name: partialToolCall.name,
                arguments: JSON.parse(partialToolCall.arguments),
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
}
