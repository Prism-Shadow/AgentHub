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

import Anthropic from "@anthropic-ai/sdk";
import {
  BetaMessageParam,
  BetaRawMessageStreamEvent,
} from "@anthropic-ai/sdk/resources/beta/messages";
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

/**
 * Claude 4.5-specific LLM client implementation.
 */
export class Claude4_5Client extends LLMClient {
  protected _model: string;
  private _client: Anthropic;

  /**
   * Initialize Claude 4.5 client with model and API key.
   */
  constructor(options: {
    model: string;
    apiKey?: string;
    baseUrl?: string | null;
    clientType?: string | null;
  }) {
    super();
    this._model = options.model;
    const key = options.apiKey || process.env.ANTHROPIC_API_KEY || undefined;
    const url = options.baseUrl || process.env.ANTHROPIC_BASE_URL || undefined;

    this._client = new Anthropic({
      apiKey: key,
      baseURL: url,
    });
  }

  /**
   * Convert ThinkingLevel enum to Claude's budget_tokens.
   */
  private _convertThinkingLevelToBudget(thinkingLevel: ThinkingLevel): {
    type: string;
    budget_tokens?: number;
  } {
    const mapping: { [key: string]: { type: string; budget_tokens?: number } } =
      {
        [ThinkingLevel.NONE]: { type: "disabled" },
        [ThinkingLevel.LOW]: { type: "enabled", budget_tokens: 1024 },
        [ThinkingLevel.MEDIUM]: { type: "enabled", budget_tokens: 4096 },
        [ThinkingLevel.HIGH]: { type: "enabled", budget_tokens: 16384 },
      };
    return mapping[thinkingLevel];
  }

  /**
   * Convert ToolChoice to Claude's tool_choice format.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _convertToolChoice(toolChoice: ToolChoice): any {
    if (Array.isArray(toolChoice)) {
      if (toolChoice.length > 1) {
        throw new Error("Claude supports only one tool choice.");
      }
      return { type: "any", name: toolChoice[0] };
    } else if (toolChoice === "none") {
      return { type: "none" };
    } else if (toolChoice === "auto") {
      return { type: "auto" };
    } else if (toolChoice === "required") {
      return { type: "any" };
    }
  }

  /**
   * Transform universal configuration to Claude-specific configuration.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformUniConfigToModelConfig(config: UniConfig): any {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const claudeConfig: any = {
      model: this._model,
      betas: ["interleaved-thinking-2025-05-14"],
    };

    if (config.system_prompt !== undefined) {
      claudeConfig.system = config.system_prompt;
    }

    if (config.max_tokens !== undefined) {
      claudeConfig.max_tokens = config.max_tokens;
    } else {
      claudeConfig.max_tokens = 32768;
    }

    if (config.temperature !== undefined) {
      claudeConfig.temperature = config.temperature;
    }

    if (config.thinking_level !== undefined) {
      claudeConfig.temperature = 1.0; // `temperature` may only be set to 1 when thinking is enabled
      claudeConfig.thinking = this._convertThinkingLevelToBudget(
        config.thinking_level,
      );
    }

    if (config.tools !== undefined) {
      const claudeTools = [];
      for (const tool of config.tools) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const claudeTool: any = {};
        for (const [key, value] of Object.entries(tool)) {
          claudeTool[key.replace("parameters", "input_schema")] = value;
        }
        claudeTools.push(claudeTool);
      }
      claudeConfig.tools = claudeTools;
    }

    if (config.tool_choice !== undefined) {
      claudeConfig.tool_choice = this._convertToolChoice(config.tool_choice);
    }

    return claudeConfig;
  }

  /**
   * Transform universal message format to Claude's MessageParam format.
   */
  transformUniMessageToModelInput(messages: UniMessage[]): BetaMessageParam[] {
    const claudeMessages: BetaMessageParam[] = [];

    for (const msg of messages) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const contentBlocks: any[] = [];
      for (const item of msg.content_items) {
        if (item.type === "text") {
          contentBlocks.push({ type: "text", text: item.text });
        } else if (item.type === "image_url") {
          const imageUrl = item.image_url;
          if (imageUrl.startsWith("data:")) {
            const match = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
            if (match) {
              const mediaType = match[1];
              const base64Data = match[2];
              contentBlocks.push({
                type: "image",
                source: {
                  type: "base64",
                  media_type: mediaType,
                  data: base64Data,
                },
              });
            } else {
              throw new Error(`Invalid base64 image: ${imageUrl}`);
            }
          } else {
            contentBlocks.push({
              type: "image",
              source: { type: "url", url: imageUrl },
            });
          }
        } else if (item.type === "thinking") {
          contentBlocks.push({
            type: "thinking",
            thinking: item.thinking,
            signature: item.signature,
          });
        } else if (item.type === "tool_call") {
          contentBlocks.push({
            type: "tool_use",
            id: item.tool_call_id,
            name: item.name,
            input: item.arguments,
          });
        } else if (item.type === "tool_result") {
          if (!item.tool_call_id) {
            throw new Error("tool_call_id is required for tool result.");
          }

          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const toolResult: any[] = [{ type: "text", text: item.text }];

          if (item.images) {
            for (const imageUrl of item.images) {
              if (imageUrl.startsWith("data:")) {
                const match = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
                if (match) {
                  const mediaType = match[1];
                  const base64Data = match[2];
                  toolResult.push({
                    type: "image",
                    source: {
                      type: "base64",
                      media_type: mediaType,
                      data: base64Data,
                    },
                  });
                } else {
                  throw new Error(`Invalid base64 image: ${imageUrl}`);
                }
              } else {
                toolResult.push({
                  type: "image",
                  source: { type: "url", url: imageUrl },
                });
              }
            }
          }

          contentBlocks.push({
            type: "tool_result",
            content: toolResult,
            tool_use_id: item.tool_call_id,
          });
        } else {
          throw new Error(`Unknown item: ${JSON.stringify(item)}`);
        }
      }

      claudeMessages.push({
        role: msg.role,
        content: contentBlocks,
      });
    }

    return claudeMessages;
  }

  /**
   * Transform Claude model output to universal event format.
   */
  transformModelOutputToUniEvent(
    modelOutput: BetaRawMessageStreamEvent,
  ): UniEvent {
    let eventType: EventType | null = null;
    const contentItems: PartialContentItem[] = [];
    let usageMetadata: UsageMetadata | null = null;
    let finishReason: FinishReason | null = null;

    const claudeEventType = modelOutput.type;
    if (claudeEventType === "content_block_start") {
      eventType = "start";
      const block = modelOutput.content_block;
      if (block.type === "tool_use") {
        contentItems.push({
          type: "partial_tool_call",
          name: block.name,
          arguments: "",
          tool_call_id: block.id,
        });
      }
    } else if (claudeEventType === "content_block_delta") {
      eventType = "delta";
      const delta = modelOutput.delta;
      if (delta.type === "thinking_delta") {
        contentItems.push({ type: "thinking", thinking: delta.thinking });
      } else if (delta.type === "text_delta") {
        contentItems.push({ type: "text", text: delta.text });
      } else if (delta.type === "input_json_delta") {
        contentItems.push({
          type: "partial_tool_call",
          name: "",
          arguments: delta.partial_json,
          tool_call_id: "",
        });
      } else if (delta.type === "signature_delta") {
        contentItems.push({
          type: "thinking",
          thinking: "",
          signature: delta.signature,
        });
      }
    } else if (claudeEventType === "content_block_stop") {
      eventType = "stop";
    } else if (claudeEventType === "message_start") {
      eventType = "start";
      const message = modelOutput.message;
      if (message.usage) {
        const cacheCreationTokens =
          message.usage.cache_creation_input_tokens || 0;
        usageMetadata = {
          cached_tokens: message.usage.cache_read_input_tokens,
          prompt_tokens: message.usage.input_tokens + cacheCreationTokens,
          thoughts_tokens: null,
          response_tokens: null,
        };
      }
    } else if (claudeEventType === "message_delta") {
      eventType = "stop";
      const delta = modelOutput.delta;
      if (delta.stop_reason) {
        const stopReasonMapping: { [key: string]: FinishReason } = {
          end_turn: "stop",
          max_tokens: "length",
          stop_sequence: "stop",
          tool_use: "stop",
        };
        finishReason = stopReasonMapping[delta.stop_reason] || "unknown";
      }

      const usage = modelOutput.usage;
      if (usage) {
        // In message_delta, we only update response_tokens
        usageMetadata = {
          cached_tokens: null,
          prompt_tokens: null,
          thoughts_tokens: null,
          response_tokens: usage.output_tokens,
        };
      }
    } else if (claudeEventType === "message_stop") {
      eventType = "stop";
    } else if (
      ["text", "thinking", "signature", "input_json"].includes(claudeEventType)
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
   * Stream generate using Claude SDK with unified conversion methods.
   */
  async *streamingResponse(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncGenerator<UniEvent> {
    const claudeConfig = this.transformUniConfigToModelConfig(options.config);
    const claudeMessages = this.transformUniMessageToModelInput(
      options.messages,
    );

    // Add cache_control to last user message's last item if enabled
    const promptCaching = options.config.prompt_caching || PromptCaching.ENABLE;
    if (promptCaching !== PromptCaching.DISABLE && claudeMessages.length > 0) {
      try {
        const reversedMessages = [...claudeMessages].reverse();
        const lastUserMessage = reversedMessages.find((x) => x.role === "user");
        if (lastUserMessage && Array.isArray(lastUserMessage.content)) {
          const lastContentItem =
            lastUserMessage.content[lastUserMessage.content.length - 1];
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (lastContentItem as any).cache_control = {
            type: "ephemeral",
            ttl: promptCaching === PromptCaching.ENHANCE ? "1h" : "5m",
          };
        }
      } catch {
        // Ignore errors in cache_control setup
      }
    }

    // Stream generate
    const partialToolCall: {
      name?: string;
      arguments?: string;
      tool_call_id?: string;
    } = {};
    const partialUsage: {
      prompt_tokens?: number | null;
      cached_tokens?: number | null;
    } = {};

    const stream = await this._client.beta.messages.stream({
      ...claudeConfig,
      messages: claudeMessages,
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

        if (uniEvent.usage_metadata !== null) {
          partialUsage.prompt_tokens = uniEvent.usage_metadata.prompt_tokens;
          partialUsage.cached_tokens = uniEvent.usage_metadata.cached_tokens;
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

        if (
          partialUsage.prompt_tokens !== undefined &&
          partialUsage.prompt_tokens !== null &&
          uniEvent.usage_metadata !== null
        ) {
          yield {
            role: "assistant",
            event_type: "stop",
            content_items: [],
            usage_metadata: {
              prompt_tokens: partialUsage.prompt_tokens,
              thoughts_tokens: null,
              response_tokens: uniEvent.usage_metadata.response_tokens,
              cached_tokens: partialUsage.cached_tokens || null,
            },
            finish_reason: uniEvent.finish_reason,
          };
          partialUsage.prompt_tokens = undefined;
          partialUsage.cached_tokens = undefined;
        }
      }
    }
  }
}
