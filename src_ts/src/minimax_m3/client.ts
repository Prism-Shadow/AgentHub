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
  Response,
  ResponseCreateParamsStreaming,
  ResponseStreamEvent,
} from "openai/resources/responses/responses";
import { LLMClient } from "../baseClient";
import {
  AgentHubError,
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

const DEFAULT_BASE_URL = "https://api.minimax.io/v1";

type JsonValue = string | number | boolean | null | JsonValue[] | JsonObject;

interface JsonObject {
  [key: string]: JsonValue;
}

type MiniMaxReasoningEffort = "none" | "low" | "medium" | "high";

interface MiniMaxFunctionTool {
  type: "function";
  name: string;
  description: string;
  parameters?: Record<string, unknown>;
}

interface MiniMaxResponseConfig {
  model: string;
  store: false;
  instructions?: string;
  max_output_tokens?: number;
  temperature?: number;
  reasoning?: { effort: MiniMaxReasoningEffort };
  tools?: MiniMaxFunctionTool[];
  tool_choice?: "auto" | "none";
}

interface MiniMaxInputTextContent extends JsonObject {
  type: "input_text";
  text: string;
}

interface MiniMaxOutputTextContent extends JsonObject {
  type: "output_text";
  text: string;
}

interface MiniMaxInputImageContent extends JsonObject {
  type: "input_image";
  image_url: string;
}

type MiniMaxMessageContent =
  | MiniMaxInputTextContent
  | MiniMaxOutputTextContent
  | MiniMaxInputImageContent;

type MiniMaxToolOutputContent =
  | MiniMaxInputTextContent
  | MiniMaxInputImageContent;

interface MiniMaxMessageInput {
  type?: "message";
  role: UniMessage["role"];
  content: JsonObject[];
}

// `summary` and `content` are optional on a MiniMax reasoning item, so they ride along through the
// JsonObject index signature rather than being required here.
interface MiniMaxReasoningInput extends JsonObject {
  id: string;
  type: "reasoning";
}

interface MiniMaxFunctionCallInput extends JsonObject {
  type: "function_call";
  call_id: string;
  name: string;
  arguments: string;
}

interface MiniMaxFunctionCallOutputInput extends JsonObject {
  type: "function_call_output";
  call_id: string;
  output: string | MiniMaxToolOutputContent[];
}

type MiniMaxInputItem =
  | MiniMaxMessageInput
  | MiniMaxReasoningInput
  | MiniMaxFunctionCallInput
  | MiniMaxFunctionCallOutputInput;

interface MiniMaxResponseCreateParamsStreaming extends MiniMaxResponseConfig {
  input: MiniMaxInputItem[];
  stream: true;
}


function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isJsonValue(value: unknown): value is JsonValue {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "boolean"
  ) {
    return true;
  }
  if (typeof value === "number") {
    return Number.isFinite(value);
  }
  if (Array.isArray(value)) {
    return value.every(isJsonValue);
  }
  return isJsonObject(value);
}

function isJsonObject(value: unknown): value is JsonObject {
  if (!isRecord(value)) {
    return false;
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) {
    return false;
  }
  return Object.values(value).every(isJsonValue);
}

function requireJsonObject(value: unknown, context: string): JsonObject {
  if (!isJsonObject(value)) {
    throw new Error(`${context} must contain only JSON-style values.`);
  }
  return value;
}





// Reporting zeros for a request that consumed real tokens would corrupt cost accounting, so a
// response with no usage block returns null and leaves the base client's "last event must carry
// usage_metadata" validation in charge.
function transformUsage(response: Response): UsageMetadata | null {
  const usage = response.usage;
  if (!usage) {
    return null;
  }

  // MiniMax sends the detail objects as null when it has nothing to report and drops them
  // entirely on truncated responses, so every field needs its own default, parents included.
  const cachedTokens = usage.input_tokens_details?.cached_tokens ?? 0;
  const reasoningTokens = usage.output_tokens_details?.reasoning_tokens ?? 0;
  return {
    cached_tokens: cachedTokens,
    prompt_tokens: (usage.input_tokens ?? 0) - cachedTokens,
    thoughts_tokens: reasoningTokens,
    response_tokens: (usage.output_tokens ?? 0) - reasoningTokens,
  };
}

function transformFinishReason(
  response: Response,
  eventType: "response.completed" | "response.incomplete",
): FinishReason {
  // Trust the response status over the event name: a truncated answer reported through
  // response.completed must still finish as a truncation, not as a normal stop.
  if (eventType === "response.incomplete" || response.status === "incomplete") {
    const incompleteReason = response.incomplete_details?.reason;
    if (incompleteReason === "max_output_tokens") {
      return "length";
    }
    if (incompleteReason === "content_filter") {
      return "stop";
    }
    return "unknown";
  }
  if ((response.output ?? []).some((item) => item.type === "function_call")) {
    return "tool_call";
  }
  return "stop";
}

// MiniMax reports stream errors either flat on the event or nested under `error`, and the Python
// client accepts both, so read whichever shape arrived rather than emitting `undefined`.
function streamError(modelOutput: object): Error {
  const flat = modelOutput as Record<string, unknown>;
  const nested = flat.error;
  const source = isRecord(nested) ? nested : flat;
  const details: string[] = [];
  for (const field of ["code", "message", "param"] as const) {
    const value = source[field];
    if (value !== undefined && value !== null) {
      details.push(`${field}=${JSON.stringify(value)}`);
    }
  }
  if (details.length === 0) {
    details.push("no error details were provided");
  }
  return new AgentHubError(`MiniMax stream error: ${details.join(", ")}`);
}

function responseFailure(response: Response): Error {
  if (response.error) {
    return new AgentHubError(
      `MiniMax response ${response.id} failed (${response.error.code}): ${response.error.message}`,
    );
  }
  return new AgentHubError(
    `MiniMax response ${response.id} failed without error details from the API.`,
  );
}

/** MiniMax M3 client using MiniMax's Responses API. */
export class MiniMaxM3Client extends LLMClient {
  protected _model: string;
  private _client: OpenAI;

  constructor(options: {
    model: string;
    apiKey?: string;
    baseUrl?: string | null;
    clientType?: string | null;
  }) {
    super();
    this._model = options.model;
    // The wrapped OpenAI SDK falls back to OPENAI_API_KEY when handed undefined, which would send
    // an OpenAI credential to the MiniMax host, so resolve the key here and fail loudly instead.
    const apiKey = options.apiKey || process.env.MINIMAX_API_KEY;
    if (!apiKey) {
      throw new Error("MINIMAX_API_KEY is required for MiniMaxM3Client.");
    }
    this._client = new OpenAI({
      apiKey,
      baseURL:
        options.baseUrl || process.env.MINIMAX_BASE_URL || DEFAULT_BASE_URL,
    });
  }

  private _convertThinkingLevelToEffort(
    thinkingLevel: ThinkingLevel,
  ): MiniMaxReasoningEffort {
    const mapping: Record<ThinkingLevel, MiniMaxReasoningEffort> = {
      [ThinkingLevel.NONE]: "none",
      [ThinkingLevel.LOW]: "low",
      [ThinkingLevel.MEDIUM]: "medium",
      [ThinkingLevel.HIGH]: "high",
      [ThinkingLevel.XHIGH]: "high",
    };
    return mapping[thinkingLevel];
  }

  private _convertToolChoice(toolChoice: ToolChoice): "auto" | "none" {
    if (toolChoice === "auto" || toolChoice === "none") {
      return toolChoice;
    }
    throw new UnsupportedParameterError({
      client: this.constructor.name,
      parameter: "tool_choice",
      message: "MiniMax Responses API does not support required or named tool selection.",
    });
  }

  transformUniConfigToModelConfig(config: UniConfig): MiniMaxResponseConfig {
    const minimaxConfig: MiniMaxResponseConfig = {
      model: this._model,
      store: false,
    };

    if (config.system_prompt !== undefined) {
      minimaxConfig.instructions = config.system_prompt;
    }
    if (config.max_tokens !== undefined) {
      minimaxConfig.max_output_tokens = config.max_tokens;
    }
    if (config.temperature !== undefined) {
      // Written as a positive range test so NaN is rejected too: every `<`/`>` comparison against
      // NaN is false, which would let it through to the provider as a null temperature.
      if (!(config.temperature >= 0 && config.temperature <= 1)) {
        throw new UnsupportedParameterError({
          client: this.constructor.name,
          parameter: "temperature",
          message: "MiniMax Responses API does not support temperatures outside the range 0 to 1.",
        });
      }
      minimaxConfig.temperature = config.temperature;
    }
    if (config.thinking_level !== undefined) {
      minimaxConfig.reasoning = {
        effort: this._convertThinkingLevelToEffort(config.thinking_level),
      };
    }
    if (config.tools !== undefined) {
      minimaxConfig.tools = config.tools.map((tool) => ({
        type: "function",
        name: tool.name,
        description: tool.description,
        ...(tool.parameters !== undefined
          ? { parameters: tool.parameters }
          : {}),
      }));
    }
    if (config.tool_choice !== undefined) {
      minimaxConfig.tool_choice = this._convertToolChoice(config.tool_choice);
    }
    if (config.prompt_caching === PromptCaching.DISABLE) {
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "prompt_caching",
        message: "MiniMax Responses API does not support disabling its automatic prompt cache.",
      });
    }
    if (config.prompt_caching === PromptCaching.ENHANCE) {
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "prompt_caching",
        message: "MiniMax Responses API does not support enhancing its automatic prompt cache.",
      });
    }

    return minimaxConfig;
  }

  transformUniMessageToModelInput(messages: UniMessage[]): MiniMaxInputItem[] {
    const inputList: MiniMaxInputItem[] = [];

    for (const message of messages) {
      let contentItems: MiniMaxMessageContent[] = [];
      const flushContentItems = (): void => {
        if (contentItems.length > 0) {
          inputList.push({ role: message.role, content: contentItems });
          contentItems = [];
        }
      };

      for (const item of message.content_items) {
        if (item.type === "text") {
          if (message.role === "user") {
            contentItems.push({ type: "input_text", text: item.text });
          } else {
            contentItems.push({ type: "output_text", text: item.text });
          }
        } else if (item.type === "image_url") {
          contentItems.push({ type: "input_image", image_url: item.image_url });
        } else if (item.type === "thinking") {
          flushContentItems();
          // Rebuild the reasoning item from the universal thinking text plus the one field that
          // cannot be reconstructed, the provider's item id.
          const reasoningId = item.fidelity?.id;
          if (typeof reasoningId !== "string") {
            throw new Error(
              "MiniMax reasoning replay requires an id in its fidelity.",
            );
          }
          inputList.push({
            type: "reasoning",
            id: reasoningId,
            content: item.thinking
              ? [{ type: "reasoning_text", text: item.thinking }]
              : [],
          });
        } else if (item.type === "tool_call") {
          flushContentItems();
          inputList.push({
            type: "function_call",
            call_id: item.tool_call_id,
            name: item.name,
            arguments: JSON.stringify(item.arguments),
          });
        } else if (item.type === "tool_result") {
          flushContentItems();
          if (item.tool_call_id === undefined) {
            throw new Error("tool_call_id is required for tool result.");
          }
          let output: string | MiniMaxToolOutputContent[] = item.text;
          if (item.images?.length) {
            const outputItems: MiniMaxToolOutputContent[] = [
              { type: "input_text", text: item.text },
            ];
            for (const imageUrl of item.images) {
              outputItems.push({ type: "input_image", image_url: imageUrl });
            }
            output = outputItems;
          }
          inputList.push({
            type: "function_call_output",
            call_id: item.tool_call_id,
            output,
          });
        } else {
          throw new Error(`Unknown item: ${JSON.stringify(item)}`);
        }
      }

      flushContentItems();
    }

    return inputList;
  }

  transformModelOutputToUniEvent(modelOutput: ResponseStreamEvent): UniEvent {
    let eventType: EventType = "unused";
    const contentItems: PartialContentItem[] = [];
    let usageMetadata: UsageMetadata | null = null;
    let finishReason: FinishReason | null = null;

    switch (modelOutput.type) {
      case "response.output_text.delta":
        eventType = "delta";
        contentItems.push({
          type: "text",
          text: modelOutput.delta,
        });
        break;
      case "response.reasoning_text.delta":
        eventType = "delta";
        contentItems.push({
          type: "thinking",
          thinking: modelOutput.delta,
        });
        break;
      case "response.output_item.added":
        if (modelOutput.item.type === "function_call") {
          eventType = "start";
          contentItems.push({
            type: "partial_tool_call",
            name: modelOutput.item.name ?? "",
            arguments: "",
            tool_call_id: modelOutput.item.call_id ?? "",
            // The output-item id is optional on the wire; record null rather than aborting the
            // turn, matching the Python client.
            fidelity: { item_id: modelOutput.item.id ?? null },
          });
        }
        break;
      case "response.function_call_arguments.delta":
        eventType = "delta";
        contentItems.push({
          type: "partial_tool_call",
          name: "",
          arguments: modelOutput.delta,
          tool_call_id: "",
          fidelity: { item_id: modelOutput.item_id },
        });
        break;
      case "response.output_item.done": {
        // Validate the whole item now: a non-JSON value captured here would otherwise only fail on
        // the next request, where it can no longer be traced back to the response that produced it.
        requireJsonObject(modelOutput.item, "MiniMax output item");
        if (modelOutput.item.type === "reasoning") {
          eventType = "delta";
          contentItems.push({
            type: "thinking",
            thinking: "",
            fidelity: { id: modelOutput.item.id ?? null },
          });
        } else if (modelOutput.item.type === "function_call") {
          eventType = "delta";
          contentItems.push({
            type: "tool_call",
            name: modelOutput.item.name,
            arguments: parseToolCallArguments(
              modelOutput.item.arguments,
              this.constructor.name,
              modelOutput.item.name,
              modelOutput.item.call_id,
            ),
            tool_call_id: modelOutput.item.call_id,
          });
        }
        // A completed message needs no content item: its text already arrived as output_text
        // deltas, so emitting an empty one here would only add a standalone empty assistant
        // message when the item produced no deltas at all.
        break;
      }
      case "response.completed":
      case "response.incomplete":
        eventType = "stop";
        usageMetadata = transformUsage(modelOutput.response);
        finishReason = transformFinishReason(
          modelOutput.response,
          modelOutput.type,
        );
        break;
      case "response.failed":
        throw responseFailure(modelOutput.response);
      case "error":
        throw streamError(modelOutput);
      case "response.created":
      case "response.in_progress":
      case "response.output_text.done":
      case "response.reasoning_text.done":
      case "response.content_part.added":
      case "response.content_part.done":
      case "response.function_call_arguments.done":
        break;
      default:
        // `response.error` is absent from the SDK's event union but handled by the Python client,
        // so match it here rather than reporting a provider error as an unknown event.
        if ((modelOutput as { type?: string }).type === "response.error") {
          throw streamError(modelOutput);
        }
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

  async *_streamingResponseInternal(options: {
    messages: UniMessage[];
    config: UniConfig;
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent> {
    const minimaxConfig = this.transformUniConfigToModelConfig(options.config);
    const inputList = this.transformUniMessageToModelInput(options.messages);
    const params: MiniMaxResponseCreateParamsStreaming = {
      ...minimaxConfig,
      input: inputList,
      stream: true,
    };

    // MiniMax accepts output_text assistant inputs and function tools without
    // OpenAI's required strict field, so narrow the compatibility cast to this boundary.
    const stream = await this._client.responses.create(
      params as ResponseCreateParamsStreaming,
      { signal: options.signal },
    );
    for await (const modelEvent of stream) {
      const event = this.transformModelOutputToUniEvent(modelEvent);
      if (event.event_type !== "unused") {
        yield event;
      }
    }
  }
}
