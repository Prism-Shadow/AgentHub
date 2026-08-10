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

import { isDeepStrictEqual } from "node:util";
import OpenAI from "openai";
import type {
  Response,
  ResponseCreateParamsStreaming,
  ResponseStreamEvent,
} from "openai/resources/responses/responses";
import { LLMClient } from "../baseClient";
import { parseToolCallArguments, UnsupportedParameterError } from "../errors";
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

interface MiniMaxReasoningInput extends JsonObject {
  id: string;
  type: "reasoning";
  summary: JsonValue[];
  content: JsonValue[];
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

const WIRE_ITEM_FIDELITY_KEY = "wire_item";

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

function isJsonArray(value: JsonValue | undefined): value is JsonValue[] {
  return Array.isArray(value);
}

function wireItemFromFidelity(
  fidelity: unknown,
  context: string,
): JsonObject | undefined {
  if (!isRecord(fidelity) || !(WIRE_ITEM_FIDELITY_KEY in fidelity)) {
    return undefined;
  }
  return requireJsonObject(
    fidelity[WIRE_ITEM_FIDELITY_KEY],
    `${context} wire item`,
  );
}

function legacyWireItemFromFidelity(fidelity: JsonObject): JsonObject {
  const wireItem: JsonObject = {};
  for (const [key, value] of Object.entries(fidelity)) {
    if (key !== WIRE_ITEM_FIDELITY_KEY && key !== "phase") {
      wireItem[key] = value;
    }
  }
  return wireItem;
}

function wireContentText(
  wireItem: JsonObject,
  contentType: string,
): string | undefined {
  const content = wireItem.content;
  if (!Array.isArray(content)) {
    return undefined;
  }

  const textParts: string[] = [];
  for (const part of content) {
    if (!isJsonObject(part) || part.type !== contentType) {
      continue;
    }
    if (typeof part.text !== "string") {
      return undefined;
    }
    textParts.push(part.text);
  }
  return textParts.length > 0 ? textParts.join("") : undefined;
}

function wireMessageContent(wireItem: JsonObject): JsonObject[] {
  const content = wireItem.content;
  if (!Array.isArray(content) || !content.every(isJsonObject)) {
    throw new Error(
      "MiniMax message wire fidelity must contain JSON object content parts.",
    );
  }
  return content;
}

function transformUsage(response: Response): UsageMetadata | null {
  const usage = response.usage;
  if (!usage) {
    return null;
  }

  const cachedTokens = usage.input_tokens_details.cached_tokens ?? 0;
  const reasoningTokens = usage.output_tokens_details.reasoning_tokens ?? 0;
  return {
    cached_tokens: cachedTokens,
    prompt_tokens: usage.input_tokens - cachedTokens,
    thoughts_tokens: reasoningTokens,
    response_tokens: usage.output_tokens - reasoningTokens,
  };
}

function transformFinishReason(
  response: Response,
  eventType: "response.completed" | "response.incomplete",
): FinishReason {
  if (eventType === "response.incomplete") {
    const incompleteReason = response.incomplete_details?.reason;
    if (incompleteReason === "max_output_tokens") {
      return "length";
    }
    if (incompleteReason === "content_filter") {
      return "stop";
    }
    return "unknown";
  }
  if (response.output.some((item) => item.type === "function_call")) {
    return "tool_call";
  }
  return "stop";
}

function responseFailure(response: Response): Error {
  if (response.error) {
    return new Error(
      `MiniMax response ${response.id} failed (${response.error.code}): ${response.error.message}`,
    );
  }
  return new Error(
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
    if (options.model.toLowerCase() !== "minimax-m3") {
      throw new Error(`${options.model} is not supported by MiniMaxM3Client.`);
    }
    this._model = options.model;
    this._client = new OpenAI({
      apiKey: options.apiKey || process.env.MINIMAX_API_KEY || undefined,
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
      if (config.temperature < 0 || config.temperature > 1) {
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
          const wireItem = wireItemFromFidelity(
            item.fidelity,
            "MiniMax message fidelity",
          );
          if (wireItem === undefined) {
            if (message.role === "user") {
              contentItems.push({ type: "input_text", text: item.text });
            } else {
              contentItems.push({ type: "output_text", text: item.text });
            }
          } else {
            if (message.role !== "assistant" || wireItem.type !== "message") {
              throw new Error(
                "MiniMax message wire fidelity requires an assistant message item.",
              );
            }
            flushContentItems();
            const wireText = wireContentText(wireItem, "output_text");
            const content =
              wireText === item.text && wireItem.role === message.role
                ? wireMessageContent(wireItem)
                : [{ type: "output_text", text: item.text }];
            inputList.push({
              ...wireItem,
              type: "message",
              role: message.role,
              content,
            });
          }
        } else if (item.type === "image_url") {
          contentItems.push({ type: "input_image", image_url: item.image_url });
        } else if (item.type === "thinking") {
          flushContentItems();
          const fidelity = requireJsonObject(
            item.fidelity ?? {},
            "MiniMax reasoning fidelity",
          );
          const wireItem =
            wireItemFromFidelity(fidelity, "MiniMax reasoning fidelity") ??
            legacyWireItemFromFidelity(fidelity);
          const id = wireItem.id;
          const summary = wireItem.summary;
          const wireContent = wireItem.content;
          if (
            typeof id !== "string" ||
            !isJsonArray(summary) ||
            !isJsonArray(wireContent)
          ) {
            throw new Error(
              "MiniMax reasoning replay requires valid id, summary, and content fidelity.",
            );
          }
          const content =
            wireContentText(wireItem, "reasoning_text") === item.thinking
              ? wireContent
              : [{ type: "reasoning_text", text: item.thinking }];
          inputList.push({
            ...wireItem,
            id,
            type: "reasoning",
            summary,
            content,
          });
        } else if (item.type === "tool_call") {
          flushContentItems();
          let serializedArguments = JSON.stringify(item.arguments);
          if (serializedArguments === undefined) {
            throw new Error(
              `MiniMax tool call ${item.name} arguments could not be serialized.`,
            );
          }

          const fidelity = requireJsonObject(
            item.fidelity ?? {},
            "MiniMax function-call fidelity",
          );
          const wireItem =
            wireItemFromFidelity(fidelity, "MiniMax function-call fidelity") ??
            legacyWireItemFromFidelity(fidelity);
          const rawArguments = wireItem.arguments;
          if (
            typeof rawArguments === "string" &&
            isDeepStrictEqual(
              parseToolCallArguments(
                rawArguments,
                this.constructor.name,
                item.name,
                item.tool_call_id,
              ),
              item.arguments,
            )
          ) {
            serializedArguments = rawArguments;
          }
          inputList.push({
            ...wireItem,
            type: "function_call",
            call_id: item.tool_call_id,
            name: item.name,
            arguments: serializedArguments,
          });
        } else if (item.type === "tool_result") {
          flushContentItems();
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
          fidelity: { phase: modelOutput.item_id },
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
          if (modelOutput.item.id === undefined) {
            throw new Error(
              "MiniMax function-call start event is missing its output item id.",
            );
          }
          eventType = "start";
          contentItems.push({
            type: "partial_tool_call",
            name: modelOutput.item.name,
            arguments: "",
            tool_call_id: modelOutput.item.call_id,
            fidelity: {
              item_id: modelOutput.item.id,
              output_index: modelOutput.output_index,
            },
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
          fidelity: {
            item_id: modelOutput.item_id,
            output_index: modelOutput.output_index,
          },
        });
        break;
      case "response.output_item.done": {
        const wireItem = requireJsonObject(
          modelOutput.item,
          "MiniMax output item",
        );
        if (modelOutput.item.type === "reasoning") {
          eventType = "delta";
          contentItems.push({
            type: "thinking",
            thinking: "",
            fidelity: { [WIRE_ITEM_FIDELITY_KEY]: wireItem },
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
            fidelity: { [WIRE_ITEM_FIDELITY_KEY]: wireItem },
          });
        } else if (modelOutput.item.type === "message") {
          eventType = "delta";
          const phase =
            typeof wireItem.id === "string"
              ? wireItem.id
              : `output-${modelOutput.output_index}`;
          contentItems.push({
            type: "text",
            text: "",
            fidelity: {
              phase,
              [WIRE_ITEM_FIDELITY_KEY]: wireItem,
            },
          });
        }
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
      case "error": {
        const code = modelOutput.code ? ` (${modelOutput.code})` : "";
        const parameter = modelOutput.param
          ? ` for parameter ${modelOutput.param}`
          : "";
        throw new Error(
          `MiniMax stream error${code}${parameter}: ${modelOutput.message}`,
        );
      }
      case "response.created":
      case "response.in_progress":
      case "response.output_text.done":
      case "response.reasoning_text.done":
      case "response.content_part.added":
      case "response.content_part.done":
      case "response.function_call_arguments.done":
        break;
      default:
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
