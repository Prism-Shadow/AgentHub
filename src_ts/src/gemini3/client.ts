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

import {
  GoogleGenAI,
  Content,
  GenerateContentConfig,
  Part,
  FunctionCall,
  ThinkingConfig,
  ThinkingLevel as GeminiThinkingLevel,
  FunctionCallingConfig,
  Tool,
  ToolConfig,
  GenerateContentResponse,
  FinishReason as GeminiFinishReason,
  FunctionResponsePart,
  FunctionResponseBlob,
  FunctionResponse,
} from "@google/genai";
import * as path from "path";
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
 * Gemini 3-specific LLM client implementation.
 */
export class Gemini3Client extends LLMClient {
  protected _model: string;
  private _client: GoogleGenAI;

  /**
   * Initialize Gemini 3 client with model and API key.
   */
  constructor(options: {
    model: string;
    apiKey?: string;
    baseUrl?: string | null;
    clientType?: string | null;
  }) {
    super();
    this._model = options.model;
    const key =
      options.apiKey ||
      process.env.GEMINI_API_KEY ||
      process.env.GOOGLE_API_KEY ||
      undefined;
    const url =
      options.baseUrl || process.env.GOOGLE_GEMINI_BASE_URL || undefined;

    const httpOptions = url ? { baseUrl: url } : undefined;
    this._client = new GoogleGenAI({
      apiKey: key,
      httpOptions: httpOptions,
    });
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
   * Convert ThinkingLevel enum to Gemini's ThinkingLevel.
   */
  private _convertThinkingLevel(
    thinkingLevel: ThinkingLevel | undefined,
  ): GeminiThinkingLevel | undefined {
    if (!thinkingLevel) return undefined;

    const mapping: { [key: string]: GeminiThinkingLevel } = {
      [ThinkingLevel.NONE]: GeminiThinkingLevel.MINIMAL,
      [ThinkingLevel.LOW]: GeminiThinkingLevel.LOW,
      [ThinkingLevel.MEDIUM]: GeminiThinkingLevel.MEDIUM,
      [ThinkingLevel.HIGH]: GeminiThinkingLevel.HIGH,
    };
    return mapping[thinkingLevel];
  }

  /**
   * Convert ToolChoice to Gemini's tool config.
   */
  private _convertToolChoice(
    toolChoice: ToolChoice,
  ): FunctionCallingConfig | undefined {
    if (Array.isArray(toolChoice)) {
      return {
        mode: "ANY",
        allowedFunctionNames: toolChoice,
      } as FunctionCallingConfig;
    } else if (toolChoice === "none") {
      return { mode: "NONE" } as FunctionCallingConfig;
    } else if (toolChoice === "auto") {
      return { mode: "AUTO" } as FunctionCallingConfig;
    } else if (toolChoice === "required") {
      return { mode: "ANY" } as FunctionCallingConfig;
    }
    return undefined;
  }

  /**
   * Transform universal configuration to Gemini-specific configuration.
   */
  transformUniConfigToModelConfig(
    config: UniConfig,
  ): GenerateContentConfig | undefined {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const configParams: any = {};

    if (config.system_prompt !== undefined) {
      configParams.systemInstruction = config.system_prompt;
    }

    if (config.max_tokens !== undefined) {
      configParams.maxOutputTokens = config.max_tokens;
    }

    if (config.temperature !== undefined) {
      configParams.temperature = config.temperature;
    }

    const thinkingSummary = config.thinking_summary;
    const thinkingLevel = config.thinking_level;
    if (thinkingSummary !== undefined || thinkingLevel !== undefined) {
      configParams.thinkingConfig = {
        includeThoughts: thinkingSummary,
        thinkingLevel: this._convertThinkingLevel(thinkingLevel),
      } as ThinkingConfig;
    }

    if (config.tools !== undefined) {
      configParams.tools = [{ functionDeclarations: config.tools } as Tool];
      const toolChoice = config.tool_choice;
      if (toolChoice !== undefined) {
        const toolConfig = this._convertToolChoice(toolChoice);
        if (toolConfig) {
          configParams.toolConfig = {
            functionCallingConfig: toolConfig,
          } as ToolConfig;
        }
      }
    }

    if (
      config.prompt_caching !== undefined &&
      config.prompt_caching !== PromptCaching.ENABLE
    ) {
      throw new Error("prompt_caching must be ENABLE for Gemini 3.");
    }

    return Object.keys(configParams).length > 0
      ? (configParams as GenerateContentConfig)
      : undefined;
  }

  /**
   * Transform universal message format to Gemini's Content format.
   */
  async transformUniMessageToModelInput(
    messages: UniMessage[],
  ): Promise<Content[]> {
    const mapping: { [key: string]: string } = {
      user: "user",
      assistant: "model",
    };

    const contents: Content[] = [];
    for (const msg of messages) {
      const parts: Part[] = [];
      for (const item of msg.content_items) {
        if (item.type === "text") {
          parts.push({
            text: item.text,
            thoughtSignature: item.signature as string | undefined,
          } as Part);
        } else if (item.type === "image_url") {
          const urlValue = item.image_url;
          if (urlValue.startsWith("data:")) {
            const match = urlValue.match(/^data:([^;]+);base64,(.+)$/);
            if (match) {
              const mimeType = match[1];
              const base64Data = match[2];
              parts.push({
                inlineData: {
                  mimeType: mimeType,
                  data: base64Data,
                },
              } as Part);
            } else {
              throw new Error(`Invalid base64 image: ${urlValue}`);
            }
          } else {
            const mimeType = this._detectImageMimeType(urlValue);
            parts.push({
              fileData: {
                fileUri: urlValue,
                mimeType: mimeType,
              },
            } as Part);
          }
        } else if (item.type === "thinking") {
          parts.push({
            text: item.thinking,
            thought: true,
            thoughtSignature: item.signature as string | undefined,
          } as Part);
        } else if (item.type === "tool_call") {
          const functionCall: FunctionCall = {
            name: item.name,
            args: item.arguments,
          };
          parts.push({
            functionCall: functionCall,
            thoughtSignature: item.signature as string | undefined,
          } as Part);
        } else if (item.type === "tool_result") {
          if (!item.tool_call_id) {
            throw new Error("tool_call_id is required for tool result.");
          }

          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const toolResult: Record<string, any> = { result: item.text };
          const multimodalParts: FunctionResponsePart[] = [];

          if (item.images) {
            for (const imageUrl of item.images) {
              let imageBytes: Uint8Array;
              let mimeType: string;
              if (imageUrl.startsWith("data:")) {
                const match = imageUrl.match(/^data:([^;]+);base64,(.+)$/);
                if (match) {
                  mimeType = match[1];
                  const base64Data = match[2];
                  imageBytes = Uint8Array.from(
                    Buffer.from(base64Data, "base64"),
                  );
                } else {
                  throw new Error(`Invalid base64 image: ${imageUrl}`);
                }
              } else {
                const response = await fetch(imageUrl);
                if (!response.ok) {
                  throw new Error(`Failed to fetch image: ${imageUrl}`);
                }
                const arrayBuffer = await response.arrayBuffer();
                imageBytes = new Uint8Array(arrayBuffer);
                mimeType = this._detectImageMimeType(imageUrl);
              }
              const base64Data = Buffer.from(imageBytes).toString("base64");
              multimodalParts.push({
                inlineData: {
                  mimeType: mimeType,
                  data: base64Data,
                } as FunctionResponseBlob,
              } as FunctionResponsePart);
            }
          }

          parts.push({
            functionResponse: {
              name: item.tool_call_id,
              response: toolResult,
              parts: multimodalParts.length > 0 ? multimodalParts : undefined,
            } as FunctionResponse,
          } as Part);
        } else {
          throw new Error(`Unknown item: ${JSON.stringify(item)}`);
        }
      }

      contents.push({
        role: mapping[msg.role],
        parts: parts,
      } as Content);
    }

    return contents;
  }

  /**
   * Transform Gemini model output to universal event format.
   */
  transformModelOutputToUniEvent(
    modelOutput: GenerateContentResponse,
  ): UniEvent {
    let eventType: EventType = "delta";
    const contentItems: PartialContentItem[] = [];
    let usageMetadata: UsageMetadata | null = null;
    let finishReason: FinishReason | null = null;

    const candidate = modelOutput.candidates?.[0];
    if (!candidate) {
      throw new Error("No candidate in response");
    }

    for (const part of candidate.content?.parts || []) {
      if (part.functionCall) {
        contentItems.push({
          type: "tool_call",
          name: part.functionCall.name || "",
          arguments: part.functionCall.args || {},
          tool_call_id: part.functionCall.name || "",
          signature: part.thoughtSignature as string | undefined,
        });
      } else if (part.text !== undefined && part.thought) {
        contentItems.push({
          type: "thinking",
          thinking: part.text,
          signature: part.thoughtSignature as string | undefined,
        });
      } else if (part.text !== undefined) {
        contentItems.push({
          type: "text",
          text: part.text,
          signature: part.thoughtSignature as string | undefined,
        });
      } else {
        throw new Error(`Unknown output: ${JSON.stringify(part)}`);
      }
    }

    if (candidate.finishReason) {
      eventType = "stop";
      const stopReasonMapping: { [key: string]: FinishReason } = {
        [GeminiFinishReason.STOP]: "stop",
        [GeminiFinishReason.MAX_TOKENS]: "length",
      };
      finishReason = stopReasonMapping[candidate.finishReason] || "unknown";
    }

    if (modelOutput.usageMetadata) {
      eventType = eventType || "delta"; // deal with separate usage data

      const promptTokens = modelOutput.usageMetadata.promptTokenCount || 0;
      const cachedTokens =
        modelOutput.usageMetadata.cachedContentTokenCount || 0;
      usageMetadata = {
        cached_tokens:
          modelOutput.usageMetadata.cachedContentTokenCount || null,
        prompt_tokens: promptTokens - cachedTokens,
        thoughts_tokens: modelOutput.usageMetadata.thoughtsTokenCount || null,
        response_tokens: modelOutput.usageMetadata.candidatesTokenCount || null,
      };
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
   * Stream generate using Gemini SDK with unified conversion methods.
   */
  async *streamingResponse(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncGenerator<UniEvent> {
    const geminiConfig = this.transformUniConfigToModelConfig(options.config);
    const contents = await this.transformUniMessageToModelInput(
      options.messages,
    );

    const responseStream = await this._client.models.generateContentStream({
      model: this._model,
      contents: contents,
      config: geminiConfig,
    });

    let lastEvent: UniEvent | null = null;
    for await (const chunk of responseStream) {
      const event = this.transformModelOutputToUniEvent(chunk);
      for (const item of event.content_items) {
        if (item.type === "tool_call") {
          lastEvent = {
            role: "assistant",
            event_type: "delta",
            content_items: [
              {
                type: "partial_tool_call",
                name: item.name,
                arguments: JSON.stringify(item.arguments),
                tool_call_id: item.tool_call_id,
                signature: item.signature,
              },
            ],
            usage_metadata: null,
            finish_reason: null,
          };
          yield lastEvent;
        }
      }

      lastEvent = event;
      yield event;
    }
    LLMClient._validateLastEvent(lastEvent);
  }
}
