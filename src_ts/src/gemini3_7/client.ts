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
  ImageConfig as GeminiImageConfig,
  MultiSpeakerVoiceConfig,
  Part,
  PrebuiltVoiceConfig,
  FunctionCall,
  ThinkingConfig,
  ThinkingLevel as GeminiThinkingLevel,
  FunctionCallingConfig,
  SpeakerVoiceConfig,
  SpeechConfig,
  Tool,
  ToolConfig,
  GenerateContentResponse,
  FinishReason as GeminiFinishReason,
  FunctionResponsePart,
  FunctionResponseBlob,
  FunctionResponse,
  VoiceConfig,
  EmbedContentConfig,
} from "@google/genai";
import * as path from "path";
import { LLMClient } from "../baseClient";
import { UnsupportedParameterError } from "../errors";
import {
  EventType,
  Fidelity,
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
 * Wrap a part's thought signature as a fidelity payload, or nothing when absent.
 */
function partFidelity(part: Part): { fidelity?: Fidelity } {
  if (part.thoughtSignature == null) {
    return {};
  }
  return { fidelity: { signature: part.thoughtSignature } };
}

/**
 * Read the thought signature recorded in an item's fidelity payload.
 */
function itemThoughtSignature(item: {
  fidelity?: Fidelity;
}): string | undefined {
  return item.fidelity?.signature;
}

/**
 * Unified client for the Gemini family, named for the newest generation it
 * serves (3.7). It serves every generateContent model generation (3.7 back
 * through 3.x text, image, TTS, and embedding models, with the 2.5 series
 * reachable via an explicit clientType), and applies the 3.6-generation
 * parameter contract to the whole family: temperature is rejected everywhere.
 *
 * Starting with the 3.6 generation the API deprecates the temperature/top_p/top_k
 * sampling parameters (silently ignored today, HTTP 400 in future
 * generations), so this client rejects them instead of sending a no-op.
 */
export class Gemini3_7Client extends LLMClient {
  protected _model: string;
  private _client: GoogleGenAI;

  /**
   * Initialize Gemini 3.7 client with model and API key.
   */
  constructor(options: {
    model: string;
    apiKey?: string;
    baseUrl?: string | null;
    clientType?: string | null;
  }) {
    super();
    this._model = options.model;
    const key = options.apiKey || process.env.GEMINI_API_KEY || undefined;
    const url = options.baseUrl || process.env.GEMINI_BASE_URL || undefined;
    const httpOptions = url ? { baseUrl: url } : undefined;
    if (key && key.startsWith("{")) {
      const credentials = JSON.parse(key);
      const googleAuthOptions = {
        credentials,
        scopes: ["https://www.googleapis.com/auth/cloud-platform"],
      };
      this._client = new GoogleGenAI({
        vertexai: true,
        location: "global",
        project: credentials.project_id,
        googleAuthOptions,
        httpOptions,
      });
    } else {
      this._client = new GoogleGenAI({
        apiKey: key,
        httpOptions,
      });
    }
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
   * Get image bytes and MIME type from URL.
   */
  private async _getImageBytesAndMimeType(
    url: string,
    signal?: AbortSignal,
  ): Promise<{ data: Buffer; mimeType: string }> {
    if (url.startsWith("data:")) {
      const match = url.match(/^data:([^;]+);base64,(.+)$/);
      if (match) {
        const mimeType = match[1];
        const base64Data = match[2];
        const data = Buffer.from(base64Data, "base64");
        return { data, mimeType };
      } else {
        throw new Error(`Invalid base64 image: ${url}`);
      }
    } else {
      const response = await fetch(url, { signal });
      if (!response.ok) {
        throw new Error(`Failed to fetch image: ${url}`);
      }
      const arrayBuffer = await response.arrayBuffer();
      const data = Buffer.from(arrayBuffer);
      const mimeType = this._detectImageMimeType(url);
      return { data, mimeType };
    }
  }

  // Gemini thinking levels from weakest to strongest, used to pick the
  // closest supported level when a model rejects the requested one.
  private static readonly GEMINI_LEVEL_ORDER: GeminiThinkingLevel[] = [
    GeminiThinkingLevel.MINIMAL,
    GeminiThinkingLevel.LOW,
    GeminiThinkingLevel.MEDIUM,
    GeminiThinkingLevel.HIGH,
  ];

  /**
   * Thinking levels the target model accepts (llmsdk_docs/gemini3_7/docs/thinking.md).
   *
   * An empty array means the model rejects the thinking_level parameter
   * entirely, so it must be omitted from the request.
   */
  private _supportedThinkingLevels(): GeminiThinkingLevel[] {
    if (this._model.includes("gemini-2.5")) {
      // The vendor table claims low/medium/high, but the live API rejects
      // every thinking_level value for the 2.5 series (verified 2026-07-24).
      return [];
    }
    if (this._model.includes("-image")) {
      return [GeminiThinkingLevel.MINIMAL, GeminiThinkingLevel.HIGH];
    }
    if (this._model.includes("gemini-3-pro")) {
      // The only pro generation without "medium".
      return [GeminiThinkingLevel.LOW, GeminiThinkingLevel.HIGH];
    }
    if (this._model.includes("-pro")) {
      // Every pro generation rejects "minimal"; matching broadly keeps
      // future pro models on the safe side (clamping a level the model
      // would have accepted costs a little accuracy, forwarding an
      // unsupported one is a 400).
      return [
        GeminiThinkingLevel.LOW,
        GeminiThinkingLevel.MEDIUM,
        GeminiThinkingLevel.HIGH,
      ];
    }
    if (this._model.includes("gemini-3.7")) {
      // The 3.7 generation rejects "minimal" with a 400 (verified live 2026-08-13).
      return [
        GeminiThinkingLevel.LOW,
        GeminiThinkingLevel.MEDIUM,
        GeminiThinkingLevel.HIGH,
      ];
    }
    return Gemini3_7Client.GEMINI_LEVEL_ORDER;
  }

  /**
   * Convert ThinkingLevel enum to the closest Gemini ThinkingLevel the model supports.
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
      [ThinkingLevel.XHIGH]: GeminiThinkingLevel.HIGH,
      // Gemini stops at "high", so both top levels land there before per-model clamping
      [ThinkingLevel.MAX]: GeminiThinkingLevel.HIGH,
    };
    const level = mapping[thinkingLevel];
    if (level === undefined) {
      return undefined;
    }
    const supported = this._supportedThinkingLevels();
    if (supported.length === 0) {
      // The model takes no thinking_level at all; drop the parameter and
      // let the model use its default instead of forwarding a 400.
      return undefined;
    }
    if (supported.includes(level)) {
      return level;
    }
    // Degrade silently to the nearest supported level; ties round up,
    // e.g. MEDIUM becomes HIGH on gemini-3-pro and NONE maps to LOW on
    // gemini-3.7-flash. `supported` is non-empty here, so the
    // initial-value-less reduce cannot throw.
    const order = Gemini3_7Client.GEMINI_LEVEL_ORDER;
    const index = order.indexOf(level);
    return supported.reduce((best, candidate) => {
      const bestDistance = Math.abs(order.indexOf(best) - index);
      const candidateDistance = Math.abs(order.indexOf(candidate) - index);
      if (candidateDistance !== bestDistance) {
        return candidateDistance < bestDistance ? candidate : best;
      }
      return order.indexOf(candidate) > order.indexOf(best) ? candidate : best;
    });
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

  private _withAbortSignal<T extends { abortSignal?: AbortSignal }>(
    config: T | undefined,
    signal?: AbortSignal,
  ): T | undefined {
    if (!signal) {
      return config;
    }
    return { ...(config ?? {}), abortSignal: signal } as T;
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
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "temperature",
        message:
          "Gemini models do not support setting temperature; the API deprecated " +
          "sampling parameters starting with the 3.6 generation.",
      });
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

    if (config.fast_mode) {
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "fast_mode",
        message: "Gemini does not support fast mode.",
      });
    }

    if (
      config.prompt_caching !== undefined &&
      config.prompt_caching !== PromptCaching.ENABLE
    ) {
      throw new UnsupportedParameterError({
        client: this.constructor.name,
        parameter: "prompt_caching",
        message: "prompt_caching must be ENABLE for Gemini.",
      });
    }

    if (config.image_config !== undefined) {
      configParams.imageConfig = {
        aspectRatio: config.image_config.aspect_ratio,
        imageSize: config.image_config.image_size,
      } as GeminiImageConfig;
    }

    const isTtsModel = this._model.toLowerCase().includes("tts");
    if (isTtsModel) {
      configParams.responseModalities = ["AUDIO"];
      const ttsConfig = config.tts_config ?? [{ voice: "Kore" }];
      if (![1, 2].includes(ttsConfig.length)) {
        throw new Error("tts_config must contain 1 or 2 entries.");
      }

      if (ttsConfig.length === 1) {
        configParams.speechConfig = {
          voiceConfig: {
            prebuiltVoiceConfig: {
              voiceName: ttsConfig[0].voice,
            } as PrebuiltVoiceConfig,
          } as VoiceConfig,
        } as SpeechConfig;
      } else {
        const speakerVoiceConfigs = ttsConfig.map((speakerConfig) => {
          if (!speakerConfig.speaker) {
            throw new Error(
              "speaker is required when tts_config has 2 entries.",
            );
          }

          return {
            speaker: speakerConfig.speaker,
            voiceConfig: {
              prebuiltVoiceConfig: {
                voiceName: speakerConfig.voice,
              } as PrebuiltVoiceConfig,
            } as VoiceConfig,
          } as SpeakerVoiceConfig;
        });

        configParams.speechConfig = {
          multiSpeakerVoiceConfig: {
            speakerVoiceConfigs,
          } as MultiSpeakerVoiceConfig,
        } as SpeechConfig;
      }
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
    signal?: AbortSignal,
  ): Promise<Content[]> {
    const mapping: { [key: string]: string } = {
      user: "user",
      assistant: "model",
    };

    const contents: Content[] = [];
    // The generateContent API wants both the call id and the function name on a function
    // response, but a universal tool_result carries only the id, so remember each call's name.
    const callNames = new Map<string, string>();
    for (const msg of messages) {
      const parts: Part[] = [];
      for (const item of msg.content_items) {
        if (item.type === "text") {
          parts.push({
            text: item.text,
            thoughtSignature: itemThoughtSignature(item),
          } as Part);
        } else if (item.type === "image_url") {
          const urlValue = item.image_url;
          const imageData = await this._getImageBytesAndMimeType(
            urlValue,
            signal,
          );
          parts.push({
            inlineData: {
              mimeType: imageData.mimeType,
              data: imageData.data.toString("base64"),
            },
          } as Part);
        } else if (item.type === "inline_data") {
          parts.push({
            inlineData: {
              mimeType: item.mime_type,
              data: item.data.toString("base64"),
            },
            thoughtSignature: itemThoughtSignature(item),
          } as Part);
        } else if (item.type === "thinking") {
          parts.push({
            text: item.thinking,
            thought: true,
            thoughtSignature: itemThoughtSignature(item),
          } as Part);
        } else if (item.type === "inline_thinking") {
          parts.push({
            inlineData: {
              mimeType: item.mime_type,
              data: item.data.toString("base64"),
            },
            thought: true,
            thoughtSignature: itemThoughtSignature(item),
          } as Part);
        } else if (item.type === "tool_call") {
          callNames.set(item.tool_call_id, item.name);
          // Histories from before ids were stored carry the name as the tool_call_id;
          // replay those without an id, exactly as they arrived.
          const functionCall: FunctionCall = {
            ...(item.tool_call_id !== item.name
              ? { id: item.tool_call_id }
              : {}),
            name: item.name,
            args: item.arguments,
          };
          parts.push({
            functionCall: functionCall,
            thoughtSignature: itemThoughtSignature(item),
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
              const imageData = await this._getImageBytesAndMimeType(
                imageUrl,
                signal,
              );
              multimodalParts.push({
                inlineData: {
                  mimeType: imageData.mimeType,
                  data: imageData.data.toString("base64"),
                } as FunctionResponseBlob,
              } as FunctionResponsePart);
            }
          }

          const functionName =
            callNames.get(item.tool_call_id) ?? item.tool_call_id;
          parts.push({
            functionResponse: {
              ...(item.tool_call_id !== functionName
                ? { id: item.tool_call_id }
                : {}),
              name: functionName,
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

    if (
      modelOutput.candidates?.length !== undefined &&
      modelOutput.candidates?.length > 0
    ) {
      const candidate = modelOutput.candidates?.[0];
      for (const part of candidate.content?.parts || []) {
        if (part.functionCall) {
          contentItems.push({
            type: "tool_call",
            name: part.functionCall.name || "",
            arguments: part.functionCall.args || {},
            tool_call_id: part.functionCall.id || part.functionCall.name || "",
            ...partFidelity(part),
          });
        } else if (part.thought) {
          if (part.text !== undefined) {
            contentItems.push({
              type: "thinking",
              thinking: part.text,
              ...partFidelity(part),
            });
          } else if (part.inlineData) {
            contentItems.push({
              type: "inline_thinking",
              data: Buffer.from(part.inlineData.data || "", "base64"),
              mime_type: part.inlineData.mimeType || "application/octet-stream",
              ...partFidelity(part),
            });
          }
        } else if (part.inlineData) {
          contentItems.push({
            type: "inline_data",
            data: Buffer.from(part.inlineData.data || "", "base64"),
            mime_type: part.inlineData.mimeType || "application/octet-stream",
            ...partFidelity(part),
          });
        } else if (part.text !== undefined) {
          contentItems.push({
            type: "text",
            text: part.text,
            ...partFidelity(part),
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

    if (
      contentItems.length === 0 &&
      usageMetadata === null &&
      finishReason === null
    ) {
      // nothing was read out of the chunk, so there is nothing to emit: a gateway
      // heartbeat looks like this, and so does any other chunk we take no value from
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

  private async *_embedMessagesInternal(options: {
    messages: UniMessage[];
    config: UniConfig;
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent> {
    // Embed transformed messages and return them as a streaming event.
    const contents = await this.transformUniMessageToModelInput(
      options.messages,
      options.signal,
    );

    const geminiConfig = this._withAbortSignal<EmbedContentConfig>(
      options.config.embedding_config?.dimensions != null
        ? {
            outputDimensionality: options.config.embedding_config.dimensions,
          }
        : undefined,
      options.signal,
    );

    const result = await this._client.models.embedContent({
      model: this._model,
      contents,
      config: geminiConfig,
    });

    yield {
      role: "assistant",
      event_type: "stop",
      content_items:
        result.embeddings?.map((embedding) => ({
          type: "embedding",
          embedding: embedding.values ?? [],
        })) ?? [],
      usage_metadata: {
        cached_tokens: null,
        prompt_tokens: result.metadata?.billableCharacterCount ?? null,
        thoughts_tokens: null,
        response_tokens: null,
      },
      finish_reason: "stop",
    };
  }

  /**
   * Stream generate using Gemini SDK with unified conversion methods.
   */
  async *_streamingResponseInternal(options: {
    messages: UniMessage[];
    config: UniConfig;
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent> {
    if (this._model.toLowerCase().includes("embedding")) {
      for await (const event of this._embedMessagesInternal(options)) {
        yield event;
      }
      return;
    }

    // check if all items are text for tts model
    const isTtsModel = this._model.toLowerCase().includes("tts");
    if (isTtsModel) {
      const invalidItem = options.messages
        .flatMap((message) => message.content_items)
        .find((item) => item.type !== "text");
      if (invalidItem) {
        throw new Error(
          `Gemini TTS only supports text input, got content item type=${JSON.stringify(invalidItem.type)}.`,
        );
      }
    }

    const geminiConfig = this._withAbortSignal<GenerateContentConfig>(
      this.transformUniConfigToModelConfig(options.config),
      options.signal,
    );
    const contents = await this.transformUniMessageToModelInput(
      options.messages,
      options.signal,
    );

    const responseStream = await this._client.models.generateContentStream({
      model: this._model,
      contents: contents,
      config: geminiConfig,
    });

    for await (const chunk of responseStream) {
      const event = this.transformModelOutputToUniEvent(chunk);
      if (event.event_type === "unused") {
        continue;
      }

      for (const item of event.content_items) {
        if (item.type === "tool_call") {
          // gemini 3.7 does not support partial tool call, mock a partial tool call event
          yield {
            role: "assistant",
            event_type: "delta",
            content_items: [
              {
                type: "partial_tool_call",
                name: item.name,
                arguments: JSON.stringify(item.arguments),
                tool_call_id: item.tool_call_id,
                fidelity: item.fidelity,
              },
            ],
            usage_metadata: null,
            finish_reason: null,
          };
        }
      }

      yield event;
    }
  }
}
