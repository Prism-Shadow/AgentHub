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

import { LLMClient } from "./baseClient";
import { Gemini3Client } from "./gemini3";
import { Gemini3_6Client } from "./gemini3_6";
import { Claude4_6Client } from "./claude4_6";
import { Claude5Client } from "./claude5";
import { GPT5_5Client } from "./gpt5_5";
import { GLM5_1Client } from "./glm5_1";
import { GLM5_2Client } from "./glm5_2";
import { KimiK2_6Client } from "./kimi_k2_6";
import { KimiK3Client } from "./kimi_k3";
import { OpenaiClient } from "./openai";
import { OpenaiEmbeddingClient } from "./openai_embedding";
import { DeepSeekV4Client } from "./deepseek_v4";
import { MiniMaxM3Client } from "./minimax_m3";
import { UniConfig, UniEvent, UniMessage } from "./types";

/**
 * Auto-routing LLM client that dispatches to appropriate model-specific client.
 *
 * This client is stateful - it knows the model name at initialization and maintains
 * conversation history for that specific model.
 */
export class AutoLLMClient extends LLMClient {
  private _client: LLMClient;

  /**
   * Initialize AutoLLMClient with a specific model.
   *
   * @param options - Configuration object with model, apiKey, baseUrl, and clientType
   */
  constructor(options: {
    model: string;
    apiKey?: string;
    baseUrl?: string | null;
    clientType?: string | null;
  }) {
    super();
    this._client = this._createClientForModel(
      options.model,
      options.apiKey,
      options.baseUrl,
      options.clientType,
    );
  }

  /**
   * Create the appropriate client for the given model.
   *
   * @param model - Model identifier
   * @param apiKey - API key to be passed to the client implementation (unused until clients are implemented)
   * @param baseUrl - Base URL to be passed to the client implementation (unused until clients are implemented)
   * @param clientType - Optional client type override
   * @returns Instance of the appropriate client
   * @throws Error when the requested client is not yet implemented
   */
  private _createClientForModel(
    model: string,
    apiKey?: string,
    baseUrl?: string | null,
    clientType?: string | null,
  ): LLMClient {
    const explicitClientType = clientType || process.env.CLIENT_TYPE;
    clientType = (explicitClientType || model).toLowerCase();

    if (
      clientType === "minimax-m3" ||
      (!explicitClientType &&
        (clientType === "minimax-m2.7" ||
          clientType === "minimax-m2.7-highspeed"))
    ) {
      return new MiniMaxM3Client({ model, apiKey, baseUrl });
    }
    // gemini-3.6 must be matched before the broader gemini-3 prefix below
    if (
      clientType.includes("gemini-3.6") ||
      clientType.includes("gemini-3.5-flash-lite")
    ) {
      // gemini-3.5-flash-lite shares the sampling-parameter deprecation with gemini-3.6
      return new Gemini3_6Client({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("gemini-3") ||
      clientType.includes("gemini-embedding")
    ) {
      return new Gemini3Client({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("claude") &&
      (clientType.includes("4-7") ||
        clientType.includes("4-8") ||
        clientType.includes("-5"))
    ) {
      return new Claude5Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("claude") && clientType.includes("4-6")) {
      return new Claude4_6Client({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("gpt-5.4") ||
      clientType.includes("gpt-5.5")
    ) {
      return new GPT5_5Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("glm-5.2")) {
      return new GLM5_2Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("glm-5") || clientType.includes("glm-5.1")) {
      return new GLM5_1Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("kimi-k3")) {
      return new KimiK3Client({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("kimi-k2.5") ||
      clientType.includes("kimi-k2.6")
    ) {
      return new KimiK2_6Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("deepseek-v4")) {
      return new DeepSeekV4Client({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("openai") &&
      clientType.includes("embedding")
    ) {
      return new OpenaiEmbeddingClient({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("openai") &&
      !clientType.includes("embedding")
    ) {
      return new OpenaiClient({ model, apiKey, baseUrl });
    } else {
      throw new Error(
        `${clientType} is not supported. ` +
          "Supported client types: minimax-m3, gemini-3.6, gemini-3, " +
          "claude-5, claude-4-8, claude-4-7, " +
          "claude-4-6, gpt-5.5, gpt-5.4, glm-5.2, glm-5.1, kimi-k3, kimi-k2.6, kimi-k2.5, " +
          "deepseek-v4, openai-embedding, openai.",
      );
    }
  }

  /**
   * Delegate to underlying client's transformUniConfigToModelConfig.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformUniConfigToModelConfig(config: UniConfig): any {
    return this._client.transformUniConfigToModelConfig(config);
  }

  /**
   * Delegate to underlying client's transformUniMessageToModelInput.
   */
  /* eslint-disable @typescript-eslint/no-explicit-any */
  transformUniMessageToModelInput(
    messages: UniMessage[],
    signal?: AbortSignal,
  ): any {
    return this._client.transformUniMessageToModelInput(messages, signal);
  }
  /* eslint-enable @typescript-eslint/no-explicit-any */

  /**
   * Delegate to underlying client's transformModelOutputToUniEvent.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformModelOutputToUniEvent(modelOutput: any): UniEvent {
    return this._client.transformModelOutputToUniEvent(modelOutput);
  }

  /**
   * Not implemented - use streamingResponse instead.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async *_streamingResponseInternal(_options: any): AsyncGenerator<UniEvent> {
    yield {
      role: "assistant",
      event_type: "delta",
      content_items: [],
      usage_metadata: null,
      finish_reason: null,
    };
    throw new Error("Please use streamingResponse instead.");
  }

  /**
   * Route to underlying client's streamingResponse.
   */
  async *streamingResponse(options: {
    messages: UniMessage[];
    config: UniConfig;
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent> {
    for await (const event of this._client.streamingResponse({
      messages: options.messages,
      config: options.config,
      signal: options.signal,
    })) {
      yield event;
    }
  }

  /**
   * Route to underlying client's streamingResponseStateful.
   */
  async *streamingResponseStateful(options: {
    message: UniMessage;
    config: UniConfig;
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent> {
    for await (const event of this._client.streamingResponseStateful({
      message: options.message,
      config: options.config,
      signal: options.signal,
    })) {
      yield event;
    }
  }

  /**
   * Clear history in the underlying client.
   */
  clearHistory(): void {
    this._client.clearHistory();
  }

  /**
   * Get history from the underlying client.
   */
  getHistory(): UniMessage[] {
    return this._client.getHistory();
  }

  /**
   * Set history in the underlying client.
   */
  setHistory(history: UniMessage[]): void {
    this._client.setHistory(history);
  }
}
