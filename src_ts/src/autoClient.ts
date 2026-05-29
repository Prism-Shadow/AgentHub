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
import { Claude4_6Client } from "./claude4_6";
import { Claude4_7Client } from "./claude4_7";
import { GPT5_5Client } from "./gpt5_5";
import { GLM5_1Client } from "./glm5_1";
import { KimiK2_6Client } from "./kimi_k2_6";
import { OpenaiClient } from "./openai";
import { DeepSeekV4Client } from "./deepseek_v4";
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
    clientType = (
      clientType ||
      process.env.CLIENT_TYPE ||
      model.toLowerCase()
    ).toLowerCase();

    if (
      clientType.includes("gemini-3-") ||
      clientType.includes("gemini-3.1-") ||
      clientType.includes("gemini-3.5-") ||
      clientType.includes("gemini-embedding")
    ) {
      return new Gemini3Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("claude") && clientType.includes("4-7")) {
      return new Claude4_7Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("claude") && clientType.includes("4-6")) {
      return new Claude4_6Client({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("gpt-5.4") ||
      clientType.includes("gpt-5.5")
    ) {
      return new GPT5_5Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("glm-5") || clientType.includes("glm-5.1")) {
      return new GLM5_1Client({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("kimi-k2.5") ||
      clientType.includes("kimi-k2.6")
    ) {
      return new KimiK2_6Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("openai")) {
      return new OpenaiClient({ model, apiKey, baseUrl });
    } else if (clientType.includes("deepseek-v4-")) {
      return new DeepSeekV4Client({ model, apiKey, baseUrl });
    } else {
      throw new Error(
        `${clientType} is not supported. ` +
          "Supported client types: gemini-3, claude-4-7, claude-4-6, gpt-5.4, gpt-5.5, glm-5.1, kimi-k2.5, kimi-k2.6, openai, deepseek-v4.",
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
