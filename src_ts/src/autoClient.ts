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
import { Gemini3_7Client } from "./gemini3_7";
import { Claude5Client } from "./claude5";
import { GPT5_6Client } from "./gpt5_6";
import { GLM5_3Client } from "./glm5_3";
import { KimiK3Client } from "./kimi_k3";
import { OpenaiChatClient } from "./openai_chat";
import { OpenaiResponsesClient } from "./openai_responses";
import { AntMessagesClient } from "./ant_messages";
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
    clientType = (clientType || process.env.CLIENT_TYPE || model).toLowerCase();

    // every Gemini generation shares the unified client ("gemini-3" also matches the
    // gemini-3.7/gemini-3.6/gemini-3.5-flash-lite client types)
    if (
      clientType.includes("gemini-3") ||
      clientType.includes("gemini-embedding")
    ) {
      return new Gemini3_7Client({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("claude") &&
      (clientType.includes("4-6") ||
        clientType.includes("4-7") ||
        clientType.includes("4-8") ||
        clientType.includes("-5"))
    ) {
      // the whole Claude 4.6+ series shares the unified client
      return new Claude5Client({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("gpt-5.4") ||
      clientType.includes("gpt-5.5") ||
      clientType.includes("gpt-5.6")
    ) {
      return new GPT5_6Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("glm-5")) {
      // the whole GLM series shares the unified client
      return new GLM5_3Client({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("kimi-k3") ||
      clientType.includes("kimi-k2.5") ||
      clientType.includes("kimi-k2.6")
    ) {
      // the whole Kimi K2.5+ series shares the unified client
      return new KimiK3Client({ model, apiKey, baseUrl });
    } else if (clientType === "minimax-m3") {
      return new MiniMaxM3Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("deepseek-v4")) {
      return new DeepSeekV4Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("ant-messages")) {
      return new AntMessagesClient({ model, apiKey, baseUrl });
    } else if (clientType.includes("openai-responses")) {
      return new OpenaiResponsesClient({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("openai") &&
      clientType.includes("embedding")
    ) {
      return new OpenaiEmbeddingClient({ model, apiKey, baseUrl });
    } else if (
      clientType.includes("openai") &&
      !clientType.includes("embedding")
    ) {
      // openai-chat, plus bare "openai" as alias
      return new OpenaiChatClient({ model, apiKey, baseUrl });
    } else {
      throw new Error(
        `${clientType} is not supported. ` +
          "Supported client types: minimax-m3, gemini-3.7, gemini-3.6, gemini-3, " +
          "claude-5, claude-4-8, claude-4-7, " +
          "claude-4-6, gpt-5.6, gpt-5.5, gpt-5.4, glm-5.3, glm-5.2, glm-5.1, kimi-k3, kimi-k2.6, kimi-k2.5, " +
          "deepseek-v4, openai-embedding, ant-messages, openai-responses, openai-chat.",
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

  /**
   * List the model ids the endpoint serves in the underlying client.
   */
  async listModels(): Promise<string[]> {
    return this._client.listModels();
  }
}
