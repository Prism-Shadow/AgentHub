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
import { Claude4_5Client } from "./claude4_5";
import { GPT5_2Client } from "./gpt5_2";
import { GLM4_7Client } from "./glm4_7";
import { Qwen3Client } from "./qwen3";
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
      options.clientType
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
    clientType?: string | null
  ): LLMClient {
    clientType = clientType || process.env.CLIENT_TYPE || model.toLowerCase();

    if (clientType.includes("gemini-3")) {
      return new Gemini3Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("claude") && clientType.includes("4-5")) {
      return new Claude4_5Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("gpt-5.1") || clientType.includes("gpt-5.2")) {
      return new GPT5_2Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("glm-4.7")) {
      return new GLM4_7Client({ model, apiKey, baseUrl });
    } else if (clientType.includes("qwen3")) {
      return new Qwen3Client({ model, apiKey, baseUrl });
    } else {
      throw new Error(
        `${clientType} is not supported. ` +
          "Supported client types: gemini-3, claude-4-5, gpt-5.2, glm-4.7, qwen3."
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformUniMessageToModelInput(messages: UniMessage[]): any {
    return this._client.transformUniMessageToModelInput(messages);
  }

  /**
   * Delegate to underlying client's transformModelOutputToUniEvent.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transformModelOutputToUniEvent(modelOutput: any): UniEvent {
    return this._client.transformModelOutputToUniEvent(modelOutput);
  }

  /**
   * Route to underlying client's streamingResponse.
   */
  async *streamingResponse(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncGenerator<UniEvent> {
    for await (const event of this._client.streamingResponse({
      messages: options.messages,
      config: options.config
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
  }): AsyncGenerator<UniEvent> {
    for await (const event of this._client.streamingResponseStateful({
      message: options.message,
      config: options.config,
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
}
