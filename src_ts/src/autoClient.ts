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
   * @param model - Model identifier (determines which client to use)
   * @param apiKey - Optional API key
   */
  constructor(model: string, apiKey?: string) {
    super();
    this._client = this._createClientForModel(model, apiKey);
  }

  /**
   * Create the appropriate client for the given model.
   *
   * @param model - Model identifier
   * @param _apiKey - Optional API key
   * @returns Instance of the appropriate client
   */
  private _createClientForModel(
    model: string,
    _apiKey?: string
  ): LLMClient {
    const clientType = process.env.CLIENT_TYPE || model.toLowerCase();

    if (clientType.includes("gemini-3")) {
      throw new Error(
        "Gemini-3 client is not implemented in TypeScript yet. " +
          "Please implement it following the Python version."
      );
    } else if (clientType.includes("claude") && clientType.includes("4-5")) {
      throw new Error(
        "Claude 4-5 client is not implemented in TypeScript yet. " +
          "Please implement it following the Python version."
      );
    } else if (clientType.includes("gpt-5.2")) {
      throw new Error(
        "GPT-5.2 client is not implemented in TypeScript yet. " +
          "Please implement it following the Python version."
      );
    } else if (clientType.includes("glm-4.7")) {
      throw new Error(
        "GLM-4.7 client is not implemented in TypeScript yet. " +
          "Please implement it following the Python version."
      );
    } else if (clientType.includes("qwen3")) {
      console.warn("Warning: Qwen3 client is only compatible with vLLM Server.");
      throw new Error(
        "Qwen3 client is not implemented in TypeScript yet. " +
          "Please implement it following the Python version."
      );
    } else {
      throw new Error(
        `${clientType} is not supported. ` +
          "Supported models: gemini-3, claude-xxx-4-5, gpt-5.2, glm-4.7, qwen3-xxx."
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
  async *streamingResponse(
    messages: UniMessage[],
    config: UniConfig
  ): AsyncGenerator<UniEvent> {
    for await (const event of this._client.streamingResponse(
      messages,
      config
    )) {
      yield event;
    }
  }

  /**
   * Route to underlying client's streamingResponseStateful.
   */
  async *streamingResponseStateful(
    message: UniMessage,
    config: UniConfig
  ): AsyncGenerator<UniEvent> {
    for await (const event of this._client.streamingResponseStateful(
      message,
      config
    )) {
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
