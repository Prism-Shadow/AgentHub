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
  ContentItem,
  FinishReason,
  UniConfig,
  UniEvent,
  UniMessage,
  UsageMetadata,
} from "./types";

/**
 * Abstract base class for LLM clients.
 *
 * All model-specific clients must inherit from this class and implement
 * the required abstract methods for complete SDK abstraction.
 */
export abstract class LLMClient {
  protected _history: UniMessage[];

  constructor() {
    this._history = [];
  }

  /**
   * Transform universal configuration to model-specific configuration.
   *
   * @param config - Universal configuration object
   * @returns Model-specific configuration object
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  abstract transformUniConfigToModelConfig(config: UniConfig): any;

  /**
   * Transform universal message format to model-specific input format.
   *
   * @param messages - List of universal message objects
   * @returns Model-specific input format (e.g., Gemini's Content list, OpenAI's messages array)
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  abstract transformUniMessageToModelInput(messages: UniMessage[]): any;

  /**
   * Transform model output to universal event format.
   *
   * @param modelOutput - Model-specific output object (streaming chunk)
   * @returns Universal event object
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  abstract transformModelOutputToUniEvent(modelOutput: any): UniEvent;

  /**
   * Concatenate a stream of universal events into a single universal message.
   *
   * This is a concrete method implemented in the base class that can be reused
   * by all model clients. It accumulates events and builds a complete message.
   *
   * @param events - List of universal events from streaming response
   * @returns Complete universal message object
   */
  concatUniEventsToUniMessage(events: UniEvent[]): UniMessage {
    const contentItems: ContentItem[] = [];
    let usageMetadata: UsageMetadata | null = null;
    let finishReason: FinishReason | null = null;

    for (const event of events) {
      for (const item of event.content_items) {
        if (item.type === "text") {
          const lastItem = contentItems[contentItems.length - 1];
          if (lastItem && lastItem.type === "text") {
            lastItem.text += item.text;
            if (item.signature) {
              lastItem.signature = item.signature;
            }
          } else if (item.text) {
            contentItems.push({ ...item });
          }
        } else if (item.type === "thinking") {
          const lastItem = contentItems[contentItems.length - 1];
          if (lastItem && lastItem.type === "thinking") {
            lastItem.thinking += item.thinking;
            if (item.signature) {
              lastItem.signature = item.signature;
            }
          } else if (item.thinking || item.signature) {
            contentItems.push({ ...item });
          }
        } else if (item.type === "partial_tool_call") {
          // Skip partial_tool_call items - they should already be converted to tool_call
        } else {
          contentItems.push({ ...item });
        }
      }

      usageMetadata = event.usage_metadata;
      finishReason = event.finish_reason;
    }

    return {
      role: "assistant",
      content_items: contentItems,
      usage_metadata: usageMetadata,
      finish_reason: finishReason,
    };
  }

  /**
   * Generate content in streaming mode (stateless).
   *
   * This method should use transformUniConfigToModelConfig and
   * transformUniMessageToModelInput to prepare the request, then
   * transformModelOutputToUniEvent to convert each chunk.
   *
   * @param messages - List of universal message objects containing conversation history
   * @param config - Universal configuration object
   * @yields Universal events from the streaming response
   */
  abstract streamingResponse(
    messages: UniMessage[],
    config: UniConfig
  ): AsyncGenerator<UniEvent>;

  /**
   * Generate content in streaming mode (stateful).
   *
   * This method should use transformUniConfigToModelConfig,
   * transformUniMessageToModelInput, transformModelOutputToUniEvent,
   * and concatUniEventsToUniMessage to manage the conversation flow.
   *
   * @param message - Latest universal message object to add to conversation
   * @param config - Universal configuration object
   * @yields Universal events from the streaming response
   */
  async *streamingResponseStateful(
    message: UniMessage,
    config: UniConfig
  ): AsyncGenerator<UniEvent> {
    this._history.push(message);

    const events: UniEvent[] = [];
    for await (const event of this.streamingResponse(this._history, config)) {
      events.push(event);
      yield event;
    }

    if (events.length > 0) {
      const assistantMessage = this.concatUniEventsToUniMessage(events);
      this._history.push(assistantMessage);
    }

    if (config.trace_id) {
      const { Tracer } = await import("./integration/tracer");
      const tracer = new Tracer();
      tracer.saveHistory(this._history, config.trace_id, config);
    }
  }

  /**
   * Clear the message history.
   */
  async clearHistory(): Promise<void> {
    this._history = [];
  }

  /**
   * Get the current message history.
   *
   * @returns Copy of the current message history
   */
  async getHistory(): Promise<UniMessage[]> {
    return [...this._history];
  }
}
