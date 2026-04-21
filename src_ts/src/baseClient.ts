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
  FinishReason,
  ContentItem,
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
  protected _model: string;
  private _history: UniMessage[];

  constructor() {
    this._model = "";
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
    let createdAt: number | undefined = undefined;

    for (const event of events) {
      for (const item of event.content_items) {
        if (item.type === "text") {
          const lastItem = contentItems[contentItems.length - 1];
          if (
            lastItem &&
            lastItem.type === "text" &&
            lastItem.signature == null && // no signature yet
            item.phase == null // no new phase
          ) {
            lastItem.text += item.text;
            if (item.signature) {
              lastItem.signature = item.signature;
            }
          } else if (item.text || item.phase != null) {
            // text or new phase starts an item
            contentItems.push({ ...item });
          }
        } else if (item.type === "thinking") {
          const lastItem = contentItems[contentItems.length - 1];
          if (
            lastItem &&
            lastItem.type === "thinking" &&
            lastItem.signature == null
          ) {
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
      createdAt = event.created_at;
    }

    return {
      role: "assistant",
      content_items: contentItems,
      usage_metadata: usageMetadata,
      finish_reason: finishReason,
      created_at: createdAt,
    };
  }

  /**
   * Internal method to handle streaming response.
   *
   * This method should be implemented by each model client to handle
   * the actual streaming request and yield model-specific events.
   *
   * @param options - Object containing messages and config
   * @yields Model-specific events from the streaming response
   */
  abstract _streamingResponseInternal(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncGenerator<UniEvent>;

  /**
   * Generate content in streaming mode (stateless).
   *
   * This method should use transformUniConfigToModelConfig and
   * transformUniMessageToModelInput to prepare the request, then
   * transformModelOutputToUniEvent to convert each chunk.
   *
   * @param options - Object containing messages and config
   * @yields Universal events from the streaming response
   */
  async *streamingResponse(options: {
    messages: UniMessage[];
    config: UniConfig;
  }): AsyncGenerator<UniEvent> {
    const { messages, config } = options;

    // Stamp any messages that don't yet have a created_at timestamp
    for (const msg of messages) {
      if (msg.created_at == null) {
        msg.created_at = Date.now();
      }
    }

    let lastEvent: UniEvent | null = null;
    const events: UniEvent[] = [];
    for await (const event of this._streamingResponseInternal(options)) {
      event.created_at = Date.now();
      lastEvent = event;
      events.push(event);
      yield event;
    }
    LLMClient._validateLastEvent(lastEvent);

    // Save history to file if trace_id is specified
    if (config.trace_id && events.length > 0) {
      const { Tracer } = await import("./integration/tracer");
      const assistantMessage = this.concatUniEventsToUniMessage(events);
      const tracer = new Tracer();
      tracer.saveHistory(
        this._model,
        [...messages, assistantMessage],
        config.trace_id,
        config,
      );
    }
  }

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
  async *streamingResponseStateful(options: {
    message: UniMessage;
    config: UniConfig;
  }): AsyncGenerator<UniEvent> {
    const { message, config } = options;

    const tempMessages = [...this._history, message];

    const events: UniEvent[] = [];
    for await (const event of this.streamingResponse({
      messages: tempMessages,
      config,
    })) {
      events.push(event);
      yield event;
    }

    // tempMessages[-1] is the user message, now stamped with created_at by streamingResponse
    if (events.length > 0) {
      const assistantMessage = this.concatUniEventsToUniMessage(events);
      this._history.push(tempMessages[tempMessages.length - 1]);
      this._history.push(assistantMessage);
    }
  }

  /**
   * Validate that the last event has usage_metadata and finish_reason.
   *
   * This validation guards against servers that silently terminate streaming
   * output partway through without sending a proper final event.
   *
   * @param lastEvent - The last event yielded by streamingResponse
   * @throws Error if lastEvent is null or missing usage_metadata/finish_reason
   */
  protected static _validateLastEvent(lastEvent: UniEvent | null): void {
    if (lastEvent === null) {
      throw new Error("Streaming response yielded no events");
    }
    if (lastEvent.usage_metadata === null) {
      throw new Error(
        `Last event must carry usage_metadata, got: ${JSON.stringify(lastEvent)}`,
      );
    }
    if (lastEvent.finish_reason === null) {
      throw new Error(
        `Last event must carry finish_reason, got: ${JSON.stringify(lastEvent)}`,
      );
    }
  }

  /**
   * Clear the message history.
   */
  clearHistory(): void {
    this._history = [];
  }

  /**
   * Get the current message history.
   *
   * @returns Copy of the current message history
   */
  getHistory(): UniMessage[] {
    return [...this._history];
  }

  /**
   * Replace the message history with a copy of the provided history.
   *
   * @param history - List of universal message objects to set as the new history
   */
  setHistory(history: UniMessage[]): void {
    this._history = [...history];
  }
}
