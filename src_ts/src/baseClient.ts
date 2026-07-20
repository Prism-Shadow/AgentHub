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

import { EmptyResponseError } from "./errors";
import {
  Fidelity,
  FinishReason,
  ContentItem,
  UniConfig,
  UniEvent,
  UniMessage,
  UsageMetadata,
} from "./types";

/**
 * Whether a content item carries a non-empty fidelity payload.
 */
function hasFidelity(fidelity?: Fidelity): boolean {
  return fidelity != null && Object.keys(fidelity).length > 0;
}

/**
 * Compare two fidelity payloads by value. Fidelity dicts are built with a
 * stable key order by each client, so JSON serialization is a faithful
 * equality check.
 */
function fidelityEquals(a?: Fidelity, b?: Fidelity): boolean {
  return JSON.stringify(a ?? {}) === JSON.stringify(b ?? {});
}

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
  abstract transformUniMessageToModelInput(
    messages: UniMessage[],
    signal?: AbortSignal,
  ): any; // eslint-disable-line @typescript-eslint/no-explicit-any

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
          const itemFidelity = item.fidelity ?? {};
          // a delta announcing a different phase starts a new item; same-phase and
          // phaseless deltas merge until a signature finishes the item
          if (
            lastItem &&
            lastItem.type === "text" &&
            lastItem.fidelity?.signature == null && // not finished by a signature yet
            (itemFidelity.phase == null || // phaseless deltas continue the item
              itemFidelity.phase === lastItem.fidelity?.phase) // same phase merges
          ) {
            lastItem.text += item.text;
            if (hasFidelity(item.fidelity)) {
              // a signature finishes the current item
              lastItem.fidelity = { ...lastItem.fidelity, ...item.fidelity };
            }
          } else if (item.text || itemFidelity.phase != null) {
            // text or new phase starts an item
            contentItems.push({ ...item });
          }
        } else if (item.type === "thinking") {
          const lastItem = contentItems[contentItems.length - 1];
          // a new item starts only when the open item's fidelity is non-empty and
          // differs from the incoming delta's; everything else merges into it
          if (
            lastItem &&
            lastItem.type === "thinking" &&
            (!hasFidelity(lastItem.fidelity) || // not finished by fidelity yet
              // a run of equal fidelity is one item
              fidelityEquals(lastItem.fidelity, item.fidelity))
          ) {
            lastItem.thinking += item.thinking;
            if (hasFidelity(item.fidelity)) {
              // fidelity finishes the current item
              lastItem.fidelity = item.fidelity;
            }
          } else if (item.thinking || hasFidelity(item.fidelity)) {
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
    signal?: AbortSignal;
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
    signal?: AbortSignal;
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
    this._validateNonThinkingOutput(events);

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
    signal?: AbortSignal;
  }): AsyncGenerator<UniEvent> {
    const { message, config } = options;

    const tempMessages = [...this._history, message];

    const events: UniEvent[] = [];
    for await (const event of this.streamingResponse({
      messages: tempMessages,
      config,
      signal: options.signal,
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
   * Validate that the completed response carries content other than thinking.
   *
   * Replaying a thinking-only assistant message on the next turn fails with a 400
   * error, so the response is rejected as soon as the stream completes.
   *
   * @param events - All events yielded by streamingResponse
   * @throws EmptyResponseError if every content item in the response is thinking
   */
  protected _validateNonThinkingOutput(events: UniEvent[]): void {
    const thinkingOnly = events.every((event) =>
      event.content_items.every(
        (item) => item.type === "thinking" || item.type === "inline_thinking",
      ),
    );
    if (thinkingOnly) {
      const finishReason =
        events.length > 0 ? events[events.length - 1].finish_reason : null;
      throw new EmptyResponseError({
        client: this.constructor.name,
        finishReason,
      });
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
