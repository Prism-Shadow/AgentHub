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

function previewToolCallArguments(raw: string): string {
  const maxLength = 160;
  if (raw.length <= maxLength) {
    return raw;
  }
  const edgeLength = 72;
  return `${raw.slice(0, edgeLength)}...[truncated]...${raw.slice(-edgeLength)}`;
}

function describeJsonValue(value: unknown): string {
  if (value === null) {
    return "null";
  }
  if (Array.isArray(value)) {
    return "an array";
  }
  if (typeof value === "string") {
    return "a string";
  }
  if (typeof value === "number") {
    return "a number";
  }
  if (typeof value === "boolean") {
    return "a boolean";
  }
  return typeof value;
}

export class AgentHubError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "AgentHubError";
  }
}

/**
 * Raised when a completed response carries no non-thinking content and no tool calls.
 *
 * Models occasionally finish a turn with thinking output only (reasoning models in
 * particular); replaying such an assistant message on the next turn fails with a 400
 * error, so the response is rejected as soon as the stream completes.
 */
export class EmptyResponseError extends AgentHubError {
  readonly client: string;
  readonly finishReason: string | null;

  constructor(args: { client: string; finishReason: string | null }) {
    super(
      `Response from ${args.client} finished (finish_reason=${JSON.stringify(args.finishReason)}) ` +
        `with no non-thinking content and no tool calls; sending it back on the next turn would ` +
        `fail with a 400 error.`,
    );
    this.name = "EmptyResponseError";
    this.client = args.client;
    this.finishReason = args.finishReason;
  }
}

export class ToolCallArgumentParseError extends AgentHubError {
  readonly client: string;
  readonly toolName: string;
  readonly toolCallId: string;
  readonly rawArgumentsLength: number;
  readonly rawArgumentsPreview: string;

  constructor(args: {
    client: string;
    toolName: string;
    toolCallId: string;
    rawArguments: string;
    reason: string;
  }) {
    const preview = previewToolCallArguments(args.rawArguments);
    super(
      `Invalid streamed tool call arguments from ${args.client} for tool "${args.toolName}" ` +
        `(tool_call_id="${args.toolCallId}", length=${args.rawArguments.length}, ` +
        `preview=${JSON.stringify(preview)}): ${args.reason}`,
    );
    this.name = "ToolCallArgumentParseError";
    this.client = args.client;
    this.toolName = args.toolName;
    this.toolCallId = args.toolCallId;
    this.rawArgumentsLength = args.rawArguments.length;
    this.rawArgumentsPreview = preview;
  }
}

export function parseToolCallArguments(
  rawArguments: string | undefined,
  client: string,
  toolName: string,
  toolCallId: string,
): Record<string, unknown> {
  const raw = rawArguments || "{}";
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    throw new ToolCallArgumentParseError({
      client,
      toolName,
      toolCallId,
      rawArguments: raw,
      reason,
    });
  }

  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new ToolCallArgumentParseError({
      client,
      toolName,
      toolCallId,
      rawArguments: raw,
      reason: `expected a JSON object, got ${describeJsonValue(parsed)}`,
    });
  }

  return parsed as Record<string, unknown>;
}
