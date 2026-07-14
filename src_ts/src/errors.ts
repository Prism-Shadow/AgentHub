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

export class ToolCallArgumentParseError extends Error {
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

/** Thrown when DeepSeek reaches the output token limit while still thinking. */
export class EmptyAssistantResponseError extends Error {
  constructor() {
    super(
      "DeepSeek reached the output token limit before producing content or tool calls.",
    );
    this.name = "EmptyAssistantResponseError";
  }
}

export function parseToolCallArguments(
  rawArguments: string | undefined,
  client: string,
  toolName: string,
  toolCallId: string,
): Record<string, unknown> {
  const raw = rawArguments || "{}";
  try {
    const parsed = JSON.parse(raw);
    if (
      parsed !== null &&
      typeof parsed === "object" &&
      !Array.isArray(parsed)
    ) {
      return parsed as Record<string, unknown>;
    }
    throw new Error("expected a JSON object");
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
}
