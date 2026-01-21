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

/**
 * Thinking level for model reasoning.
 */
export enum ThinkingLevel {
  NONE = "none",
  LOW = "low",
  MEDIUM = "medium",
  HIGH = "high",
}

/**
 * Prompt cache configuration for Claude models.
 */
export enum PromptCaching {
  ENABLE = "enable",
  DISABLE = "disable",
  ENHANCE = "enhance",
}

export type ToolChoice = ("auto" | "required" | "none") | string[];
export type Role = "user" | "assistant";
export type EventType = "start" | "delta" | "stop" | "unused";
export type FinishReason = "stop" | "length" | "unknown";

export interface TextContentItem {
  type: "text";
  text: string;
  signature?: string | Buffer;
}

export interface ImageContentItem {
  type: "image_url";
  image_url: string;
}

export interface ThinkingContentItem {
  type: "thinking";
  thinking: string;
  signature?: string | Buffer;
}

export interface ToolCallContentItem {
  type: "tool_call";
  name: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  arguments: Record<string, any>;
  tool_call_id: string;
  signature?: string | Buffer;
}

export interface PartialToolCallContentItem {
  type: "partial_tool_call";
  name: string;
  arguments: string;
  tool_call_id: string;
  signature?: string | Buffer;
}

export interface ToolResultContentItem {
  type: "tool_result";
  result: string;
  tool_call_id: string;
}

export type ContentItem =
  | TextContentItem
  | ImageContentItem
  | ThinkingContentItem
  | ToolCallContentItem
  | ToolResultContentItem;

export type PartialContentItem = ContentItem | PartialToolCallContentItem;

/**
 * Usage metadata for model response.
 */
export interface UsageMetadata {
  prompt_tokens: number | null;
  thoughts_tokens: number | null;
  response_tokens: number | null;
  cached_tokens: number | null;
}

/**
 * Universal message format for LLM communication.
 */
export interface UniMessage {
  role: Role;
  content_items: ContentItem[];
  usage_metadata?: UsageMetadata;
  finish_reason?: FinishReason;
}

/**
 * Universal event format for streaming responses.
 */
export interface UniEvent {
  role: Role;
  content_items: ContentItem[];
  usage_metadata?: UsageMetadata;
  finish_reason?: FinishReason;
}

/**
 * Partial universal event format for streaming responses.
 */
export interface PartialUniEvent {
  role: Role;
  event_type: EventType;
  content_items: PartialContentItem[];
  usage_metadata?: UsageMetadata;
  finish_reason?: FinishReason;
}

/**
 * Available tool schema.
 */
export interface ToolSchema {
  name: string;
  description: string;
  parameters?: string;
}

/**
 * Universal configuration format for LLM requests.
 */
export interface UniConfig {
  max_tokens?: number;
  temperature?: number;
  tools?: ToolSchema[];
  thinking_summary?: boolean;
  thinking_level?: ThinkingLevel;
  tool_choice?: ToolChoice;
  system_prompt?: string;
  prompt_caching?: PromptCaching;
  trace_id?: string;
}
