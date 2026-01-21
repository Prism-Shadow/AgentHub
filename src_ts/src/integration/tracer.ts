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
 * Conversation tracer module for saving and viewing conversation history.
 *
 * This module provides functionality to save conversation history to local files
 * and serve them via a web interface for real-time monitoring.
 */

import * as fs from "fs";
import * as path from "path";
import { UniConfig, UniMessage } from "../types";

/**
 * Tracer for saving conversation history to local files.
 *
 * This class handles saving conversation history to files in a cache directory.
 */
export class Tracer {
  private cacheDir: string;

  /**
   * Initialize the tracer.
   *
   * @param cacheDir - Directory to store conversation history files
   */
  constructor(cacheDir?: string) {
    this.cacheDir = path.resolve(
      cacheDir || process.env.AGENTHUB_CACHE_DIR || "cache"
    );
    this._ensureDirectoryExists(this.cacheDir);
  }

  /**
   * Ensure directory exists, create if it doesn't.
   *
   * @param dirPath - Directory path to create
   */
  private _ensureDirectoryExists(dirPath: string): void {
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
  }

  /**
   * Recursively serialize objects for JSON, converting Buffer to base64.
   *
   * @param obj - Object to serialize
   * @returns JSON-serializable object
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private _serializeForJson(obj: any): any {
    if (Buffer.isBuffer(obj)) {
      return obj.toString("base64");
    } else if (obj && typeof obj === "object") {
      if (Array.isArray(obj)) {
        return obj.map((item) => this._serializeForJson(item));
      } else {
        const result: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(obj)) {
          result[key] = this._serializeForJson(value);
        }
        return result;
      }
    }
    return obj;
  }

  /**
   * Save conversation history to files.
   *
   * @param history - List of UniMessage objects representing the conversation
   * @param fileId - File identifier without extension (e.g., "agent1/00001")
   * @param config - The UniConfig used for this conversation
   */
  saveHistory(
    history: UniMessage[],
    fileId: string,
    config: UniConfig
  ): void {
    const filePathBase = path.join(this.cacheDir, fileId);
    const dirPath = path.dirname(filePathBase);
    this._ensureDirectoryExists(dirPath);

    const jsonPath = filePathBase + ".json";
    const jsonData = {
      history: this._serializeForJson(history),
      config: this._serializeForJson(config),
      timestamp: new Date().toISOString(),
    };
    fs.writeFileSync(jsonPath, JSON.stringify(jsonData, null, 2), "utf-8");

    const txtPath = filePathBase + ".txt";
    const formattedContent = this._formatHistory(history, config);
    fs.writeFileSync(txtPath, formattedContent, "utf-8");
  }

  /**
   * Format conversation history in a readable text format.
   *
   * @param history - List of UniMessage objects
   * @param config - The UniConfig used for this conversation
   * @returns Formatted string representation of the conversation
   */
  private _formatHistory(history: UniMessage[], config: UniConfig): string {
    const lines: string[] = [];
    lines.push("=".repeat(80));
    lines.push(
      `Conversation History - ${new Date().toLocaleString()}`
    );
    lines.push("=".repeat(80));
    lines.push("");

    lines.push("Configuration:");
    for (const [key, value] of Object.entries(config)) {
      if (key !== "trace_id") {
        lines.push(`  ${key}: ${JSON.stringify(value)}`);
      }
    }
    lines.push("");

    for (let i = 0; i < history.length; i++) {
      const message = history[i];
      const role = message.role.toUpperCase();
      lines.push(`[${i + 1}] ${role}:`);
      lines.push("-".repeat(80));

      for (const item of message.content_items) {
        if (item.type === "text") {
          lines.push(`Text: ${item.text}`);
        } else if (item.type === "thinking") {
          lines.push(`Thinking: ${item.thinking}`);
        } else if (item.type === "image_url") {
          lines.push(`Image URL: ${item.image_url}`);
        } else if (item.type === "tool_call") {
          lines.push(`Tool Call: ${item.name}`);
          lines.push(
            `  Arguments: ${JSON.stringify(item.arguments, null, 2)}`
          );
          lines.push(`  Tool Call ID: ${item.tool_call_id}`);
        } else if (item.type === "tool_result") {
          lines.push(
            `Tool Result (ID: ${item.tool_call_id}): ${item.result}`
          );
        }
      }

      if (message.usage_metadata) {
        const metadata = message.usage_metadata;
        lines.push("\nUsage Metadata:");
        if (metadata.prompt_tokens !== null) {
          lines.push(`  Prompt Tokens: ${metadata.prompt_tokens}`);
        }
        if (metadata.thoughts_tokens !== null) {
          lines.push(`  Thoughts Tokens: ${metadata.thoughts_tokens}`);
        }
        if (metadata.response_tokens !== null) {
          lines.push(`  Response Tokens: ${metadata.response_tokens}`);
        }
        if (metadata.cached_tokens !== null) {
          lines.push(`  Cached Tokens: ${metadata.cached_tokens}`);
        }
      }

      if (message.finish_reason) {
        lines.push(`\nFinish Reason: ${message.finish_reason}`);
      }

      lines.push("");
    }

    return lines.join("\n");
  }
}
