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
import express, { Express, Request, Response } from "express";
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
      cacheDir || process.env.AGENTHUB_CACHE_DIR || "cache",
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
    model: string,
    history: UniMessage[],
    fileId: string,
    config: UniConfig,
  ): void {
    const filePathBase = path.join(this.cacheDir, fileId);
    const dirPath = path.dirname(filePathBase);
    this._ensureDirectoryExists(dirPath);

    const configWithModel: UniConfig & { model: string } = {
      ...(config as UniConfig),
      model,
    };

    const jsonPath = filePathBase + ".json";
    const jsonData = {
      history: this._serializeForJson(history),
      config: this._serializeForJson(configWithModel),
      timestamp: new Date().toISOString(),
    };
    fs.writeFileSync(jsonPath, JSON.stringify(jsonData, null, 2), "utf-8");

    const txtPath = filePathBase + ".txt";
    const formattedContent = this._formatHistory(history, configWithModel);
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
    lines.push(`Conversation History - ${new Date().toLocaleString()}`);
    lines.push("=".repeat(80));
    lines.push("");

    lines.push("Configuration:");
    for (const [key, value] of Object.entries(config)) {
      if (key !== "trace_id") {
        if (key === "tools" && Array.isArray(value)) {
          lines.push(`  ${key}:`);
          lines.push(`    ${JSON.stringify(value, null, 2)}`);
        } else {
          lines.push(`  ${key}: ${value}`);
        }
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
          lines.push(`  Arguments: ${JSON.stringify(item.arguments, null, 2)}`);
          lines.push(`  Tool Call ID: ${item.tool_call_id}`);
        } else if (item.type === "tool_result") {
          lines.push(`Tool Result (ID: ${item.tool_call_id}): ${item.text}`);
          if (item.images && item.images.length > 0) {
            item.images.forEach((imageUrl, i) => {
              lines.push(`  Image ${i + 1}: ${imageUrl}`);
            });
          }
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

  /**
   * Create an Express web application for browsing conversation files.
   *
   * @returns Express application instance
   */
  createWebApp(): Express {
    const app = express();

    const DIRECTORY_TEMPLATE = (breadcrumb: string, itemsHtml: string) => `
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tracer</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50 min-h-screen">
        <div class="max-w-5xl mx-auto p-6">
            <h1 class="text-3xl font-bold text-gray-900 mb-6">Tracer</h1>
            <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
                <p class="text-sm text-gray-600"><strong>Path:</strong> ${breadcrumb}</p>
            </div>
            <div class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
                ${itemsHtml}
            </div>
        </div>
    </body>
    </html>
    `;

    const JSON_VIEWER_TEMPLATE = (
      filename: string,
      breadcrumb: string,
      backUrl: string,
      configHtml: string,
      historyHtml: string,
      numMessages: number,
    ) => `
    <!DOCTYPE html>
    <html>
    <head>
        <title>${filename}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50 min-h-screen">
        <div class="max-w-5xl mx-auto p-6">
            <h1 class="text-3xl font-bold text-gray-900 mb-4">${filename}</h1>
            <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
                <p class="text-sm text-gray-600"><strong>Path:</strong> ${breadcrumb}</p>
            </div>
            <a href="${backUrl}" class="inline-block mb-6 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-md border border-gray-300 text-sm transition-colors">
                ← Back to Directory
            </a>
            ${configHtml}
            ${historyHtml}
        </div>
        <script>
            function toggleMessage(idx) {
                const content = document.getElementById('content-' + idx);
                const icon = document.getElementById('icon-' + idx);
                content.classList.toggle('hidden');
                icon.classList.toggle('rotate-90');
            }
            // Expand all messages by default
            const numMessages = ${numMessages};
            for (let i = 0; i < numMessages; i++) {
                toggleMessage(i);
            }
        </script>
    </body>
    </html>
    `;

    const TEXT_VIEWER_TEMPLATE = (
      filename: string,
      breadcrumb: string,
      backUrl: string,
      content: string,
    ) => `
    <!DOCTYPE html>
    <html>
    <head>
        <title>${filename}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50 min-h-screen">
        <div class="max-w-5xl mx-auto p-6">
            <h1 class="text-3xl font-bold text-gray-900 mb-4">${filename}</h1>
            <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
                <p class="text-sm text-gray-600"><strong>Path:</strong> ${breadcrumb}</p>
            </div>
            <a href="${backUrl}" class="inline-block mb-6 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-md border border-gray-300 text-sm transition-colors">
                ← Back to Directory
            </a>
            <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6 overflow-x-auto">
                <pre class="font-mono text-sm whitespace-pre-wrap text-gray-800">${content}</pre>
            </div>
        </div>
    </body>
    </html>
    `;

    app.get("*", (req: Request, res: Response) => {
      const subpath = req.path.slice(1);
      const fullPath = path.resolve(path.join(this.cacheDir, subpath));

      if (!fullPath.startsWith(path.resolve(this.cacheDir))) {
        return res.status(403).send("Access denied");
      }

      if (!fs.existsSync(fullPath)) {
        return res.status(404).send("Path not found");
      }

      if (fs.statSync(fullPath).isFile()) {
        try {
          const parts = subpath ? subpath.split("/") : [];
          const breadcrumbParts = [
            '<a href="/" class="text-blue-600 hover:underline">cache</a>',
          ];

          for (let i = 0; i < parts.length - 1; i++) {
            const pathToPart = parts.slice(0, i + 1).join("/");
            breadcrumbParts.push(
              `<a href="/${pathToPart}" class="text-blue-600 hover:underline">${parts[i]}</a>`,
            );
          }

          if (parts.length > 0) {
            breadcrumbParts.push(`<strong>${parts[parts.length - 1]}</strong>`);
          }

          const breadcrumb = breadcrumbParts.join(" / ");
          const backUrl =
            parts.length > 1 ? "/" + parts.slice(0, -1).join("/") : "/";

          if (fullPath.endsWith(".json")) {
            const data = JSON.parse(fs.readFileSync(fullPath, "utf-8"));
            const configItems = Object.entries(data.config || {})
              .filter(([key]) => key !== "trace_id")
              .map(([key, value]) => {
                if (key === "system_prompt" && value != null) {
                  return {
                    key,
                    value: value,
                    isSystemPrompt: true,
                  };
                } else if (key === "tools" && Array.isArray(value)) {
                  return {
                    key,
                    value: JSON.stringify(value, null, 2),
                    isTools: true,
                  };
                } else {
                  return {
                    key,
                    value:
                      typeof value === "object"
                        ? JSON.stringify(value, null, 2)
                        : String(value),
                  };
                }
              });

            const configHtml =
              configItems.length > 0
                ? `<div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
                  <h2 class="text-xl font-semibold text-gray-900 mb-4">Configuration</h2>
                  ${configItems
                    .map((item) => {
                      if (item.isSystemPrompt) {
                        return `<div class="py-2 text-sm"><strong class="text-gray-900">${item.key}:</strong><pre style="margin: 4px 0 0 0; padding: 8px; background-color: #f6f8fa; border-radius: 4px; font-size: 12px; overflow-x: auto; white-space: pre-wrap;">${this._escapeHtml(String(item.value))}</pre></div>`;
                      } else if (item.isTools) {
                        return `<div class="py-2 text-sm"><strong class="text-gray-900">${item.key}:</strong><pre style="margin: 4px 0 0 0; padding: 8px; background-color: #f6f8fa; border-radius: 4px; font-size: 12px; overflow-x: auto;">${this._escapeHtml(String(item.value))}</pre></div>`;
                      } else {
                        return `<div class="py-2 text-sm"><strong class="text-gray-900">${item.key}:</strong><span class="text-gray-600">${this._escapeHtml(String(item.value))}</span></div>`;
                      }
                    })
                    .join("")}
                </div>`
                : "";

            const historyHtml = (data.history || [])
              .map((msg: UniMessage, idx: number) => {
                const contentItemsHtml = msg.content_items
                  .map((item) => {
                    let itemHtml = `<div class="mb-4 pb-4 border-b border-gray-100 last:border-b-0 last:mb-0 last:pb-0"><div class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">${item.type}</div>`;
                    if (item.type === "text") {
                      itemHtml += `<div class="bg-gray-50 p-4 rounded-md font-mono text-sm whitespace-pre-wrap text-gray-800">${this._escapeHtml(item.text)}</div>`;
                    } else if (item.type === "thinking") {
                      itemHtml += `<div class="bg-blue-50 p-4 rounded-md border-l-4 border-blue-500 font-mono text-sm whitespace-pre-wrap text-gray-800">${this._escapeHtml(item.thinking)}</div>`;
                    } else if (item.type === "tool_call") {
                      const entries = Object.entries(item.arguments);
                      let args = "";
                      for (let i = 0; i < entries.length; i++) {
                        const [key, value] = entries[i];
                        args += `${key}="${this._escapeHtml(String(value))}"`;
                        if (i !== entries.length - 1) {
                          args += ", ";
                        }
                      }
                      itemHtml += `<div class="bg-yellow-50 p-4 rounded-md border-l-4 border-yellow-500"><div class="font-mono text-sm text-gray-800">${item.name}(${args})</div></div>`;
                    } else if (item.type === "tool_result") {
                      let resultHtml = `<div class="bg-green-50 p-4 rounded-md border-l-4 border-green-500"><strong class="text-sm text-gray-900">Result:</strong> <span class="text-sm text-gray-700">${this._escapeHtml(item.text)}</span><br><strong class="text-sm text-gray-900">Call ID:</strong> <span class="text-sm text-gray-700">${item.tool_call_id}</span>`;
                      if (item.images && item.images.length > 0) {
                        resultHtml += `<div class="mt-2 flex flex-wrap gap-2">`;
                        for (const imageUrl of item.images) {
                          resultHtml += `<img src="${this._escapeHtml(imageUrl)}" class="max-w-xs max-h-48 rounded-md" alt="Tool Result Image">`;
                        }
                        resultHtml += `</div>`;
                      }
                      resultHtml += `</div>`;
                      itemHtml += resultHtml;
                    } else if (item.type === "image_url") {
                      itemHtml += `<div class="bg-gray-50 p-4 rounded-md"><img src="${this._escapeHtml(item.image_url)}" class="max-w-xs max-h-48 rounded-md" alt="Preview"></div>`;
                    }
                    itemHtml += "</div>";
                    return itemHtml;
                  })
                  .join("");

                let metadataHtml = "";
                if (msg.usage_metadata || msg.finish_reason) {
                  metadataHtml +=
                    '<div class="mt-4 pt-4 border-t border-gray-200 text-right text-xs text-gray-500">';
                  if (msg.usage_metadata) {
                    const parts = [];
                    if (msg.usage_metadata.prompt_tokens !== null)
                      parts.push(
                        `Prompt: ${msg.usage_metadata.prompt_tokens} tokens`,
                      );
                    if (msg.usage_metadata.thoughts_tokens !== null)
                      parts.push(
                        `Thoughts: ${msg.usage_metadata.thoughts_tokens} tokens`,
                      );
                    if (msg.usage_metadata.response_tokens !== null)
                      parts.push(
                        `Response: ${msg.usage_metadata.response_tokens} tokens`,
                      );
                    if (msg.usage_metadata.cached_tokens !== null)
                      parts.push(
                        `Cached: ${msg.usage_metadata.cached_tokens} tokens`,
                      );
                    metadataHtml += parts.join(" • ");
                  }
                  if (msg.finish_reason) {
                    if (msg.usage_metadata) metadataHtml += " • ";
                    metadataHtml += `Finish: ${msg.finish_reason}`;
                  }
                  metadataHtml += "</div>";
                }

                const roleClass =
                  msg.role === "user" ? "text-blue-600" : "text-green-600";
                return `
                  <div class="bg-white rounded-lg shadow-sm border border-gray-200 mb-4 overflow-hidden">
                    <div class="bg-gray-50 border-b border-gray-200 p-4 cursor-pointer hover:bg-gray-100 transition-colors" onclick="toggleMessage(${idx})">
                      <div class="flex justify-between items-center">
                        <div>
                          <span class="font-semibold text-sm uppercase ${roleClass}">${msg.role}</span>
                          <span class="text-xs text-gray-500 ml-2">• ${msg.content_items.length} item(s)</span>
                        </div>
                        <span class="text-gray-400 transform transition-transform" id="icon-${idx}">▶</span>
                      </div>
                    </div>
                    <div class="p-6 hidden" id="content-${idx}">
                      ${contentItemsHtml}
                      ${metadataHtml}
                    </div>
                  </div>
                `;
              })
              .join("");

            const html = JSON_VIEWER_TEMPLATE(
              path.basename(fullPath),
              breadcrumb,
              backUrl,
              configHtml,
              historyHtml,
              data.history?.length || 0,
            );

            return res.send(html);
          } else {
            const content = fs.readFileSync(fullPath, "utf-8");
            const html = TEXT_VIEWER_TEMPLATE(
              path.basename(fullPath),
              breadcrumb,
              backUrl,
              this._escapeHtml(content),
            );

            return res.send(html);
          }
        } catch (error) {
          return res.status(500).send(`Error reading file: ${error}`);
        }
      }

      try {
        const items = fs
          .readdirSync(fullPath)
          .sort((a: string, b: string) => {
            const aIsDir = fs.statSync(path.join(fullPath, a)).isDirectory();
            const bIsDir = fs.statSync(path.join(fullPath, b)).isDirectory();
            if (aIsDir !== bIsDir) return aIsDir ? -1 : 1;
            return a.localeCompare(b);
          })
          .map((entry: string) => {
            const entryPath = path.join(fullPath, entry);
            const relativePath = path.relative(this.cacheDir, entryPath);
            const isDir = fs.statSync(entryPath).isDirectory();
            let size = "";
            if (!isDir) {
              const stat = fs.statSync(entryPath);
              if (stat.size < 1024) {
                size = `${stat.size} B`;
              } else if (stat.size < 1024 * 1024) {
                size = `${(stat.size / 1024).toFixed(1)} KB`;
              } else {
                size = `${(stat.size / (1024 * 1024)).toFixed(1)} MB`;
              }
            }
            return {
              name: entry,
              is_dir: isDir,
              url: "/" + relativePath.replace(/\\/g, "/"),
              size,
            };
          });

        const parts = subpath ? subpath.split("/") : [];
        const breadcrumbParts = [
          '<a href="/" class="text-blue-600 hover:underline">cache</a>',
        ];

        for (let i = 0; i < parts.length; i++) {
          if (parts[i]) {
            const pathToPart = parts.slice(0, i + 1).join("/");
            breadcrumbParts.push(
              `<a href="/${pathToPart}" class="text-blue-600 hover:underline">${parts[i]}</a>`,
            );
          }
        }

        const breadcrumb = breadcrumbParts.join(" / ");

        const itemsHtml = items.length
          ? items
              .map(
                (item: {
                  is_dir: boolean;
                  url: string;
                  name: string;
                  size: string;
                }) => {
                  const icon = item.is_dir ? "📁" : "📄";
                  const sizeHtml = item.size
                    ? `<span class="text-xs text-gray-500">${item.size}</span>`
                    : "";
                  return `
                  <div class="border-b border-gray-200 last:border-b-0 hover:bg-gray-50 transition-colors">
                    <a href="${item.url}" class="flex items-center justify-between p-4 text-blue-600 hover:text-blue-800">
                      <span class="flex items-center">
                        <span class="mr-2">${icon}</span>
                        <span class="text-sm">${item.name}</span>
                      </span>
                      ${sizeHtml}
                    </a>
                  </div>
                `;
                },
              )
              .join("")
          : '<div class="p-8 text-center text-gray-500 italic">No files or directories found.</div>';

        const html = DIRECTORY_TEMPLATE(breadcrumb, itemsHtml);

        return res.send(html);
      } catch (error) {
        return res.status(500).send(`Error listing directory: ${error}`);
      }
    });

    return app;
  }

  /**
   * Escape HTML special characters.
   *
   * @param text - Text to escape
   * @returns Escaped text
   */
  private _escapeHtml(text: string): string {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  /**
   * Start the web server for browsing conversation files.
   *
   * @param host - Host address to bind to
   * @param port - Port number to listen on
   */
  startWebServer(host: string = "127.0.0.1", port: number = 5000): void {
    const app = this.createWebApp();
    app.listen(port, host, () => {
      console.log(`Starting tracer web server at http://${host}:${port}`);
      console.log(`Cache directory: ${path.resolve(this.cacheDir)}`);
    });
  }
}
