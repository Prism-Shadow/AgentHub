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

import { UsageMetadata } from "./types";

/**
 * Fix the usage metadata for OpenRouter.
 *
 * OpenRouter occasionally does not include the reasoning tokens to the completion tokens.
 *
 * @param usageMetadata - The usage metadata.
 * @param baseUrl - The API URL.
 * @returns The fixed usage metadata.
 */
export function fixOpenrouterUsageMetadata(
  usageMetadata: UsageMetadata,
  baseUrl: string,
): UsageMetadata {
  const fixedUsageMetadata = { ...usageMetadata };
  if (
    baseUrl.includes("openrouter.ai") &&
    fixedUsageMetadata.response_tokens !== null &&
    fixedUsageMetadata.response_tokens < 0
  ) {
    fixedUsageMetadata.response_tokens +=
      fixedUsageMetadata.thoughts_tokens || 0;
  }

  return fixedUsageMetadata;
}

/**
 * Whether a stream event came from outside the protocol and carries nothing.
 *
 * Gateways in front of a model API (one-api-style proxies, OpenRouter) inject
 * their own events into the SSE stream — heartbeats, cost tickers — and the
 * unknown-event guard used to kill the whole stream on one, e.g.
 * `{"type":"ping","cost":"@"}`. Skipping is safe only where all three hold: the
 * event type sits outside the protocol's own namespace, so a provider event the
 * client has not learned yet — `response.output_text.annotation.added`, say —
 * still raises; the type does not name an error, so a gateway reporting an
 * upstream failure still raises; and no field holds a non-empty object or
 * array, so an event carrying a payload the client would silently drop still
 * raises.
 *
 * @param modelOutput - The stream event.
 * @param protocolPrefixes - The event type prefixes the protocol owns.
 * @returns Whether the event can be skipped.
 */
export function isForeignNoOpEvent(
  modelOutput: unknown,
  protocolPrefixes: readonly string[],
): boolean {
  if (typeof modelOutput !== "object" || modelOutput === null) {
    return false;
  }

  const fields = modelOutput as Record<string, unknown>;
  const eventType = typeof fields.type === "string" ? fields.type : "";
  if (
    protocolPrefixes.some((prefix) => eventType.startsWith(prefix)) ||
    eventType.includes("error") ||
    eventType.includes("fail")
  ) {
    return false;
  }

  return !Object.values(fields).some((value) => {
    if (typeof value !== "object" || value === null) {
      return false;
    }

    return Array.isArray(value)
      ? value.length > 0
      : Object.keys(value).length > 0;
  });
}
