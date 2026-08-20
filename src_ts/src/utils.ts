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
 * Whether AGENTHUB_DEBUG asks the clients to fail loudly on output they do not recognize.
 *
 * Streaming clients skip an unrecognized event so that a gateway's own frames cannot
 * kill a long generation. The same silence hides a genuinely new provider event, so the
 * guards stay one environment variable away.
 *
 * @returns Whether debug mode is on.
 */
export function isDebugEnabled(): boolean {
  const flag = (process.env.AGENTHUB_DEBUG || "").trim().toLowerCase();
  return !["", "0", "false", "no", "off"].includes(flag);
}
