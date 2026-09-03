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

import { OpenaiChatClient } from "../openai_chat";
import { ThinkingLevel, UniConfig } from "../types";

/** Qwen models served through vLLM's OpenAI-compatible Chat Completions API. */
export class QwenVllmClient extends OpenaiChatClient {
  /** Map AgentHub's level to the boolean switch consumed by Qwen chat templates. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  override transformUniConfigToModelConfig(config: UniConfig): any {
    const qwenConfig = super.transformUniConfigToModelConfig(config);

    if (config.thinking_level !== undefined) {
      qwenConfig.chat_template_kwargs = {
        enable_thinking: config.thinking_level !== ThinkingLevel.NONE,
      };
    }

    return qwenConfig;
  }
}
