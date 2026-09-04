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

type ChatTemplateKwargs = Record<string, unknown>;
type ThinkingProfile = Record<ThinkingLevel, ChatTemplateKwargs>;

// vLLM passes chat_template_kwargs straight to the served model's chat template, so the
// switch that turns thinking on is whatever that template happens to read. Each profile
// below maps an AgentHub level onto one family's kwargs; an empty mapping means the
// request carries no chat_template_kwargs at all.

// Qwen3 templates read a single enable_thinking boolean.
const QWEN3_THINKING: ThinkingProfile = {
  [ThinkingLevel.NONE]: { enable_thinking: false },
  [ThinkingLevel.LOW]: { enable_thinking: true },
  [ThinkingLevel.MEDIUM]: { enable_thinking: true },
  [ThinkingLevel.HIGH]: { enable_thinking: true },
  [ThinkingLevel.XHIGH]: { enable_thinking: true },
  [ThinkingLevel.MAX]: { enable_thinking: true },
};

// Qwen3.8-27B keeps enable_thinking as the off switch and takes its adaptive modes as
// reasoning_effort, which accepts only low/medium/xhigh, so high and max clamp to xhigh.
const QWEN3_8_27B_THINKING: ThinkingProfile = {
  [ThinkingLevel.NONE]: { enable_thinking: false },
  [ThinkingLevel.LOW]: { reasoning_effort: "low" },
  [ThinkingLevel.MEDIUM]: { reasoning_effort: "medium" },
  [ThinkingLevel.HIGH]: { reasoning_effort: "xhigh" },
  [ThinkingLevel.XHIGH]: { reasoning_effort: "xhigh" },
  [ThinkingLevel.MAX]: { reasoning_effort: "xhigh" },
};

// DeepSeek V4 templates read a thinking flag paired with reasoning_effort, which accepts
// only low/high/max, so medium and xhigh clamp to high. Thinking is off whenever the flag
// is absent, which is what NONE sends.
const DEEPSEEK_V4_THINKING: ThinkingProfile = {
  [ThinkingLevel.NONE]: {},
  [ThinkingLevel.LOW]: { thinking: true, reasoning_effort: "low" },
  [ThinkingLevel.MEDIUM]: { thinking: true, reasoning_effort: "high" },
  [ThinkingLevel.HIGH]: { thinking: true, reasoning_effort: "high" },
  [ThinkingLevel.XHIGH]: { thinking: true, reasoning_effort: "high" },
  [ThinkingLevel.MAX]: { thinking: true, reasoning_effort: "max" },
};

// Keys are matched as substrings of the lowercased model id, so a served id keeps whatever
// prefix the deployment gave it (Qwen/Qwen3.6-35B-A3B, deepseek-ai/DeepSeek-V4-Pro). The
// first match wins, so a key that contains another must come first: deepseek-v4-flash is a
// prefix of deepseek-v4-flash-vision-exp.
const MODEL_THINKING_PROFILES: readonly (readonly [string, ThinkingProfile])[] =
  [
    ["qwen3.8-flash-next", QWEN3_THINKING],
    ["qwen3.8-27b", QWEN3_8_27B_THINKING],
    ["qwen3.6-35b-a3b", QWEN3_THINKING],
    ["qwen3.5-0.8b", QWEN3_THINKING],
    ["qwen3.5-9b", QWEN3_THINKING],
    ["deepseek-v4-flash-vision-exp", DEEPSEEK_V4_THINKING],
    ["deepseek-v4-pro", DEEPSEEK_V4_THINKING],
    ["deepseek-v4-flash", DEEPSEEK_V4_THINKING],
  ];

/** Models served through vLLM's OpenAI-compatible Chat Completions API. */
export class OpenaiChatVllmAdapterClient extends OpenaiChatClient {
  /**
   * Return the chat_template_kwargs this model's template reads for the level.
   *
   * A model outside the table falls back to Qwen3's enable_thinking, the most widespread
   * of the conventions and inert on a template that ignores the key.
   */
  private _thinkingChatTemplateKwargs(
    thinkingLevel: ThinkingLevel,
  ): ChatTemplateKwargs {
    const model = this._model.toLowerCase();
    for (const [name, profile] of MODEL_THINKING_PROFILES) {
      if (model.includes(name)) {
        return { ...profile[thinkingLevel] };
      }
    }

    return { ...QWEN3_THINKING[thinkingLevel] };
  }

  /** Map AgentHub's level onto the thinking switch this model's chat template reads. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  override transformUniConfigToModelConfig(config: UniConfig): any {
    const vllmConfig = super.transformUniConfigToModelConfig(config);

    if (config.thinking_level !== undefined) {
      const chatTemplateKwargs = this._thinkingChatTemplateKwargs(
        config.thinking_level,
      );
      if (Object.keys(chatTemplateKwargs).length > 0) {
        vllmConfig.chat_template_kwargs = chatTemplateKwargs;
      }
    }

    return vllmConfig;
  }
}
