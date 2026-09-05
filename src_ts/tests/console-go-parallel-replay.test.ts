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

import { expect, test } from "@jest/globals";
import { AutoLLMClient, UniMessage } from "../src";

/**
 * Regression coverage for the Console Go ("opencode_go") parallel-tool-call replay.
 *
 * A turn in which the model emits TWO parallel function calls is answered by two
 * independent `tool_result` user messages (the harness settles each tool separately).
 * The replayed transcript therefore contains:
 *
 *   assistant: [thinking(fidelity-only), text, FC1, FC2]
 *   user:      [tool_result FC1]
 *   user:      [tool_result FC2]
 *
 * Before the fix, the fidelity-only thinking item (the reasoning block a Responses
 * server streams with an encrypted blob and no visible text) was pushed to the wire
 * right before the two function calls. Console Go then rejected the request with
 * `400 No function call found for function_call_output …` — a replay of an encrypted
 * reasoning blob adjacent to parallel function calls breaks the gateway's call replay.
 */
function client(): AutoLLMClient {
  return new AutoLLMClient({
    model: "muse-spark-1.3-contributor",
    apiKey: "test-key",
    clientType: "openai-responses",
  });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function inputOf(c: AutoLLMClient, messages: UniMessage[]): any[] {
  const routed = (
    c as unknown as {
      _client: {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        transformUniMessageToModelInput(messages: UniMessage[]): any[];
      };
    }
  )._client;
  return routed.transformUniMessageToModelInput(messages);
}

test("fidelity-only encrypted reasoning is omitted from a parallel tool replay", () => {
  const c = client();
  const messages: UniMessage[] = [
    {
      role: "user",
      content_items: [{ type: "text", text: "run both commands now" }],
    },
    {
      role: "assistant",
      content_items: [
        {
          type: "thinking",
          thinking: "",
          fidelity: {
            encrypted_content: "Q-PaDgGTAx3nSg5xMxtJJRsVL-fake",
            signature: "sig-fake",
            format: "fake",
          },
        },
        { type: "text", text: "Running both now." },
        {
          type: "tool_call",
          name: "exec_command",
          arguments: { cmd: "echo a" },
          tool_call_id: "call_a",
        },
        {
          type: "tool_call",
          name: "exec_command",
          arguments: { cmd: "echo b" },
          tool_call_id: "call_b",
        },
      ],
    },
    {
      role: "user",
      content_items: [
        { type: "tool_result", text: "A", tool_call_id: "call_a" },
      ],
    },
    {
      role: "user",
      content_items: [
        { type: "tool_result", text: "B", tool_call_id: "call_b" },
      ],
    },
  ];

  const input = inputOf(c, messages);
  const types = input.map((item) => item.type ?? `message:${item.role}`);
  expect(types).toEqual([
    "message:user",
    "message:assistant",
    "function_call",
    "function_call",
    "function_call_output",
    "function_call_output",
  ]);
  expect(
    input.some(
      (item) => item.type === "reasoning" || item.encrypted_content !== undefined,
    ),
  ).toBe(false);
  const outputs = input.filter((item) => item.type === "function_call_output");
  expect(outputs.map((o) => o.call_id)).toEqual(["call_a", "call_b"]);
});

test("thinking WITH visible text still replays as a reasoning item", () => {
  const c = client();
  const messages: UniMessage[] = [
    {
      role: "user",
      content_items: [{ type: "text", text: "run the command" }],
    },
    {
      role: "assistant",
      content_items: [
        {
          type: "thinking",
          thinking: "I should call the tool.",
          fidelity: { encrypted_content: "enc-fake", signature: "sig-fake" },
        },
        {
          type: "tool_call",
          name: "exec_command",
          arguments: { cmd: "echo a" },
          tool_call_id: "call_a",
        },
      ],
    },
    {
      role: "user",
      content_items: [
        { type: "tool_result", text: "A", tool_call_id: "call_a" },
      ],
    },
  ];

  const input = inputOf(c, messages);
  const types = input.map((item) => item.type ?? `message:${item.role}`);
  expect(types).toEqual([
    "message:user",
    "reasoning",
    "function_call",
    "function_call_output",
  ]);
  const reasoning = input.find((item) => item.type === "reasoning");
  expect(reasoning.content).toEqual([
    { type: "reasoning_text", text: "I should call the tool." },
  ]);
});
