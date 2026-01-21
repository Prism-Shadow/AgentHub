#!/usr/bin/env node
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

const { Gemini3Client } = require("./dist/gemini3");

async function main() {
  if (!process.env.GEMINI_API_KEY && !process.env.GOOGLE_API_KEY) {
    console.log("No API key found. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.");
    process.exit(1);
  }

  const client = new Gemini3Client("gemini-3-flash-preview");
  const messages = [
    {
      role: "user",
      content_items: [{ type: "text", text: "What is 2+3? Please answer briefly." }],
    },
  ];
  const config = {};

  console.log("Testing Gemini3Client...");
  console.log("Question: What is 2+3?");
  console.log("Response:");

  let fullText = "";
  for await (const event of client.streamingResponse(messages, config)) {
    for (const item of event.content_items) {
      if (item.type === "text") {
        process.stdout.write(item.text);
        fullText += item.text;
      }
    }
  }

  console.log("\n\nTest completed!");
  if (fullText.includes("5")) {
    console.log("✓ Response contains '5' - test passed!");
  } else {
    console.log("✗ Response does not contain '5' - test failed!");
    console.log("Full response:", fullText);
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
