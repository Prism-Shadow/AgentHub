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
 * Example demonstrating base64 image understanding capability.
 * 
 * This example shows how to use the AutoLLMClient to analyze images using base64 encoding.
 */

import * as fs from "fs";
import * as path from "path";
import { AutoLLMClient } from "../src/autoClient";

async function main() {
  console.log("=".repeat(60));
  console.log("Base64 Image Understanding Example");
  console.log("=".repeat(60));

  // Get model from environment variable, default to gpt-5.2
  const model = process.env.MODEL || "gpt-5.2";
  console.log(`Using model: ${model}`);

  const client = new AutoLLMClient({ model });
  const config = {};

  // Read a test image and encode to base64
  const imagePath = path.join(__dirname, "../../.github/images/agenthub.png");
  const imageBuffer = fs.readFileSync(imagePath);
  const base64Image = imageBuffer.toString("base64");

  // Create data URI
  const dataUri = `data:image/png;base64,${base64Image}`;

  const query = "What's in this image? Please describe what you see.";
  console.log(`User: ${query}`);
  console.log(`Image: base64 encoded (${base64Image.length} characters)`);
  console.log("Assistant:");

  for await (const event of client.streamingResponse({
    messages: [
      {
        role: "user",
        content_items: [
          { type: "text", text: query },
          { type: "image_url", image_url: dataUri },
        ],
      },
    ],
    config,
  })) {
    for (const item of event.content_items) {
      if (item.type === "text") {
        process.stdout.write(item.text);
      }
    }
  }

  console.log("\n" + "=".repeat(60));
  console.log("Image analysis complete!");
  console.log("=".repeat(60));
}

main().catch(console.error);
