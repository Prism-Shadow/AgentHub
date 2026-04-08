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
 * Example demonstrating streamed Gemini image generation/editing.
 *
 * This example sends a public cat image plus a text instruction, prints each
 * streamed event with binary payloads redacted, and saves any generated image
 * chunks to the current working directory.
 */

import { writeFile } from "fs/promises";
import path from "path";
import { AutoLLMClient } from "../src";

const CAT_IMAGE_URL =
  "https://images.unsplash.com/photo-1519052537078-e6302a4968d4?auto=format&fit=crop&w=800&q=80";

function withoutBinaryData(value: unknown): unknown {
  // Recursively redact binary data of image in the event object
  if (Buffer.isBuffer(value)) {
    return "<hidden_binary_data>";
  }
  if (Array.isArray(value)) {
    return value.map((item) => withoutBinaryData(item));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        withoutBinaryData(item),
      ]),
    );
  }
  return value;
}

function extensionFromMimeType(mimeType: string): string {
  const mimeTypeMap: Record<string, string> = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
  };
  return mimeTypeMap[mimeType] || "bin";
}

async function main(): Promise<void> {
  console.log("=".repeat(60));
  console.log("Image Generation Example");
  console.log("=".repeat(60));

  const model = process.env.MODEL || "gemini-3.1-flash-image-preview";
  console.log(`Using model: ${model}`);

  const client = new AutoLLMClient({ model });
  const prompt =
    "Create a picture of my cat eating a nano-banana in a fancy restaurant under the Gemini constellation.";

  console.log(`User: ${prompt}`);
  console.log(`Image: ${CAT_IMAGE_URL}`);
  console.log("Assistant:");

  let savedImageCount = 0;
  for await (const event of client.streamingResponse({
    messages: [
      {
        role: "user",
        content_items: [
          { type: "text", text: prompt },
          // Uncomment to edit the image from URL
          // { type: "image_url", image_url: CAT_IMAGE_URL },
        ],
      },
    ],
    config: {
      image_config: { aspect_ratio: "16:9", image_size: "1K" },
    },
  })) {
    console.log(withoutBinaryData(event));

    for (const item of event.content_items) {
      if (item.type === "text") {
        console.log(`Text: ${item.text}`);
      } else if (item.type === "inline_data") {
        savedImageCount += 1;
        const extension = extensionFromMimeType(item.mime_type);
        const outputPath = path.resolve(
          process.cwd(),
          `generated_image_${savedImageCount}.${extension}`,
        );
        await writeFile(outputPath, item.data);
        console.log(`Saved image chunk to: ${outputPath}`);
      }
    }
  }

  console.log("\n" + "=".repeat(60));
  console.log("Image generation complete!");
  console.log("=".repeat(60));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
