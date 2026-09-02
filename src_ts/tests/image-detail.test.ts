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

import { expect, describe, test } from "@jest/globals";
import { AutoLLMClient, UniMessage } from "../src";
import { exceedsOpenaiPatchLimit, imageDimensions } from "../src/utils";

// Header builders. Only the bytes the parser reads have to be right, so none of these is a
// decodable picture; a real file carries the same leading bytes.
function png(width: number, height: number): Buffer {
  const header = Buffer.alloc(24);
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(header);
  header.writeUInt32BE(13, 8);
  header.write("IHDR", 12, "latin1");
  header.writeUInt32BE(width, 16);
  header.writeUInt32BE(height, 20);
  return header;
}

function gif(width: number, height: number): Buffer {
  const header = Buffer.alloc(13);
  header.write("GIF89a", 0, "latin1");
  header.writeUInt16LE(width, 6);
  header.writeUInt16LE(height, 8);
  return header;
}

// The frame header follows `metadataBytes` of APP1 payload, the way EXIF does; a segment
// holds at most 65,533 bytes, so a larger amount spans several of them.
function jpeg(width: number, height: number, metadataBytes = 0): Buffer {
  const parts: Buffer[] = [Buffer.from([0xff, 0xd8])];
  let remaining = metadataBytes;
  do {
    const payload = Math.min(remaining, 65533);
    const app1 = Buffer.alloc(4 + payload);
    app1[0] = 0xff;
    app1[1] = 0xe1;
    app1.writeUInt16BE(2 + payload, 2);
    parts.push(app1);
    remaining -= payload;
  } while (remaining > 0);
  const sof0 = Buffer.alloc(10);
  sof0[0] = 0xff;
  sof0[1] = 0xc0;
  sof0.writeUInt16BE(8, 2);
  sof0[4] = 8;
  sof0.writeUInt16BE(height, 5);
  sof0.writeUInt16BE(width, 7);
  sof0[9] = 3;
  return Buffer.concat([...parts, sof0]);
}

function riff(chunk: string, payload: Buffer): Buffer {
  const header = Buffer.alloc(20);
  header.write("RIFF", 0, "latin1");
  header.writeUInt32LE(4 + 8 + payload.length, 4);
  header.write("WEBP", 8, "latin1");
  header.write(chunk, 12, "latin1");
  header.writeUInt32LE(payload.length, 16);
  return Buffer.concat([header, payload]);
}

// The scaling bits above the 14-bit size are set, to prove they are masked off.
function webpLossy(width: number, height: number): Buffer {
  const payload = Buffer.alloc(10);
  payload[3] = 0x9d;
  payload[4] = 0x01;
  payload[5] = 0x2a;
  payload.writeUInt16LE(width | (2 << 14), 6);
  payload.writeUInt16LE(height | (1 << 14), 8);
  return riff("VP8 ", payload);
}

// The alpha flag above the two 14-bit sizes is set, to prove it is masked off.
function webpLossless(width: number, height: number): Buffer {
  const payload = Buffer.alloc(10);
  payload[0] = 0x2f;
  payload.writeUInt32LE(
    ((width - 1) | ((height - 1) << 14) | (1 << 28)) >>> 0,
    1,
  );
  return riff("VP8L", payload);
}

function webpExtended(width: number, height: number): Buffer {
  const payload = Buffer.alloc(10);
  payload[0] = 0x10;
  payload.writeUIntLE(width - 1, 4, 3);
  payload.writeUIntLE(height - 1, 7, 3);
  return riff("VP8X", payload);
}

function dataUrl(bytes: Buffer, mime = "image/png"): string {
  return `data:${mime};base64,${bytes.toString("base64")}`;
}

describe("imageDimensions", () => {
  test.each([
    ["PNG", png(6400, 8608), { width: 6400, height: 8608 }],
    ["GIF", gif(640, 480), { width: 640, height: 480 }],
    ["JPEG", jpeg(4032, 3024), { width: 4032, height: 3024 }],
    [
      "JPEG behind 100 KiB of metadata",
      jpeg(4032, 3024, 100 * 1024),
      { width: 4032, height: 3024 },
    ],
    ["lossy WebP", webpLossy(1920, 1080), { width: 1920, height: 1080 }],
    ["lossless WebP", webpLossless(1920, 1080), { width: 1920, height: 1080 }],
    ["extended WebP", webpExtended(1920, 1080), { width: 1920, height: 1080 }],
  ])("reads the size of a %s header", (_name, bytes, expected) => {
    expect(imageDimensions(bytes)).toEqual(expected);
  });

  test.each([
    ["empty input", Buffer.alloc(0)],
    ["text", Buffer.from("not an image")],
    ["a PNG cut before its IHDR chunk", png(6400, 8608).subarray(0, 16)],
    [
      "a JPEG cut inside a metadata segment",
      jpeg(4032, 3024, 1024).subarray(0, 512),
    ],
    [
      "a JPEG whose scan starts before any frame header",
      Buffer.from([0xff, 0xd8, 0xff, 0xda, 0x00, 0x02, 0x00, 0x00]),
    ],
    [
      "a WebP whose first chunk is not a bitstream",
      riff("ALPH", Buffer.alloc(10)),
    ],
  ])("reads %s as unrecognized", (_name, bytes) => {
    expect(imageDimensions(bytes)).toBeNull();
  });
});

describe("exceedsOpenaiPatchLimit", () => {
  test.each([
    ["a 6400x8608 screenshot (53,800 patches)", dataUrl(png(6400, 8608)), true],
    ["5600x5600 (30,625 patches)", dataUrl(png(5600, 5600)), true],
    ["5504x5504 (29,584 patches)", dataUrl(png(5504, 5504)), false],
    ["1024x1024", dataUrl(png(1024, 1024)), false],
    [
      "an 8000x8000 JPEG whose frame header sits behind 100 KiB of metadata",
      dataUrl(jpeg(8000, 8000, 100 * 1024), "image/jpeg"),
      true,
    ],
    [
      "a 131070x500 strip, once scaled to the 65,535-pixel side",
      dataUrl(png(131070, 500)),
      false,
    ],
    [
      "a 131070x1500 strip, still over the limit once scaled",
      dataUrl(png(131070, 1500)),
      true,
    ],
    ["an http URL", "https://example.com/huge.png", false],
    [
      "a data URL that is not base64",
      `data:image/png,${encodeURIComponent("not base64")}`,
      false,
    ],
    [
      "a data URL of something that is not an image",
      dataUrl(Buffer.from("hello")),
      false,
    ],
  ])("%s", (_name, url, expected) => {
    expect(exceedsOpenaiPatchLimit(url)).toBe(expected);
  });
});

interface ImageDetailCase {
  expectedClient: string;
  model: string;
  clientType?: string;
  protocol: "responses" | "chat";
  // whether an image over the patch limit goes out at high detail
  shrinks: boolean;
}

const IMAGE_DETAIL_CASES: ImageDetailCase[] = [
  {
    expectedClient: "GPT5_6Client",
    model: "gpt-5.6-terra",
    protocol: "responses",
    shrinks: true,
  },
  {
    expectedClient: "GPT5_6Client",
    model: "gpt-5.5",
    protocol: "responses",
    shrinks: false,
  },
  {
    expectedClient: "OpenaiResponsesClient",
    model: "openai/gpt-5.6-terra",
    clientType: "openai-responses",
    protocol: "responses",
    shrinks: true,
  },
  {
    expectedClient: "OpenaiResponsesClient",
    model: "deepseek-v4-flash-vision-exp",
    clientType: "openai-responses",
    protocol: "responses",
    shrinks: false,
  },
  {
    expectedClient: "OpenaiChatClient",
    model: "GPT-5.6-Sol",
    clientType: "openai-chat",
    protocol: "chat",
    shrinks: true,
  },
  {
    expectedClient: "OpenaiChatClient",
    model: "gpt-5.5",
    clientType: "openai-chat",
    protocol: "chat",
    shrinks: false,
  },
];

const OVERSIZED = dataUrl(png(6400, 8608));
const SMALL = dataUrl(png(1024, 1024));

// A prompt carrying an oversized image next to a small one, and a tool result carrying both again.
const MESSAGES: UniMessage[] = [
  {
    role: "user",
    content_items: [
      { type: "text", text: "What is in these?" },
      { type: "image_url", image_url: OVERSIZED },
      { type: "image_url", image_url: SMALL },
    ],
  },
  {
    role: "assistant",
    content_items: [
      {
        type: "tool_call",
        name: "read_image",
        arguments: { path: "shot.png" },
        tool_call_id: "call_1",
      },
    ],
  },
  {
    role: "user",
    content_items: [
      {
        type: "tool_result",
        text: "image/png",
        images: [OVERSIZED, SMALL],
        tool_call_id: "call_1",
      },
    ],
  },
];

// The detail of each image part, in order: the prompt's two images, then the tool result's two.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function details(testCase: ImageDetailCase, modelInput: any[]): unknown[] {
  if (testCase.protocol === "responses") {
    return [
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ...modelInput[0].content.slice(1).map((part: any) => part.detail),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ...modelInput[2].output.slice(1).map((part: any) => part.detail),
    ];
  }

  return [
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ...modelInput[0].content.slice(1).map((part: any) => part.image_url.detail),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ...modelInput[2].content.slice(1).map((part: any) => part.image_url.detail),
  ];
}

describe.each(IMAGE_DETAIL_CASES)(
  "Image detail for $expectedClient on $model",
  (testCase) => {
    test(
      testCase.shrinks
        ? "sends an image over the patch limit at high detail"
        : "leaves the detail level to the model",
      async () => {
        const client = new AutoLLMClient({
          model: testCase.model,
          apiKey: "test-key",
          clientType: testCase.clientType,
        });
        const routedClient = (
          client as unknown as {
            _client: {
              constructor: { name: string };
              transformUniMessageToModelInput(
                messages: UniMessage[],
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
              ): Promise<any[]> | any[];
            };
          }
        )._client;
        expect(routedClient.constructor.name).toBe(testCase.expectedClient);

        const modelInput =
          await routedClient.transformUniMessageToModelInput(MESSAGES);

        const shrunk = testCase.shrinks ? "high" : undefined;
        expect(details(testCase, modelInput)).toEqual([
          shrunk,
          undefined,
          shrunk,
          undefined,
        ]);
      },
    );
  },
);
