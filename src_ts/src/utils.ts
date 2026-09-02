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

/** Pixel dimensions read from an image header. */
export interface ImageDimensions {
  width: number;
  height: number;
}

/**
 * Read the pixel dimensions from the header of a PNG, JPEG, GIF or WebP image.
 *
 * Only the header is inspected, so the bytes may be a prefix of the file; a
 * prefix that ends before the dimensions are reached reads as unrecognized.
 *
 * @param bytes - The image bytes, or a prefix of them.
 * @returns The dimensions, or null when the bytes are not a recognized image.
 */
export function imageDimensions(bytes: Uint8Array): ImageDimensions | null {
  const ascii = (offset: number, length: number): string =>
    String.fromCharCode(...bytes.subarray(offset, offset + length));
  const u16be = (offset: number): number =>
    (bytes[offset] << 8) | bytes[offset + 1];
  const u16le = (offset: number): number =>
    bytes[offset] | (bytes[offset + 1] << 8);
  const u24le = (offset: number): number =>
    bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
  const u32be = (offset: number): number =>
    ((bytes[offset] << 24) |
      (bytes[offset + 1] << 16) |
      (bytes[offset + 2] << 8) |
      bytes[offset + 3]) >>>
    0;

  // PNG: an 8-byte signature, then the IHDR chunk with width and height
  if (
    bytes.length >= 24 &&
    ascii(0, 8) === "\x89PNG\r\n\x1a\n" &&
    ascii(12, 4) === "IHDR"
  ) {
    return { width: u32be(16), height: u32be(20) };
  }

  // GIF: the logical screen size follows the 6-byte signature
  if (bytes.length >= 10 && ["GIF87a", "GIF89a"].includes(ascii(0, 6))) {
    return { width: u16le(6), height: u16le(8) };
  }

  // WebP: a RIFF container whose first chunk names the bitstream flavour
  if (bytes.length >= 30 && ascii(0, 4) === "RIFF" && ascii(8, 4) === "WEBP") {
    const chunk = ascii(12, 4);
    if (
      chunk === "VP8 " &&
      bytes[23] === 0x9d &&
      bytes[24] === 0x01 &&
      bytes[25] === 0x2a
    ) {
      // lossy: a 3-byte frame tag and the key frame start code precede the size,
      // whose top two bits are a scaling hint
      return { width: u16le(26) & 0x3fff, height: u16le(28) & 0x3fff };
    }
    if (chunk === "VP8L" && bytes[20] === 0x2f) {
      // lossless: 14 bits of width minus one, then 14 bits of height minus one
      const bits = u24le(21) | (bytes[24] << 24);
      return {
        width: (bits & 0x3fff) + 1,
        height: ((bits >>> 14) & 0x3fff) + 1,
      };
    }
    if (chunk === "VP8X") {
      // extended: the canvas size minus one, 24 bits each, after the flags
      return { width: u24le(24) + 1, height: u24le(27) + 1 };
    }
    return null;
  }

  // JPEG: walk the marker segments to the first frame header (SOFn)
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    let offset = 2;
    while (offset + 3 < bytes.length) {
      if (bytes[offset] !== 0xff) {
        return null;
      }
      const marker = bytes[offset + 1];
      if (marker === 0xff) {
        // fill byte ahead of a marker
        offset += 1;
        continue;
      }
      if (
        marker === 0x01 ||
        marker === 0xd8 ||
        (marker >= 0xd0 && marker <= 0xd7)
      ) {
        // TEM, SOI and RSTn stand alone, without a length
        offset += 2;
        continue;
      }
      if (marker === 0xd9 || marker === 0xda) {
        // end of image, or scan data before any frame header
        return null;
      }
      const frameHeader =
        marker >= 0xc0 &&
        marker <= 0xcf &&
        marker !== 0xc4 &&
        marker !== 0xc8 &&
        marker !== 0xcc;
      if (frameHeader) {
        if (offset + 8 >= bytes.length) {
          return null;
        }
        // length, precision, then height before width
        return { width: u16be(offset + 7), height: u16be(offset + 5) };
      }
      offset += 2 + u16be(offset + 2);
    }
    return null;
  }

  return null;
}

/**
 * The patch count above which the OpenAI vision API rejects an image instead
 * of resizing it.
 */
const OPENAI_PATCH_LIMIT = 30000;

/**
 * The longest side the OpenAI vision API keeps at `original` detail; a larger
 * image is scaled down to fit it before the patches are counted.
 */
const OPENAI_ORIGINAL_MAX_SIDE = 65535;

/**
 * Base64 characters decoded on the first attempt: enough for any PNG, GIF or
 * WebP header, and for a JPEG whose metadata segments are of ordinary size.
 */
const HEADER_PREFIX_CHARS = 64 * 1024;

/**
 * Decode the leading bytes of a base64 data URL.
 *
 * @param dataUrl - The URL.
 * @param maxChars - How many base64 characters to decode at most.
 * @returns The decoded bytes, or null when the URL is not a base64 data URL.
 */
function dataUrlBytes(dataUrl: string, maxChars: number): Buffer | null {
  if (!dataUrl.startsWith("data:")) {
    return null;
  }
  const comma = dataUrl.indexOf(",");
  if (comma < 0 || !dataUrl.slice(5, comma).split(";").includes("base64")) {
    return null;
  }
  // whole 4-character groups only, so the cut cannot land inside one
  const chars = Math.min(dataUrl.length - comma - 1, maxChars) & ~3;
  return Buffer.from(dataUrl.slice(comma + 1, comma + 1 + chars), "base64");
}

/**
 * Whether the OpenAI vision API would reject an image at `original` detail.
 *
 * The API covers an image with 32-pixel patches and rejects one that needs
 * more than 30,000 of them after its own resizing; at `original` detail the
 * only resizing is the 65,535-pixel cap on either side. Only a base64 data URL
 * can be measured here: an HTTP(S) URL is fetched by the API itself and
 * answers false.
 *
 * @param imageUrl - The image URL.
 * @returns Whether the API would reject the image.
 */
export function exceedsOpenaiPatchLimit(imageUrl: string): boolean {
  let bytes = dataUrlBytes(imageUrl, HEADER_PREFIX_CHARS);
  if (bytes === null) {
    return false;
  }
  let size = imageDimensions(bytes);
  if (size === null && bytes[0] === 0xff && bytes[1] === 0xd8) {
    // a JPEG may carry more metadata than the prefix before its frame header
    bytes = dataUrlBytes(imageUrl, Infinity) as Buffer;
    size = imageDimensions(bytes);
  }
  if (size === null) {
    return false;
  }

  let { width, height } = size;
  const longest = Math.max(width, height);
  if (longest > OPENAI_ORIGINAL_MAX_SIDE) {
    width = Math.round((width * OPENAI_ORIGINAL_MAX_SIDE) / longest);
    height = Math.round((height * OPENAI_ORIGINAL_MAX_SIDE) / longest);
  }
  return Math.ceil(width / 32) * Math.ceil(height / 32) > OPENAI_PATCH_LIMIT;
}
