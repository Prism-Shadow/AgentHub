# Copyright 2025 Prism Shadow. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import binascii
import math
import os

from .types import UsageMetadata


def fix_openrouter_usage_metadata(usage_metadata: UsageMetadata, base_url: str) -> UsageMetadata:
    """
    Fix the usage metadata for OpenRouter.

    OpenRouter occasionally does not include the reasoning tokens to the completion tokens.

    Args:
        usage_metadata (UsageMetadata): The usage metadata.
        base_url (str): The API URL.

    Returns:
        UsageMetadata: The fixed usage metadata.
    """
    fixed_usage_metadata = usage_metadata.copy()
    if "openrouter.ai" in base_url and fixed_usage_metadata["response_tokens"] < 0:
        fixed_usage_metadata["response_tokens"] += fixed_usage_metadata["thoughts_tokens"] or 0

    return fixed_usage_metadata


def is_debug_enabled() -> bool:
    """
    Whether AGENTHUB_DEBUG asks the clients to fail loudly on output they do not recognize.

    Streaming clients skip an unrecognized event so that a gateway's own frames cannot kill a
    long generation. The same silence hides a genuinely new provider event, so the guards stay
    one environment variable away.

    Returns:
        bool: Whether debug mode is on.
    """
    return os.getenv("AGENTHUB_DEBUG", "").strip().lower() not in ("", "0", "false", "no", "off")


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    """
    Read the pixel dimensions from the header of a PNG, JPEG, GIF or WebP image.

    Only the header is inspected, so the bytes may be a prefix of the file; a prefix that ends
    before the dimensions are reached reads as unrecognized.

    Args:
        data (bytes): The image bytes, or a prefix of them.

    Returns:
        tuple[int, int] | None: The width and height, or None when the bytes are not a recognized image.
    """
    # PNG: an 8-byte signature, then the IHDR chunk with width and height
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")

    # GIF: the logical screen size follows the 6-byte signature
    if len(data) >= 10 and data[:6] in (b"GIF87a", b"GIF89a"):
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")

    # WebP: a RIFF container whose first chunk names the bitstream flavour
    if len(data) >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
            # lossy: a 3-byte frame tag and the key frame start code precede the size, whose top
            # two bits are a scaling hint
            width = int.from_bytes(data[26:28], "little") & 0x3FFF
            height = int.from_bytes(data[28:30], "little") & 0x3FFF
            return width, height
        if chunk == b"VP8L" and data[20] == 0x2F:
            # lossless: 14 bits of width minus one, then 14 bits of height minus one
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8X":
            # extended: the canvas size minus one, 24 bits each, after the flags
            return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
        return None

    # JPEG: walk the marker segments to the first frame header (SOFn)
    if len(data) >= 4 and data[:2] == b"\xff\xd8":
        offset = 2
        while offset + 3 < len(data):
            if data[offset] != 0xFF:
                return None
            marker = data[offset + 1]
            if marker == 0xFF:  # fill byte ahead of a marker
                offset += 1
                continue
            if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:  # TEM, SOI and RSTn stand alone, without a length
                offset += 2
                continue
            if marker in (0xD9, 0xDA):  # end of image, or scan data before any frame header
                return None
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                if offset + 8 >= len(data):
                    return None
                # length, precision, then height before width
                height = int.from_bytes(data[offset + 5 : offset + 7], "big")
                width = int.from_bytes(data[offset + 7 : offset + 9], "big")
                return width, height
            offset += 2 + int.from_bytes(data[offset + 2 : offset + 4], "big")
        return None

    return None


# The patch count above which the OpenAI vision API rejects an image instead of resizing it.
_OPENAI_PATCH_LIMIT = 30000

# The longest side the OpenAI vision API keeps at `original` detail; a larger image is scaled
# down to fit it before the patches are counted.
_OPENAI_ORIGINAL_MAX_SIDE = 65535

# Base64 characters decoded on the first attempt: enough for any PNG, GIF or WebP header, and
# for a JPEG whose metadata segments are of ordinary size.
_HEADER_PREFIX_CHARS = 64 * 1024


def _data_url_bytes(data_url: str, max_chars: int | None) -> bytes | None:
    """
    Decode the leading bytes of a base64 data URL.

    Args:
        data_url (str): The URL.
        max_chars (int | None): How many base64 characters to decode at most, or None for all of them.

    Returns:
        bytes | None: The decoded bytes, or None when the URL is not a base64 data URL.
    """
    if not data_url.startswith("data:"):
        return None
    comma = data_url.find(",")
    if comma < 0 or "base64" not in data_url[5:comma].split(";"):
        return None
    payload = data_url[comma + 1 :]
    if max_chars is not None:
        # whole 4-character groups only, so the cut cannot land inside one
        payload = payload[: min(len(payload), max_chars) & ~3]
    try:
        return base64.b64decode(payload)
    except (binascii.Error, ValueError):
        return None


def exceeds_openai_patch_limit(image_url: str) -> bool:
    """
    Whether the OpenAI vision API would reject an image at `original` detail.

    The API covers an image with 32-pixel patches and rejects one that needs more than 30,000 of
    them after its own resizing; at `original` detail the only resizing is the 65,535-pixel cap
    on either side. Only a base64 data URL can be measured here: an HTTP(S) URL is fetched by
    the API itself and answers False.

    Args:
        image_url (str): The image URL.

    Returns:
        bool: Whether the API would reject the image.
    """
    data = _data_url_bytes(image_url, _HEADER_PREFIX_CHARS)
    if data is None:
        return False
    size = image_dimensions(data)
    if size is None and data[:2] == b"\xff\xd8":
        # a JPEG may carry more metadata than the prefix before its frame header
        data = _data_url_bytes(image_url, None)
        size = image_dimensions(data) if data is not None else None
    if size is None:
        return False

    width, height = size
    longest = max(width, height)
    if longest > _OPENAI_ORIGINAL_MAX_SIDE:
        width = math.floor(width * _OPENAI_ORIGINAL_MAX_SIDE / longest + 0.5)
        height = math.floor(height * _OPENAI_ORIGINAL_MAX_SIDE / longest + 0.5)
    return math.ceil(width / 32) * math.ceil(height / 32) > _OPENAI_PATCH_LIMIT
