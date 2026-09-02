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
import inspect
import struct
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import pytest

from agenthub import AutoLLMClient
from agenthub.utils import exceeds_openai_patch_limit, image_dimensions


# Header builders. Only the bytes the parser reads have to be right, so none of these is a
# decodable picture; a real file carries the same leading bytes.
def _png(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)


def _gif(width: int, height: int) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + bytes(3)


def _jpeg(width: int, height: int, metadata_bytes: int = 0) -> bytes:
    """
    The frame header follows `metadata_bytes` of APP1 payload, the way EXIF does; a segment
    holds at most 65,533 bytes, so a larger amount spans several of them.
    """
    parts = [b"\xff\xd8"]
    remaining = metadata_bytes
    while True:
        payload = min(remaining, 65533)
        parts.append(b"\xff\xe1" + struct.pack(">H", 2 + payload) + bytes(payload))
        remaining -= payload
        if remaining <= 0:
            break

    parts.append(b"\xff\xc0" + struct.pack(">HBHHB", 8, 8, height, width, 3))
    return b"".join(parts)


def _riff(chunk: bytes, payload: bytes) -> bytes:
    return (
        b"RIFF" + struct.pack("<I", 4 + 8 + len(payload)) + b"WEBP" + chunk + struct.pack("<I", len(payload)) + payload
    )


def _webp_lossy(width: int, height: int) -> bytes:
    """The scaling bits above the 14-bit size are set, to prove they are masked off."""
    return _riff(b"VP8 ", bytes(3) + b"\x9d\x01\x2a" + struct.pack("<HH", width | (2 << 14), height | (1 << 14)))


def _webp_lossless(width: int, height: int) -> bytes:
    """The alpha flag above the two 14-bit sizes is set, to prove it is masked off."""
    bits = (width - 1) | ((height - 1) << 14) | (1 << 28)
    return _riff(b"VP8L", b"\x2f" + struct.pack("<I", bits) + bytes(5))


def _webp_extended(width: int, height: int) -> bytes:
    return _riff(b"VP8X", b"\x10\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little"))


def _data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


@pytest.mark.parametrize(
    ("name", "data", "expected"),
    [
        ("PNG", _png(6400, 8608), (6400, 8608)),
        ("GIF", _gif(640, 480), (640, 480)),
        ("JPEG", _jpeg(4032, 3024), (4032, 3024)),
        ("JPEG behind 100 KiB of metadata", _jpeg(4032, 3024, 100 * 1024), (4032, 3024)),
        ("lossy WebP", _webp_lossy(1920, 1080), (1920, 1080)),
        ("lossless WebP", _webp_lossless(1920, 1080), (1920, 1080)),
        ("extended WebP", _webp_extended(1920, 1080), (1920, 1080)),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_image_dimensions_reads_the_header(name: str, data: bytes, expected: tuple[int, int]):
    assert image_dimensions(data) == expected


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("empty input", b""),
        ("text", b"not an image"),
        ("a PNG cut before its IHDR chunk", _png(6400, 8608)[:16]),
        ("a JPEG cut inside a metadata segment", _jpeg(4032, 3024, 1024)[:512]),
        ("a JPEG whose scan starts before any frame header", b"\xff\xd8\xff\xda\x00\x02\x00\x00"),
        ("a WebP whose first chunk is not a bitstream", _riff(b"ALPH", bytes(10))),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_image_dimensions_reads_unrecognized_bytes_as_none(name: str, data: bytes):
    assert image_dimensions(data) is None


@pytest.mark.parametrize(
    ("name", "url", "expected"),
    [
        ("a 6400x8608 screenshot (53,800 patches)", _data_url(_png(6400, 8608)), True),
        ("5600x5600 (30,625 patches)", _data_url(_png(5600, 5600)), True),
        ("5504x5504 (29,584 patches)", _data_url(_png(5504, 5504)), False),
        ("1024x1024", _data_url(_png(1024, 1024)), False),
        (
            "an 8000x8000 JPEG whose frame header sits behind 100 KiB of metadata",
            _data_url(_jpeg(8000, 8000, 100 * 1024), "image/jpeg"),
            True,
        ),
        ("a 131070x500 strip, once scaled to the 65,535-pixel side", _data_url(_png(131070, 500)), False),
        ("a 131070x1500 strip, still over the limit once scaled", _data_url(_png(131070, 1500)), True),
        ("an http URL", "https://example.com/huge.png", False),
        ("a data URL that is not base64", "data:image/png," + quote("not base64"), False),
        ("a data URL of something that is not an image", _data_url(b"hello"), False),
    ],
    ids=lambda value: value if isinstance(value, str) and not value.startswith(("data:", "http")) else "",
)
def test_exceeds_openai_patch_limit(name: str, url: str, expected: bool):
    assert exceeds_openai_patch_limit(url) is expected


@dataclass
class ImageDetailCase:
    expected_client: str
    model: str
    client_type: str | None
    protocol: str
    # whether an image over the patch limit goes out at high detail
    shrinks: bool


IMAGE_DETAIL_CASES = [
    ImageDetailCase("GPT5_6Client", "gpt-5.6-terra", None, "responses", True),
    ImageDetailCase("GPT5_6Client", "gpt-5.5", None, "responses", False),
    ImageDetailCase("OpenaiResponsesClient", "openai/gpt-5.6-terra", "openai-responses", "responses", True),
    ImageDetailCase("OpenaiResponsesClient", "deepseek-v4-flash-vision-exp", "openai-responses", "responses", False),
    ImageDetailCase("OpenaiChatClient", "GPT-5.6-Sol", "openai-chat", "chat", True),
    ImageDetailCase("OpenaiChatClient", "gpt-5.5", "openai-chat", "chat", False),
]

OVERSIZED = _data_url(_png(6400, 8608))
SMALL = _data_url(_png(1024, 1024))

# A prompt carrying an oversized image next to a small one, and a tool result carrying both again.
MESSAGES: list[dict[str, Any]] = [
    {
        "role": "user",
        "content_items": [
            {"type": "text", "text": "What is in these?"},
            {"type": "image_url", "image_url": OVERSIZED},
            {"type": "image_url", "image_url": SMALL},
        ],
    },
    {
        "role": "assistant",
        "content_items": [
            {"type": "tool_call", "name": "read_image", "arguments": {"path": "shot.png"}, "tool_call_id": "call_1"}
        ],
    },
    {
        "role": "user",
        "content_items": [
            {"type": "tool_result", "text": "image/png", "images": [OVERSIZED, SMALL], "tool_call_id": "call_1"}
        ],
    },
]


def _details(case: ImageDetailCase, model_input: list[dict[str, Any]]) -> list[str | None]:
    """The detail of each image part, in order: the prompt's two images, then the tool result's two."""
    if case.protocol == "responses":
        return [part.get("detail") for part in model_input[0]["content"][1:]] + [
            part.get("detail") for part in model_input[2]["output"][1:]
        ]

    return [part["image_url"].get("detail") for part in model_input[0]["content"][1:]] + [
        part["image_url"].get("detail") for part in model_input[2]["content"][1:]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    IMAGE_DETAIL_CASES,
    ids=[f"{case.model}:{case.client_type or 'auto'}" for case in IMAGE_DETAIL_CASES],
)
async def test_image_over_the_patch_limit_goes_out_at_high_detail_on_gpt_5_6(case: ImageDetailCase):
    client = AutoLLMClient(model=case.model, api_key="test-key", client_type=case.client_type)
    assert client._client.__class__.__name__ == case.expected_client  # noqa: SLF001

    model_input = client._client.transform_uni_message_to_model_input(MESSAGES)  # noqa: SLF001
    if inspect.isawaitable(model_input):
        model_input = await model_input

    shrunk = "high" if case.shrinks else None
    assert _details(case, model_input) == [shrunk, None, shrunk, None]
