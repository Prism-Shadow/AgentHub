"""
Example demonstrating streamed Gemini image editing.

This example sends a local cat image plus a text edit instruction, then prints
each streamed event with binary fields redacted so the event structure is easy
to inspect in the terminal.
"""

import asyncio
import base64
import copy
import mimetypes
import os
from pathlib import Path
from typing import Any

from google.genai import types as gemini_types

from agenthub import AutoLLMClient


def _guess_image_extension(mime_type: str) -> str:
    """Guess a file extension from a MIME type."""
    extension = mimetypes.guess_extension(mime_type)
    return extension or ".bin"


def _image_file_to_data_uri(path: Path) -> str:
    """Convert a local image file to a data URI."""
    mime_type, _ = mimetypes.guess_type(path.name)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type or 'image/jpeg'};base64,{encoded}"


def without_binary_data(obj: Any) -> Any:
    """Deep-copy an object and replace bytes values with empty bytes."""

    def _replace_bytes(value: Any) -> Any:
        if isinstance(value, bytes):
            return b"hidden_binary_data"
        if isinstance(value, dict):
            return {key: _replace_bytes(child) for key, child in value.items()}
        if isinstance(value, list):
            return [_replace_bytes(child) for child in value]
        return value

    return _replace_bytes(copy.deepcopy(obj))


def _inline_image_item_as_image(item: dict):
    """Convert an inline_image content item back to a Gemini Part image."""
    part = gemini_types.Part.from_bytes(data=item["data"], mime_type=item["mime_type"])
    return part.as_image()


async def main() -> None:
    print("=" * 60)
    print("Gemini Streamed Image Generation Example")
    print("=" * 60)

    model = os.getenv("MODEL", "gemini-3.1-flash-image-preview")
    client = AutoLLMClient(model=model)
    input_image_path = Path(__file__).with_name("cat.jpg")
    input_image_data_uri = _image_file_to_data_uri(input_image_path)

    prompt = "Create a picture of my cat eating a nano-banana in a fancy restaurant under the Gemini constellation."
    config = {
        "response_modalities": ["TEXT", "IMAGE"],
        "image_config": {"aspect_ratio": "16:9", "image_size": "1K"},
    }

    async for event in client.streaming_response(
        messages=[
            {
                "role": "user",
                "content_items": [
                    {"type": "text", "text": prompt},
                    # edit the input image or generate a new one
                    {"type": "image_url", "image_url": input_image_data_uri},
                ],
            }
        ],
        config=config,
    ):
        print(without_binary_data(event))
        for item in event["content_items"]:
            if item["type"] == "text":
                print(f"Text: {item['text']}")
            elif item["type"] == "inline_image":
                extension = _guess_image_extension(item["mime_type"])
                output_path = Path.cwd() / f"generated_image{extension}"
                image = _inline_image_item_as_image(item)
                if image is None:
                    raise ValueError(f"Failed to decode inline image with mime type {item['mime_type']}")
                image.save(output_path)
                print(f"Saved image chunk to: {output_path}")

        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
