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

"""
Example demonstrating base64 image understanding capability.

This example shows how to use the AutoLLMClient to analyze images using base64 encoding.
"""

import asyncio
import base64
import os

from agenthub import AutoLLMClient


async def main():
    """Example of base64 image understanding with AutoLLMClient."""
    print("=" * 60)
    print("Base64 Image Understanding Example")
    print("=" * 60)

    # Get model from environment variable, default to gpt-5.2
    model = os.getenv("MODEL", "gpt-5.2")
    print(f"Using model: {model}")

    client = AutoLLMClient(model=model)
    config = {}

    # Read a test image and encode to base64
    image_path = os.path.join(os.path.dirname(__file__), "../../.github/images/agenthub.png")
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Create data URI
    data_uri = f"data:image/png;base64,{base64_image}"

    query = "What's in this image? Please describe what you see."
    print(f"User: {query}")
    print(f"Image: base64 encoded ({len(base64_image)} characters)")
    print("Assistant:")

    async for event in client.streaming_response(
        messages=[
            {
                "role": "user",
                "content_items": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": data_uri},
                ],
            }
        ],
        config=config,
    ):
        for item in event["content_items"]:
            if item["type"] == "text":
                print(item["text"], end="", flush=True)

    print("\n" + "=" * 60)
    print("Image analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
