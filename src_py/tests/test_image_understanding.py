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

import os

import pytest

from agent_adapter import AutoLLMClient


# Skip tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
    reason="No Gemini API key available",
)


@pytest.mark.asyncio
async def test_image_understanding_basic():
    """Test basic image understanding with a URL."""
    client = AutoLLMClient(model="gemini-3-flash-preview")
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Narcissus_poeticus_subsp._radiiflorus.1658.jpg/500px-Narcissus_poeticus_subsp._radiiflorus.1658.jpg"

    messages = [
        {
            "role": "user",
            "content_items": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": image_url},
            ],
        }
    ]
    config = {}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0
    # Check that we got some text response
    has_text = False
    for event in events:
        for item in event.get("content_items", []):
            if item.get("type") == "text" and item.get("text"):
                has_text = True
                break
        if has_text:
            break

    assert has_text, "Expected text response from image understanding"


@pytest.mark.asyncio
async def test_image_understanding_with_specific_question():
    """Test image understanding with a specific question about the image."""
    client = AutoLLMClient(model="gemini-3-flash-preview")
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Narcissus_poeticus_subsp._radiiflorus.1658.jpg/500px-Narcissus_poeticus_subsp._radiiflorus.1658.jpg"

    messages = [
        {
            "role": "user",
            "content_items": [
                {"type": "text", "text": "What type of flower is in this image?"},
                {"type": "image_url", "image_url": image_url},
            ],
        }
    ]
    config = {"temperature": 0.7}

    events = []
    async for event in client.streaming_response(messages=messages, config=config):
        events.append(event)

    assert len(events) > 0
    # Verify we got a response
    has_text_response = any(
        item.get("type") == "text" and item.get("text") for event in events for item in event.get("content_items", [])
    )
    assert has_text_response, "Expected text response about the flower"


@pytest.mark.asyncio
async def test_image_understanding_stateful():
    """Test stateful image understanding with follow-up questions."""
    client = AutoLLMClient(model="gemini-3-flash-preview")
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Narcissus_poeticus_subsp._radiiflorus.1658.jpg/500px-Narcissus_poeticus_subsp._radiiflorus.1658.jpg"

    config = {}

    # First message with image
    events1 = []
    async for event in client.streaming_response_stateful(
        message={
            "role": "user",
            "content_items": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": image_url},
            ],
        },
        config=config,
    ):
        events1.append(event)

    assert len(events1) > 0
    assert len(client.get_history()) == 2  # User message + assistant response

    # Follow-up question without the image (should remember context)
    events2 = []
    async for event in client.streaming_response_stateful(
        message={"role": "user", "content_items": [{"type": "text", "text": "What color is it?"}]}, config=config
    ):
        events2.append(event)

    assert len(events2) > 0
    assert len(client.get_history()) == 4  # 2 previous + 2 new
