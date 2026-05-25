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

from flask import Flask

from agenthub.integration import playground
from agenthub.integration.playground import create_chat_app


def test_create_chat_app():
    """Test chat app creation."""
    app = create_chat_app()
    assert app is not None
    assert isinstance(app, Flask)


def test_chat_app_index_route():
    """Test that the index route serves the chat UI."""
    app = create_chat_app()

    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200
        assert b"LLM Playground" in response.data
        assert b'<h1 class="text-xl font-semibold">AgentHub</h1>' in response.data
        assert "🤖 LLM Playground".encode() not in response.data
        assert b"messagesContainer" in response.data
        assert b"messageInput" in response.data
        assert b"apiKeyInput" in response.data
        assert b"apiKeyVisibilityToggle" in response.data
        assert b"toggleApiKeyVisibility()" in response.data
        assert b'id="apiKeyVisibilityShowIcon" class="hidden"' in response.data
        assert b'id="apiKeyVisibilityHideIcon" xmlns=' in response.data
        assert b"baseUrlInput" in response.data
        assert b'href="/tracer/"' in response.data
        assert b'target="_blank"' in response.data
        assert b"Open Tracer" in response.data
        assert response.data.index(b'<h1 class="text-xl font-semibold">') < response.data.index(b">GitHub<")
        assert response.data.index(b">GitHub<") < response.data.index(b">Open Tracer<")
        assert b"temperatureInput" not in response.data
        assert b"maxTokensInput" not in response.data


def test_chat_app_api_chat_no_message():
    """Test that API returns error when no message is provided."""
    app = create_chat_app()

    with app.test_client() as client:
        response = client.post("/api/chat", json={})
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "No message provided" in data["error"]


def test_chat_app_mounts_tracer():
    """Test that the playground app also serves tracer on the same port."""
    app = create_chat_app()

    with app.test_client() as client:
        response = client.get("/tracer/")
        assert response.status_code == 200
        assert b"Tracer" in response.data
        assert b'href="/tracer/"' in response.data


def test_chat_app_uses_client_connection_options(monkeypatch):
    """Test that playground client options do not leak into request config."""
    captured = {}

    class FakeClient:
        def __init__(self, model, api_key=None, base_url=None):
            captured["client_options"] = {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
            }

        async def streaming_response_stateful(self, message, config):
            captured["request_config"] = config
            yield {
                "role": "assistant",
                "event_type": "stop",
                "content_items": [],
                "usage_metadata": None,
                "finish_reason": "stop",
                "created_at": 0,
            }

        def clear_history(self):
            pass

    playground._session_clients.clear()
    playground._session_client_options.clear()
    monkeypatch.setattr(playground, "AutoLLMClient", FakeClient)

    app = create_chat_app()
    with app.test_client() as client:
        response = client.post(
            "/api/chat",
            json={
                "session_id": "connection-options",
                "message": {"role": "user", "content_items": [{"type": "text", "text": "Hello"}]},
                "config": {
                    "model": "gpt-5.5",
                    "api_key": "test-key",
                    "base_url": "https://example.test/v1",
                    "thinking_level": "low",
                },
            },
        )

        assert response.status_code == 200
        assert b"data:" in response.data

    assert captured["client_options"] == {
        "model": "gpt-5.5",
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
    }
    assert captured["request_config"] == {"thinking_level": "low"}


def test_chat_app_accepts_large_image_payload(monkeypatch):
    """Test that playground accepts image payloads above small JSON defaults."""
    captured = {}

    class FakeClient:
        def __init__(self, model, api_key=None, base_url=None):
            pass

        async def streaming_response_stateful(self, message, config):
            captured["message"] = message
            yield {
                "role": "assistant",
                "event_type": "stop",
                "content_items": [],
                "usage_metadata": None,
                "finish_reason": "stop",
                "created_at": 0,
            }

        def clear_history(self):
            pass

    playground._session_clients.clear()
    playground._session_client_options.clear()
    monkeypatch.setattr(playground, "AutoLLMClient", FakeClient)

    large_image = f"data:image/png;base64,{'a' * 150_000}"
    app = create_chat_app()
    with app.test_client() as client:
        response = client.post(
            "/api/chat",
            json={
                "session_id": "large-image",
                "message": {"role": "user", "content_items": [{"type": "image_url", "image_url": large_image}]},
                "config": {"model": "gpt-5.5"},
            },
        )

        assert response.status_code == 200
        assert b"data:" in response.data

    assert captured["message"]["content_items"][0] == {"type": "image_url", "image_url": large_image}
