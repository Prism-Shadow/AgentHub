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

from agenthub.abort_signal import AbortSignal
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
        assert b"AgentHub Playground" in response.data
        assert b'<h1 class="text-xl font-semibold">AgentHub</h1>' in response.data
        assert b"messagesContainer" in response.data
        assert b"messageInput" in response.data
        assert b'id="modelCombobox"' in response.data
        assert b'id="thinkingLevelCombobox"' in response.data
        assert b'id="thinkingSummaryCombobox"' in response.data
        assert b'id="toolChoiceCombobox"' in response.data
        assert b'data-combobox-option data-value="gpt-5.6-luna"' in response.data
        assert b'data-value="text-embedding-3-large" data-client-type="openai-embedding"' in response.data
        assert b"getSelectedClientType()" in response.data
        assert b"toggleCombobox('modelCombobox')" in response.data
        assert b"selectComboboxOption('modelCombobox', this)" in response.data
        assert b"customModelInput" in response.data
        assert b"handleModelSelectChange()" in response.data
        assert b"modelDropdown" not in response.data
        assert b"toggleModelMenu()" not in response.data
        assert b"<select" not in response.data
        assert b"<datalist" not in response.data
        assert b"apiKeyInput" in response.data
        assert b'id="listModelsButton"' in response.data
        assert b'id="extraHeadersInput"' in response.data
        assert b'id="listModelsError"' in response.data
        assert b"addListedModels(" in response.data
        assert b"getSelectedClientType()" in response.data
        assert b"selectedOptionClientType()" in response.data
        assert b"handleClientTypeInput()" in response.data
        assert b">Connection</span>" in response.data
        assert b">Generation</span>" in response.data
        assert b"getExtraHeaders()" in response.data
        assert b"listModels()" in response.data
        assert b"/api/models" in response.data
        assert b"apiKeyVisibilityToggle" in response.data
        assert b"toggleApiKeyVisibility()" in response.data
        assert b'id="stopButton"' in response.data
        assert b"stopGeneration()" in response.data
        assert b"currentAbortController.abort()" in response.data
        assert b"/api/abort" in response.data
        assert b'id="apiKeyVisibilityShowIcon" class="hidden"' in response.data
        assert b'id="apiKeyVisibilityHideIcon" xmlns=' in response.data
        assert b"baseUrlInput" in response.data
        assert b"renderEmbedding" in response.data
        assert b"item.embedding.slice(0, 5)" in response.data
        assert b"appendAudioChunk(contentDiv, item, audioStream)" in response.data
        assert b"finalizeAudioStream(audioStream)" in response.data
        assert b"renderAudioPlayer(audioStream.mimeType, audioStream.chunks)" in response.data
        assert b"pcmBase64ToWavDataUrl" not in response.data
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


def test_chat_app_lists_the_models_the_endpoint_serves(monkeypatch):
    """Test that the playground lists models through the configured client options."""
    captured = {}

    class FakeClient:
        def __init__(self, model, api_key=None, base_url=None, client_type=None, default_headers=None):
            captured["client_options"] = {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "client_type": client_type,
                "default_headers": default_headers,
            }

        async def list_models(self):
            return ["gpt-5.6", "claude-sonnet-5"]

    monkeypatch.setattr(playground, "AutoLLMClient", FakeClient)

    app = create_chat_app()
    with app.test_client() as client:
        response = client.post(
            "/api/models",
            json={"config": {"model": "gpt-5.6", "api_key": "test-key", "base_url": "https://relay.test/v1"}},
        )

    assert response.status_code == 200
    assert response.get_json() == {"models": ["gpt-5.6", "claude-sonnet-5"]}
    assert captured["client_options"] == {
        "model": "gpt-5.6",
        "api_key": "test-key",
        "base_url": "https://relay.test/v1",
        "client_type": None,
        "default_headers": None,
    }


def test_chat_app_reports_a_failed_model_listing(monkeypatch):
    """Test that a rejected listing reaches the UI as an error rather than an empty list."""

    class FailingClient:
        def __init__(self, model, api_key=None, base_url=None, client_type=None, default_headers=None):
            pass

        async def list_models(self):
            raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(playground, "AutoLLMClient", FailingClient)

    app = create_chat_app()
    with app.test_client() as client:
        response = client.post("/api/models", json={"config": {"model": "gpt-5.6"}})

    assert response.status_code == 400
    assert "401 unauthorized" in response.get_json()["error"]


def test_chat_app_uses_client_connection_options(monkeypatch):
    """Test that playground client options do not leak into request config."""
    captured = {}

    class FakeClient:
        def __init__(self, model, api_key=None, base_url=None, client_type=None, default_headers=None):
            captured["client_options"] = {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "client_type": client_type,
                "default_headers": default_headers,
            }

        async def streaming_response_stateful(self, message, config, signal=None):
            captured["request_config"] = config
            captured["signal"] = signal
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
                    "client_type": "gpt-5.5",
                    "default_headers": {"X-Title": "AgentHub"},
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
        "client_type": "gpt-5.5",
        "default_headers": {"X-Title": "AgentHub"},
    }
    assert captured["request_config"] == {"thinking_level": "low"}
    assert isinstance(captured["signal"], AbortSignal)


def test_chat_app_accepts_large_image_payload(monkeypatch):
    """Test that playground accepts image payloads above small JSON defaults."""
    captured = {}

    class FakeClient:
        def __init__(self, model, api_key=None, base_url=None, client_type=None, default_headers=None):
            pass

        async def streaming_response_stateful(self, message, config, signal=None):
            captured["message"] = message
            captured["signal"] = signal
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
    assert isinstance(captured["signal"], AbortSignal)


def test_chat_app_abort_route_interrupts_active_signal():
    """Test that the abort route interrupts the active request signal."""
    playground._session_abort_signals.clear()
    signal = AbortSignal()
    playground._session_abort_signals["abort-session"] = signal

    app = create_chat_app()
    with app.test_client() as client:
        response = client.post("/api/abort", json={"session_id": "abort-session"})

        assert response.status_code == 200
        assert response.get_json() == {"status": "aborted"}

    assert signal.aborted
    playground._session_abort_signals.clear()
