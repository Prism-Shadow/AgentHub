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

import pytest
from flask import Flask

from integration.web_ui import create_chat_app


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
        assert b"AgentHub Chat" in response.data
        assert b"messagesContainer" in response.data
        assert b"messageInput" in response.data


def test_chat_app_api_chat_no_message():
    """Test that API returns error when no message is provided."""
    app = create_chat_app()

    with app.test_client() as client:
        response = client.post("/api/chat", json={})
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "No message provided" in data["error"]
