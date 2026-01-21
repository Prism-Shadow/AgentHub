#!/usr/bin/env python
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
Playground for interacting with LLMs.

This module provides a web interface for chatting with language models,
with support for config editing, streaming responses, and message cards
showing token usage and stop reasons.
"""

import asyncio
import base64
import json
import threading
from typing import Any

from flask import Flask, Response, jsonify, render_template_string, request

from .. import AutoLLMClient


# Global event loop and lock for thread-safe async operations
_event_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()
_session_clients: dict[str, AutoLLMClient] = {}


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create the global event loop for async operations."""
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        with _loop_lock:
            # Double-check after acquiring lock
            if _event_loop is None or _event_loop.is_closed():
                _event_loop = asyncio.new_event_loop()

                # Start the loop in a background thread
                def run_loop():
                    asyncio.set_event_loop(_event_loop)
                    _event_loop.run_forever()

                loop_thread = threading.Thread(target=run_loop, daemon=True)
                loop_thread.start()

    return _event_loop


def _serialize_for_json(obj: Any) -> Any:
    """Recursively serialize objects for JSON, converting bytes to base64."""
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    return obj


def create_chat_app() -> Flask:
    """
    Create a Flask web application for chatting with LLMs.

    Returns:
        Flask application instance
    """
    app = Flask(__name__)

    # HTML template for the chat UI
    CHAT_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Playground</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: #f5f5f5;
                color: #24292f;
                display: flex;
                flex-direction: column;
                height: 100vh;
            }
            .header {
                background-color: #24292f;
                color: white;
                padding: 16px 24px;
                border-bottom: 1px solid #d0d7de;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .header h1 {
                font-size: 20px;
                font-weight: 600;
            }
            .config-toggle {
                background-color: #238636;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
            }
            .config-toggle:hover {
                background-color: #2ea043;
            }
            .config-panel {
                background-color: #fff;
                border-bottom: 1px solid #d0d7de;
                padding: 16px 24px;
                display: none;
            }
            .config-panel.visible {
                display: block;
            }
            .config-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 12px;
            }
            .config-field {
                display: flex;
                flex-direction: column;
            }
            .config-field label {
                font-size: 14px;
                font-weight: 600;
                margin-bottom: 4px;
                color: #24292f;
            }
            .config-field input,
            .config-field select {
                padding: 8px;
                border: 1px solid #d0d7de;
                border-radius: 6px;
                font-size: 14px;
            }
            .messages-container {
                flex: 1;
                overflow-y: auto;
                padding: 24px;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            .message-card {
                background-color: #fff;
                border-radius: 8px;
                border: 1px solid #d0d7de;
                padding: 16px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                max-width: 800px;
                animation: slideIn 0.3s ease-out;
            }
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            .message-card.user {
                align-self: flex-end;
                background-color: #ddf4ff;
                border-color: #54aeff;
            }
            .message-card.assistant {
                align-self: flex-start;
                background-color: #fff;
            }
            .message-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }
            .message-role {
                font-weight: 600;
                font-size: 14px;
                text-transform: uppercase;
            }
            .role-user {
                color: #0969da;
            }
            .role-assistant {
                color: #1a7f37;
            }
            .message-content {
                font-size: 14px;
                line-height: 1.6;
                white-space: pre-wrap;
                word-wrap: break-word;
                margin-bottom: 8px;
            }
            .message-metadata {
                display: flex;
                justify-content: flex-end;
                gap: 12px;
                font-size: 12px;
                color: #656d76;
                padding-top: 8px;
                border-top: 1px solid #f6f8fa;
            }
            .metadata-item {
                display: flex;
                align-items: center;
                gap: 4px;
            }
            .input-container {
                background-color: #fff;
                border-top: 1px solid #d0d7de;
                padding: 16px 24px;
            }
            .input-wrapper {
                display: flex;
                gap: 12px;
                max-width: 1200px;
                margin: 0 auto;
            }
            .input-box {
                flex: 1;
                padding: 12px;
                border: 1px solid #d0d7de;
                border-radius: 6px;
                font-size: 14px;
                font-family: inherit;
                resize: vertical;
                min-height: 44px;
                max-height: 200px;
            }
            .send-button {
                background-color: #238636;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                white-space: nowrap;
            }
            .send-button:hover:not(:disabled) {
                background-color: #2ea043;
            }
            .send-button:disabled {
                background-color: #94d3a2;
                cursor: not-allowed;
            }
            .clear-button {
                background-color: #cf222e;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
            }
            .clear-button:hover {
                background-color: #a40e26;
            }
            .loading {
                display: inline-block;
                width: 12px;
                height: 12px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #1a7f37;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 LLM Playground</h1>
            <button class="config-toggle" onclick="toggleConfig()">⚙️ Config</button>
        </div>

        <div class="config-panel" id="configPanel">
            <div class="config-grid">
                <div class="config-field">
                    <label for="modelSelect">Model</label>
                    <select id="modelSelect">
                        <option value="gemini-3-flash-preview">Gemini 3 Flash</option>
                        <option value="claude-sonnet-4-5-20250929">Claude Sonnet 4.5</option>
                        <option value="gpt-5.2">GPT 5.2</option>
                        <option value="glm-4.7">GLM 4.7</option>
                        <option value="Qwen/Qwen3-0.6B">Qwen3-0.6B</option>
                        <option value="Qwen/Qwen3-8B">Qwen3-8B</option>
                        <option value="Qwen/Qwen3-4B-Instruct-2507">Qwen3-4B-Instruct-2507</option>
                    </select>
                </div>
                <div class="config-field">
                    <label for="temperatureInput">Temperature</label>
                    <input type="number" id="temperatureInput" min="0" max="2" step="0.1" value="1.0">
                </div>
                <div class="config-field">
                    <label for="maxTokensInput">Max Tokens</label>
                    <input type="number" id="maxTokensInput" min="1" max="100000" step="1" value="4096">
                </div>
                <div class="config-field">
                    <label for="thinkingLevelSelect">Thinking Level</label>
                    <select id="thinkingLevelSelect">
                        <option value="">Unspecified</option>
                        <option value="none">None</option>
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                    </select>
                </div>
                <div class="config-field">
                    <label for="thinkingSummaryCheckbox">Thinking Summary</label>
                    <select id="thinkingSummaryCheckbox">
                        <option value="">Unspecified</option>
                        <option value="true">True</option>
                        <option value="false">False</option>
                    </select>
                </div>
                <div class="config-field">
                    <label for="toolChoiceSelect">Tool Choice</label>
                    <select id="toolChoiceSelect">
                        <option value="">Unspecified</option>
                        <option value="auto">Auto</option>
                        <option value="required">Required</option>
                        <option value="none">None</option>
                    </select>
                </div>
            </div>
            <div class="config-grid">
                <div class="config-field">
                    <label for="systemPromptInput">System Prompt</label>
                    <textarea id="systemPromptInput" rows="2" style="width: 100%; padding: 8px; border: 1px solid #d0d7de; border-radius: 6px; font-size: 14px;"></textarea>
                </div>
                <div class="config-field">
                    <label for="toolsInput">Tools (JSON Array)</label>
                    <textarea id="toolsInput" rows="3" placeholder='[{"name": "function_name", "description": "...", "parameters": {...}}]' style="width: 100%; padding: 8px; border: 1px solid #d0d7de; border-radius: 6px; font-size: 14px; font-family: monospace;"></textarea>
                </div>
                <div class="config-field">
                    <label for="traceIdInput">Trace ID</label>
                    <input type="text" id="traceIdInput" placeholder="e.g., session_001" style="width: 100%; padding: 8px; border: 1px solid #d0d7de; border-radius: 6px; font-size: 14px;">
                </div>
            </div>
        </div>

        <div class="messages-container" id="messagesContainer">
            <div style="text-align: center; color: #656d76; padding: 40px;">
                <h2>Start a conversation</h2>
                <p>Type your message below to begin chatting with the AI.</p>
            </div>
        </div>

        <div class="input-container">
            <div class="input-wrapper">
                <textarea id="messageInput" class="input-box" placeholder="Type your message here..." rows="1"></textarea>
                <button class="send-button" id="sendButton" onclick="sendMessage()">Send</button>
                <button class="clear-button" onclick="clearChat()">Clear</button>
            </div>
        </div>

        <script>
            let isStreaming = false;
            let sessionId = Math.random().toString(36).substring(7);

            function toggleConfig() {
                const panel = document.getElementById('configPanel');
                panel.classList.toggle('visible');
            }

            function getConfig() {
                const config = {
                    model: document.getElementById('modelSelect').value,
                    temperature: parseFloat(document.getElementById('temperatureInput').value),
                    max_tokens: parseInt(document.getElementById('maxTokensInput').value)
                };

                const thinkingLevel = document.getElementById('thinkingLevelSelect').value;
                if (thinkingLevel) {
                    config.thinking_level = thinkingLevel;
                }

                const thinkingSummary = document.getElementById('thinkingSummaryCheckbox').value;
                if (thinkingSummary) {
                    config.thinking_summary = JSON.parse(thinkingSummary);
                }

                const toolChoice = document.getElementById('toolChoiceSelect').value;
                if (toolChoice && toolChoice !== 'auto') {
                    config.tool_choice = toolChoice;
                }

                const systemPrompt = document.getElementById('systemPromptInput').value.trim();
                if (systemPrompt) {
                    config.system_prompt = systemPrompt;
                }

                const toolsInput = document.getElementById('toolsInput').value.trim();
                if (toolsInput) {
                    try {
                        config.tools = JSON.parse(toolsInput);
                    } catch (e) {
                        console.error('Invalid JSON in tools field:', e);
                        alert('Invalid JSON format in Tools field. Please check your syntax.');
                    }
                }

                const traceId = document.getElementById('traceIdInput').value.trim();
                if (traceId) {
                    config.trace_id = traceId;
                }

                return config;
            }

            function addMessageCard(role, content, metadata = null) {
                const container = document.getElementById('messagesContainer');

                // Remove welcome message if it exists
                if (container.children.length === 1 && container.children[0].style.textAlign === 'center') {
                    container.innerHTML = '';
                }

                const card = document.createElement('div');
                card.className = `message-card ${role}`;

                let html = `
                    <div class="message-header">
                        <span class="message-role role-${role}">${role}</span>
                    </div>
                    <div class="message-content">${content || ''}</div>
                `;

                if (metadata) {
                    html += '<div class="message-metadata">';
                    if (metadata.tokens) {
                        html += `<div class="metadata-item">📊 ${metadata.tokens} tokens</div>`;
                    }
                    if (metadata.finish_reason) {
                        html += `<div class="metadata-item">🏁 ${metadata.finish_reason}</div>`;
                    }
                    html += '</div>';
                }

                card.innerHTML = html;
                container.appendChild(card);
                container.scrollTop = container.scrollHeight;

                return card;
            }

            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');
                const message = input.value.trim();

                if (!message || isStreaming) return;

                isStreaming = true;
                sendButton.disabled = true;
                input.value = '';

                // Add user message to UI
                addMessageCard('user', message);

                // Create assistant card for streaming
                const assistantCard = addMessageCard('assistant', '');
                const contentDiv = assistantCard.querySelector('.message-content');

                try {
                    const config = getConfig();
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: {
                                role: 'user',
                                content_items: [{ type: 'text', text: message }]
                            },
                            config: config,
                            session_id: sessionId
                        })
                    });

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let fullResponse = '';
                    let fullThinking = '';
                    let metadata = null;

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\\n');

                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = line.slice(6);
                                if (data === '[DONE]') continue;

                                try {
                                    const event = JSON.parse(data);

                                    // Handle all content types
                                    for (const item of event.content_items || []) {
                                        if (item.type === 'text') {
                                            fullResponse += item.text;
                                            // Find or create text container
                                            let textContainer = contentDiv.querySelector('.text-content');
                                            if (!textContainer) {
                                                textContainer = document.createElement('div');
                                                textContainer.className = 'text-content';
                                                contentDiv.appendChild(textContainer);
                                            }
                                            textContainer.textContent = fullResponse;
                                        } else if (item.type === 'thinking') {
                                            fullThinking += item.thinking;
                                            // Find or create thinking container
                                            let thinkingContainer = contentDiv.querySelector('.thinking-content');
                                            if (!thinkingContainer) {
                                                thinkingContainer = document.createElement('div');
                                                thinkingContainer.className = 'thinking-content';
                                                thinkingContainer.style.cssText = 'background-color: #ddf4ff; padding: 12px; border-radius: 4px; border-left: 3px solid #0969da; margin-bottom: 8px; font-style: italic;';
                                                contentDiv.appendChild(thinkingContainer);
                                            }
                                            thinkingContainer.textContent = `💭 ${fullThinking}`;
                                        } else if (item.type === 'partial_tool_call') {
                                            // Playground shows streaming tool calls (partial_tool_call)
                                            const toolCallDiv = document.createElement('div');
                                            toolCallDiv.style.cssText = 'background-color: #fff8c5; padding: 12px; border-radius: 4px; border-left: 3px solid #d4a72c; margin-bottom: 8px;';
                                            toolCallDiv.innerHTML = `<strong>🛠️ Tool Call:</strong> ${item.name || '...'}<br><pre style="margin: 4px 0 0 0; font-size: 12px;">${item.arguments || ''}</pre>`;
                                            contentDiv.appendChild(toolCallDiv);
                                        } else if (item.type === 'tool_call') {
                                            // Skip complete tool_call - playground only shows streaming (partial) tool calls
                                            continue;
                                        } else if (item.type === 'tool_result') {
                                            const toolResultDiv = document.createElement('div');
                                            toolResultDiv.style.cssText = 'background-color: #d1f0e8; padding: 12px; border-radius: 4px; border-left: 3px solid #1a7f37; margin-bottom: 8px;';
                                            toolResultDiv.innerHTML = `<strong>✅ Tool Result:</strong><br><pre style="margin: 4px 0 0 0; font-size: 12px;">${item.result}</pre>`;
                                            contentDiv.appendChild(toolResultDiv);
                                        }
                                    }

                                    // Store detailed metadata
                                    if (event.usage_metadata) {
                                        const usage = event.usage_metadata;
                                        metadata = {
                                            prompt_tokens: usage.prompt_tokens || 0,
                                            thoughts_tokens: usage.thoughts_tokens || 0,
                                            response_tokens: usage.response_tokens || 0,
                                            total_tokens: (usage.prompt_tokens || 0) + (usage.thoughts_tokens || 0) + (usage.response_tokens || 0)
                                        };
                                    }
                                    if (event.finish_reason) {
                                        metadata = metadata || {};
                                        metadata.finish_reason = event.finish_reason;
                                    }
                                } catch (e) {
                                    console.error('Error parsing event:', e);
                                }
                            }
                        }
                    }

                    // Update card with metadata
                    if (metadata) {
                        let metadataHtml = '<div class="message-metadata">';
                        if (metadata.prompt_tokens || metadata.thoughts_tokens || metadata.response_tokens) {
                            const parts = [];
                            if (metadata.prompt_tokens) parts.push(`Prompt: ${metadata.prompt_tokens}`);
                            if (metadata.thoughts_tokens) parts.push(`Thoughts: ${metadata.thoughts_tokens}`);
                            if (metadata.response_tokens) parts.push(`Response: ${metadata.response_tokens}`);
                            if (metadata.total_tokens) parts.push(`Total: ${metadata.total_tokens}`);
                            metadataHtml += `<div class="metadata-item">📊 ${parts.join(' | ')}</div>`;
                        }
                        if (metadata.finish_reason) {
                            metadataHtml += `<div class="metadata-item">🏁 ${metadata.finish_reason}</div>`;
                        }
                        metadataHtml += '</div>';
                        assistantCard.innerHTML += metadataHtml;
                    }


                } catch (error) {
                    contentDiv.textContent = `Error: ${error.message}`;
                    console.error('Error:', error);
                }

                isStreaming = false;
                sendButton.disabled = false;
            }

            function clearChat() {
                if (confirm('Are you sure you want to clear the conversation?')) {
                    // Call API to clear server-side history
                    fetch('/api/clear', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            session_id: sessionId
                        })
                    }).then(() => {
                        // Generate new session ID to start fresh
                        sessionId = Math.random().toString(36).substring(7);
                        const container = document.getElementById('messagesContainer');
                        container.innerHTML = `
                            <div style="text-align: center; color: #656d76; padding: 40px;">
                                <h2>Start a conversation</h2>
                                <p>Type your message below to begin chatting with the AI.</p>
                            </div>
                        `;
                    }).catch(error => {
                        console.error('Error clearing chat:', error);
                    });
                }
            }

            // Handle Enter key to send message (Shift+Enter for new line)
            document.getElementById('messageInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            // Auto-resize textarea
            const textarea = document.getElementById('messageInput');
            textarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 200) + 'px';
            });
        </script>
    </body>
    </html>
    """

    @app.route("/")
    def index() -> str:
        """Serve the chat UI."""
        return render_template_string(CHAT_TEMPLATE)

    @app.route("/api/chat", methods=["POST"])
    def chat() -> Response:
        """Handle chat requests with streaming responses."""
        data = request.json
        message = data.get("message")
        config = data.get("config", {})
        session_id = data.get("session_id", "default")

        if not message:
            return jsonify({"error": "No message provided"}), 400

        def generate():
            """Generate streaming response using the persistent event loop."""
            try:
                # Get or create client for this session
                if session_id not in _session_clients:
                    model = config.get("model", "gemini-3-flash-preview")
                    _session_clients[session_id] = AutoLLMClient(model=model)

                client = _session_clients[session_id]

                # Get the persistent event loop
                loop = _get_event_loop()

                # Create async function to collect events
                async def stream_events():
                    async for event in client.streaming_response_stateful(message=message, config=config):
                        # Serialize event to handle bytes objects
                        serialized_event = _serialize_for_json(event)
                        yield f"data: {json.dumps(serialized_event, ensure_ascii=False)}\n\n"

                async_gen = stream_events()
                while True:
                    try:
                        event = asyncio.run_coroutine_threadsafe(async_gen.__anext__(), loop).result()
                        yield event
                    except StopAsyncIteration:
                        break

            except Exception as e:
                error_event = {
                    "role": "assistant",
                    "content_items": [{"type": "text", "text": f"Error: {str(e)}"}],
                    "finish_reason": "error",
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return Response(generate(), mimetype="text/event-stream")

    @app.route("/api/clear", methods=["POST"])
    def clear() -> Response:
        """Clear chat history for a session."""
        data = request.json
        session_id = data.get("session_id", "default")

        # Clear the client history if it exists
        if session_id in _session_clients:
            _session_clients[session_id].clear_history()
            del _session_clients[session_id]

        return jsonify({"status": "success"})

    return app


def start_playground_server(host: str = "127.0.0.1", port: int = 5001, debug: bool = False) -> None:
    """
    Start the playground web server.

    Args:
        host: Host address to bind to
        port: Port number to listen on
        debug: Enable debug mode
    """
    app = create_chat_app()
    print(f"Starting LLM Playground at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the LLM Playground web server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port number to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    start_playground_server(host=args.host, port=args.port, debug=args.debug)
