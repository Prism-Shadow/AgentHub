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
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-50 flex flex-col h-screen">
        <div class="bg-gray-900 text-white px-6 py-4 border-b border-gray-700 flex justify-between items-center">
            <h1 class="text-xl font-semibold">🤖 LLM Playground</h1>
            <button class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md text-sm transition-colors" onclick="toggleConfig()">
                ⚙️ Config
            </button>
        </div>

        <div class="bg-white border-b border-gray-200 px-6 py-4 hidden" id="configPanel">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="modelSelect">Model</label>
                    <input
                        id="modelSelect"
                        list="modelList"
                        placeholder="Select or enter a model name"
                        class="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                    <datalist id="modelList">
                        <option value="gpt-5.2">GPT 5.2</option>
                        <option value="gemini-3-flash-preview">Gemini 3 Flash</option>
                        <option value="claude-sonnet-4-5-20250929">Claude Sonnet 4.5</option>
                        <option value="glm-5">GLM 5</option>
                    </datalist>
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="temperatureInput">Temperature</label>
                    <input type="number" id="temperatureInput" min="0" max="2" step="0.1" value="1.0" class="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="maxTokensInput">Max Tokens</label>
                    <input type="number" id="maxTokensInput" min="1" max="100000" step="1" value="4096" class="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="thinkingLevelSelect">Thinking Level</label>
                    <select id="thinkingLevelSelect" class="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                        <option value="">Unspecified</option>
                        <option value="none">None</option>
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                    </select>
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="thinkingSummaryCheckbox">Thinking Summary</label>
                    <select id="thinkingSummaryCheckbox" class="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                        <option value="">Unspecified</option>
                        <option value="true">True</option>
                        <option value="false">False</option>
                    </select>
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="toolChoiceSelect">Tool Choice</label>
                    <select id="toolChoiceSelect" class="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                        <option value="">Unspecified</option>
                        <option value="auto">Auto</option>
                        <option value="required">Required</option>
                        <option value="none">None</option>
                    </select>
                </div>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="systemPromptInput">System Prompt</label>
                    <textarea id="systemPromptInput" rows="2" class="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"></textarea>
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="toolsInput">Tools (JSON Array)</label>
                    <textarea id="toolsInput" rows="3" placeholder='[{"name": "function_name", "description": "...", "parameters": {...}}]' class="px-3 py-2 border border-gray-300 rounded-md text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"></textarea>
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="traceIdInput">Trace ID</label>
                    <input type="text" id="traceIdInput" placeholder="e.g., session_001" class="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
            </div>
        </div>

        <div class="flex-1 overflow-y-auto px-6 py-6" id="messagesContainer">
            <div class="text-center text-gray-500 py-10">
                <h2 class="text-2xl font-semibold mb-2">Start a conversation</h2>
                <p class="text-sm">Type your message below to begin chatting with the AI.</p>
            </div>
        </div>

        <div class="bg-white border-t border-gray-200 px-6 py-4">
            <div id="imagePreviewContainer" class="mb-3 max-w-5xl mx-auto hidden"></div>
            <div class="flex gap-3 max-w-5xl mx-auto">
                <input type="file" id="imageInput" accept="image/*" multiple class="hidden" onchange="handleImageSelect(event)">
                <button class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-3 rounded-md text-sm font-semibold whitespace-nowrap transition-colors" onclick="document.getElementById('imageInput').click()">📎 Image</button>
                <textarea id="messageInput" class="flex-1 px-4 py-3 border border-gray-300 rounded-md text-sm resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="Type your message here..." rows="1"></textarea>
                <button class="bg-green-600 hover:bg-green-700 disabled:bg-green-300 disabled:cursor-not-allowed text-white px-6 py-3 rounded-md text-sm font-semibold whitespace-nowrap transition-colors" id="sendButton" onclick="sendMessage()">Send</button>
                <button class="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-md text-sm font-semibold transition-colors" onclick="clearChat()">Clear</button>
            </div>
        </div>

        <script>
            let isStreaming = false;
            let sessionId = Math.random().toString(36).substring(7);
            let selectedImages = [];

            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            function handleImageSelect(event) {
                const files = event.target.files;
                if (!files || files.length === 0) return;

                const maxFileSize = 10 * 1024 * 1024;
                const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];

                Array.from(files).forEach(file => {
                    if (!allowedTypes.includes(file.type)) {
                        alert(`File "${file.name}" is not a valid image type. Please upload JPEG, PNG, GIF, or WebP images.`);
                        return;
                    }

                    if (file.size > maxFileSize) {
                        alert(`File "${file.name}" is too large. Maximum file size is 10MB.`);
                        return;
                    }

                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const base64Data = e.target.result;
                        if (typeof base64Data === 'string' && base64Data.startsWith('data:image/')) {
                            selectedImages.push(base64Data);
                            updateImagePreview();
                        }
                    };
                    reader.readAsDataURL(file);
                });

                event.target.value = '';
            }

            function updateImagePreview() {
                const container = document.getElementById('imagePreviewContainer');
                if (selectedImages.length === 0) {
                    container.classList.add('hidden');
                    container.innerHTML = '';
                    return;
                }

                container.classList.remove('hidden');
                container.innerHTML = selectedImages.map((img, idx) => `
                    <div class="inline-block relative mr-2 mb-2">
                        <img src="${img}" class="h-20 w-20 object-cover rounded border border-gray-300">
                        <button onclick="removeImage(${idx})" class="absolute -top-2 -right-2 bg-red-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs hover:bg-red-700">×</button>
                    </div>
                `).join('');
            }

            function removeImage(idx) {
                selectedImages.splice(idx, 1);
                updateImagePreview();
            }

            function toggleConfig() {
                const panel = document.getElementById('configPanel');
                panel.classList.toggle('hidden');
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

            function addMessageCard(role, content, metadata = null, images = []) {
                const container = document.getElementById('messagesContainer');

                if (container.children.length === 1 && container.children[0].className.includes('text-center')) {
                    container.innerHTML = '';
                }

                const card = document.createElement('div');
                const isUser = role === 'user';
                card.className = `max-w-3xl rounded-lg shadow-sm border p-4 mb-4 ${isUser ? 'ml-auto bg-blue-50 border-blue-200' : 'mr-auto bg-white border-gray-200'}`;

                let html = `
                    <div class="flex justify-between items-center mb-3">
                        <span class="font-semibold text-sm uppercase ${isUser ? 'text-blue-600' : 'text-green-600'}">${role}</span>
                    </div>
                `;

                if (images && images.length > 0) {
                    html += '<div class="mb-3 flex flex-wrap gap-2">';
                    images.forEach(img => {
                        html += `<img src="${img}" class="max-w-xs rounded border border-gray-300">`;
                    });
                    html += '</div>';
                }

                html += `<div class="message-content text-sm leading-relaxed whitespace-pre-wrap">${escapeHtml(content || '')}</div>`;

                if (metadata) {
                    html += '<div class="flex justify-end gap-3 mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">';
                    if (metadata.tokens) {
                        html += `<div class="flex items-center gap-1">📊 ${metadata.tokens} tokens</div>`;
                    }
                    if (metadata.finish_reason) {
                        html += `<div class="flex items-center gap-1">🏁 ${metadata.finish_reason}</div>`;
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

                if ((!message && selectedImages.length === 0) || isStreaming) return;

                isStreaming = true;
                sendButton.disabled = true;
                input.value = '';

                const currentImages = [...selectedImages];
                selectedImages = [];
                updateImagePreview();

                addMessageCard('user', message, null, currentImages);

                const assistantCard = addMessageCard('assistant', '');
                const contentDiv = assistantCard.querySelector('.message-content');

                try {
                    const config = getConfig();
                    const content_items = [];

                    if (message) {
                        content_items.push({ type: 'text', text: message });
                    }

                    currentImages.forEach(img => {
                        content_items.push({ type: 'image_url', image_url: img });
                    });

                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: {
                                role: 'user',
                                content_items: content_items
                            },
                            config: config,
                            session_id: sessionId
                        })
                    });

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let fullResponse = '';
                    let fullThinking = '';
                    let fullToolName = '';
                    let fullToolArgs = '';
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

                                    for (const item of event.content_items || []) {
                                        if (item.type === 'text') {
                                            fullResponse += item.text;
                                            let textContainer = contentDiv.querySelector('.text-content');
                                            if (!textContainer) {
                                                textContainer = document.createElement('div');
                                                textContainer.className = 'text-content';
                                                contentDiv.appendChild(textContainer);
                                            }
                                            textContainer.textContent = fullResponse;
                                        } else if (item.type === 'thinking') {
                                            fullThinking += item.thinking;
                                            let thinkingContainer = contentDiv.querySelector('.thinking-content');
                                            if (!thinkingContainer) {
                                                thinkingContainer = document.createElement('div');
                                                thinkingContainer.className = 'thinking-content bg-blue-50 p-3 rounded-md border-l-4 border-blue-500 mb-2 italic';
                                                contentDiv.appendChild(thinkingContainer);
                                            }
                                            thinkingContainer.textContent = `💭 ${fullThinking}`;
                                        } else if (item.type === 'partial_tool_call') {
                                            fullToolName += item.name || '';
                                            fullToolArgs += item.arguments || '';
                                            let toolcallContainer = contentDiv.querySelector('.toolcall-content');
                                            if (!toolcallContainer) {
                                                toolcallContainer = document.createElement('div');
                                                toolcallContainer.className = 'toolcall-content bg-yellow-50 p-3 rounded-md border-l-4 border-yellow-500 mb-2';
                                                contentDiv.appendChild(toolcallContainer);
                                            }
                                            toolcallContainer.innerHTML = `<strong class="text-sm">🛠️ Tool Call:</strong> ${escapeHtml(fullToolName || '...')}<br><pre class="mt-1 text-xs">${escapeHtml(fullToolArgs || '')}</pre>`;
                                        } else if (item.type === 'tool_result') {
                                            const toolResultDiv = document.createElement('div');
                                            toolResultDiv.className = 'bg-green-50 p-3 rounded-md border-l-4 border-green-500 mb-2';
                                            toolResultDiv.innerHTML = `<strong class="text-sm">✅ Tool Result:</strong><br><pre class="mt-1 text-xs">${escapeHtml(item.text)}</pre>`;
                                            contentDiv.appendChild(toolResultDiv);
                                        }
                                    }

                                    if (event.usage_metadata) {
                                        const usage = event.usage_metadata;
                                        metadata = {
                                            prompt_tokens: usage.prompt_tokens || 0,
                                            thoughts_tokens: usage.thoughts_tokens || 0,
                                            response_tokens: usage.response_tokens || 0,
                                            cached_tokens: usage.cached_tokens || 0,
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

                    if (metadata) {
                        let metadataHtml = '<div class="flex justify-end gap-3 mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">';
                        if (metadata.prompt_tokens || metadata.thoughts_tokens || metadata.response_tokens) {
                            const parts = [];
                            if (metadata.prompt_tokens) parts.push(`Prompt: ${metadata.prompt_tokens}`);
                            if (metadata.thoughts_tokens) parts.push(`Thoughts: ${metadata.thoughts_tokens}`);
                            if (metadata.response_tokens) parts.push(`Response: ${metadata.response_tokens}`);
                            if (metadata.cached_tokens) parts.push(`Cached: ${metadata.cached_tokens}`);
                            if (metadata.total_tokens) parts.push(`Total: ${metadata.total_tokens}`);
                            metadataHtml += `<div class="flex items-center gap-1">📊 ${parts.join(' | ')}</div>`;
                        }
                        if (metadata.finish_reason) {
                            metadataHtml += `<div class="flex items-center gap-1">🏁 ${metadata.finish_reason}</div>`;
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
                    fetch('/api/clear', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            session_id: sessionId
                        })
                    }).then(() => {
                        sessionId = Math.random().toString(36).substring(7);
                        const container = document.getElementById('messagesContainer');
                        container.innerHTML = `
                            <div class="text-center text-gray-500 py-10">
                                <h2 class="text-2xl font-semibold mb-2">Start a conversation</h2>
                                <p class="text-sm">Type your message below to begin chatting with the AI.</p>
                            </div>
                        `;
                    }).catch(error => {
                        console.error('Error clearing chat:', error);
                    });
                }
            }

            document.getElementById('messageInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });

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
                    model = config.get("model") or "gpt-5.2"
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
