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
import concurrent.futures
import json
import threading
from typing import Any

from flask import Flask, Response, jsonify, render_template_string, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from .. import AutoLLMClient
from ..abort_signal import AbortSignal
from .tracer import Tracer


# Global event loop and lock for thread-safe async operations
_event_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()
_session_clients: dict[str, AutoLLMClient] = {}
_session_client_options: dict[str, tuple[str, str | None, str | None]] = {}
_session_abort_signals: dict[str, AbortSignal] = {}


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


def _normalize_optional_string(value: Any) -> str | None:
    """Return a trimmed string, or None for empty/non-string values."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _get_client_options(config: dict[str, Any]) -> tuple[str, str | None, str | None]:
    """Extract client construction options from playground config."""
    model = _normalize_optional_string(config.get("model")) or "gpt-5.5"
    api_key = _normalize_optional_string(config.get("api_key"))
    base_url = _normalize_optional_string(config.get("base_url"))
    return model, api_key, base_url


def _get_request_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove client construction options before sending request config."""
    request_config = dict(config)
    request_config.pop("model", None)
    request_config.pop("api_key", None)
    request_config.pop("base_url", None)
    return request_config


def create_chat_app() -> Flask:
    """
    Create a Flask web application for chatting with LLMs.

    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

    @app.errorhandler(RequestEntityTooLarge)
    def request_entity_too_large(_error: RequestEntityTooLarge) -> tuple[Response, int]:
        """Return a JSON error when uploaded images make the request too large."""
        return jsonify({"error": "Request body is too large. Please upload fewer or smaller images."}), 413

    # HTML template for the chat UI
    CHAT_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Playground</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            [data-combobox-menu] [data-combobox-option] {
                font-size: 0.875rem;
                line-height: 1.25rem;
            }

            [data-combobox-menu] [data-combobox-option] span {
                font-size: inherit;
                line-height: inherit;
            }

            [data-combobox-menu] [data-combobox-option]::after {
                content: attr(data-description);
                display: block;
                margin-top: 0.125rem;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                font-size: 0.75rem;
                line-height: 1rem;
                color: rgb(107 114 128);
            }
        </style>
    </head>
    <body class="bg-gray-50 flex flex-col h-screen">
        <div class="bg-gray-900 text-white px-6 py-4 border-b border-gray-700 flex justify-between items-center">
            <h1 class="text-xl font-semibold">AgentHub</h1>
            <div class="flex items-center gap-4">
                <a href="https://github.com/Prism-Shadow/AgentHub" target="_blank" rel="noopener noreferrer" class="text-gray-400 hover:text-white text-sm transition-colors">GitHub</a>
                <a href="/tracer/" target="_blank" rel="noopener noreferrer" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm transition-colors">Open Tracer</a>
                <button class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md text-sm transition-colors" onclick="toggleConfig()">
                    ⚙️ Config
                </button>
            </div>
        </div>

        <div class="bg-white border-b border-gray-200 px-6 py-4" id="configPanel">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="modelComboboxButton">Model</label>
                    <div id="modelCombobox" class="relative" data-combobox>
                        <input id="modelSelect" type="hidden" value="gpt-5.5" data-combobox-value>
                        <button
                            id="modelComboboxButton"
                            type="button"
                            role="combobox"
                            aria-controls="modelComboboxMenu"
                            aria-expanded="false"
                            class="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-left shadow-sm transition hover:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            onclick="toggleCombobox('modelCombobox')"
                            onkeydown="handleComboboxKeydown(event, 'modelCombobox')"
                            data-combobox-button
                        >
                            <span class="flex items-center justify-between gap-3">
                                <span class="min-w-0">
                                    <span class="block truncate text-sm font-medium text-gray-900" data-combobox-label>GPT 5.5</span>
                                </span>
                                <svg class="h-4 w-4 flex-none text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                    <path d="m6 9 6 6 6-6"></path>
                                </svg>
                            </span>
                        </button>
                        <div id="modelComboboxMenu" class="hidden absolute z-30 mt-2 max-h-72 w-full overflow-y-auto rounded-md border border-gray-200 bg-white py-1 shadow-lg" role="listbox" aria-labelledby="modelComboboxButton" data-combobox-menu>
                            <button type="button" role="option" aria-selected="true" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none bg-blue-50" data-combobox-option data-value="gpt-5.5" data-label="GPT 5.5" data-description="gpt-5.5" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">GPT 5.5</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="gemini-3.5-flash" data-label="Gemini 3.5 Flash" data-description="gemini-3.5-flash" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">Gemini 3.5 Flash</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="gemini-embedding-2" data-label="Gemini Embedding 2" data-description="gemini-embedding-2" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">Gemini Embedding 2</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="claude-opus-4-7" data-label="Claude Opus 4.7" data-description="claude-opus-4-7" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">Claude Opus 4.7</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="claude-sonnet-4-6" data-label="Claude Sonnet 4.6" data-description="claude-sonnet-4-6" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">Claude Sonnet 4.6</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="kimi-k2.6" data-label="Kimi K2.6" data-description="kimi-k2.6" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">Kimi K2.6</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="glm-5.1" data-label="GLM 5.1" data-description="glm-5.1" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">GLM 5.1</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="qwen/qwen3.6-35b-a3b" data-label="Qwen3.6 35B" data-description="qwen/qwen3.6-35b-a3b" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">Qwen3.6 35B</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="gemini-3.1-flash-image-preview" data-label="Gemini 3.1 Flash Image" data-description="gemini-3.1-flash-image-preview" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">Gemini 3.1 Flash Image</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="gemini-3.1-flash-tts-preview" data-label="Gemini 3.1 Flash TTS" data-description="gemini-3.1-flash-tts-preview" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">Gemini 3.1 Flash TTS</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="deepseek-v4-pro" data-label="DeepSeek V4 Pro" data-description="deepseek-v4-pro" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">DeepSeek V4 Pro</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="deepseek-v4-flash" data-label="DeepSeek V4 Flash" data-description="deepseek-v4-flash" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">DeepSeek V4 Flash</span>
                            </button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="__custom__" data-label="Custom model" data-description="Enter a model id" onclick="selectComboboxOption('modelCombobox', this)">
                                <span class="block truncate text-sm font-medium text-gray-900">Custom model</span>
                            </button>
                        </div>
                    </div>
                    <div id="customModelWrapper" class="hidden mt-2">
                        <input
                            id="customModelInput"
                            type="text"
                            autocomplete="off"
                            placeholder="Custom model id"
                            class="w-full px-3 py-2 border border-gray-300 rounded-md text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                        />
                    </div>
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="apiKeyInput">API Key</label>
                    <div class="relative">
                        <input type="password" id="apiKeyInput" autocomplete="off" placeholder="Use environment variable when empty" class="w-full px-3 py-2 pr-12 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                        <button type="button" id="apiKeyVisibilityToggle" aria-label="Show API key" title="Show API key" class="absolute inset-y-0 right-1 my-1 px-3 text-gray-500 hover:text-gray-900 rounded focus:outline-none focus:ring-2 focus:ring-blue-500" onclick="toggleApiKeyVisibility()">
                            <svg id="apiKeyVisibilityShowIcon" class="hidden" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"></path>
                                <circle cx="12" cy="12" r="3"></circle>
                            </svg>
                            <svg id="apiKeyVisibilityHideIcon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                <path d="M10.7 5.1A10.9 10.9 0 0 1 12 5c6.5 0 10 7 10 7a18.5 18.5 0 0 1-3.3 4.3"></path>
                                <path d="M6.6 6.6C3.8 8.4 2 12 2 12s3.5 7 10 7a10.9 10.9 0 0 0 5.4-1.4"></path>
                                <path d="M9.9 9.9A3 3 0 0 0 14.1 14.1"></path>
                                <path d="M3 3l18 18"></path>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="baseUrlInput">Base URL</label>
                    <input type="url" id="baseUrlInput" placeholder="Use provider default when empty" class="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500">
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="thinkingLevelComboboxButton">Thinking Level</label>
                    <div id="thinkingLevelCombobox" class="relative" data-combobox>
                        <input id="thinkingLevelSelect" type="hidden" value="" data-combobox-value>
                        <button id="thinkingLevelComboboxButton" type="button" role="combobox" aria-controls="thinkingLevelComboboxMenu" aria-expanded="false" class="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-left shadow-sm transition hover:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500" onclick="toggleCombobox('thinkingLevelCombobox')" onkeydown="handleComboboxKeydown(event, 'thinkingLevelCombobox')" data-combobox-button>
                            <span class="flex items-center justify-between gap-3">
                                <span class="min-w-0">
                                    <span class="block truncate text-sm font-medium text-gray-900" data-combobox-label>Unspecified</span>
                                </span>
                                <svg class="h-4 w-4 flex-none text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                    <path d="m6 9 6 6 6-6"></path>
                                </svg>
                            </span>
                        </button>
                        <div id="thinkingLevelComboboxMenu" class="hidden absolute z-30 mt-2 max-h-72 w-full overflow-y-auto rounded-md border border-gray-200 bg-white py-1 shadow-lg" role="listbox" aria-labelledby="thinkingLevelComboboxButton" data-combobox-menu>
                            <button type="button" role="option" aria-selected="true" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none bg-blue-50" data-combobox-option data-value="" data-label="Unspecified" data-description="Provider default" onclick="selectComboboxOption('thinkingLevelCombobox', this)">Unspecified</button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="none" data-label="None" data-description="Disable thinking" onclick="selectComboboxOption('thinkingLevelCombobox', this)">None</button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="low" data-label="Low" data-description="Low thinking budget" onclick="selectComboboxOption('thinkingLevelCombobox', this)">Low</button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="medium" data-label="Medium" data-description="Medium thinking budget" onclick="selectComboboxOption('thinkingLevelCombobox', this)">Medium</button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="high" data-label="High" data-description="High thinking budget" onclick="selectComboboxOption('thinkingLevelCombobox', this)">High</button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="xhigh" data-label="XHigh" data-description="Maximum thinking budget" onclick="selectComboboxOption('thinkingLevelCombobox', this)">XHigh</button>
                        </div>
                    </div>
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="thinkingSummaryComboboxButton">Thinking Summary</label>
                    <div id="thinkingSummaryCombobox" class="relative" data-combobox>
                        <input id="thinkingSummaryCheckbox" type="hidden" value="" data-combobox-value>
                        <button id="thinkingSummaryComboboxButton" type="button" role="combobox" aria-controls="thinkingSummaryComboboxMenu" aria-expanded="false" class="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-left shadow-sm transition hover:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500" onclick="toggleCombobox('thinkingSummaryCombobox')" onkeydown="handleComboboxKeydown(event, 'thinkingSummaryCombobox')" data-combobox-button>
                            <span class="flex items-center justify-between gap-3">
                                <span class="min-w-0">
                                    <span class="block truncate text-sm font-medium text-gray-900" data-combobox-label>Unspecified</span>
                                </span>
                                <svg class="h-4 w-4 flex-none text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                    <path d="m6 9 6 6 6-6"></path>
                                </svg>
                            </span>
                        </button>
                        <div id="thinkingSummaryComboboxMenu" class="hidden absolute z-30 mt-2 max-h-72 w-full overflow-y-auto rounded-md border border-gray-200 bg-white py-1 shadow-lg" role="listbox" aria-labelledby="thinkingSummaryComboboxButton" data-combobox-menu>
                            <button type="button" role="option" aria-selected="true" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none bg-blue-50" data-combobox-option data-value="" data-label="Unspecified" data-description="Provider default" onclick="selectComboboxOption('thinkingSummaryCombobox', this)">Unspecified</button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="true" data-label="True" data-description="Request summaries" onclick="selectComboboxOption('thinkingSummaryCombobox', this)">True</button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="false" data-label="False" data-description="Hide summaries" onclick="selectComboboxOption('thinkingSummaryCombobox', this)">False</button>
                        </div>
                    </div>
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="toolChoiceComboboxButton">Tool Choice</label>
                    <div id="toolChoiceCombobox" class="relative" data-combobox>
                        <input id="toolChoiceSelect" type="hidden" value="" data-combobox-value>
                        <button id="toolChoiceComboboxButton" type="button" role="combobox" aria-controls="toolChoiceComboboxMenu" aria-expanded="false" class="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-left shadow-sm transition hover:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-500" onclick="toggleCombobox('toolChoiceCombobox')" onkeydown="handleComboboxKeydown(event, 'toolChoiceCombobox')" data-combobox-button>
                            <span class="flex items-center justify-between gap-3">
                                <span class="min-w-0">
                                    <span class="block truncate text-sm font-medium text-gray-900" data-combobox-label>Unspecified</span>
                                </span>
                                <svg class="h-4 w-4 flex-none text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                                    <path d="m6 9 6 6 6-6"></path>
                                </svg>
                            </span>
                        </button>
                        <div id="toolChoiceComboboxMenu" class="hidden absolute z-30 mt-2 max-h-72 w-full overflow-y-auto rounded-md border border-gray-200 bg-white py-1 shadow-lg" role="listbox" aria-labelledby="toolChoiceComboboxButton" data-combobox-menu>
                            <button type="button" role="option" aria-selected="true" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none bg-blue-50" data-combobox-option data-value="" data-label="Unspecified" data-description="SDK default" onclick="selectComboboxOption('toolChoiceCombobox', this)">Unspecified</button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="auto" data-label="Auto" data-description="Model may call tools" onclick="selectComboboxOption('toolChoiceCombobox', this)">Auto</button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="required" data-label="Required" data-description="Model must call tools" onclick="selectComboboxOption('toolChoiceCombobox', this)">Required</button>
                            <button type="button" role="option" aria-selected="false" class="w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none" data-combobox-option data-value="none" data-label="None" data-description="Do not call tools" onclick="selectComboboxOption('toolChoiceCombobox', this)">None</button>
                        </div>
                    </div>
                </div>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="systemPromptInput">System Prompt</label>
                    <textarea id="systemPromptInput" rows="2" class="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"></textarea>
                </div>
                <div class="flex flex-col">
                    <label class="text-sm font-semibold text-gray-900 mb-1" for="toolsInput">Tools (JSON Array)</label>
                    <textarea id="toolsInput" rows="3" placeholder='[{"name": "function_name", "description": "...", "parameters": {...}}]' class="px-3 py-2 border border-gray-300 rounded-md text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-y"></textarea>
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
                <textarea id="messageInput" class="flex-1 px-4 py-3 border border-gray-300 rounded-md text-sm resize-none overflow-y-hidden focus:ring-2 focus:ring-blue-500 focus:border-blue-500" placeholder="Type your message here..." rows="1"></textarea>
                <button class="bg-green-600 hover:bg-green-700 disabled:bg-green-300 disabled:cursor-not-allowed text-white px-6 py-3 rounded-md text-sm font-semibold whitespace-nowrap transition-colors" id="sendButton" onclick="sendMessage()">Send</button>
                <button class="hidden bg-orange-600 hover:bg-orange-700 disabled:bg-orange-300 disabled:cursor-not-allowed text-white px-6 py-3 rounded-md text-sm font-semibold whitespace-nowrap transition-colors" id="stopButton" onclick="stopGeneration()" disabled>Stop</button>
                <button class="bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-md text-sm font-semibold transition-colors" onclick="clearChat()">Clear</button>
            </div>
        </div>

        <script>
            let isStreaming = false;
            let sessionId = Math.random().toString(36).substring(7);
            let selectedImages = [];
            let lastMessageTimestamp = null;
            let currentAbortController = null;

            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            function formatTimestamp(ms) {
                if (!ms) return '';
                const d = new Date(ms);
                const pad = n => n.toString().padStart(2, '0');
                return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
            }

            function pcmBase64ToWavDataUrl(pcmBase64, sampleRate = 24000, channels = 1, bitsPerSample = 16) {
                const binary = atob(pcmBase64);
                const pcmBytes = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) {
                    pcmBytes[i] = binary.charCodeAt(i);
                }

                const header = new ArrayBuffer(44);
                const view = new DataView(header);
                const byteRate = sampleRate * channels * bitsPerSample / 8;
                const blockAlign = channels * bitsPerSample / 8;

                const writeString = (offset, value) => {
                    for (let i = 0; i < value.length; i++) {
                        view.setUint8(offset + i, value.charCodeAt(i));
                    }
                };

                writeString(0, 'RIFF');
                view.setUint32(4, 36 + pcmBytes.length, true);
                writeString(8, 'WAVE');
                writeString(12, 'fmt ');
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true);
                view.setUint16(22, channels, true);
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, byteRate, true);
                view.setUint16(32, blockAlign, true);
                view.setUint16(34, bitsPerSample, true);
                writeString(36, 'data');
                view.setUint32(40, pcmBytes.length, true);

                const wavBytes = new Uint8Array(44 + pcmBytes.length);
                wavBytes.set(new Uint8Array(header), 0);
                wavBytes.set(pcmBytes, 44);

                let wavBinary = '';
                const chunkSize = 0x8000;
                for (let i = 0; i < wavBytes.length; i += chunkSize) {
                    wavBinary += String.fromCharCode(...wavBytes.subarray(i, i + chunkSize));
                }
                return `data:audio/wav;base64,${btoa(wavBinary)}`;
            }

            function renderInlineData(item) {
                const mimeType = (item.mime_type || '').toLowerCase();
                if (mimeType.startsWith('image/')) {
                    return `<div class="mb-3"><img src="data:${mimeType || 'image/png'};base64,${item.data}" class="max-w-xs rounded border border-gray-300"></div>`;
                }

                const audioMimeTypes = ['audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/webm', 'audio/flac', 'audio/aac', 'audio/mp4'];
                const isAudio = !mimeType || mimeType === 'application/octet-stream' || mimeType.startsWith('audio/');
                if (!isAudio) {
                    return `<div class="mb-3 rounded border border-gray-300 bg-gray-50 px-3 py-2 text-xs text-gray-600">Inline data: ${escapeHtml(item.mime_type || 'application/octet-stream')}</div>`;
                }

                const playableMimeType = audioMimeTypes.includes(mimeType);
                const audioSrc = playableMimeType
                    ? `data:${mimeType || 'application/octet-stream'};base64,${item.data}`
                    : pcmBase64ToWavDataUrl(item.data);
                return `<div class="mb-3"><audio controls preload="metadata" class="max-w-xs"><source src="${audioSrc}" type="${playableMimeType ? mimeType : 'audio/wav'}"></audio></div>`;
            }

            function renderEmbedding(item) {
                const values = Array.isArray(item.embedding) ? item.embedding.slice(0, 5) : [];
                const preview = escapeHtml(`[${values.join(', ')}]`);
                return `<div class="embedding-content mb-2 rounded-md border-l-4 border-indigo-500 bg-indigo-50 p-3 whitespace-normal"><div class="flex items-start gap-2 text-sm"><strong class="shrink-0 text-gray-900">Embedding:</strong><code class="font-mono text-xs text-gray-800 break-all">${preview}</code></div></div>`;
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

            function closeCombobox(comboboxId) {
                const root = document.getElementById(comboboxId);
                if (!root) {
                    return;
                }
                const menu = root.querySelector('[data-combobox-menu]');
                const button = root.querySelector('[data-combobox-button]');
                menu.classList.add('hidden');
                button.setAttribute('aria-expanded', 'false');
            }

            function closeComboboxes(exceptId) {
                document.querySelectorAll('[data-combobox]').forEach((root) => {
                    if (root.id !== exceptId) {
                        closeCombobox(root.id);
                    }
                });
            }

            function toggleCombobox(comboboxId) {
                const root = document.getElementById(comboboxId);
                const menu = root.querySelector('[data-combobox-menu]');
                const isOpen = !menu.classList.contains('hidden');
                closeComboboxes(comboboxId);
                if (isOpen) {
                    closeCombobox(comboboxId);
                    return;
                }
                menu.classList.remove('hidden');
                root.querySelector('[data-combobox-button]').setAttribute('aria-expanded', 'true');
            }

            function selectComboboxOption(comboboxId, option) {
                const root = document.getElementById(comboboxId);
                root.querySelector('[data-combobox-value]').value = option.dataset.value || '';
                root.querySelector('[data-combobox-label]').textContent = option.dataset.label || 'Unspecified';
                const description = root.querySelector('[data-combobox-description]');
                if (description) {
                    description.textContent = option.dataset.description || option.dataset.value || 'Default';
                }

                root.querySelectorAll('[data-combobox-option]').forEach((item) => {
                    const isSelected = item === option;
                    item.setAttribute('aria-selected', isSelected ? 'true' : 'false');
                    item.classList.toggle('bg-blue-50', isSelected);
                });

                closeCombobox(comboboxId);
                if (comboboxId === 'modelCombobox') {
                    handleModelSelectChange();
                }
            }

            function handleComboboxKeydown(event, comboboxId) {
                if (event.key === 'Enter' || event.key === ' ' || event.key === 'ArrowDown') {
                    event.preventDefault();
                    toggleCombobox(comboboxId);
                } else if (event.key === 'Escape') {
                    closeCombobox(comboboxId);
                }
            }

            document.addEventListener('click', (event) => {
                const target = event.target;
                if (!(target instanceof Element)) {
                    return;
                }
                if (!target.closest('[data-combobox]')) {
                    closeComboboxes();
                }
            });

            function handleModelSelectChange() {
                const useCustom = document.getElementById('modelSelect').value === '__custom__';
                const wrapper = document.getElementById('customModelWrapper');
                wrapper.classList.toggle('hidden', !useCustom);
                if (useCustom) {
                    document.getElementById('customModelInput').focus();
                }
            }

            function getSelectedModel() {
                const modelSelect = document.getElementById('modelSelect');
                if (modelSelect.value === '__custom__') {
                    return document.getElementById('customModelInput').value.trim();
                }
                return modelSelect.value;
            }

            function toggleApiKeyVisibility() {
                const input = document.getElementById('apiKeyInput');
                const toggle = document.getElementById('apiKeyVisibilityToggle');
                const showIcon = document.getElementById('apiKeyVisibilityShowIcon');
                const hideIcon = document.getElementById('apiKeyVisibilityHideIcon');
                const shouldShow = input.type === 'password';

                input.type = shouldShow ? 'text' : 'password';
                toggle.setAttribute('aria-label', shouldShow ? 'Hide API key' : 'Show API key');
                toggle.setAttribute('title', shouldShow ? 'Hide API key' : 'Show API key');
                showIcon.classList.toggle('hidden', !shouldShow);
                hideIcon.classList.toggle('hidden', shouldShow);
            }

            function getConfig() {
                const config = {
                    model: getSelectedModel()
                };

                const apiKey = document.getElementById('apiKeyInput').value.trim();
                if (apiKey) {
                    config.api_key = apiKey;
                }

                const baseUrl = document.getElementById('baseUrlInput').value.trim();
                if (baseUrl) {
                    config.base_url = baseUrl;
                }

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

            function addMessageCard(role, content, metadata = null, images = [], timestamp = null, tookMs = null) {
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
                        <div class="flex items-center gap-2">
                            <span class="msg-took text-xs text-gray-400">${tookMs !== null ? 'Took ' + tookMs + ' ms' : ''}</span>
                            <span class="text-xs text-gray-400 msg-timestamp">${timestamp ? formatTimestamp(timestamp) : ''}</span>
                        </div>
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

            function setStreamingControls(streaming) {
                const sendButton = document.getElementById('sendButton');
                const stopButton = document.getElementById('stopButton');
                sendButton.disabled = streaming;
                stopButton.disabled = !streaming;
                stopButton.classList.toggle('hidden', !streaming);
            }

            function stopGeneration() {
                if (!isStreaming) return;

                fetch('/api/abort', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: sessionId
                    }),
                    keepalive: true
                }).catch(error => {
                    console.error('Error interrupting chat:', error);
                });

                if (currentAbortController) {
                    currentAbortController.abort();
                }
            }

            function markInterrupted(contentDiv) {
                if (!contentDiv.textContent.trim() && contentDiv.children.length === 0) {
                    contentDiv.textContent = 'Interrupted.';
                    return;
                }

                const interruptedDiv = document.createElement('div');
                interruptedDiv.className = 'mt-3 text-xs font-semibold text-orange-600';
                interruptedDiv.textContent = 'Interrupted';
                contentDiv.appendChild(interruptedDiv);
            }

            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const container = document.getElementById('messagesContainer');
                const message = input.value.trim();

                if ((!message && selectedImages.length === 0) || isStreaming) return;

                isStreaming = true;
                currentAbortController = new AbortController();
                setStreamingControls(true);
                input.value = '';
                resizeMessageInput();

                const currentImages = [...selectedImages];
                selectedImages = [];
                updateImagePreview();

                const userSendTime = Date.now();
                const timeSinceLastResponse = lastMessageTimestamp !== null ? userSendTime - lastMessageTimestamp : null;
                addMessageCard('user', message, null, currentImages, userSendTime, timeSinceLastResponse);

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
                        }),
                        signal: currentAbortController.signal
                    });

                    if (!response.ok || !response.body) {
                        let errorMessage = `Request failed with status ${response.status}`;
                        try {
                            const errorPayload = await response.clone().json();
                            if (errorPayload && errorPayload.error) {
                                errorMessage = errorPayload.error;
                            }
                        } catch (e) {
                            const errorText = await response.text();
                            if (errorText) {
                                errorMessage = errorText;
                            }
                        }
                        throw new Error(errorMessage);
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let fullResponse = '';
                    let fullThinking = '';
                    let fullToolName = '';
                    let fullToolArgs = '';
                    let metadata = null;
                    let lastCreatedAt = null;
                    let buffer = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;

                        const chunk = decoder.decode(value);
                        buffer += chunk;
                        if (!buffer.endsWith('\\n\\n')) continue;

                        const lines = buffer.split('\\n');
                        buffer = '';
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
                                                // Always insert thinking before any text content so it appears on top
                                                const textContainer = contentDiv.querySelector('.text-content');
                                                contentDiv.insertBefore(thinkingContainer, textContainer || contentDiv.firstChild);
                                            }
                                            thinkingContainer.textContent = `💭 ${fullThinking}`;
                                        } else if (item.type === 'inline_thinking') {
                                            // Ignore thinking inline data
                                            continue;
                                        } else if (item.type === 'partial_tool_call') {
                                            fullToolName += item.name || '';
                                            fullToolArgs += item.arguments || '';
                                            let toolcallContainer = contentDiv.querySelector('.toolcall-content');
                                            if (!toolcallContainer) {
                                                toolcallContainer = document.createElement('div');
                                                toolcallContainer.className = 'toolcall-content bg-yellow-50 p-3 rounded-md border-l-4 border-yellow-500 mb-2';
                                                contentDiv.appendChild(toolcallContainer);
                                            }
                                            toolcallContainer.innerHTML = `<strong class="text-sm">🛠️ Tool Call:</strong> ${escapeHtml(fullToolName || '...')}<br><div class="mt-1 text-xs whitespace-pre-wrap">${escapeHtml(fullToolArgs || '')}</div>`;
                                        } else if (item.type === 'tool_result') {
                                            const toolResultDiv = document.createElement('div');
                                            toolResultDiv.className = 'bg-green-50 p-3 rounded-md border-l-4 border-green-500 mb-2';
                                            toolResultDiv.innerHTML = `<strong class="text-sm">✅ Tool Result:</strong><br><div class="mt-1 text-xs whitespace-pre-wrap">${escapeHtml(item.text)}</div>`;
                                            contentDiv.appendChild(toolResultDiv);
                                        } else if (item.type === 'inline_data') {
                                            const inlineDataDiv = document.createElement('div');
                                            inlineDataDiv.innerHTML = renderInlineData(item);
                                            if (inlineDataDiv.firstChild) {
                                                contentDiv.appendChild(inlineDataDiv.firstChild);
                                            }
                                        } else if (item.type === 'embedding') {
                                            const embeddingDiv = document.createElement('div');
                                            embeddingDiv.innerHTML = renderEmbedding(item);
                                            if (embeddingDiv.firstChild) {
                                                contentDiv.appendChild(embeddingDiv.firstChild);
                                            }
                                        }
                                    }

                                    container.scrollTop = container.scrollHeight;
                                    if (event.usage_metadata) {
                                        const usage = event.usage_metadata;
                                        const inputTokens = (usage.cached_tokens || 0) + (usage.prompt_tokens || 0);
                                        const outputTokens = (usage.thoughts_tokens || 0) + (usage.response_tokens || 0);
                                        const totalTokens = inputTokens + outputTokens;
                                        metadata = {
                                            cached_tokens: usage.cached_tokens || 0,
                                            prompt_tokens: usage.prompt_tokens || 0,
                                            thoughts_tokens: usage.thoughts_tokens || 0,
                                            response_tokens: usage.response_tokens || 0,
                                            total_tokens: totalTokens
                                        };
                                    }
                                    if (event.finish_reason) {
                                        metadata = metadata || {};
                                        metadata.finish_reason = event.finish_reason;
                                    }
                                    if (event.created_at) {
                                        lastCreatedAt = event.created_at;
                                    }
                                } catch (e) {
                                    console.error('Error parsing event:', e);
                                }
                            }
                        }
                    }

                    if (lastCreatedAt) {
                        const timestampEl = assistantCard.querySelector('.msg-timestamp');
                        if (timestampEl) {
                            timestampEl.textContent = formatTimestamp(lastCreatedAt);
                        }
                    }

                    const endTime = Date.now();
                    const responseTimeMs = endTime - userSendTime;
                    lastMessageTimestamp = endTime;
                    const tookEl = assistantCard.querySelector('.msg-took');
                    if (tookEl) {
                        tookEl.textContent = `Took ${responseTimeMs} ms`;
                    }

                    if (metadata) {
                        let metadataHtml = '<div class="flex justify-end gap-3 mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">';
                        const parts = [];
                        if (metadata.cached_tokens) parts.push(`Cached: ${metadata.cached_tokens}`);
                        if (metadata.prompt_tokens) parts.push(`Prompt: ${metadata.prompt_tokens}`);
                        if (metadata.thoughts_tokens) parts.push(`Thoughts: ${metadata.thoughts_tokens}`);
                        if (metadata.response_tokens) parts.push(`Response: ${metadata.response_tokens}`);
                        if (metadata.total_tokens) parts.push(`Total: ${metadata.total_tokens}`);
                        metadataHtml += `<div class="flex items-center gap-1">📊 ${parts.join(' | ')}</div>`;
                        if (metadata.finish_reason) {
                            metadataHtml += `<div class="flex items-center gap-1">🏁 ${metadata.finish_reason}</div>`;
                        }
                        metadataHtml += '</div>';
                        assistantCard.innerHTML += metadataHtml;
                    }

                } catch (error) {
                    if (error.name === 'AbortError') {
                        markInterrupted(contentDiv);
                    } else {
                        contentDiv.textContent = `Error: ${error.message}`;
                        console.error('Error:', error);
                    }
                    lastMessageTimestamp = Date.now();
                } finally {
                    isStreaming = false;
                    currentAbortController = null;
                    setStreamingControls(false);
                }
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
                        lastMessageTimestamp = null;
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
                if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            const textarea = document.getElementById('messageInput');
            function resizeMessageInput() {
                const maxHeight = 200;
                textarea.style.height = 'auto';
                const nextHeight = Math.min(textarea.scrollHeight, maxHeight);
                textarea.style.height = nextHeight + 'px';
                textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden';
            }
            textarea.addEventListener('input', resizeMessageInput);
            resizeMessageInput();

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
        data = request.json or {}
        message = data.get("message")
        config = data.get("config", {})
        session_id = data.get("session_id", "default")

        if not message:
            return jsonify({"error": "No message provided"}), 400

        def generate():
            """Generate streaming response using the persistent event loop."""
            signal = AbortSignal()
            previous_signal = _session_abort_signals.get(session_id)
            if previous_signal is not None:
                previous_signal.abort("replaced")
            _session_abort_signals[session_id] = signal
            async_gen = None
            loop = _get_event_loop()
            try:
                # Get or create client for this session
                client_options = _get_client_options(config)
                if session_id not in _session_clients or _session_client_options.get(session_id) != client_options:
                    model, api_key, base_url = client_options
                    _session_clients[session_id] = AutoLLMClient(model=model, api_key=api_key, base_url=base_url)
                    _session_client_options[session_id] = client_options

                client = _session_clients[session_id]
                request_config = _get_request_config(config)

                # Create async function to collect events
                async def stream_events():
                    async for event in client.streaming_response_stateful(
                        message=message,
                        config=request_config,
                        signal=signal,
                    ):
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

                yield "data: [DONE]\n\n"
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                yield "data: [DONE]\n\n"
            except GeneratorExit:
                signal.abort("client disconnected")
                if async_gen is not None:
                    try:
                        asyncio.run_coroutine_threadsafe(async_gen.aclose(), loop).result(timeout=1)
                    except Exception:
                        pass
                raise
            except Exception as e:
                error_event = {
                    "role": "assistant",
                    "content_items": [{"type": "text", "text": f"Error: {str(e)}"}],
                    "finish_reason": "error",
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                signal.abort("request ended")
                if _session_abort_signals.get(session_id) is signal:
                    del _session_abort_signals[session_id]

        return Response(generate(), mimetype="text/event-stream")

    @app.route("/api/abort", methods=["POST"])
    def abort() -> Response:
        """Interrupt the active streaming response for a session."""
        data = request.json or {}
        session_id = data.get("session_id", "default")
        signal = _session_abort_signals.get(session_id)
        if signal is None:
            return jsonify({"status": "idle"})

        signal.abort("interrupted")
        return jsonify({"status": "aborted"})

    @app.route("/api/clear", methods=["POST"])
    def clear() -> Response:
        """Clear chat history for a session."""
        data = request.json or {}
        session_id = data.get("session_id", "default")

        signal = _session_abort_signals.pop(session_id, None)
        if signal is not None:
            signal.abort("cleared")

        # Clear the client history if it exists
        if session_id in _session_clients:
            _session_clients[session_id].clear_history()
            del _session_clients[session_id]
            _session_client_options.pop(session_id, None)

        return jsonify({"status": "success"})

    tracer_app = Tracer().create_web_app(base_path="/tracer")
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/tracer": tracer_app.wsgi_app})

    return app


def start_playground_server(host: str = "127.0.0.1", port: int = 25751, debug: bool = False) -> None:
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
    parser.add_argument("--port", type=int, default=25751, help="Port number to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    start_playground_server(host=args.host, port=args.port, debug=args.debug)
