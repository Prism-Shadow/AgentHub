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
Conversation tracer module for saving and viewing conversation history.

This module provides functionality to save conversation history to local files
and serve them via a web interface for real-time monitoring.
"""

import base64
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, Response, render_template_string

from ..types import UniMessage


@dataclass
class Tracer:
    """
    Tracer for saving conversation history to local files.

    This class handles saving conversation history to files in a cache directory
    and provides a web server for browsing and viewing the saved conversations.
    """

    cache_dir: Path = field(default=None, init=True)

    def __post_init__(self) -> None:
        """Initialize cache directory after instance creation."""
        if self.cache_dir is None:
            cache_dir_str = os.getenv("AGENTHUB_CACHE_DIR", "cache")
            self.cache_dir = Path(cache_dir_str).absolute()
        elif isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir).absolute()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _serialize_for_json(self, obj: Any) -> Any:
        """
        Recursively serialize objects for JSON, converting bytes to base64.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("utf-8")
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        return obj

    def save_history(self, model: str, history: list[UniMessage], file_id: str, config: dict[str, Any]) -> None:
        """
        Save conversation history to files.

        Args:
            model: The model name used for this conversation
            history: List of UniMessage objects representing the conversation
            file_id: File identifier without extension (e.g., "agent1/00001")
            config: The UniConfig used for this conversation
        """
        # Create directory if needed
        file_path_base = self.cache_dir / file_id
        file_path_base.parent.mkdir(parents=True, exist_ok=True)

        config_with_model = config.copy()
        config_with_model["model"] = model
        # Save as JSON
        json_path = file_path_base.with_suffix(".json")
        json_data = {
            "history": self._serialize_for_json(history),
            "config": self._serialize_for_json(config_with_model),
            "timestamp": datetime.now().isoformat(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # Save as human-readable text
        txt_path = file_path_base.with_suffix(".txt")
        formatted_content = self._format_history(history, config_with_model)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(formatted_content)

    def _format_history(self, history: list[UniMessage], config: dict[str, Any]) -> str:
        """
        Format conversation history in a readable text format.

        Args:
            history: List of UniMessage objects
            config: The UniConfig used for this conversation

        Returns:
            Formatted string representation of the conversation
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Conversation History - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        # Add config information
        lines.append("Configuration:")
        for key, value in config.items():
            if key != "trace_id":  # Don't include trace_id itself
                if key == "tools" and isinstance(value, list):
                    lines.append(f"  {key}:")
                    lines.append(f"    {json.dumps(value, indent=2, ensure_ascii=False)}")
                else:
                    lines.append(f"  {key}: {value}")
        lines.append("")

        for i, message in enumerate(history, 1):
            role = message["role"].upper()
            lines.append(f"[{i}] {role}:")
            lines.append("-" * 80)

            for item in message["content_items"]:
                if item["type"] == "text":
                    lines.append(f"Text: {item['text']}")
                elif item["type"] == "thinking":
                    lines.append(f"Thinking: {item['thinking']}")
                elif item["type"] == "image_url":
                    lines.append(f"Image URL: {item['image_url']}")
                elif item["type"] == "tool_call":
                    lines.append(f"Tool Call: {item['name']}")
                    lines.append(f"  Arguments: {json.dumps(item['arguments'], indent=2, ensure_ascii=False)}")
                    lines.append(f"  Tool Call ID: {item['tool_call_id']}")
                elif item["type"] == "partial_tool_call":
                    # Skip partial_tool_call - tracer only shows complete tool calls
                    pass
                elif item["type"] == "tool_result":
                    lines.append(f"Tool Result (ID: {item['tool_call_id']}): {item['text']}")
                    if "images" in item and item["images"]:
                        for i, image_url in enumerate(item["images"], 1):
                            lines.append(f"  Image {i}: {image_url}")

            # Add usage metadata if available
            if "usage_metadata" in message and message["usage_metadata"]:
                metadata = message["usage_metadata"]
                lines.append("\nUsage Metadata:")
                if metadata.get("prompt_tokens"):
                    lines.append(f"  Prompt Tokens: {metadata['prompt_tokens']}")
                if metadata.get("thoughts_tokens"):
                    lines.append(f"  Thoughts Tokens: {metadata['thoughts_tokens']}")
                if metadata.get("response_tokens"):
                    lines.append(f"  Response Tokens: {metadata['response_tokens']}")
                if metadata.get("cached_tokens"):
                    lines.append(f"  Cached Tokens: {metadata['cached_tokens']}")

            # Add finish reason if available
            if "finish_reason" in message:
                lines.append(f"\nFinish Reason: {message['finish_reason']}")

            lines.append("")

        return "\n".join(lines)

    def create_web_app(self) -> Flask:
        """
        Create a Flask web application for browsing conversation files.

        Returns:
            Flask application instance
        """
        app = Flask(__name__)
        app.jinja_env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}

        # HTML template for directory listing
        DIRECTORY_TEMPLATE = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tracer</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-50 min-h-screen">
            <div class="max-w-5xl mx-auto p-6">
                <h1 class="text-3xl font-bold text-gray-900 mb-6">Tracer</h1>
                <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
                    <p class="text-sm text-gray-600"><strong>Path:</strong> {{ breadcrumb|safe }}</p>
                </div>
                <div class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
                    {% if items %}
                        {% for item in items %}
                            <div class="border-b border-gray-200 last:border-b-0 hover:bg-gray-50 transition-colors">
                                <a href="{{ item.url }}" class="flex items-center justify-between p-4 text-blue-600 hover:text-blue-800">
                                    <span class="flex items-center">
                                        <span class="mr-2">{% if item.is_dir %}📁{% else %}📄{% endif %}</span>
                                        <span class="text-sm">{{ item.name }}</span>
                                    </span>
                                    {% if item.size %}
                                    <span class="text-xs text-gray-500">{{ item.size }}</span>
                                    {% endif %}
                                </a>
                            </div>
                        {% endfor %}
                    {% else %}
                        <div class="p-8 text-center text-gray-500 italic">No files or directories found.</div>
                    {% endif %}
                </div>
            </div>
        </body>
        </html>
        """

        # HTML template for JSON conversation viewing
        JSON_VIEWER_TEMPLATE = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ filename }}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-50 min-h-screen">
            <div class="max-w-5xl mx-auto p-6">
                <h1 class="text-3xl font-bold text-gray-900 mb-4">{{ filename }}</h1>
                <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
                    <p class="text-sm text-gray-600"><strong>Path:</strong> {{ breadcrumb|safe }}</p>
                </div>
                <a href="{{ back_url }}" class="inline-block mb-6 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-md border border-gray-300 text-sm transition-colors">
                    ← Back to Directory
                </a>
                {% if config %}
                <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
                    <h2 class="text-xl font-semibold text-gray-900 mb-4">Configuration</h2>
                    {% for key, value in config.items() %}
                        {% if key != 'trace_id' %}
                        <div class="py-2 text-sm">
                            <strong class="text-gray-900">{{ key|e }}:</strong>
                            {% if key == 'system_prompt' and value is not none %}
                                <pre style="margin: 4px 0 0 0; padding: 8px; background-color: #f6f8fa; border-radius: 4px; font-size: 12px; overflow-x: auto; white-space: pre-wrap;">{{ value|e }}</pre>
                            {% elif key == 'tools' and value is iterable and value is not string %}
                                <pre style="margin: 4px 0 0 0; padding: 8px; background-color: #f6f8fa; border-radius: 4px; font-size: 12px; overflow-x: auto;">{{ value|tojson(indent=2)|e }}</pre>
                            {% else %}
                                <span class="text-gray-600">{{ value|e }}</span>
                            {% endif %}
                        </div>
                        {% endif %}
                    {% endfor %}
                </div>
                {% endif %}

                {% for msg_idx, message in enumerate(history) %}
                <div class="bg-white rounded-lg shadow-sm border border-gray-200 mb-4 overflow-hidden">
                    <div class="bg-gray-50 border-b border-gray-200 p-4 cursor-pointer hover:bg-gray-100 transition-colors" onclick="toggleMessage({{ msg_idx }})">
                        <div class="flex justify-between items-center">
                            <div>
                                <span class="font-semibold text-sm uppercase {% if message.role == 'user' %}text-blue-600{% else %}text-green-600{% endif %}">{{ message.role }}</span>
                                <span class="text-xs text-gray-500 ml-2">• {{ message.content_items|length }} item(s)</span>
                            </div>
                            <span class="text-gray-400 transform transition-transform" id="icon-{{ msg_idx }}">▶</span>
                        </div>
                    </div>
                    <div class="p-6 hidden" id="content-{{ msg_idx }}">
                        {% for item in message.content_items %}
                            <div class="mb-4 pb-4 border-b border-gray-100 last:border-b-0 last:mb-0 last:pb-0">
                                <div class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">{{ item.type|e }}</div>
                                {% if item.type == 'text' %}
                                    <div class="bg-gray-50 p-4 rounded-md font-mono text-sm whitespace-pre-wrap text-gray-800">{{ item.text|e }}</div>
                                {% elif item.type == 'thinking' %}
                                    <div class="bg-blue-50 p-4 rounded-md border-l-4 border-blue-500 font-mono text-sm whitespace-pre-wrap text-gray-800">{{ item.thinking|e }}</div>
                                {% elif item.type == 'tool_call' %}
                                    <div class="bg-yellow-50 p-4 rounded-md border-l-4 border-yellow-500">
                                        <div class="font-mono text-sm text-gray-800">{{ item.name|e }}({% for key, value in item.arguments.items() %}{{ key|e }}="{{ value|e }}"{% if not loop.last %}, {% endif %}{% endfor %})</div>
                                    </div>
                                {% elif item.type == 'tool_result' %}
                                    <div class="bg-green-50 p-4 rounded-md border-l-4 border-green-500">
                                        <strong class="text-sm text-gray-900">Result:</strong> <span class="text-sm text-gray-700">{{ item.text|e }}</span><br>
                                        <strong class="text-sm text-gray-900">Call ID:</strong> <span class="text-sm text-gray-700">{{ item.tool_call_id|e }}</span>
                                        {% if item.images %}
                                            <div class="mt-2 flex flex-wrap gap-2">
                                                {% for image_url in item.images %}
                                                    <img src="{{ image_url|e }}" class="max-w-xs max-h-48 rounded-md" alt="Tool Result Image">
                                                {% endfor %}
                                            </div>
                                        {% endif %}
                                    </div>
                                {% elif item.type == 'image_url' %}
                                    <div class="bg-gray-50 p-4 rounded-md">
                                        <img src="{{ item.image_url|e }}" class="max-w-xs max-h-48 rounded-md" alt="Preview">
                                    </div>
                                {% endif %}
                            </div>
                        {% endfor %}

                        {% if message.usage_metadata or message.finish_reason %}
                        <div class="mt-4 pt-4 border-t border-gray-200 text-right text-xs text-gray-500">
                            {% if message.usage_metadata %}
                                {% if message.usage_metadata.prompt_tokens %}Prompt: {{ message.usage_metadata.prompt_tokens }} tokens{% endif %}
                                {% if message.usage_metadata.thoughts_tokens %} • Thoughts: {{ message.usage_metadata.thoughts_tokens }} tokens{% endif %}
                                {% if message.usage_metadata.response_tokens %} • Response: {{ message.usage_metadata.response_tokens }} tokens{% endif %}
                                {% if message.usage_metadata.cached_tokens %} • Cached: {{ message.usage_metadata.cached_tokens }} tokens{% endif %}
                            {% endif %}
                            {% if message.finish_reason %}{% if message.usage_metadata %} • {% endif %}Finish: {{ message.finish_reason|e }}{% endif %}
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>
            <script>
                function toggleMessage(idx) {
                    const content = document.getElementById('content-' + idx);
                    const icon = document.getElementById('icon-' + idx);
                    content.classList.toggle('hidden');
                    icon.classList.toggle('rotate-90');
                }
                // Expand all messages by default
                const numMessages = {{ history|length }};
                for (let i = 0; i < numMessages; i++) {
                    toggleMessage(i);
                }
            </script>
        </body>
        </html>
        """

        # HTML template for text file viewing
        TEXT_VIEWER_TEMPLATE = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ filename }}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-50 min-h-screen">
            <div class="max-w-5xl mx-auto p-6">
                <h1 class="text-3xl font-bold text-gray-900 mb-4">{{ filename }}</h1>
                <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
                    <p class="text-sm text-gray-600"><strong>Path:</strong> {{ breadcrumb|safe }}</p>
                </div>
                <a href="{{ back_url|e }}" class="inline-block mb-6 px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-md border border-gray-300 text-sm transition-colors">
                    ← Back to Directory
                </a>
                <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6 overflow-x-auto">
                    <pre class="font-mono text-sm whitespace-pre-wrap text-gray-800">{{ content|e }}</pre>
                </div>
            </div>
        </body>
        </html>
        """

        @app.route("/")
        @app.route("/<path:subpath>")
        def browse(subpath: str = "") -> str | Response:
            """Browse files and directories in the cache folder."""
            full_path = self.cache_dir / subpath
            full_path = full_path.resolve()

            # Security check: ensure path is within cache_dir
            if not str(full_path).startswith(str(self.cache_dir.resolve())):
                return "Access denied", 403

            # If path doesn't exist
            if not full_path.exists():
                return "Path not found", 404

            # If it's a file, display its content
            if full_path.is_file():
                try:
                    # Build breadcrumb
                    parts = subpath.split("/") if subpath else []
                    breadcrumb_parts = ['<a href="/">cache</a>']
                    for i, part in enumerate(parts[:-1]):
                        path_to_part = "/".join(parts[: i + 1])
                        breadcrumb_parts.append(f'<a href="/{path_to_part}">{part}</a>')
                    breadcrumb_parts.append(f"<strong>{parts[-1]}</strong>" if parts else "")
                    breadcrumb = " / ".join(breadcrumb_parts)

                    # Determine back URL
                    back_url = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"

                    # If it's a JSON file, render with the JSON viewer
                    if full_path.suffix == ".json":
                        with open(full_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        return render_template_string(
                            JSON_VIEWER_TEMPLATE,
                            filename=full_path.name,
                            breadcrumb=breadcrumb,
                            back_url=back_url,
                            history=data.get("history", []),
                            config=data.get("config", {}),
                            enumerate=enumerate,
                        )
                    else:
                        # For text files, use simple viewer
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        return render_template_string(
                            TEXT_VIEWER_TEMPLATE,
                            filename=full_path.name,
                            content=content,
                            breadcrumb=breadcrumb,
                            back_url=back_url,
                        )
                except Exception as e:
                    return f"Error reading file: {str(e)}", 500

            # If it's a directory, list its contents
            items = []
            try:
                for entry in sorted(full_path.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
                    # Calculate relative path from cache_dir
                    try:
                        relative_path = entry.resolve().relative_to(self.cache_dir.resolve())
                    except ValueError:
                        # If relative_to fails, skip this entry for security
                        continue
                    item_info: dict[str, Any] = {
                        "name": entry.name,
                        "is_dir": entry.is_dir(),
                        "url": f"/{relative_path}",
                    }
                    if entry.is_file():
                        size = entry.stat().st_size
                        if size < 1024:
                            item_info["size"] = f"{size} B"
                        elif size < 1024 * 1024:
                            item_info["size"] = f"{size / 1024:.1f} KB"
                        else:
                            item_info["size"] = f"{size / (1024 * 1024):.1f} MB"
                    items.append(item_info)
            except Exception as e:
                return f"Error listing directory: {str(e)}", 500

            # Build breadcrumb
            parts = subpath.split("/") if subpath else []
            breadcrumb_parts = ['<a href="/">cache</a>']
            for i, part in enumerate(parts):
                if part:
                    path_to_part = "/".join(parts[: i + 1])
                    breadcrumb_parts.append(f'<a href="/{path_to_part}">{part}</a>')
            breadcrumb = " / ".join(breadcrumb_parts)

            return render_template_string(DIRECTORY_TEMPLATE, items=items, breadcrumb=breadcrumb)

        return app

    def start_web_server(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
        """
        Start the web server for browsing conversation files.

        Args:
            host: Host address to bind to
            port: Port number to listen on
            debug: Enable debug mode
        """
        app = self.create_web_app()
        print(f"Starting tracer web server at http://{host}:{port}")
        print(f"Cache directory: {self.cache_dir.resolve()}")
        app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the Tracer web server for browsing conversation files")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to store conversation history files")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port number to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    tracer = Tracer(cache_dir=args.cache_dir)
    tracer.start_web_server(host=args.host, port=args.port, debug=args.debug)
