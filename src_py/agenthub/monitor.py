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
Conversation monitoring module for saving and viewing conversation history.

This module provides functionality to save conversation history to local files
and serve them via a web interface for real-time monitoring.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, Response, render_template_string

from .types import UniMessage


class ConversationMonitor:
    """
    Monitor for saving conversation history to local files.

    This class handles saving conversation history to files in a cache directory
    and provides a web server for browsing and viewing the saved conversations.
    """

    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize the conversation monitor.

        Args:
            cache_dir: Directory to store conversation history files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_history(self, history: list[UniMessage], relative_path: str) -> None:
        """
        Save conversation history to a file.

        Args:
            history: List of UniMessage objects representing the conversation
            relative_path: Relative file path within cache directory (e.g., "agent1/19293.txt")
        """
        file_path = self.cache_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Format history in a human-readable way
        formatted_content = self._format_history(history)

        # Overwrite the file with latest history
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(formatted_content)

    def _format_history(self, history: list[UniMessage]) -> str:
        """
        Format conversation history in a readable text format.

        Args:
            history: List of UniMessage objects

        Returns:
            Formatted string representation of the conversation
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Conversation History - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
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
                    lines.append(f"  Arguments: {json.dumps(item['argument'], indent=2)}")
                    lines.append(f"  Tool Call ID: {item['tool_call_id']}")
                elif item["type"] == "tool_result":
                    lines.append(f"Tool Result (ID: {item['tool_call_id']}): {item['result']}")

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

        # HTML template for directory listing
        DIRECTORY_TEMPLATE = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conversation Monitor</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                h1 {
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }
                .breadcrumb {
                    margin: 20px 0;
                    padding: 10px;
                    background-color: #fff;
                    border-radius: 5px;
                }
                .breadcrumb a {
                    color: #4CAF50;
                    text-decoration: none;
                    margin-right: 5px;
                }
                .breadcrumb a:hover {
                    text-decoration: underline;
                }
                .file-list {
                    background-color: #fff;
                    border-radius: 5px;
                    padding: 20px;
                }
                .file-item, .dir-item {
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 3px;
                    display: flex;
                    align-items: center;
                }
                .file-item:hover, .dir-item:hover {
                    background-color: #f0f0f0;
                }
                .file-item a, .dir-item a {
                    color: #333;
                    text-decoration: none;
                    flex-grow: 1;
                }
                .dir-item a::before {
                    content: "📁 ";
                    margin-right: 5px;
                }
                .file-item a::before {
                    content: "📄 ";
                    margin-right: 5px;
                }
                .file-size {
                    color: #888;
                    font-size: 0.9em;
                    margin-left: 10px;
                }
                .empty {
                    color: #888;
                    font-style: italic;
                    padding: 20px;
                }
            </style>
        </head>
        <body>
            <h1>Conversation Monitor</h1>
            <div class="breadcrumb">
                <strong>Path:</strong> {{ breadcrumb|safe }}
            </div>
            <div class="file-list">
                {% if items %}
                    {% for item in items %}
                        {% if item.is_dir %}
                            <div class="dir-item">
                                <a href="{{ item.url }}">{{ item.name }}</a>
                            </div>
                        {% else %}
                            <div class="file-item">
                                <a href="{{ item.url }}">{{ item.name }}</a>
                                <span class="file-size">{{ item.size }}</span>
                            </div>
                        {% endif %}
                    {% endfor %}
                {% else %}
                    <div class="empty">No files or directories found.</div>
                {% endif %}
            </div>
        </body>
        </html>
        """

        # HTML template for file viewing
        FILE_TEMPLATE = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ filename }} - Conversation Monitor</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                h1 {
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }
                .breadcrumb {
                    margin: 20px 0;
                    padding: 10px;
                    background-color: #fff;
                    border-radius: 5px;
                }
                .breadcrumb a {
                    color: #4CAF50;
                    text-decoration: none;
                    margin-right: 5px;
                }
                .breadcrumb a:hover {
                    text-decoration: underline;
                }
                .file-content {
                    background-color: #fff;
                    border-radius: 5px;
                    padding: 20px;
                    white-space: pre-wrap;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 1.5;
                    overflow-x: auto;
                }
                .back-button {
                    display: inline-block;
                    margin: 20px 0;
                    padding: 10px 20px;
                    background-color: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }
                .back-button:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <h1>{{ filename }}</h1>
            <div class="breadcrumb">
                <strong>Path:</strong> {{ breadcrumb|safe }}
            </div>
            <a href="{{ back_url }}" class="back-button">← Back to Directory</a>
            <div class="file-content">{{ content }}</div>
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
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()

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

                    return render_template_string(
                        FILE_TEMPLATE,
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
                    relative_path = entry.relative_to(self.cache_dir)
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
        print(f"Starting conversation monitor web server at http://{host}:{port}")
        print(f"Cache directory: {self.cache_dir.resolve()}")
        app.run(host=host, port=port, debug=debug)


# Global monitor instance
_global_monitor: ConversationMonitor | None = None


def get_monitor(cache_dir: str = "cache") -> ConversationMonitor:
    """
    Get the global conversation monitor instance.

    Args:
        cache_dir: Directory to store conversation history files

    Returns:
        ConversationMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ConversationMonitor(cache_dir=cache_dir)
    return _global_monitor
