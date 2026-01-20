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
Example demonstrating the AgentHub Chat Web UI.

This example shows how to start the web UI server for interactive
chat with LLMs. The web UI supports:
- Config editing (model, temperature, max_tokens)
- Streaming chat responses
- Message cards with token usage and finish reasons
"""

from agenthub.integration.web_ui import start_chat_server


if __name__ == "__main__":
    print("=" * 60)
    print("AgentHub Chat Web UI")
    print("=" * 60)
    print("\nStarting web server...")
    print("\nOpen http://127.0.0.1:5001 in your browser to start chatting!")
    print("Press Ctrl+C to stop the server.\n")

    start_chat_server(host="127.0.0.1", port=5001, debug=False)
