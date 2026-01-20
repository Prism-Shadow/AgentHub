# Integration Module

This module contains integration tools for AgentHub, including conversation tracing and a web UI for interacting with LLMs.

## Components

### Tracer

The `Tracer` class provides functionality to save conversation history to local files and browse them via a web interface.

**Features:**
- Save conversation history as JSON and text files
- Web interface for browsing saved conversations
- Automatic serialization of complex data types

**Usage:**

```python
from integration import Tracer

# Initialize tracer
tracer = Tracer(cache_dir="cache")

# Save conversation history
history = [
    {"role": "user", "content_items": [{"type": "text", "text": "Hello"}]},
    {"role": "assistant", "content_items": [{"type": "text", "text": "Hi there!"}]}
]
config = {"temperature": 0.7}
tracer.save_history(history, "conversation_001", config)

# Start web server to browse saved conversations
tracer.start_web_server(host="127.0.0.1", port=5000)
```

**With AutoLLMClient:**

```python
from agenthub import AutoLLMClient

client = AutoLLMClient(model="gemini-3-flash-preview")
config = {"trace_id": "agent1/conversation_001", "temperature": 0.7}

message = {"role": "user", "content_items": [{"type": "text", "text": "Hello"}]}
async for _ in client.streaming_response_stateful(message=message, config=config):
    pass
# Conversation is automatically saved to cache/agent1/conversation_001.json and .txt
```

### Web UI

The Web UI provides an interactive chat interface for communicating with LLMs through a web browser.

**Features:**
- Interactive chat interface
- Editable configuration (model, temperature, max_tokens)
- Streaming message display
- Message cards showing token usage and finish reason
- Support for multiple AI models (Gemini, Claude, GLM)

**Usage:**

```python
from integration import start_chat_server

# Start the chat server
start_chat_server(host="127.0.0.1", port=5001, debug=False)
# Open http://127.0.0.1:5001 in your browser
```

**Or run the example:**

```bash
python examples/web_ui_example.py
```

**Features:**
- **Config Panel**: Click the "⚙️ Config" button to edit:
  - Model selection (Gemini 3 Flash, Claude Sonnet 4.5, GLM 4.7)
  - Temperature (0-2)
  - Max Tokens
- **Message Input**: Type messages in the input box (Shift+Enter for new line, Enter to send)
- **Message Cards**: Each message is displayed as a card with:
  - Role indicator (User/Assistant)
  - Message content
  - Token usage (in bottom-right corner)
  - Finish reason (in bottom-right corner)
- **Streaming**: Responses stream in real-time as the model generates them
- **Clear**: Clear the conversation history

## Examples

See the `examples/` directory for usage examples:
- `trace_example.py` - Demonstrates conversation tracing
- `web_ui_example.py` - Starts the web UI server

## API Endpoints

The Web UI exposes the following endpoints:

- `GET /` - Main chat interface
- `POST /api/chat` - Chat API endpoint (accepts streaming requests)
  - Request body: `{"message": {...}, "config": {...}, "history": [...]}`
  - Response: Server-Sent Events (SSE) stream with chat events
