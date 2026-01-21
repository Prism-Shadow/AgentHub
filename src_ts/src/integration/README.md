# Integration Module

This module provides tools for conversation tracing and interactive playground for AgentHub.

## Features

### Tracer

The Tracer module allows you to save conversation history to local files and view them through a web interface with real-time monitoring.

**Features:**
- Save conversation history to JSON and text files
- Web interface for browsing saved conversations
- Beautiful UI with Tailwind CSS
- Support for all content types (text, thinking, tool calls, images)
- Display usage metadata and finish reasons

**Usage:**

```typescript
import { Tracer } from "agenthub";

// Create a tracer instance
const tracer = new Tracer("./cache");

// Save conversation history
tracer.saveHistory(history, "session/conv_001", config);

// Start web server to view saved conversations
tracer.startWebServer("127.0.0.1", 5000);
// Open http://127.0.0.1:5000 in your browser
```

### Playground

The Playground module provides an interactive web interface for chatting with LLMs.

**Features:**
- Real-time streaming chat interface
- Configuration editor for model parameters
- Beautiful UI with Tailwind CSS
- Support for:
  - Model selection
  - Temperature and max tokens
  - Thinking level and summary
  - Tool choice
  - System prompts
  - Tools (JSON format)
  - Trace ID for saving conversations
- Display message cards with token usage and finish reasons

**Usage:**

```typescript
import { startPlaygroundServer } from "agenthub";

// Start the playground server
startPlaygroundServer("127.0.0.1", 5001);
// Open http://127.0.0.1:5001 in your browser
```

## Examples

See the `examples` directory for complete working examples:

- `tracerExample.ts` - Demonstrates how to use the Tracer
- `playgroundExample.ts` - Demonstrates how to start the Playground

To run the examples:

```bash
# Build the project
npm run build

# Run tracer example
node dist/examples/tracerExample.js

# Run playground example
node dist/examples/playgroundExample.js
```

## UI Design

Both the Tracer and Playground use Tailwind CSS for a modern, responsive UI:

- **Tracer UI:**
  - File browser with directory navigation
  - JSON viewer with collapsible message cards
  - Syntax highlighting for different content types
  - Token usage and finish reason display

- **Playground UI:**
  - Clean chat interface with message cards
  - Collapsible configuration panel
  - Streaming response visualization
  - Real-time token usage display

## API Reference

### Tracer

#### Constructor

```typescript
new Tracer(cacheDir?: string)
```

Creates a new Tracer instance.

**Parameters:**
- `cacheDir` (optional): Directory to store conversation history files. Defaults to `AGENTHUB_CACHE_DIR` environment variable or `"cache"`.

#### Methods

##### `saveHistory(history: UniMessage[], fileId: string, config: UniConfig): void`

Save conversation history to files.

**Parameters:**
- `history`: Array of UniMessage objects representing the conversation
- `fileId`: File identifier without extension (e.g., "agent1/00001")
- `config`: The UniConfig used for this conversation

##### `createWebApp(): Express`

Create an Express web application for browsing conversation files.

**Returns:** Express application instance

##### `startWebServer(host?: string, port?: number): void`

Start the web server for browsing conversation files.

**Parameters:**
- `host` (optional): Host address to bind to. Default: "127.0.0.1"
- `port` (optional): Port number to listen on. Default: 5000

### Playground

#### Functions

##### `createChatApp(): Express`

Create an Express web application for chatting with LLMs.

**Returns:** Express application instance

##### `startPlaygroundServer(host?: string, port?: number): void`

Start the playground web server.

**Parameters:**
- `host` (optional): Host address to bind to. Default: "127.0.0.1"
- `port` (optional): Port number to listen on. Default: 5001

## Testing

Run the tests with:

```bash
npm test
```

The test suite includes:
- Tracer initialization tests
- History saving tests
- Directory creation tests
- File overwrite tests
- Serialization tests
- Metadata formatting tests
