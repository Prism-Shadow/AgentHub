# AgentHub TypeScript Implementation

This directory contains the TypeScript implementation of AgentHub, mirroring the Python implementation in `src_py/`.

## Building

```bash
make install  # Install dependencies
make build    # Build TypeScript to JavaScript
make lint     # Run ESLint
make test     # Run tests
```

## Usage

### Basic Client Usage

```typescript
import { AutoLLMClient } from 'agenthub';

// Initialize with model name
const client = new AutoLLMClient('gemini3flash');
```

### Tracer Usage

Save and browse conversation history with a web interface:

```typescript
import { Tracer } from 'agenthub/integration/tracer';

// Create a tracer instance
const tracer = new Tracer('./cache');

// Save conversation history
tracer.saveHistory(history, 'session/conv_001', config);

// Start web server to view saved conversations
tracer.startWebServer('127.0.0.1', 5000);
// Open http://127.0.0.1:5000 in your browser
```

### Playground Usage

Interactive web interface for chatting with LLMs:

```typescript
import { startPlaygroundServer } from 'agenthub/integration/playground';

// Start the playground server
startPlaygroundServer('127.0.0.1', 5001);
// Open http://127.0.0.1:5001 in your browser
```

## Examples

Run the examples:

```bash
# Build the project
npm run build

# Run tracer example
node dist/examples/tracerExample.js

# Run playground example
node dist/examples/playgroundExample.js
```
