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
const client = new AutoLLMClient({ model: 'gemini-3-flash-preview' });

// With API key and base URL
const client = new AutoLLMClient({
  model: 'claude-sonnet-4-5-20250929',
  apiKey: 'your-api-key',
  baseUrl: 'https://api.anthropic.com'
});

// Using streaming methods with options
const message = {
  role: 'user',
  content_items: [{ type: 'text', text: 'Hello!' }]
};
const config = { temperature: 0.7 };

for await (const event of client.streamingResponseStateful({ message, config })) {
  console.log(event);
}
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
npm run tracer

# Run playground example
npm run playground
```
