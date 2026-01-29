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
import { AutoLLMClient } from '@prismshadow/agenthub';

process.env.OPENAI_API_KEY = 'your-openai-api-key';

async function main() {
  const client = new AutoLLMClient({ model: 'gpt-5.2' });

  for await (const event of client.streamingResponseStateful({
    message: {
      role: 'user',
      content_items: [{ type: 'text', text: 'Hello!' }]
    },
    config: {}
  })) {
    console.log(event);
  }
}

main().catch(console.error);
```

### Image Understanding

AgentHub supports both URL and base64 encoded images:

**Using URL:**
```typescript
import { AutoLLMClient } from '@prismshadow/agenthub';

async function main() {
  const client = new AutoLLMClient({ model: 'gpt-5.2' });

  for await (const event of client.streamingResponse({
    messages: [{
      role: 'user',
      content_items: [
        { type: 'text', text: "What's in this image?" },
        { type: 'image_url', image_url: 'https://example.com/image.jpg' }
      ]
    }],
    config: {}
  })) {
    for (const item of event.content_items) {
      if (item.type === 'text') {
        process.stdout.write(item.text);
      }
    }
  }
}

main().catch(console.error);
```

**Using Base64 Encoding:**
```typescript
import * as fs from 'fs';
import { AutoLLMClient } from '@prismshadow/agenthub';

async function main() {
  const client = new AutoLLMClient({ model: 'gpt-5.2' });

  // Read and encode image
  const imageBuffer = fs.readFileSync('image.png');
  const base64Image = imageBuffer.toString('base64');
  
  // Create data URI
  const dataUri = `data:image/png;base64,${base64Image}`;

  for await (const event of client.streamingResponse({
    messages: [{
      role: 'user',
      content_items: [
        { type: 'text', text: "What's in this image?" },
        { type: 'image_url', image_url: dataUri }
      ]
    }],
    config: {}
  })) {
    for (const item of event.content_items) {
      if (item.type === 'text') {
        process.stdout.write(item.text);
      }
    }
  }
}

main().catch(console.error);
```

Supported models with image understanding:
- GPT-5.2 (`gpt-5.2`)
- Claude 4.5 (`claude-sonnet-4-5-20250929`)
- Gemini 3 (`gemini-3-flash-preview`)

### Tracer Usage

Save and browse conversation history with a web interface:

```typescript
import { Tracer } from '@prismshadow/agenthub/integration/tracer';

// Create a tracer instance
const tracer = new Tracer('./cache');

// Save conversation history
const model = 'gpt-5.2';
const history = [
  { role: 'user', content_items: [{ type: 'text', text: 'Hello!' }] },
  { role: 'assistant', content_items: [{ type: 'text', text: 'Hi there!' }] }
];
const config = {};
tracer.saveHistory(model, history, 'session/conv_001', config);

// Start web server to view saved conversations
tracer.startWebServer('127.0.0.1', 5000);
// Open http://127.0.0.1:5000 in your browser
```

### Playground Usage

Interactive web interface for chatting with LLMs:

```typescript
import { startPlaygroundServer } from '@prismshadow/agenthub/integration/playground';

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
