# AgentHub TypeScript Implementation

This directory contains the TypeScript implementation of AgentHub, mirroring the Python implementation in `src_py/`.

## Structure

- `src/types.ts` - Type definitions for all data structures
- `src/baseClient.ts` - Abstract base class for LLM clients
- `src/autoClient.ts` - Auto-routing client that dispatches to model-specific clients
- `src/integration/` - Integration tools (tracer and playground)
  - `tracer.ts` - Conversation history tracer with web UI
  - `playground.ts` - Interactive LLM playground with web UI
  - `index.ts` - Integration module exports
  - `README.md` - Integration module documentation
- `src/index.ts` - Main exports
- `examples/` - Example applications
  - `tracerExample.ts` - Tracer usage example
  - `playgroundExample.ts` - Playground usage example

## Implementation Status

### Completed ✅
- Core type system (types.ts)
- Base client abstract class (baseClient.ts)
- Auto-routing client (autoClient.ts)
- **Tracer for conversation history with web UI (integration/tracer.ts)**
- **Playground web interface with streaming chat (integration/playground.ts)**
- **Tailwind CSS UI for both tracer and playground**
- **Example applications and comprehensive tests**

### Not Implemented ❌
- Model-specific clients (claude4_5, gemini3, gpt5_2, glm4_7, qwen3)

Model-specific clients should be implemented following the Python version's pattern. Each model client should:
1. Extend the `LLMClient` base class
2. Implement the abstract methods:
   - `transformUniConfigToModelConfig()`
   - `transformUniMessageToModelInput()`
   - `transformModelOutputToUniEvent()`
   - `streamingResponse()`

## Building

```bash
make build    # Build TypeScript to JavaScript
make lint     # Run ESLint
make test     # Run tests
```

## Usage

### Basic Client Usage

```typescript
import { AutoLLMClient, ThinkingLevel, PromptCaching } from 'agenthub';

// Note: Model-specific clients are not yet implemented
// This will throw an error until you implement the specific client
const client = new AutoLLMClient('gemini-3-flash-preview');

// The client provides the same interface as the Python version
```

### Tracer Usage

```typescript
import { Tracer } from 'agenthub';

// Create a tracer instance
const tracer = new Tracer('./cache');

// Save conversation history
tracer.saveHistory(history, 'session/conv_001', config);

// Start web server to view saved conversations
tracer.startWebServer('127.0.0.1', 5000);
// Open http://127.0.0.1:5000 in your browser
```

### Playground Usage

```typescript
import { startPlaygroundServer } from 'agenthub';

// Start the playground server
startPlaygroundServer('127.0.0.1', 5001);
// Open http://127.0.0.1:5001 in your browser
```

## Integration Features

The integration module provides two powerful tools:

### 1. Tracer
- Save conversation history to JSON and text files
- Web interface for browsing saved conversations with Tailwind CSS
- Support for all content types (text, thinking, tool calls, images)
- Display usage metadata and finish reasons
- Beautiful, responsive UI

### 2. Playground
- Interactive web interface for chatting with LLMs
- Real-time streaming chat with Tailwind CSS UI
- Configuration editor for model parameters
- Display message cards with token usage and finish reasons
- Support for tools, system prompts, and advanced settings

See `src/integration/README.md` for detailed documentation.

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

## Notes

- This implementation maintains consistency with the Python version
- The code logic mirrors the Python implementation as closely as possible
- Integration features use Tailwind CSS for modern, responsive UI
- ESLint is configured to allow unused parameters prefixed with underscore
- TypeScript strict mode is enabled for type safety
