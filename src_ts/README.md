# AgentHub TypeScript Implementation

This directory contains the TypeScript implementation of AgentHub, mirroring the Python implementation in `src_py/`.

## Structure

- `src/types.ts` - Type definitions for all data structures
- `src/baseClient.ts` - Abstract base class for LLM clients
- `src/autoClient.ts` - Auto-routing client that dispatches to model-specific clients
- `src/integration/tracer.ts` - Conversation history tracer
- `src/index.ts` - Main exports

## Implementation Status

### Completed ✅
- Core type system (types.ts)
- Base client abstract class (baseClient.ts)
- Auto-routing client (autoClient.ts)
- Tracer for conversation history (integration/tracer.ts)

### Not Implemented ❌
- Model-specific clients (claude4_5, gemini3, gpt5_2, glm4_7, qwen3)
- Playground web interface (integration/playground.ts)

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
make test     # Run tests (not yet implemented)
```

## Usage

```typescript
import { AutoLLMClient, ThinkingLevel, PromptCaching } from 'agenthub';

// Note: Model-specific clients are not yet implemented
// This will throw an error until you implement the specific client
const client = new AutoLLMClient('gemini-3-flash-preview');

// The client provides the same interface as the Python version
```

## Notes

- This implementation maintains consistency with the Python version
- The code logic mirrors the Python implementation as closely as possible
- ESLint is configured to allow unused parameters prefixed with underscore
- TypeScript strict mode is enabled for type safety
