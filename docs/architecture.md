# Architecture Guide

## Project Structure

AgentAdapter is a monorepo containing two independent but parallel implementations:

```
AgentAdapter/
├── .github/
│   └── copilot-instructions.md    # GitHub Copilot configuration
├── docs/                           # Agent reference documentation
│   ├── README.md                  # Documentation index
│   ├── architecture.md            # This file
│   ├── development.md             # Development guide
│   └── conventions.md             # Code conventions
├── src_py/                        # Python package
│   ├── agent_adapter/             # Python source code
│   ├── tests/                     # Python tests
│   ├── pyproject.toml            # Python project config
│   └── Makefile                  # Python build commands
├── src_ts/                        # TypeScript package
│   ├── src/                      # TypeScript source code
│   ├── package.json              # Node.js package config
│   ├── tsconfig.json             # TypeScript config
│   └── Makefile                  # TypeScript build commands
├── CONTRIBUTING.md                # Contribution guidelines
├── LICENSE                        # Apache 2.0 License
└── README.md                      # Main project README
```

## Design Principles

### 1. Dual Implementation Strategy

The project maintains separate Python and TypeScript implementations to:
- Demonstrate cross-language compatibility
- Serve as examples for different ecosystems
- Allow independent evolution of each implementation
- Provide templates for both Python and Node.js projects

### 2. Separation of Concerns

- **Language-specific code**: Kept in dedicated `src_py/` and `src_ts/` directories
- **Build tools**: Each directory has its own `Makefile` for consistency
- **Dependencies**: Managed independently per language (pyproject.toml vs package.json)
- **Testing**: Each implementation has its own test suite

### 3. Minimal Dependencies

Both implementations intentionally maintain minimal dependencies to:
- Reduce complexity
- Improve maintainability
- Serve as clean templates
- Minimize security surface area

## Component Overview

### Python Package (`src_py/`)

- **Package name**: `agent-adapter`
- **Module structure**: `agent_adapter.hello_world`
- **Build system**: uv (modern Python package manager)
- **Testing**: pytest
- **Linting**: ruff (fast Python linter and formatter)

### TypeScript Package (`src_ts/`)

- **Package name**: `agent-adapter`
- **Module structure**: ES6 modules
- **Build system**: TypeScript compiler (tsc)
- **Testing**: Placeholder (extensible)
- **Linting**: ESLint with TypeScript plugin

## Future Considerations

### Scalability

When extending this project:
1. Maintain the dual-language structure
2. Add new modules in respective `agent_adapter/` or `src/` directories
3. Keep build processes language-specific
4. Document cross-language compatibility requirements

### Integration Points

Currently, the two implementations are independent. If integration is needed:
- Consider using protocol buffers or JSON for data exchange
- Implement consistent API interfaces
- Document shared contracts in this directory
