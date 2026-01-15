# Directory Tree

This document shows the overall structure of the AgentAdapter repository.

```
.
├── LICENSE
├── README.md
├── docs/
│   ├── claude4_5/
│   │   ├── docs/          # Claude 4.5 documentation files
│   │   ├── examples/      # Claude 4.5 example code
│   │   ├── readme.python.md
│   │   └── readme.typescript.md
│   ├── gemini3/
│   │   ├── docs/          # Gemini 3 documentation files
│   │   ├── readme.python.md
│   │   └── readme.typescript.md
│   └── gpt5_2/
│       ├── docs/          # GPT-5.2 documentation files
│       ├── examples/      # GPT-5.2 example code
│       ├── readme.python.md
│       └── readme.typescript.md
├── src_py/
│   ├── Makefile
│   ├── agent_adapter/
│   │   ├── __init__.py
│   │   └── hello_world.py
│   ├── pyproject.toml
│   └── tests/
│       └── test_hello_world.py
└── src_ts/
    ├── Makefile
    ├── eslint.config.cjs
    ├── package-lock.json
    ├── package.json
    ├── src/
    │   └── index.ts
    └── tsconfig.json
```

## Directory Structure Overview

- **docs/** - Documentation for different AI model adapters
  - **claude4_5/** - Claude 4.5 API documentation and examples
  - **gemini3/** - Gemini 3 API documentation and examples
  - **gpt5_2/** - GPT-5.2 API documentation and examples
  
- **src_py/** - Python package source code
  - **agent_adapter/** - Main Python package
  - **tests/** - Python test files
  
- **src_ts/** - TypeScript package source code
  - **src/** - TypeScript source files
