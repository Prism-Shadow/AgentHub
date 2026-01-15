# Code Conventions

This document outlines the coding standards and best practices for the AgentAdapter project.

## General Principles

1. **Consistency**: Follow existing patterns in the codebase
2. **Simplicity**: Prefer simple, readable code over clever solutions
3. **Documentation**: Document non-obvious decisions and complex logic
4. **Testing**: Write tests for new functionality
5. **License headers**: Include Apache 2.0 license headers in source files

## Python Conventions

### Style Guide

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

### Formatting

Use **ruff** for automatic formatting:
```bash
uv run ruff format .
```

**Key settings** (from `pyproject.toml`):
- Line length: 119 characters
- Indent: 4 spaces
- Quote style: Double quotes
- Python version: 3.11+

### Linting Rules

Enabled ruff rules:
- `C`: McCabe complexity
- `E`: pycodestyle errors
- `F`: Pyflakes
- `I`: isort (import sorting)
- `W`: pycodestyle warnings
- `RUF022`: Ruff-specific checks

Ignored rules:
- `C901`: Complex functions (case-by-case basis)
- `E501`: Line too long (handled by formatter)
- `E741`: Ambiguous variable names (case-by-case)
- `W605`: Invalid escape sequences
- `C408`: Unnecessary dict call

### Import Organization

```python
# Standard library imports
import os
import sys

# Third-party imports
import requests
import numpy as np

# Local imports
from agent_adapter import module_name
```

ruff automatically organizes imports with 2 lines after imports.

### Type Hints

Use type hints for function signatures:

```python
def get_message() -> str:
    return "Hello from Agent Adapter!"

def process_data(input_data: dict[str, any]) -> list[str]:
    # Implementation
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of function.

    Longer description if needed, explaining the purpose,
    behavior, and any important details.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        Description of return value.

    Raises:
        ValueError: Description of when this is raised.
    """
    pass
```

### Testing Conventions

```python
# File: tests/test_module_name.py

def test_function_does_expected_behavior():
    """Test that function performs expected behavior."""
    # Arrange
    expected = "expected value"
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected


def test_function_handles_edge_case():
    """Test edge case handling."""
    # Test implementation
    pass
```

## TypeScript Conventions

### Style Guide

Follow TypeScript and ESLint best practices.

### Formatting

- Indent: 2 spaces
- Semicolons: Use them
- Quotes: Double quotes (consistent with Python)
- Line length: Reasonable (around 100-120 characters)

### Type Safety

Always use TypeScript types:

```typescript
// Good: Explicit types
export const getMessage = (): string => "Hello from Agent Adapter!";

export const printHelloWorld = (): void => {
  console.log(getMessage());
};

// Avoid: Implicit 'any' types
const processData = (data) => { // Bad: implicit 'any'
  return data.value;
};
```

### Export Conventions

Use named exports for better refactorability:

```typescript
// Prefer named exports
export const functionName = (): string => {
  return "value";
};

export class ClassName {
  // Implementation
}

// Avoid default exports unless necessary
export default something; // Use sparingly
```

### Function Style

Use arrow functions for consistency:

```typescript
// Preferred
export const functionName = (param: string): string => {
  return param.toUpperCase();
};

// Also acceptable for methods
class Example {
  public methodName(param: string): string {
    return param.toLowerCase();
  }
}
```

### Module Organization

```typescript
// Type definitions and interfaces first
interface DataStructure {
  id: string;
  value: number;
}

// Constants
const MAX_RETRIES = 3;

// Helper functions
const helperFunction = (): void => {
  // Implementation
};

// Main exports
export const mainFunction = (): void => {
  // Implementation
};
```

## File Naming

### Python

- Module files: `lowercase_with_underscores.py`
- Test files: `test_module_name.py`
- Package directories: `lowercase_no_underscores/`

### TypeScript

- Source files: `camelCase.ts` or `PascalCase.ts` for classes
- Test files: `moduleName.test.ts` or `moduleName.spec.ts`
- Configuration: `lowercase-with-dashes.config.js`

## License Headers

All source code files should include the Apache 2.0 license header:

### Python
```python
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
```

### TypeScript
```typescript
/*
 * Copyright 2025 Prism Shadow. and/or its affiliates
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```

## Git Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Use imperative mood: "Move cursor to..." not "Moves cursor to..."
- First line: Brief summary (50 characters or less)
- Blank line, then detailed description if needed
- Reference issues: "Fixes #123" or "Related to #456"

Example:
```
Add user authentication module

Implement JWT-based authentication with refresh tokens.
Includes middleware for protected routes and token validation.

Fixes #42
```

## Code Review Checklist

Before submitting for review:

- [ ] Code follows style guidelines (ruff/ESLint passes)
- [ ] Tests added/updated and passing
- [ ] License headers present
- [ ] Documentation updated if needed
- [ ] No debug code or commented-out code
- [ ] Commits are logical and well-described
- [ ] Changes are minimal and focused
