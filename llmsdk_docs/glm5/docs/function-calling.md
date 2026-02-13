> ## Documentation Index
> Fetch the complete documentation index at: https://docs.z.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Function Calling

<Tip>
  Function Calling allows AI models to call external functions and APIs, greatly expanding the capability boundaries of intelligent agents, enabling them to perform specific operations and obtain real-time data.
</Tip>

## Features

Function calling provides AI models with the ability to interact with external systems, supporting various complex application scenarios and integration requirements.

### Core Parameter Description

* **`tools`**: Defines the list of callable functions, including function names, descriptions, and parameter specifications
* **`tool_choice`**: Controls function calling strategy, default is `auto` (only supports `auto`)
* **`model`**: Uses models that support function calling, such as `glm-4-plus`, `glm-4.6`, etc.

### Response Parameter Description

Key fields in function calling responses:

* **`tool_calls`**: Contains information about functions the model decides to call
* **`function.name`**: Name of the called function
* **`function.arguments`**: Function call parameters (JSON format string)
* **`id`**: Unique identifier for the tool call

## Code Examples

By defining function tools and handling function calls, AI models can perform various external operations:

<Tabs>
  <Tab title="Python SDK">
    **Install SDK**

    ```bash  theme={null}
    # Install latest version
    pip install zai-sdk

    # Or specify version
    pip install zai-sdk==0.1.0
    ```

    **Verify Installation**

    ```python  theme={null}
    import zai
    print(zai.__version__)
    ```

    **Complete Example**

    ```python  theme={null}
    import json
    from zai import ZaiClient

    # Initialize client
    client = ZaiClient(api_key='your_api_key')

    # Define weather query function
    def get_weather(city: str) -> dict:
        """Get weather information for specified city"""
        # This should call a real weather API
        weather_data = {
            "city": city,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%",
            "wind_speed": "5 km/h"
        }
        return weather_data

    # Define function tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for specified city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name, e.g.: Beijing, Shanghai"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    # Make conversation request
    response = client.chat.completions.create(
        model="glm-5",  # Use model that supports function calling
        messages=[
            {"role": "user", "content": "How's the weather in Beijing today?"}
        ],
        tools=tools,         # Pass function tools
        tool_choice="auto"   # Automatically choose whether to call functions
    )

    # Handle function calls
    message = response.choices[0].message
    messages = [{"role": "user", "content": "How's the weather in Beijing today?"}]
    messages.append(message.model_dump())

    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "get_weather":
                # Parse parameters and call function
                args = json.loads(tool_call.function.arguments)
                weather_result = get_weather(args.get("city"))

                # Return function result to model
                messages.append({
                    "role": "tool",
                    "content": json.dumps(weather_result, ensure_ascii=False),
                    "tool_call_id": tool_call.id
                })

        # Get final answer
        final_response = client.chat.completions.create(
            model="glm-5",
            messages=messages,
            tools=tools
        )

        print(final_response.choices[0].message.content)
    else:
        print(message.content)
    ```
  </Tab>
</Tabs>

## Scenario Examples

<Warning>
  When using function calling, please ensure proper security validation and permission control for external APIs and database operations.
</Warning>

<Accordion title="Multi-function Assistant">
  ```python  theme={null}
  import json
  import requests
  from datetime import datetime
  from zai import ZaiClient

  class FunctionAgent:
      def __init__(self, api_key):
          self.client = ZaiClient(api_key=api_key)
          self.tools = self._define_tools()

      def _define_tools(self):
          return [
              {
                  "type": "function",
                  "function": {
                      "name": "get_current_time",
                      "description": "Get current time",
                      "parameters": {
                          "type": "object",
                          "properties": {},
                          "required": []
                      }
                  }
              },
              {
                  "type": "function",
                  "function": {
                      "name": "calculate",
                      "description": "Perform mathematical calculations",
                      "parameters": {
                          "type": "object",
                          "properties": {
                              "expression": {
                                  "type": "string",
                                  "description": "Mathematical expression, e.g.: 2+3*4"
                              }
                          },
                          "required": ["expression"]
                      }
                  }
              },
              {
                  "type": "function",
                  "function": {
                      "name": "search_web",
                      "description": "Search web information",
                      "parameters": {
                          "type": "object",
                          "properties": {
                              "query": {
                                  "type": "string",
                                  "description": "Search keywords"
                              }
                          },
                          "required": ["query"]
                      }
                  }
              }
          ]

      def get_current_time(self):
          """Get current time"""
          return {
              "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "timezone": "Asia/Shanghai"
          }

      def calculate(self, expression: str):
          """Safe mathematical calculation"""
          try:
              # Simple security check
              allowed_chars = set('0123456789+-*/().')
              if not all(c in allowed_chars or c.isspace() for c in expression):
                  return {"error": "Expression contains disallowed characters"}

              result = eval(expression)
              return {
                  "expression": expression,
                  "result": result
              }
          except Exception as e:
              return {"error": f"Calculation error: {str(e)}"}

      def search_web(self, query: str):
          """Simulate web search"""
          # This should call a real search API
          return {
              "query": query,
              "results": [
                  {"title": f"Search result 1 about {query}", "url": "https://example1.com"},
                  {"title": f"Search result 2 about {query}", "url": "https://example2.com"}
              ]
          }

      def execute_function(self, function_name: str, arguments: dict):
          """Execute function call"""
          if function_name == "get_current_time":
              return self.get_current_time()
          elif function_name == "calculate":
              return self.calculate(arguments.get("expression", ""))
          elif function_name == "search_web":
              return self.search_web(arguments.get("query", ""))
          else:
              return {"error": f"Unknown function: {function_name}"}

      def chat(self, user_message: str):
          """Handle user message"""
          messages = [{"role": "user", "content": user_message}]

          response = self.client.chat.completions.create(
              model="glm-5",
              messages=messages,
              tools=self.tools,
              tool_choice="auto"
          )

          message = response.choices[0].message
          messages.append(message.model_dump())

          # Handle function calls
          if message.tool_calls:
              for tool_call in message.tool_calls:
                  function_name = tool_call.function.name
                  arguments = json.loads(tool_call.function.arguments)

                  # Execute function
                  result = self.execute_function(function_name, arguments)

                  # Add function result
                  messages.append({
                      "role": "tool",
                      "content": json.dumps(result, ensure_ascii=False),
                      "tool_call_id": tool_call.id
                  })

              # Get final answer
              final_response = self.client.chat.completions.create(
                  model="glm-5",
                  messages=messages,
                  tools=self.tools
              )

              return final_response.choices[0].message.content
          else:
              return message.content

  # Usage example
  agent = FunctionAgent("your_api_key")

  # Test different types of requests
  print(agent.chat("What time is it now?"))
  print(agent.chat("Help me calculate 15 * 23 + 7"))
  print(agent.chat("Search for the latest developments in artificial intelligence"))
  ```
</Accordion>

<Accordion title="Database Query">
  ```python  theme={null}
  import sqlite3

  def query_database(sql: str) -> dict:
      """Execute database query"""
      try:
          conn = sqlite3.connect('example.db')
          cursor = conn.cursor()
          cursor.execute(sql)
          results = cursor.fetchall()
          conn.close()

          return {
              "success": True,
              "data": results,
              "row_count": len(results)
          }
      except Exception as e:
          return {
              "success": False,
              "error": str(e)
          }

  # Function definition
  db_tool = {
      "type": "function",
      "function": {
          "name": "query_database",
          "description": "Execute SQL query",
          "parameters": {
              "type": "object",
              "properties": {
                  "sql": {
                      "type": "string",
                      "description": "SQL query statement"
                  }
              },
              "required": ["sql"]
          }
      }
  }
  ```
</Accordion>

<Accordion title="File Operations">
  ```python  theme={null}
  import os
  import json

  def file_operations(operation: str, file_path: str, content: str = None) -> dict:
      """File operation function"""
      try:
          if operation == "read":
              with open(file_path, 'r', encoding='utf-8') as f:
                  content = f.read()
              return {"success": True, "content": content}

          elif operation == "write":
              with open(file_path, 'w', encoding='utf-8') as f:
                  f.write(content)
              return {"success": True, "message": "File written successfully"}

          elif operation == "list":
              files = os.listdir(file_path)
              return {"success": True, "files": files}

          else:
              return {"success": False, "error": "Unsupported operation"}

      except Exception as e:
          return {"success": False, "error": str(e)}

  # Function definition
  file_tool = {
      "type": "function",
      "function": {
          "name": "file_operations",
          "description": "Execute file operations",
          "parameters": {
              "type": "object",
              "properties": {
                  "operation": {
                      "type": "string",
                      "enum": ["read", "write", "list"],
                      "description": "Operation type"
                  },
                  "file_path": {
                      "type": "string",
                      "description": "File path"
                  },
                  "content": {
                      "type": "string",
                      "description": "Content to write (only required for write operation)"
                  }
              },
              "required": ["operation", "file_path"]
          }
      }
  }
  ```
</Accordion>

<Accordion title="API Integration">
  ```python  theme={null}
  import requests

  def call_external_api(url: str, method: str = "GET", headers: dict = None, data: dict = None) -> dict:
      """Call external API"""
      try:
          if method.upper() == "GET":
              response = requests.get(url, headers=headers, params=data)
          elif method.upper() == "POST":
              response = requests.post(url, headers=headers, json=data)
          else:
              return {"success": False, "error": "Unsupported HTTP method"}

          return {
              "success": True,
              "status_code": response.status_code,
              "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
          }

      except Exception as e:
          return {"success": False, "error": str(e)}

  # Function definition
  api_tool = {
      "type": "function",
      "function": {
          "name": "call_external_api",
          "description": "Call external API",
          "parameters": {
              "type": "object",
              "properties": {
                  "url": {
                      "type": "string",
                      "description": "API endpoint URL"
                  },
                  "method": {
                      "type": "string",
                      "enum": ["GET", "POST"],
                      "description": "HTTP method"
                  },
                  "headers": {
                      "type": "object",
                      "description": "Request headers"
                  },
                  "data": {
                      "type": "object",
                      "description": "Request data"
                  }
              },
              "required": ["url"]
          }
      }
  }
  ```
</Accordion>

## Best Practices

<CardGroup cols={2}>
  <Card title="Function Design Principles" icon="code">
    * Single responsibility: Each function should do one thing
    * Clear naming: Function and parameter names should be meaningful
    * Complete description: Provide detailed function and parameter descriptions
  </Card>

  <Card title="Security Considerations" icon="shield">
    * Input validation: Strictly validate all input parameters
    * Permission control: Limit function access permissions
    * Logging: Record function call logs
  </Card>
</CardGroup>

### Parameter Design

```python  theme={null}
# Good parameter design
{
    "type": "object",
    "properties": {
        "city": {
            "type": "string",
            "description": "City name, supports Chinese and English, e.g.: Beijing, Shanghai, New York",
            "examples": ["Beijing", "Shanghai", "New York"]
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit",
            "default": "celsius"
        }
    },
    "required": ["city"]
}
```

### Error Handling

```python  theme={null}
def robust_function(param: str) -> dict:
    """Robust function implementation"""
    try:
        # Parameter validation
        if not param or not isinstance(param, str):
            return {
                "success": False,
                "error": "Invalid parameter",
                "error_code": "INVALID_PARAM"
            }

        # Business logic
        result = process_data(param)

        return {
            "success": True,
            "data": result,
            "timestamp": datetime.now().isoformat()
        }

    except ValueError as e:
        return {
            "success": False,
            "error": f"Data error: {str(e)}",
            "error_code": "DATA_ERROR"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"System error: {str(e)}",
            "error_code": "SYSTEM_ERROR"
        }
```

### Input Validation

```python  theme={null}
def secure_function(user_input: str) -> dict:
    """Secure function implementation"""
    # Input length limit
    if len(user_input) > 1000:
        return {"error": "Input too long"}

    # Dangerous character filtering
    dangerous_chars = ['<', '>', '&', '"', "'"]
    if any(char in user_input for char in dangerous_chars):
        return {"error": "Input contains dangerous characters"}

    # SQL injection protection
    sql_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT']
    if any(keyword in user_input.upper() for keyword in sql_keywords):
        return {"error": "Input contains dangerous keywords"}

    return {"success": True, "processed_input": user_input}
```

### Permission Control

```python  theme={null}
def check_permissions(user_id: str, operation: str) -> bool:
    """Check user permissions"""
    user_permissions = get_user_permissions(user_id)
    return operation in user_permissions

def protected_function(user_id: str, operation: str, data: dict) -> dict:
    """Function requiring permission validation"""
    if not check_permissions(user_id, operation):
        return {
            "success": False,
            "error": "Insufficient permissions",
            "error_code": "PERMISSION_DENIED"
        }

    # Execute operation
    return perform_operation(operation, data)
```

<Tip>
  It is recommended to provide detailed documentation and examples for each function to help the model better understand the function's purpose and usage.
</Tip>

<Warning>
  Function calling involves code execution. Please ensure appropriate security measures are implemented, including input validation, permission control, and error handling.
</Warning>
