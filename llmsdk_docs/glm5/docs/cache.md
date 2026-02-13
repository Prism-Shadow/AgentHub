# Context Caching

<Tip>
  Context caching functionality significantly reduces token consumption and response latency by caching repeated context content. When you repeatedly use the same system prompts or conversation history in dialogues, the caching mechanism automatically identifies and reuses this content, thereby improving performance and reducing costs.
</Tip>

## Features

* **Automatic Cache Recognition**: Implicit caching that intelligently identifies repeated context content without manual configuration
* **Significant Cost Reduction**: Cached tokens are billed at lower prices, dramatically saving costs
* **Improved Response Speed**: Reduces processing time for repeated content, accelerating model responses
* **Transparent Billing**: Detailed display of cached token counts in response field `usage.prompt_tokens_details.cached_tokens`
* **Wide Compatibility**: Supports all mainstream models, including GLM-4.6, GLM-4.5 series, etc.

> Context caching works by computing input message content and identifying content that is identical or highly similar to previous requests. When repeated content is detected, the system reuses previous computation results, avoiding redundant token processing.

This mechanism is particularly suitable for the following scenarios:

* System prompt reuse: In multi-turn conversations, system prompts usually remain unchanged, and caching can significantly reduce token consumption for this part.
* Repetitive tasks: For tasks that process similar content with consistent instructions multiple times, caching can improve efficiency.
* Multi-turn conversation history: In complex conversations, historical messages often contain a lot of repeated information, and caching can effectively reduce token usage for this part.

## Code Examples

<Tabs>
  <Tab title="cURL">
    **Basic Caching Example**

    ```bash  theme={null}
    # First request - establish cache
    curl --location 'https://api.z.ai/api/paas/v4/chat/completions' \
    --header 'Authorization: Bearer YOUR_API_KEY' \
    --header 'Content-Type: application/json' \
    --data '{
        "model": "glm-4.7",
        "messages": [
            {
                "role": "system",
                "content": "You are a professional data analyst, skilled at explaining data trends and providing business insights."
            },
            {
                "role": "user",
                "content": "How to analyze user retention rate?"
            }
        ]
    }'
    ```

    **Cache Reuse Example**

    ```bash  theme={null}
    # Second request - reuse system prompt cache
    curl --location 'https://api.z.ai/api/paas/v4/chat/completions' \
    --header 'Authorization: Bearer YOUR_API_KEY' \
    --header 'Content-Type: application/json' \
    --data '{
        "model": "glm-4.7",
        "messages": [
            {
                "role": "system",
                "content": "You are a professional data analyst, skilled at explaining data trends and providing business insights."
            },
            {
                "role": "user",
                "content": "What is funnel analysis?"
            }
        ]
    }'
    ```
  </Tab>

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

    **Basic Conversation Example**

    ```python  theme={null}
    from zai import ZaiClient

    # Initialize client
    client = ZaiClient(api_key='Your API key')

    # First request - establish cache
    response1 = client.chat.completions.create(
        model="glm-4.7",
        messages=[
            {
                "role": "system",
                "content": "You are a professional technical documentation assistant, skilled at explaining complex technical concepts. Please answer user questions with clear and concise language, and provide practical code examples."
            },
            {
                "role": "user",
                "content": "What is RESTful API?"
            }
        ]
    )

    print("First request result:")
    print(f"Reply: {response1.choices[0].message.content}")
    print(f"Total tokens: {response1.usage.total_tokens}")
    print(f"Cached tokens: {response1.usage.prompt_tokens_details.cached_tokens if hasattr(response1.usage, 'prompt_tokens_details') else 0}")

    # Second request - reuse system prompt cache
    response2 = client.chat.completions.create(
        model="glm-4.7",
        messages=[
            {
                "role": "system",
                "content": "You are a professional technical documentation assistant, skilled at explaining complex technical concepts. Please answer user questions with clear and concise language, and provide practical code examples."  # Same system prompt
            },
            {
                "role": "user",
                "content": "What are the differences between GraphQL and RESTful API?"
            }
        ]
    )

    print("\nSecond request result:")
    print(f"Reply: {response2.choices[0].message.content}")
    print(f"Total tokens: {response2.usage.total_tokens}")
    print(f"Cached tokens: {response2.usage.prompt_tokens_details.cached_tokens if hasattr(response2.usage, 'prompt_tokens_details') else 0}")
    ```

    **Long Document Analysis Example**

    ```python  theme={null}
    from zai import ZaiClient

    # Initialize client
    client = ZaiClient(api_key='Your API key')

    # Long document content (simulated)
    long_document = """
    This is a detailed technical specification document that includes system architecture, API design, database structure, and many other aspects.
    The document is very long and contains a lot of technical details and implementation instructions...
    [Large amount of document content omitted here]
    """

    # First analysis - establish document cache
    response1 = client.chat.completions.create(
        model="glm-4.7",
        messages=[
            {
                "role": "system",
                "content": f"Please answer user questions based on the following technical document:\n\n{long_document}"
            },
            {
                "role": "user",
                "content": "What is the main architecture of this system?"
            }
        ]
    )

    print("First analysis:")
    print(f"Total tokens: {response1.usage.total_tokens}")
    print(f"Cached tokens: {response1.usage.prompt_tokens_details.cached_tokens if hasattr(response1.usage, 'prompt_tokens_details') else 0}")

    # Second analysis - reuse document cache
    response2 = client.chat.completions.create(
        model="glm-4.7",
        messages=[
            {
                "role": "system",
                "content": f"Please answer user questions based on the following technical document:\n\n{long_document}"  # Same document content
            },
            {
                "role": "user",
                "content": "What are the characteristics of the API design?"
            }
        ]
    )

    print("\nSecond analysis:")
    print(f"Total tokens: {response2.usage.total_tokens}")
    print(f"Cached tokens: {response2.usage.prompt_tokens_details.cached_tokens if hasattr(response2.usage, 'prompt_tokens_details') else 0}")
    print(f"Cache savings: {response2.usage.prompt_tokens_details.cached_tokens / response2.usage.total_tokens * 100:.1f}%")
    ```

    **Multi-turn Conversation Caching Example**

    ```python  theme={null}
    from zai import ZaiClient

    # Initialize client
    client = ZaiClient(api_key='Your API key')

    # Build conversation history
    conversation_history = [
        {"role": "system", "content": "You are a Python programming assistant, helping users solve programming problems."},
        {"role": "user", "content": "How to create a simple Flask application?"},
        {"role": "assistant", "content": "Creating a Flask application is simple, first install Flask..."},
        {"role": "user", "content": "How to add routes?"},
        {"role": "assistant", "content": "In Flask, add routes using the @app.route decorator..."},
    ]

    # Continue conversation - reuse conversation history cache
    response = client.chat.completions.create(
        model="glm-4.7",
        messages=conversation_history + [
            {"role": "user", "content": "How to handle POST requests?"}
        ]
    )

    print("Conversation reply:")
    print(f"Content: {response.choices[0].message.content}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens if hasattr(response.usage, 'prompt_tokens_details') else 0}")

    # Calculate cache efficiency
    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details.cached_tokens:
        cache_ratio = response.usage.prompt_tokens_details.cached_tokens / response.usage.prompt_tokens * 100
        print(f"Cache hit rate: {cache_ratio:.1f}%")
    ```

    **Batch Processing Optimization Example**

    ````python  theme={null}
    from zai import ZaiClient
    import time

    # Initialize client
    client = ZaiClient(api_key='Your API key')

    # Common system prompt
    system_prompt = """
    You are a professional code review assistant. Please analyze the provided code from the following aspects:
    1. Code quality and readability
    2. Performance optimization suggestions
    3. Security considerations
    4. Best practice recommendations
    Please provide specific improvement suggestions.
    """

    # List of code snippets to review
    code_snippets = [
        "def calculate_sum(numbers): return sum(numbers)",
        "class User: def __init__(self, name): self.name = name",
        "for i in range(len(items)): print(items[i])",
        "if user_input == 'yes' or user_input == 'y': return True"
    ]

    results = []
    total_cached_tokens = 0

    for i, code in enumerate(code_snippets):
        start_time = time.time()

        response = client.chat.completions.create(
            model="glm-4.7",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please review the following code:\n```python\n{code}\n```"}
            ]
        )

        end_time = time.time()

        # Count cache effects
        cached_tokens = 0
        if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details.cached_tokens:
            cached_tokens = response.usage.prompt_tokens_details.cached_tokens
            total_cached_tokens += cached_tokens

        results.append({
            'code': code,
            'review': response.choices[0].message.content,
            'total_tokens': response.usage.total_tokens,
            'cached_tokens': cached_tokens,
            'response_time': end_time - start_time
        })

        print(f"Code snippet {i+1} review completed:")
        print(f"  Response time: {end_time - start_time:.2f}s")
        print(f"  Cached tokens: {cached_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")
        print()

    print(f"Batch processing completed, total cached tokens: {total_cached_tokens}")
    ````
  </Tab>
</Tabs>

Response contains context cache token usage information:

```json  theme={null}
{
  "usage": {
    "prompt_tokens": 1200,
    "completion_tokens": 300,
    "total_tokens": 1500,
    "prompt_tokens_details": {
      "cached_tokens": 800
    }
  }
}
```

## Best Practices

<Tabs>
  <Tab title="System Prompt Optimization">
    Use stable system prompts

    ```python  theme={null}
    # Recommended: Use stable system prompts
    system_prompt = """
    You are a professional technical consultant with the following characteristics:
    - Deep technical background and rich project experience
    - Able to provide accurate and practical technical advice
    - Good at explaining complex concepts in clear and concise language
    Please provide professional technical guidance based on user questions.
    """
    ```
  </Tab>

  <Tab title="Document Content Reuse">
    Use long documents as system messages

    ```python  theme={null}
    # Recommended: Use long documents as system messages
    def create_document_based_chat(document_content, user_question):
        return client.chat.completions.create(
            model="glm-4.7",
            messages=[
                {
                    "role": "system",
                    "content": f"Please answer user questions based on the following document content:\n\n{document_content}"
                },
                {
                    "role": "user",
                    "content": user_question
                }
            ]
        )

    # Multiple calls with the same document, system prompts will be cached
    questions = ["What is the main content of the document?", "What are the key points?", "How to implement these suggestions?"]
    for question in questions:
        response = create_document_based_chat(document_content, question)
        # Second and subsequent calls will hit the cache
    ```
  </Tab>

  <Tab title="Conversation History Management">
    Manage conversation history to improve cache efficiency

    ```python  theme={null}
    class ConversationManager:
        def __init__(self, client, system_prompt):
            self.client = client
            self.system_prompt = system_prompt
            self.history = [{"role": "system", "content": system_prompt}]

        def add_message(self, role, content):
            self.history.append({"role": role, "content": content})

        def get_response(self, user_message):
            # Add user message
            self.add_message("user", user_message)

            # Get reply (conversation history will be cached)
            response = self.client.chat.completions.create(
                model="glm-4.7",
                messages=self.history
            )

            # Add assistant reply to history
            assistant_message = response.choices[0].message.content
            self.add_message("assistant", assistant_message)

            return response

        def get_cache_stats(self, response):
            """Get cache statistics"""
            if hasattr(response.usage, 'prompt_tokens_details'):
                cached = response.usage.prompt_tokens_details.cached_tokens or 0
                total = response.usage.prompt_tokens
                return f"Cache hit: {cached}/{total} ({cached/total*100:.1f}%)"
            return "No cache information"

    # Usage example
    manager = ConversationManager(client, "You are a programming assistant...")
    response1 = manager.get_response("How to learn Python?")
    response2 = manager.get_response("Recommend some learning resources")  # Will reuse previous conversation cache
    ```
  </Tab>
</Tabs>

## Use Cases

<CardGroup cols={2}>
  <Card title="Multi-turn Conversations" icon="headset">
    * Intelligent customer service systems
    * Personal assistant services
  </Card>

  <Card title="Batch Processing" icon="cubes">
    * Code review batch processing
    * Content batch analysis
  </Card>

  <Card title="Template Applications" icon="rectangle-list">
    * Report generation templates
    * Standardized process handling
  </Card>

  <Card title="Education and Training" icon="glasses">
    * Homework grading assistance
    * Learning material analysis
  </Card>
</CardGroup>

## Important Notes

<Tabs>
  <Tab title="Understanding Cache Mechanism">
    * Caching is automatically triggered based on content similarity, no manual configuration required
    * Identical content has the highest cache hit rate
    * Minor formatting differences may affect cache effectiveness
    * Cache has reasonable time limits, will recalculate after expiration
  </Tab>

  <Tab title="Cost Optimization Suggestions">
    * Cached tokens are billed at lower prices
    * Long documents and repeated content have the most significant cache effects
    * Design system prompts reasonably to improve reuse rates
    * Monitor cache hit rates and optimize usage patterns
  </Tab>

  <Tab title="Performance Considerations">
    * Caching can significantly improve response speed
    * First request to establish cache may be slightly slower
    * Manage conversation history length reasonably
    * Avoid overly frequent content changes
  </Tab>

  <Tab title="Best Practices">
    * Use stable system prompt templates
    * Process long documents as system messages
    * Organize conversation history structure reasonably
    * Regularly analyze cache effectiveness and optimize
  </Tab>
</Tabs>

## Billing Information

Context caching uses a differentiated billing strategy:

* New content tokens: Billed at standard prices
* Cache hit tokens: Billed at discounted prices (usually 50% of standard price)
* Output tokens: Billed at standard prices

Billing example:

```
Assuming standard price is 0.01 /1K tokens:

Request details:
- Total input tokens: 2000
- Cache hit tokens: 1200
- New content tokens: 800
- Output tokens: 500

Billing calculation:
- New content cost: 800 × 0.01/1000 = 0.008
- Cache cost: 1200 × 0.005/1000 = 0.006
- Output cost: 500 × 0.01/1000 = 0.005
- Total cost: 0.019

Compared to no cache (2500 × 0.01/1000 = 0.025), saves 24%
```


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.z.ai/llms.txt
