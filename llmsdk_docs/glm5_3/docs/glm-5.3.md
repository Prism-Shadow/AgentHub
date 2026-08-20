> ## Documentation Index
> Fetch the complete documentation index at: https://docs.z.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# GLM-5.3

<Tip>
  GLM-5.3 is now available to all GLM Coding Plan users, with 50% better coding performance! [**Subscribe now**](https://z.ai/subscribe)
</Tip>

## Overview

**GLM-5.3** is Z.ai's latest flagship model, delivering comprehensive advancements in complex software engineering and agent capabilities. It uses the same base model as GLM-5.2, with all improvements driven by post-training. Compared with GLM-5.2, it delivers significantly stronger performance in complex programming and long-horizon tasks:

**Stronger Coding Capability**

GLM-5.3 delivers a significant improvement in coding capabilities, achieving a 50% performance gain over GLM-5.2 on Z.ai Code Bench. It also reaches state-of-the-art (SOTA) performance among open-source models on public benchmarks, including Terminal Bench 3.0 and Agents’ Last Exam (CLI).

**Emergent Cyber Capability**

As the scale of post-training continues to expand, the model’s cybersecurity capabilities have improved at a rate that exceeds expectations. GLM-5.3 achieves the best performance to date on the CyberGym vulnerability discovery benchmark. Moreover, the deeper it progresses along the vulnerability exploitation chain, the more pronounced its gains over GLM-5.2 become, with scores on vulnerability exploitation benchmarks exceeding twice those of GLM-5.2.

[↗ Blog](https://z.ai/blog/glm-5.3)

## Feature Changes

GLM-5.3 currently supports text-only inputs, with a 1M-token context window and a maximum output length of 128K tokens.
GLM-5.3 always operates with reasoning enabled and supports three reasoning effort levels: `low`, `high`, and `max`. Disabling reasoning is no longer supported. The reasoning parameters are described below:

| Parameter          | Values               | Default   | Description                                                                        |
| :----------------- | :------------------- | :-------- | :--------------------------------------------------------------------------------- |
| `thinking.type`    | `enabled`            | `enabled` | Supports reasoning only; disabling reasoning is not supported.                     |
| `reasoning_effort` | `low`, `high`, `max` | `max`     | `low` – Lightweight Reasoning; `high` – Enhanced Reasoning; `max` – Deep Reasoning |

<Note>
  Migration Notice: If your application currently uses **`thinking.type: "disabled"`**, please change it to **`enabled`** and set **`reasoning_effort`** to **`low`** before updating the model ID to **`glm-5.3`**. Otherwise, the request will fail.
</Note>

For complex tasks such as coding, we recommend using `max`. An example is provided below:

```json theme={null}
{
  "model": "glm-5.3",
  "thinking": { "type": "enabled" },
  "reasoning_effort": "max"
}
```

For a detailed introduction to GLM-5.3’s advanced reasoning capabilities, please refer to the documentation: [Deep Thinking](https://docs.z.ai/guides/capabilities/thinking)

## How to Use

#### **Model API**

The supported protocols and endpoints are as follows:

| Protocol                        | Base URL                               |
| :------------------------------ | :------------------------------------- |
| OpenAI Chat Completion Protocol | `	https://api.z.ai/api/coding/paas/v4` |
| OpenAI Response Protocol        | `	https://api.z.ai/api/v1`             |
| Anthropic Message Protocol      | `https://api.z.ai/api/anthropic`       |

[↗ API Documentation](https://docs.z.ai/api-reference/llm/chat-completion)：Learn how to call the API

<Note>
  If you have previously subscribed to a GLM Coding Plan, including an expired subscription, you can currently access the model API only through the OpenAI Chat Completion-compatible protocol. We will continue to improve and optimize this in upcoming iterations.
</Note>

#### **GLM Coding Plan**

GLM-5.3 is now fully available, and you can use it with your preferred coding agents. [Guide](https://docs.z.ai/devpack/latest-model)

The new GLM Coding Plan adopts a points-based quota system, providing transparent and predictable usage. Model calls made during off-peak hours, including all day on weekends, consume only 50% of the standard points.
Subscribe now: [Individual Plan](https://z.ai/subscribe)、[Team Plan](https://z.ai/subscribe?plantype=team).

## Capability Support

* [Thinking Mode](https://docs.z.ai/guides/capabilities/thinking-mode)：Provides multiple thinking modes to accommodate different task requirements
* [Streaming Messages](https://docs.z.ai/guides/capabilities/streaming)：Supports real-time streaming responses for an enhanced user interaction experience
* [Function Calling](https://docs.z.ai/guides/capabilities/function-calling)：Provides powerful tool-calling capabilities and supports integration with a wide range of external tools
* [Context Caching](https://docs.z.ai/guides/capabilities/cache)：Uses an intelligent caching mechanism to optimize performance in long-context conversations
* [Structured Output](https://docs.z.ai/guides/capabilities/struct-output)：Supports structured output formats such as JSON for seamless system integration

## Key Advancements

GLM-5.3 delivers significant improvements across complex software engineering, terminal operations, and a broader range of real-world Agent tasks.

![Description](https://cdn.bigmodel.cn/markdown/1786683281203image.png?attname=image.png)

#### Stronger Coding

For GLM-5.3, we pushed environment scaling toward tasks that look less like coding exercises and more like real units of expert work. The environments now cover a much broader range of production workflows, with tasks designed around how engineering and research work is actually carried out in practice. Some represent several days of work for an experienced engineer.

In an ML infrastructure task, for example, the model may be given the same working environment as an engineer, with access to compute clusters, storage systems, internal documentation, codebases, and experiment results. It must diagnose bottlenecks across the training stack, implement optimizations, run experiments, and deliver a measurable end-to-end speedup while preserving correctness. Training on environments at this level pushes the model toward taking ownership of substantial work end to end, rather than relying on users to decompose the problem and supervise each step.

As agent capability improves, much of the difficulty in scaling post-training moves from the model to the environment. A useful task environment has to be executable, verifiable, and close to real professional work — and we need many of them, not a handful of hand-built ones. To scale this process, we built pipelines that synthesize environments end to end, and for a subset of tasks, the RL reward signal as well. Research agents collect task patterns from real work and turn them into runnable long-horizon environments with multi-step dependencies and hidden state; a judge agent then attempts each task to verify that it is actually solvable. Verifiers are synthesized without access to the reference solution, while solver trajectories are used to discover and close reward shortcuts. A verifier that passes oracle, no-op, and unsolved-state checks produces a binary reward reliable enough to train on directly.

It carries over the RL strategies introduced in GLM-5.2, including SAO with compaction, which helps these gains hold on long-horizon tasks rather than only on short ones. The effect shows up across both coding and general agent tasks. GLM-5.3 improves from 4.6 to 28.3 on Terminal-Bench 3.0, from 46.2 to 66.9 on DeepSWE v1.1, and from 23.8 to 28.5 on Agents' Last Exam. These pipelines still require a meaningful amount of human-in-the-loop work; making environment generation and verification more autonomous is one of the next steps.

Beyond public benchmarks, we introduce Z.ai Code Bench, an in-house benchmark designed to evaluate coding agents under realistic user scenarios. It covers diverse task categories and places agents in complex local development environments. At different effort levels, we evaluate agents along two dimensions: end-to-end task completion rate and fine-grained checklist accuracy. As a private benchmark, Z.ai Code Bench also reduces the risk of contamination from public test sets and gives us a more faithful measure of real-world user experience.

![Description](https://cdn.bigmodel.cn/markdown/1786683377150image.png?attname=image.png)

As shown in the figure, GLM-5.3 improves both performance and token efficiency. It delivers markedly stronger agentic coding results than GLM-5.2 at every effort level while consuming fewer output tokens. At Max effort, GLM-5.3 reaches 34.5% at roughly 75K output tokens per task, compared with 23.4% at 96K for GLM-5.2. The same shift holds against closed models. At High effort, GLM-5.3 reaches 31.4% at around 50K output tokens, surpassing Claude Opus 4.8 at 29.5% with 120K. GLM-5.3 remains behind Claude Fable 5, which reaches 39.5% at Max effort.

#### Emergent Cyber Capability

As part of post-training, we introduced vulnerability discovery data and environments into the training mix. We expected this to make the model better at finding and reasoning about vulnerabilities. What surprised us was how quickly the capability continued to develop as training scaled. GLM-5.3 did not simply become better at identifying isolated flaws: it began to reason across multiple stages of exploitation, forming coherent plans for complete exploitation chains.

We evaluate GLM-5.3 across three benchmarks covering different stages of vulnerability analysis and exploitation.

![Description](https://cdn.bigmodel.cn/markdown/1786683405531image.png?attname=image.png)

We evaluate GLM-5.3 across three benchmarks covering different stages of vulnerability analysis and exploitation. On CyberGym, which starts from white-box source code and tests whether the model can identify and validate vulnerabilities by triggering faults, GLM-5.3 scores 84.5%, up from GLM-5.2's 77.2% — the best result on the benchmark, ahead of Mythos 5 (83.8%) and GPT-5.6 Sol (83.6%). On ExploitBench, which requires deeper reasoning about real vulnerabilities and their exploitation, GLM-5.3 reaches 54.4%, more than doubling GLM-5.2's 24.4%, while Mythos 5 and GPT-5.6 Sol score 78.0% and 76.5%, respectively. On ExploitGym, which measures how many exploitation tasks a model can complete under time-normalized budgets, GLM-5.3 completes 105 tasks within two hours and 130 within six hours, compared with 29 and 39 for GLM-5.2; budgets are normalized across models using per-model throughput figures, detailed in the footnotes. Mythos 5 remains well ahead at 181 and 247 tasks.

The pattern across the three is consistent: the further up the exploitation chain a benchmark sits, the larger the gain from GLM-5.2 — and also the wider the remaining gap to the closed frontier. Capability is growing fastest exactly where we are furthest behind.

We then tested whether these capabilities transfer beyond controlled benchmarks. Since GLM-5.2, we have been working with several security teams in China to run our models against real-world codebases. After expert review, screening, and deduplication, the model identified 2,436 vulnerabilities across 269 projects, including 1,097 medium-to-high severity issues. The findings span system kernels, operating systems, browser engines, open-source infrastructure, web applications, and network protocols. Many had remained unnoticed for years or even decades, with the oldest dating back roughly 40 years.

This work has since grown into an ongoing disclosure effort. We built the Z.ai Security Disclosure Ledger to maintain a public record of the findings as they move through the disclosure process. The ledger is continuously updated as new vulnerabilities are reviewed and disclosed, distinguishing issues that have already been made public from those that are still under disclosure. For disclosed issues, it records information including the affected project, severity, CVE where available, and how long the vulnerability had remained in the codebase.

## Quick Start

The following is a full sample code to help you onboard GLM-5.3 with ease.

<Tabs>
  <Tab title="cURL">
    **Basic Call**

    ```bash theme={null}
    curl -X POST "https://api.z.ai/api/paas/v4/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer your-api-key" \
    -d '{
      "model": "glm-5.3",
      "messages": [
        {
          "role": "system",
          "content": "You are a senior full-stack software engineer, proficient in frontend development, backend architecture design, and modern web technology stacks."
        },
        {
          "role": "user",
          "content": "Design and build a personal blog website for me, including a homepage, article list page, and article detail page, using React + Node.js technology stack."
        }
      ],
      "thinking": {
        "type": "enabled"
      },
      "reasoning_effort": "max",
      "max_tokens": 4096,
      "temperature": 1.0
    }'
    ```

    **Streaming Call**

    ```bash theme={null}
    curl -X POST "https://api.z.ai/api/paas/v4/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer your-api-key" \
    -d '{
      "model": "glm-5.3",
      "messages": [
        {
          "role": "system",
          "content": "You are a senior full-stack software engineer, proficient in frontend development, backend architecture design, and modern web technology stacks."
        },
        {
          "role": "user",
          "content": "Design and build a personal blog website for me, including a homepage, article list page, and article detail page, using React + Node.js technology stack."
        }
      ],
      "thinking": {
        "type": "enabled"
      },
      "reasoning_effort": "max",
      "stream": true,
      "max_tokens": 4096,
      "temperature": 1.0
    }'
    ```
  </Tab>

  <Tab title="Official Python SDK">
    **Install SDK**

    ```bash theme={null}
    # Install latest version
    pip install zai-sdk

    # Or specify version
    pip install zai-sdk==0.2.3
    ```

    **Verify Installation**

    ```python theme={null}
    import zai

    print(zai.__version__)
    ```

    **Basic Call**

    ```python theme={null}
    from zai import ZaiClient

    client = ZaiClient(api_key="your-api-key")  # Your API Key

    response = client.chat.completions.create(
        model="glm-5.3",
        messages=[
            {
                "role": "system",
                "content": "You are a senior full-stack software engineer, proficient in frontend development, backend architecture design, and modern web technology stacks.",
            },
            {
                "role": "user",
                "content": "Design and build a personal blog website for me, including a homepage, article list page, and article detail page, using React + Node.js technology stack.",
            },
        ],
        thinking={
            "type": "enabled",  # Optional: "enabled" only, reasoning is always enabled
        },
        reasoning_effort="max",
        max_tokens=4096,
        temperature=1.0,
    )

    # Get complete response
    print(response.choices[0].message)
    ```

    **Streaming Call**

    ```python theme={null}
    from zai import ZaiClient

    client = ZaiClient(api_key="your-api-key")  # Your API Key

    response = client.chat.completions.create(
        model="glm-5.3",
        messages=[
            {
                "role": "system",
                "content": "You are a senior full-stack software engineer, proficient in frontend development, backend architecture design, and modern web technology stacks.",
            },
            {
                "role": "user",
                "content": "Design and build a personal blog website for me, including a homepage, article list page, and article detail page, using React + Node.js technology stack.",
            },
        ],
        thinking={
            "type": "enabled",  # Optional: "enabled" only, reasoning is always enabled
        },
        reasoning_effort="max",
        stream=True,
        max_tokens=4096,
        temperature=1.0,
    )

    # Stream response
    for chunk in response:
        if chunk.choices[0].delta.reasoning_content:
            print(chunk.choices[0].delta.reasoning_content, end="", flush=True)

        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    ```
  </Tab>

  <Tab title="Official Java SDK">
    **Install SDK**

    **Maven**

    ```xml theme={null}
    <dependency>
        <groupId>ai.z.openapi</groupId>
        <artifactId>zai-sdk</artifactId>
        <version>0.3.5</version>
    </dependency>
    ```

    **Gradle (Groovy)**

    ```groovy theme={null}
    implementation 'ai.z.openapi:zai-sdk:0.3.5'
    ```

    **Basic Call**

    ```java theme={null}
    import ai.z.openapi.ZaiClient;
    import ai.z.openapi.service.model.ChatCompletionCreateParams;
    import ai.z.openapi.service.model.ChatCompletionResponse;
    import ai.z.openapi.service.model.ChatMessage;
    import ai.z.openapi.service.model.ChatMessageRole;
    import ai.z.openapi.service.model.ChatThinking;
    import java.util.Arrays;

    public class BasicChat {
        public static void main(String[] args) {
            // Initialize client
            ZaiClient client = ZaiClient.builder().ofZAI().apiKey("your-api-key").build();

            // Create chat completion request
            ChatCompletionCreateParams request = ChatCompletionCreateParams.builder()
                .model("glm-5.3")
                .messages(
                    Arrays.asList(
                        ChatMessage.builder()
                            .role(ChatMessageRole.SYSTEM.value())
                            .content(
                                "You are a senior full-stack software engineer, proficient in frontend development, backend architecture design, and modern web technology stacks.")
                            .build(),
                        ChatMessage.builder()
                            .role(ChatMessageRole.USER.value())
                            .content(
                                "Design and build a personal blog website for me, including a homepage, article list page, and article detail page, using React + Node.js technology stack.")
                            .build()))
                .thinking(ChatThinking.builder().type("enabled").build())
                .reasoningEffort("max")
                .maxTokens(4096)
                .temperature(1.0f)
                .build();

            // Send request
            ChatCompletionResponse response = client.chat().createChatCompletion(request);

            // Get response
            if (response.isSuccess()) {
                Object reply = response.getData().getChoices().get(0).getMessage();
                System.out.println("AI Response: " + reply);
            } else {
                System.err.println("Error: " + response.getMsg());
            }
        }
    }
    ```

    **Streaming Call**

    ```java theme={null}
    import ai.z.openapi.ZaiClient;
    import ai.z.openapi.service.model.ChatCompletionCreateParams;
    import ai.z.openapi.service.model.ChatCompletionResponse;
    import ai.z.openapi.service.model.ChatMessage;
    import ai.z.openapi.service.model.ChatMessageRole;
    import ai.z.openapi.service.model.ChatThinking;
    import ai.z.openapi.service.model.Delta;
    import java.util.Arrays;

    public class StreamingChat {
        public static void main(String[] args) {
            // Initialize client
            ZaiClient client = ZaiClient.builder().ofZAI().apiKey("your-api-key").build();

            // Create streaming chat completion request
            ChatCompletionCreateParams request = ChatCompletionCreateParams.builder()
                .model("glm-5.3")
                .messages(
                    Arrays.asList(
                        ChatMessage.builder()
                            .role(ChatMessageRole.SYSTEM.value())
                            .content(
                                "You are a senior full-stack software engineer, proficient in frontend development, backend architecture design, and modern web technology stacks.")
                            .build(),
                        ChatMessage.builder()
                            .role(ChatMessageRole.USER.value())
                            .content(
                                "Design and build a personal blog website for me, including a homepage, article list page, and article detail page, using React + Node.js technology stack.")
                            .build()))
                .thinking(ChatThinking.builder().type("enabled").build())
                .reasoningEffort("max")
                .stream(true) // Enable streaming output
                .maxTokens(4096)
                .temperature(1.0f)
                .build();

            ChatCompletionResponse response = client.chat().createChatCompletion(request);

            if (response.isSuccess()) {
                response.getFlowable()
                    .subscribe(
                        // Process streaming message data
                        data -> {
                            if (data.getChoices() != null && !data.getChoices().isEmpty()) {
                                Delta delta = data.getChoices().get(0).getDelta();
                                System.out.print(delta + "\n");
                            }
                        },
                        // Process streaming response error
                        error -> System.err.println("\nStream error: " + error.getMessage()),
                        // Process streaming response completion event
                        () -> System.out.println("\nStreaming response completed"));
            } else {
                System.err.println("Error: " + response.getMsg());
            }
        }
    }
    ```
  </Tab>

  <Tab title="OpenAI Python SDK">
    **Install SDK**

    ```bash theme={null}
    # Install or upgrade to latest version
    pip install --upgrade 'openai>=1.0'
    ```

    **Verify Installation**

    ```python theme={null}
    python -c "import openai; print(openai.__version__)"
    ```

    **Usage Example**

    ```python theme={null}
    from openai import OpenAI

    client = OpenAI(
        api_key="your-Z.AI-api-key",
        base_url="https://api.z.ai/api/paas/v4/",
    )

    completion = client.chat.completions.create(
        model="glm-5.3",
        messages=[
            {"role": "system", "content": "You are a senior full-stack software engineer, proficient in frontend development, backend architecture design, and modern web technology stacks."},
            {
                "role": "user",
                "content": "Design and build a personal blog website for me, including a homepage, article list page, and article detail page, using React + Node.js technology stack.",
            },
        ],
    )

    print(completion.choices[0].message.content)
    ```
  </Tab>
</Tabs>
