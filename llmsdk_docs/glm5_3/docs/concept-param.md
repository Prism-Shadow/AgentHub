> ## Documentation Index
> Fetch the complete documentation index at: https://docs.z.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Core Parameters

<Tip>
  When interacting with models, you can control the model's output by adjusting different parameters to meet the needs of various scenarios. Understanding these core parameters will help you better utilize the model's capabilities.
</Tip>

## Quick Reference

| Parameter                              | Type    | Default Value         | Description                                                                                   |
| :------------------------------------- | :------ | :-------------------- | :-------------------------------------------------------------------------------------------- |
| [do\_sample](#do_sample)               | Boolean | `true`                | Whether to sample the output to increase diversity.                                           |
| [temperature](#temperature)            | Float   | (Model dependent)     | Controls the randomness of output, higher values are more random.                             |
| [top\_p](#top_p)                       | Float   | (Model dependent)     | Controls diversity through nucleus sampling, recommended to use either this or `temperature`. |
| [max\_tokens](#max_tokens)             | Integer | (Model dependent)     | Limits the maximum number of tokens generated in a single call.                               |
| [stream](#stream)                      | Boolean | `false`               | Whether to return responses in streaming mode.                                                |
| [thinking](#thinking)                  | Object  | `{"type": "enabled"}` | Whether to enable chain-of-thought deep thinking, only supported by `GLM-4.5` and above.      |
| [reasoning\_effort](#reasoning_effort) | String  | `max` `high` `low`    | Controls the model's reasoning effort level, only supported by `GLM-5.2` and above.           |

***

## Parameter Details

### do\_sample

`do_sample` is a boolean value (`true` or `false`) that determines whether to sample the model's output.

* `true` (default): Performs random sampling based on the probability distribution of each token, increasing text diversity and creativity. Suitable for content creation, dialogue, and other scenarios.
* `false`: Uses a greedy strategy, always selecting the token with the highest probability. Provides high deterministic output, suitable for scenarios requiring precise, factual answers.

Best Practices:

* Set to `false` when you need reproducible, deterministic output.
* Set to `true` when you want the model to generate more diverse and interesting content, and use it in combination with `temperature` or `top_p`.

### temperature

The `temperature` parameter controls the randomness of the model's output.

* Lower values (e.g., 0.2): Make the probability distribution more "sharp", resulting in more deterministic and conservative output.
* Higher values (e.g., 0.8): Make the probability distribution more "flat", resulting in more random and diverse output.

Best Practices:

* For scenarios requiring rigor and factual accuracy (such as knowledge Q\&A), it's recommended to use lower `temperature`.
* For scenarios requiring creativity (such as content creation), you can try higher `temperature`.
* It's recommended to use only one of `temperature` and `top_p`.

### top\_p

`top_p` (nucleus sampling) controls diversity by sampling from the smallest set of tokens whose cumulative probability exceeds the threshold.

* Lower values (e.g., 0.2): Limit the sampling range, resulting in more deterministic output.
* Higher values (e.g., 0.9): Expand the sampling range, resulting in more diverse output.

Best Practices:

* If you want to achieve some diversity while ensuring content quality, `top_p` is a good choice (recommended values 0.8-0.95).
* It's generally not recommended to modify both `temperature` and `top_p` simultaneously.

### max\_tokens

`max_tokens` is used to limit the maximum number of tokens the model can generate in a single call. GLM-4.6 supports a maximum output length of 128K, GLM-4.5 supports a maximum output length of 96K, and it's recommended to set it to no less than 1024. Tokens are the basic units of text, typically 1 token equals approximately 0.75 English words or 1.5 Chinese characters. Setting an appropriate max\_tokens can control response length and cost, avoiding overly long outputs. If the model completes its answer before reaching the max\_tokens limit, it will naturally end; if it reaches the limit, the output may be truncated.

* Purpose: Prevents generating overly long text and controls API call costs.
* Note: `max_tokens` limits the length of generated content, not including input.

Best Practices:

* Set `max_tokens` reasonably according to your application scenario. If you need short answers, you can set it to a smaller value (e.g., 50).

Default `max_tokens` and maximum supported `max_tokens` for each model:

| Model Code          | Default max\_tokens | Maximum max\_tokens |
| :------------------ | :-----------------: | :-----------------: |
| glm-5.3             |        65536        |        131072       |
| glm-5.2             |        65536        |        131072       |
| glm-5.1             |        65536        |        131072       |
| glm-5v-turbo        |        65536        |        131072       |
| glm-5-turbo         |        65536        |        131072       |
| glm-5               |        65536        |        131072       |
| glm-4.7             |        65536        |        131072       |
| glm-4.6             |        65536        |        131072       |
| glm-4.6v            |        16384        |        32768        |
| glm-4.6v-flash      |        16384        |        32768        |
| glm-4.6v-flashx     |        16384        |        32768        |
| glm-4.5             |        65536        |        98304        |
| glm-4.5-air         |        65536        |        98304        |
| glm-4.5-x           |        65536        |        98304        |
| glm-4.5-airx        |        65536        |        98304        |
| glm-4.5-flash       |        65536        |        98304        |
| glm-4.5v            |        16384        |        16384        |
| glm-4-32b-0414-128k |        16384        |        16384        |

### stream

`stream` is a boolean value used to control the API's response method.

* `false` (default): Returns the complete response at once, simple to implement but with long waiting times.
* `true`: Returns content in streaming (SSE) mode, significantly improving the experience of real-time interactive applications.

Best Practices:

* For chatbots, real-time code generation, and other applications, it's strongly recommended to set this to `true`.

### thinking

The `thinking` parameter controls whether the model enables "Chain of Thought" for deeper thinking and reasoning.

* Type: Object
* Supported Models: `GLM-4.5` and above

Properties:

* `type` (string):
  * `enabled` (default): Enable chain of thought. `GLM-5.3` uses forced thinking. For other models, `GLM-5.2`, `GLM-5.1`, `GLM-5`, `GLM-5-Turbo`, `GLM-5V-Turbo`, `GLM-4.6`, and `GLM-4.5` auto-determine whether to think, while `GLM-4.7` and `GLM-4.5V` use forced thinking.
  * `disabled`: Disable chain of thought.

Best Practices:

* It's recommended to enable this when you need the model to perform complex reasoning and planning.
* For simple tasks, you can disable it to get faster responses.

### reasoning\_effort

The `reasoning_effort` parameter is used to control the model's reasoning effort level when chain-of-thought thinking is enabled.

* Type: String
* Supported Models: `GLM-5.2` and above
* Allowed values: `max`, `high`, `low`
* Default: `max`

Parameter values:

* `low`: Light thinking
* `high`: Enhanced thinking
* `max`: Deep thinking (default)

Notes:

* The `GLM-5.2` model supports `max`, `xhigh`, `high`, `medium`, `low`, `minimal`, `none`; passing `none` or `minimal` will cause the model to skip thinking; `low` and `medium` will be mapped to `high`; `xhigh` will be mapped to `max`.
* This parameter only takes effect when `thinking.type` is set to `enabled`.
* The `GLM-5.3` model only supports `max`, `high`, `low`.

***

## Related Concepts

<AccordionGroup>
  <Accordion title="Token Usage Calculation">
    Tokens are the basic units for model text processing. Usage calculation includes both input and output parts.

    * **Input Token Count:** The number of tokens contained in the text you send to the model.
    * **Output Token Count:** The number of tokens contained in the text generated by the model.
    * **Total Token Count:** The sum of input and output, usually used as the billing basis.

    You can call the `tokenizer` API to estimate the token count of text.
  </Accordion>

  <Accordion title="Maximum Output Tokens">
    Maximum Output Tokens refers to the maximum number of tokens a model can generate in a single request. It's different from the `max_tokens` parameter - `max_tokens` is the upper limit you set in your request, while Maximum Output Tokens is the architectural limitation of the model itself.

    For example, a model's context window might be 8k tokens, but its maximum output capability might be limited to 4k tokens.
  </Accordion>

  <Accordion title="Context Window">
    The Context Window refers to the total number of tokens a model can process in a single interaction, including all tokens from both **input text** and **generated text**.

    * **Importance:** The context window determines how much historical information the model can "remember". If the total length of input and expected output exceeds the model's context window, the model will be unable to process it.
    * **Note:** Different models have different context window sizes. When conducting long conversations or processing long documents, special attention should be paid to context window limitations.
  </Accordion>

  <Accordion title="Concurrency Limits">
    Concurrency refers to the number of API requests you can initiate simultaneously. This is set by the platform to ensure service stability and fair resource allocation.

    * **Limits:** Different users or subscription plans may have different concurrency quotas.
    * **Overages:** If you exceed the concurrency limit, new requests may fail or need to wait in queue.

    If your application requires high concurrency processing, please check your account limits or contact platform support.
  </Accordion>
</AccordionGroup>
