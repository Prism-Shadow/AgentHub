# Context Caching

The DeepSeek API Context Caching on Disk Technology is enabled by default for all users, allowing them to benefit without needing to modify their code.

Each user request will trigger the construction of a hard disk cache. If subsequent requests have overlapping prefixes with previous requests, the overlapping part will only be fetched from the cache, which counts as a "cache hit."

## Cache Persistence and Hit Rules

A cache hit requires that the corresponding prefix has already been "persisted" (written to the disk cache). Due to the Sliding Window Attention mechanism, the storage and matching of cached prefixes differs from before. Each cached prefix is an independent, complete unit. A subsequent request can only hit the cache if it fully matches a cache prefix unit.

### When cache prefixes are persisted:

1. Persistence at request boundaries: Each request will produce two cache prefix units at the end position of the user input and the end position of the model output. A subsequent request can hit the cache if it fully matches them.
2. Common prefix detection persistence: When the system detects a common prefix across multiple requests, it will persist that common prefix as an independent cache prefix unit. A subsequent request can hit the cache if it fully reuses that cache prefix unit.
3. Persistence at fixed token intervals: For long inputs or long outputs, the system will carve out cache prefix units at fixed token intervals, to avoid long prefixes from being completely uncacheable due to never reaching an end position.

Example 1: A user's first-round request is `A + B`, and the second-round request is `A + B + C`. The second request can fully match the cache prefix unit `A + B`, hitting the cache for `A + B`. See Example 1 below.

Example 2: A user's first-round request is `A + B`, and the second-round request is `A + C`. The second request cannot hit the cache, because `A + C` does not fully match the first round's cache prefix unit (`A + B`). However, at this point the system will detect that the two requests share a common prefix `A`, and persist `A` as a cache prefix unit. When a third-round request `A + D` arrives, it can fully match the cache prefix unit `A`, hitting the cache for `A`. See Example 2 below.

---

### Example 1: Multi-round Conversation

First Request

```text
messages: [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is the capital of China?"}
]
```

Second Request

```text
messages: [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is the capital of China?"},
    {"role": "assistant", "content": "The capital of China is Beijing."},
    {"role": "user", "content": "What is the capital of the United States?"}
]
```

In this example, the second request can fully reuse the cache prefix unit from the first request, which will count as a "cache hit."

### Example 2: Long Text Q&A

First Request

```text
messages: [
    {"role": "system", "content": "You are an experienced financial report analyst..."}
    {"role": "user", "content": "<financial report content>\n\nPlease summarize the key information of this financial report."}
]
```

Second Request

```text
messages: [
    {"role": "system", "content": "You are an experienced financial report analyst..."}
    {"role": "user", "content": "<financial report content>\n\nPlease analyze the profitability of this financial report."}
]
```

Third Request

```text
messages: [
    {"role": "system", "content": "You are an experienced financial report analyst..."}
    {"role": "user", "content": "<financial report content>\n\nPlease analyze the ratio of the company's revenue to expenses."}
]
```

In the above example, the first two requests will not hit the cache. After the first two requests are completed, the system will identify the `system` message + <financial report content> in the `user` message as a cache prefix unit and persist it. In the third request, since it fully matches the previously persisted cache prefix unit, it can hit the cache.

---

## Checking Cache Hit Status

In the response from the DeepSeek API, we have added two fields in the `usage` section to reflect the cache hit status of the request:

1. `prompt_cache_hit_tokens`: The number of tokens in the input of this request that resulted in a cache hit.
2. `prompt_cache_miss_tokens`: The number of tokens in the input of this request that did not result in a cache hit.

## Hard Disk Cache and Output Randomness

The hard disk cache only matches the prefix part of the user's input. The output is still generated through computation and inference, and it is influenced by parameters such as temperature, introducing randomness.

## Additional Notes

1. The cache system works on a "best-effort" basis and does not guarantee a 100% cache hit rate.
2. Cache construction takes seconds. Once the cache is no longer in use, it will be automatically cleared, usually within a few hours to a few days.
