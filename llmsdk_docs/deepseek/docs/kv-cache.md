# Context Caching

DeepSeek enables disk-based context caching by default for all users. No code changes are required to benefit from it.

Each request can build cache prefix units on disk. Later requests with matching prefixes can reuse the cached prefix, which is counted as a cache hit.

## Cache Persistence and Hit Rules

A request can hit the cache only when its prefix fully matches a persisted cache prefix unit. With Sliding Window Attention, each cached prefix is an independent complete unit; partial matches against an existing unit do not count as hits.

DeepSeek persists cache prefix units in three ways:

1. Request-boundary persistence: each request can persist prefix units at the end of the user input and at the end of the model output.
2. Common-prefix detection: when multiple requests share a common prefix, DeepSeek can persist that common prefix as its own cache prefix unit.
3. Fixed token intervals: for long inputs or outputs, DeepSeek can create cache prefix units at fixed token intervals so long prefixes can still become cacheable.

## Example: Multi-round Conversation

First request:

```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant"
  },
  {
    "role": "user",
    "content": "What is the capital of China?"
  }
]
```

Second request:

```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant"
  },
  {
    "role": "user",
    "content": "What is the capital of China?"
  },
  {
    "role": "assistant",
    "content": "The capital of China is Beijing."
  },
  {
    "role": "user",
    "content": "What is the capital of the United States?"
  }
]
```

The second request fully reuses the first request as a prefix, so that prefix can count as a cache hit.

## Example: Long Text Q&A

First request:

```json
[
  {
    "role": "system",
    "content": "You are an experienced financial report analyst..."
  },
  {
    "role": "user",
    "content": "<financial report content>\n\nPlease summarize the key information of this financial report."
  }
]
```

Second request:

```json
[
  {
    "role": "system",
    "content": "You are an experienced financial report analyst..."
  },
  {
    "role": "user",
    "content": "<financial report content>\n\nPlease analyze the profitability of this financial report."
  }
]
```

Third request:

```json
[
  {
    "role": "system",
    "content": "You are an experienced financial report analyst..."
  },
  {
    "role": "user",
    "content": "<financial report content>\n\nPlease analyze the ratio of the company's revenue to expenses."
  }
]
```

The first two requests do not hit the cache because their full prefixes differ. After both requests complete, DeepSeek can detect and persist the shared `system` message plus the financial report content as a cache prefix unit. The third request can then hit that persisted prefix.

## Checking Cache Hit Status

DeepSeek includes cache status fields in the response `usage` object:

```json
{
  "usage": {
    "prompt_cache_hit_tokens": 1280,
    "prompt_cache_miss_tokens": 320
  }
}
```

- `prompt_cache_hit_tokens`: input tokens served from cache.
- `prompt_cache_miss_tokens`: input tokens not served from cache.

## Output Randomness

The disk cache matches only the prefix part of the user input. The model output is still generated through normal inference, so generation parameters such as `temperature` can still affect the output.

## Notes

- Context caching is best effort and does not guarantee a 100% hit rate.
- Cache construction can take seconds.
- Unused cache entries are cleared automatically, usually within a few hours to a few days.
