# Automatic caching for Claude 4.6; tracer timestamps and round index

- Claude 4.6 switched to automatic prompt caching by moving `cache_control` from message content items to a top-level API parameter (#95); Bedrock still uses per-message cache control.
- `UniMessage`/`UniEvent` gained `created_at` timestamps, and the tracer tracks message rounds (#96).

*Backfilled when the changelog was reorganized by release version; see the git history of the release range for full context.*
