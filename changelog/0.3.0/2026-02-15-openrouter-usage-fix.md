# Fix token usage calculation from the OpenRouter provider

- OpenRouter occasionally omits reasoning tokens from completion tokens; the usage metadata calculation compensates for it in both Python and TypeScript (#73).

*Backfilled when the changelog was reorganized by release version; see the git history of the release range for full context.*
