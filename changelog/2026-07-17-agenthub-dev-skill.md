# Add the agenthub-dev skill and the changelog details directory

- Added `skills/agenthub-dev/SKILL.md`, fixing the model-support development workflow: sync official docs into `llmsdk_docs/`, capture a live streaming tool-call exchange with thinking into the git-ignored `api_captures/`, implement paired Python/TypeScript protocol clients with bijective message conversion, and verify with model-scoped e2e tests only.
- The workflow makes four situations hard stops that require asking the user: unclear or unfetchable official docs, a missing provider API key, any live API request error, and non-obvious `UniConfig` key mappings.
- Added `api_captures/` to `.gitignore` as the home for raw API captures; where docs and captures disagree, the capture wins.
- Added the `changelog/` directory: each `CHANGELOG.md` entry keeps one brief line and links to a detail file here (see `changelog/README.md`).
