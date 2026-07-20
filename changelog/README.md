# Changelog Details

One file per entry in [../CHANGELOG.md](../CHANGELOG.md). The root file keeps a single brief line per change; the full story lives here, grouped by the release that ships the change.

- File path: `<version>/YYYY-MM-DD-short-slug.md` — one folder per release version, file names matching the entry date.
- Changes not yet released go into the upcoming version's folder; rename the folder during release preparation if the number changes.
- Content: what changed and why, affected modules, and decisions worth keeping (protocol differences, config mappings, migration notes).
- Link the root entry to its file: `- [YYYY-MM-DD] Brief description. ([details](changelog/<version>/YYYY-MM-DD-short-slug.md))`
