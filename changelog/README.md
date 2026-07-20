# Changelog Details

The root [../CHANGELOG.md](../CHANGELOG.md) keeps one brief line per release. Each release owns a folder here:

- `<version>/README.md` — the release summary: one brief line per change, each linking its detail file, e.g. `- [YYYY-MM-DD] Brief description. ([details](YYYY-MM-DD-short-slug.md))`
- `<version>/YYYY-MM-DD-short-slug.md` — one detail file per entry, named by the entry date: what changed and why, affected modules, and decisions worth keeping (protocol differences, config mappings, migration notes).
- Changes not yet released go into the upcoming version's folder; during release preparation, rename the folder if the number changed and add the release line to the root file.
