# Compatibility Fixtures

This directory contains declarative fixtures used by `npm run test:compat-fixtures`.

- `cases.json` is the source of truth for expected outcomes.
- Each case references a fixture folder with:
  - `package.json`
  - `SKILL.md`
  - `hooks/clawvault/HOOK.md`
  - `hooks/clawvault/handler.js`

Current fixture scenarios:

- `healthy` — expected strict pass.
- `missing-requires-bin` — warning for missing `metadata.openclaw.requires.bins`.
- `non-auto-profile` — warning for missing `--profile auto` delegation.
- `missing-events` — error for missing required hook events.
- `missing-package-hook` — error for missing `openclaw.hooks` registration.
