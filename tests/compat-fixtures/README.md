# Compatibility Fixtures

This directory contains declarative fixtures used by `npm run test:compat-fixtures`.
Use `npm run test:compat-contract` when you only need contract validation (manifest/docs/runtime-label parity) without executing every fixture case.
Summary artifacts include `summarySchemaVersion` to support stable downstream parsing as telemetry fields evolve.
Summary header generation rejects malformed/duplicate array entries for case and check-label lists, keeping report metadata contract-safe.
Fixture runner validates full summary artifact shape/invariants before writing (schema version, totals/failures coherence, telemetry/result array structure).
Standalone summary artifact validation is available via `scripts/validate-compat-summary.mjs` for post-run/CI artifact checks.
Use `npm run test:compat-summary:verify -- <path-to-summary.json>` to validate existing summary artifacts without re-running fixture generation.
Standalone validator behavior is unit-tested in `scripts/validate-compat-summary.test.js` (success/failure/env fallback paths).
Summary workflow scripts respect `COMPAT_REPORT_DIR` from the caller environment (falling back to `.compat-reports` when unset).
Runner utility tests also cover malformed summary-loading paths and fixtures-summary case-report failure validation for artifact-level error contracts.
CI publishes the generated summary artifact (`compat-summary`) for each run, with full per-case report bundles uploaded on failures.
- Additional mode rules are enforced: `contract` summaries cannot contain case results, and `fixtures` summaries must keep totals aligned with selected cases.
- Fixtures-mode summary validation now also enforces per-result schema + status coherence (`passedCases`/`failedCases`) and selected-case ordering parity for emitted result lists.
- `slowestCases` is contract-validated against emitted fixture results (exact `min(3,total)` length, descending order, and duration parity).
- Result semantics are validated: passed entries must have no mismatches and matched expected/actual exits; failed entries must include mismatch details.
- Summary shape validation is centralized at summary write time, so any future reporter writing `summary.json` inherits the same contract checks.
- Per-case JSON report artifacts are also shape-validated at write time to keep case-report and summary-report contracts aligned.

- `cases.json` is the source of truth for expected outcomes.
  - includes `schemaVersion` for explicit contract evolution (current: `2`).
  - case `name` values must be lowercase kebab-case identifiers.
  - includes `expectedCheckLabels` to lock the compatibility check-label contract.
  - `expectedCheckStatuses` labels must be declared in `expectedCheckLabels`.
  - supports `expectedCheckStatuses`, `expectedDetailIncludes`, and `expectedHintIncludes`.
  - `expectedDetailIncludes` / `expectedHintIncludes` labels must also exist in `expectedCheckStatuses`.
  - supports `openclawExitCode` to simulate non-zero OpenClaw CLI behavior per fixture case.
  - `allowMissingFiles` entries must reference known required fixture paths (typo-safe validation).
  - non-ready OpenClaw simulations (`openclawExitCode != 0` or `openclawSignal`) must assert `expectedCheckStatuses["openclaw CLI available"] = "warn"`.
  - enforces strict expectation consistency (`expectedExitCode` in `0|1`, non-negative warning/error counts, and non-empty `expectedCheckStatuses`).
- Each case references a fixture folder with:
  - `package.json`
  - `SKILL.md`
  - `hooks/clawvault/HOOK.md`
  - `hooks/clawvault/handler.js`

Current fixture scenarios:

- order is intentionally kept in sync with `cases.json` and validated by the fixture runner.
- duplicate scenario bullet entries are rejected by fixture README contract validation.
- `healthy` — expected strict pass.
  - also asserts `ok` status for every declared compatibility check label.
- `broken-openclaw-cli` — warning when openclaw CLI exists but is not runnable.
- `broken-openclaw-signal` — warning when openclaw CLI invocation is terminated by signal.
- `missing-openclaw-cli` — warning when openclaw CLI is missing from PATH.
- `missing-requires-bin` — warning for missing metadata.openclaw.requires.bins.
- `non-auto-profile` — warning for missing --profile auto delegation.
- `missing-events` — error for missing required hook events.
- `missing-package-hook` — error for missing openclaw.hooks registration.
- `missing-skill-openclaw` — warning for missing metadata.openclaw in SKILL.md.
- `unsafe-hook-handler` — warning for unsafe hook execution conventions (execSync, missing shared profile delegation).
- `unsafe-shell-handler` — warning for shell-enabled hook execution options.
- `missing-skill-file` — warning for missing SKILL.md file entirely.
