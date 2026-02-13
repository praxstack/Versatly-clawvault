import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { spawnSync } from 'child_process';
import { fileURLToPath } from 'url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..');
const fixturesRoot = path.join(repoRoot, 'tests', 'compat-fixtures');
const fixtureCasesPath = path.join(fixturesRoot, 'cases.json');
const VALID_CHECK_STATUSES = new Set(['ok', 'warn', 'error']);
const compatReportDir = process.env.COMPAT_REPORT_DIR
  ? path.resolve(process.env.COMPAT_REPORT_DIR)
  : '';

function loadCases() {
  const raw = fs.readFileSync(fixtureCasesPath, 'utf-8');
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error('compat fixture cases must be a non-empty array');
  }

  const names = new Set();
  for (const [index, testCase] of parsed.entries()) {
    if (!testCase || typeof testCase !== 'object') {
      throw new Error(`compat fixture case[${index}] must be an object`);
    }
    if (typeof testCase.name !== 'string' || testCase.name.length === 0) {
      throw new Error(`compat fixture case[${index}] missing name`);
    }
    if (names.has(testCase.name)) {
      throw new Error(`compat fixture case[${index}] duplicates name "${testCase.name}"`);
    }
    names.add(testCase.name);
    if (!Number.isInteger(testCase.expectedExitCode)) {
      throw new Error(`compat fixture case[${index}] missing expectedExitCode`);
    }
    if (!Number.isInteger(testCase.expectedWarnings)) {
      throw new Error(`compat fixture case[${index}] missing expectedWarnings`);
    }
    if (!Number.isInteger(testCase.expectedErrors)) {
      throw new Error(`compat fixture case[${index}] missing expectedErrors`);
    }
    if (!testCase.expectedCheckStatuses || typeof testCase.expectedCheckStatuses !== 'object') {
      throw new Error(`compat fixture case[${index}] missing expectedCheckStatuses`);
    }
    for (const [label, status] of Object.entries(testCase.expectedCheckStatuses)) {
      if (typeof label !== 'string' || !label) {
        throw new Error(`compat fixture case[${index}] has invalid status label`);
      }
      if (!VALID_CHECK_STATUSES.has(status)) {
        throw new Error(`compat fixture case[${index}] has invalid status "${status}" for "${label}"`);
      }
    }

    if (testCase.expectedDetailIncludes !== undefined) {
      if (!testCase.expectedDetailIncludes || typeof testCase.expectedDetailIncludes !== 'object') {
        throw new Error(`compat fixture case[${index}] expectedDetailIncludes must be an object`);
      }
      for (const [label, snippet] of Object.entries(testCase.expectedDetailIncludes)) {
        if (typeof label !== 'string' || !label || typeof snippet !== 'string' || !snippet) {
          throw new Error(`compat fixture case[${index}] has invalid detail expectation`);
        }
      }
    }
  }

  return parsed;
}

function selectCases(cases) {
  const rawSelection = process.env.COMPAT_CASES;
  if (!rawSelection || !rawSelection.trim()) {
    return cases;
  }

  const selected = rawSelection
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);

  const selectedSet = new Set(selected);
  const missing = selected.filter((name) => !cases.some((testCase) => testCase.name === name));
  if (missing.length > 0) {
    throw new Error(`Unknown COMPAT_CASES entries: ${missing.join(', ')}`);
  }

  return cases.filter((testCase) => selectedSet.has(testCase.name));
}

function createOpenClawShim() {
  const shimDir = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-openclaw-shim-'));
  const shimPath = path.join(shimDir, 'openclaw');
  fs.writeFileSync(shimPath, '#!/usr/bin/env bash\nexit 0\n', 'utf-8');
  fs.chmodSync(shimPath, 0o755);
  return { shimDir, shimPath };
}

function ensureCompatReportShape(report, caseName) {
  if (!report || typeof report !== 'object') {
    throw new Error(`fixture=${caseName} emitted non-object JSON report`);
  }
  if (typeof report.generatedAt !== 'string') {
    throw new Error(`fixture=${caseName} report missing generatedAt`);
  }
  if (!Array.isArray(report.checks)) {
    throw new Error(`fixture=${caseName} report missing checks[]`);
  }
  if (typeof report.warnings !== 'number' || typeof report.errors !== 'number') {
    throw new Error(`fixture=${caseName} report missing warnings/errors counts`);
  }
}

function parseCompatReport(stdout, caseName) {
  try {
    const parsed = JSON.parse(stdout);
    ensureCompatReportShape(parsed, caseName);
    return parsed;
  } catch (err) {
    throw new Error(`fixture=${caseName} produced invalid JSON report: ${err?.message || String(err)}`);
  }
}

function ensureReportDir() {
  if (!compatReportDir) return;
  fs.mkdirSync(compatReportDir, { recursive: true });
}

function writeCaseReport(testCase, report) {
  if (!compatReportDir || !report) return;
  const reportPath = path.join(compatReportDir, `${testCase.name}.json`);
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2), 'utf-8');
}

function assertFixtureFiles(caseName, fixturePath) {
  const requiredPaths = [
    'package.json',
    'SKILL.md',
    path.join('hooks', 'clawvault', 'HOOK.md'),
    path.join('hooks', 'clawvault', 'handler.js')
  ];

  const missing = requiredPaths.filter((relativePath) => !fs.existsSync(path.join(fixturePath, relativePath)));
  if (missing.length > 0) {
    throw new Error(`fixture=${caseName} missing required files: ${missing.join(', ')}`);
  }
}

function runCase(testCase, env) {
  const fixturePath = path.join(fixturesRoot, testCase.name);
  assertFixtureFiles(testCase.name, fixturePath);
  const result = spawnSync(
    process.execPath,
    ['./bin/clawvault.js', 'compat', '--strict', '--base-dir', fixturePath, '--json'],
    {
      cwd: repoRoot,
      env,
      encoding: 'utf-8'
    }
  );

  const actualExitCode = result.status ?? 1;
  const exitMatches = actualExitCode === testCase.expectedExitCode;
  let report = null;
  let outputMatches = false;
  let outputError = null;

  try {
    report = parseCompatReport(result.stdout, testCase.name);
    writeCaseReport(testCase, report);
    const warningsMatch = report.warnings === testCase.expectedWarnings;
    const errorsMatch = report.errors === testCase.expectedErrors;
    const statusMatches = Object.entries(testCase.expectedCheckStatuses).every(([label, expectedStatus]) => {
      const check = report.checks.find((candidate) => candidate?.label === label);
      return check?.status === expectedStatus;
    });
    const detailMatches = Object.entries(testCase.expectedDetailIncludes ?? {}).every(([label, expectedSnippet]) => {
      const check = report.checks.find((candidate) => candidate?.label === label);
      return typeof check?.detail === 'string' && check.detail.includes(expectedSnippet);
    });
    outputMatches = warningsMatch && errorsMatch && statusMatches && detailMatches;
  } catch (err) {
    outputError = err;
    outputMatches = false;
  }

  const passed = exitMatches && outputMatches;
  const summary = `${passed ? '✓' : '✗'} fixture=${testCase.name} expectedExit=${testCase.expectedExitCode} actualExit=${actualExitCode}`;
  console.log(summary);

  if (!passed) {
    if (outputError) {
      console.error(outputError.message);
    }
    if (report) {
      console.error(`expected warnings/errors=${testCase.expectedWarnings}/${testCase.expectedErrors} actual=${report.warnings}/${report.errors}`);
    }
    console.error(result.stdout);
    console.error(result.stderr);
  }

  return passed;
}

function main() {
  const cases = selectCases(loadCases());
  ensureReportDir();
  const { shimDir } = createOpenClawShim();
  const env = {
    ...process.env,
    PATH: `${shimDir}:${process.env.PATH ?? ''}`
  };

  try {
    const failures = cases
      .map((testCase) => runCase(testCase, env))
      .filter((passed) => !passed).length;

    if (failures > 0) {
      console.error(`Compatibility fixture check failed: ${failures} case(s).`);
      process.exit(1);
    }

    console.log('Compatibility fixture check passed.');
  } finally {
    fs.rmSync(shimDir, { recursive: true, force: true });
  }
}

main();
