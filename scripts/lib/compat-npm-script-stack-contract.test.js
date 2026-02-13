import { describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

function loadPackageScripts() {
  const packageJsonPath = path.resolve(process.cwd(), 'package.json');
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf-8'));
  return packageJson.scripts ?? {};
}

function expectContainsInOrder(value, parts, label) {
  let cursor = -1;
  for (const part of parts) {
    const nextIndex = value.indexOf(part, cursor + 1);
    expect(nextIndex, `${label} is missing segment: ${part}`).toBeGreaterThanOrEqual(0);
    cursor = nextIndex;
  }
}

describe('compat npm script stack contracts', () => {
  it('keeps dedicated artifact CLI drift script wired to both validator suites', () => {
    const scripts = loadPackageScripts();
    const cliDriftScript = scripts['test:compat-artifact-cli-drift:fast'];
    expect(typeof cliDriftScript).toBe('string');
    expect(cliDriftScript).toContain('scripts/validate-compat-artifact-bundle-manifest.test.js');
    expect(cliDriftScript).toContain('scripts/validate-compat-artifact-bundle.test.js');
  });

  it('keeps fast artifact stack ordering aligned with required contract gates', () => {
    const scripts = loadPackageScripts();
    const artifactStackScript = scripts['test:compat-artifact-stack:fast'];
    expect(typeof artifactStackScript).toBe('string');
    expectContainsInOrder(
      artifactStackScript,
      [
        'npm run test:compat-artifact-alignment:fast',
        'npm run test:compat-artifact-cli-drift:fast',
        'npm run test:compat-artifact-bundle:manifest:schema',
        'npm run test:compat-artifact-bundle:manifest:verify:report',
        'npm run test:compat-artifact-bundle:manifest:verify:schema',
        'npm run test:compat-artifact-bundle:verify:report',
        'npm run test:compat-artifact-bundle:verify:schema'
      ],
      'test:compat-artifact-stack:fast'
    );
  });

  it('keeps fast summary stack chained through report stack', () => {
    const scripts = loadPackageScripts();
    const summaryFastScript = scripts['test:compat-summary:fast'];
    expect(typeof summaryFastScript).toBe('string');
    expectContainsInOrder(
      summaryFastScript,
      [
        'npm run test:compat-fixtures:fast',
        'node scripts/validate-compat-summary.mjs --out',
        'npm run test:compat-report-stack:fast'
      ],
      'test:compat-summary:fast'
    );
  });
});
