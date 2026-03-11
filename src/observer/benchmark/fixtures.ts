import * as fs from 'fs';
import * as path from 'path';
import type { ObserverBenchmarkFixture, ObserverBenchmarkFixtureConfig } from './types.js';

function readJsonConfig(configPath: string): ObserverBenchmarkFixtureConfig {
  if (!fs.existsSync(configPath)) {
    return {};
  }
  const raw = fs.readFileSync(configPath, 'utf-8').trim();
  if (!raw) {
    return {};
  }
  const parsed = JSON.parse(raw) as unknown;
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`Invalid fixture config (expected object): ${configPath}`);
  }
  return parsed as ObserverBenchmarkFixtureConfig;
}

function ensureFile(filePath: string, label: string): void {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Missing ${label}: ${filePath}`);
  }
  const stat = fs.statSync(filePath);
  if (!stat.isFile()) {
    throw new Error(`${label} is not a file: ${filePath}`);
  }
}

export function loadObserverBenchmarkFixtures(fixturesDir: string): ObserverBenchmarkFixture[] {
  const resolvedRoot = path.resolve(fixturesDir);
  if (!fs.existsSync(resolvedRoot)) {
    throw new Error(`Fixtures directory not found: ${resolvedRoot}`);
  }
  if (!fs.statSync(resolvedRoot).isDirectory()) {
    throw new Error(`Fixtures path is not a directory: ${resolvedRoot}`);
  }

  const fixtureDirs = fs.readdirSync(resolvedRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort((left, right) => left.localeCompare(right));

  const fixtures: ObserverBenchmarkFixture[] = [];
  for (const fixtureId of fixtureDirs) {
    const fixtureDir = path.join(resolvedRoot, fixtureId);
    const transcriptPath = path.join(fixtureDir, 'transcript.md');
    const expectedPath = path.join(fixtureDir, 'expected.md');
    const configPath = path.join(fixtureDir, 'config.json');

    ensureFile(transcriptPath, 'transcript fixture');
    ensureFile(expectedPath, 'expected observations fixture');

    const transcript = fs.readFileSync(transcriptPath, 'utf-8');
    const expected = fs.readFileSync(expectedPath, 'utf-8');
    const config = readJsonConfig(configPath);

    fixtures.push({
      id: fixtureId,
      dir: fixtureDir,
      transcriptPath,
      expectedPath,
      transcript,
      expected,
      config
    });
  }

  if (fixtures.length === 0) {
    throw new Error(`No fixture directories found in: ${resolvedRoot}`);
  }

  return fixtures;
}
