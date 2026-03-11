import { afterEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { runObserverBenchmark } from './runner.js';

function makeFixtureRoot(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-observer-benchmark-'));
}

function writeFixture(
  root: string,
  name: string,
  payload: {
    transcript: string;
    expected: string;
    config?: Record<string, unknown>;
  }
): void {
  const dir = path.join(root, name);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'transcript.md'), payload.transcript, 'utf-8');
  fs.writeFileSync(path.join(dir, 'expected.md'), payload.expected, 'utf-8');
  if (payload.config) {
    fs.writeFileSync(path.join(dir, 'config.json'), JSON.stringify(payload.config, null, 2), 'utf-8');
  }
}

describe('runObserverBenchmark', () => {
  const roots: string[] = [];

  afterEach(() => {
    for (const root of roots) {
      fs.rmSync(root, { recursive: true, force: true });
    }
    roots.length = 0;
  });

  it('loads fixtures, runs compressor in mock mode, and scores outputs', async () => {
    const fixturesRoot = makeFixtureRoot();
    roots.push(fixturesRoot);

    writeFixture(fixturesRoot, 'coding-session', {
      transcript: [
        'user: We decided to ship PostgreSQL first.',
        'assistant: TODO: publish rollback guide.',
        'user: Client approved launch.'
      ].join('\n'),
      expected: [
        '## 2026-03-10',
        '',
        '- [decision|c=0.95|i=0.90] 10:00 decided to ship PostgreSQL first',
        '- [todo|c=0.88|i=0.72] 10:05 TODO: publish rollback guide',
        '- [commitment|c=0.84|i=0.68] 10:10 Client approved launch'
      ].join('\n'),
      config: {
        now: '2026-03-10T10:10:00.000Z'
      }
    });

    const report = await runObserverBenchmark({
      fixturesDir: fixturesRoot,
      provider: 'mock'
    });

    expect(report.fixtureCount).toBe(1);
    expect(report.fixtures[0]?.fixtureId).toBe('coding-session');
    expect(report.fixtures[0]?.provider).toBe('mock');
    expect(report.fixtures[0]?.actualCount).toBeGreaterThan(0);
    expect(report.fixtures[0]?.scoring.metrics.precision).toBeGreaterThanOrEqual(0);
    expect(report.aggregate.overall).toBeGreaterThanOrEqual(0);
    expect(report.aggregate.overall).toBeLessThanOrEqual(1);
  });

  it('honors fixture-level scoring overrides', async () => {
    const fixturesRoot = makeFixtureRoot();
    roots.push(fixturesRoot);

    writeFixture(fixturesRoot, 'maintenance', {
      transcript: [
        'user: Routine maintenance complete.',
        'assistant: no incidents.',
        'user: heartbeat checks remained healthy.'
      ].join('\n'),
      expected: [
        '## 2026-03-10',
        '',
        '- [fact|c=0.72|i=0.30] 09:00 Routine maintenance completed without incidents'
      ].join('\n'),
      config: {
        now: '2026-03-10T09:00:00.000Z',
        scoring: {
          minimumImportanceForRecall: 0.25,
          matchThreshold: 0.2,
          keywordHints: ['heartbeat checks']
        }
      }
    });

    const report = await runObserverBenchmark({
      fixturesDir: fixturesRoot,
      provider: 'mock'
    });

    expect(report.fixtures[0]?.scoring.keywordSet).toContain('heartbeat checks');
    expect(report.fixtures[0]?.scoring.metrics.recall).toBeGreaterThanOrEqual(0);
  });
});
