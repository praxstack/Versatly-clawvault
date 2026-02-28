import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  ensureLedgerStructure,
  ensureParentDir,
  formatIsoWeekKey,
  getArchiveObservationPath,
  getIsoWeek,
  getIsoWeekRange,
  getLegacyObservationPath,
  getObservationPath,
  getRawTranscriptPath,
  listArchiveObservationFiles,
  listLedgerObservationFiles,
  listLegacyObservationFiles,
  listObservationFiles,
  listRawTranscriptFiles,
  parseDateKey,
  toDateKey,
} from './ledger.js';

function makeTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-ledger-'));
}

function writeFile(filePath: string, content: string): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content, 'utf-8');
}

describe('ledger', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = makeTempDir();
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  it('formats/parses date keys and builds date-based paths', () => {
    const date = new Date('2026-02-14T12:34:56.000Z');

    expect(toDateKey(date)).toBe('2026-02-14');
    expect(parseDateKey('2026-02-14')?.toISOString()).toBe('2026-02-14T00:00:00.000Z');
    expect(parseDateKey('2026/02/14')).toBeNull();

    expect(getObservationPath(tempDir, '2026-02-14')).toBe(
      path.join(tempDir, 'ledger', 'observations', '2026', '02', '14.md')
    );
    expect(getArchiveObservationPath(tempDir, date)).toBe(
      path.join(tempDir, 'ledger', 'archive', 'observations', '2026', '02', '14.md')
    );
    expect(getLegacyObservationPath(tempDir, date)).toBe(
      path.join(tempDir, 'observations', '2026-02-14.md')
    );
    expect(getRawTranscriptPath(tempDir, 'openclaw', date)).toBe(
      path.join(tempDir, 'ledger', 'raw', 'openclaw', '2026', '02', '14.jsonl')
    );

    expect(() => getObservationPath(tempDir, '2026/02/14')).toThrow('Invalid date key');
  });

  it('creates ledger structure and ensures parent directories', () => {
    ensureLedgerStructure(tempDir);

    const expectedDirs = [
      path.join(tempDir, 'ledger', 'raw', 'openclaw'),
      path.join(tempDir, 'ledger', 'raw', 'chatgpt'),
      path.join(tempDir, 'ledger', 'raw', 'claude'),
      path.join(tempDir, 'ledger', 'raw', 'opencode'),
      path.join(tempDir, 'ledger', 'observations'),
      path.join(tempDir, 'ledger', 'reflections'),
      path.join(tempDir, 'ledger', 'archive', 'observations'),
    ];

    for (const dir of expectedDirs) {
      expect(fs.existsSync(dir)).toBe(true);
      expect(fs.statSync(dir).isDirectory()).toBe(true);
    }

    const nestedFile = path.join(tempDir, 'ledger', 'raw', 'custom', '2026', '02', '14.jsonl');
    ensureParentDir(nestedFile);
    expect(fs.existsSync(path.dirname(nestedFile))).toBe(true);
  });

  it('lists observation files across ledger, legacy, and archive with filtering and dedupe', () => {
    ensureLedgerStructure(tempDir);

    writeFile(getObservationPath(tempDir, '2026-02-10'), '# Ledger observation');
    writeFile(getObservationPath(tempDir, '2026-02-11'), '# Ledger observation');
    writeFile(getArchiveObservationPath(tempDir, '2026-02-10'), '# Archive observation');
    writeFile(getArchiveObservationPath(tempDir, '2026-02-09'), '# Archive observation');
    writeFile(getLegacyObservationPath(tempDir, '2026-02-10'), '# Legacy observation');
    writeFile(path.join(tempDir, 'observations', 'not-a-date.md'), '# ignored');

    const ledgerOnly = listLedgerObservationFiles(tempDir);
    expect(ledgerOnly.map((entry) => entry.date)).toEqual(['2026-02-10', '2026-02-11']);
    expect(ledgerOnly.every((entry) => entry.location === 'ledger')).toBe(true);

    const archiveOnly = listArchiveObservationFiles(tempDir, { fromDate: '2026-02-10' });
    expect(archiveOnly.map((entry) => entry.date)).toEqual(['2026-02-10']);
    expect(archiveOnly[0].location).toBe('archive');

    const legacyOnly = listLegacyObservationFiles(tempDir);
    expect(legacyOnly.map((entry) => entry.date)).toEqual(['2026-02-10']);
    expect(legacyOnly[0].location).toBe('legacy');

    const mergedDedupe = listObservationFiles(tempDir, { includeArchive: true });
    expect(mergedDedupe.map((entry) => `${entry.date}:${entry.location}`)).toEqual([
      '2026-02-09:archive',
      '2026-02-10:ledger',
      '2026-02-11:ledger',
    ]);

    const mergedNoDedupe = listObservationFiles(tempDir, {
      includeArchive: true,
      dedupeByDate: false,
    });
    expect(mergedNoDedupe.filter((entry) => entry.date === '2026-02-10')).toHaveLength(3);
  });

  it('lists raw transcripts with source/date filtering', () => {
    ensureLedgerStructure(tempDir);

    writeFile(
      path.join(tempDir, 'ledger', 'raw', 'openclaw', '2026', '02', '14.jsonl'),
      '{"event":"openclaw"}\n'
    );
    writeFile(
      path.join(tempDir, 'ledger', 'raw', 'claude', '2026', '02', '13.jsonl'),
      '{"event":"claude"}\n'
    );
    writeFile(
      path.join(tempDir, 'ledger', 'raw', 'openclaw', '2026', '02', '14.txt'),
      'ignore me'
    );

    const all = listRawTranscriptFiles(tempDir);
    expect(all.map((entry) => `${entry.date}:${entry.source}`)).toEqual([
      '2026-02-13:claude',
      '2026-02-14:openclaw',
    ]);

    const filtered = listRawTranscriptFiles(tempDir, { source: 'openclaw', fromDate: '2026-02-14' });
    expect(filtered).toHaveLength(1);
    expect(filtered[0].source).toBe('openclaw');
  });

  it('computes ISO week helpers consistently', () => {
    expect(getIsoWeek(new Date('2021-01-01T00:00:00.000Z'))).toEqual({ year: 2020, week: 53 });
    expect(formatIsoWeekKey({ year: 2026, week: 7 })).toBe('2026-W07');
    expect(formatIsoWeekKey(new Date('2026-02-14T00:00:00.000Z'))).toBe('2026-W07');

    const range = getIsoWeekRange(2026, 7);
    expect(toDateKey(range.start)).toBe('2026-02-09');
    expect(toDateKey(range.end)).toBe('2026-02-15');
  });
});
