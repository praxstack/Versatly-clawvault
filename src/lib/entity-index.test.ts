import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { buildEntityIndex, getSortedAliases } from './entity-index.js';

function makeTempDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-entity-index-'));
}

function writeMarkdown(vaultPath: string, relativePath: string, content: string): void {
  const absolutePath = path.join(vaultPath, relativePath);
  fs.mkdirSync(path.dirname(absolutePath), { recursive: true });
  fs.writeFileSync(absolutePath, content, 'utf-8');
}

describe('entity-index', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = makeTempDir();
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  it('builds aliases from filename, title, and frontmatter aliases', () => {
    writeMarkdown(
      tempDir,
      'people/pedro.md',
      `---
title: Pedro Duarte
aliases:
  - Pduarte
  - The Claw
---
# Pedro
`
    );

    const index = buildEntityIndex(tempDir);

    expect(index.entries.get('pedro')).toBe('people/pedro');
    expect(index.entries.get('pedro duarte')).toBe('people/pedro');
    expect(index.entries.get('pduarte')).toBe('people/pedro');
    expect(index.entries.get('the claw')).toBe('people/pedro');

    const entry = index.byPath.get('people/pedro');
    expect(entry?.aliases).toEqual(['pedro', 'Pedro Duarte', 'Pduarte', 'The Claw']);
  });

  it('keeps the first mapping when duplicate aliases collide', () => {
    writeMarkdown(
      tempDir,
      'people/ops.md',
      `---
aliases:
  - shared
---
People note
`
    );
    writeMarkdown(
      tempDir,
      'projects/platform.md',
      `---
aliases:
  - shared
---
Project note
`
    );

    const index = buildEntityIndex(tempDir);
    expect(index.entries.get('shared')).toBe('people/ops');
  });

  it('ignores non-entity folders and non-markdown files', () => {
    writeMarkdown(tempDir, 'misc/ignored.md', '# Not indexed');
    fs.mkdirSync(path.join(tempDir, 'projects'), { recursive: true });
    fs.writeFileSync(path.join(tempDir, 'projects', 'readme.txt'), 'ignore me', 'utf-8');
    writeMarkdown(tempDir, 'projects/core-api.md', '# Core API');

    const index = buildEntityIndex(tempDir);

    expect(index.byPath.has('misc/ignored')).toBe(false);
    expect(index.entries.has('readme')).toBe(false);
    expect(index.byPath.has('projects/core-api')).toBe(true);
  });

  it('does not duplicate title alias when title matches filename ignoring case', () => {
    writeMarkdown(
      tempDir,
      'decisions/review.md',
      `---
title: Review
---
Decision
`
    );

    const index = buildEntityIndex(tempDir);
    expect(index.byPath.get('decisions/review')?.aliases).toEqual(['review']);
  });

  it('sorts aliases by descending length', () => {
    writeMarkdown(
      tempDir,
      'people/justin.md',
      `---
title: Justin Dukes
aliases:
  - JD
---
Person
`
    );

    const sorted = getSortedAliases(buildEntityIndex(tempDir));
    const sortedAliases = sorted.map((entry) => entry.alias);

    expect(sortedAliases[0]).toBe('justin dukes');
    expect(sortedAliases.at(-1)).toBe('jd');
  });
});
