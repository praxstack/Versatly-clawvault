import { describe, expect, it } from 'vitest';
import { autoLink, dryRunLink, findUnlinkedMentions } from './auto-linker.js';
import type { EntityIndex } from './entity-index.js';

function createIndex(entries: Array<[string, string]>): EntityIndex {
  return {
    entries: new Map(entries),
    byPath: new Map(),
  };
}

describe('auto-linker', () => {
  it('links only first unprotected occurrence per entity', () => {
    const index = createIndex([
      ['justin', 'people/justin'],
    ]);

    const input = `---
owner: Justin
---

\`\`\`
Justin
\`\`\`

Already linked [[people/justin]].
URL https://example.com/Justin should stay plain.

Real Justin mention.
Another Justin mention.
`;

    const output = autoLink(input, index);

    expect(output).toContain('Real [[people/justin]] mention.');
    expect(output).toContain('Another Justin mention.');
    expect(output).toContain('owner: Justin');
    expect(output).toContain('https://example.com/Justin');
  });

  it('renders alias links and escapes regex metacharacters', () => {
    const index = createIndex([
      ['node.js', 'projects/nodejs'],
      ['justin', 'people/justin'],
    ]);

    const output = autoLink('Node.js integrates with Justin.', index);

    expect(output).toContain('[[projects/nodejs|Node.js]]');
    expect(output).toContain('[[people/justin]]');
  });

  it('prefers longer aliases before shorter overlaps', () => {
    const index = createIndex([
      ['core api', 'projects/core-api'],
      ['api', 'projects/api'],
    ]);

    const output = autoLink('Core API is replacing the API endpoint.', index);

    expect(output).toContain('[[projects/core-api|Core API]]');
    expect(output).toContain('the [[projects/api]] endpoint.');
  });

  it('reports dry-run links with line numbers and protected-range filtering', () => {
    const index = createIndex([
      ['core api', 'projects/core-api'],
      ['justin', 'people/justin'],
    ]);

    const content = [
      'Title',
      'Mention Justin here.',
      '`Core API` should not be linked here.',
      'Core API should be linked here.',
    ].join('\n');

    const dryRun = dryRunLink(content, index);
    const unlinked = findUnlinkedMentions(content, index);

    expect(dryRun).toEqual([
      { alias: 'Core API', path: 'projects/core-api', line: 4 },
      { alias: 'Justin', path: 'people/justin', line: 2 },
    ]);
    expect(unlinked).toEqual(dryRun);
  });
});
