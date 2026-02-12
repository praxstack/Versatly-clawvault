import { afterEach, describe, expect, it, vi } from 'vitest';
import { estimateTokens } from '../lib/token-counter.js';

const {
  loadMock,
  listMock,
  vsearchMock,
  readObservationsMock,
  parseObservationLinesMock
} = vi.hoisted(() => ({
  loadMock: vi.fn(),
  listMock: vi.fn(),
  vsearchMock: vi.fn(),
  readObservationsMock: vi.fn(),
  parseObservationLinesMock: vi.fn()
}));

vi.mock('../lib/vault.js', () => ({
  ClawVault: class {
    private readonly vaultPath: string;

    constructor(vaultPath: string) {
      this.vaultPath = vaultPath;
    }

    async load(): Promise<void> {
      await loadMock();
    }

    async list(): Promise<unknown[]> {
      return listMock();
    }

    async vsearch(task: string, options: unknown): Promise<unknown[]> {
      return vsearchMock(task, options);
    }

    getPath(): string {
      return this.vaultPath;
    }
  }
}));

vi.mock('../lib/observation-reader.js', () => ({
  readObservations: (vaultPath: string, days: number) => readObservationsMock(vaultPath, days),
  parseObservationLines: (markdown: string) => parseObservationLinesMock(markdown)
}));

import { buildContext } from './context.js';

afterEach(() => {
  vi.clearAllMocks();
});

describe('buildContext budget handling', () => {
  it('greedily keeps high-priority entries and enforces markdown budget', async () => {
    loadMock.mockResolvedValue(undefined);
    listMock.mockResolvedValue([
      {
        path: '/vault/daily/2026-02-11.md',
        title: '2026-02-11',
        category: 'daily',
        content: 'Daily summary '.repeat(40),
        modified: new Date('2026-02-11T08:00:00.000Z'),
        frontmatter: { date: '2026-02-11', type: 'daily' }
      }
    ]);
    vsearchMock.mockResolvedValue([
      {
        score: 0.9,
        snippet: 'Search context '.repeat(60),
        document: {
          path: '/vault/notes/architecture.md',
          title: 'Architecture',
          category: 'notes',
          content: '',
          modified: new Date('2026-02-10T10:00:00.000Z'),
          frontmatter: {}
        }
      }
    ]);
    readObservationsMock.mockReturnValue('## 2026-02-11');
    parseObservationLinesMock.mockReturnValue([
      { priority: '🔴', content: 'Critical deployment gate remains open', date: '2026-02-11' },
      { priority: '🟢', content: 'Low priority chatter '.repeat(50), date: '2026-02-11' }
    ]);

    const budget = 40;
    const result = await buildContext('ship release', {
      vaultPath: '/vault',
      budget
    });

    expect(estimateTokens(result.markdown)).toBeLessThanOrEqual(budget);
    expect(result.markdown).toContain('🔴 observation (2026-02-11)');
    expect(result.markdown).not.toContain('Low priority chatter');
  });
});
