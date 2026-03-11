import { afterEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import type { Document } from '../types.js';

const { embedTextMock, crossEncoderRerankMock } = vi.hoisted(() => ({
  embedTextMock: vi.fn(),
  crossEncoderRerankMock: vi.fn()
}));

vi.mock('./hosted-embeddings.js', async () => {
  const actual = await vi.importActual<typeof import('./hosted-embeddings.js')>('./hosted-embeddings.js');
  return {
    ...actual,
    embedText: embedTextMock
  };
});

vi.mock('./hosted-rerank.js', async () => {
  const actual = await vi.importActual<typeof import('./hosted-rerank.js')>('./hosted-rerank.js');
  return {
    ...actual,
    crossEncoderRerank: crossEncoderRerankMock
  };
});

import { InProcessSearchEngine } from './in-process-search.js';

const createdTempDirs: string[] = [];

function makeTempVault(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-in-process-search-'));
  createdTempDirs.push(dir);
  return dir;
}

function doc(id: string, content: string, modified?: Date): Document {
  return {
    id,
    path: `/vault/${id}.md`,
    category: id.split('/')[0] || 'notes',
    title: id.split('/').pop() || id,
    content,
    frontmatter: {},
    links: [],
    tags: [],
    modified: modified ?? new Date()
  };
}

afterEach(() => {
  vi.clearAllMocks();
  while (createdTempDirs.length > 0) {
    const dir = createdTempDirs.pop();
    if (dir) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe('InProcessSearchEngine', () => {
  it('returns multiple hits per document from chunk-level BM25', async () => {
    const engine = new InProcessSearchEngine();
    engine.setConfig({ chunkSize: 30, chunkOverlap: 0 });
    engine.addDocument(doc(
      'notes/multi-hit',
      'alpha one two three four five six seven eight nine ten eleven twelve '
      + 'beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi '
      + 'alpha red blue green yellow orange purple black white gray cyan magenta '
      + 'tau upsilon phi chi psi omega alpha terminal'
    ));

    const results = await engine.search('alpha', { limit: 5 });
    expect(results.length).toBeGreaterThan(1);
    expect(results.every((entry) => entry.document.id === 'notes/multi-hit')).toBe(true);
  });

  it('runs semantic vsearch using hosted embeddings cache', async () => {
    const vaultPath = makeTempVault();
    const cacheDir = path.join(vaultPath, '.clawvault');
    fs.mkdirSync(cacheDir, { recursive: true });
    fs.writeFileSync(path.join(cacheDir, 'embeddings.bin.json'), JSON.stringify({
      version: 2,
      provider: 'openai',
      model: 'text-embedding-3-small',
      vectors: {
        'notes/a': { hash: 'a', embedding: [1, 0] },
        'notes/b': { hash: 'b', embedding: [0, 1] }
      }
    }));

    embedTextMock.mockResolvedValue(new Float32Array([1, 0]));

    const engine = new InProcessSearchEngine();
    engine.setVaultPath(vaultPath);
    engine.setConfig({
      embeddings: {
        provider: 'openai',
        model: 'text-embedding-3-small',
        apiKey: 'test'
      }
    });
    engine.addDocument(doc('notes/a', 'first semantic note'));
    engine.addDocument(doc('notes/b', 'second semantic note'));

    const results = await engine.vsearch('semantic query', { limit: 2 });
    expect(results).toHaveLength(2);
    expect(results[0].document.id).toBe('notes/a');
  });

  it('applies cross-encoder reranking when configured', async () => {
    crossEncoderRerankMock.mockResolvedValue([0.1, 0.9]);

    const engine = new InProcessSearchEngine();
    engine.setConfig({
      rerank: {
        provider: 'jina',
        apiKey: 'test',
        weight: 1
      },
      chunkSize: 120
    });

    engine.addDocument(doc('notes/one', 'query term query term query term'));
    engine.addDocument(doc('notes/two', 'query term'));

    const results = await engine.search('query', { limit: 2 });
    expect(results).toHaveLength(2);
    expect(results[0].document.id).toBe('notes/two');
  });
});
