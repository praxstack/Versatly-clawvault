import { afterEach, describe, expect, it, vi } from 'vitest';

const { execFileSyncMock } = vi.hoisted(() => ({
  execFileSyncMock: vi.fn()
}));

vi.mock('child_process', () => ({
  execFileSync: execFileSyncMock
}));

async function loadModule() {
  vi.resetModules();
  return await import('./qmd-collections.js');
}

afterEach(() => {
  vi.clearAllMocks();
});

describe('parseQmdCollectionList', () => {
  it('parses collection list output', async () => {
    const { parseQmdCollectionList } = await loadModule();
    const output = `my-vault (qmd://my-vault)
  Root: /home/user/my-vault
  Files: 100
  Vectors: 95

workspace-main (qmd://workspace-main)
  Root: /home/user/workspace
  Files: 50
  Vectors: 50
`;
    const collections = parseQmdCollectionList(output);
    expect(collections).toHaveLength(2);
    expect(collections[0].name).toBe('my-vault');
    expect(collections[0].root).toBe('/home/user/my-vault');
    expect(collections[0].files).toBe(100);
    expect(collections[0].vectors).toBe(95);
    expect(collections[0].pendingEmbeddings).toBe(5);
    expect(collections[1].name).toBe('workspace-main');
    expect(collections[1].root).toBe('/home/user/workspace');
  });

  it('handles empty output', async () => {
    const { parseQmdCollectionList } = await loadModule();
    expect(parseQmdCollectionList('')).toEqual([]);
    expect(parseQmdCollectionList('\n\n')).toEqual([]);
  });
});

describe('collectionExists', () => {
  it('returns true when collection exists', async () => {
    execFileSyncMock.mockReturnValue(`my-vault (qmd://my-vault)
  Root: /home/user/my-vault
`);
    const { collectionExists } = await loadModule();
    expect(collectionExists('my-vault')).toBe(true);
  });

  it('returns false when collection does not exist', async () => {
    execFileSyncMock.mockReturnValue(`other-vault (qmd://other-vault)
  Root: /home/user/other
`);
    const { collectionExists } = await loadModule();
    expect(collectionExists('my-vault')).toBe(false);
  });

  it('returns false when qmd fails', async () => {
    execFileSyncMock.mockImplementation(() => {
      throw new Error('qmd not found');
    });
    const { collectionExists } = await loadModule();
    expect(collectionExists('my-vault')).toBe(false);
  });
});

describe('findCollectionByRoot', () => {
  it('finds collection by root path', async () => {
    execFileSyncMock.mockReturnValue(`my-vault (qmd://my-vault)
  Root: /home/user/my-vault

workspace (qmd://workspace)
  Root: /home/user/workspace
`);
    const { findCollectionByRoot } = await loadModule();
    const result = findCollectionByRoot('/home/user/workspace');
    expect(result).toBeDefined();
    expect(result?.name).toBe('workspace');
  });

  it('handles trailing slashes', async () => {
    execFileSyncMock.mockReturnValue(`my-vault (qmd://my-vault)
  Root: /home/user/my-vault/
`);
    const { findCollectionByRoot } = await loadModule();
    const result = findCollectionByRoot('/home/user/my-vault');
    expect(result).toBeDefined();
    expect(result?.name).toBe('my-vault');
  });

  it('returns undefined when no match', async () => {
    execFileSyncMock.mockReturnValue(`my-vault (qmd://my-vault)
  Root: /home/user/my-vault
`);
    const { findCollectionByRoot } = await loadModule();
    expect(findCollectionByRoot('/other/path')).toBeUndefined();
  });

  it('returns undefined when qmd fails', async () => {
    execFileSyncMock.mockImplementation(() => {
      throw new Error('qmd not found');
    });
    const { findCollectionByRoot } = await loadModule();
    expect(findCollectionByRoot('/any/path')).toBeUndefined();
  });
});

describe('getFirstCollection', () => {
  it('returns first collection', async () => {
    execFileSyncMock.mockReturnValue(`first-vault (qmd://first-vault)
  Root: /home/user/first

second-vault (qmd://second-vault)
  Root: /home/user/second
`);
    const { getFirstCollection } = await loadModule();
    const result = getFirstCollection();
    expect(result).toBeDefined();
    expect(result?.name).toBe('first-vault');
  });

  it('returns undefined when no collections', async () => {
    execFileSyncMock.mockReturnValue('');
    const { getFirstCollection } = await loadModule();
    expect(getFirstCollection()).toBeUndefined();
  });

  it('returns undefined when qmd fails', async () => {
    execFileSyncMock.mockImplementation(() => {
      throw new Error('qmd not found');
    });
    const { getFirstCollection } = await loadModule();
    expect(getFirstCollection()).toBeUndefined();
  });
});
