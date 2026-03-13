import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

const {
  hasQmdMock,
  listQmdCollectionsMock,
  spawnSyncMock
} = vi.hoisted(() => ({
  hasQmdMock: vi.fn(),
  listQmdCollectionsMock: vi.fn(),
  spawnSyncMock: vi.fn()
}));

let mockQmdCollection = 'vault';
let mockQmdRoot = '/tmp/vault';

vi.mock('../lib/search.js', async () => {
  const actual = await vi.importActual<typeof import('../lib/search.js')>('../lib/search.js');
  return {
    ...actual,
    hasQmd: hasQmdMock
  };
});

vi.mock('../lib/qmd-collections.js', () => ({
  listQmdCollections: listQmdCollectionsMock,
  removeQmdCollection: vi.fn(),
  findCollectionByRoot: vi.fn().mockReturnValue(undefined),
  collectionExists: vi.fn().mockReturnValue(true),
  getFirstCollection: vi.fn().mockReturnValue(undefined)
}));

vi.mock('child_process', async (importOriginal) => {
  const actual = await importOriginal<typeof import('child_process')>();
  return {
    ...actual,
    spawnSync: spawnSyncMock
  };
});

vi.mock('../lib/vault.js', () => ({
  ClawVault: class {
    private vaultPath: string;

    constructor(vaultPath: string) {
      this.vaultPath = vaultPath;
    }

    async load(): Promise<void> {
      return;
    }

    getPath(): string {
      return this.vaultPath;
    }

    getQmdCollection(): string {
      return mockQmdCollection;
    }

    getQmdRoot(): string {
      return mockQmdRoot;
    }
  },
  findVault: async () => null
}));

import { doctor } from './doctor.js';

function makeTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function writeConfig(vaultPath: string, payload: unknown): void {
  fs.writeFileSync(path.join(vaultPath, '.clawvault.json'), JSON.stringify(payload, null, 2));
}

function checkByLabel(report: Awaited<ReturnType<typeof doctor>>, label: string) {
  return report.checks.find((check) => check.label === label);
}

function installSpawnDefaults(prefixPath: string): void {
  spawnSyncMock.mockImplementation((command: string, args: string[]) => {
    if (command === 'npm' && args[0] === '--version') {
      return { status: 0, stdout: '10.9.4\n', stderr: '' };
    }
    if (command === 'npm' && args[0] === 'config' && args[1] === 'get' && args[2] === 'prefix') {
      return { status: 0, stdout: `${prefixPath}\n`, stderr: '' };
    }
    if (command === 'openclaw') {
      return { status: 0, stdout: 'clawvault enabled\n', stderr: '' };
    }
    if (command === 'git') {
      return { status: 0, stdout: 'git version 2.44.0\n', stderr: '' };
    }
    return { status: 0, stdout: '', stderr: '' };
  });
}

beforeEach(() => {
  hasQmdMock.mockReturnValue(false);
  listQmdCollectionsMock.mockReturnValue([]);
  mockQmdCollection = 'vault';
  mockQmdRoot = '/tmp/vault';
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('doctor', () => {
  it('treats missing qmd as warning and keeps in-process BM25 available', async () => {
    const vaultPath = makeTempDir('clawvault-doctor-qmd-');
    writeConfig(vaultPath, { name: 'test-vault' });
    installSpawnDefaults(vaultPath);

    try {
      const report = await doctor({ vaultPath });
      expect(checkByLabel(report, 'qmd availability')?.status).toBe('warn');
      expect(checkByLabel(report, 'in-process BM25 engine')?.status).toBe('ok');
      expect(checkByLabel(report, 'vault config file')?.status).toBe('ok');
    } finally {
      fs.rmSync(vaultPath, { recursive: true, force: true });
    }
  });

  it('reports config parse errors and downstream search warnings', async () => {
    const vaultPath = makeTempDir('clawvault-doctor-config-');
    fs.writeFileSync(path.join(vaultPath, '.clawvault.json'), '{"name": ');
    installSpawnDefaults(vaultPath);

    try {
      const report = await doctor({ vaultPath });
      expect(checkByLabel(report, 'vault config file')?.status).toBe('error');
      expect(checkByLabel(report, 'semantic embeddings')?.status).toBe('warn');
      expect(report.errors).toBeGreaterThan(0);
    } finally {
      fs.rmSync(vaultPath, { recursive: true, force: true });
    }
  });

  it('warns when npm global prefix is not writable', async () => {
    const vaultPath = makeTempDir('clawvault-doctor-prefix-');
    writeConfig(vaultPath, { name: 'test-vault' });
    installSpawnDefaults('/root/locked-prefix');

    try {
      const report = await doctor({ vaultPath });
      const npmCheck = checkByLabel(report, 'npm global install location');
      expect(npmCheck?.status).toBe('warn');
      expect(npmCheck?.hint).toContain('npm config set prefix ~/.npm-global');
    } finally {
      fs.rmSync(vaultPath, { recursive: true, force: true });
    }
  });

  it('warns when OpenClaw does not list clawvault plugin', async () => {
    const vaultPath = makeTempDir('clawvault-doctor-openclaw-');
    writeConfig(vaultPath, { name: 'test-vault' });
    installSpawnDefaults(vaultPath);
    spawnSyncMock.mockImplementation((command: string, args: string[]) => {
      if (command === 'openclaw') {
        return { status: 0, stdout: 'other-plugin enabled\n', stderr: '' };
      }
      if (command === 'npm' && args[0] === '--version') {
        return { status: 0, stdout: '10.9.4\n', stderr: '' };
      }
      if (command === 'npm' && args[0] === 'config' && args[1] === 'get' && args[2] === 'prefix') {
        return { status: 0, stdout: `${vaultPath}\n`, stderr: '' };
      }
      if (command === 'git') {
        return { status: 0, stdout: 'git version 2.44.0\n', stderr: '' };
      }
      return { status: 0, stdout: '', stderr: '' };
    });

    try {
      const report = await doctor({ vaultPath });
      expect(checkByLabel(report, 'OpenClaw plugin registration')?.status).toBe('warn');
    } finally {
      fs.rmSync(vaultPath, { recursive: true, force: true });
    }
  });

  it('handles object-shaped vault path options without crashing', async () => {
    const vaultPath = makeTempDir('clawvault-doctor-object-path-');
    writeConfig(vaultPath, { name: 'test-vault' });
    installSpawnDefaults(vaultPath);

    try {
      const report = await doctor({ vaultPath: { path: vaultPath } as unknown as string });
      expect(report.vaultPath).toBe(path.resolve(vaultPath));
      expect(checkByLabel(report, 'vault directory')?.status).toBe('ok');
      expect(checkByLabel(report, 'vault config file')?.status).toBe('ok');
    } finally {
      fs.rmSync(vaultPath, { recursive: true, force: true });
    }
  });

  it('retains migration diagnostics for migrate command compatibility', async () => {
    const vaultPath = makeTempDir('clawvault-doctor-migration-');
    writeConfig(vaultPath, { name: 'my-vault', qmdCollection: 'my-vault', qmdRoot: vaultPath });
    installSpawnDefaults(vaultPath);

    hasQmdMock.mockReturnValue(true);
    listQmdCollectionsMock.mockReturnValue([]);
    mockQmdCollection = 'my-vault';
    mockQmdRoot = vaultPath;

    try {
      const report = await doctor({ vaultPath });
      const missingCollection = report.migrationIssues.find((issue) => issue.type === 'missing_qmd_collection');
      expect(missingCollection).toBeDefined();
      expect(checkByLabel(report, 'migration: missing qmd collection')?.status).toBe('warn');
    } finally {
      fs.rmSync(vaultPath, { recursive: true, force: true });
    }
  });
});
