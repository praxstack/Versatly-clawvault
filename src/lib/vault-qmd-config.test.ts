import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

const { execFileSyncMock } = vi.hoisted(() => ({
  execFileSyncMock: vi.fn()
}));

vi.mock('child_process', () => ({
  execFileSync: execFileSyncMock
}));

async function loadModule() {
  vi.resetModules();
  return await import('./vault-qmd-config.js');
}

function makeTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

afterEach(() => {
  vi.clearAllMocks();
});

describe('loadVaultQmdConfig', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = makeTempDir('clawvault-vault-qmd-config-');
  });

  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  it('reads qmdCollection from config file', async () => {
    fs.writeFileSync(
      path.join(tempDir, '.clawvault.json'),
      JSON.stringify({ name: 'my-vault', qmdCollection: 'custom-collection' })
    );
    execFileSyncMock.mockReturnValue(`custom-collection (qmd://custom-collection)
  Root: ${tempDir}
`);

    const { loadVaultQmdConfig } = await loadModule();
    const config = loadVaultQmdConfig(tempDir);
    expect(config.qmdCollection).toBe('custom-collection');
    expect(config.autoDetected).toBeUndefined();
  });

  it('falls back to vault name when qmdCollection not in config', async () => {
    fs.writeFileSync(
      path.join(tempDir, '.clawvault.json'),
      JSON.stringify({ name: 'my-vault' })
    );
    execFileSyncMock.mockReturnValue(`my-vault (qmd://my-vault)
  Root: ${tempDir}
`);

    const { loadVaultQmdConfig } = await loadModule();
    const config = loadVaultQmdConfig(tempDir);
    expect(config.qmdCollection).toBe('my-vault');
  });

  it('auto-detects collection by root when configured collection does not exist', async () => {
    fs.writeFileSync(
      path.join(tempDir, '.clawvault.json'),
      JSON.stringify({ name: 'old-name', qmdCollection: 'old-collection' })
    );
    execFileSyncMock.mockReturnValue(`new-collection (qmd://new-collection)
  Root: ${tempDir}
`);

    const { loadVaultQmdConfig } = await loadModule();
    const config = loadVaultQmdConfig(tempDir);
    expect(config.qmdCollection).toBe('new-collection');
    expect(config.autoDetected).toBe(true);
  });

  it('falls back to first collection when no root match', async () => {
    fs.writeFileSync(
      path.join(tempDir, '.clawvault.json'),
      JSON.stringify({ name: 'my-vault', qmdCollection: 'nonexistent' })
    );
    execFileSyncMock.mockReturnValue(`first-collection (qmd://first-collection)
  Root: /other/path

second-collection (qmd://second-collection)
  Root: /another/path
`);

    const { loadVaultQmdConfig } = await loadModule();
    const config = loadVaultQmdConfig(tempDir);
    expect(config.qmdCollection).toBe('first-collection');
    expect(config.autoDetected).toBe(true);
  });

  it('uses directory basename when no config file exists', async () => {
    execFileSyncMock.mockReturnValue(`${path.basename(tempDir)} (qmd://${path.basename(tempDir)})
  Root: ${tempDir}
`);

    const { loadVaultQmdConfig } = await loadModule();
    const config = loadVaultQmdConfig(tempDir);
    expect(config.qmdCollection).toBe(path.basename(tempDir));
  });

  it('handles qmd failure gracefully by using configured collection', async () => {
    fs.writeFileSync(
      path.join(tempDir, '.clawvault.json'),
      JSON.stringify({ name: 'my-vault', qmdCollection: 'my-collection' })
    );
    execFileSyncMock.mockImplementation(() => {
      throw new Error('qmd not found');
    });

    const { loadVaultQmdConfig } = await loadModule();
    const config = loadVaultQmdConfig(tempDir);
    expect(config.qmdCollection).toBe('my-collection');
  });

  it('resolves relative qmdRoot paths', async () => {
    fs.writeFileSync(
      path.join(tempDir, '.clawvault.json'),
      JSON.stringify({ name: 'my-vault', qmdRoot: './subdir' })
    );
    execFileSyncMock.mockReturnValue('');

    const { loadVaultQmdConfig } = await loadModule();
    const config = loadVaultQmdConfig(tempDir);
    expect(config.qmdRoot).toBe(path.join(tempDir, 'subdir'));
  });
});
