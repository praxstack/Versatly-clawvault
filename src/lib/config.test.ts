import { afterEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { findNearestVaultPath, resolveVaultPath, resolveAgentVaultPath } from './config.js';

const originalVaultEnv = process.env.CLAWVAULT_PATH;

function makeTempDir(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function makeVaultFixture(prefix: string): string {
  const dir = makeTempDir(prefix);
  fs.writeFileSync(path.join(dir, '.clawvault.json'), '{}', 'utf-8');
  return dir;
}

afterEach(() => {
  if (originalVaultEnv === undefined) {
    delete process.env.CLAWVAULT_PATH;
  } else {
    process.env.CLAWVAULT_PATH = originalVaultEnv;
  }
});

describe('config path resolution', () => {
  it('finds nearest vault from cwd hierarchy', () => {
    const root = makeTempDir('clawvault-config-');
    const nested = path.join(root, 'a', 'b');
    fs.mkdirSync(nested, { recursive: true });
    fs.writeFileSync(path.join(root, '.clawvault.json'), '{}', 'utf-8');

    try {
      const found = findNearestVaultPath(nested);
      expect(found).toBe(root);
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it('resolves explicit path before env or discovery', () => {
    const explicit = makeTempDir('clawvault-explicit-');
    const env = makeTempDir('clawvault-env-');
    process.env.CLAWVAULT_PATH = env;
    try {
      const resolved = resolveVaultPath({ explicitPath: explicit });
      expect(resolved).toBe(path.resolve(explicit));
    } finally {
      fs.rmSync(explicit, { recursive: true, force: true });
      fs.rmSync(env, { recursive: true, force: true });
    }
  });

  it('accepts object-shaped explicitPath values', () => {
    const explicit = makeTempDir('clawvault-explicit-object-');
    const env = makeTempDir('clawvault-env-object-');
    process.env.CLAWVAULT_PATH = env;
    try {
      const resolved = resolveVaultPath({
        explicitPath: { path: explicit } as unknown as string
      });
      expect(resolved).toBe(path.resolve(explicit));
    } finally {
      fs.rmSync(explicit, { recursive: true, force: true });
      fs.rmSync(env, { recursive: true, force: true });
    }
  });

  it('resolves env path before cwd discovery', () => {
    const env = makeTempDir('clawvault-env-');
    const cwdRoot = makeTempDir('clawvault-cwd-');
    const nested = path.join(cwdRoot, 'nested');
    fs.mkdirSync(nested, { recursive: true });
    fs.writeFileSync(path.join(cwdRoot, '.clawvault.json'), '{}', 'utf-8');
    process.env.CLAWVAULT_PATH = env;

    try {
      const resolved = resolveVaultPath({ cwd: nested });
      expect(resolved).toBe(path.resolve(env));
    } finally {
      fs.rmSync(env, { recursive: true, force: true });
      fs.rmSync(cwdRoot, { recursive: true, force: true });
    }
  });

  it('throws when no explicit path, env, or discovered vault exists', () => {
    const cwd = makeTempDir('clawvault-missing-');
    delete process.env.CLAWVAULT_PATH;
    try {
      expect(() => resolveVaultPath({ cwd })).toThrow(
        'No vault path found. Set CLAWVAULT_PATH, use --vault, or run inside a vault.'
      );
    } finally {
      fs.rmSync(cwd, { recursive: true, force: true });
    }
  });
});

describe('per-agent vault resolution', () => {
  it('resolves agent-specific vault path from agentVaults config', () => {
    const agent1Vault = makeVaultFixture('clawvault-agent1-');
    const agent2Vault = makeVaultFixture('clawvault-agent2-');
    const fallbackVault = makeVaultFixture('clawvault-fallback-');
    delete process.env.CLAWVAULT_PATH;

    try {
      const resolved = resolveVaultPath({
        agentId: 'agent1',
        pluginConfig: {
          vaultPath: fallbackVault,
          agentVaults: {
            agent1: agent1Vault,
            agent2: agent2Vault
          }
        }
      });
      expect(resolved).toBe(agent1Vault);
    } finally {
      fs.rmSync(agent1Vault, { recursive: true, force: true });
      fs.rmSync(agent2Vault, { recursive: true, force: true });
      fs.rmSync(fallbackVault, { recursive: true, force: true });
    }
  });

  it('falls back to vaultPath when agent not in agentVaults', () => {
    const agent1Vault = makeVaultFixture('clawvault-agent1-');
    const fallbackVault = makeVaultFixture('clawvault-fallback-');
    delete process.env.CLAWVAULT_PATH;

    try {
      const resolved = resolveVaultPath({
        agentId: 'unknown-agent',
        pluginConfig: {
          vaultPath: fallbackVault,
          agentVaults: {
            agent1: agent1Vault
          }
        }
      });
      expect(resolved).toBe(fallbackVault);
    } finally {
      fs.rmSync(agent1Vault, { recursive: true, force: true });
      fs.rmSync(fallbackVault, { recursive: true, force: true });
    }
  });

  it('falls back to vaultPath when agentVaults is not set', () => {
    const fallbackVault = makeVaultFixture('clawvault-fallback-');
    delete process.env.CLAWVAULT_PATH;

    try {
      const resolved = resolveVaultPath({
        agentId: 'agent1',
        pluginConfig: {
          vaultPath: fallbackVault
        }
      });
      expect(resolved).toBe(fallbackVault);
    } finally {
      fs.rmSync(fallbackVault, { recursive: true, force: true });
    }
  });

  it('explicit path takes precedence over agentVaults', () => {
    const explicitVault = makeVaultFixture('clawvault-explicit-');
    const agent1Vault = makeVaultFixture('clawvault-agent1-');
    delete process.env.CLAWVAULT_PATH;

    try {
      const resolved = resolveVaultPath({
        explicitPath: explicitVault,
        agentId: 'agent1',
        pluginConfig: {
          agentVaults: {
            agent1: agent1Vault
          }
        }
      });
      expect(resolved).toBe(explicitVault);
    } finally {
      fs.rmSync(explicitVault, { recursive: true, force: true });
      fs.rmSync(agent1Vault, { recursive: true, force: true });
    }
  });

  it('resolveAgentVaultPath returns null for invalid inputs', () => {
    expect(resolveAgentVaultPath(undefined, 'agent1')).toBeNull();
    expect(resolveAgentVaultPath({}, undefined)).toBeNull();
    expect(resolveAgentVaultPath({}, '')).toBeNull();
    expect(resolveAgentVaultPath({ agent1: '/nonexistent/path' }, 'agent1')).toBeNull();
  });

  it('resolveAgentVaultPath returns null when agent not found', () => {
    const vault = makeVaultFixture('clawvault-agent-');
    try {
      const result = resolveAgentVaultPath({ agent1: vault }, 'agent2');
      expect(result).toBeNull();
    } finally {
      fs.rmSync(vault, { recursive: true, force: true });
    }
  });

  it('resolveAgentVaultPath returns validated path when agent found', () => {
    const vault = makeVaultFixture('clawvault-agent-');
    try {
      const result = resolveAgentVaultPath({ agent1: vault }, 'agent1');
      expect(result).toBe(vault);
    } finally {
      fs.rmSync(vault, { recursive: true, force: true });
    }
  });
});
