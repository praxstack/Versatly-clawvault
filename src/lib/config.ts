import * as fs from 'fs';
import * as path from 'path';

export interface AgentVaultsConfig {
  [agentName: string]: string;
}

export interface PluginConfig {
  vaultPath?: string;
  agentVaults?: AgentVaultsConfig;
}

/**
 * Get the vault path from CLAWVAULT_PATH env var or throw
 */
export function getVaultPath(): string {
  const vaultPath = process.env.CLAWVAULT_PATH;
  if (!vaultPath) {
    throw new Error('CLAWVAULT_PATH environment variable not set');
  }
  return path.resolve(vaultPath);
}

export function findNearestVaultPath(startPath: string = process.cwd()): string | null {
  let current = path.resolve(startPath);
  while (true) {
    if (fs.existsSync(path.join(current, '.clawvault.json'))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      return null;
    }
    current = parent;
  }
}

/**
 * Validate that a path is a valid vault directory
 */
function validateVaultPath(vaultPath: string): string | null {
  if (!vaultPath || typeof vaultPath !== 'string') return null;
  
  const resolved = path.resolve(vaultPath);
  if (!path.isAbsolute(resolved)) return null;
  
  try {
    const stat = fs.statSync(resolved);
    if (!stat.isDirectory()) return null;
  } catch {
    return null;
  }
  
  const configPath = path.join(resolved, '.clawvault.json');
  if (!fs.existsSync(configPath)) return null;
  
  return resolved;
}

/**
 * Resolve vault path for a specific agent from agentVaults config
 */
export function resolveAgentVaultPath(
  agentVaults: AgentVaultsConfig | undefined,
  agentId: string | undefined
): string | null {
  if (!agentId || typeof agentId !== 'string') return null;
  if (!agentVaults || typeof agentVaults !== 'object' || Array.isArray(agentVaults)) {
    return null;
  }
  
  const agentPath = agentVaults[agentId];
  if (!agentPath || typeof agentPath !== 'string') return null;
  
  return validateVaultPath(agentPath);
}

export interface ResolveVaultPathOptions {
  explicitPath?: string;
  cwd?: string;
  agentId?: string;
  pluginConfig?: PluginConfig;
}

function toNonEmptyPathString(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function coercePathCandidate(value: unknown): string | null {
  const direct = toNonEmptyPathString(value);
  if (direct) return direct;
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }

  const record = value as Record<string, unknown>;
  const candidates = [record.explicitPath, record.vaultPath, record.path, record.vault];
  for (const candidate of candidates) {
    const normalized = toNonEmptyPathString(candidate);
    if (normalized) {
      return normalized;
    }
  }
  return null;
}

/**
 * Resolve vault path with support for per-agent vault paths.
 * 
 * Resolution order:
 * 1. Explicit path (--vault flag)
 * 2. Per-agent vault from agentVaults config (if agentId provided)
 * 3. Plugin config vaultPath (fallback for all agents)
 * 4. CLAWVAULT_PATH environment variable
 * 5. Walk up from cwd to find nearest vault
 */
export function resolveVaultPath(options: ResolveVaultPathOptions = {}): string {
  // 1. Explicit path takes precedence
  const explicitPath = coercePathCandidate(options.explicitPath);
  if (explicitPath) {
    return path.resolve(explicitPath);
  }

  // 2. Check agentVaults for per-agent vault path
  if (options.agentId && options.pluginConfig?.agentVaults) {
    const agentVaultPath = resolveAgentVaultPath(
      options.pluginConfig.agentVaults,
      options.agentId
    );
    if (agentVaultPath) {
      return agentVaultPath;
    }
  }

  // 3. Check plugin config vaultPath (fallback)
  const configuredVaultPath = coercePathCandidate(options.pluginConfig?.vaultPath);
  if (configuredVaultPath) {
    const validated = validateVaultPath(configuredVaultPath);
    if (validated) return validated;
  }

  // 4. Check CLAWVAULT_PATH environment variable
  const envVaultPath = toNonEmptyPathString(process.env.CLAWVAULT_PATH);
  if (envVaultPath) {
    return path.resolve(envVaultPath);
  }

  // 5. Walk up from cwd to find nearest vault
  const discovered = findNearestVaultPath(options.cwd ?? process.cwd());
  if (discovered) {
    return discovered;
  }

  throw new Error('No vault path found. Set CLAWVAULT_PATH, use --vault, or run inside a vault.');
}
