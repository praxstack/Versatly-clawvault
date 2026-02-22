import * as fs from 'fs';
import * as path from 'path';
import { collectionExists, findCollectionByRoot, getFirstCollection } from './qmd-collections.js';

const CONFIG_FILE = '.clawvault.json';

interface VaultConfigPayload {
  name?: unknown;
  qmdCollection?: unknown;
  qmdRoot?: unknown;
}

export interface VaultQmdConfig {
  vaultPath: string;
  qmdCollection: string;
  qmdRoot: string;
  autoDetected?: boolean;
}

function readTrimmedString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function autoDetectCollection(vaultPath: string, fallbackName: string): { collection: string; autoDetected: boolean } {
  const byRoot = findCollectionByRoot(vaultPath);
  if (byRoot) {
    return { collection: byRoot.name, autoDetected: true };
  }

  if (collectionExists(fallbackName)) {
    return { collection: fallbackName, autoDetected: false };
  }

  const first = getFirstCollection();
  if (first) {
    return { collection: first.name, autoDetected: true };
  }

  return { collection: fallbackName, autoDetected: false };
}

export function loadVaultQmdConfig(vaultPath: string): VaultQmdConfig {
  const resolvedVaultPath = path.resolve(vaultPath);
  const fallbackName = path.basename(resolvedVaultPath);
  const fallbackRoot = resolvedVaultPath;
  const configPath = path.join(resolvedVaultPath, CONFIG_FILE);

  if (!fs.existsSync(configPath)) {
    const { collection, autoDetected } = autoDetectCollection(resolvedVaultPath, fallbackName);
    return {
      vaultPath: resolvedVaultPath,
      qmdCollection: collection,
      qmdRoot: fallbackRoot,
      autoDetected
    };
  }

  try {
    const raw = JSON.parse(fs.readFileSync(configPath, 'utf-8')) as VaultConfigPayload;
    const configuredName = readTrimmedString(raw.name) ?? fallbackName;
    const configuredCollection = readTrimmedString(raw.qmdCollection);
    const rawRoot = readTrimmedString(raw.qmdRoot) ?? fallbackRoot;
    const qmdRoot = path.isAbsolute(rawRoot)
      ? path.resolve(rawRoot)
      : path.resolve(resolvedVaultPath, rawRoot);

    if (configuredCollection && collectionExists(configuredCollection)) {
      return {
        vaultPath: resolvedVaultPath,
        qmdCollection: configuredCollection,
        qmdRoot
      };
    }

    const { collection, autoDetected } = autoDetectCollection(qmdRoot, configuredCollection ?? configuredName);
    return {
      vaultPath: resolvedVaultPath,
      qmdCollection: collection,
      qmdRoot,
      autoDetected
    };
  } catch {
    const { collection, autoDetected } = autoDetectCollection(resolvedVaultPath, fallbackName);
    return {
      vaultPath: resolvedVaultPath,
      qmdCollection: collection,
      qmdRoot: fallbackRoot,
      autoDetected
    };
  }
}
