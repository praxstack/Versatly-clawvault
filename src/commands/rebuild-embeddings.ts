/**
 * Rebuild embedding cache for hybrid search.
 * Uses hosted embedding providers (OpenAI/Gemini/Ollama).
 */
import * as fs from 'fs';
import * as path from 'path';
import { resolveVaultPath } from '../lib/config.js';
import { listConfig } from '../lib/config-manager.js';
import { computeEmbeddingHash, EmbeddingStore } from '../lib/embedding-store.js';
import { embedText, resolveEmbeddingConfig } from '../lib/hosted-embeddings.js';
import type { VaultSearchConfig } from '../types.js';

export interface RebuildEmbeddingsOptions {
  force?: boolean;
  onProgress?: (current: number, total: number) => void;
}

export interface RebuildEmbeddingsResult {
  total: number;
  added: number;
  skipped: number;
}

function walkDir(dir: string, base: string): string[] {
  const files: string[] = [];
  try {
    for (const entry of fs.readdirSync(dir)) {
      if (entry.startsWith('.') || entry === 'node_modules') continue;
      const full = path.join(dir, entry);
      const stat = fs.statSync(full);
      if (stat.isDirectory()) {
        files.push(...walkDir(full, base));
      } else if (entry.endsWith('.md')) {
        files.push(path.relative(base, full));
      }
    }
  } catch {
    // Skip inaccessible directories
  }
  return files;
}

function resolveSearchConfig(vaultPath: string): VaultSearchConfig {
  try {
    const config = listConfig(vaultPath) as Record<string, unknown>;
    const search = config.search;
    if (search && typeof search === 'object' && !Array.isArray(search)) {
      return search as VaultSearchConfig;
    }
    return {};
  } catch {
    return {};
  }
}

/**
 * Rebuild embeddings for all markdown files in a vault.
 */
export async function rebuildEmbeddingsForVault(
  vaultPath: string,
  options: RebuildEmbeddingsOptions = {}
): Promise<RebuildEmbeddingsResult> {
  const { force = false, onProgress } = options;
  const searchConfig = resolveSearchConfig(vaultPath);
  const embeddingConfig = resolveEmbeddingConfig(searchConfig);
  if (!embeddingConfig) {
    throw new Error(
      'No hosted embedding provider configured. Set search.embeddings.provider to openai, gemini, or ollama.'
    );
  }

  const store = new EmbeddingStore(vaultPath);
  store.load();
  if (force || !store.isCompatible(embeddingConfig.provider, embeddingConfig.model)) {
    store.clear();
  }
  store.setSignature(embeddingConfig.provider, embeddingConfig.model);

  const mdFiles = walkDir(vaultPath, vaultPath).filter(
    (f) => !f.startsWith('node_modules') && !f.startsWith('.')
  );
  const validDocIds = new Set<string>();

  let added = 0;
  let skipped = 0;

  for (let i = 0; i < mdFiles.length; i++) {
    const file = mdFiles[i];
    const docId = file.replace(/\.md$/, '');

    validDocIds.add(docId);

    try {
      const content = fs.readFileSync(path.join(vaultPath, file), 'utf-8');
      if (content.length < 20) {
        skipped++;
        if (onProgress) onProgress(i + 1, mdFiles.length);
        continue;
      }

      const hash = computeEmbeddingHash(content.slice(0, 12000));
      if (!force && store.getHash(docId) === hash) {
        skipped++;
        if (onProgress) onProgress(i + 1, mdFiles.length);
        continue;
      }

      const embedding = await embedText(content, embeddingConfig, { isQuery: false });
      store.set(docId, hash, embedding);
      added++;
    } catch {
      skipped++;
    }

    if (onProgress) onProgress(i + 1, mdFiles.length);
  }

  store.prune(validDocIds);
  store.save();

  return {
    total: store.size,
    added,
    skipped
  };
}

export interface RebuildEmbeddingsCommandOptions {
  vaultPath?: string;
  force?: boolean;
  quiet?: boolean;
}

export async function rebuildEmbeddingsCommand(
  options: RebuildEmbeddingsCommandOptions = {}
): Promise<RebuildEmbeddingsResult> {
  const vaultPath = resolveVaultPath({ explicitPath: options.vaultPath });

  if (!options.quiet) {
    console.log(`Rebuilding embedding cache for vault: ${vaultPath}`);
  }

  const result = await rebuildEmbeddingsForVault(vaultPath, {
    force: options.force,
    onProgress: options.quiet
      ? undefined
      : (current, total) => {
          process.stdout.write(`\r  Embedding ${current}/${total} documents...`);
        }
  });

  if (!options.quiet) {
    console.log(`\n✓ Done. ${result.total} embeddings (${result.added} new, ${result.skipped} cached)`);
  }

  return result;
}
