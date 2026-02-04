import * as fs from 'fs';
import * as path from 'path';
import { execFileSync } from 'child_process';
import { ClawVault } from '../lib/vault.js';
import { hasQmd, QmdUnavailableError } from '../lib/search.js';
import { formatAge } from '../lib/time.js';
import { scanVaultLinks } from '../lib/backlinks.js';
import type { CheckpointData } from './checkpoint.js';

export interface VaultStatus {
  vaultName: string;
  vaultPath: string;
  health: 'ok' | 'warning';
  issues: string[];
  checkpoint: {
    exists: boolean;
    timestamp?: string;
    age?: string;
    sessionKey?: string;
    model?: string;
    tokenEstimate?: number;
  };
  qmd: {
    collection: string;
    root: string;
    indexStatus: 'present' | 'missing' | 'root-mismatch';
    error?: string;
  };
  git?: {
    repoRoot: string;
    clean: boolean;
    dirtyCount: number;
  };
  links: {
    total: number;
    orphans: number;
  };
  documents: number;
  categories: Record<string, number>;
}

const CLAWVAULT_DIR = '.clawvault';
const CHECKPOINT_FILE = 'last-checkpoint.json';
const DIRTY_DEATH_FLAG = 'dirty-death.flag';

function findGitRoot(startPath: string): string | null {
  let current = path.resolve(startPath);
  while (true) {
    if (fs.existsSync(path.join(current, '.git'))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) return null;
    current = parent;
  }
}

function getGitStatus(repoRoot: string): { clean: boolean; dirtyCount: number } {
  const output = execFileSync('git', ['-C', repoRoot, 'status', '--porcelain'], {
    encoding: 'utf-8'
  });
  const lines = output.split('\n').filter(Boolean);
  return { clean: lines.length === 0, dirtyCount: lines.length };
}

function parseQmdCollections(raw: string): Array<Record<string, unknown> | string> {
  const parsed = JSON.parse(raw) as unknown;
  if (Array.isArray(parsed)) return parsed;
  if (parsed && typeof parsed === 'object') {
    const candidate = (parsed as { collections?: unknown; items?: unknown; data?: unknown }).collections
      ?? (parsed as { collections?: unknown; items?: unknown; data?: unknown }).items
      ?? (parsed as { collections?: unknown; items?: unknown; data?: unknown }).data;
    if (Array.isArray(candidate)) return candidate as Array<Record<string, unknown> | string>;
  }
  throw new Error('qmd collection list returned an unexpected JSON shape.');
}

function getQmdIndexStatus(collection: string, root: string): 'present' | 'missing' | 'root-mismatch' {
  const output = execFileSync('qmd', ['collection', 'list', '--json'], { encoding: 'utf-8' });
  const entries = parseQmdCollections(output);
  const resolvedRoot = path.resolve(root);

  for (const entry of entries) {
    if (typeof entry === 'string') {
      if (entry === collection) {
        return 'present';
      }
      continue;
    }

    const name = String(
      (entry as { name?: unknown; collection?: unknown; id?: unknown }).name
        ?? (entry as { name?: unknown; collection?: unknown; id?: unknown }).collection
        ?? (entry as { name?: unknown; collection?: unknown; id?: unknown }).id
        ?? ''
    );
    if (name !== collection) continue;

    const entryRoot = (entry as { root?: unknown; path?: unknown; dir?: unknown }).root
      ?? (entry as { root?: unknown; path?: unknown; dir?: unknown }).path
      ?? (entry as { root?: unknown; path?: unknown; dir?: unknown }).dir;
    if (entryRoot) {
      const resolvedEntryRoot = path.resolve(String(entryRoot));
      return resolvedEntryRoot === resolvedRoot ? 'present' : 'root-mismatch';
    }
    return 'present';
  }

  return 'missing';
}

function loadCheckpoint(vaultPath: string): { data: CheckpointData | null; error?: string } {
  const checkpointPath = path.join(vaultPath, CLAWVAULT_DIR, CHECKPOINT_FILE);
  if (!fs.existsSync(checkpointPath)) {
    return { data: null };
  }
  try {
    const data = JSON.parse(fs.readFileSync(checkpointPath, 'utf-8')) as CheckpointData;
    return { data };
  } catch (err: any) {
    return { data: null, error: err?.message || 'Failed to parse checkpoint' };
  }
}

export async function getStatus(vaultPath: string): Promise<VaultStatus> {
  if (!hasQmd()) {
    throw new QmdUnavailableError();
  }

  const vault = new ClawVault(path.resolve(vaultPath));
  await vault.load();
  const stats = await vault.stats();
  const linkScan = scanVaultLinks(vault.getPath());

  const issues: string[] = [];
  const checkpointInfo = loadCheckpoint(vault.getPath());
  const checkpoint = checkpointInfo.data;

  if (checkpointInfo.error) {
    issues.push(`Checkpoint parse error: ${checkpointInfo.error}`);
  }

  const checkpointStatus: VaultStatus['checkpoint'] = {
    exists: Boolean(checkpoint),
    timestamp: checkpoint?.timestamp,
    age: checkpoint?.timestamp
      ? formatAge(Date.now() - new Date(checkpoint.timestamp).getTime())
      : undefined,
    sessionKey: checkpoint?.sessionKey,
    model: checkpoint?.model,
    tokenEstimate: checkpoint?.tokenEstimate
  };

  if (!checkpointStatus.exists) {
    issues.push('No checkpoint found');
  }

  const dirtyFlagPath = path.join(vault.getPath(), CLAWVAULT_DIR, DIRTY_DEATH_FLAG);
  if (fs.existsSync(dirtyFlagPath)) {
    issues.push('Dirty death flag is set');
  }

  const qmdCollection = vault.getQmdCollection();
  const qmdRoot = vault.getQmdRoot();
  let qmdIndexStatus: VaultStatus['qmd']['indexStatus'] = 'missing';
  let qmdError: string | undefined;
  try {
    qmdIndexStatus = getQmdIndexStatus(qmdCollection, qmdRoot);
    if (qmdIndexStatus !== 'present') {
      issues.push(`qmd collection ${qmdIndexStatus.replace('-', ' ')}`);
    }
  } catch (err: any) {
    qmdError = err?.message || 'Failed to check qmd index';
    issues.push(`qmd status error: ${qmdError}`);
  }

  let gitStatus: VaultStatus['git'] | undefined;
  const gitRoot = findGitRoot(vault.getPath());
  if (gitRoot) {
    try {
      const gitInfo = getGitStatus(gitRoot);
      gitStatus = { repoRoot: gitRoot, ...gitInfo };
      if (!gitInfo.clean) {
        issues.push(`Uncommitted changes: ${gitInfo.dirtyCount}`);
      }
    } catch (err: any) {
      issues.push(`Git status error: ${err?.message || 'unknown error'}`);
    }
  }

  return {
    vaultName: vault.getName(),
    vaultPath: vault.getPath(),
    health: issues.length === 0 ? 'ok' : 'warning',
    issues,
    checkpoint: checkpointStatus,
    qmd: {
      collection: qmdCollection,
      root: qmdRoot,
      indexStatus: qmdIndexStatus,
      error: qmdError
    },
    git: gitStatus,
    links: {
      total: linkScan.linkCount,
      orphans: linkScan.orphans.length
    },
    documents: stats.documents,
    categories: stats.categories
  };
}

export function formatStatus(status: VaultStatus): string {
  let output = 'ClawVault Status\n';
  output += '-'.repeat(40) + '\n';
  output += `Vault: ${status.vaultName}\n`;
  output += `Path: ${status.vaultPath}\n`;
  output += `Health: ${status.health}\n`;
  if (status.issues.length > 0) {
    output += `Issues: ${status.issues.join('; ')}\n`;
  } else {
    output += 'Issues: none\n';
  }

  output += '\nCheckpoint:\n';
  if (!status.checkpoint.exists) {
    output += '  - none\n';
  } else {
    output += `  - Timestamp: ${status.checkpoint.timestamp}\n`;
    if (status.checkpoint.age) {
      output += `  - Age: ${status.checkpoint.age}\n`;
    }
    if (status.checkpoint.sessionKey) {
      output += `  - Session key: ${status.checkpoint.sessionKey}\n`;
    }
    if (status.checkpoint.model) {
      output += `  - Model: ${status.checkpoint.model}\n`;
    }
    if (status.checkpoint.tokenEstimate !== undefined) {
      output += `  - Token estimate: ${status.checkpoint.tokenEstimate}\n`;
    }
  }

  output += '\nqmd:\n';
  output += `  - Collection: ${status.qmd.collection}\n`;
  output += `  - Root: ${status.qmd.root}\n`;
  output += `  - Index: ${status.qmd.indexStatus}\n`;
  if (status.qmd.error) {
    output += `  - Error: ${status.qmd.error}\n`;
  }

  if (status.git) {
    output += '\nGit:\n';
    output += `  - Repo: ${status.git.repoRoot}\n`;
    output += `  - Status: ${status.git.clean ? 'clean' : 'dirty'} (${status.git.dirtyCount} change(s))\n`;
  }

  output += '\nLinks:\n';
  output += `  - Total: ${status.links.total}\n`;
  if (status.links.orphans > 0) {
    output += `  - Orphans: ${status.links.orphans}\n`;
  }

  output += '\nDocuments:\n';
  output += `  - Total: ${status.documents}\n`;
  output += '  - By category:\n';
  for (const [category, count] of Object.entries(status.categories)) {
    output += `    * ${category}: ${count}\n`;
  }

  output += '-'.repeat(40) + '\n';
  return output;
}

export async function statusCommand(
  vaultPath: string,
  options: { json?: boolean } = {}
): Promise<void> {
  const status = await getStatus(vaultPath);
  if (options.json) {
    console.log(JSON.stringify(status, null, 2));
    return;
  }
  console.log(formatStatus(status));
}
