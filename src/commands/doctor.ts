import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { spawnSync } from 'child_process';
import { parse as parseYaml } from 'yaml';
import { ClawVault, findVault } from '../lib/vault.js';
import { findNearestVaultPath } from '../lib/config.js';
import { hasQmd, QMD_INSTALL_COMMAND, QMD_INSTALL_URL } from '../lib/search.js';
import { listQmdCollections, type QmdCollectionInfo } from '../lib/qmd-collections.js';
import { loadVaultQmdConfig } from '../lib/vault-qmd-config.js';

export type DoctorStatus = 'ok' | 'warn' | 'error';

export interface DoctorCheck {
  label: string;
  status: DoctorStatus;
  detail?: string;
  hint?: string;
  category?: 'system' | 'vault' | 'config' | 'search' | 'integration' | 'storage' | 'migration';
}

export interface DoctorOptions {
  vaultPath?: string;
  fix?: boolean;
}

export interface DoctorReport {
  generatedAt: string;
  vaultPath?: string;
  checks: DoctorCheck[];
  warnings: number;
  errors: number;
  migrationIssues: MigrationIssue[];
}

export interface MigrationIssue {
  type: MigrationIssueType;
  description: string;
  autoFixable: boolean;
  details?: Record<string, unknown>;
}

export type MigrationIssueType =
  | 'stale_collection_name'
  | 'missing_qmd_collection'
  | 'wrong_vault_path'
  | 'orphaned_collection'
  | 'missing_qmd_config'
  | 'legacy_config_format';

const CONFIG_FILE_CANDIDATES = ['.clawvault.json', '.clawvault.yaml', '.clawvault.yml'] as const;
const RECOMMENDED_NODE_MAJOR = 22;
const MINIMUM_NODE_MAJOR = 18;
const MINIMUM_NPM_MAJOR = 9;
const DISK_ERROR_THRESHOLD_BYTES = 512 * 1024 * 1024;
const DISK_WARN_THRESHOLD_BYTES = 5 * 1024 * 1024 * 1024;
const V2_COLLECTION_PATTERNS = [/^clawvault$/i, /^vault$/i, /^memory$/i, /^notes$/i];

function humanBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function majorVersionOf(versionText: string): number | null {
  const match = versionText.trim().match(/^v?(\d+)/);
  if (!match) return null;
  const parsed = Number.parseInt(match[1], 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function toNonEmptyString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed ? trimmed : undefined;
}

function coerceVaultPath(value: unknown): string | undefined {
  const direct = toNonEmptyString(value);
  if (direct) return direct;
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return undefined;
  }

  const record = value as Record<string, unknown>;
  const candidates = [record.vaultPath, record.path, record.vault, record.explicitPath];
  for (const candidate of candidates) {
    const coerced = toNonEmptyString(candidate);
    if (coerced) {
      return coerced;
    }
  }
  return undefined;
}

function normalizeDoctorOptions(input?: string | DoctorOptions): DoctorOptions {
  if (typeof input === 'string') {
    return { vaultPath: input };
  }
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    return {};
  }

  const raw = input as Record<string, unknown>;
  const normalized: DoctorOptions = {};
  const vaultPath = coerceVaultPath(raw.vaultPath);
  if (vaultPath) {
    normalized.vaultPath = vaultPath;
  }
  if (typeof raw.fix === 'boolean') {
    normalized.fix = raw.fix;
  }
  return normalized;
}

function resolveDoctorVaultPath(explicitPath?: string): string | undefined {
  const normalizedExplicitPath = coerceVaultPath(explicitPath);
  if (normalizedExplicitPath) {
    return path.resolve(normalizedExplicitPath);
  }
  const envVaultPath = toNonEmptyString(process.env.CLAWVAULT_PATH);
  if (envVaultPath) {
    return path.resolve(envVaultPath);
  }
  const nearest = findNearestVaultPath(process.cwd());
  return nearest ? path.resolve(nearest) : undefined;
}

function selectConfigPath(vaultPath: string): string | undefined {
  for (const filename of CONFIG_FILE_CANDIDATES) {
    const candidate = path.join(vaultPath, filename);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

function parseConfigDocument(configPath: string): { data?: Record<string, unknown>; error?: string; format: 'json' | 'yaml' } {
  try {
    const raw = fs.readFileSync(configPath, 'utf-8');
    const isJson = configPath.endsWith('.json');
    const parsed = isJson ? JSON.parse(raw) : parseYaml(raw);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {
        format: isJson ? 'json' : 'yaml',
        error: 'Config root must be an object.'
      };
    }
    return {
      format: isJson ? 'json' : 'yaml',
      data: parsed as Record<string, unknown>
    };
  } catch (error: any) {
    return {
      format: configPath.endsWith('.json') ? 'json' : 'yaml',
      error: error?.message || 'Unable to parse config file.'
    };
  }
}

function isLikelyV2CollectionName(name: string): boolean {
  return V2_COLLECTION_PATTERNS.some((pattern) => pattern.test(name));
}

function checkQmdCollectionExists(
  collections: QmdCollectionInfo[],
  expectedName: string
): { exists: boolean; collection?: QmdCollectionInfo } {
  const found = collections.find((collection) => collection.name === expectedName);
  return { exists: Boolean(found), collection: found };
}

function checkCollectionPathMatches(collection: QmdCollectionInfo, expectedRoot: string): boolean {
  if (!collection.root) return false;
  return path.resolve(collection.root) === path.resolve(expectedRoot);
}

function detectMigrationIssues(
  vaultPath: string,
  configuredCollection: string | undefined,
  configuredRoot: string | undefined
): MigrationIssue[] {
  const issues: MigrationIssue[] = [];
  if (!hasQmd()) return issues;

  let collections: QmdCollectionInfo[];
  try {
    collections = listQmdCollections();
  } catch {
    return issues;
  }

  const vaultConfig = loadVaultQmdConfig(vaultPath);
  const expectedCollection = configuredCollection || vaultConfig.qmdCollection;
  const expectedRoot = configuredRoot || vaultConfig.qmdRoot;
  if (!expectedCollection || !expectedRoot) return issues;

  const { exists, collection } = checkQmdCollectionExists(collections, expectedCollection);
  if (!exists) {
    const potentialV2Collections = collections.filter(
      (entry) => isLikelyV2CollectionName(entry.name)
        && entry.root
        && path.resolve(entry.root) === path.resolve(expectedRoot)
    );

    if (potentialV2Collections.length > 0) {
      issues.push({
        type: 'stale_collection_name',
        description: `Found v2-style collection "${potentialV2Collections[0].name}" that should be renamed to "${expectedCollection}"`,
        autoFixable: true,
        details: {
          oldName: potentialV2Collections[0].name,
          newName: expectedCollection,
          root: potentialV2Collections[0].root
        }
      });
    } else {
      issues.push({
        type: 'missing_qmd_collection',
        description: `qmd collection "${expectedCollection}" does not exist`,
        autoFixable: true,
        details: {
          collectionName: expectedCollection,
          expectedRoot
        }
      });
    }
  } else if (collection && !checkCollectionPathMatches(collection, expectedRoot)) {
    issues.push({
      type: 'wrong_vault_path',
      description: `Collection "${expectedCollection}" points to "${collection.root}" but vault is at "${expectedRoot}"`,
      autoFixable: true,
      details: {
        collectionName: expectedCollection,
        currentRoot: collection.root,
        expectedRoot
      }
    });
  }

  const orphanedCollections = collections.filter((entry) => {
    if (entry.name === expectedCollection || !entry.root) return false;
    const collectionRoot = path.resolve(entry.root);
    const vaultRoot = path.resolve(expectedRoot);
    return collectionRoot === vaultRoot || collectionRoot.startsWith(`${vaultRoot}${path.sep}`);
  });
  for (const orphan of orphanedCollections) {
    issues.push({
      type: 'orphaned_collection',
      description: `Orphaned collection "${orphan.name}" points to vault path but is not the configured collection`,
      autoFixable: true,
      details: {
        collectionName: orphan.name,
        root: orphan.root
      }
    });
  }

  const jsonConfigPath = path.join(vaultPath, '.clawvault.json');
  if (fs.existsSync(jsonConfigPath)) {
    try {
      const config = JSON.parse(fs.readFileSync(jsonConfigPath, 'utf-8')) as Record<string, unknown>;
      if (!config.qmdCollection || !config.qmdRoot) {
        issues.push({
          type: 'missing_qmd_config',
          description: 'Vault config is missing qmdCollection or qmdRoot settings',
          autoFixable: true,
          details: {
            hasQmdCollection: Boolean(config.qmdCollection),
            hasQmdRoot: Boolean(config.qmdRoot)
          }
        });
      }
    } catch {
      issues.push({
        type: 'legacy_config_format',
        description: 'Unable to parse .clawvault.json - may need migration',
        autoFixable: false
      });
    }
  }

  return issues;
}

function migrationIssuesToChecks(issues: MigrationIssue[]): DoctorCheck[] {
  return issues.map((issue) => ({
    label: `migration: ${issue.type.replace(/_/g, ' ')}`,
    status: 'warn',
    detail: issue.description,
    hint: issue.autoFixable
      ? 'Run `clawvault migrate` to auto-fix this issue.'
      : 'Manual intervention required.',
    category: 'migration'
  }));
}

async function resolveVault(vaultPath?: string): Promise<ClawVault> {
  if (vaultPath) {
    const vault = new ClawVault(path.resolve(vaultPath));
    await vault.load();
    return vault;
  }
  const found = await findVault();
  if (!found) {
    throw new Error('No ClawVault found. Run `clawvault init` first.');
  }
  return found;
}

function computeAvailableDiskBytes(targetPath: string): number | null {
  const statfsSync = (fs as typeof fs & {
    statfsSync?: (path: fs.PathLike, options?: { bigint?: boolean }) => {
      bsize?: number | bigint;
      frsize?: number | bigint;
      bavail: number | bigint;
    };
  }).statfsSync;
  if (typeof statfsSync !== 'function') {
    return null;
  }

  const stats = statfsSync(targetPath, { bigint: true });
  if (typeof stats.bsize === 'undefined') {
    return null;
  }
  const blockSize = typeof stats.bsize === 'bigint' ? stats.bsize : BigInt(stats.bsize);
  const availableBlocks = typeof stats.bavail === 'bigint' ? stats.bavail : BigInt(stats.bavail);
  const availableBytes = blockSize * availableBlocks;
  return Number(availableBytes);
}

export async function doctor(input?: string | DoctorOptions): Promise<DoctorReport> {
  const options = normalizeDoctorOptions(input);
  const checks: DoctorCheck[] = [];
  const migrationIssues: MigrationIssue[] = [];

  const pushCheck = (check: DoctorCheck): void => {
    checks.push(check);
  };

  const nodeVersion = process.versions.node;
  const nodeMajor = majorVersionOf(nodeVersion);
  if (nodeMajor === null) {
    pushCheck({
      label: 'Node.js version',
      status: 'error',
      detail: `Unable to parse Node.js version "${nodeVersion}"`,
      hint: `Install Node.js ${MINIMUM_NODE_MAJOR}+ (recommended ${RECOMMENDED_NODE_MAJOR}+).`,
      category: 'system'
    });
  } else if (nodeMajor < MINIMUM_NODE_MAJOR) {
    pushCheck({
      label: 'Node.js version',
      status: 'error',
      detail: `Detected ${nodeVersion}; minimum supported is ${MINIMUM_NODE_MAJOR}.`,
      hint: `Upgrade Node.js to ${RECOMMENDED_NODE_MAJOR}+ and reinstall clawvault.`,
      category: 'system'
    });
  } else if (nodeMajor < RECOMMENDED_NODE_MAJOR) {
    pushCheck({
      label: 'Node.js version',
      status: 'warn',
      detail: `Detected ${nodeVersion}; recommended runtime is ${RECOMMENDED_NODE_MAJOR}+.`,
      hint: `Upgrade Node.js for best compatibility: use nvm/fnm and install Node.js ${RECOMMENDED_NODE_MAJOR}+.`,
      category: 'system'
    });
  } else {
    pushCheck({
      label: 'Node.js version',
      status: 'ok',
      detail: nodeVersion,
      category: 'system'
    });
  }

  const npmVersionResult = spawnSync('npm', ['--version'], { encoding: 'utf-8', shell: process.platform === 'win32' });
  const npmPrefixResult = spawnSync('npm', ['config', 'get', 'prefix'], {
    encoding: 'utf-8',
    shell: process.platform === 'win32'
  });
  if (npmPrefixResult.error || npmPrefixResult.status !== 0) {
    pushCheck({
      label: 'npm global install location',
      status: 'error',
      detail: npmPrefixResult.error?.message || npmPrefixResult.stderr?.trim() || 'Unable to read npm global prefix.',
      hint: 'Verify npm is installed and available in PATH.',
      category: 'system'
    });
  } else {
    const prefix = String(npmPrefixResult.stdout ?? '').trim();
    const npmVersionText = npmVersionResult.status === 0
      ? String(npmVersionResult.stdout ?? '').trim()
      : 'unknown';
    const npmMajor = majorVersionOf(npmVersionText);
    if (!prefix) {
      pushCheck({
        label: 'npm global install location',
        status: 'warn',
        detail: `npm ${npmVersionText}, but global prefix is empty.`,
        hint: 'Run `npm config get prefix` and configure a writable prefix if needed.',
        category: 'system'
      });
    } else {
      let writable = false;
      try {
        fs.accessSync(prefix, fs.constants.W_OK);
        writable = true;
      } catch {
        writable = false;
      }

      if (!writable) {
        pushCheck({
          label: 'npm global install location',
          status: 'warn',
          detail: `npm ${npmVersionText}, prefix ${prefix} is not writable.`,
          hint: 'Run `npm config set prefix ~/.npm-global` and add `~/.npm-global/bin` to PATH.',
          category: 'system'
        });
      } else if (npmMajor !== null && npmMajor < MINIMUM_NPM_MAJOR) {
        pushCheck({
          label: 'npm global install location',
          status: 'warn',
          detail: `npm ${npmVersionText}, prefix ${prefix} is writable.`,
          hint: `Upgrade npm to ${MINIMUM_NPM_MAJOR}+ for best install compatibility.`,
          category: 'system'
        });
      } else {
        pushCheck({
          label: 'npm global install location',
          status: 'ok',
          detail: `npm ${npmVersionText}, prefix ${prefix} is writable.`,
          category: 'system'
        });
      }
    }
  }

  const qmdInstalled = hasQmd();
  if (qmdInstalled) {
    pushCheck({
      label: 'qmd availability',
      status: 'ok',
      detail: 'qmd detected in PATH (optional fallback enabled).',
      category: 'search'
    });
  } else {
    pushCheck({
      label: 'qmd availability',
      status: 'warn',
      detail: 'qmd not found in PATH (optional dependency).',
      hint: `In-process BM25 is available by default. To enable qmd fallback: ${QMD_INSTALL_COMMAND} (${QMD_INSTALL_URL})`,
      category: 'search'
    });
  }

  const resolvedVaultPath = resolveDoctorVaultPath(options.vaultPath);
  let vaultPathForChecks: string | undefined = resolvedVaultPath;
  let vaultReadyForDeepChecks = false;
  if (!resolvedVaultPath) {
    pushCheck({
      label: 'vault directory',
      status: 'warn',
      detail: 'No vault path detected from --vault, CLAWVAULT_PATH, or current directory.',
      hint: 'Run `clawvault init <path>` and set `CLAWVAULT_PATH` (or use `--vault`).',
      category: 'vault'
    });
  } else {
    try {
      const stats = fs.statSync(resolvedVaultPath);
      if (!stats.isDirectory()) {
        pushCheck({
          label: 'vault directory',
          status: 'error',
          detail: `${resolvedVaultPath} exists but is not a directory.`,
          hint: 'Use `clawvault doctor --vault <vault-directory>` with a valid vault folder.',
          category: 'vault'
        });
      } else {
        try {
          fs.accessSync(resolvedVaultPath, fs.constants.W_OK);
          pushCheck({
            label: 'vault directory',
            status: 'ok',
            detail: `${resolvedVaultPath} is writable.`,
            category: 'vault'
          });
          vaultReadyForDeepChecks = true;
        } catch {
          pushCheck({
            label: 'vault directory',
            status: 'error',
            detail: `${resolvedVaultPath} exists but is not writable.`,
            hint: 'Fix directory permissions (`chmod`/`chown`) or select another vault path.',
            category: 'vault'
          });
        }
      }
    } catch {
      pushCheck({
        label: 'vault directory',
        status: 'error',
        detail: `${resolvedVaultPath} does not exist.`,
        hint: 'Create the vault with `clawvault init <path>` or point to an existing vault via `--vault`.',
        category: 'vault'
      });
    }
  }

  let parsedConfig: Record<string, unknown> | undefined;
  if (!vaultReadyForDeepChecks || !vaultPathForChecks) {
    pushCheck({
      label: 'vault config file',
      status: 'warn',
      detail: 'Skipped because no usable vault directory was detected.',
      hint: 'Resolve vault directory issues first, then rerun `clawvault doctor`.',
      category: 'config'
    });
  } else {
    const configPath = selectConfigPath(vaultPathForChecks);
    if (!configPath) {
      pushCheck({
        label: 'vault config file',
        status: 'error',
        detail: 'No config file found (.clawvault.json/.yaml/.yml).',
        hint: 'Run `clawvault init` or create a valid vault config file.',
        category: 'config'
      });
    } else {
      const parsed = parseConfigDocument(configPath);
      if (parsed.error || !parsed.data) {
        pushCheck({
          label: 'vault config file',
          status: 'error',
          detail: `${path.basename(configPath)} is invalid ${parsed.format.toUpperCase()}: ${parsed.error}`,
          hint: `Fix ${path.basename(configPath)} syntax and rerun \`clawvault doctor\`.`,
          category: 'config'
        });
      } else {
        parsedConfig = parsed.data;
        pushCheck({
          label: 'vault config file',
          status: 'ok',
          detail: `${path.basename(configPath)} (${parsed.format.toUpperCase()})`,
          category: 'config'
        });
      }
    }
  }

  pushCheck({
    label: 'in-process BM25 engine',
    status: 'ok',
    detail: 'Built-in search backend is available.',
    category: 'search'
  });

  if (!parsedConfig) {
    pushCheck({
      label: 'semantic embeddings',
      status: 'warn',
      detail: 'Could not evaluate embeddings configuration without a valid vault config.',
      hint: 'Fix vault config issues, then configure embeddings with `clawvault config set search.embeddings.provider <provider>`.',
      category: 'search'
    });
  } else {
    const search = (
      parsedConfig.search && typeof parsedConfig.search === 'object' && !Array.isArray(parsedConfig.search)
        ? parsedConfig.search
        : {}
    ) as Record<string, unknown>;
    const embeddings = (
      search.embeddings && typeof search.embeddings === 'object' && !Array.isArray(search.embeddings)
        ? search.embeddings
        : {}
    ) as Record<string, unknown>;
    const backend = typeof search.backend === 'string' ? search.backend : 'in-process';
    const provider = typeof embeddings.provider === 'string' ? embeddings.provider : 'none';
    const apiKey = typeof embeddings.apiKey === 'string' ? embeddings.apiKey.trim() : '';

    if (backend === 'qmd' && !qmdInstalled) {
      pushCheck({
        label: 'search backend configuration',
        status: 'warn',
        detail: 'Config prefers qmd backend, but qmd is not installed.',
        hint: 'Install qmd or run `clawvault config set search.backend in-process`.',
        category: 'search'
      });
    } else {
      pushCheck({
        label: 'search backend configuration',
        status: 'ok',
        detail: `backend=${backend}`,
        category: 'search'
      });
    }

    if (provider === 'none') {
      pushCheck({
        label: 'semantic embeddings',
        status: 'warn',
        detail: 'Embeddings provider is not configured.',
        hint: 'Configure semantic search: `clawvault config set search.embeddings.provider openai|gemini|ollama`.',
        category: 'search'
      });
    } else if ((provider === 'openai' || provider === 'gemini') && !apiKey
      && !process.env.OPENAI_API_KEY
      && !process.env.GEMINI_API_KEY
      && !process.env.GOOGLE_API_KEY) {
      pushCheck({
        label: 'semantic embeddings',
        status: 'warn',
        detail: `${provider} provider configured, but no API key was found.`,
        hint: provider === 'openai'
          ? 'Set `search.embeddings.apiKey` or export `OPENAI_API_KEY`.'
          : 'Set `search.embeddings.apiKey` or export `GEMINI_API_KEY`/`GOOGLE_API_KEY`.',
        category: 'search'
      });
    } else {
      pushCheck({
        label: 'semantic embeddings',
        status: 'ok',
        detail: `${provider} provider configured.`,
        category: 'search'
      });
    }
  }

  const openClawResult = spawnSync('openclaw', ['hooks', 'list', '--verbose'], {
    encoding: 'utf-8',
    shell: process.platform === 'win32'
  });
  const openClawError = openClawResult.error as NodeJS.ErrnoException | undefined;
  if (openClawError?.code === 'ENOENT') {
    pushCheck({
      label: 'OpenClaw plugin registration',
      status: 'warn',
      detail: 'openclaw CLI not found; integration check skipped.',
      hint: 'Install OpenClaw CLI, then run `openclaw hooks install clawvault && openclaw hooks enable clawvault`.',
      category: 'integration'
    });
  } else if (openClawError || openClawResult.status !== 0) {
    pushCheck({
      label: 'OpenClaw plugin registration',
      status: 'warn',
      detail: openClawError?.message || openClawResult.stderr?.trim() || 'Unable to read OpenClaw hooks list.',
      hint: 'Run `openclaw hooks list --verbose` to inspect hook/plugin state.',
      category: 'integration'
    });
  } else if (/clawvault/i.test(`${openClawResult.stdout}\n${openClawResult.stderr}`)) {
    pushCheck({
      label: 'OpenClaw plugin registration',
      status: 'ok',
      detail: 'clawvault appears in OpenClaw hook/plugin list.',
      category: 'integration'
    });
  } else {
    pushCheck({
      label: 'OpenClaw plugin registration',
      status: 'warn',
      detail: 'clawvault is not currently registered in OpenClaw hooks/plugins.',
      hint: 'Run `openclaw hooks install clawvault && openclaw hooks enable clawvault`.',
      category: 'integration'
    });
  }

  if (!vaultReadyForDeepChecks || !vaultPathForChecks) {
    pushCheck({
      label: 'disk space',
      status: 'warn',
      detail: 'Skipped because vault directory is unavailable.',
      hint: 'Resolve vault directory issues and rerun doctor.',
      category: 'storage'
    });
  } else {
    try {
      const availableBytes = computeAvailableDiskBytes(vaultPathForChecks);
      if (availableBytes === null) {
        pushCheck({
          label: 'disk space',
          status: 'warn',
          detail: 'Disk space check is not supported on this platform/runtime.',
          hint: 'Use `df -h` (Linux/macOS) or File Explorer properties (Windows) to verify free space.',
          category: 'storage'
        });
      } else if (availableBytes < DISK_ERROR_THRESHOLD_BYTES) {
        pushCheck({
          label: 'disk space',
          status: 'error',
          detail: `${humanBytes(availableBytes)} free in vault filesystem.`,
          hint: 'Free disk space immediately or move the vault to a filesystem with more capacity.',
          category: 'storage'
        });
      } else if (availableBytes < DISK_WARN_THRESHOLD_BYTES) {
        pushCheck({
          label: 'disk space',
          status: 'warn',
          detail: `${humanBytes(availableBytes)} free in vault filesystem.`,
          hint: 'Consider freeing space to avoid indexing and write failures.',
          category: 'storage'
        });
      } else {
        pushCheck({
          label: 'disk space',
          status: 'ok',
          detail: `${humanBytes(availableBytes)} free in vault filesystem.`,
          category: 'storage'
        });
      }
    } catch (error: any) {
      pushCheck({
        label: 'disk space',
        status: 'warn',
        detail: error?.message || 'Unable to read filesystem stats.',
        hint: 'Verify disk free space manually and rerun doctor.',
        category: 'storage'
      });
    }
  }

  const gitResult = spawnSync('git', ['--version'], { encoding: 'utf-8', shell: process.platform === 'win32' });
  if (gitResult.error || gitResult.status !== 0) {
    pushCheck({
      label: 'git availability',
      status: 'warn',
      detail: gitResult.error?.message || gitResult.stderr?.trim() || 'git is not available in PATH.',
      hint: 'Install git and ensure `git --version` works in this shell.',
      category: 'system'
    });
  } else {
    pushCheck({
      label: 'git availability',
      status: 'ok',
      detail: String(gitResult.stdout ?? '').trim() || 'git available',
      category: 'system'
    });
  }

  pushCheck({
    label: 'OS / architecture',
    status: 'ok',
    detail: `${os.platform()} ${os.release()} (${os.arch()})`,
    category: 'system'
  });

  if (qmdInstalled && vaultReadyForDeepChecks && vaultPathForChecks) {
    try {
      const vault = await resolveVault(vaultPathForChecks);
      const detectedMigrationIssues = detectMigrationIssues(
        vault.getPath(),
        vault.getQmdCollection(),
        vault.getQmdRoot()
      );
      migrationIssues.push(...detectedMigrationIssues);
      for (const check of migrationIssuesToChecks(detectedMigrationIssues)) {
        pushCheck(check);
      }
      vaultPathForChecks = vault.getPath();
    } catch {
      // Migration diagnostics are best effort and should not block doctor output.
    }
  }

  const warnings = checks.filter((check) => check.status === 'warn').length;
  const errors = checks.filter((check) => check.status === 'error').length;
  return {
    generatedAt: new Date().toISOString(),
    vaultPath: vaultPathForChecks,
    checks,
    warnings,
    errors,
    migrationIssues
  };
}
