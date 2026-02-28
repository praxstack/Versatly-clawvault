import * as fs from 'fs';
import * as path from 'path';
import { buildEntityIndex, type EntityIndex } from './entity-index.js';
import { extractRawWikiLinks, normalizeWikiLinkTarget } from './wiki-links.js';

const CLAWVAULT_DIR = '.clawvault';
const BACKLINKS_FILE = 'backlinks.json';

export interface BacklinksScanResult {
  backlinks: Map<string, string[]>;
  orphans: Array<{ source: string; target: string }>;
  linkCount: number;
}

function ensureClawvaultDir(vaultPath: string): string {
  const dir = path.join(vaultPath, CLAWVAULT_DIR);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  return dir;
}

function toVaultId(vaultPath: string, filePath: string): string {
  const relative = path.relative(vaultPath, filePath).replace(/\.md$/, '');
  return relative.split(path.sep).join('/');
}

function normalizeLinkTarget(raw: string): string {
  return normalizeWikiLinkTarget(raw);
}

function normalizeLookupCandidate(value: string): string {
  const normalized = normalizeLinkTarget(value);
  if (!normalized) return '';

  const resolved = path.posix.normalize(normalized).replace(/^\/+/, '');
  if (!resolved || resolved === '.' || resolved.startsWith('../')) {
    return '';
  }

  return resolved;
}

function buildLookupCandidates(target: string, sourceId: string): string[] {
  const candidates: string[] = [];
  const sourceDir = path.posix.dirname(sourceId);
  const hasSourceDir = sourceDir !== '.';
  const isRelativeTarget = target.startsWith('./') || target.startsWith('../');

  const addCandidate = (candidate: string): void => {
    const normalized = normalizeLookupCandidate(candidate);
    if (!normalized || candidates.includes(normalized)) return;
    candidates.push(normalized);
  };

  if (isRelativeTarget) {
    if (hasSourceDir) {
      addCandidate(path.posix.join(sourceDir, target));
    } else {
      addCandidate(target);
    }
    if (target.startsWith('./')) {
      addCandidate(target.slice(2));
    }
    return candidates;
  }

  if (!target.includes('/')) {
    if (hasSourceDir) {
      addCandidate(`${sourceDir}/${target}`);
    }
    addCandidate(target);
    return candidates;
  }

  addCandidate(target);
  return candidates;
}

function listMarkdownFiles(vaultPath: string): string[] {
  const files: string[] = [];
  const skipDirs = new Set(['archive', 'templates', 'node_modules']);

  function walk(dir: string): void {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name.startsWith('.') || skipDirs.has(entry.name)) continue;
        walk(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        files.push(fullPath);
      }
    }
  }

  walk(vaultPath);
  return files;
}

function buildKnownIds(vaultPath: string, files: string[]): {
  ids: Set<string>;
  idsLower: Map<string, string>;
} {
  const ids = new Set<string>();
  const idsLower = new Map<string, string>();

  for (const file of files) {
    const id = toVaultId(vaultPath, file);
    ids.add(id);
    const lower = id.toLowerCase();
    if (!idsLower.has(lower)) {
      idsLower.set(lower, id);
    }
  }

  return { ids, idsLower };
}

function resolveTarget(
  target: string,
  sourceId: string,
  known: { ids: Set<string>; idsLower: Map<string, string> },
  entityIndex?: EntityIndex
): string | null {
  if (!target) return null;

  for (const candidate of buildLookupCandidates(target, sourceId)) {
    if (known.ids.has(candidate)) return candidate;
    const lowerCandidate = candidate.toLowerCase();
    if (known.idsLower.has(lowerCandidate)) {
      return known.idsLower.get(lowerCandidate)!;
    }
  }

  const lower = target.toLowerCase();
  if (entityIndex?.entries.has(lower)) return entityIndex.entries.get(lower)!;
  return null;
}

export function scanVaultLinks(
  vaultPath: string,
  options: { entityIndex?: EntityIndex } = {}
): BacklinksScanResult {
  const files = listMarkdownFiles(vaultPath);
  const known = buildKnownIds(vaultPath, files);
  const entityIndex = options.entityIndex ?? buildEntityIndex(vaultPath);

  const backlinks = new Map<string, Set<string>>();
  const orphans: Array<{ source: string; target: string }> = [];
  let linkCount = 0;

  for (const file of files) {
    const sourceId = toVaultId(vaultPath, file);
    const content = fs.readFileSync(file, 'utf-8');
    const matches = extractRawWikiLinks(content);
    linkCount += matches.length;

    for (const match of matches) {
      const target = normalizeLinkTarget(match);
      if (!target) continue;
      const resolved = resolveTarget(target, sourceId, known, entityIndex);
      if (!resolved) {
        orphans.push({ source: sourceId, target });
        continue;
      }
      if (!backlinks.has(resolved)) {
        backlinks.set(resolved, new Set());
      }
      backlinks.get(resolved)!.add(sourceId);
    }
  }

  const backlinksMap = new Map<string, string[]>();
  for (const [target, sources] of backlinks) {
    backlinksMap.set(target, [...sources].sort());
  }

  return { backlinks: backlinksMap, orphans, linkCount };
}

export function writeBacklinksIndex(vaultPath: string, backlinks: Map<string, string[]>): void {
  const dir = ensureClawvaultDir(vaultPath);
  const output: Record<string, string[]> = {};

  const targets = [...backlinks.keys()].sort();
  for (const target of targets) {
    const sources = backlinks.get(target) || [];
    output[target] = [...new Set(sources)].sort();
  }

  fs.writeFileSync(path.join(dir, BACKLINKS_FILE), JSON.stringify(output, null, 2));
}

export function readBacklinksIndex(vaultPath: string): Map<string, string[]> | null {
  const filePath = path.join(vaultPath, CLAWVAULT_DIR, BACKLINKS_FILE);
  if (!fs.existsSync(filePath)) return null;
  try {
    const raw = JSON.parse(fs.readFileSync(filePath, 'utf-8')) as Record<string, string[]>;
    const map = new Map<string, string[]>();
    for (const [target, sources] of Object.entries(raw)) {
      if (Array.isArray(sources)) {
        map.set(target, sources);
      }
    }
    return map;
  } catch {
    return null;
  }
}

export function rebuildBacklinksIndex(
  vaultPath: string,
  options: { entityIndex?: EntityIndex } = {}
): BacklinksScanResult {
  const result = scanVaultLinks(vaultPath, options);
  writeBacklinksIndex(vaultPath, result.backlinks);
  return result;
}
