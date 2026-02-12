import * as path from 'path';
import type { Command } from 'commander';
import { ClawVault } from '../lib/vault.js';
import { parseObservationLines, readObservations } from '../lib/observation-reader.js';
import { estimateTokens } from '../lib/token-counter.js';
import type { Document, SearchResult } from '../types.js';

const DEFAULT_LIMIT = 5;
const MAX_SNIPPET_LENGTH = 320;
const OBSERVATION_LOOKBACK_DAYS = 7;

export type ContextFormat = 'markdown' | 'json';

export interface ContextOptions {
  vaultPath: string;
  limit?: number;
  format?: ContextFormat;
  recent?: boolean;
  includeObservations?: boolean;
  budget?: number;
}

export interface ContextEntry {
  title: string;
  path: string;
  category: string;
  score: number;
  snippet: string;
  modified: string;
  age: string;
  source: 'observation' | 'daily-note' | 'search';
}

export interface ContextResult {
  task: string;
  generated: string;
  context: ContextEntry[];
  markdown: string;
}

function formatRelativeAge(date: Date, now: number = Date.now()): string {
  const ageMs = Math.max(0, now - date.getTime());
  const days = Math.floor(ageMs / (24 * 60 * 60 * 1000));

  if (days === 0) return 'today';
  if (days < 7) return `${days} day${days === 1 ? '' : 's'} ago`;

  const weeks = Math.floor(days / 7);
  if (weeks < 5) return `${weeks} week${weeks === 1 ? '' : 's'} ago`;

  const months = Math.floor(days / 30);
  if (months < 12) return `${months} month${months === 1 ? '' : 's'} ago`;

  const years = Math.floor(days / 365);
  return `${years} year${years === 1 ? '' : 's'} ago`;
}

function normalizeSnippet(result: SearchResult): string {
  const source = (result.snippet || result.document.content || '').trim();
  if (!source) return 'No snippet available.';
  return source
    .replace(/\s+/g, ' ')
    .slice(0, MAX_SNIPPET_LENGTH);
}

export function formatContextMarkdown(task: string, entries: ContextEntry[]): string {
  let output = `## Relevant Context for: ${task}\n\n`;

  if (entries.length === 0) {
    output += '_No relevant context found._\n';
    return output;
  }

  for (const entry of entries) {
    output += `### ${entry.title} (${entry.source}, score: ${entry.score.toFixed(2)}, ${entry.age})\n`;
    output += `${entry.snippet}\n\n`;
  }

  return output.trimEnd();
}

interface PrioritizedContextItem {
  entry: ContextEntry;
  priority: number;
}

function estimateSnippet(source: string): string {
  const normalized = source.replace(/\s+/g, ' ').trim();
  if (!normalized) {
    return 'No snippet available.';
  }
  return normalized.slice(0, MAX_SNIPPET_LENGTH);
}

function parseIsoDate(value: unknown): string | null {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) {
      return trimmed;
    }
    if (/^\d{4}-\d{2}-\d{2}T/.test(trimmed)) {
      return trimmed.slice(0, 10);
    }
  }

  if (value instanceof Date) {
    const time = value.getTime();
    if (!Number.isNaN(time)) {
      return value.toISOString().slice(0, 10);
    }
  }

  return null;
}

function asDate(value: string | null, fallback: Date = new Date(0)): Date {
  if (!value) {
    return fallback;
  }
  const parsed = new Date(`${value}T00:00:00.000Z`);
  return Number.isNaN(parsed.getTime()) ? fallback : parsed;
}

function observationPriorityToRank(priority: string): number {
  if (priority === '🔴') return 1;
  if (priority === '🟡') return 4;
  return 5;
}

function observationPriorityScore(priority: string): number {
  if (priority === '🔴') return 1.0;
  if (priority === '🟡') return 0.7;
  return 0.4;
}

function isLikelyDailyNote(document: Document): boolean {
  const normalizedPath = document.path.split(path.sep).join('/').toLowerCase();
  if (normalizedPath.includes('/daily/')) {
    return true;
  }

  const category = document.category.toLowerCase();
  if (category.includes('daily')) {
    return true;
  }

  const type = typeof document.frontmatter.type === 'string'
    ? document.frontmatter.type.toLowerCase()
    : '';
  return type === 'daily';
}

function findDailyDate(document: Document, targetDates: Set<string>): string | null {
  const frontmatterDate = parseIsoDate(document.frontmatter.date);
  const titleDate = parseIsoDate(document.title);
  const fileDate = parseIsoDate(path.basename(document.path, '.md'));
  const candidates = [frontmatterDate, titleDate, fileDate].filter((value): value is string => Boolean(value));

  for (const candidate of candidates) {
    if (!targetDates.has(candidate)) {
      continue;
    }
    if (isLikelyDailyNote(document) || titleDate === candidate || fileDate === candidate) {
      return candidate;
    }
  }

  return null;
}

function getTargetDailyDates(now: Date = new Date()): string[] {
  const today = now.toISOString().slice(0, 10);
  const yesterdayDate = new Date(now);
  yesterdayDate.setDate(yesterdayDate.getDate() - 1);
  const yesterday = yesterdayDate.toISOString().slice(0, 10);
  return [today, yesterday];
}

async function buildDailyContextItems(vault: ClawVault): Promise<PrioritizedContextItem[]> {
  const allDocuments = await vault.list();
  const targetDates = getTargetDailyDates();
  const targetDateSet = new Set(targetDates);
  const byDate = new Map<string, Document>();

  for (const document of allDocuments) {
    const dailyDate = findDailyDate(document, targetDateSet);
    if (!dailyDate) {
      continue;
    }

    const existing = byDate.get(dailyDate);
    if (!existing || document.modified.getTime() > existing.modified.getTime()) {
      byDate.set(dailyDate, document);
    }
  }

  const items: PrioritizedContextItem[] = [];
  for (const date of targetDates) {
    const document = byDate.get(date);
    if (!document) {
      continue;
    }

    const relativePath = path.relative(vault.getPath(), document.path).split(path.sep).join('/');
    const snippet = estimateSnippet(document.content);
    items.push({
      priority: 2,
      entry: {
        title: `Daily note ${date}`,
        path: relativePath,
        category: document.category,
        score: 0.9,
        snippet,
        modified: document.modified.toISOString(),
        age: formatRelativeAge(document.modified),
        source: 'daily-note'
      }
    });
  }

  return items;
}

function buildObservationContextItems(vaultPath: string): PrioritizedContextItem[] {
  const observationMarkdown = readObservations(vaultPath, OBSERVATION_LOOKBACK_DAYS);
  const parsed = parseObservationLines(observationMarkdown);
  const items: PrioritizedContextItem[] = [];

  for (const observation of parsed) {
    const priority = observationPriorityToRank(observation.priority);
    const modifiedDate = asDate(observation.date, new Date());
    const date = observation.date || modifiedDate.toISOString().slice(0, 10);
    const snippet = estimateSnippet(observation.content);

    items.push({
      priority,
      entry: {
        title: `${observation.priority} observation (${date})`,
        path: `observations/${date}.md`,
        category: 'observations',
        score: observationPriorityScore(observation.priority),
        snippet,
        modified: modifiedDate.toISOString(),
        age: formatRelativeAge(modifiedDate),
        source: 'observation'
      }
    });
  }

  return items;
}

function buildSearchContextItems(vault: ClawVault, results: SearchResult[]): PrioritizedContextItem[] {
  return results.map((result): PrioritizedContextItem => {
    const relativePath = path.relative(vault.getPath(), result.document.path).split(path.sep).join('/');
    const entry: ContextEntry = {
      title: result.document.title,
      path: relativePath,
      category: result.document.category,
      score: result.score,
      snippet: normalizeSnippet(result),
      modified: result.document.modified.toISOString(),
      age: formatRelativeAge(result.document.modified),
      source: 'search'
    };

    return {
      priority: 3,
      entry
    };
  });
}

function renderEntryBlock(entry: ContextEntry): string {
  return `### ${entry.title} (${entry.source}, score: ${entry.score.toFixed(2)}, ${entry.age})\n${entry.snippet}\n\n`;
}

function truncateToBudget(text: string, budget: number): string {
  if (!Number.isFinite(budget) || budget <= 0) {
    return '';
  }

  const maxChars = Math.max(0, Math.floor(budget) * 4);
  if (text.length <= maxChars) {
    return text;
  }
  return text.slice(0, maxChars).trimEnd();
}

function applyTokenBudget(items: PrioritizedContextItem[], task: string, budget?: number): { context: ContextEntry[]; markdown: string } {
  const fullContext = items.map((item) => item.entry);
  const fullMarkdown = formatContextMarkdown(task, fullContext);

  if (budget === undefined) {
    return { context: fullContext, markdown: fullMarkdown };
  }

  const normalizedBudget = Math.max(1, Math.floor(budget));
  const header = `## Relevant Context for: ${task}\n\n`;
  let remaining = normalizedBudget - estimateTokens(header);
  const selectedEntries: ContextEntry[] = [];

  for (const item of [...items].sort((a, b) => a.priority - b.priority)) {
    if (remaining <= 0) {
      break;
    }
    const cost = estimateTokens(renderEntryBlock(item.entry));
    if (cost <= remaining) {
      selectedEntries.push(item.entry);
      remaining -= cost;
    }
  }

  const markdown = truncateToBudget(formatContextMarkdown(task, selectedEntries), normalizedBudget);
  return {
    context: selectedEntries,
    markdown
  };
}

export async function buildContext(task: string, options: ContextOptions): Promise<ContextResult> {
  const normalizedTask = task.trim();
  if (!normalizedTask) {
    throw new Error('Task description is required.');
  }

  const vault = new ClawVault(path.resolve(options.vaultPath));
  await vault.load();

  const limit = Math.max(1, options.limit ?? DEFAULT_LIMIT);
  const recent = options.recent ?? true;
  const includeObservations = options.includeObservations ?? true;

  const searchResults = await vault.vsearch(normalizedTask, {
    limit,
    temporalBoost: recent
  });

  const searchItems = buildSearchContextItems(vault, searchResults);
  const dailyItems = await buildDailyContextItems(vault);
  const observationItems = includeObservations
    ? buildObservationContextItems(vault.getPath())
    : [];

  const redObservations = observationItems.filter((item) => item.priority === 1);
  const yellowObservations = observationItems.filter((item) => item.priority === 4);
  const greenObservations = observationItems.filter((item) => item.priority === 5);

  const ordered = [
    ...redObservations,
    ...dailyItems,
    ...searchItems,
    ...yellowObservations,
    ...greenObservations
  ];

  const { context, markdown } = applyTokenBudget(ordered, normalizedTask, options.budget);

  return {
    task: normalizedTask,
    generated: new Date().toISOString(),
    context,
    markdown
  };
}

export async function contextCommand(task: string, options: ContextOptions): Promise<void> {
  const result = await buildContext(task, options);
  const format = options.format ?? 'markdown';

  if (format === 'json') {
    console.log(JSON.stringify({
      task: result.task,
      generated: result.generated,
      count: result.context.length,
      context: result.context
    }, null, 2));
    return;
  }

  console.log(result.markdown);
}

function parsePositiveInteger(raw: string, label: string): number {
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`Invalid ${label}: ${raw}`);
  }
  return parsed;
}

export function registerContextCommand(program: Command): void {
  program
    .command('context <task>')
    .description('Generate task-relevant context for prompt injection')
    .option('-n, --limit <n>', 'Max results', '5')
    .option('--format <format>', 'Output format (markdown|json)', 'markdown')
    .option('--recent', 'Boost recent documents (enabled by default)', true)
    .option('--include-observations', 'Include observation memories in output', true)
    .option('--budget <number>', 'Optional token budget for assembled context')
    .option('-v, --vault <path>', 'Vault path')
    .action(async (
      task: string,
      rawOptions: {
        limit: string;
        format: string;
        recent?: boolean;
        includeObservations?: boolean;
        budget?: string;
        vault?: string;
      }
    ) => {
      const format = rawOptions.format === 'json' ? 'json' : 'markdown';
      const budget = rawOptions.budget
        ? parsePositiveInteger(rawOptions.budget, 'budget')
        : undefined;
      const limit = parsePositiveInteger(rawOptions.limit, 'limit');
      const vaultPath = rawOptions.vault
        ?? process.env.CLAWVAULT_PATH
        ?? process.cwd();

      await contextCommand(task, {
        vaultPath,
        limit,
        format,
        recent: rawOptions.recent ?? true,
        includeObservations: rawOptions.includeObservations ?? true,
        budget
      });
    });
}
