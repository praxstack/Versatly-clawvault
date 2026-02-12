import * as fs from 'fs';
import * as path from 'path';
import { Compressor } from './compressor.js';
import { Reflector } from './reflector.js';
import { Router } from './router.js';

type ObservationPriority = '🔴' | '🟡' | '🟢';

interface ObservationLine {
  priority: ObservationPriority;
  content: string;
}

const DATE_HEADING_RE = /^##\s+(\d{4}-\d{2}-\d{2})\s*$/;
const OBSERVATION_LINE_RE = /^(🔴|🟡|🟢)\s+(.+)$/u;

export interface ObserverCompressor {
  compress(messages: string[], existingObservations: string): Promise<string>;
}

export interface ObserverReflector {
  reflect(observations: string): string;
}

export interface ObserverOptions {
  tokenThreshold?: number;
  reflectThreshold?: number;
  model?: string;
  compressor?: ObserverCompressor;
  reflector?: ObserverReflector;
  now?: () => Date;
}

export class Observer {
  private readonly vaultPath: string;
  private readonly observationsDir: string;
  private readonly tokenThreshold: number;
  private readonly reflectThreshold: number;
  private readonly compressor: ObserverCompressor;
  private readonly reflector: ObserverReflector;
  private readonly now: () => Date;

  private readonly router: Router;
  private pendingMessages: string[] = [];
  private observationsCache = '';
  private lastRoutingSummary = '';

  constructor(vaultPath: string, options: ObserverOptions = {}) {
    this.vaultPath = path.resolve(vaultPath);
    this.observationsDir = path.join(this.vaultPath, 'observations');
    this.tokenThreshold = options.tokenThreshold ?? 30000;
    this.reflectThreshold = options.reflectThreshold ?? 40000;
    this.now = options.now ?? (() => new Date());
    this.compressor = options.compressor ?? new Compressor({ model: options.model, now: this.now });
    this.reflector = options.reflector ?? new Reflector({ now: this.now });

    this.router = new Router(vaultPath);

    fs.mkdirSync(this.observationsDir, { recursive: true });
    this.observationsCache = this.readTodayObservations();
  }

  async processMessages(messages: string[]): Promise<void> {
    const incoming = messages.map((message) => message.trim()).filter(Boolean);
    if (incoming.length === 0) {
      return;
    }

    this.pendingMessages.push(...incoming);
    const buffered = this.pendingMessages.join('\n');
    if (this.estimateTokens(buffered) < this.tokenThreshold) {
      return;
    }

    const todayPath = this.getObservationPath(this.now());
    const existingRaw = this.readObservationFile(todayPath);
    const existing = this.deduplicateObservationMarkdown(existingRaw);
    if (existingRaw.trim() !== existing) {
      this.writeObservationFile(todayPath, existing);
    }
    const compressedRaw = (await this.compressor.compress(this.pendingMessages, existing)).trim();
    this.pendingMessages = [];
    const compressed = this.deduplicateObservationMarkdown(compressedRaw);

    if (!compressed) {
      return;
    }

    this.writeObservationFile(todayPath, compressed);
    this.observationsCache = compressed;

    // Route observations to vault categories (decisions/, lessons/, etc.)
    const { summary } = this.router.route(compressed);
    if (summary) {
      this.lastRoutingSummary = summary;
    }

    await this.reflectIfNeeded();
  }

  /**
   * Force-flush pending messages regardless of threshold.
   * Call this on session end to capture everything.
   */
  async flush(): Promise<{ observations: string; routingSummary: string }> {
    if (this.pendingMessages.length === 0) {
      return { observations: this.observationsCache, routingSummary: this.lastRoutingSummary };
    }

    const todayPath = this.getObservationPath(this.now());
    const existingRaw = this.readObservationFile(todayPath);
    const existing = this.deduplicateObservationMarkdown(existingRaw);
    if (existingRaw.trim() !== existing) {
      this.writeObservationFile(todayPath, existing);
    }
    const compressedRaw = (await this.compressor.compress(this.pendingMessages, existing)).trim();
    this.pendingMessages = [];
    const compressed = this.deduplicateObservationMarkdown(compressedRaw);

    if (compressed) {
      this.writeObservationFile(todayPath, compressed);
      this.observationsCache = compressed;
      const { summary } = this.router.route(compressed);
      this.lastRoutingSummary = summary;
      await this.reflectIfNeeded();
    }

    return { observations: this.observationsCache, routingSummary: this.lastRoutingSummary };
  }

  getObservations(): string {
    this.observationsCache = this.readTodayObservations();
    return this.observationsCache;
  }

  private estimateTokens(input: string): number {
    return Math.ceil(input.length / 4);
  }

  private getObservationPath(date: Date): string {
    const datePart = date.toISOString().split('T')[0];
    return path.join(this.observationsDir, `${datePart}.md`);
  }

  private readTodayObservations(): string {
    const todayPath = this.getObservationPath(this.now());
    return this.readObservationFile(todayPath);
  }

  private readObservationFile(filePath: string): string {
    if (!fs.existsSync(filePath)) {
      return '';
    }
    return fs.readFileSync(filePath, 'utf-8').trim();
  }

  private writeObservationFile(filePath: string, content: string): void {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, `${content.trim()}\n`, 'utf-8');
  }

  private getObservationFiles(): string[] {
    if (!fs.existsSync(this.observationsDir)) {
      return [];
    }

    return fs.readdirSync(this.observationsDir)
      .filter((name) => name.endsWith('.md'))
      .sort((a, b) => a.localeCompare(b))
      .map((name) => path.join(this.observationsDir, name));
  }

  private readObservationCorpus(): string {
    const files = this.getObservationFiles();
    if (files.length === 0) {
      return '';
    }
    return files
      .map((filePath) => this.readObservationFile(filePath))
      .filter(Boolean)
      .join('\n\n');
  }

  private deduplicateObservationMarkdown(markdown: string): string {
    const parsed = this.parseSections(markdown);
    if (parsed.size === 0) {
      return markdown.trim();
    }

    for (const [date, lines] of parsed.entries()) {
      const seen = new Set<string>();
      const deduped: ObservationLine[] = [];
      for (const line of lines) {
        const normalized = this.normalizeObservationContent(line.content);
        if (!normalized || seen.has(normalized)) {
          continue;
        }
        seen.add(normalized);
        deduped.push(line);
      }
      parsed.set(date, deduped);
    }

    return this.renderSections(parsed);
  }

  private parseSections(markdown: string): Map<string, ObservationLine[]> {
    const sections = new Map<string, ObservationLine[]>();
    let currentDate: string | null = null;

    for (const rawLine of markdown.split(/\r?\n/)) {
      const dateMatch = rawLine.match(DATE_HEADING_RE);
      if (dateMatch) {
        currentDate = dateMatch[1];
        if (!sections.has(currentDate)) {
          sections.set(currentDate, []);
        }
        continue;
      }

      if (!currentDate) {
        continue;
      }

      const lineMatch = rawLine.match(OBSERVATION_LINE_RE);
      if (!lineMatch) {
        continue;
      }

      const current = sections.get(currentDate) ?? [];
      current.push({
        priority: lineMatch[1] as ObservationPriority,
        content: lineMatch[2].trim()
      });
      sections.set(currentDate, current);
    }

    return sections;
  }

  private renderSections(sections: Map<string, ObservationLine[]>): string {
    const chunks: string[] = [];
    const dates = [...sections.keys()].sort((a, b) => a.localeCompare(b));

    for (const date of dates) {
      const lines = sections.get(date) ?? [];
      if (lines.length === 0) {
        continue;
      }
      chunks.push(`## ${date}`);
      chunks.push('');
      for (const line of lines) {
        chunks.push(`${line.priority} ${line.content}`);
      }
      chunks.push('');
    }

    return chunks.join('\n').trim();
  }

  private normalizeObservationContent(content: string): string {
    return content
      .replace(/^\d{2}:\d{2}\s+/, '')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();
  }

  private async reflectIfNeeded(): Promise<void> {
    const corpus = this.readObservationCorpus();
    if (this.estimateTokens(corpus) < this.reflectThreshold) {
      return;
    }

    for (const filePath of this.getObservationFiles()) {
      const current = this.readObservationFile(filePath);
      if (!current) continue;

      const reflected = this.reflector.reflect(current).trim();
      if (!reflected) {
        fs.rmSync(filePath, { force: true });
        continue;
      }

      this.writeObservationFile(filePath, reflected);
    }

    this.observationsCache = this.readTodayObservations();
  }
}
