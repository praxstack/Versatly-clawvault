import * as fs from 'fs';
import * as path from 'path';
import {
  normalizeObservationContent,
  parseObservationLine,
  parseObservationMarkdown,
  renderScoredObservationLine,
  type ObservationType
} from '../lib/observation-format.js';

/**
 * Routes observations into the appropriate vault category files.
 * Takes compressed observations and updates decisions/, people/, lessons/, etc.
 */

interface RoutedItem {
  category: string;
  title: string;
  content: string;
  type: ObservationType;
  confidence: number;
  importance: number;
  date: string;
}

const CATEGORY_PATTERNS: Array<{ category: string; patterns: RegExp[] }> = [
  {
    category: 'decisions',
    patterns: [
      /\b(decid(?:e|ed|ing|ion)|chose|picked|went with|selected|opted)\b/i,
      /\b(decision|trade[- ]?off|alternative|rationale)\b/i,
    ],
  },
  {
    category: 'lessons',
    patterns: [
      /\b(learn(?:ed|ing|t)|lesson|mistake|insight|realized|discovered)\b/i,
      /\b(note to self|remember|important|don'?t forget|never again)\b/i,
    ],
  },
  {
    category: 'people',
    patterns: [
      /\b(said|asked|told|mentioned|emailed|called|messaged|met with)\b/i,
      /\b(client|partner|team|colleague|contact)\b/i,
      /\b(?:Pedro|Justin|Maria|Sarah|[A-Z][a-z]+ (?:said|asked|told|mentioned))\b/,
      /\b(?:talked to|met with|spoke with|chatted with|discussed with)\s+[A-Z][a-z]+\b/i,
      /\b[A-Z][a-z]+\s+(?:from|at)\s+[A-Z]/,
      /\b[A-Z][a-z]+\s+from\b/,
    ],
  },
  {
    category: 'preferences',
    patterns: [
      /\b(prefer(?:s|red|ence)?|like(?:s|d)?|want(?:s|ed)?|style|convention)\b/i,
      /\b(always use|never use|default to)\b/i,
    ],
  },
  {
    category: 'commitments',
    patterns: [
      /\b(promised|committed|deadline|due|scheduled|will do|agreed to)\b/i,
      /\b(todo|task|action item|follow[- ]?up)\b/i,
    ],
  },
  {
    category: 'projects',
    patterns: [
      /\b(deployed|shipped|launched|released|merged|built|created)\b/i,
      /\b(project|repo|service|api|feature|bug fix)\b/i,
    ],
  },
];

const TYPE_TO_CATEGORY: Record<ObservationType, string> = {
  decision: 'decisions',
  preference: 'preferences',
  fact: 'facts',
  commitment: 'commitments',
  milestone: 'projects',
  lesson: 'lessons',
  relationship: 'people',
  project: 'projects'
};

export class Router {
  private readonly vaultPath: string;

  constructor(vaultPath: string) {
    this.vaultPath = path.resolve(vaultPath);
  }

  /**
   * Takes observation markdown and routes items to appropriate vault categories.
   * Routes only items with importance >= 0.4.
   * Returns a summary of what was routed where.
   */
  route(observationMarkdown: string): { routed: RoutedItem[]; summary: string } {
    const items = this.parseObservations(observationMarkdown);
    const routed: RoutedItem[] = [];

    for (const item of items) {
      if (item.importance < 0.4) continue;

      const category = this.categorize(item.type, item.content);
      if (!category) continue;

      const routedItem: RoutedItem = {
        category,
        title: item.title,
        content: item.content,
        type: item.type,
        confidence: item.confidence,
        importance: item.importance,
        date: item.date
      };
      routed.push(routedItem);
      this.appendToCategory(category, routedItem);
    }

    const summary = this.buildSummary(routed);
    return { routed, summary };
  }

  private parseObservations(markdown: string): Array<{
    type: ObservationType;
    confidence: number;
    importance: number;
    content: string;
    date: string;
    title: string;
  }> {
    const records = parseObservationMarkdown(markdown);
    return records.map((record) => ({
      type: record.type,
      confidence: record.confidence,
      importance: record.importance,
      content: record.content,
      date: record.date,
      title: record.content.slice(0, 80).replace(/[^a-zA-Z0-9\s-]/g, '').trim()
    }));
  }

  private categorize(type: ObservationType, content: string): string | null {
    const typedCategory = TYPE_TO_CATEGORY[type];
    if (typedCategory) {
      return typedCategory;
    }

    for (const { category, patterns } of CATEGORY_PATTERNS) {
      if (patterns.some((p) => p.test(content))) {
        return category;
      }
    }
    return null;
  }

  private normalizeForDedup(content: string): string {
    return normalizeObservationContent(
      content.replace(/\[\[[^\]]*\]\]/g, (match) => match.replace(/\[\[|\]\]/g, ''))
    );
  }

  /**
   * Extract entity slug from observation content for people/projects routing.
   * Returns null if no entity can be identified.
   */
  private extractEntitySlug(content: string, category: string): string | null {
    if (category !== 'people' && category !== 'projects') return null;

    if (category === 'people') {
      // Match patterns like "talked to Pedro", "met with Maria", "Justin said"
      // Note: name patterns are case-SENSITIVE to only match capitalized proper nouns
      const patterns = [
        /(?:talked to|met with|spoke with|chatted with|discussed with|emailed|called|messaged)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/,
        /([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:said|asked|told|mentioned|from|at)\b/,
        /\b(?:client|partner|colleague|contact)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/i,
      ];
      for (const pattern of patterns) {
        const match = content.match(pattern);
        if (match?.[1]) return this.toSlug(match[1]);
      }
    }

    if (category === 'projects') {
      // Match project-like names (capitalized, or in quotes)
      const patterns = [
        /(?:deployed|shipped|launched|released|built|created|working on)\s+([A-Z][a-zA-Z0-9-]+)/,
        /"([^"]+)"\s+(?:project|repo|service)/i,
      ];
      for (const pattern of patterns) {
        const match = content.match(pattern);
        if (match?.[1]) return this.toSlug(match[1]);
      }
    }

    return null;
  }

  private toSlug(name: string): string {
    return name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');
  }

  /**
   * Resolve the file path for a routed item.
   * For people/projects: entity-slug subfolder with date file (e.g., people/pedro/2026-02-12.md)
   * For other categories: category/date.md
   */
  private resolveFilePath(category: string, item: RoutedItem): string {
    const entitySlug = this.extractEntitySlug(item.content, category);
    if (entitySlug) {
      const entityDir = path.join(this.vaultPath, category, entitySlug);
      fs.mkdirSync(entityDir, { recursive: true });
      return path.join(entityDir, `${item.date}.md`);
    }
    const categoryDir = path.join(this.vaultPath, category);
    fs.mkdirSync(categoryDir, { recursive: true });
    return path.join(categoryDir, `${item.date}.md`);
  }

  private appendToCategory(category: string, item: RoutedItem): void {
    // Resolve file path (entity-aware for people/projects)
    const filePath = this.resolveFilePath(category, item);
    // Ensure parent dir exists (resolveFilePath handles entity dirs, but be safe)
    fs.mkdirSync(path.dirname(filePath), { recursive: true });

    const existing = fs.existsSync(filePath)
      ? fs.readFileSync(filePath, 'utf-8').trim()
      : '';

    // Normalized dedup: strip timestamps, wiki-links, whitespace, case
    const normalizedNew = this.normalizeForDedup(item.content);
    const existingLines = existing.split(/\r?\n/);
    for (const line of existingLines) {
      const lineContent = line.replace(/^-\s*/, '').trim();
      const parsed = parseObservationLine(lineContent, item.date);
      const candidate = parsed ? parsed.content : lineContent;
      if (this.normalizeForDedup(candidate) === normalizedNew) return;
    }

    // Also check similarity (>80% overlap = likely duplicate)
    for (const line of existingLines) {
      const lineContent = line.replace(/^-\s*/, '').trim();
      const parsed = parseObservationLine(lineContent, item.date);
      const normalizedExisting = this.normalizeForDedup(parsed ? parsed.content : lineContent);
      if (normalizedExisting.length > 10 && normalizedNew.length > 10) {
        const shorter = normalizedNew.length < normalizedExisting.length ? normalizedNew : normalizedExisting;
        const longer = normalizedNew.length >= normalizedExisting.length ? normalizedNew : normalizedExisting;
        if (longer.includes(shorter) || this.similarity(normalizedNew, normalizedExisting) > 0.8) return;
      }
    }

    const linkedContent = this.addWikiLinks(item.content);
    const entry = renderScoredObservationLine({
      type: item.type,
      confidence: item.confidence,
      importance: item.importance,
      content: linkedContent
    });
    const entitySlug = this.extractEntitySlug(item.content, category);
    const headerLabel = entitySlug ? `${category}/${entitySlug}` : category;
    const header = existing ? '' : `# ${headerLabel} — ${item.date}\n`;
    const newContent = existing
      ? `${existing}\n${entry}\n`
      : `${header}\n${entry}\n`;

    fs.writeFileSync(filePath, newContent, 'utf-8');
  }

  /**
   * Auto-link proper nouns and known entities with [[wiki-links]].
   * Scans for capitalized names, project names, and tool names.
   * Skips content already inside [[brackets]].
   */
  private addWikiLinks(content: string): string {
    // Don't double-link
    if (content.includes('[[')) return content;

    // Match capitalized proper nouns (2+ chars, not at start of sentence after emoji/time)
    // Pattern: standalone capitalized word that looks like a name/entity
    const namePattern = /\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b/g;

    // Words to skip (common English words that happen to appear capitalized)
    const skipWords = new Set([
      'The', 'This', 'That', 'These', 'Those', 'There', 'Then', 'Than',
      'When', 'Where', 'What', 'Which', 'While', 'With', 'Would', 'Will',
      'Should', 'Could', 'About', 'After', 'Before', 'Between', 'Because',
      'Also', 'Always', 'Already', 'Another', 'Any', 'Each', 'Every',
      'From', 'Have', 'Has', 'Had', 'Into', 'Just', 'Keep', 'Like',
      'Made', 'Make', 'Many', 'More', 'Most', 'Much', 'Must', 'Need',
      'Never', 'Next', 'None', 'Not', 'Now', 'Only', 'Other', 'Over',
      'Same', 'Some', 'Such', 'Sure', 'Take', 'Them', 'They', 'Too',
      'Under', 'Until', 'Upon', 'Very', 'Want', 'Were', 'Work', 'Yet',
      'Decision', 'Error', 'Deadline', 'Friday', 'Monday', 'Tuesday',
      'Wednesday', 'Thursday', 'Saturday', 'Sunday', 'January', 'February',
      'March', 'April', 'May', 'June', 'July', 'August', 'September',
      'October', 'November', 'December', 'Today', 'Tomorrow', 'Yesterday',
      'Message', 'Feature', 'Session', 'Update', 'System', 'User',
      'Processed', 'Working', 'Built', 'Deployed', 'Discussed', 'Talked',
      'Mentioned', 'Requested', 'Asked', 'Said',
    ]);

    // Known tool/project names to always link (lowercase for matching)
    const knownEntities = new Set([
      'PostgreSQL', 'MongoDB', 'Railway', 'Vercel', 'React', 'Vue', 'Svelte',
      'Express', 'NestJS', 'Prisma', 'Docker', 'Kubernetes', 'Redis',
      'GraphQL', 'Stripe', 'ClawVault', 'OpenClaw', 'GitHub', 'Obsidian',
    ]);

    return content.replace(namePattern, (match) => {
      if (skipWords.has(match)) return match;
      if (knownEntities.has(match)) return `[[${match}]]`;
      // Link proper nouns (likely people/orgs)
      if (/^[A-Z][a-z]+$/.test(match) && match.length >= 3) {
        return `[[${match}]]`;
      }
      // Link multi-word proper nouns (e.g., "Justin Dukes")
      if (/^[A-Z][a-z]+ [A-Z][a-z]+$/.test(match)) {
        return `[[${match}]]`;
      }
      return match;
    });
  }

  /**
   * Jaccard similarity on word bigrams — cheap approximation.
   */
  private similarity(a: string, b: string): number {
    const bigrams = (s: string): Set<string> => {
      const words = s.split(' ');
      const bg = new Set<string>();
      for (let i = 0; i < words.length - 1; i++) bg.add(`${words[i]} ${words[i + 1]}`);
      return bg;
    };
    const setA = bigrams(a);
    const setB = bigrams(b);
    if (setA.size === 0 || setB.size === 0) return 0;
    let intersection = 0;
    for (const bg of setA) if (setB.has(bg)) intersection++;
    return intersection / (setA.size + setB.size - intersection);
  }

  private buildSummary(routed: RoutedItem[]): string {
    if (routed.length === 0) return 'No items routed to vault categories.';

    const byCat = new Map<string, number>();
    for (const item of routed) {
      byCat.set(item.category, (byCat.get(item.category) ?? 0) + 1);
    }

    const parts = [...byCat.entries()].map(([cat, count]) => `${cat}: ${count}`);
    return `Routed ${routed.length} observations → ${parts.join(', ')}`;
  }
}
