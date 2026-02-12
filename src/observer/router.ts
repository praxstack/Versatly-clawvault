import * as fs from 'fs';
import * as path from 'path';

/**
 * Routes observations into the appropriate vault category files.
 * Takes compressed observations and updates decisions/, people/, lessons/, etc.
 */

interface RoutedItem {
  category: string;
  title: string;
  content: string;
  priority: '🔴' | '🟡' | '🟢';
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

const OBSERVATION_LINE_RE = /^(🔴|🟡|🟢)\s+(\d{2}:\d{2})?\s*(.+)$/u;
const DATE_HEADING_RE = /^##\s+(\d{4}-\d{2}-\d{2})\s*$/;

export class Router {
  private readonly vaultPath: string;

  constructor(vaultPath: string) {
    this.vaultPath = path.resolve(vaultPath);
  }

  /**
   * Takes observation markdown and routes items to appropriate vault categories.
   * Only routes 🔴 and 🟡 items — 🟢 stays only in observations.
   * Returns a summary of what was routed where.
   */
  route(observationMarkdown: string): { routed: RoutedItem[]; summary: string } {
    const items = this.parseObservations(observationMarkdown);
    const routed: RoutedItem[] = [];

    for (const item of items) {
      // Only route critical and notable items
      if (item.priority === '🟢') continue;

      const category = this.categorize(item.content);
      if (!category) continue;

      const routedItem: RoutedItem = { category, title: item.title, content: item.content, priority: item.priority, date: item.date };
      routed.push(routedItem);
      this.appendToCategory(category, routedItem);
    }

    const summary = this.buildSummary(routed);
    return { routed, summary };
  }

  private parseObservations(markdown: string): Array<{ priority: '🔴' | '🟡' | '🟢'; content: string; date: string; title: string }> {
    const results: Array<{ priority: '🔴' | '🟡' | '🟢'; content: string; date: string; title: string }> = [];
    let currentDate = new Date().toISOString().split('T')[0];

    for (const line of markdown.split(/\r?\n/)) {
      const dateMatch = line.match(DATE_HEADING_RE);
      if (dateMatch) {
        currentDate = dateMatch[1];
        continue;
      }

      const obsMatch = line.match(OBSERVATION_LINE_RE);
      if (!obsMatch) continue;

      const priority = obsMatch[1] as '🔴' | '🟡' | '🟢';
      const content = obsMatch[3].trim();
      const title = content.slice(0, 80).replace(/[^a-zA-Z0-9\s-]/g, '').trim();

      results.push({ priority, content, date: currentDate, title });
    }

    return results;
  }

  private categorize(content: string): string | null {
    for (const { category, patterns } of CATEGORY_PATTERNS) {
      if (patterns.some((p) => p.test(content))) {
        return category;
      }
    }
    return null;
  }

  private appendToCategory(category: string, item: RoutedItem): void {
    const categoryDir = path.join(this.vaultPath, category);
    fs.mkdirSync(categoryDir, { recursive: true });

    // Append to the date-based category file
    const filePath = path.join(categoryDir, `${item.date}.md`);
    const existing = fs.existsSync(filePath)
      ? fs.readFileSync(filePath, 'utf-8').trim()
      : '';

    // Don't duplicate
    if (existing.includes(item.content)) return;

    const linkedContent = this.addWikiLinks(item.content);
    const entry = `- ${item.priority} ${linkedContent}`;
    const header = existing ? '' : `# ${category} — ${item.date}\n`;
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
