export interface CompressorOptions {
  model?: string;
  now?: () => Date;
  fetchImpl?: typeof fetch;
}

type Priority = '🔴' | '🟡' | '🟢';

interface ObservationLine {
  priority: Priority;
  content: string;
}

const DATE_HEADING_RE = /^##\s+(\d{4}-\d{2}-\d{2})\s*$/;
const OBSERVATION_LINE_RE = /^(🔴|🟡|🟢)\s+(.+)$/u;
const CRITICAL_RE =
  /(?:\b(?:decision|decided|chose|chosen|selected|picked|opted|switched to)\s*:?|\bdecid(?:e|ed|ing|ion)\b|\berror\b|\bfail(?:ed|ure|ing)?\b|\bblock(?:ed|er)?\b|\bbreaking(?:\s+change)?s?\b|\bcritical\b|\b\w+\s+chosen\s+(?:for|over|as)\b|\bpublish(?:ed)?\b.*@?\d+\.\d+|\bmerge[d]?\s+(?:PR|pull\s+request)\b|\bshipped\b|\breleased?\b.*v?\d+\.\d+)/i;
const DEADLINE_WITH_DATE_RE = /(?:(?:\bdeadline\b|\bdue(?:\s+date)?\b|\bcutoff\b).*(?:\d{4}-\d{2}-\d{2}|\d{1,2}\/\d{1,2}(?:\/\d{2,4})?|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})|(?:\d{4}-\d{2}-\d{2}|\d{1,2}\/\d{1,2}(?:\/\d{2,4})?|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}).*(?:\bdeadline\b|\bdue(?:\s+date)?\b|\bcutoff\b))/i;
const NOTABLE_RE = /\b(prefer(?:ence|s)?|likes?|dislikes?|context|pattern|architecture|approach|trade[- ]?off|milestone|stakeholder|teammate|collaborat(?:e|ed|ion)|discussion|notable|deadline|due|timeline|deploy(?:ed|ment)?|built|configured|launched)\b/i;

export class Compressor {
  private readonly model?: string;
  private readonly now: () => Date;
  private readonly fetchImpl: typeof fetch;

  constructor(options: CompressorOptions = {}) {
    this.model = options.model;
    this.now = options.now ?? (() => new Date());
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async compress(messages: string[], existingObservations: string): Promise<string> {
    const cleanedMessages = messages.map((message) => message.trim()).filter(Boolean);
    if (cleanedMessages.length === 0) {
      return existingObservations.trim();
    }

    const prompt = this.buildPrompt(cleanedMessages, existingObservations);
    const provider = this.resolveProvider();

    if (provider) {
      try {
        const llmOutput = provider === 'anthropic'
          ? await this.callAnthropic(prompt)
          : provider === 'gemini'
          ? await this.callGemini(prompt)
          : await this.callOpenAI(prompt);
        const normalized = this.normalizeLlmOutput(llmOutput);
        if (normalized) {
          return this.mergeObservations(existingObservations, normalized);
        }
      } catch {
        // Fall back to deterministic extraction to keep observer flow stable.
      }
    }

    const fallback = this.fallbackCompression(cleanedMessages);
    return this.mergeObservations(existingObservations, fallback);
  }

  private resolveProvider(): 'anthropic' | 'openai' | 'gemini' | null {
    if (process.env.CLAWVAULT_NO_LLM) return null;
    if (process.env.ANTHROPIC_API_KEY) {
      return 'anthropic';
    }
    if (process.env.OPENAI_API_KEY) {
      return 'openai';
    }
    if (process.env.GEMINI_API_KEY) {
      return 'gemini';
    }
    return null;
  }

  private buildPrompt(messages: string[], existingObservations: string): string {
    return [
      'You are an observer that compresses raw AI session messages into durable, human-meaningful observations.',
      '',
      'Rules:',
      '- Output markdown only.',
      '- Group observations by date heading: ## YYYY-MM-DD',
      '- Each line must follow: <emoji> <HH:MM> <observation>',
      '- Priority emojis: 🔴 critical, 🟡 notable, 🟢 info',
      '- 🔴 for: decisions between alternatives, blockers, deadlines with explicit dates, breaking changes, commitments made to people, version releases/publishes, merged PRs, shipped features',
      '- 🟡 for: preferences, architecture discussions, trade-offs, milestones, people interactions, notable context, deployments to production, new tools/commands built, project configuration changes',
      '- 🟢 for: routine tasks, intermediate build steps, general progress, minor fixes',
      '- Preserve source tags when present (e.g., [main], [telegram-dm], [discord], [telegram-group]).',
      '',
      'QUALITY FILTERS (important):',
      '- DO NOT observe: CLI errors, command failures, tool output parsing issues, retry attempts, debug logs.',
      '  These are transient noise, not memories. Only observe errors if they represent a BLOCKER or an unresolved problem.',
      '- DO NOT observe: "acknowledged the conversation", "said okay", routine confirmations.',
      '- MERGE related events into single observations. If 5 images were generated, say "Generated 5 images for X" not 5 separate lines.',
      '- MERGE retry sequences: "Tried X, failed, tried Y, succeeded" → "Resolved X using Y (after initial failure)"',
      '- Prefer OUTCOMES over PROCESSES: "Deployed v1.2 to Railway" not "Started deploy... build finished... deploy succeeded"',
      '',
      'AGENT ATTRIBUTION:',
      '- If the transcript shows multiple speakers/agents, prefix observations with who did it: "Pedro asked...", "Clawdious deployed...", "Zeca generated..."',
      '- If only one agent is acting, attribution is optional.',
      '',
      'PROJECT MILESTONES (critical — these are the most valuable observations):',
      '- Version releases: "🔴 HH:MM Published project@version to npm/PyPI/etc" — ALWAYS 🔴',
      '- PR merges: "🔴 HH:MM Merged PR #N: <title> into <branch>" — ALWAYS 🔴',
      '- Feature completion: "🔴 HH:MM Shipped <feature> in <project>" — ALWAYS 🔴',
      '- Production deploys: "🟡 HH:MM Deployed <project> to <url>" — ALWAYS at least 🟡',
      '- New commands/APIs: "🟡 HH:MM Built <command/endpoint> for <project>"',
      '- Config/infra changes: "🟡 HH:MM Changed <what> in <project> (<why>)"',
      '- Do NOT collapse multiple releases into one line — each version matters for history.',
      '',
      'COMMITMENT FORMAT (when someone promises/agrees to something):',
      '- Use: "🔴 HH:MM [COMMITMENT] <who> committed to <what> by <when>" (include deadline if mentioned)',
      '',
      'Keep observations concise and factual. Aim for signal, not completeness.',
      '',
      'Existing observations (may be empty):',
      existingObservations.trim() || '(none)',
      '',
      'Raw messages:',
      ...messages.map((message, index) => `[${index + 1}] ${message}`),
      '',
      'Return only the updated observation markdown.'
    ].join('\n');
  }

  private async callAnthropic(prompt: string): Promise<string> {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return '';
    }

    const response = await this.fetchImpl('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: this.model ?? 'claude-3-5-haiku-latest',
        temperature: 0.1,
        max_tokens: 1400,
        messages: [{ role: 'user', content: prompt }]
      })
    });

    if (!response.ok) {
      throw new Error(`Anthropic request failed (${response.status})`);
    }

    const payload = await response.json() as {
      content?: Array<{ type: string; text?: string }>;
    };

    return payload.content
      ?.filter((part) => part.type === 'text' && part.text)
      .map((part) => part.text as string)
      .join('\n')
      .trim() ?? '';
  }

  private async callOpenAI(prompt: string): Promise<string> {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      return '';
    }

    const response = await this.fetchImpl('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: this.model ?? 'gpt-4o-mini',
        temperature: 0.1,
        messages: [
          { role: 'system', content: 'You transform session logs into concise observations.' },
          { role: 'user', content: prompt }
        ]
      })
    });

    if (!response.ok) {
      throw new Error(`OpenAI request failed (${response.status})`);
    }

    const payload = await response.json() as {
      choices?: Array<{ message?: { content?: string } }>;
    };
    return payload.choices?.[0]?.message?.content?.trim() ?? '';
  }

  private async callGemini(prompt: string): Promise<string> {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return '';
    }

    const model = this.model ?? 'gemini-2.0-flash';
    const response = await this.fetchImpl(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: { temperature: 0.1, maxOutputTokens: 1400 }
        })
      }
    );

    if (!response.ok) {
      throw new Error(`Gemini request failed (${response.status})`);
    }

    const payload = await response.json() as {
      candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }>;
    };
    return payload.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? '';
  }

  private normalizeLlmOutput(output: string): string {
    if (!output.trim()) {
      return '';
    }

    const cleaned = output
      .replace(/^```(?:markdown)?\s*/i, '')
      .replace(/\s*```$/, '')
      .trim();

    const lines = cleaned.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    const hasObservationLine = lines.some((line) => OBSERVATION_LINE_RE.test(line));
    if (!hasObservationLine) {
      return '';
    }

    const hasDateHeading = lines.some((line) => DATE_HEADING_RE.test(line));
    const result = hasDateHeading ? cleaned : `## ${this.formatDate(this.now())}\n\n${cleaned}`;

    // Post-process: fix wiki-link corruption, then enforce priority rules
    // LLMs often fuse preceding words into wiki-links: "reque[[people/pedro]]"
    const sanitized = this.sanitizeWikiLinks(result);
    return this.enforcePriorityRules(sanitized);
  }

  /**
   * Fix wiki-link corruption from LLM compression.
   * LLMs often fuse preceding word fragments into wiki-links during rewriting:
   *   "reque[[people/pedro]]" → "[[people/pedro]]"
   *   "Linke[[agents/zeca]]" → "[[agents/zeca]]"
   *   "taske[[people/pedro]]a" → "[[people/pedro]]"
   * Also fixes trailing word fragments fused after closing brackets.
   */
  private sanitizeWikiLinks(markdown: string): string {
    // Fix word fragments fused BEFORE opening [[ — e.g. "reque[[foo]]" → " [[foo]]"
    let result = markdown.replace(/\w+\[\[/g, ' [[');
    // Fix word fragments fused AFTER closing ]] — e.g. "[[foo]]sted" → "[[foo]]"
    result = result.replace(/\]\]\w+/g, ']]');
    // Clean up any double spaces introduced
    result = result.replace(/ {2,}/g, ' ');
    return result;
  }

  /**
   * Post-process LLM output to enforce priority rules.
   * Lines matching critical rules get upgraded to 🔴, notable rules to 🟡.
   */
  private enforcePriorityRules(markdown: string): string {
    return markdown.split(/\r?\n/).map((line) => {
      const match = line.match(OBSERVATION_LINE_RE);
      if (!match) return line;

      const currentPriority = match[1] as Priority;
      const content = match[2];

      if (this.isCriticalContent(content) && currentPriority !== '🔴') {
        return line.replace(/^🟡|^🟢/u, '🔴');
      }
      if (this.isNotableContent(content) && currentPriority === '🟢') {
        return line.replace(/^🟢/u, '🟡');
      }

      return line;
    }).join('\n');
  }

  private fallbackCompression(messages: string[]): string {
    const sections = new Map<string, ObservationLine[]>();
    const seen = new Set<string>();

    for (const message of messages) {
      const normalized = this.normalizeText(message);
      if (!normalized) continue;

      const date = this.extractDate(message) ?? this.formatDate(this.now());
      const time = this.extractTime(message) ?? this.formatTime(this.now());
      const priority = this.inferPriority(normalized);
      const line = `${time} ${normalized}`;
      const dedupeKey = `${date}|${priority}|${this.normalizeText(line)}`;
      if (seen.has(dedupeKey)) continue;
      seen.add(dedupeKey);

      const bucket = sections.get(date) ?? [];
      bucket.push({ priority, content: line });
      sections.set(date, bucket);
    }

    if (sections.size === 0) {
      const date = this.formatDate(this.now());
      sections.set(date, [{ priority: '🟢', content: `${this.formatTime(this.now())} Processed session updates.` }]);
    }

    return this.renderSections(sections);
  }

  private mergeObservations(existing: string, incoming: string): string {
    const existingSections = this.parseSections(existing);
    const incomingSections = this.parseSections(incoming);

    if (incomingSections.size === 0) {
      return existing.trim();
    }

    if (existingSections.size === 0) {
      return this.renderSections(incomingSections);
    }

    for (const [date, lines] of existingSections.entries()) {
      existingSections.set(date, this.deduplicateObservationLines(lines));
    }

    for (const [date, lines] of incomingSections.entries()) {
      const current = this.deduplicateObservationLines(existingSections.get(date) ?? []);
      const seen = new Set(current.map((line) => this.normalizeObservationContent(line.content)));
      for (const line of lines) {
        const normalized = this.normalizeObservationContent(line.content);
        if (!normalized || seen.has(normalized)) {
          continue;
        }
        seen.add(normalized);
        current.push(line);
      }
      existingSections.set(date, current);
    }

    return this.renderSections(existingSections);
  }

  private deduplicateObservationLines(lines: ObservationLine[]): ObservationLine[] {
    const deduped: ObservationLine[] = [];
    const seen = new Set<string>();
    for (const line of lines) {
      const normalized = this.normalizeObservationContent(line.content);
      if (!normalized || seen.has(normalized)) {
        continue;
      }
      seen.add(normalized);
      deduped.push(line);
    }
    return deduped;
  }

  private normalizeObservationContent(content: string): string {
    return content
      .replace(/^\d{2}:\d{2}\s+/, '')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();
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

      if (!currentDate) continue;
      const lineMatch = rawLine.match(OBSERVATION_LINE_RE);
      if (!lineMatch) continue;

      const bucket = sections.get(currentDate) ?? [];
      bucket.push({
        priority: lineMatch[1] as Priority,
        content: lineMatch[2].trim()
      });
      sections.set(currentDate, bucket);
    }

    return sections;
  }

  private renderSections(sections: Map<string, ObservationLine[]>): string {
    const chunks: string[] = [];
    const sortedDates = [...sections.keys()].sort((a, b) => a.localeCompare(b));

    for (const date of sortedDates) {
      const lines = sections.get(date) ?? [];
      if (lines.length === 0) continue;
      chunks.push(`## ${date}`);
      chunks.push('');
      for (const line of lines) {
        chunks.push(`${line.priority} ${line.content}`);
      }
      chunks.push('');
    }

    return chunks.join('\n').trim();
  }

  private inferPriority(text: string): Priority {
    if (this.isCriticalContent(text)) return '🔴';
    if (this.isNotableContent(text)) return '🟡';
    return '🟢';
  }

  private isCriticalContent(text: string): boolean {
    return CRITICAL_RE.test(text) || DEADLINE_WITH_DATE_RE.test(text);
  }

  private isNotableContent(text: string): boolean {
    return NOTABLE_RE.test(text);
  }

  private normalizeText(text: string): string {
    return text
      .replace(/\s+/g, ' ')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .trim()
      .slice(0, 280);
  }

  private extractDate(text: string): string | null {
    const match = text.match(/\b(\d{4}-\d{2}-\d{2})\b/);
    return match?.[1] ?? null;
  }

  private extractTime(text: string): string | null {
    const match = text.match(/\b([01]\d|2[0-3]):([0-5]\d)\b/);
    if (!match) {
      return null;
    }
    return `${match[1]}:${match[2]}`;
  }

  private formatDate(date: Date): string {
    return date.toISOString().split('T')[0];
  }

  private formatTime(date: Date): string {
    return date.toISOString().slice(11, 16);
  }
}
