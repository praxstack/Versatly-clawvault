import {
  normalizeObservationContent,
  parseObservationMarkdown,
  type ParsedObservationRecord
} from '../../lib/observation-format.js';
import type {
  FixtureScoringMetrics,
  FixtureScoringResult,
  KeywordRewriteFlag,
  ObservationMatch,
  ScoreFixtureOptions
} from './types.js';

const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from', 'had', 'has', 'have',
  'he', 'her', 'his', 'i', 'in', 'is', 'it', 'its', 'me', 'my', 'of', 'on', 'or', 'our', 'she',
  'that', 'the', 'their', 'them', 'they', 'this', 'to', 'was', 'we', 'were', 'will', 'with', 'you',
  'your', 'about', 'after', 'before', 'can', 'did', 'do', 'done', 'if', 'into', 'just', 'not', 'now',
  'out', 'over', 'should', 'than', 'then', 'there', 'these', 'those', 'today', 'tomorrow', 'update',
  'updated', 'working', 'session', 'assistant', 'user'
]);

const WORD_TOKEN_RE = /[a-z0-9]+(?:[._/-][a-z0-9]+)*/g;
const TERM_TOKEN_RE = /[A-Za-z0-9][A-Za-z0-9._/-]{2,}/g;

function clamp01(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

function round(value: number): number {
  return Math.round(clamp01(value) * 10000) / 10000;
}

function splitTokens(text: string): string[] {
  const lowered = normalizeObservationContent(text);
  const matches = lowered.match(WORD_TOKEN_RE) ?? [];
  return matches.filter((token) => token.length >= 2 && !STOPWORDS.has(token));
}

function toTokenSet(text: string): Set<string> {
  return new Set(splitTokens(text));
}

function overlapScore(expectedTokens: Set<string>, actualTokens: Set<string>): number {
  if (expectedTokens.size === 0 || actualTokens.size === 0) {
    return 0;
  }
  let overlap = 0;
  for (const token of expectedTokens) {
    if (actualTokens.has(token)) {
      overlap += 1;
    }
  }
  return overlap / expectedTokens.size;
}

function containsTerm(haystack: string, term: string): boolean {
  const normalizedHaystack = haystack.toLowerCase();
  const normalizedTerm = term.toLowerCase();
  if (!normalizedTerm) {
    return false;
  }
  if (/[^a-z0-9]/.test(normalizedTerm)) {
    return normalizedHaystack.includes(normalizedTerm);
  }
  const escaped = normalizedTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const regex = new RegExp(`\\b${escaped}\\b`, 'i');
  return regex.test(normalizedHaystack);
}

function extractSearchTerms(text: string, minimumLength = 6): string[] {
  const matches = text.match(TERM_TOKEN_RE) ?? [];
  const seen = new Set<string>();
  const terms: string[] = [];

  for (const raw of matches) {
    const normalized = raw.toLowerCase();
    if (normalized.length < minimumLength) {
      continue;
    }
    if (STOPWORDS.has(normalized)) {
      continue;
    }

    const hasSignal = /[A-Z]/.test(raw) || /\d/.test(raw) || /[._/-]/.test(raw) || normalized.length >= 8;
    if (!hasSignal) {
      continue;
    }

    if (!seen.has(normalized)) {
      seen.add(normalized);
      terms.push(raw);
    }
  }

  return terms;
}

export function extractKeywordSetFromTranscript(transcript: string, hints: string[] = []): string[] {
  const counts = new Map<string, { raw: string; count: number }>();

  for (const term of extractSearchTerms(transcript, 4)) {
    const key = term.toLowerCase();
    const current = counts.get(key) ?? { raw: term, count: 0 };
    current.count += 1;
    counts.set(key, current);
  }

  for (const hint of hints) {
    const trimmed = hint.trim();
    if (!trimmed) {
      continue;
    }
    const key = trimmed.toLowerCase();
    const current = counts.get(key) ?? { raw: trimmed, count: 0 };
    current.count += 2;
    counts.set(key, current);
  }

  const ranked = [...counts.values()]
    .sort((left, right) => right.count - left.count || right.raw.length - left.raw.length || left.raw.localeCompare(right.raw))
    .map((entry) => entry.raw);

  if (ranked.length > 0) {
    return ranked.slice(0, 25);
  }

  // Fallback: pull meaningful plain words if no high-signal terms were found.
  const fallback = [...new Set(splitTokens(transcript))]
    .filter((token) => token.length >= 5)
    .slice(0, 12);
  return fallback;
}

export function matchObservationRecords(
  expected: ParsedObservationRecord[],
  actual: ParsedObservationRecord[],
  threshold = 0.35
): ObservationMatch[] {
  const expectedTokens = expected.map((record) => toTokenSet(record.content));
  const actualTokens = actual.map((record) => toTokenSet(record.content));

  const expectedOrder = expected
    .map((record, index) => ({ index, importance: record.importance }))
    .sort((left, right) => right.importance - left.importance)
    .map((entry) => entry.index);

  const usedActual = new Set<number>();
  const matches: ObservationMatch[] = [];

  for (const expectedIndex of expectedOrder) {
    let bestIndex = -1;
    let bestScore = 0;
    for (let actualIndex = 0; actualIndex < actual.length; actualIndex += 1) {
      if (usedActual.has(actualIndex)) {
        continue;
      }
      const score = overlapScore(expectedTokens[expectedIndex] ?? new Set<string>(), actualTokens[actualIndex] ?? new Set<string>());
      if (score > bestScore) {
        bestScore = score;
        bestIndex = actualIndex;
      }
    }

    if (bestIndex >= 0 && bestScore >= threshold) {
      usedActual.add(bestIndex);
      matches.push({
        expectedIndex,
        actualIndex: bestIndex,
        similarity: round(bestScore)
      });
    }
  }

  return matches.sort((left, right) => left.expectedIndex - right.expectedIndex);
}

function computeMetrics(
  expected: ParsedObservationRecord[],
  actual: ParsedObservationRecord[],
  matches: ObservationMatch[],
  keywordSet: string[],
  minimumImportanceForRecall: number
): {
  metrics: FixtureScoringMetrics;
  missedImportant: ParsedObservationRecord[];
  noiseObservations: ParsedObservationRecord[];
  missingKeywords: string[];
  keywordRewriteFlags: KeywordRewriteFlag[];
} {
  const matchedExpected = new Set(matches.map((match) => match.expectedIndex));
  const matchedActual = new Set(matches.map((match) => match.actualIndex));

  const precision = actual.length === 0
    ? (expected.length === 0 ? 1 : 0)
    : matchedActual.size / actual.length;
  const noiseRatio = actual.length === 0 ? 0 : (actual.length - matchedActual.size) / actual.length;

  const importantExpected = expected.filter((record) => record.importance >= minimumImportanceForRecall);
  const totalImportantWeight = importantExpected.reduce((sum, record) => sum + record.importance, 0);
  const matchedImportantWeight = matches.reduce((sum, match) => {
    const record = expected[match.expectedIndex];
    if (!record || record.importance < minimumImportanceForRecall) {
      return sum;
    }
    return sum + record.importance;
  }, 0);
  const recall = totalImportantWeight === 0 ? 1 : matchedImportantWeight / totalImportantWeight;

  let typeNumerator = 0;
  let typeDenominator = 0;
  for (const match of matches) {
    const expectedRecord = expected[match.expectedIndex];
    const actualRecord = actual[match.actualIndex];
    if (!expectedRecord || !actualRecord) {
      continue;
    }
    const weight = Math.max(0.2, expectedRecord.importance);
    typeDenominator += weight;
    if (expectedRecord.type === actualRecord.type) {
      typeNumerator += weight;
    }
  }
  const typeAccuracy = typeDenominator === 0
    ? (expected.length === 0 && actual.length === 0 ? 1 : 0)
    : typeNumerator / typeDenominator;

  const actualCombined = actual.map((record) => record.content).join('\n');
  const missingKeywords = keywordSet.filter((keyword) => !containsTerm(actualCombined, keyword));
  const keywordPreservation = keywordSet.length === 0
    ? 1
    : (keywordSet.length - missingKeywords.length) / keywordSet.length;

  const keywordRewriteFlags: KeywordRewriteFlag[] = [];
  for (const match of matches) {
    const expectedRecord = expected[match.expectedIndex];
    const actualRecord = actual[match.actualIndex];
    if (!expectedRecord || !actualRecord) {
      continue;
    }
    const expectedTerms = extractSearchTerms(expectedRecord.content, 5);
    if (expectedTerms.length === 0) {
      continue;
    }
    const missing = expectedTerms.filter((term) => !containsTerm(actualRecord.content, term));
    if (missing.length > 0) {
      keywordRewriteFlags.push({
        expected: expectedRecord.content,
        actual: actualRecord.content,
        missingTerms: missing
      });
    }
  }

  const overall = (
    precision * 0.30 +
    recall * 0.35 +
    keywordPreservation * 0.20 +
    typeAccuracy * 0.15
  );

  return {
    metrics: {
      precision: round(precision),
      noiseRatio: round(noiseRatio),
      recall: round(recall),
      typeAccuracy: round(typeAccuracy),
      keywordPreservation: round(keywordPreservation),
      overall: round(overall)
    },
    missedImportant: expected.filter((record, index) => (
      record.importance >= minimumImportanceForRecall && !matchedExpected.has(index)
    )),
    noiseObservations: actual.filter((_, index) => !matchedActual.has(index)),
    missingKeywords,
    keywordRewriteFlags
  };
}

export function scoreFixtureObservations(input: {
  transcript: string;
  expectedMarkdown: string;
  actualMarkdown: string;
  options?: ScoreFixtureOptions;
}): FixtureScoringResult {
  const options = input.options ?? {};
  const minimumImportanceForRecall = clamp01(options.minimumImportanceForRecall ?? 0.5);
  const threshold = clamp01(options.matchThreshold ?? 0.35);

  const expected = parseObservationMarkdown(input.expectedMarkdown);
  const actual = parseObservationMarkdown(input.actualMarkdown);
  const matches = matchObservationRecords(expected, actual, threshold);
  const keywordSet = extractKeywordSetFromTranscript(input.transcript, options.keywordHints ?? []);
  const computed = computeMetrics(expected, actual, matches, keywordSet, minimumImportanceForRecall);

  return {
    metrics: computed.metrics,
    matches,
    missedImportant: computed.missedImportant,
    noiseObservations: computed.noiseObservations,
    keywordSet,
    missingKeywords: computed.missingKeywords,
    keywordRewriteFlags: computed.keywordRewriteFlags
  };
}

export function compareObservationText(left: string, right: string): boolean {
  return normalizeObservationContent(left) === normalizeObservationContent(right);
}
