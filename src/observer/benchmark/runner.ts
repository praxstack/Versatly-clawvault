import { Compressor } from '../compressor.js';
import { parseSessionFile } from '../session-parser.js';
import { parseObservationMarkdown } from '../../lib/observation-format.js';
import { loadObserverBenchmarkFixtures } from './fixtures.js';
import { scoreFixtureObservations } from './scoring.js';
import type {
  BenchmarkFixtureResult,
  FixtureScoringMetrics,
  ObserverBenchmarkProvider,
  ObserverBenchmarkReport,
  ObserverBenchmarkRunOptions
} from './types.js';

const DEFAULT_PROVIDER: ObserverBenchmarkProvider = 'mock';
const DEFAULT_NOW = new Date('2026-03-10T12:00:00.000Z');

function parseFixtureNow(raw: string | undefined): Date {
  if (!raw) {
    return DEFAULT_NOW;
  }
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    throw new Error(`Invalid fixture config now timestamp: ${raw}`);
  }
  return parsed;
}

function parseTranscriptMessages(rawTranscript: string, transcriptPath: string): string[] {
  const lineMessages = rawTranscript
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lineMessages.length > 0) {
    return lineMessages;
  }
  return parseSessionFile(transcriptPath);
}

function withNoLlm<T>(enabled: boolean, task: () => Promise<T>): Promise<T> {
  if (!enabled) {
    return task();
  }
  const previous = process.env.CLAWVAULT_NO_LLM;
  process.env.CLAWVAULT_NO_LLM = '1';
  return task().finally(() => {
    if (previous === undefined) {
      delete process.env.CLAWVAULT_NO_LLM;
      return;
    }
    process.env.CLAWVAULT_NO_LLM = previous;
  });
}

function averageMetrics(rows: FixtureScoringMetrics[]): FixtureScoringMetrics {
  if (rows.length === 0) {
    return {
      precision: 0,
      noiseRatio: 0,
      recall: 0,
      typeAccuracy: 0,
      keywordPreservation: 0,
      overall: 0
    };
  }

  const totals = rows.reduce((acc, metrics) => {
    acc.precision += metrics.precision;
    acc.noiseRatio += metrics.noiseRatio;
    acc.recall += metrics.recall;
    acc.typeAccuracy += metrics.typeAccuracy;
    acc.keywordPreservation += metrics.keywordPreservation;
    acc.overall += metrics.overall;
    return acc;
  }, {
    precision: 0,
    noiseRatio: 0,
    recall: 0,
    typeAccuracy: 0,
    keywordPreservation: 0,
    overall: 0
  });

  const count = rows.length;
  const round = (value: number): number => Math.round((value / count) * 10000) / 10000;

  return {
    precision: round(totals.precision),
    noiseRatio: round(totals.noiseRatio),
    recall: round(totals.recall),
    typeAccuracy: round(totals.typeAccuracy),
    keywordPreservation: round(totals.keywordPreservation),
    overall: round(totals.overall)
  };
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

export async function runObserverBenchmark(options: ObserverBenchmarkRunOptions): Promise<ObserverBenchmarkReport> {
  const fixtures = loadObserverBenchmarkFixtures(options.fixturesDir);
  const fixtureResults: BenchmarkFixtureResult[] = [];

  for (const fixture of fixtures) {
    const fixtureProvider = fixture.config.compression?.provider ?? options.provider ?? DEFAULT_PROVIDER;
    const fixtureModel = fixture.config.compression?.model ?? options.model;
    const fixtureNow = parseFixtureNow(fixture.config.now);
    const messages = parseTranscriptMessages(fixture.transcript, fixture.transcriptPath);

    const compressor = new Compressor({
      provider: fixtureProvider === 'mock' ? undefined : fixtureProvider,
      model: fixtureModel,
      baseUrl: fixture.config.compression?.baseUrl,
      apiKey: fixture.config.compression?.apiKey,
      now: () => fixtureNow
    });

    const outputObservations = await withNoLlm(
      fixtureProvider === 'mock',
      async () => compressor.compress(messages, fixture.config.existingObservations ?? '')
    );

    const scoring = scoreFixtureObservations({
      transcript: fixture.transcript,
      expectedMarkdown: fixture.expected,
      actualMarkdown: outputObservations,
      options: {
        matchThreshold: fixture.config.scoring?.matchThreshold,
        minimumImportanceForRecall: fixture.config.scoring?.minimumImportanceForRecall,
        keywordHints: fixture.config.scoring?.keywordHints
      }
    });

    const expectedRecords = parseObservationMarkdown(fixture.expected);
    const actualRecords = parseObservationMarkdown(outputObservations);

    fixtureResults.push({
      fixtureId: fixture.id,
      transcriptPath: fixture.transcriptPath,
      expectedPath: fixture.expectedPath,
      provider: fixtureProvider,
      model: fixtureModel,
      outputObservations,
      expectedCount: expectedRecords.length,
      actualCount: actualRecords.length,
      expectedTypes: unique(expectedRecords.map((record) => record.type)),
      actualTypes: unique(actualRecords.map((record) => record.type)),
      scoring
    });
  }

  const reportProvider = options.provider ?? DEFAULT_PROVIDER;
  return {
    generatedAt: new Date().toISOString(),
    fixturesDir: options.fixturesDir,
    provider: reportProvider,
    model: options.model,
    fixtureCount: fixtureResults.length,
    aggregate: averageMetrics(fixtureResults.map((row) => row.scoring.metrics)),
    fixtures: fixtureResults
  };
}
