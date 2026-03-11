import type { CompressionProvider } from '../compressor.js';
import type { ObservationType, ParsedObservationRecord } from '../../lib/observation-format.js';

export type ObserverBenchmarkProvider = CompressionProvider | 'mock';
export type ObserverBenchmarkReportFormat = 'json' | 'text';

export interface ObserverBenchmarkFixtureConfig {
  now?: string;
  existingObservations?: string;
  compression?: {
    provider?: ObserverBenchmarkProvider;
    model?: string;
    baseUrl?: string;
    apiKey?: string;
  };
  scoring?: {
    matchThreshold?: number;
    minimumImportanceForRecall?: number;
    keywordHints?: string[];
  };
}

export interface ObserverBenchmarkFixture {
  id: string;
  dir: string;
  transcriptPath: string;
  expectedPath: string;
  transcript: string;
  expected: string;
  config: ObserverBenchmarkFixtureConfig;
}

export interface ObservationMatch {
  expectedIndex: number;
  actualIndex: number;
  similarity: number;
}

export interface KeywordRewriteFlag {
  expected: string;
  actual: string;
  missingTerms: string[];
}

export interface FixtureScoringMetrics {
  precision: number;
  noiseRatio: number;
  recall: number;
  typeAccuracy: number;
  keywordPreservation: number;
  overall: number;
}

export interface FixtureScoringResult {
  metrics: FixtureScoringMetrics;
  matches: ObservationMatch[];
  missedImportant: ParsedObservationRecord[];
  noiseObservations: ParsedObservationRecord[];
  keywordSet: string[];
  missingKeywords: string[];
  keywordRewriteFlags: KeywordRewriteFlag[];
}

export interface BenchmarkFixtureResult {
  fixtureId: string;
  transcriptPath: string;
  expectedPath: string;
  provider: ObserverBenchmarkProvider;
  model?: string;
  outputObservations: string;
  expectedCount: number;
  actualCount: number;
  expectedTypes: ObservationType[];
  actualTypes: ObservationType[];
  scoring: FixtureScoringResult;
}

export interface ObserverBenchmarkReport {
  generatedAt: string;
  fixturesDir: string;
  provider: ObserverBenchmarkProvider;
  model?: string;
  fixtureCount: number;
  aggregate: FixtureScoringMetrics;
  fixtures: BenchmarkFixtureResult[];
}

export interface ObserverBenchmarkRunOptions {
  fixturesDir: string;
  provider?: ObserverBenchmarkProvider;
  model?: string;
}

export interface ScoreFixtureOptions {
  minimumImportanceForRecall?: number;
  matchThreshold?: number;
  keywordHints?: string[];
}
