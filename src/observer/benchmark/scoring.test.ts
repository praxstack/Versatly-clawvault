import { describe, expect, it } from 'vitest';
import {
  compareObservationText,
  extractKeywordSetFromTranscript,
  matchObservationRecords,
  scoreFixtureObservations
} from './scoring.js';
import { parseObservationMarkdown } from '../../lib/observation-format.js';

const transcript = [
  'user: We decided to migrate auth to PostgreSQL and Redis-backed queues.',
  'assistant: TODO: ship rollback runbook before Friday for Project Artemis.',
  'user: Meeting with Acme client confirms launch date.'
].join('\n');

const expected = [
  '## 2026-03-10',
  '',
  '- [decision|c=0.95|i=0.92] 10:00 Decided to migrate auth to PostgreSQL and Redis-backed queues',
  '- [todo|c=0.90|i=0.80] 10:10 TODO: ship rollback runbook before Friday for Project Artemis',
  '- [commitment|c=0.84|i=0.74] 10:15 Confirmed launch date with Acme client'
].join('\n');

describe('observer benchmark scoring', () => {
  it('matches expected and actual observations by token overlap', () => {
    const actual = [
      '## 2026-03-10',
      '',
      '- [decision|c=0.90|i=0.88] 10:03 Migrated auth to PostgreSQL with Redis queues',
      '- [todo|c=0.88|i=0.70] 10:11 TODO: ship rollback runbook before Friday for Project Artemis',
      '- [fact|c=0.70|i=0.22] 10:20 Added unrelated status chatter'
    ].join('\n');

    const expectedRecords = parseObservationMarkdown(expected);
    const actualRecords = parseObservationMarkdown(actual);
    const matches = matchObservationRecords(expectedRecords, actualRecords, 0.35);

    expect(matches).toHaveLength(2);
    expect(matches[0]?.expectedIndex).toBe(0);
    expect(matches[1]?.expectedIndex).toBe(1);
  });

  it('produces meaningful differentiation between strong and weak output', () => {
    const strongActual = [
      '## 2026-03-10',
      '',
      '- [decision|c=0.93|i=0.90] 10:01 Decided to migrate auth to PostgreSQL and Redis-backed queues',
      '- [todo|c=0.89|i=0.78] 10:12 TODO: ship rollback runbook before Friday for Project Artemis',
      '- [commitment|c=0.86|i=0.72] 10:16 Confirmed launch date with Acme client'
    ].join('\n');

    const weakActual = [
      '## 2026-03-10',
      '',
      '- [fact|c=0.62|i=0.20] 10:01 Team discussed architecture updates',
      '- [fact|c=0.61|i=0.21] 10:05 Session continued with general planning notes'
    ].join('\n');

    const strong = scoreFixtureObservations({
      transcript,
      expectedMarkdown: expected,
      actualMarkdown: strongActual
    });
    const weak = scoreFixtureObservations({
      transcript,
      expectedMarkdown: expected,
      actualMarkdown: weakActual
    });

    expect(strong.metrics.overall).toBeGreaterThan(weak.metrics.overall);
    expect(strong.metrics.recall).toBeGreaterThan(weak.metrics.recall);
    expect(strong.metrics.keywordPreservation).toBeGreaterThan(weak.metrics.keywordPreservation);
    expect(weak.missedImportant.length).toBeGreaterThan(0);
    expect(weak.noiseObservations.length).toBeGreaterThan(0);
  });

  it('extracts searchable keyword anchors from transcript text', () => {
    const keywords = extractKeywordSetFromTranscript(transcript, ['Project Artemis']);
    expect(keywords.some((term) => term.toLowerCase().includes('postgresql'))).toBe(true);
    expect(keywords.some((term) => term.toLowerCase().includes('redis'))).toBe(true);
    expect(keywords.some((term) => term.toLowerCase().includes('project artemis'))).toBe(true);
  });

  it('normalizes observation text when comparing equivalence', () => {
    expect(compareObservationText('10:00   Ship release', 'Ship release')).toBe(true);
    expect(compareObservationText('Ship release', 'Ship another release')).toBe(false);
  });
});
