import { afterEach, describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { FactStore } from './fact-store.js';
import { ExtractedFact, factId } from './fact-extractor.js';

function makeTempVault(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'clawvault-factstore-'));
  fs.mkdirSync(path.join(dir, '.clawvault'), { recursive: true });
  return dir;
}

const tempDirs: string[] = [];
afterEach(() => {
  while (tempDirs.length) {
    const dir = tempDirs.pop();
    if (dir) fs.rmSync(dir, { recursive: true, force: true });
  }
});

function makeFact(overrides: Partial<ExtractedFact> = {}): ExtractedFact {
  const entity = overrides.entity || 'user';
  const relation = overrides.relation || 'prefers';
  const value = overrides.value || 'pizza';
  return {
    id: factId(entity, relation, value),
    entity,
    entityNorm: entity.toLowerCase(),
    relation,
    value,
    validFrom: '2026-02-20T00:00:00Z',
    validUntil: null,
    confidence: 0.8,
    category: 'preference',
    source: 'test.md',
    rawText: `${entity} ${relation} ${value}`,
    ...overrides
  };
}

describe('FactStore', () => {
  it('stores and retrieves facts', () => {
    const vault = makeTempVault();
    tempDirs.push(vault);
    const store = new FactStore(vault);
    store.load();

    const fact = makeFact();
    store.addFacts([fact]);

    const results = store.getEntityFacts('user');
    expect(results.length).toBe(1);
    expect(results[0].value).toBe('pizza');
  });

  it('persists to disk and reloads', () => {
    const vault = makeTempVault();
    tempDirs.push(vault);

    // Write
    const store1 = new FactStore(vault);
    store1.load();
    store1.addFacts([makeFact(), makeFact({ relation: 'lives_in', value: 'NYC', category: 'fact' })]);
    store1.save();

    // Read
    const store2 = new FactStore(vault);
    store2.load();
    expect(store2.getAllFacts().length).toBe(2);
    expect(store2.getEntityFacts('user').length).toBe(2);
  });

  it('resolves conflicts by superseding old facts', () => {
    const vault = makeTempVault();
    tempDirs.push(vault);
    const store = new FactStore(vault);
    store.load();

    // Old fact
    store.addFacts([makeFact({ relation: 'lives_in', value: 'NYC', category: 'fact', validFrom: '2026-01-01T00:00:00Z' })]);

    // New conflicting fact (moved)
    const conflicts = store.addFacts([makeFact({ relation: 'lives_in', value: 'London', category: 'fact', validFrom: '2026-02-01T00:00:00Z' })]);

    expect(conflicts).toBe(1);

    // Only London should be active
    const active = store.getEntityFacts('user');
    const livingFacts = active.filter(f => f.relation === 'lives_in');
    expect(livingFacts.length).toBe(1);
    expect(livingFacts[0].value).toBe('London');

    // But total should be 2 (NYC superseded)
    const stats = store.stats();
    expect(stats.totalFacts).toBe(2);
    expect(stats.supersededFacts).toBe(1);
  });

  it('does not conflict non-exclusive relations', () => {
    const vault = makeTempVault();
    tempDirs.push(vault);
    const store = new FactStore(vault);
    store.load();

    store.addFacts([
      makeFact({ relation: 'bought', value: 'laptop', category: 'event' }),
      makeFact({ relation: 'bought', value: 'camera', category: 'event' })
    ]);

    const facts = store.getEntityFacts('user');
    expect(facts.length).toBe(2);
  });

  it('queries by category', () => {
    const vault = makeTempVault();
    tempDirs.push(vault);
    const store = new FactStore(vault);
    store.load();

    store.addFacts([
      makeFact({ category: 'preference' }),
      makeFact({ relation: 'works_at', value: 'Google', category: 'fact' }),
      makeFact({ relation: 'decided', value: 'use TypeScript', category: 'decision' })
    ]);

    expect(store.getPreferences().length).toBe(1);
    expect(store.getCategoryFacts('fact').length).toBe(1);
    expect(store.getCategoryFacts('decision').length).toBe(1);
  });

  it('searches by keyword', () => {
    const vault = makeTempVault();
    tempDirs.push(vault);
    const store = new FactStore(vault);
    store.load();

    store.addFacts([
      makeFact({ value: 'Italian food' }),
      makeFact({ relation: 'works_at', value: 'Google', category: 'fact' }),
      makeFact({ relation: 'lives_in', value: 'New York City', category: 'fact' })
    ]);

    const results = store.searchFacts('Italian');
    expect(results.length).toBe(1);
    expect(results[0].value).toBe('Italian food');
  });

  it('queries facts at a point in time', () => {
    const vault = makeTempVault();
    tempDirs.push(vault);
    const store = new FactStore(vault);
    store.load();

    store.addFacts([makeFact({
      relation: 'lives_in', value: 'NYC', category: 'fact',
      validFrom: '2025-01-01T00:00:00Z'
    })]);
    store.addFacts([makeFact({
      relation: 'lives_in', value: 'London', category: 'fact',
      validFrom: '2026-01-01T00:00:00Z'
    })]);

    // In 2025, should see NYC
    const in2025 = store.getFactsAt('2025-06-01T00:00:00Z');
    const living2025 = in2025.filter(f => f.relation === 'lives_in');
    expect(living2025.length).toBe(1);
    expect(living2025[0].value).toBe('NYC');

    // In 2026, should see London
    const in2026 = store.getFactsAt('2026-06-01T00:00:00Z');
    const living2026 = in2026.filter(f => f.relation === 'lives_in');
    expect(living2026.length).toBe(1);
    expect(living2026[0].value).toBe('London');
  });

  it('returns correct stats', () => {
    const vault = makeTempVault();
    tempDirs.push(vault);
    const store = new FactStore(vault);
    store.load();

    store.addFacts([
      makeFact(),
      makeFact({ entity: 'John', relation: 'works_at', value: 'Google', category: 'fact' })
    ]);

    const stats = store.stats();
    expect(stats.totalFacts).toBe(2);
    expect(stats.activeFacts).toBe(2);
    expect(stats.supersededFacts).toBe(0);
    expect(stats.entities).toBe(2);
  });
});
