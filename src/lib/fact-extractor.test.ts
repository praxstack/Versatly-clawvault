import { describe, expect, it } from 'vitest';
import {
  extractFactsRuleBased,
  normalizeEntity,
  factId
} from './fact-extractor.js';

describe('normalizeEntity', () => {
  it('lowercases and strips punctuation', () => {
    expect(normalizeEntity('Pedro Sobral')).toBe('pedro sobral');
    expect(normalizeEntity('Dr. Smith')).toBe('dr smith');
    expect(normalizeEntity("O'Brien")).toBe('obrien');
  });

  it('collapses whitespace', () => {
    expect(normalizeEntity('  John   Doe  ')).toBe('john doe');
  });
});

describe('factId', () => {
  it('produces deterministic IDs', () => {
    const id1 = factId('user', 'prefers', 'pizza');
    const id2 = factId('user', 'prefers', 'pizza');
    expect(id1).toBe(id2);
  });

  it('produces different IDs for different facts', () => {
    const id1 = factId('user', 'prefers', 'pizza');
    const id2 = factId('user', 'prefers', 'sushi');
    expect(id1).not.toBe(id2);
  });
});

describe('extractFactsRuleBased', () => {
  it('extracts preferences', () => {
    const facts = extractFactsRuleBased(
      'I prefer dark mode. I like Italian food. My favorite color is blue.',
      'test.md'
    );
    expect(facts.length).toBeGreaterThanOrEqual(3);
    expect(facts.some(f => f.relation === 'prefers' && f.value.includes('dark mode'))).toBe(true);
    expect(facts.some(f => f.relation === 'prefers' && f.value.includes('Italian food'))).toBe(true);
    expect(facts.some(f => f.relation === 'favorite_color' && f.value.includes('blue'))).toBe(true);
  });

  it('extracts dislikes', () => {
    const facts = extractFactsRuleBased("I don't like spicy food.", 'test.md');
    expect(facts.length).toBeGreaterThanOrEqual(1);
    expect(facts[0].relation).toBe('dislikes');
    expect(facts[0].value).toContain('spicy food');
  });

  it('extracts location facts', () => {
    const facts = extractFactsRuleBased('John lives in New York. Alice moved to London.', 'test.md');
    expect(facts.length).toBeGreaterThanOrEqual(2);
    expect(facts.some(f => f.entity === 'John' && f.relation === 'lives_in')).toBe(true);
    expect(facts.some(f => f.entity === 'Alice' && f.relation === 'lives_in')).toBe(true);
  });

  it('extracts work facts', () => {
    const facts = extractFactsRuleBased('Pedro works at Versatly.', 'test.md');
    expect(facts.length).toBeGreaterThanOrEqual(1);
    expect(facts[0].relation).toBe('works_at');
    expect(facts[0].value).toContain('Versatly');
  });

  it('extracts spending facts', () => {
    const facts = extractFactsRuleBased('I spent $500 on a new camera.', 'test.md');
    expect(facts.length).toBeGreaterThanOrEqual(1);
    expect(facts[0].relation).toBe('spent_on');
    expect(facts[0].value).toContain('$500');
  });

  it('extracts decisions', () => {
    const facts = extractFactsRuleBased('I decided to use TypeScript for the project.', 'test.md');
    expect(facts.length).toBeGreaterThanOrEqual(1);
    expect(facts[0].category).toBe('decision');
  });

  it('sets metadata correctly', () => {
    const facts = extractFactsRuleBased('I prefer tea.', 'memory/daily.md', '2026-02-23T00:00:00Z');
    expect(facts.length).toBe(1);
    expect(facts[0].source).toBe('memory/daily.md');
    expect(facts[0].validFrom).toBe('2026-02-23T00:00:00Z');
    expect(facts[0].validUntil).toBeNull();
    expect(facts[0].confidence).toBe(0.7);
    expect(facts[0].category).toBe('preference');
  });

  it('returns empty for irrelevant text', () => {
    const facts = extractFactsRuleBased('The weather is nice today.', 'test.md');
    expect(facts.length).toBe(0);
  });

  it('handles multi-sentence paragraphs', () => {
    const text = `Last week I bought a new laptop for $1200. 
    I prefer MacBook over Windows. 
    My colleague John works at Google.
    I decided to switch to Linux for development.`;
    const facts = extractFactsRuleBased(text, 'test.md');
    expect(facts.length).toBeGreaterThanOrEqual(3);
  });
});
