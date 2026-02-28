import { describe, expect, it } from 'vitest';
import { formatAge } from './time.js';

describe('time utilities', () => {
  it('returns unknown for non-finite values', () => {
    expect(formatAge(Number.NaN)).toBe('unknown');
    expect(formatAge(Number.POSITIVE_INFINITY)).toBe('unknown');
  });

  it('clamps negative durations to zero seconds', () => {
    expect(formatAge(-5000)).toBe('0s');
  });

  it('formats seconds', () => {
    expect(formatAge(42_000)).toBe('42s');
  });

  it('formats minutes and seconds', () => {
    expect(formatAge(125_000)).toBe('2m 5s');
  });

  it('formats hours/minutes and days/hours', () => {
    expect(formatAge((3 * 60 * 60 + 15 * 60) * 1000)).toBe('3h 15m');
    expect(formatAge((2 * 24 * 60 * 60 + 5 * 60 * 60) * 1000)).toBe('2d 5h');
  });
});
