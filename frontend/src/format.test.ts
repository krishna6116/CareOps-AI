import { describe, expect, it } from 'vitest';

import { display, formatCount, formatPercent, formatSigned } from './format';

describe('formatCount', () => {
  it('shows counts as whole numbers', () => {
    expect(formatCount(338.4)).toBe('338');
    expect(formatCount(338.5)).toBe('339');
    expect(formatCount('144.0')).toBe('144');
    expect(formatCount(6718)).toBe('6,718');
  });

  it('never shows a negative count and reports missing values', () => {
    expect(formatCount(-2)).toBe('0');
    expect(formatCount(null)).toBe('Unavailable');
    expect(formatCount('n/a')).toBe('Unavailable');
    expect(formatCount(true)).toBe('Unavailable');
  });

  it('keeps decimals out of every count it formats', () => {
    for (const value of [0.49, 1.5, 12.0001, 99.99, 230.7]) {
      expect(formatCount(value)).not.toMatch(/[.]/);
    }
  });
});

describe('other formats', () => {
  it('keeps rates as percentages with a decimal', () => {
    expect(formatPercent(0.9873)).toBe('98.7%');
    expect(formatPercent(null)).toBe('Unavailable');
  });

  it('signs changes', () => {
    expect(formatSigned(23)).toBe('+23');
    expect(formatSigned(-4)).toBe('−4');
    expect(formatSigned(0)).toBe('0');
    expect(formatSigned(null)).toBe('—');
  });

  it('prints integral numbers without decimals and keeps metric precision', () => {
    expect(display(4174)).toBe('4174');
    expect(display(0.59971)).toBe('0.6');
    expect(display(2.6234)).toBe('2.623');
    expect(display(null)).toBe('—');
    expect(display(false)).toBe('no');
  });
});
