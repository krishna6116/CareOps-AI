// Presentation rules. Counts (beds, admissions, staff, requests) are shown as
// whole numbers, rounded half away from zero and never negative; the backend
// keeps full precision for training, optimization and evaluation. Rates and
// metrics keep their decimals.

export function toNumber(value: unknown): number | null {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function formatCount(value: unknown): string {
  const parsed = toNumber(value);
  if (parsed === null) return 'Unavailable';
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(Math.max(0, Math.round(parsed)));
}

export function formatDecimal(value: unknown, digits = 2): string {
  const parsed = toNumber(value);
  if (parsed === null) return 'Unavailable';
  return new Intl.NumberFormat('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(parsed);
}

export function formatPercent(fraction: unknown, digits = 1): string {
  const parsed = toNumber(fraction);
  if (parsed === null) return 'Unavailable';
  return `${(parsed * 100).toFixed(digits)}%`;
}

export function formatSigned(value: unknown, formatter: (value: unknown) => string = formatCount): string {
  const parsed = toNumber(value);
  if (parsed === null) return '—';
  if (parsed === 0) return '0';
  const magnitude = formatter(Math.abs(parsed));
  return parsed > 0 ? `+${magnitude}` : `−${magnitude}`;
}

// Table cells: integral numbers print without decimals; other numbers keep
// up to three decimals so metrics and probabilities are not distorted.
export function display(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number') {
    return Number.isInteger(value)
      ? new Intl.NumberFormat('en-US', { useGrouping: false }).format(value)
      : new Intl.NumberFormat('en-US', { maximumFractionDigits: 3, useGrouping: false }).format(value);
  }
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

export function label(column: string): string {
  return column.replace(/_/g, ' ');
}
